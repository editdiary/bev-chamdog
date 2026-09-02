"""`L_range`의 dead zone이 실제로 무엇을 면제하고 있는지 잰다 (2026-08-27).

## 무엇을 묻는가

`L_range`는 광선당 **스칼라 하나**에 걸린다 -- `e_j = â(θ_j) − a_gt(θ_j)`이고 `â`는 그 광선
200 표본에 걸친 `p(free)`의 **부호 있는 합**이다(`range_loss.compute_range_loss`). 그리고
`|e_j| ≤ δ_R`이면 gradient가 정확히 0이 된다(설계 문서 §8.5). 여기서 두 가지가 걸린다.

1. **상쇄.** 합이라서 광선 앞쪽의 과대예측과 뒤쪽의 과소예측이 서로 지운다. 그러면 경계가
   전부 틀린 광선도 `|e_j|`가 작아져 dead zone에 들어갈 수 있다. 즉 dead zone 진입은
   "이 방향을 맞혔다"보다 **약한 조건**이고, 정확히 얼마나 약한지는 측정된 적이 없다.
2. **`D_range_s0`의 val `range_arc_mae`가 40 epoch 내내 0.244~0.25에서 멈춰 δ_R = 0.20
   밑으로 안 내려온다**(train은 epoch 3에 진입해 0.049까지 간다). 그 0.244가 어디서 오는지
   -- 경계 번짐인지, 벽 뒤 free 섬인지, 계통 편향인지 -- 를 갈라야 한다.

`range_arc_mae`(= `mean |e_j|`)는 상쇄된 **뒤**의 값이라 위 둘 중 어느 것도 답하지 못한다.
그래서 이 도구는 상쇄 **전**의 양을 따로 만든다. 표본별 잔차 `r_jk = p_jk·v_jk·in_jk −
free^gt_jk`를 부호로 갈라서

    e_over  = Δr · Σ_k max(0, r_jk)      (자유공간 과대예측 = `fatal` 방향)
    e_under = Δr · Σ_k max(0, −r_jk)     (과소예측 = 보수 방향)

로 두면 **`e_signed = e_over − e_under`가 loss가 보는 값이고, `e_abs = e_over + e_under`가
상쇄가 지우기 전의 총 불일치다.** 두 값의 비가 상쇄의 크기다.

## 무엇을 답하지 못하는가

`e_abs`는 확률을 그대로 적분하므로 **soft target을 정확히 맞힌 경계 대역도 0이 아니다.**
`y(d) + y(−d) = 1`이라 `e_signed` 쪽에서는 정확히 상쇄되지만 `e_abs`에는 대역 폭만큼 남는다
(δ = 0.15 m 대역 하나당 약 0.15 m). 그래서 `e_abs`의 절대값을 "오차"로 읽으면 안 되고,
**`|d|`로 층화한 §B의 far 성분**을 봐야 한다. 그쪽은 경계에서 먼 자리라 soft target이
설명하지 못한다.

실행:
    CUDA_VISIBLE_DEVICES=1 python tools/diagnose_range_dead_zone.py \\
        --run=D_range_s0 --log_root=runs/ablation
"""
import sys
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.common.free_space import decompose  # noqa: E402
from projects.common.polar import build_ray_index  # noqa: E402
from projects.common.range_loss import (  # noqa: E402
    DEFAULT_DELTA_R_M,
    RayGather,
    ray_is_ok,
)
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    GRID_SPEC,
    RobotBEVDataset,
)
from projects.datasets.simplebev_vox import height_config_for_ckpt_dirs  # noqa: E402
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.pixel_grid import convention_for_run_dirs  # noqa: E402
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402

# `|d|` 층. **`band`의 폭은 loss의 δ와 같아야 한다** -- 그래야 "soft 대역이 설명하는 몫"과
# "설명하지 못하는 몫"이 정확히 갈린다. `far`는 경계에서 0.5 m 넘게 떨어진 자리이고, 벽 뒤
# free 섬처럼 **per-cell BCE가 거의 벌하지 않는 오차 모드**가 사는 곳이다(설계 문서 §8.1).
_BANDS = (("band |d|<=0.15", 0.0, 0.15), ("near 0.15~0.5", 0.15, 0.5), ("far >0.5", 0.5, np.inf))

_DELTA_R_SWEEP = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def _width(text) -> int:
    from unicodedata import east_asian_width
    return sum(2 if east_asian_width(ch) in "WF" else 1 for ch in str(text))


def _pad(text, width, align=">") -> str:
    fill = max(0, width - _width(text))
    return f"{text}{' ' * fill}" if align == "<" else f"{' ' * fill}{text}"


def _row(cells, widths):
    return "  " + " ".join(_pad(c, w) for c, w in zip(cells, widths))


def collect(model, loader, vox_util, gather, device):
    """val/train 한 split을 훑어 **광선 단위** 원자료를 모은다.

    반환은 `RAY_OK` 광선만 담은 1차원 배열들의 dict다. 프레임 경계를 지우는 이유는 loss가
    `sum / |R_OK|`로 배치 전체를 한 번에 평균하기 때문이다 -- 프레임별로 먼저 평균하면
    광선 수가 다른 프레임에 다른 가중치를 주게 되어 loss와 다른 값이 나온다.
    """
    step_m = gather.step_m
    inside = gather.inside
    out = {k: [] for k in ("signed", "over", "under", "band_over", "band_under")}
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["rgb_camXs"].to(device) - 0.5,
                           batch["pix_T_cams"].to(device),
                           batch["cam0_T_camXs"].to(device), vox_util)[2]
            valid = batch["valid_bev_g"].to(device)
            free_gt = decompose(batch["seg_bev_g"].to(device),
                                batch["vis_bev_g"].to(device), valid)["free"]
            # `p(free)`는 loss가 쓰는 것과 **같은 softmax**에서 나와야 한다
            # (`binary_metrics.run_batch_soft_boundary`의 규약).
            prob = torch.softmax(logits, dim=1)[:, 1:2]

            valid_ray = gather(valid.to(prob.dtype)) * inside
            sampled_gt = gather(free_gt.to(torch.bool)) & inside
            # `compute_range_loss`와 **같은 마스킹**이다. 여기서 어긋나면 `signed`가 loss가
            # 보는 값과 달라져 이 도구 전체가 다른 양을 재게 된다(아래에서 assert로 고정).
            resid = gather(prob) * valid_ray - sampled_gt.to(prob.dtype)
            d_ray = gather(batch["d_bev_g"].to(device)).abs()

            ok = ray_is_ok(sampled_gt, inside)
            over = resid.clamp_min(0.0) * step_m
            under = (-resid).clamp_min(0.0) * step_m

            out["signed"].append((over.sum(-1) - under.sum(-1))[ok].cpu().numpy())
            out["over"].append(over.sum(-1)[ok].cpu().numpy())
            out["under"].append(under.sum(-1)[ok].cpu().numpy())
            for lo, hi in ((b[1], b[2]) for b in _BANDS):
                sel = (d_ray >= lo) & (d_ray < hi)
                out["band_over"].append((over * sel).sum(-1)[ok].cpu().numpy())
                out["band_under"].append((under * sel).sum(-1)[ok].cpu().numpy())
            print(f"\r  {len(out['signed'])}배치", end="", flush=True)
    print("\r" + " " * 24 + "\r", end="")

    n_band = len(_BANDS)
    packed = {k: np.concatenate(out[k]) for k in ("signed", "over", "under")}
    for key in ("band_over", "band_under"):
        # 배치마다 층 3개를 순서대로 넣었으므로 stride로 되찾는다.
        packed[key] = np.stack([np.concatenate(out[key][i::n_band]) for i in range(n_band)])
    return packed


def report(name, r, delta_r):
    n = r["signed"].size
    abs_signed = np.abs(r["signed"])
    e_abs = r["over"] + r["under"]
    in_dead = abs_signed <= delta_r

    print(f"\n=== [{name}] `RAY_OK` 광선 {n}개 ===")

    print("\n[A] 상쇄 -- loss가 보는 값 대 상쇄 전 총 불일치")
    print("  `e_signed`는 loss가 보는 값, `e_abs = e_over + e_under`는 부호로 지워지기 전 총량이다.")
    w = (26, 10, 10, 10, 10)
    print(_row(("", "평균", "중앙값", "p90", "p99"), w))
    for label, v in (("|e_signed| (loss가 보는 값)", abs_signed), ("e_abs (상쇄 전)", e_abs),
                     ("e_over (과대예측)", r["over"]), ("e_under (과소예측)", r["under"])):
        print(_row((label, f"{v.mean():.4f}", f"{np.median(v):.4f}",
                    f"{np.percentile(v, 90):.4f}", f"{np.percentile(v, 99):.4f}"), w))
    print(f"\n  상쇄율 = 1 − mean|e_signed|/mean e_abs = "
          f"**{1 - abs_signed.mean() / e_abs.mean():.3f}** "
          f"(1에 가까울수록 부호가 서로 다 지운다)")
    print(f"  부호 편향 mean e_signed = {r['signed'].mean():+.4f} m "
          f"(양수 = 자유공간 과대예측 = `fatal` 방향)")

    print(f"\n[B] dead zone(δ_R = {delta_r:.2f} m)이 무엇을 면제하는가")
    print(f"  dead zone 안 광선: {in_dead.sum()}/{n} = **{in_dead.mean() * 100:.1f} %** "
          f"-- 이 광선들은 gradient가 정확히 0이다")
    if in_dead.any():
        d_abs = e_abs[in_dead]
        print(f"  그 광선들의 `e_abs` 평균 {d_abs.mean():.4f} m / p90 {np.percentile(d_abs, 90):.4f} m")
        masked = int((d_abs > delta_r).sum())
        print(f"  그중 `e_abs > δ_R`인 광선: {masked}/{in_dead.sum()} = "
              f"**{masked / in_dead.sum() * 100:.1f} %**")
        print("     ^ **상쇄가 아니었다면 벌을 받았을 광선이다.** 이것이 dead zone 진입이")
        print("       '이 방향을 맞혔다'보다 약한 조건이라는 것의 크기다.")

    print("\n[C] 불일치가 경계에서 얼마나 떨어진 자리에 있나 (`|d|`로 층화)")
    print("  **`band`는 soft target이 설명한다** -- 대역 안에서 목표가 0/1이 아니므로 `e_abs`에")
    print("  남는 것이 정상이고, `y(d)+y(−d)=1`이라 `e_signed` 쪽에서는 정확히 상쇄된다.")
    print("  **`far`는 설명되지 않는다** -- 경계에서 0.5 m 넘게 떨어진 자리의 불일치이고,")
    print("  벽 뒤 free 섬처럼 `L_range`가 잡으려던 바로 그 오차 모드다.")
    w = (16, 12, 12, 12, 12, 14)
    print(_row(("층", "e_over", "e_under", "합", "총량 중 몫", "dead zone 안"), w))
    tot = e_abs.mean()
    for i, (label, _, _) in enumerate(_BANDS):
        bo, bu = r["band_over"][i], r["band_under"][i]
        s = (bo + bu).mean()
        dead_share = (bo + bu)[in_dead].mean() if in_dead.any() else float("nan")
        print(_row((label, f"{bo.mean():.4f}", f"{bu.mean():.4f}", f"{s:.4f}",
                    f"{s / tot * 100:.1f} %", f"{dead_share:.4f}"), w))

    print("\n[C2] 그 층들이 **loss가 보는 값**(`e_signed`)에 어떻게 들어가나")
    print("  [C]는 상쇄 전 총량이라 '어디가 시끄러운가'만 말한다. 실제로 `|e_signed|`를 만드는")
    print("  것이 어느 층인지는 층별 **부호 있는** 기여 `s_b`로 봐야 한다(Σ_b s_b = e_signed).")
    print("  `mean s_b`는 그 층의 계통 편향, `mean|s_b|`는 광선마다 흔들리는 크기,")
    print("  `Var 몫`은 `Cov(s_b, e_signed)/Var(e_signed)`로 **광선 간 분산을 누가 만드는가**다.")
    w = (16, 12, 12, 12, 14)
    print(_row(("층", "mean s_b", "mean|s_b|", "Var 몫", "band 폭 대비"), w))
    var_total = r["signed"].var()
    for i, (label, lo, hi) in enumerate(_BANDS):
        s_b = r["band_over"][i] - r["band_under"][i]
        cov = np.cov(s_b, r["signed"])[0, 1]
        # soft target이 설명할 수 있는 상한. 대역을 정확히 맞혀도 `e_abs`에는 대역 폭만큼
        # 남지만 `s_b`에는 **0만 남는다**(`y(d)+y(−d)=1`). 그래서 `mean|s_b|`가 0이 아니면
        # 그만큼은 대역이 **밀렸거나 번진** 것이지 soft target 자체가 아니다.
        note = "설명 상한 0.00" if i == 0 else ""
        print(_row((label, f"{s_b.mean():+.4f}", f"{np.abs(s_b).mean():.4f}",
                    f"{cov / var_total * 100:.1f} %", note), w))

    print("\n[D] δ_R을 옮기면 항이 얼마나 살아나나")
    w = (10, 14, 14, 12)
    print(_row(("δ_R", "밖 광선 비율", "mean e^eff", "loss(β=0.10)"), w))
    for dr in _DELTA_R_SWEEP:
        eff = np.clip(abs_signed - dr, 0.0, None)
        beta = 0.10
        rho = np.where(eff <= beta, eff ** 2 / (2 * beta), eff - beta / 2)
        print(_row((f"{dr:.2f}", f"{(eff > 0).mean() * 100:.1f} %",
                    f"{eff.mean():.4f}", f"{rho.mean():.4f}"), w))


def main(
    run="D_range_s0",
    log_root="runs/ablation",
    dataset_root=DEFAULT_DATASET_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    encoder_type="res101",
    delta_r=DEFAULT_DELTA_R_M,
    batch_size=4,
    num_workers=4,
    # train split은 val보다 크다. 같은 크기로 잘라야 두 숫자를 나란히 읽을 수 있다.
    max_train_frames=75,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = Path(log_root) / "logs" / run
    ckpt_dir = Path(log_root) / "ckpt" / run
    found = sorted(ckpt_dir.glob("model_best-*.pth"))
    if not found:
        raise FileNotFoundError(f"체크포인트가 없다: {ckpt_dir}")
    root = Path(dataset_root)

    # 표본 규약과 `Y`는 **런에서 되찾는다** -- 기본값을 쓰면 legacy 런을 새 기하로 재채점해
    # 조용히 다른 숫자가 나온다(AGENTS.md의 절단선).
    convention, offset = convention_for_run_dirs([log_dir])
    height = height_config_for_ckpt_dirs([ckpt_dir])
    print(f"[run] {run}  ckpt={found[-1].name}")
    print(f"[geom] height_bins={height['height_bins']} "
          f"({height['height_min_m']}~{height['height_max_m']} m), pixel={convention}+{offset}")

    splits = {}
    for tag, fname, cap in (("val", "split_val_samples.txt", None),
                            ("train", "split_train_samples.txt", max_train_frames)):
        lines = [ln.strip() for ln in (log_dir / fname).read_text().splitlines() if ln.strip()]
        if cap:
            lines = lines[:cap]
        splits[tag] = [(root / ln.split("/")[0], ln.split("/")[1]) for ln in lines]
        print(f"[split] {tag} {len(splits[tag])}프레임 ({fname})")

    first = RobotBEVDataset(splits["val"], common_root=common_root, augment=False)
    vox_util = build_double_sphere_vox_util(GRID_SPEC, first.cameras, device=device,
                                            pixel_convention=convention, pixel_offset=offset,
                                            height_bins=height["height_bins"],
                                            height_min_m=height["height_min_m"],
                                            height_max_m=height["height_max_m"])
    model = ThreeClassSegnet(GRID_SPEC.n_rows, vox_util.Y, GRID_SPEC.n_cols, vox_util,
                             use_radar=False, use_lidar=False, do_rgbcompress=True,
                             encoder_type=encoder_type, rand_flip=False, num_classes=2).to(device)
    state = torch.load(found[-1], map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=True)
    model.eval()

    rays = build_ray_index(GRID_SPEC)
    gather = RayGather(rays, (GRID_SPEC.n_rows, GRID_SPEC.n_cols), device=device)

    raw = {}
    for tag in ("val", "train"):
        dataset = (first if tag == "val"
                   else RobotBEVDataset(splits[tag], common_root=common_root, augment=False))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
        packed = collect(model, loader, vox_util, gather, device)
        report(tag, packed, delta_r)
        raw.update({f"{tag}_{k}": v for k, v in packed.items()})

    # **원자료를 남긴다** -- 층 경계나 δ_R을 바꿔 다시 읽을 때 GPU를 다시 쓰지 않기 위해서다.
    out = Path(log_root) / "analysis" / f"range_dead_zone_{run}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **raw)
    print(f"\n[저장] 광선 단위 원자료 -> {out}")

    print("\n판독:")
    print("  [A] 상쇄율이 크면 `range_arc_mae`는 광선의 품질이 아니라 **상쇄 후의 잔차**를 재는 것이다.")
    print("  [B] 'dead zone 안인데 e_abs > δ_R'인 비율이 크면 항이 면제하는 광선의 상당수가")
    print("      실제로는 틀린 광선이고, 그러면 dead zone은 설계가 의도한 '라벨 불확실성 면제'보다")
    print("      넓게 면제하고 있는 것이다.")
    print("  [C] val의 `far` 몫이 train보다 크면 **일반화가 안 되는 부분이 벽 뒤 free 섬 쪽**이고,")
    print("      그것이 val `arc_mae`가 0.2 밑으로 안 내려오는 이유다. 반대로 `band`가 지배하면")
    print("      경계 번짐이고 δ_R을 줄여도 얻을 것이 없다.")


if __name__ == "__main__":
    Fire(main)
