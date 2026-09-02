"""시드를 바꾸면 **예측 경계가 몇 cm 흔들리나** -- `experiment_history.md` §5 "다음 세션이 할 일" 2번.

## 왜 이 숫자인가

살아남은 주장 ①(재학습 재현성 5~22배)은 지금 `fatal_rate`의 시드 간 σ가 0.0063 -> 0.0007
이라는 **무차원 비율**로 진술돼 있다. 로봇 논문에서 그 문장은 읽히지 않는다. 같은 사실을
물리 단위로 바꾸면

    "시드를 바꿔 다시 학습하면 자유공간 경계가 median X cm, 최악 10 %에서 Y cm 움직인다"

가 되고, 이건 곧바로 안전 여유(clearance)와 비교할 수 있는 양이다.

## 정의

광선 θ, 프레임 f, 시드 s에 대해 `R̂_s(f, θ)` = 예측 free의 첫 장애물까지 거리
(`first_free_range`, 학습·평가와 **같은 함수**). 세 시드가 **모두** `RAY_OK`인 광선에서

    σ(f, θ) = std_s R̂_s(f, θ)        (표본 표준편차, ddof=1)

를 내고, 그 분포의 median과 P90을 보고한다. 단위는 cm.

**상태가 갈리는 광선은 σ가 볼 수 없다.** 어떤 시드는 격자 끝까지 free(`censored`)라고 보고
어떤 시드는 장애물을 봤다면 그건 σ 0이 아니라 가장 큰 불안정인데 표본에서 빠진다. 그래서
`상태 불일치` 비율을 **같은 표에** 찍는다 -- 빼놓으면 불안정한 셀이 유리해진다.

## 품질과 같이 읽는다

설계 문서 §15.6의 교훈: 안정성 지표를 단독으로 읽으면 "아무것도 안 배우는 것"이 1등이다.
그래서 `iou_free`와 `range_mae`를 같은 표에 둔다. **`range_mae`와 나란히 보는 것이 요점**이다 --
시드 jitter가 오차 자체보다 훨씬 작아야 "같은 모델을 일관되게 준다"는 주장이 성립한다.

실행:

    python tools/report_seed_jitter.py --log_root=runs/ablation --cells=A_ce,C_soft,D_range
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.models\._utils")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.common.free_space import decompose  # noqa: E402
from projects.common.npsafe import bool_not  # noqa: E402
from projects.common.free_space_metrics import iou_free as iou_free_fn  # noqa: E402
from projects.common.polar import RAY_OK, build_ray_index, first_free_range  # noqa: E402
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    GRID_SPEC,
    RobotBEVDataset,
    parse_sequence_names,
    split_samples_by_sequence,
)
from projects.datasets.simplebev_vox import (  # noqa: E402
    height_config_for_ckpt_dirs,
    vox_dims,
)
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.pixel_grid import convention_for_run_dirs  # noqa: E402
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402


def _width(text) -> int:
    from unicodedata import east_asian_width
    return sum(2 if east_asian_width(ch) in "WF" else 1 for ch in str(text))


def _pad(text, width, align="<") -> str:
    fill = max(0, width - _width(text))
    return f"{text}{' ' * fill}" if align == "<" else f"{' ' * fill}{text}"


def _csv(value):
    """Fire는 `--cells=A_ce,C_soft`를 **tuple**로 파싱한다 -- 스칼라만 가정하면 조용히 깨진다."""
    if isinstance(value, (tuple, list)):
        return tuple(str(v).strip() for v in value)
    return tuple(v.strip() for v in str(value).split(",") if v.strip())


def _load(ckpt, vox, encoder_type, device):
    # `Y`는 vox_util이 이미 알고 있다 -- 리터럴을 다시 쓰면 둘이 어긋날 수 있다.
    model = ThreeClassSegnet(GRID_SPEC.n_rows, vox.Y, GRID_SPEC.n_cols, vox, use_radar=False,
                             use_lidar=False, do_rgbcompress=True, encoder_type=encoder_type,
                             rand_flip=False, num_classes=2).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=True)
    model.eval()
    return model


def main(
    log_root="runs/ablation",
    cells="A_ce,C_soft,D_range",
    seeds="0,1,2",
    # τ=0.5는 argmax와 같다 -- 기존 표의 동작점이다. §16.3처럼 τ를 옮겨 가며 보려면 바꾼다.
    tau=0.5,
    # 광선 반지름 표본 간격 [셀]. 기본 0.5셀 = 2.5 cm이고, 그게 곧 `R̂`의 **양자화 눈금**이다.
    # **σ_ray를 읽을 때 이 값이 바닥을 만든다** -- 시드 3개가 격자 눈금 위에 있으면 σ가
    # 가질 수 있는 최소 비영 값이 `간격/√3`이다. 세밀하게 줄여 보면 "정말 같은가"와
    # "눈금이 거칠어 구분이 안 되는가"가 갈린다. 단, 예측 자체가 5 cm 셀의 이진 맵이므로
    # 간격을 줄여도 셀 크기 아래로는 못 내려간다.
    step_cells=0.5,
    train_sequences="raws2,raws3,rawos1,rawos2,rawos4",
    val_sequences="raws1,rawos3",
    dataset_root=DEFAULT_DATASET_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    encoder_type="res101",
    batch_size=8,
    num_workers=8,
    # 요약 dict를 그대로 떨어뜨릴 JSON 경로. **광선별 원시 배열은 이것과 별개로**
    # `{log_root}/analysis/seed_jitter/*.npz`에 이미 캐시된다(프레임x광선 R̂와 RAY_OK).
    json_out=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cells = _csv(cells)
    seeds = tuple(int(s) for s in _csv(seeds))

    root = Path(dataset_root)
    names = parse_sequence_names(train_sequences)
    val_names = parse_sequence_names(val_sequences)
    _, val_samples = split_samples_by_sequence([root / n for n in names + val_names], val_names)
    dataset = RobotBEVDataset(val_samples, common_root=common_root, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # 표본 규약은 **런의 config.json에서 되찾는다** -- 기본값을 쓰면 `runs/ablation`(legacy)을
    # 새 기하로 재채점해 조용히 다른 숫자가 나온다.
    convention, offset = convention_for_run_dirs(
        [Path(log_root) / "logs" / f"{cell}_s{seed}" for cell in cells for seed in seeds])
    height = height_config_for_ckpt_dirs(
        [Path(log_root) / "ckpt" / f"{cell}_s{seed}" for cell in cells for seed in seeds])
    vox_util = build_double_sphere_vox_util(GRID_SPEC, dataset.cameras, device=device,
                                           pixel_convention=convention, pixel_offset=offset,
                                           height_bins=height["height_bins"],
                                           height_min_m=height["height_min_m"],
                                           height_max_m=height["height_max_m"])
    rays = build_ray_index(GRID_SPEC, step_cells=float(step_cells))
    quant_cm = float(step_cells) * GRID_SPEC.cell_m * 100.0

    # 라벨은 모델과 무관하므로 한 번만 분해한다. GT 광선거리도 같이 캐시해 둔다 --
    # 시드 jitter를 **오차 자체**와 나란히 놓는 것이 이 표의 요점이기 때문이다.
    labels, gt_r, gt_ok = [], [], []
    for batch in loader:
        seg, vis, valid = (batch[k].to(device) for k in ("seg_bev_g", "vis_bev_g", "valid_bev_g"))
        parts = decompose(seg, vis, valid)
        labels.append((parts, valid))
        gt_free = (parts["free"].bool() & valid.bool()).cpu().numpy()
        for i in range(gt_free.shape[0]):
            r, s = first_free_range(gt_free[i, 0], rays)
            gt_r.append(r)
            gt_ok.append(s == RAY_OK)
    gt_r = np.stack(gt_r)                       # (F, n_theta)
    gt_ok = np.stack(gt_ok)

    def ray_ranges(model):
        """모델 하나에 대해 (F, n_theta) 광선거리와 `RAY_OK` 마스크, 그리고 `iou_free`."""
        r_all, ok_all, ious, counts = [], [], [], []
        with torch.no_grad():
            for batch, (gt_parts, valid) in zip(loader, labels):
                _, _, logits, _, _ = model(batch["rgb_camXs"].to(device) - 0.5,
                                           batch["pix_T_cams"].to(device),
                                           batch["cam0_T_camXs"].to(device), vox_util)
                valid_b = valid.bool()
                pred_free = (torch.softmax(logits, dim=1)[:, 1:2] > tau) & valid_b
                value, count = iou_free_fn(pred_free, gt_parts["free"], valid)
                ious.append(value)
                counts.append(count)
                free_np = pred_free.cpu().numpy()
                for i in range(free_np.shape[0]):
                    r, s = first_free_range(free_np[i, 0], rays)
                    r_all.append(r)
                    ok_all.append(s == RAY_OK)
        iou = float(np.average(ious, weights=np.maximum(counts, 1e-9)))
        return np.stack(r_all), np.stack(ok_all), iou

    # forward 결과를 캐시한다. 집계 방식을 바꿀 때마다 9개 체크포인트를 다시 돌리면
    # GPU가 다른 학습과 경합할 때 못 쓴다 -- 캐시는 체크포인트 파일명에 묶여 있으므로
    # 체크포인트가 바뀌면 자동으로 무효가 된다.
    cache_dir = Path(log_root) / "analysis" / "seed_jitter"
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for cell in cells:
        per_seed_r, per_seed_ok, per_seed_iou = [], [], []
        for seed in seeds:
            found = sorted(Path(f"{log_root}/ckpt/{cell}_s{seed}").glob("model_best-*.pth"))
            if not found:
                print(f"!! 체크포인트가 없어 건너뛴다: {log_root}/ckpt/{cell}_s{seed}")
                continue
            # **캐시 키에 광선 파라미터가 들어가야 한다** -- 안 넣으면 `step_cells`를 바꿔도
            # 옛 눈금으로 계산한 배열이 조용히 재사용된다.
            cache = (cache_dir /
                     f"{cell}_s{seed}_{found[-1].stem}_tau{tau:.2f}_step{float(step_cells):g}.npz")
            if cache.exists():
                blob = np.load(cache)
                r, ok, iou = blob["r"], blob["ok"], float(blob["iou"])
                print(f"  {cell}_s{seed} (캐시) iou_free {iou:.4f}", flush=True)
            else:
                model = _load(found[-1], vox_util, encoder_type, device)
                r, ok, iou = ray_ranges(model)
                del model
                torch.cuda.empty_cache()
                np.savez_compressed(cache, r=r, ok=ok, iou=iou)
                print(f"  {cell}_s{seed} ({found[-1].name}) iou_free {iou:.4f}", flush=True)
            per_seed_r.append(r)
            per_seed_ok.append(ok)
            per_seed_iou.append(iou)
        if len(per_seed_r) < 2:
            print(f"!! {cell}: 시드가 2개 미만이라 σ를 낼 수 없다")
            continue

        r = np.stack(per_seed_r)                 # (S, F, n_theta)
        ok = np.stack(per_seed_ok)
        all_ok = ok.all(axis=0)                  # 세 시드 모두 RAY_OK
        any_ok = ok.any(axis=0)
        # 상태 불일치: 한 시드라도 OK인데 전부 OK는 아닌 광선. σ가 볼 수 없는 불안정이다.
        # **`~all_ok`를 직접 쓰면 안 된다** -- numpy 임시 소거가 `all_ok`를 제자리에서
        # 뒤집고, 바로 아래 `r[:, all_ok]`가 정반대 광선을 고르게 된다
        # (`projects/common/npsafe.py`).
        disagree = float((any_ok & bool_not(all_ok)).sum()) / max(int(any_ok.sum()), 1)

        r_ok = r[:, all_ok]                      # (S, N_rays)
        sigma_cm = np.std(r_ok, axis=0, ddof=1) * 100.0

        # **분해.** 주장 ①(`range_bias` σ 8.4배)은 집계 지표의 σ이고, 그건 시드 간
        # **전역 편이**다. 광선별 σ는 거기에 광선마다 독립인 잔차가 더해진 것이라 잔차가
        # 크면 전역 성분이 줄어도 거의 안 움직인다. 둘을 나눠 찍어야 두 숫자가 화해한다.
        #   R̂_s(f,θ) = M_s + resid_s(f,θ),  M_s = mean_{f,θ} R̂_s
        m_s = r_ok.mean(axis=1)                  # (S,) 시드별 평균 경계 위치
        sigma_global_cm = float(np.std(m_s, ddof=1) * 100.0)
        resid_cm = np.std(r_ok - m_s[:, None], axis=0, ddof=1) * 100.0
        # 오차 자체와의 대조 -- GT도 OK인 광선에서만 |R̂ - R_gt|를 낸다.
        paired = all_ok & gt_ok
        err_cm = np.abs(r[:, paired].mean(axis=0) - gt_r[paired]) * 100.0

        results[cell] = {
            "iou_free": float(np.mean(per_seed_iou)),
            "iou_sd": float(np.std(per_seed_iou, ddof=1)),
            "median": float(np.median(sigma_cm)),
            "p90": float(np.percentile(sigma_cm, 90)),
            "global": sigma_global_cm,
            "resid": float(np.median(resid_cm)),
            # 세 시드가 **완전히 같은 눈금**에 떨어진 광선의 비율. 이게 크면 σ_ray는
            # 모델의 성질이 아니라 양자화 바닥을 재고 있는 것이다.
            "zero": float((sigma_cm == 0).mean() * 100.0),
            "disagree": disagree * 100.0,
            "err_median": float(np.median(err_cm)),
            "n_rays": int(all_ok.sum()),
            "n_seeds": len(per_seed_r),
        }

    if not results:
        return

    if json_out:
        import json as _json
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(_json.dumps(
            {"tau": float(tau), "step_cells": float(step_cells),
             "quant_cm": float(quant_cm), "seeds": list(seeds), "cells": results},
            indent=2, ensure_ascii=False))
        print(f"  요약 -> {json_out}")

    print(f"\n=== seed boundary jitter -- 시드를 바꾸면 예측 경계가 몇 cm 움직이나 (τ={tau:.2f}) ===")
    print("`σ_ray`는 광선별 시드 간 표준편차 [cm]. 세 시드가 모두 RAY_OK인 광선만 들어간다.")
    print("**품질과 같이 읽는다**(§15.6) -- `iou_free`가 나쁜 셀의 안정성은 뜻이 없다.")
    print("`상태 불일치`는 σ가 볼 수 없는 불안정이다(한 시드는 장애물을 보고 다른 시드는 못 봤다).")
    print("`|오차| median`은 같은 광선에서 GT 대비 오차 -- jitter를 이것과 견줘 읽는다.\n")

    n_ref = max((v["n_seeds"] for v in results.values()), default=len(seeds))
    print(f"광선 거리의 **양자화 눈금은 {quant_cm:.2f} cm**(step_cells={step_cells})이고,"
          f" 시드 {n_ref}개에서 σ가 가질 수 있는\n최소 비영 값은"
          f" {quant_cm / n_ref ** 0.5:.2f} cm다."
          " `σ=0 비율`이 크면 σ_ray는 모델이 아니라 눈금을 재고 있다.\n"
          "예측 자체가 5 cm 셀의 이진 맵이므로 눈금을 줄여도 셀 크기가 바닥으로 남는다.\n")

    head = [("cell", 9), ("n", 3), ("iou_free", 9), ("σ_전역", 8), ("σ_ray med", 10),
            ("σ_ray P90", 10), ("σ_잔차 med", 11), ("σ=0 비율", 10), ("상태 불일치", 12),
            ("|오차| med", 11), ("광선 수", 9)]
    print("  " + " ".join(_pad(h, w, ">") for h, w in head))
    for cell, v in results.items():
        row = [cell, str(v["n_seeds"]), f"{v['iou_free']:.4f}", f"{v['global']:.2f}",
               f"{v['median']:.2f}", f"{v['p90']:.2f}", f"{v['resid']:.2f}",
               f"{v['zero']:.1f} %", f"{v['disagree']:.2f} %", f"{v['err_median']:.2f}",
               f"{v['n_rays']}"]
        print("  " + " ".join(_pad(c, w, ">") for c, (_, w) in zip(row, head)))

    live = [c for c in cells if c in results]
    if len(live) >= 2:
        a, d = results[live[0]], results[live[-1]]
        print(f"\n  {live[0]} -> {live[-1]}")
        for key, label in (("global", "σ_전역"), ("median", "σ_ray median"),
                           ("p90", "σ_ray P90"), ("resid", "σ_잔차 median")):
            print(f"    {_pad(label, 14)} {a[key]:6.2f} -> {d[key]:6.2f} cm"
                  f"  ({a[key] / max(d[key], 1e-9):.1f}x)")
        print("\n  **σ_전역과 σ_ray는 다른 것을 잰다.** 집계 지표(`range_bias`)의 시드 간 σ가"
              " 곧 σ_전역이고,\n  광선별 σ는 거기에 광선마다 독립인 잔차가 더해진 값이다"
              " -- 잔차가 크면 전역 성분이 줄어도\n  광선별 σ는 거의 안 움직인다."
              " **주장 ①은 σ_전역에 대한 것이므로 광선별 σ로 재진술되지 않는다.**")
        print(f"  **n={a['n_seeds']}의 σ이므로 배율의 신뢰구간은 넓다** -- 방향과 크기만 읽고"
              " 유의성은\n  `report_ablation.py`의 F 검정과 같이 판단한다.")


if __name__ == "__main__":
    Fire(main)
