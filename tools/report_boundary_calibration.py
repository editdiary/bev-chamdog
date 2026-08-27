"""경계 항의 품질을 **config끼리 비교 가능한 눈금**으로 잰다 (2026-08-27).

## 왜 이 도구가 필요한가 -- 로그의 `kl_boundary`는 비교할 수 없다

`kl_boundary`는 각 런이 **자기 `Ω_B`에서 자기 target에 대해** 잰 평균이다. 그래서 손잡이를
바꾸면 값이 줄어드는데, 그 줄어듦의 상당 부분이 **재는 대상이 쉬워진 것**이다.

- **`δ`가 커지면 셀 집합이 바뀐다.** `δ = 0.45`의 `Ω_B`는 셀의 32 %이고 벽에서 40 cm
  떨어진 셀까지 들어오는데 거기 target은 0.958이라 쉽다. `δ = 0.15`의 `Ω_B`는 11.6 %이고
  전부 경계 15 cm 안, 전부 어렵다.
- **`α`·`κ`·`ε`이 바뀌면 target이 바뀐다.** 평평한 target은 본질적으로 맞히기 쉽다.

실측으로 `δ = 0.45` 런의 `kl_boundary` 0.0870을 분해하면

    0.0870  ->  0.1052 (셀 집합을 |d| <= 0.15로 통일)  ->  0.2691 (target도 공통으로 통일)

이고 대조군은 같은 눈금에서 0.4780이다. **로그가 보여준 "96 % 개선"이 실제로는 44 %다.**

## 이 도구가 하는 일

모든 런의 최종 체크포인트를 **같은 셀**(`|d| <= core_m`, 기본 0.15)에서 **같은 target**
(`δ = 0.15`, `α = 1.0`, `κ = 1`, `ε = 0`)에 대해 채점한다. 그러면 남는 차이는 오직
**모델이 경계에서 실제로 무엇을 예측하는가**다.

**이 눈금은 대조군에게 유리하다** -- 공통 target이 대조군 자신의 target이기 때문이다.
그러니 대조군이 여기서 지면 그 개입의 개선은 실재하고, 오히려 과소평가된 값이다.

같이 내는 것 둘.

- **예측 엔트로피** -- 모델이 경계에서 얼마나 확신하는가. `kl`이 줄어든 이유가 "위치를 더
  잘 맞혀서"인지 "덜 확신해서"인지 가른다. 후자면 `f1@10cm`이 안 좋아진다.
- **`|d|` 층별 kl** -- 개선이 경계 바로 옆에서 났는지 대역 가장자리에서 났는지.

실행:
    CUDA_VISIBLE_DEVICES=1 python tools/report_boundary_calibration.py \\
        --runs=runs/alpha_y4/logs/a100_s0,runs/loss_convergence/logs/d045_s0
"""
import json
import sys
from pathlib import Path

import torch
from fire import Fire
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.common.soft_boundary import (  # noqa: E402
    TARGET_GAUSSIAN,
    soft_target,
    target_entropy,
)
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    FINETUNE_CAMERA_NAMES,
    GRID_SPEC,
    RobotBEVDataset,
    build_bev_masks,
)
from projects.datasets.simplebev_vox import height_config_for_ckpt_dirs  # noqa: E402
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.pixel_grid import convention_for_run_dirs  # noqa: E402
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402

# **공통 눈금의 target.** 확정 config 계열의 대역 폭(δ=0.15)에 α=1.0이다. 무엇을 고르든
# 모든 런에 **같은 것**을 쓰는 한 비교는 성립한다 -- 절대값이 아니라 순서를 읽는 도구다.
_COMMON = {"delta": 0.15, "alpha": 1.0}

# `|d|` 층. **경계를 이산 EDT 값 사이에 놓는다.** 5 cm 격자의 유클리드 거리변환은
# {0.05, 0.0707, 0.10, 0.1118, 0.1414, 0.15, ...}만 내놓으므로, 구간을 그 사이에 두지
# 않으면 빈 층이 생긴다(첫 시도의 [0.075, 0.10)이 그랬다). 아래 끝이 0이 아니라 한 셀인
# 이유는 `signed_distance_field`가 셀 **중심** 사이로 재기 때문이다 -- 경계에 붙은 셀도
# `|d| = cell_m = 0.05`다.
_LAYERS = (("5~8 cm", 0.05, 0.08), ("8~12 cm", 0.08, 0.12), ("12~15 cm", 0.12, 0.1501))


def _width(text) -> int:
    from unicodedata import east_asian_width
    return sum(2 if east_asian_width(ch) in "WF" else 1 for ch in str(text))


def _pad(text, width, align=">") -> str:
    fill = max(0, width - _width(text))
    return f"{text}{' ' * fill}" if align == "<" else f"{' ' * fill}{text}"


def _row(cells, widths):
    return "  " + " ".join(_pad(c, w) for c, w in zip(cells, widths))


def _shape_params(config):
    """대역 모양을 `(sigma, alpha)`로 돌려준다 -- 쓰지 않는 쪽은 `None`.

    **둘 중 하나만 설정된다.** `(σ, k)` 재매개변수화(2026-08-28) 이후 런은 `sigma_m`만
    갖고 `sigma_alpha`가 `None`이라, `float(config["sigma_alpha"])`로 읽으면 죽는다.
    """
    alpha, sigma = config.get("sigma_alpha"), config.get("sigma_m")
    if alpha is not None:
        return None, float(alpha)
    if sigma is not None:
        return float(sigma), None
    raise ValueError(f"sigma_alpha와 sigma_m이 둘 다 없다: {config.get('exp_name')}")


def _label(config):
    delta = float(config["delta_m"])
    sigma, alpha = _shape_params(config)
    if config.get("soft_target") == "linear":
        shape = "linear"
    elif sigma is not None:
        # `σ`로 준 런은 `k = δ/σ`로 적는다 -- "몇 σ에서 자르는가"가 바로 읽힌다.
        shape = f"σ{sigma:g} k{delta / sigma:g}"
    else:
        shape = f"α{alpha:g}"
    return (f"δ{delta:.2f} {shape}"
            f" κ{float(config.get('band_kappa', 1.0)):.2f}"
            f" ε{float(config.get('label_eps', 0.0)):.2f}")


def _kl(y, log_free, log_not_free):
    """`KL(y ‖ p)`. `−(y log p + (1−y) log(1−p)) − H(y)`이고 `y = p`에서 정확히 0이다."""
    return -(y * log_free + (1.0 - y) * log_not_free) - target_entropy(y)


def main(runs, core_m=0.15, dataset_root=DEFAULT_DATASET_ROOT,
         common_root=DEFAULT_COMMON_ROOT, encoder_type="res101",
         split_file="runs/height_bins/logs/y4_s0/split_val_samples.txt",
         batch_size=4, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dirs = [Path(r.strip()) for r in
                (runs if isinstance(runs, (list, tuple)) else str(runs).split(","))
                if str(r).strip()]

    blind, invalid = build_bev_masks(common_root, GRID_SPEC, FINETUNE_CAMERA_NAMES)
    root = Path(dataset_root)
    lines = [ln.strip() for ln in Path(split_file).read_text().splitlines() if ln.strip()]
    dataset = RobotBEVDataset([(root / ln.split("/")[0], ln.split("/")[1]) for ln in lines],
                              common_root=common_root, augment=False)
    # `num_workers=0`이다 -- 이 파일은 `if __name__` 가드 안에서 도는 게 아니라 Fire가
    # 부르므로, worker를 켜면 spawn이 main 모듈을 다시 import하려다 실패한다.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    blind_t = torch.from_numpy(blind).view(1, 1, *blind.shape).to(device).bool()

    print(f"=== 경계 항의 **공통 눈금** 채점 (val {len(lines)}프레임) ===")
    print(f"  고정 셀: |d| <= {core_m:.2f} m  |  공통 target: δ={_COMMON['delta']:.2f}, "
          f"α={_COMMON['alpha']:g}, κ=1, ε=0")
    print("  **이 눈금은 대조군에게 유리하다** -- 공통 target이 대조군 자신의 target이다.")
    print("  그러니 대조군이 지면 그 개선은 실재하고 오히려 과소평가된 값이다.\n")

    w = (26, 12, 12, 14, 10, 10, 10)
    print(_row(("config", "공통 kl", "자기 kl", "예측 엔트로피",
                "5~8cm", "8~12", "12~15"), w))
    rows = []
    for run_dir in run_dirs:
        config = json.loads((run_dir / "config.json").read_text())
        config = config.get("config", config)
        ckpt_dir = run_dir.parent.parent / "ckpt" / run_dir.name
        found = sorted(ckpt_dir.glob("model_best-*.pth"))
        if not found:
            print(f"  [건너뜀] {run_dir.name}: 체크포인트 없음")
            continue
        conv, off = convention_for_run_dirs([run_dir])
        height = height_config_for_ckpt_dirs([ckpt_dir])
        vox = build_double_sphere_vox_util(
            GRID_SPEC, dataset.cameras, device=device, pixel_convention=conv, pixel_offset=off,
            height_bins=height["height_bins"], height_min_m=height["height_min_m"],
            height_max_m=height["height_max_m"])
        model = ThreeClassSegnet(GRID_SPEC.n_rows, vox.Y, GRID_SPEC.n_cols, vox,
                                 use_radar=False, use_lidar=False, do_rgbcompress=True,
                                 encoder_type=encoder_type, rand_flip=False,
                                 num_classes=2).to(device)
        state = torch.load(found[-1], map_location=device, weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state), strict=True)
        model.eval()

        # 자기 target의 손잡이. `ε`은 대역에서 `κ = 1−2ε`로 작동한다(`soft_boundary` 참조).
        own_sigma, own_alpha = _shape_params(config)
        own = {"delta": float(config["delta_m"]),
               "sigma": own_sigma, "alpha": own_alpha,
               "kappa": float(config.get("band_kappa", 1.0))
               * (1.0 - 2.0 * float(config.get("label_eps", 0.0)))}
        acc = {k: [0.0, 0] for k in ("common", "own", "ent")}
        layers = {name: [0.0, 0] for name, _, _ in _LAYERS}
        with torch.no_grad():
            for batch in loader:
                logits = model(batch["rgb_camXs"].to(device) - 0.5,
                               batch["pix_T_cams"].to(device),
                               batch["cam0_T_camXs"].to(device), vox)[2]
                log_probs = torch.log_softmax(logits, dim=1)
                log_free, log_not_free = log_probs[:, 1:2], log_probs[:, 0:1]
                d = batch["d_bev_g"].to(device)
                valid = batch["valid_bev_g"].to(device).bool()
                core = valid & ~blind_t & (d.abs() <= core_m)
                if not bool(core.any()):
                    continue
                y_common = soft_target(d, _COMMON["delta"], TARGET_GAUSSIAN,
                                       alpha=_COMMON["alpha"])
                y_own = soft_target(d, own["delta"], TARGET_GAUSSIAN,
                                    sigma=own["sigma"], alpha=own["alpha"],
                                    kappa=own["kappa"])
                kl_common = _kl(y_common, log_free, log_not_free)
                kl_own = _kl(y_own, log_free, log_not_free)
                p = log_free.exp().clamp(1e-6, 1 - 1e-6)
                ent = -(p * p.log() + (1 - p) * (1 - p).log())
                for key, field in (("common", kl_common), ("own", kl_own), ("ent", ent)):
                    acc[key][0] += float(field[core].sum())
                    acc[key][1] += int(core.sum())
                for name, lo, hi in _LAYERS:
                    sel = core & (d.abs() >= lo) & (d.abs() < hi)
                    if bool(sel.any()):
                        layers[name][0] += float(kl_common[sel].sum())
                        layers[name][1] += int(sel.sum())
        del model
        torch.cuda.empty_cache()

        mean = {k: v[0] / max(1, v[1]) for k, v in acc.items()}
        lay = {n: layers[n][0] / max(1, layers[n][1]) for n, _, _ in _LAYERS}
        rows.append((_label(config), mean, lay))
        print(_row((_label(config), f"{mean['common']:.4f}", f"{mean['own']:.4f}",
                    f"{mean['ent']:.4f}", *[f"{lay[n]:.3f}" for n, _, _ in _LAYERS]), w))

    if len(rows) > 1:
        base = rows[0][1]["common"]
        print(f"\n  첫 줄({rows[0][0]}) 대비 공통 kl 변화:")
        for label, mean, _ in rows[1:]:
            print(f"    {label:<26} {mean['common'] - base:+.4f} "
                  f"({(mean['common'] / base - 1) * 100:+.1f} %)")

    print("\n판독:")
    print("  `공통 kl`과 `자기 kl`이 크게 다르면 그 런의 로그 값은 **다른 것을 재고 있었다**.")
    print("  `공통 kl`이 줄었는데 `예측 엔트로피`가 같이 커졌으면, 개선의 정체는 '위치를 더 잘")
    print("  맞힘'이 아니라 '덜 확신함'이다 -- 그때 `f1@10cm`은 안 좋아지거나 나빠진다.")
    print("  층별로 보면 개선이 경계 바로 옆(0~5 cm)에서 났는지 가장자리에서 났는지 갈린다.")


if __name__ == "__main__":
    Fire(main)
