"""same-frame perturbation consistency -- 안정성 축 둘 중 **모션 성분이 정확히 0인 쪽**.

진단 문서 §28.7의 축이고 `experiment_history.md` §5 항목 3의 후반부다.

## 왜 이 축이 필요한가

안정성을 재는 다른 축(frame-gap)은 **장면이 실제로 변한 양**과 **모델이 흔들린 양**이 섞인다.
로봇이 곡선을 돌거나 가림 위상이 바뀌면 예측 변화가 gap에 비선형으로 생기고, 그러면 절편을
"순수 불안정성"이라고 부를 수 없다(외부 검토 지적, §28.7).

**이 축에는 그 문제가 없다.** 같은 프레임에 실제 카메라에서 일어날 만한 섭동(밝기·잡음·blur)만
주므로 **장면 기하가 완전히 동일**하다. 즉 모션 성분이 0이고 절편을 추정할 필요가 없다.
관측되는 차이는 전부 "입력이 조금 흔들렸을 때 모델이 얼마나 흔들리나"다.

## 정의

광선 θ, 프레임 f, 섭동 T에 대해 `R̂(f, θ)` = 예측 free의 첫 장애물까지 거리
(`first_free_range`, 학습·평가와 **같은 함수**). 원본과 섭동본이 **둘 다** `RAY_OK`인 광선에서

    Δ(f, θ) = |R̂_clean(f, θ) − R̂_pert(f, θ)|      [cm]

의 median과 P90을 보고한다. 시드 3개는 각각 재고 평균한다(시드 간 차이는 §17이 이미 쟀다).

**상태가 갈리는 광선은 Δ가 볼 수 없다.** 원본은 장애물을 봤는데 섭동본은 격자 끝까지
free라고 보면 그건 Δ 0이 아니라 **가장 큰 불안정**인데 표본에서 빠진다. 그래서
`상태 불일치` 비율을 **같은 표에** 찍는다 -- §17.5가 시드 축에서 겪은 함정과 같은 것이다.

## 품질 게이트 -- 이것 없이 읽으면 반드시 틀린다

**설계 문서 §15.6의 교훈: 안정성 지표를 단독으로 읽으면 "아무것도 안 배우는 것"이 1등이다.**
이 축에서 그 함정은 특히 노골적이다 -- **constant map은 입력을 아예 안 보므로 Δ가 정확히 0**,
즉 완벽한 안정성 점수를 받는다. 그래서 표에 그 행을 **직접 넣는다.** 트리비얼 기준선을
같은 표에 병기하는 이 프로젝트의 첫 번째 규약(§1)이 여기에도 그대로 걸린다.

그리고 Δ를 **모델 자신의 오차와 나란히** 읽는다. §17.6이 시드 축에서 만든 그림이 기준이다:

    모델 오차 15 cm  >>  방향별 시드 흔들림 3.8 cm  >>  전역 시드 흔들림 0.16 cm

섭동 Δ가 이 중 어디에 놓이는지가 이 표가 답하는 것이다.

## 무엇을 주장할 수 있고 없나

- **말할 수 있는 것**: 배포 환경에서 조명·센서 잡음이 이만큼 흔들릴 때 예측 자유거리가
  얼마나 움직이나.
- **말할 수 없는 것**: 이것은 **robustness가 아니다.** 섭동 하에서도 GT에 가까운지가 아니라
  **자기 자신에게 일관된지**만 잰다. 상수 지도가 만점을 받는 이유가 그것이다.

실행:

    python tools/measure_perturbation_stability.py
    python tools/measure_perturbation_stability.py --log_root=runs/pixel_offset --cells=off_0.00
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import functional as TF

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.models\._utils")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.common.baselines import as_batch, constant_free_map  # noqa: E402
from projects.common.free_space import decompose  # noqa: E402
from projects.common.free_space_metrics import iou_free as iou_free_fn  # noqa: E402
from projects.common.polar import RAY_OK, build_ray_index, first_free_range  # noqa: E402
from projects.datasets.photometric import PhotometricParams, apply_photometric  # noqa: E402
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    FINETUNE_CAMERA_NAMES,
    GRID_SPEC,
    RobotBEVDataset,
    build_bev_masks,
    load_masked_labels,
    parse_sequence_names,
    split_samples_by_sequence,
)
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.pixel_grid import convention_for_run_dirs  # noqa: E402
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402


def _csv(value):
    """Fire는 `--cells=a,b`를 **tuple**로 파싱한다 -- 스칼라만 가정하면 조용히 깨진다."""
    if isinstance(value, (tuple, list)):
        return tuple(str(v).strip() for v in value)
    return tuple(v.strip() for v in str(value).split(",") if v.strip())


def _gaussian_blur(rgb, sigma):
    """(S,3,H,W) -> 같은 shape. 커널은 `2*ceil(3σ)+1`(홀수)로 잡는다."""
    radius = int(np.ceil(3.0 * sigma))
    kernel = 2 * radius + 1
    return TF.gaussian_blur(rgb, kernel_size=[kernel, kernel], sigma=[sigma, sigma])


#: 섭동 목록. **실제 카메라에서 일어날 만한 크기**로 잡았고, 학습 증강 범위
#: (`BRIGHTNESS_RANGE` 0.8~1.2, `NOISE_STD_RANGE` 0~0.02)의 **안쪽**이다 --
#: 학습이 이미 본 적 있는 종류의 흔들림에서조차 예측이 움직이는지를 묻는 것이 요점이다.
#: blur는 학습 증강에 **없다**(광도 증강만 쓴다) -- 그래서 셋 중 가장 어려운 칸이다.
PERTURBATIONS = {
    "밝기 −15 %": lambda rgb, g: apply_photometric(rgb, PhotometricParams(brightness=0.85)),
    "밝기 +15 %": lambda rgb, g: apply_photometric(rgb, PhotometricParams(brightness=1.15)),
    "잡음 σ=0.02": lambda rgb, g: apply_photometric(
        rgb, PhotometricParams(noise_std=0.02), generator=g),
    "blur σ=1.0px": lambda rgb, g: _gaussian_blur(rgb, 1.0),
}


def _load(ckpt, vox, encoder_type, device):
    model = ThreeClassSegnet(GRID_SPEC.n_rows, 1, GRID_SPEC.n_cols, vox, use_radar=False,
                             use_lidar=False, do_rgbcompress=True, encoder_type=encoder_type,
                             rand_flip=False, num_classes=2).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=True)
    model.eval()
    return model


def _find_checkpoint(log_root, cell, seed):
    ckpt_dir = Path(log_root) / "ckpt" / f"{cell}_s{seed}"
    found = sorted(ckpt_dir.glob("model_best-*.pth"))
    if not found:
        raise FileNotFoundError(f"체크포인트가 없다: {ckpt_dir}")
    return found[-1]


def main(
    log_root="runs/pixel_offset",
    cells="off_0.00",
    seeds="0,1,2",
    tau=0.5,
    step_cells=0.5,
    train_sequences="raws2,raws3,rawos1,rawos2,rawos4",
    val_sequences="raws1,rawos3",
    dataset_root=DEFAULT_DATASET_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    encoder_type="res101",
    batch_size=4,
    num_workers=4,
    # 잡음 섭동의 시드. 고정하지 않으면 같은 표를 두 번 만들 때 숫자가 갈린다.
    noise_seed=0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cells, seeds = _csv(cells), tuple(int(s) for s in _csv(seeds))

    root = Path(dataset_root)
    names = parse_sequence_names(train_sequences)
    val_names = parse_sequence_names(val_sequences)
    train_samples, val_samples = split_samples_by_sequence(
        [root / n for n in names + val_names], val_names)
    dataset = RobotBEVDataset(val_samples, common_root=common_root, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 표본 규약은 **런의 config.json에서 되찾는다**(§28.6) -- 기본값을 쓰면 legacy 런을
    # 새 기하로 재채점해 조용히 다른 숫자가 나온다.
    convention, offset = convention_for_run_dirs(
        [Path(log_root) / "logs" / f"{c}_s{s}" for c in cells for s in seeds])
    vox_util = build_double_sphere_vox_util(GRID_SPEC, dataset.cameras, device=device,
                                           pixel_convention=convention, pixel_offset=offset)
    rays = build_ray_index(GRID_SPEC, step_cells=float(step_cells))
    quant_cm = float(step_cells) * GRID_SPEC.cell_m * 100.0

    # 라벨은 모델과 무관하므로 한 번만 분해한다. GT 광선거리도 캐시한다 --
    # 섭동 Δ를 **오차 자체**와 나란히 놓는 것이 이 표의 요점이다(§17.6).
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
    gt_r, gt_ok = np.stack(gt_r), np.stack(gt_ok)

    def ray_ranges(model, perturb=None, generator=None):
        """(F, n_theta) 광선거리 + `RAY_OK` 마스크 + `iou_free`."""
        r_all, ok_all, ious, counts = [], [], [], []
        with torch.no_grad():
            for batch, (gt_parts, valid) in zip(loader, labels):
                rgb = batch["rgb_camXs"].to(device)
                if perturb is not None:
                    b, s_cam = rgb.shape[0], rgb.shape[1]
                    flat = rgb.reshape(b * s_cam, *rgb.shape[2:])
                    rgb = perturb(flat, generator).reshape(b, s_cam, *rgb.shape[2:])
                # `- 0.5`는 학습·재채점과 **정확히 한 번씩** 걸려야 한다(§18.1의 사고).
                _, _, logits, _, _ = model(rgb - 0.5,
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

    # --- constant map 행: 이 축에서 Δ가 정확히 0인 트리비얼 해 ----------------------
    permanent_blind, invalid = build_bev_masks(common_root, GRID_SPEC, FINETUNE_CAMERA_NAMES)
    constant_map = constant_free_map([
        decompose(*load_masked_labels(r, s, permanent_blind, invalid))["free"]
        for r, s in train_samples
    ])
    const_ious, const_counts = [], []
    for _, (gt_parts, valid) in zip(loader, labels):
        pred = as_batch(constant_map, valid.shape[0], device).bool() & valid.bool()
        value, count = iou_free_fn(pred, gt_parts["free"], valid)
        const_ious.append(value)
        const_counts.append(count)
    const_iou = float(np.average(const_ious, weights=np.maximum(const_counts, 1e-9)))

    print(f"\n=== same-frame perturbation consistency (τ={tau}, "
          f"광선 표본 {quant_cm:.1f} cm, 시드 {len(seeds)}개 평균) ===")
    print("**모션 성분이 정확히 0인 축이다** -- 같은 프레임에 광도 섭동만 준다.")
    print(f"val {len(val_samples)}프레임 | 표본 규약 {convention} / offset {offset}\n")

    for cell in cells:
        # 원본 예측을 시드마다 한 번씩만 계산해 캐시한다.
        base = {}
        for seed in seeds:
            model = _load(_find_checkpoint(log_root, cell, seed), vox_util, encoder_type, device)
            base[seed] = (model, *ray_ranges(model))

        clean_iou = float(np.mean([base[s][3] for s in seeds]))
        gt_med, gt_p90 = [], []
        for seed in seeds:
            _, r, ok, _ = base[seed]
            both = ok & gt_ok
            err = np.abs(r[both] - gt_r[both]) * 100.0
            gt_med.append(np.median(err))
            gt_p90.append(np.percentile(err, 90))
        gt_med_cm, gt_p90_cm = float(np.mean(gt_med)), float(np.mean(gt_p90))

        # **Δ median은 거의 항상 0이다** -- 예측이 5 cm 셀의 이진 맵이고 광선 표본이
        # 2.5 cm라, 작은 섭동에서는 대부분의 광선이 **정확히 같은 셀에서** 멈춘다.
        # 그래서 median 대신 `Δ=0 비율`(얼마나 많은 광선이 전혀 안 움직이나)과
        # mean·P90(움직이는 광선이 얼마나 움직이나)을 같이 낸다. §17.4가 시드 축에서
        # `σ=0 비율` 23 %를 같은 이유로 보고한 것과 같은 처리다.
        header = (f"{'섭동':16}{'Δ=0 비율':>11}{'Δ mean':>10}{'Δ P90':>10}"
                  f"{'상태 불일치':>13}{'iou_free':>11}{'ΔP90/오차P90':>14}")
        print(f"[{cell}]  품질 게이트: 원본 `iou_free` = {clean_iou:.4f}"
              f"  (constant map {const_iou:.4f})")
        print(f"          GT 대비 오차: median {gt_med_cm:.1f} cm / P90 {gt_p90_cm:.1f} cm"
              "   <- Δ를 이것과 나란히 읽는다")
        print(header)
        print("-" * len(header))

        for label, fn in PERTURBATIONS.items():
            zeros, means, p90s, flips, ious = [], [], [], [], []
            for seed in seeds:
                model, r_clean, ok_clean, _ = base[seed]
                gen = torch.Generator(device=device).manual_seed(noise_seed + seed)
                r_pert, ok_pert, iou_p = ray_ranges(model, perturb=fn, generator=gen)
                both = ok_clean & ok_pert
                delta = np.abs(r_clean[both] - r_pert[both]) * 100.0
                zeros.append(float(np.mean(delta == 0.0)) * 100.0)
                means.append(float(np.mean(delta)))
                p90s.append(float(np.percentile(delta, 90)))
                # 상태 불일치: 한쪽만 RAY_OK인 광선의 비율. Δ 표본에서 빠지므로 따로 센다.
                flips.append(float(np.mean(ok_clean != ok_pert)) * 100.0)
                ious.append(iou_p)
            print(f"{label:16}{np.mean(zeros):>10.1f}%{np.mean(means):>8.2f}cm"
                  f"{np.mean(p90s):>8.2f}cm{np.mean(flips):>12.2f}%{np.mean(ious):>11.4f}"
                  f"{np.mean(p90s) / max(gt_p90_cm, 1e-9):>14.2f}")

        print(f"{'constant map':16}{100.0:>10.1f}%{0.0:>8.2f}cm{0.0:>8.2f}cm"
              f"{0.0:>12.2f}%{const_iou:>11.4f}{0.0:>14.2f}"
              "   <- **입력을 안 보므로 만점이다**")
        print()

    print("""--- 읽는 법 (설계 §15.6·§17.5, 결과를 보기 전에 정했다) ---
  1. **constant map 행을 먼저 본다.** Δ가 정확히 0이고 `iou_free`가 크게 낮다 --
     **안정성만 읽으면 아무것도 안 배우는 것이 1등**이라는 것의 직접 증거다.
  2. **`ΔP90/오차P90` 열이 판정을 만든다.** 이 값이 1보다 훨씬 작아야 "입력이 흔들려도 같은
     답을 준다"가 성립한다. GT 대비 오차(median 15 cm 규모, §17.6)와 같은 자리에 놓는다.
  3. **`상태 불일치`를 빼놓지 않는다.** Δ 표본에서 빠지는 광선이고, 빼면 불안정한
     설정이 유리해진다(§17.5가 시드 축에서 겪은 함정과 같다).
  4. **이것은 robustness가 아니다.** 섭동 하에서 GT에 가까운지가 아니라 자기 자신에게
     일관된지만 잰다. `iou_free` 열이 섭동 하의 품질이고, 그쪽이 robustness다.""")


if __name__ == "__main__":
    Fire(main)
