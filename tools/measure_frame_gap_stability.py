"""frame-gap consistency **diagnostic** -- 안정성 축 둘 중 라벨도 pose도 필요 없는 쪽.

진단 문서 §28.7의 축이고 `experiment_history.md` §5 항목 3의 전반부다.

## 이름을 조심해서 붙인 이유 -- "temporal stability"라고 부르지 않는다

원래 논리는 이랬다: "인접 프레임 사이 예측 변화 = **모션 성분**(gap에 비례) + **불안정 성분**
(gap과 무관) 이므로, gap에 대해 직선을 맞추면 **절편이 순수 불안정성**이다."

**외부 검토(2026-08-25b)가 이 논리에 제동을 걸었고 그것을 수용했다.** 선형 가정이 성립하지
않는 경우가 실제로 있다 -- 로봇이 곡선을 돌거나, 장애물이 화각에 들어오거나 나가거나,
**가림 위상(occlusion topology)이 바뀌면** 예측 변화가 gap에 **비선형**으로 생긴다. 그러면
절편은 "불안정"이 아니라 **"비선형 잔차"**를 담는다.

**그래서 둘을 고쳤다.**

1. **이 도구는 `temporal instability` 지표가 아니라 diagnostic이다.** 절편을 출력하되
   **"순수 불안정성"이라고 부르지 않는다.** 출력에서도 `g→0 외삽값`이라고만 쓴다.
2. **작은 gap에서만 국소 적합한다 -- gap 1·2·3.** 5·10까지 넣어 직선을 맞추면 위 비선형이
   기울기와 절편 **양쪽**을 오염시킨다.

**모션 교란이 아예 없는 축은 따로 있다** -- `tools/measure_perturbation_stability.py`
(same-frame perturbation consistency). **두 축을 반드시 같이 읽는다.**

## 왜 pose가 필요 없나

`sj_datasets_full-seq`가 **30 fps**라 인접 프레임의 ego 이동이 격자 0.7셀 수준이다. 즉
"프레임이 조금 바뀌었을 때 예측이 얼마나 바뀌나"를 **pose 없이** 물을 수 있다.
그리고 pose는 **쓰지 않기로 했다** -- LiDAR/LIO-SLAM 수집을 신뢰할 수 없고(§28.5.3),
"지나갔으니 free"는 **task 정의를 바꾼다**(사용자 결정).

## 품질 게이트 -- 이 축에서도 상수 지도가 1등이다

full-seq에는 **라벨이 없으므로** 이 시퀀스에서 품질을 잴 수 없다. 그래서 게이트는
**같은 체크포인트의 라벨 있는 val `iou_free`**를 옆에 적는 것으로 대신하고,
**constant map 행을 표에 직접 넣는다** -- 입력을 안 보므로 모든 gap에서 Δ가 정확히 0이다.
설계 문서 §15.6의 교훈("안정성만 읽으면 아무것도 안 배우는 것이 1등")이 여기에도 걸린다.

실행:

    python tools/measure_frame_gap_stability.py
    python tools/measure_frame_gap_stability.py --sequences=raws1 --n_anchors=100
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from fire import Fire

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.models\._utils")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.bev_gt.grid import ROBOT_GRID_SPEC as GRID_SPEC  # noqa: E402
from projects.common.polar import RAY_OK, build_ray_index, first_free_range  # noqa: E402
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    FINETUNE_CAMERA_NAMES,
    build_bev_masks,
)
from projects.datasets.simplebev_vox import ref_T_cam_from_ego_T_cam  # noqa: E402
from projects.geometry.double_sphere import load_cameras, load_ego_T_cams  # noqa: E402
from projects.datasets.simplebev_vox import height_config_for_ckpt_dirs  # noqa: E402
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.pixel_grid import convention_for_checkpoints  # noqa: E402
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402
from tools.render_prediction_video import (  # noqa: E402
    _model_input,
    list_frames,
    load_frame_images,
)

DEFAULT_FULLSEQ_ROOT = "dataset/sj_datasets_full-seq/val"
#: **gap 1·2·3만 쓴다.** 위 docstring의 (2) -- 더 큰 gap을 넣으면 비선형이 적합을 오염시킨다.
GAPS = (1, 2, 3)


def _csv(value):
    if isinstance(value, (tuple, list)):
        return tuple(str(v).strip() for v in value)
    return tuple(v.strip() for v in str(value).split(",") if v.strip())


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
    sequences="raws1,rawos3",
    fullseq_root=DEFAULT_FULLSEQ_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    # 앵커 프레임 수(시퀀스당). 앵커마다 gap 0~3의 4프레임을 추론하므로 forward 수는 4배다.
    # 시퀀스가 4000~4900프레임이라 전수 추론은 낭비고, 고르게 뽑으면 통로 전체를 덮는다.
    n_anchors=150,
    tau=0.5,
    step_cells=0.5,
    encoder_type="res101",
    # 라벨 있는 val에서 잰 같은 체크포인트의 `iou_free`. **품질 게이트로 표에 적는다.**
    #
    # **기본값을 숫자로 두면 안 된다.** full-seq에는 라벨이 없어 이 도구가 스스로 검증할
    # 수 없으므로, 다른 체크포인트로 돌렸을 때 **틀린 게이트를 조용히 찍는다** -- §28.6이
    # 재채점 도구에서 고친 것과 정확히 같은 종류의 결함이다. 그래서 `None`이면 숫자를
    # 찍지 않고 "미지정"이라고 적는다. 값은 `tools/measure_perturbation_stability.py`가
    # 같은 체크포인트에 대해 내놓는다.
    quality_iou_free=None,
    constant_map_iou_free=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cells, seeds = _csv(cells), tuple(int(s) for s in _csv(seeds))
    sequences = _csv(sequences)

    calib = Path(common_root) / "calibration/calib.yaml"
    orientation = Path(common_root) / "calibration/orientation.json"
    cameras_by_name = load_cameras(calib)
    cameras = [cameras_by_name[name] for name in FINETUNE_CAMERA_NAMES]
    ego_T_cams = load_ego_T_cams(calib)
    cam0_T_camXs = torch.from_numpy(np.stack(
        [ref_T_cam_from_ego_T_cam(ego_T_cams[name]) for name in FINETUNE_CAMERA_NAMES]
    )).float().unsqueeze(0).to(device)
    pix_T_cams = torch.eye(4).repeat(len(cameras), 1, 1).unsqueeze(0).to(device)

    # 라벨이 없으므로 `valid`는 리그 고정 마스크에서만 온다. `permanent_blind`는 `vis=0`이지
    # `valid=0`이 아니므로 여기서 빼지 않는다 -- 학습과 같은 계약이다.
    _, invalid = build_bev_masks(common_root, GRID_SPEC, FINETUNE_CAMERA_NAMES)
    valid = torch.from_numpy(~invalid).to(device).view(1, 1, *invalid.shape)

    rays = build_ray_index(GRID_SPEC, step_cells=float(step_cells))

    print(f"\n=== frame-gap consistency diagnostic (τ={tau}, gap {GAPS}, 30 fps) ===")
    print("**이것은 `temporal stability` 지표가 아니다**(§28.7). 절편을 '순수 불안정성'이라")
    print("부르지 않는다 -- 곡선 주행·가림 위상 변화가 gap에 비선형으로 들어오기 때문이다.")
    if quality_iou_free is None:
        print("품질 게이트: **미지정** -- `--quality_iou_free=`로 같은 체크포인트의 라벨 있는")
        print("            val `iou_free`를 넘겨라(`measure_perturbation_stability.py`가 낸다).")
        print("            안정성 숫자만 읽으면 상수 지도가 1등이다(설계 §15.6).\n")
    else:
        const_txt = ("" if constant_map_iou_free is None
                     else f"  (constant map {constant_map_iou_free:.4f})")
        print(f"품질 게이트: 라벨 있는 val `iou_free` = {quality_iou_free:.4f}{const_txt}\n")

    for cell in cells:
        convention, offset = convention_for_checkpoints(
            [_find_checkpoint(log_root, cell, s) for s in seeds])
        _height = height_config_for_ckpt_dirs(
            [Path(_find_checkpoint(log_root, cell, s)).parent for s in seeds])
        vox_util = build_double_sphere_vox_util(GRID_SPEC, cameras, device=device,
                                               pixel_convention=convention,
                                               pixel_offset=offset,                                               height_bins=_height["height_bins"],
                                               height_min_m=_height["height_min_m"],
                                               height_max_m=_height["height_max_m"])
        per_gap = {g: {"zero": [], "mean": [], "p90": [], "flip": []} for g in GAPS}

        for seed in seeds:
            model = ThreeClassSegnet(
                GRID_SPEC.n_rows, vox_util.Y, GRID_SPEC.n_cols, vox_util, use_radar=False,
                use_lidar=False, do_rgbcompress=True, encoder_type=encoder_type,
                rand_flip=False, num_classes=2).to(device)
            state = torch.load(_find_checkpoint(log_root, cell, seed),
                               map_location=device, weights_only=False)
            model.load_state_dict(state.get("model_state_dict", state), strict=True)
            model.eval()

            def predict(frame_dir):
                images = load_frame_images(frame_dir, FINETUNE_CAMERA_NAMES, orientation)
                rgb = _model_input(images, FINETUNE_CAMERA_NAMES).unsqueeze(0).to(device)
                with torch.no_grad():
                    # `- 0.5`는 forward 직전에 **정확히 한 번** 걸린다(§18.1의 사고).
                    _, _, logits, _, _ = model(rgb - 0.5, pix_T_cams, cam0_T_camXs, vox_util)
                pred_free = (torch.softmax(logits, dim=1)[:, 1:2] > tau) & valid.bool()
                r, s = first_free_range(pred_free[0, 0].cpu().numpy(), rays)
                return r, (s == RAY_OK)

            deltas = {g: [] for g in GAPS}
            flips = {g: [] for g in GAPS}
            for sequence in sequences:
                frames = list_frames(Path(fullseq_root) / sequence)
                usable = len(frames) - max(GAPS)
                if usable <= 0:
                    continue
                anchors = np.linspace(0, usable - 1, num=min(n_anchors, usable), dtype=int)
                for t in anchors:
                    r0, ok0 = predict(frames[t])
                    for g in GAPS:
                        rg, okg = predict(frames[t + g])
                        both = ok0 & okg
                        deltas[g].append(np.abs(r0[both] - rg[both]) * 100.0)
                        flips[g].append(float(np.mean(ok0 != okg)) * 100.0)

            for g in GAPS:
                d = np.concatenate(deltas[g]) if deltas[g] else np.array([np.nan])
                per_gap[g]["zero"].append(float(np.mean(d == 0.0)) * 100.0)
                per_gap[g]["mean"].append(float(np.mean(d)))
                per_gap[g]["p90"].append(float(np.percentile(d, 90)))
                per_gap[g]["flip"].append(float(np.mean(flips[g])))

        header = (f"{'gap':>5}{'Δt [ms]':>10}{'Δ=0 비율':>11}{'Δ mean':>10}"
                  f"{'Δ P90':>10}{'상태 불일치':>13}")
        print(f"[{cell}]  시퀀스 {','.join(sequences)} | 앵커 {n_anchors}/시퀀스 | 시드 {len(seeds)}개 평균")
        print(header)
        print("-" * len(header))
        means = []
        for g in GAPS:
            m = float(np.mean(per_gap[g]["mean"]))
            means.append(m)
            print(f"{g:>5}{g * 1000 / 30:>10.0f}{np.mean(per_gap[g]['zero']):>10.1f}%"
                  f"{m:>8.2f}cm{np.mean(per_gap[g]['p90']):>8.2f}cm"
                  f"{np.mean(per_gap[g]['flip']):>12.2f}%")
        print(f"{'constant map':>5}{'':>10}{100.0:>10.1f}%{0.0:>8.2f}cm{0.0:>8.2f}cm{0.0:>12.2f}%"
              "   <- **입력을 안 보므로 만점이다**")

        # gap 1·2·3에 대한 국소 선형 적합. **절편을 '순수 불안정성'이라 부르지 않는다.**
        slope, intercept = np.polyfit(np.array(GAPS, dtype=float), np.array(means), 1)
        print(f"\n  국소 선형 적합(gap 1·2·3): 기울기 {slope:+.2f} cm/frame"
              f" | **g→0 외삽값 {intercept:.2f} cm**")
        print("  ⚠ 이 외삽값을 **'순수 불안정성'이라고 부르지 않는다**(§28.7). 곡선 주행과")
        print("    가림 위상 변화가 gap에 비선형으로 들어오므로 절편은 비선형 잔차도 담는다.")
        print("    **모션 성분이 0인 축은 `measure_perturbation_stability.py`이고,**")
        print("    **그 결과와 반드시 같이 읽는다.**\n")

    print("""--- 읽는 법 (§28.7, 결과를 보기 전에 정했다) ---
  1. **이 표는 diagnostic이다.** 하나의 "안정성 점수"로 합치지 않는다.
  2. **기울기는 장면이 실제로 변한 양을 포함한다** -- 30 fps에서 gap 1이 ego 0.7셀
     이동이므로, 기울기가 0이 아닌 것은 정상이고 결함이 아니다.
  3. **외삽값은 상한으로만 읽는다.** 비선형 잔차가 섞여 있어 "입력이 같으면 예측도 같다"의
     척도로 쓸 수 없다 -- 그 척도는 same-frame perturbation 쪽이다.
  4. **constant map 행을 먼저 본다.** 모든 gap에서 Δ가 0이고 품질은 크게 낮다.""")


if __name__ == "__main__":
    Fire(main)
