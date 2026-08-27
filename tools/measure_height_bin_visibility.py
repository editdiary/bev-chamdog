"""높이 bin을 늘렸을 때 **그 bin이 실제로 이미지 안에 투영되는가**를 기하만으로 잰다 (CPU).

`Y>1` 실험의 **사전 확인**이다. 학습·체크포인트·데이터셋이 전혀 필요 없고 실제
`calib.yaml`만 쓴다. `tools/measure_lifting_resolution.py`(§27)와 같은 방식이다.

**왜 필요한가.** `Y=1`은 `ego z=0` 한 평면에서만 표본하지만, 라벨 파이프라인
(`dataset/sj_datasets/common/temp/slab_label.py`)의 occupancy는 **지상 0.87~1.67 m
슬래브의 기둥 질의**다("슬래브 [z_ref, z_ref+thick] 은 로봇이 통과해야 하는 높이 구간").
그 대역을 직접 표본하려고 `Y`를 늘리는데, 카메라가 지상 0.87 m쯤에 있으므로 **로봇 가까이
있는 셀의 높은 지점은 화각 밖으로 나갈 수 있다.**

화각 밖이면 `double_sphere_vox.unproject_image_to_mem`이 `valid_mem = 0`으로 마스킹하고
그 bin에는 **항상 0**이 들어간다. 그러면 `bev_compressor`가 그 채널을 무시하도록 학습되고,
실험 결과는 "Y=1과 같음"이 나오는데 그것은 **가설 기각이 아니라 "정보가 애초에 안 들어옴"**
이다. 둘을 구분하려면 학습 전에 이 표를 봐야 한다.

**유효성 판정은 학습 경로와 같다** -- `DoubleSphereCamera.project`의 정의역 조건 + 특징맵
경계 `(-0.5, W-0.5)`. 배율도 같다(native 1280x720 -> 특징맵 64x36).

실행:
    python tools/measure_height_bin_visibility.py
    python tools/measure_height_bin_visibility.py --y_min=-0.125 --y_max=1.875 --n_bins=8
"""
import sys
from pathlib import Path

import numpy as np
from fire import Fire

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.bev_gt.grid import ROBOT_GRID_SPEC, cell_centers_m  # noqa: E402
from projects.geometry.double_sphere import (  # noqa: E402
    FINETUNE_CAMERA_NAMES,
    load_cameras,
    load_ego_T_cams,
    se3_inverse,
)

_CALIB = _REPO_ROOT / "dataset/sj_datasets/common/calibration/calib.yaml"

# 실제 fine-tuning 형상. `measure_lifting_resolution.py`와 같은 값이다.
FEAT_W, FEAT_H = 64, 36
NATIVE_W, NATIVE_H = 1280, 720

# 라벨이 판정하는 높이 대역 [m, 지면 기준]. `slab_label.py`의 z_ref(=LiDAR 수평면,
# 지상 약 0.87 m)와 thick=0.8에서 온다.
LABEL_BAND_M = (0.87, 1.67)


def _visible(camera, ego_T_cam, forward, lateral, height):
    """ego 프레임 점 -> 특징맵 안에 들어오는가 (bool, forward와 같은 shape).

    `_feature_coords`(`measure_lifting_resolution.py`)와 같되 z를 인자로 받는다.
    """
    pts = np.stack([forward.ravel(), lateral.ravel(),
                    np.full(forward.size, float(height)), np.ones(forward.size)], axis=1)
    cam = (se3_inverse(ego_T_cam) @ pts.T).T[:, :3]
    u, v, valid = camera.project(cam)
    x = u * (FEAT_W / NATIVE_W)
    y = v * (FEAT_H / NATIVE_H)
    inside = valid & (x > -0.5) & (x < FEAT_W - 0.5) & (y > -0.5) & (y < FEAT_H - 0.5)
    return inside.reshape(forward.shape)


def bin_centers(y_min: float, y_max: float, n_bins: int):
    """`Vox_util.get_mem_T_ref`와 같은 규약 -- 복셀 **중심**에서 표본한다.

    `center_T_ref[:,1,3] = -YMIN - vox_size_Y/2` 이므로 mem index i의 ref 좌표는
    `YMIN + vox_size*(i + 0.5)`다.
    """
    size = (y_max - y_min) / float(n_bins)
    return y_min + size * (np.arange(n_bins) + 0.5), size


def main(y_min: float = -0.125, y_max: float = 1.875, n_bins: int = 8):
    if not _CALIB.exists():
        raise SystemExit(f"캘리브레이션이 없다: {_CALIB}")
    spec = ROBOT_GRID_SPEC
    cameras = load_cameras(_CALIB)
    ego_T_cams = load_ego_T_cams(_CALIB)
    forward_m, lateral_m = cell_centers_m(spec)
    fg, lg = np.meshgrid(forward_m, lateral_m, indexing="ij")
    rng = np.hypot(fg, lg)
    n_cells = fg.size

    centers, size = bin_centers(y_min, y_max, n_bins)
    print(f"BEV 격자 {spec.n_rows}x{spec.n_cols} = {n_cells} 셀, 셀 {spec.cell_m*100:.0f} cm "
          f"| 전방 {spec.front_m} m / 후방 {spec.rear_m} m / 횡 ±{spec.half_width_m} m")
    print(f"특징맵 {FEAT_W}x{FEAT_H} | 카메라 {', '.join(FINETUNE_CAMERA_NAMES)}")
    print(f"높이 범위 [{y_min:+.3f}, {y_max:+.3f}] m, {n_bins} bin, bin 크기 {size:.3f} m")
    print(f"라벨 판정 대역(slab_label.py) = 지상 {LABEL_BAND_M[0]}~{LABEL_BAND_M[1]} m\n")

    bands = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0)]
    per_bin = {}
    for z in centers:
        any_cam = np.zeros(fg.shape, bool)
        for name in FINETUNE_CAMERA_NAMES:
            any_cam |= _visible(cameras[name], ego_T_cams[name], fg, lg, z)
        per_bin[z] = any_cam

    print("=== bin별 유효 투영 비율 (카메라 3대 중 하나라도 보면 유효) ===")
    header = f"  {'높이 [m]':<10} {'전체':>7}  " + " ".join(f"{f'{lo:.0f}~{hi:.0f}m':>7}"
                                                            for lo, hi in bands)
    print(header)
    for z in centers:
        ok = per_bin[z]
        cols = []
        for lo, hi in bands:
            band = (rng >= lo) & (rng < hi)
            cols.append(f"{100*ok[band].mean():>6.1f}%" if band.sum() else "      -")
        mark = "  <- 라벨 대역" if LABEL_BAND_M[0] <= z <= LABEL_BAND_M[1] else ""
        print(f"  {z:>+8.3f}   {100*ok.mean():>6.1f}%  " + " ".join(f"{c:>7}" for c in cols) + mark)

    print("\n=== bin별 카메라 수 (그 셀을 보는 카메라가 몇 대인가, 전체 평균) ===")
    print(f"  {'높이 [m]':<10} {'0대':>7} {'1대':>7} {'2대':>7} {'3대':>7}")
    for z in centers:
        counts = np.zeros(fg.shape, int)
        for name in FINETUNE_CAMERA_NAMES:
            counts += _visible(cameras[name], ego_T_cams[name], fg, lg, z).astype(int)
        row = [f"{100*(counts == k).mean():>6.1f}%" for k in range(4)]
        print(f"  {z:>+8.3f}   " + " ".join(f"{c:>7}" for c in row))

    base = per_bin[centers[0]]
    print(f"\n=== 읽는 법 ===")
    print(f"  bin 0 (z={centers[0]:+.3f} m)이 현재 `Y=1`의 표본 평면이고 유효 "
          f"{100*base.mean():.1f}%다 -- 이것이 기준선이다.")
    label_bins = [z for z in centers if LABEL_BAND_M[0] <= z <= LABEL_BAND_M[1]]
    if label_bins:
        lab = np.mean([per_bin[z].mean() for z in label_bins])
        print(f"  라벨 대역 bin {len(label_bins)}개(z={', '.join(f'{z:.2f}' for z in label_bins)})의 "
              f"평균 유효 비율은 {100*lab:.1f}%다.")
        print(f"  이 값이 낮으면 **Y를 늘려도 라벨 대역 정보가 안 들어온다** -- 실험이 "
              f"'변화 없음'을 내도 가설 기각으로 읽으면 안 된다.")
    print(f"  높은 bin이 근거리에서 특히 낮게 나오는 것이 예상되는 무늬다 "
          f"(카메라가 지상 0.87 m라 코앞 높은 점은 화각 밖).")


if __name__ == "__main__":
    Fire(main)
