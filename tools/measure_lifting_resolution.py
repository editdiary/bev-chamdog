"""lifting이 BEV에서 **얼마나 세밀하게 분해하는가**를 캘리브레이션 기하만으로 잰다 (CPU).

진단 문서 §27의 재현 스크립트다. **학습·체크포인트·데이터셋이 전혀 필요 없다** -- 실제
`calib.yaml`과 지면 가정만 쓴다.

세 가지를 낸다.

1. **`--what=resolution`** (§27.2) -- **특징맵 1픽셀이 BEV에서 몇 cm인가**, 거리별.
   이것이 "이미지 특징이 BEV에서 구분할 수 있는 최소 거리"이고, 다른 모든 cm 숫자를 읽는
   **기준**이다. 실측: 0~1 m 6 cm / 1~2 m 10.5 cm / 2~3 m 16 cm / 3~4 m 20~24 cm.
   **BEV 격자가 5 cm이므로 격자가 특징보다 2~5배 촘촘하다**(§27.3).

2. **`--what=warp`** (§27.1) -- 옛 규약(`legacy_index`)의 표본 어긋남이 BEV에서 몇 cm인가.
   BEV 셀 `p`가 읽어야 할 특징 좌표 `x(p)` 대신 `x' = x·W/(W-1) - 0.5`를 읽으므로, 셀 `p`에는
   `x(q) = x'`인 **다른 세계점 `q`의 특징**이 들어간다. 그 `p → q` 거리가 왜곡이다.
   국소 선형화로 푼다: `J·(q-p) = (x'-x, y'-y)`, `J`는 유한차분.
   실측 median 8~10 cm이지만 **1 특징픽셀의 1/3**이라 위 (1)의 바닥 아래다 -- 그래서
   성능으로 드러나지 않았다(§18.3.5의 평평한 스윕).
   **P99 이상(117 cm~)은 읽지 말 것** -- 시선이 지면에 거의 평행한 먼 셀이라 선형화가 깨진다.

3. **`--what=budget`** (§27.5) -- BEV 격자가 특징맵의 **몇 픽셀**에서 정보를 끌어오나.
   실측 6,912 px 중 **2,240 px (32.4 %)**, 특징 1 px당 BEV 셀 6.4개.
   **"68 %가 낭비"가 아니다** -- 수용영역이 넓어 ROI 밖 픽셀도 ROI 안 특징에 기여한다.
   맞는 진술은 **"ROI 위 표본 밀도가 특징맵 전체의 1/3"**이고 그것이 (1)을 만든다.

실행: python tools/measure_lifting_resolution.py            # 셋 다
      python tools/measure_lifting_resolution.py --what=resolution
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

# 실제 fine-tuning 형상: native 1280x720 -> 입력 512x288 -> stride 8 특징맵 64x36.
FEAT_W, FEAT_H = 64, 36
NATIVE_W, NATIVE_H = 1280, 720
_H = 1e-3  # 유한차분 간격 [m]


def _feature_coords(camera, ego_T_cam, forward, lateral):
    """ego 평면 위 점 -> 특징맵 좌표 `(x, y)`와 유효성. 학습 경로와 같은 배율을 쓴다."""
    pts = np.stack([forward.ravel(), lateral.ravel(),
                    np.zeros(forward.size), np.ones(forward.size)], axis=1)
    cam = (se3_inverse(ego_T_cam) @ pts.T).T[:, :3]
    u, v, valid = camera.project(cam)
    x = u * (FEAT_W / NATIVE_W)
    y = v * (FEAT_H / NATIVE_H)
    inside = valid & (x > -0.5) & (x < FEAT_W - 0.5) & (y > -0.5) & (y < FEAT_H - 0.5)
    return x.reshape(forward.shape), y.reshape(forward.shape), inside.reshape(forward.shape)


def _jacobian(camera, ego_T_cam, fg, lg):
    """`J = d(특징좌표)/d(전방, 횡방향)` [특징px per m] 와 기준 좌표·유효성."""
    x0, y0, ok = _feature_coords(camera, ego_T_cam, fg, lg)
    xf, yf, _ = _feature_coords(camera, ego_T_cam, fg + _H, lg)
    xl, yl, _ = _feature_coords(camera, ego_T_cam, fg, lg + _H)
    jxf, jyf = (xf - x0) / _H, (yf - y0) / _H
    jxl, jyl = (xl - x0) / _H, (yl - y0) / _H
    return x0, y0, ok, jxf, jyf, jxl, jyl, jxf * jyl - jxl * jyf


def _legacy(coord, size):
    """`normalize_grid2d` + `grid_sample(align_corners=False)` 합성이 실제로 읽는 위치."""
    return coord * size / (size - 1.0) - 0.5


def main(what="all"):
    if not _CALIB.exists():
        raise SystemExit(f"캘리브레이션이 없다: {_CALIB}")
    spec = ROBOT_GRID_SPEC
    cameras = load_cameras(_CALIB)
    ego_T_cams = load_ego_T_cams(_CALIB)
    forward_m, lateral_m = cell_centers_m(spec)
    fg, lg = np.meshgrid(forward_m, lateral_m, indexing="ij")
    rng = np.hypot(fg, lg)
    cell_cm = spec.cell_m * 100.0

    print(f"BEV 격자 {spec.n_rows}x{spec.n_cols} = {spec.n_rows*spec.n_cols} 셀, "
          f"셀 {cell_cm:.0f} cm | 전방 {spec.front_m} m / 후방 {spec.rear_m} m "
          f"/ 횡 ±{spec.half_width_m} m")
    print(f"특징맵 {FEAT_W}x{FEAT_H} (native {NATIVE_W}x{NATIVE_H}, 배율 "
          f"{NATIVE_W // FEAT_W}x)\n")

    if what in ("all", "resolution"):
        print("=== §27.2 lifting 고유 해상도 -- 특징맵 1픽셀 = BEV 몇 cm (거리별 median) ===")
        print("이것이 '이미지 특징이 BEV에서 구분할 수 있는 최소 거리'다. 다른 cm 숫자의 기준.")
        bands = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0)]
        print(f"  {'거리 [m]':<10} " + " ".join(f"{n:>9}" for n in FINETUNE_CAMERA_NAMES))
        for lo, hi in bands:
            cells = []
            for name in FINETUNE_CAMERA_NAMES:
                _, _, ok, jxf, jyf, jxl, jyl, det = _jacobian(
                    cameras[name], ego_T_cams[name], fg, lg)
                # 특징맵에서 1 px 옮기려면 BEV에서 몇 m 가야 하나 = J^-1의 두 방향 중
                # **더 잘 분해되는 쪽**(작은 쪽). 보수적으로 유리하게 잡는다.
                res_m = np.minimum(np.hypot(jyl, jyf) / np.abs(det),
                                   np.hypot(jxl, jxf) / np.abs(det))
                band = ok & (rng >= lo) & (rng < hi) & (np.abs(det) > 1e-9)
                cells.append(f"{np.median(res_m[band]) * 100:>7.1f}cm"
                             if band.sum() > 20 else "      -")
            print(f"  {lo:.0f}~{hi:.0f} m     " + " ".join(f"{c:>9}" for c in cells))
        print(f"\n  BEV 격자 셀은 {cell_cm:.0f} cm다. 고유 해상도가 그보다 크면 **격자가"
              f" 특징보다 촘촘하다**는 뜻이다 (§27.3).\n")

    if what in ("all", "warp"):
        print("=== §27.1 옛 규약(legacy_index)의 표본 어긋남이 BEV에서 몇 cm인가 ===")
        print(f"{'카메라':<8} {'보이는 셀':>9} {'왜곡 median':>12} {'P90':>8} "
              f"{'특징px median':>14}")
        pooled = []
        for name in FINETUNE_CAMERA_NAMES:
            x0, y0, ok, jxf, jyf, jxl, jyl, det = _jacobian(
                cameras[name], ego_T_cams[name], fg, lg)
            dx = _legacy(x0, FEAT_W) - x0
            dy = _legacy(y0, FEAT_H) - y0
            good = ok & (np.abs(det) > 1e-9)
            df = (jyl * dx - jxl * dy) / det          # 2x2 역행렬
            dl = (-jyf * dx + jxf * dy) / det
            disp_cm = np.hypot(df, dl)[good] * 100.0
            featpx = np.hypot(dx, dy)[good]
            pooled.append(disp_cm)
            print(f"{name:<8} {good.sum():>9} {np.median(disp_cm):>9.2f} cm "
                  f"{np.percentile(disp_cm, 90):>5.1f} {np.median(featpx):>11.3f} px")
        pooled = np.concatenate(pooled)
        print(f"\n  전체: median {np.median(pooled):.2f} cm "
              f"(= 격자 {np.median(pooled)/cell_cm:.2f}셀) | "
              f"P90 {np.percentile(pooled, 90):.2f} cm")
        print("  **P99 이상은 읽지 말 것** -- 시선이 지면에 거의 평행한 먼 셀이라"
              " 선형화가 깨진다.")
        print("  **위 해상도 표와 나란히 읽는다** -- 어긋남은 1 특징픽셀의 약 1/3이고,"
              " 그래서 성능으로\n  드러나지 않았다(§18.3.5의 평평한 스윕).\n")

    if what in ("all", "budget"):
        print("=== §27.5 BEV 격자가 특징맵의 몇 픽셀에서 정보를 끌어오나 ===")
        print(f"{'카메라':<8} {'보이는 셀':>9} {'쓰는 특징px':>12} {'예산 대비':>9} "
              f"{'셀/특징px':>10}")
        total_used = 0
        for name in FINETUNE_CAMERA_NAMES:
            x, y, ok, *_ = _jacobian(cameras[name], ego_T_cams[name], fg, lg)
            # 이중선형은 픽셀 둘씩 만지지만 "정보량 상한"은 중심 픽셀의 개수다.
            rows = np.round(y[ok]).astype(int)
            cols = np.round(x[ok]).astype(int)
            used = len(set(zip(rows.tolist(), cols.tolist())))
            total_used += used
            print(f"{name:<8} {ok.sum():>9} {used:>12} "
                  f"{used/(FEAT_W*FEAT_H)*100:>8.1f}% {ok.sum()/max(used,1):>10.1f}")
        budget = FEAT_W * FEAT_H * len(FINETUNE_CAMERA_NAMES)
        n_cells = spec.n_rows * spec.n_cols
        print(f"\n  특징 예산 {budget} px 중 **{total_used} px "
              f"({total_used/budget*100:.1f} %)만** BEV 격자가 만진다.")
        print(f"  BEV 셀 {n_cells}개가 특징 {total_used} px에서 나온다 "
              f"-> 특징 1 px당 셀 {n_cells/total_used:.1f}개 (나머지는 보간).")
        print("  **'68 %가 낭비'가 아니다** -- 수용영역이 넓어 ROI 밖 픽셀도 ROI 안 특징에")
        print("  기여한다. 맞는 진술은 **'ROI 위 표본 밀도가 특징맵 전체의 1/3'**이다.")


if __name__ == "__main__":
    Fire(main)
