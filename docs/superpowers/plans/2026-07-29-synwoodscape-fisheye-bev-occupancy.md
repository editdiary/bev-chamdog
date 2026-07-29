# SynWoodScape Fisheye 투영 검증 및 BEV Occupancy GT 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SynWoodScape의 `radial_poly` 어안 카메라 project/unproject 함수를 검증하고, `_BEV.png` semantic 라벨로부터 로봇 기준(전방3m/후방1m/좌우±1m, 5cm cell) BEV occupancy(drivable/non-drivable) GT를 생성하는 파이프라인을 만든다.

**Architecture:** WoodScape 공식 `projection.py`(submodule, MIT 유사 라이선스)를 `projects/geometry/fisheye.py`에서 얇게 래핑해 재사용한다. LiDAR는 fisheye project/unproject 검증(Phase 1)에만 쓴다. BEV occupancy GT(Phase 2)는 LiDAR를 쓰지 않고, `_unuserd/instance_annotations` + `_unuserd/box_3d_annotations`로 1회성 스케일·원점 보정을 한 뒤 `semantic_annotations/gtLabels/*_BEV.png`를 직접 크롭해서 만든다.

**Tech Stack:** Python 3.11, numpy, scipy(`Rotation.from_quat`), opencv/Pillow, pytest. GPU/torch 불필요(순수 수치 연산).

## Global Constraints

- conda 환경 `bev-chamdog`(Python 3.11)에서 작업한다. 새 의존성은 `requirements.txt`에 추가하고 `constraints.txt`(`numpy<2`, `opencv-python<5`)를 지킨다.
- `third_party/`, `mmdetection3d/`는 git submodule — **직접 수정 금지**. WoodScape의 `projection.py`는 `sys.path.insert`로 임포트만 하고 코드를 고치지 않는다(기존 `simple_bev` 임포트 패턴과 동일, `docs/project_structure.md` 참고).
- 커스텀 코드는 `projects/`(재사용 모듈), `tools/`(스크립트)에 둔다. 스크립트는 저장소 루트에서 실행한다(`python tools/<script>.py`).
- 생성물(시각화 이미지, GT npy 등)은 `outputs/`(이미 `.gitignore`에 포함) 아래에 쓴다.
- 커밋 메시지는 영어로 작성한다. Task 단위로 한 번에 커밋한다(스텝마다 쪼개지 않음).
- 이 계획은 다음 스펙 문서를 구현한다: `docs/superpowers/specs/2026-07-29-synwoodscape-fisheye-bev-occupancy-design.md`.

---

### Task 1: 동차좌표 변환 헬퍼 + pytest 설정

**Files:**
- Create: `projects/geometry/__init__.py`
- Create: `projects/geometry/frames.py`
- Test: `tests/geometry/test_frames.py`
- Create: `pytest.ini`
- Modify: `requirements.txt` (pytest 추가)

**Interfaces:**
- Produces: `apply_4x4(transform: np.ndarray, points: np.ndarray) -> np.ndarray` — (4,4) 변환을 (N,3) 점에 적용
- Produces: `invert_4x4(transform: np.ndarray) -> np.ndarray` — (4,4) rigid transform 역행렬

- [ ] **Step 1: pytest 설정 추가**

`pytest.ini` (저장소 루트):

```ini
[pytest]
pythonpath = .
testpaths = tests
```

`requirements.txt`의 "시각화 / 지표" 섹션 아래에 추가:

```
pytest==8.3.4
```

- [ ] **Step 2: 의존성 설치**

Run: `pip install -c constraints.txt -r requirements.txt && pip check`

- [ ] **Step 3: 실패하는 테스트 작성**

`projects/geometry/__init__.py`: (빈 파일)

`tests/geometry/test_frames.py`:

```python
import numpy as np

from projects.geometry.frames import apply_4x4, invert_4x4


def test_apply_4x4_translation_only():
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    result = apply_4x4(transform, points)

    np.testing.assert_allclose(result, [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])


def test_invert_4x4_round_trip():
    rng = np.random.default_rng(0)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [4.0, -1.0, 2.5]
    points = rng.normal(size=(5, 3))

    round_tripped = apply_4x4(invert_4x4(transform), apply_4x4(transform, points))

    np.testing.assert_allclose(round_tripped, points, atol=1e-10)
```

- [ ] **Step 4: 테스트 실패 확인**

Run: `pytest tests/geometry/test_frames.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'projects.geometry.frames'`)

- [ ] **Step 5: 최소 구현**

`projects/geometry/frames.py`:

```python
import numpy as np


def apply_4x4(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to an (N, 3) array of points."""
    points = np.asarray(points, dtype=np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points, ones], axis=1)
    transformed = points_h @ np.asarray(transform, dtype=np.float64).T
    return transformed[:, :3]


def invert_4x4(transform: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid-body homogeneous transform."""
    transform = np.asarray(transform, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse
```

- [ ] **Step 6: 테스트 통과 확인**

Run: `pytest tests/geometry/test_frames.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: 커밋**

```bash
git add pytest.ini requirements.txt projects/geometry/__init__.py projects/geometry/frames.py tests/geometry/test_frames.py
git commit -m "feat: add homogeneous transform helpers and pytest setup"
```

---

### Task 2: LiDAR world→ego 변환 + 좌표계 규약 회귀 테스트

**Files:**
- Modify: `projects/geometry/frames.py`
- Modify: `tests/geometry/test_frames.py`

**Interfaces:**
- Consumes: `apply_4x4`, `invert_4x4` (Task 1)
- Produces: `LIDAR_MOUNT_OFFSET_EGO: np.ndarray` — ego frame 기준 LiDAR 장착 오프셋 `(0.0, 0.0, 2.0)`
- Produces: `world_points_to_ego(points_world: np.ndarray, world_T_lidar: np.ndarray) -> np.ndarray`
- Produces: `parse_vehicle_location(vehicle_data_txt_path) -> np.ndarray` — `vehicle_data/rgb_images/*.txt`의 `Location(x=...,y=...,z=...)`를 파싱

이 Task는 스펙 §3.3에서 확인한 "`lidar_data/*.pkl`의 `transform` = world_T_lidar" 관계를 자동화된 회귀 테스트로 고정한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/geometry/test_frames.py`에 추가:

```python
import pickle
import re
from pathlib import Path

import pytest

from projects.geometry.frames import (
    LIDAR_MOUNT_OFFSET_EGO,
    parse_vehicle_location,
    world_points_to_ego,
)

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
requires_dataset = pytest.mark.skipif(
    not DATASET_ROOT.exists(), reason="SynWoodScape dataset not available locally"
)


def test_world_points_to_ego_maps_lidar_origin_to_mount_offset():
    world_T_lidar = np.eye(4)
    world_T_lidar[:3, 3] = [10.0, -5.0, 3.0]
    points_world = np.array([[10.0, -5.0, 3.0]])  # the LiDAR sensor's own origin, in world frame

    result = world_points_to_ego(points_world, world_T_lidar)

    np.testing.assert_allclose(result, [[0.0, 0.0, 2.0]], atol=1e-10)


@requires_dataset
def test_lidar_transform_matches_vehicle_pose_for_sample_00000():
    vehicle_location = parse_vehicle_location(
        DATASET_ROOT / "vehicle_data" / "rgb_images" / "00000.txt"
    )
    with open(DATASET_ROOT / "lidar_data" / "00000.pkl", "rb") as f:
        lidar_record = pickle.load(f)
    lidar_translation = np.asarray(lidar_record["transform"])[:3, 3]

    np.testing.assert_allclose(lidar_translation[:2], vehicle_location[:2], atol=0.01)
    z_diff = lidar_translation[2] - vehicle_location[2]
    assert abs(z_diff - LIDAR_MOUNT_OFFSET_EGO[2]) < 0.05
```

(`import numpy as np`는 파일 상단에 이미 있으므로 중복 추가하지 않는다.)

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/geometry/test_frames.py -v`
Expected: FAIL (`ImportError: cannot import name 'world_points_to_ego'`)

- [ ] **Step 3: 구현**

`projects/geometry/frames.py`에 추가:

```python
LIDAR_MOUNT_OFFSET_EGO = np.array([0.0, 0.0, 2.0])  # readme.txt: LiDAR is mounted at ego (x=0, y=0, z=2.0)


def world_points_to_ego(points_world: np.ndarray, world_T_lidar: np.ndarray) -> np.ndarray:
    """Convert LiDAR points (CARLA world frame) into the ego-vehicle frame.

    `world_T_lidar` is the `transform` field stored in each lidar_data/*.pkl
    (confirmed to be the LiDAR-sensor-to-world pose by cross-referencing
    vehicle_data — see docs/superpowers/specs/2026-07-29-synwoodscape-fisheye-bev-occupancy-design.md §3.3).
    """
    points_lidar = apply_4x4(invert_4x4(world_T_lidar), points_world)
    return points_lidar + LIDAR_MOUNT_OFFSET_EGO


def parse_vehicle_location(vehicle_data_txt_path) -> np.ndarray:
    """Parse the ego world Location(x=..., y=..., z=...) out of a vehicle_data/rgb_images/*.txt file."""
    text = Path(vehicle_data_txt_path).read_text()
    match = re.search(r"Location\(x=([-\d.]+), y=([-\d.]+), z=([-\d.]+)\)", text)
    return np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])
```

파일 상단 import에 `import re`와 `from pathlib import Path`를 추가한다.

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/geometry/test_frames.py -v`
Expected: PASS (dataset이 로컬에 있으면 4 passed, 없으면 3 passed + 1 skipped)

- [ ] **Step 5: 커밋**

```bash
git add projects/geometry/frames.py tests/geometry/test_frames.py
git commit -m "feat: add LiDAR world-to-ego conversion with a coordinate-convention regression test"
```

---

### Task 3: SynWoodScape fisheye 캘리브레이션 로더 + 투영 정확성 테스트

**Files:**
- Create: `projects/geometry/fisheye.py`
- Test: `tests/geometry/test_fisheye.py`

**Interfaces:**
- Produces: `load_camera(calibration_json_path) -> projection.Camera` — WoodScape의 `Camera` 객체(`.project_3d_to_2d`, `.project_2d_to_3d`, `.cx`, `.cy`, `.width`, `.height`, `.aspect_ratio`, `.translation`, `.rotation` 보유)

WoodScape 공식 구현(`third_party/datasets/WoodScape/scripts/calibration/projection.py`)의 `Camera`/`RadialPolyCamProjection`을 그대로 임포트해서 쓴다 — radial_poly project/unproject 수식을 직접 재구현하지 않는다(같은 수식을 다시 짜면 버그 위험만 늘어남). `Camera.project_3d_to_2d(world_points)`는 **ego(vehicle) frame 점**을 받아 camera frame 변환까지 내부에서 처리한다(`world_points`라는 이름은 WoodScape 코드의 명명일 뿐, 실제로는 "카메라 pose가 표현된 좌표계" = 우리의 ego frame을 뜻한다).

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/geometry/test_fisheye.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from projects.geometry.fisheye import load_camera

CALIB_DIR = Path("dataset/synwoodscape/SynWoodScape_V0.1.0/calibration_data")
requires_dataset = pytest.mark.skipif(
    not CALIB_DIR.exists(), reason="SynWoodScape dataset not available locally"
)


@requires_dataset
def test_load_camera_parses_fv_json():
    cam = load_camera(CALIB_DIR / "FV.json")

    assert cam.width == 1280
    assert cam.height == 966
    assert abs(cam.aspect_ratio - 1.00021) < 1e-6
    np.testing.assert_allclose(cam.translation, [1.92, 0.0, 0.9])


@requires_dataset
def test_project_point_on_optical_axis_lands_on_principal_point():
    cam = load_camera(CALIB_DIR / "FV.json")
    point_in_camera_frame = np.array([0.0, 0.0, 5.0])  # 5m straight ahead of the lens
    point_in_ego_frame = cam.translation + cam.rotation @ point_in_camera_frame

    pixel = cam.project_3d_to_2d(point_in_ego_frame[np.newaxis, :])

    np.testing.assert_allclose(pixel[0], [cam.cx, cam.cy], atol=1e-6)


@requires_dataset
def test_project_then_unproject_round_trip():
    cam = load_camera(CALIB_DIR / "FV.json")
    rng = np.random.default_rng(1)
    offsets_in_camera_frame = rng.uniform(-1.0, 1.0, size=(20, 2))
    depths = rng.uniform(5.0, 15.0, size=20)
    points_in_camera_frame = np.column_stack([offsets_in_camera_frame, depths])
    points_in_ego_frame = (cam.rotation @ points_in_camera_frame.T).T + cam.translation

    pixels = cam.project_3d_to_2d(points_in_ego_frame)
    norms = np.linalg.norm(points_in_ego_frame - cam.translation, axis=1)
    recovered = cam.project_2d_to_3d(pixels, norms)

    np.testing.assert_allclose(recovered, points_in_ego_frame, atol=1e-4)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/geometry/test_fisheye.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'projects.geometry.fisheye'`)

- [ ] **Step 3: 구현**

`projects/geometry/fisheye.py`:

```python
import json
import sys
from pathlib import Path

_WOODSCAPE_CALIB_DIR = (
    Path(__file__).resolve().parents[2] / "third_party/datasets/WoodScape/scripts/calibration"
)
if str(_WOODSCAPE_CALIB_DIR) not in sys.path:
    sys.path.insert(0, str(_WOODSCAPE_CALIB_DIR))

from projection import Camera, RadialPolyCamProjection  # noqa: E402  (WoodScape submodule; see docs/project_structure.md)
from scipy.spatial.transform import Rotation as SciRot  # noqa: E402


def load_camera(calibration_json_path) -> Camera:
    """Build a WoodScape `Camera` from a SynWoodScape calibration_data/*.json file.

    SynWoodScape renames the "translation" key to
    "translation used in CARLA (CARLA reference)" compared to the real
    WoodScape calibration format, so `projection.read_cam_from_json` cannot
    be reused as-is.
    """
    with open(calibration_json_path) as f:
        config = json.load(f)

    intrinsic = config["intrinsic"]
    extrinsic = config["extrinsic"]
    translation = extrinsic.get("translation", extrinsic.get("translation used in CARLA (CARLA reference)"))

    return Camera(
        rotation=SciRot.from_quat(extrinsic["quaternion"]).as_matrix(),
        translation=translation,
        lens=RadialPolyCamProjection(
            [intrinsic["k1"], intrinsic["k2"], intrinsic["k3"], intrinsic["k4"]]
        ),
        size=(intrinsic["width"], intrinsic["height"]),
        principle_point=(intrinsic["cx_offset"], intrinsic["cy_offset"]),
        aspect_ratio=intrinsic["aspect_ratio"],
    )
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/geometry/test_fisheye.py -v`
Expected: PASS (3 passed) — 실제로 이 3개 테스트는 계획 작성 중 FV.json으로 직접 실행해 통과를 확인했다(on-axis pixel이 정확히 `(cx, cy)`, round-trip 최대 오차 `~5e-15`).

- [ ] **Step 5: 커밋**

```bash
git add projects/geometry/fisheye.py tests/geometry/test_fisheye.py
git commit -m "feat: wrap WoodScape radial_poly Camera for SynWoodScape calibration files"
```

---

### Task 4: 클래스 일치율 지표 헬퍼

**Files:**
- Create: `projects/common/__init__.py`
- Create: `projects/common/metrics.py`
- Test: `tests/common/test_metrics.py`

**Interfaces:**
- Produces: `class_consistency_rate(predicted: np.ndarray, reference: np.ndarray, valid_mask: np.ndarray | None = None) -> float`

Phase 1의 "semantic consistency rate"(스펙 §3.4-4)에 쓰는 작은 공용 지표. 재사용 가능하도록 `projects/common/`에 둔다.

- [ ] **Step 1: 실패하는 테스트 작성**

`projects/common/__init__.py`: (빈 파일)

`tests/common/test_metrics.py`:

```python
import numpy as np

from projects.common.metrics import class_consistency_rate


def test_class_consistency_rate_all_match():
    predicted = np.array([1, 2, 3])
    reference = np.array([1, 2, 3])

    assert class_consistency_rate(predicted, reference) == 1.0


def test_class_consistency_rate_partial_match():
    predicted = np.array([1, 2, 3, 4])
    reference = np.array([1, 0, 3, 0])

    assert class_consistency_rate(predicted, reference) == 0.5


def test_class_consistency_rate_respects_valid_mask():
    predicted = np.array([1, 99, 3])
    reference = np.array([1, 2, 3])
    valid_mask = np.array([True, False, True])

    assert class_consistency_rate(predicted, reference, valid_mask) == 1.0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/common/test_metrics.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'projects.common.metrics'`)

- [ ] **Step 3: 구현**

`projects/common/metrics.py`:

```python
import numpy as np


def class_consistency_rate(predicted: np.ndarray, reference: np.ndarray, valid_mask: np.ndarray = None) -> float:
    """Fraction of entries where `predicted` equals `reference`, restricted to `valid_mask` if given."""
    predicted = np.asarray(predicted)
    reference = np.asarray(reference)
    matches = predicted == reference
    if valid_mask is not None:
        matches = matches[np.asarray(valid_mask, dtype=bool)]
    return float(matches.mean()) if matches.size else float("nan")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/common/test_metrics.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 커밋**

```bash
git add projects/common/__init__.py projects/common/metrics.py tests/common/test_metrics.py
git commit -m "feat: add shared class-consistency-rate metric helper"
```

---

### Task 5: Fisheye 투영 검증 스크립트 (Phase 1 실행)

**Files:**
- Create: `tools/verify_fisheye_projection.py`

**Interfaces:**
- Consumes: `projects.geometry.frames.world_points_to_ego`, `projects.geometry.fisheye.load_camera`, `projects.common.metrics.class_consistency_rate`

**동작**: 지정한 샘플 인덱스 각각에 대해, LiDAR 포인트(+label)를 ego frame으로 옮기고 4대 카메라(FV/MVL/MVR/RV) 각각에 투영해 (a) 실제 RGB 이미지 위에 라벨 색상 오버레이 PNG를 `outputs/fisheye_verification/`에 저장하고 (b) `semantic_annotations/gtLabels`와의 class 일치율을 출력한다.

- [ ] **Step 1: 스크립트 작성**

`tools/verify_fisheye_projection.py`:

```python
"""Phase 1 검증: SynWoodScape LiDAR 포인트를 4대 fisheye 카메라에 재투영해
   투영 함수의 정확성을 육안 오버레이 + semantic consistency rate로 확인한다.

Run: python tools/verify_fisheye_projection.py --samples 00000 00001 00002
"""
import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from projects.common.metrics import class_consistency_rate
from projects.geometry.fisheye import load_camera
from projects.geometry.frames import world_points_to_ego

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
CAMERAS = ["FV", "MVL", "MVR", "RV"]
OUTPUT_DIR = Path("outputs/fisheye_verification")

# readme.txt 팔레트에서 뽑은 대표 색상 (BGR, cv2 저장용). 없는 class는 회색으로 표시한다.
PALETTE_BGR = {
    6: (50, 234, 157), 7: (128, 64, 128), 8: (232, 35, 244), 1: (70, 70, 70),
    9: (35, 142, 107), 10: (142, 0, 0), 12: (0, 220, 220),
}
DEFAULT_COLOR_BGR = (128, 128, 128)


def verify_sample(sample_idx: str) -> None:
    with open(DATASET_ROOT / "lidar_data" / f"{sample_idx}.pkl", "rb") as f:
        lidar = pickle.load(f)
    points_ego = world_points_to_ego(lidar["points"], np.asarray(lidar["transform"]))
    labels = lidar["labels"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for cam_name in CAMERAS:
        cam = load_camera(DATASET_ROOT / "calibration_data" / f"{cam_name}.json")
        pixels = cam.project_3d_to_2d(points_ego)

        valid = ~np.isnan(pixels[:, 0])
        in_bounds = (
            valid
            & (pixels[:, 0] >= 0) & (pixels[:, 0] < cam.width)
            & (pixels[:, 1] >= 0) & (pixels[:, 1] < cam.height)
        )
        n_in_bounds = int(in_bounds.sum())
        if n_in_bounds == 0:
            print(f"[{sample_idx}/{cam_name}] no points landed in the image, skipping")
            continue

        image_path = DATASET_ROOT / "rgb_images" / f"{sample_idx}_{cam_name}.png"
        image = cv2.imread(str(image_path))
        gt_path = DATASET_ROOT / "semantic_annotations" / "gtLabels" / f"{sample_idx}_{cam_name}.png"
        gt_labels_image = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

        px = pixels[in_bounds, 0].round().astype(int)
        py = pixels[in_bounds, 1].round().astype(int)
        point_labels = labels[in_bounds]
        gt_at_pixel = gt_labels_image[py, px]

        rate = class_consistency_rate(point_labels, gt_at_pixel)
        print(f"[{sample_idx}/{cam_name}] n_in_bounds={n_in_bounds} semantic_consistency_rate={rate:.3f}")

        overlay = image.copy()
        for u, v, label in zip(px, py, point_labels):
            color = PALETTE_BGR.get(int(label), DEFAULT_COLOR_BGR)
            cv2.circle(overlay, (int(u), int(v)), radius=2, color=color, thickness=-1)
        out_path = OUTPUT_DIR / f"{sample_idx}_{cam_name}_overlay.png"
        cv2.imwrite(str(out_path), overlay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=["00000", "00001", "00002"])
    args = parser.parse_args()

    for sample_idx in args.samples:
        verify_sample(sample_idx)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행**

Run: `python tools/verify_fisheye_projection.py --samples 00000 00001 00002`
Expected: 각 샘플×카메라에 대해 `semantic_consistency_rate` 값이 출력되고 `outputs/fisheye_verification/*_overlay.png`가 생성된다. 에러 없이 끝나야 한다.

- [ ] **Step 3: 육안 확인**

`outputs/fisheye_verification/00000_FV_overlay.png` 등을 열어, 색칠된 점들이 이미지의 도로/차선/건물 등 실제 내용과 위치적으로 맞아떨어지는지 확인한다. `semantic_consistency_rate`가 여러 샘플·카메라에서 대체로 0.9 이상이면 §3.5 완료 기준 충족. 크게 벗어나면(예: 절반 이하) 좌표계 부호(quaternion 순서, extrinsic 방향) 문제일 가능성이 높으니 스펙 §3.3을 다시 점검한다.

- [ ] **Step 4: 커밋**

```bash
git add tools/verify_fisheye_projection.py
git commit -m "feat: add Phase 1 fisheye projection verification script"
```

---

### Task 6: Occupancy 그리드 스펙 + drivable class remap

**Files:**
- Create: `projects/bev_gt/__init__.py`
- Create: `projects/bev_gt/grid.py`
- Test: `tests/bev_gt/test_grid.py`

**Interfaces:**
- Produces: `DRIVABLE_CLASS_IDS: frozenset[int]` — `{6, 7}` (road line, road)
- Produces: `OccupancyGridSpec` dataclass (`front_m`, `rear_m`, `half_width_m`, `cell_m`, properties `n_rows`/`n_cols`)
- Produces: `ROBOT_GRID_SPEC: OccupancyGridSpec` — 전방3m/후방1m/좌우±1m/5cm
- Produces: `remap_semantic_to_occupancy(semantic_labels: np.ndarray, drivable_class_ids=DRIVABLE_CLASS_IDS) -> np.ndarray`

- [ ] **Step 1: 실패하는 테스트 작성**

`projects/bev_gt/__init__.py`: (빈 파일)

`tests/bev_gt/test_grid.py`:

```python
import numpy as np

from projects.bev_gt.grid import ROBOT_GRID_SPEC, remap_semantic_to_occupancy


def test_robot_grid_spec_dimensions():
    assert ROBOT_GRID_SPEC.front_m == 3.0
    assert ROBOT_GRID_SPEC.rear_m == 1.0
    assert ROBOT_GRID_SPEC.half_width_m == 1.0
    assert ROBOT_GRID_SPEC.cell_m == 0.05
    assert ROBOT_GRID_SPEC.n_rows == 80
    assert ROBOT_GRID_SPEC.n_cols == 40


def test_remap_semantic_to_occupancy_marks_road_classes_drivable():
    labels = np.array([0, 6, 7, 8, 24])

    occupancy = remap_semantic_to_occupancy(labels)

    np.testing.assert_array_equal(occupancy, [0, 1, 1, 0, 0])
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/bev_gt/test_grid.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'projects.bev_gt.grid'`)

- [ ] **Step 3: 구현**

`projects/bev_gt/grid.py`:

```python
from dataclasses import dataclass

import numpy as np

DRIVABLE_CLASS_IDS = frozenset({6, 7})  # road line(6), road(7) per SynWoodScape semantic palette


@dataclass(frozen=True)
class OccupancyGridSpec:
    front_m: float
    rear_m: float
    half_width_m: float
    cell_m: float

    @property
    def n_rows(self) -> int:
        return round((self.front_m + self.rear_m) / self.cell_m)

    @property
    def n_cols(self) -> int:
        return round((2 * self.half_width_m) / self.cell_m)


ROBOT_GRID_SPEC = OccupancyGridSpec(front_m=3.0, rear_m=1.0, half_width_m=1.0, cell_m=0.05)


def remap_semantic_to_occupancy(semantic_labels: np.ndarray, drivable_class_ids=DRIVABLE_CLASS_IDS) -> np.ndarray:
    """Map semantic class ids to binary occupancy (1=drivable, 0=non-drivable)."""
    semantic_labels = np.asarray(semantic_labels)
    drivable_ids = np.fromiter(drivable_class_ids, dtype=semantic_labels.dtype)
    return np.isin(semantic_labels, drivable_ids).astype(np.uint8)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/bev_gt/test_grid.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
git add projects/bev_gt/__init__.py projects/bev_gt/grid.py tests/bev_gt/test_grid.py
git commit -m "feat: add occupancy grid spec and drivable-class remap"
```

---

### Task 7: BEV 이미지 스케일/원점 보정 모듈

**Files:**
- Create: `projects/bev_gt/bev_calibration.py`
- Test: `tests/bev_gt/test_bev_calibration.py`

**Interfaces:**
- Produces: `load_instance_bev_image(path) -> np.ndarray`
- Produces: `load_box_3d_annotations(path) -> dict[int, np.ndarray]`
- Produces: `instance_pixel_bbox(instance_image: np.ndarray, instance_id: int) -> tuple[int, int, int, int] | None` — `(x_min, x_max, y_min, y_max)`
- Produces: `box_3d_footprint_meters(corners: np.ndarray) -> tuple[float, float]` — 3D 박스 코너의 axis-aligned (X, Y) extent
- Produces: `estimate_scale_m_per_px(instance_image, box_3d_by_instance, instance_id) -> tuple[float, float] | None` — `(scale_x, scale_y)`
- Produces: `ego_vehicle_bbox_center(semantic_bev_image: np.ndarray, ego_vehicle_class_id: int = 24) -> tuple[float, float] | None`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/bev_gt/test_bev_calibration.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from projects.bev_gt.bev_calibration import (
    box_3d_footprint_meters,
    ego_vehicle_bbox_center,
    estimate_scale_m_per_px,
    instance_pixel_bbox,
    load_box_3d_annotations,
    load_instance_bev_image,
)

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
requires_dataset = pytest.mark.skipif(
    not DATASET_ROOT.exists(), reason="SynWoodScape dataset not available locally"
)


def test_instance_pixel_bbox_on_synthetic_image():
    instance_image = np.zeros((10, 10), dtype=np.uint8)
    instance_image[2:5, 3:6] = 7  # rows 2..4, cols 3..5

    bbox = instance_pixel_bbox(instance_image, 7)

    assert bbox == (3, 5, 2, 4)


def test_instance_pixel_bbox_returns_none_when_absent():
    instance_image = np.zeros((10, 10), dtype=np.uint8)

    assert instance_pixel_bbox(instance_image, 7) is None


def test_box_3d_footprint_meters_on_synthetic_corners():
    corners = np.array(
        [[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0], [0.0, 3.0, 0.0, 1.0], [2.0, 3.0, 1.5, 1.0]]
    )

    width_m, length_m = box_3d_footprint_meters(corners)

    assert width_m == pytest.approx(2.0)
    assert length_m == pytest.approx(3.0)


@requires_dataset
def test_estimate_scale_for_ego_vehicle_instance_in_sample_00000():
    instance_image = load_instance_bev_image(
        DATASET_ROOT / "_unuserd/instance_annotations/gtLabels/00000_BEV.png"
    )
    box_3d_by_instance = load_box_3d_annotations(
        DATASET_ROOT / "_unuserd/box_3d_annotations/00000.pkl"
    )

    scale_x, scale_y = estimate_scale_m_per_px(instance_image, box_3d_by_instance, instance_id=24)

    assert scale_x == pytest.approx(0.0288, abs=0.001)
    assert scale_y == pytest.approx(0.0285, abs=0.001)


@requires_dataset
def test_ego_vehicle_bbox_center_is_image_center_for_sample_00000():
    semantic_bev = load_instance_bev_image(
        DATASET_ROOT / "semantic_annotations/gtLabels/00000_BEV.png"
    )

    center = ego_vehicle_bbox_center(semantic_bev)

    assert center == pytest.approx((511.5, 511.5), abs=0.5)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/bev_gt/test_bev_calibration.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'projects.bev_gt.bev_calibration'`)

- [ ] **Step 3: 구현**

`projects/bev_gt/bev_calibration.py`:

```python
import pickle

import numpy as np
from PIL import Image


def load_instance_bev_image(path) -> np.ndarray:
    return np.array(Image.open(path))


def load_box_3d_annotations(path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def instance_pixel_bbox(instance_image: np.ndarray, instance_id: int):
    """Return (x_min, x_max, y_min, y_max) pixel bbox for instance_id, or None if absent."""
    mask = instance_image == instance_id
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def box_3d_footprint_meters(corners: np.ndarray):
    """Axis-aligned (X, Y) extent, in meters, of an (8, 4) or (8, 3) 3D box corner array."""
    corners = np.asarray(corners)[:, :3]
    extent = corners.max(axis=0) - corners.min(axis=0)
    return float(extent[0]), float(extent[1])


def estimate_scale_m_per_px(instance_image: np.ndarray, box_3d_by_instance: dict, instance_id: int):
    """(scale_x, scale_y) in meters/pixel, derived from one instance's pixel bbox vs. its 3D box footprint."""
    bbox = instance_pixel_bbox(instance_image, instance_id)
    if bbox is None or instance_id not in box_3d_by_instance:
        return None
    x_min, x_max, y_min, y_max = bbox
    px_width = x_max - x_min + 1
    px_height = y_max - y_min + 1
    width_m, length_m = box_3d_footprint_meters(box_3d_by_instance[instance_id])
    return width_m / px_width, length_m / px_height


def ego_vehicle_bbox_center(semantic_bev_image: np.ndarray, ego_vehicle_class_id: int = 24):
    """(cx, cy) pixel center of the ego-vehicle class blob in a semantic BEV image, or None if absent."""
    bbox = instance_pixel_bbox(semantic_bev_image, ego_vehicle_class_id)
    if bbox is None:
        return None
    x_min, x_max, y_min, y_max = bbox
    return (x_min + x_max) / 2, (y_min + y_max) / 2
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/bev_gt/test_bev_calibration.py -v`
Expected: PASS — 실측값(scale_x≈0.0288, scale_y≈0.0285, center≈(511.5,511.5))은 계획 작성 중 샘플 00000으로 직접 확인했다.

- [ ] **Step 5: 커밋**

```bash
git add projects/bev_gt/bev_calibration.py tests/bev_gt/test_bev_calibration.py
git commit -m "feat: add BEV image scale/origin calibration via instance-to-3d-box matching"
```

---

### Task 8: BEV 스케일 보정 스크립트 실행 (여러 샘플 집계 + 축 방향 확정)

**Files:**
- Create: `tools/calibrate_bev_scale.py`

**Interfaces:**
- Consumes: `projects.bev_gt.bev_calibration.*` (Task 7), `projects.geometry.frames.world_points_to_ego` (Task 2)

**목적**: Task 7의 단일-샘플 계산을 여러 샘플로 확장해 스케일 추정의 안정성을 확인하고, BEV 이미지 축이 ego의 전방/좌우 중 어디에 대응하는지(부호)까지 확정한다. §4.3에서 이미 확인했듯 `ego-vehicle` bbox가 66(가로)×132(세로)px이고 3D 박스가 X=1.9m(폭)/Y=3.76m(길이)이므로 **세로(행) 축 = 전후(ego X), 가로(열) 축 = 좌우(ego Y)** 로 이미 확정되어 있다 — 이 스크립트는 나머지 **부호**(어느 쪽이 전방(+)인지, 어느 쪽이 좌/우(+)인지)를 데이터로 찾는다.

- [ ] **Step 1: 스크립트 작성**

`tools/calibrate_bev_scale.py`:

```python
"""Phase 2 1회성 보정: 여러 샘플에 걸쳐 BEV 이미지의 스케일(m/px)과
   ego-vehicle 원점 일관성을 확인하고, LiDAR 포인트를 이용해 전방/좌우 부호를 찾는다.

Run: python tools/calibrate_bev_scale.py --samples 00000 00010 00100 00200 00300
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

from projects.bev_gt.bev_calibration import (
    ego_vehicle_bbox_center,
    estimate_scale_m_per_px,
    load_box_3d_annotations,
    load_instance_bev_image,
)
from projects.geometry.frames import world_points_to_ego

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")


def collect_scale_estimates(sample_indices):
    scales_x, scales_y, centers = [], [], []
    for idx in sample_indices:
        instance_image = load_instance_bev_image(
            DATASET_ROOT / f"_unuserd/instance_annotations/gtLabels/{idx}_BEV.png"
        )
        box_3d_by_instance = load_box_3d_annotations(
            DATASET_ROOT / f"_unuserd/box_3d_annotations/{idx}.pkl"
        )
        semantic_bev = load_instance_bev_image(
            DATASET_ROOT / f"semantic_annotations/gtLabels/{idx}_BEV.png"
        )

        center = ego_vehicle_bbox_center(semantic_bev)
        if center is not None:
            centers.append(center)

        for instance_id in np.unique(instance_image):
            if instance_id == 0:
                continue
            result = estimate_scale_m_per_px(instance_image, box_3d_by_instance, int(instance_id))
            if result is not None:
                scales_x.append(result[0])
                scales_y.append(result[1])

    return np.array(scales_x), np.array(scales_y), np.array(centers)


def find_axis_sign(sample_indices, meters_per_pixel, origin_px, image_shape):
    """Search sign_forward/sign_lateral (row axis = forward, col axis = lateral; see docstring above)
    against the pixel class each LiDAR point lands on, aggregated over all samples."""
    height, width = image_shape
    candidates = [(sf, sl) for sf in (1, -1) for sl in (1, -1)]
    totals = {c: [0, 0] for c in candidates}  # (matches, total)

    for idx in sample_indices:
        with open(DATASET_ROOT / f"lidar_data/{idx}.pkl", "rb") as f:
            lidar = pickle.load(f)
        points_ego = world_points_to_ego(lidar["points"], np.asarray(lidar["transform"]))
        labels = lidar["labels"]

        radius_xy = np.linalg.norm(points_ego[:, :2], axis=1)
        half_footprint_m = height / 2 * meters_per_pixel
        keep = (radius_xy > 1.0) & (radius_xy < half_footprint_m)
        if not keep.any():
            continue
        ego_x, ego_y = points_ego[keep, 0], points_ego[keep, 1]
        point_labels = labels[keep]

        semantic_bev = load_instance_bev_image(
            DATASET_ROOT / f"semantic_annotations/gtLabels/{idx}_BEV.png"
        )

        for sign_forward, sign_lateral in candidates:
            row = origin_px[1] + (sign_forward * ego_x) / meters_per_pixel
            col = origin_px[0] + (sign_lateral * ego_y) / meters_per_pixel
            row_i, col_i = row.round().astype(int), col.round().astype(int)
            valid = (row_i >= 0) & (row_i < height) & (col_i >= 0) & (col_i < width)
            if not valid.any():
                continue
            gt_classes = semantic_bev[row_i[valid], col_i[valid]]
            matches = (gt_classes == point_labels[valid]).sum()
            totals[(sign_forward, sign_lateral)][0] += int(matches)
            totals[(sign_forward, sign_lateral)][1] += int(valid.sum())

    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=["00000", "00010", "00100", "00200", "00300"])
    args = parser.parse_args()

    scales_x, scales_y, centers = collect_scale_estimates(args.samples)
    print(f"scale_x: mean={scales_x.mean():.5f} std={scales_x.std():.5f} n={len(scales_x)}")
    print(f"scale_y: mean={scales_y.mean():.5f} std={scales_y.std():.5f} n={len(scales_y)}")
    print(f"ego bbox centers: {centers.tolist()}")

    meters_per_pixel = float(np.concatenate([scales_x, scales_y]).mean())
    origin_px = tuple(centers.mean(axis=0)) if len(centers) else (511.5, 511.5)

    sample_image = load_instance_bev_image(
        DATASET_ROOT / f"semantic_annotations/gtLabels/{args.samples[0]}_BEV.png"
    )
    totals = find_axis_sign(args.samples, meters_per_pixel, origin_px, sample_image.shape)

    print("\naxis sign search (sign_forward, sign_lateral) -> match_rate (n samples):")
    for signs, (matches, total) in totals.items():
        rate = matches / total if total else float("nan")
        print(f"  {signs}: {rate:.3f} (n={total})")

    print(
        "\n지침: 위에서 가장 높은 match_rate가 다른 후보들보다 뚜렷하게 크면 "
        "그 (sign_forward, sign_lateral)을 Task 9의 SIGN_FORWARD/SIGN_LATERAL 상수로 쓴다. "
        "차이가 뚜렷하지 않으면 rgb_images/<idx>_BEV.png(컬러)를 열어 차량 전방(헤드라이트 방향)이 "
        "이미지의 어느 쪽인지 육안으로 확인해 부호를 정한다."
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행**

Run: `python tools/calibrate_bev_scale.py --samples 00000 00010 00100 00200 00300`
Expected: `scale_x`/`scale_y` 평균이 ~0.028 근방이고 표준편차가 평균의 5% 이내, `ego bbox centers`가 모두 `(511.5, 511.5)` 근처, 그리고 8개 부호 후보 중 하나의 `match_rate`가 뚜렷하게 높게 출력된다.

- [ ] **Step 3: 결과 기록**

출력된 `meters_per_pixel` 평균값과 최적 `(sign_forward, sign_lateral)`을 Task 9의 `BEV_METERS_PER_PIXEL`/`SIGN_FORWARD`/`SIGN_LATERAL` 상수에 반영할 수 있도록 적어둔다(뚜렷한 승자가 없으면 Step 1 안내대로 `rgb_images/*_BEV.png`를 육안 확인한다).

- [ ] **Step 4: 커밋**

```bash
git add tools/calibrate_bev_scale.py
git commit -m "feat: add multi-sample BEV scale and forward-axis calibration script"
```

---

### Task 9: BEV 이미지 크롭 → occupancy 그리드 변환

**Files:**
- Create: `projects/bev_gt/bev_crop.py`
- Test: `tests/bev_gt/test_bev_crop.py`

**Interfaces:**
- Consumes: `projects.bev_gt.grid.OccupancyGridSpec`, `remap_semantic_to_occupancy` (Task 6)
- Produces: `crop_bev_occupancy(semantic_bev_image: np.ndarray, grid_spec: OccupancyGridSpec, meters_per_pixel: float, origin_px: tuple[float, float], sign_forward: int = 1, sign_lateral: int = 1) -> np.ndarray` — `(grid_spec.n_rows, grid_spec.n_cols)` occupancy 배열(row 0 = 가장 후방, row -1 = 가장 전방)

행(세로) 축 = 전후, 열(가로) 축 = 좌우라는 축 매핑은 Task 7/8에서 실측으로 확정했다(§4.3). `sign_forward`/`sign_lateral`은 Task 8 실행 결과로 확정되는 부호이며, 기본값 `1, 1`은 Task 8 실행 후 실제 값으로 갱신한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/bev_gt/test_bev_crop.py`:

```python
import numpy as np

from projects.bev_gt.bev_crop import crop_bev_occupancy
from projects.bev_gt.grid import OccupancyGridSpec


def test_crop_bev_occupancy_extracts_expected_region_and_remaps():
    # 30x20 synthetic semantic image, 1m/px. Ego sits at pixel row 10.5 (a half-integer,
    # so that the grid's half-integer row offsets land exactly on integer pixel rows —
    # avoids numpy's round-half-to-even ambiguity at exact .5 boundaries).
    # Rows 0..10 are "sidewalk"(8, non-drivable); rows 11+ are "road"(7, drivable).
    semantic_image = np.full((30, 20), 8, dtype=np.uint8)
    semantic_image[11:, :] = 7

    grid_spec = OccupancyGridSpec(front_m=3.0, rear_m=1.0, half_width_m=1.0, cell_m=1.0)
    occupancy = crop_bev_occupancy(
        semantic_image, grid_spec, meters_per_pixel=1.0, origin_px=(10.5, 10.5),
        sign_forward=1, sign_lateral=1,
    )

    assert occupancy.shape == (grid_spec.n_rows, grid_spec.n_cols)
    # front-most row (image row 13) -> road -> drivable; rear-most row (image row 10) -> sidewalk -> non-drivable
    assert occupancy[-1].all()
    assert not occupancy[0].any()
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/bev_gt/test_bev_crop.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'projects.bev_gt.bev_crop'`)

- [ ] **Step 3: 구현**

`projects/bev_gt/bev_crop.py`:

```python
import numpy as np

from projects.bev_gt.grid import OccupancyGridSpec, remap_semantic_to_occupancy

# Task 8(tools/calibrate_bev_scale.py) 실행 결과로 확정되는 값. 초기값은 샘플 00000
# 단일 계측 기반(design doc §4.3)이며, Task 8을 실행한 뒤 평균값으로 갱신한다.
BEV_METERS_PER_PIXEL = 0.0286
BEV_ORIGIN_PX = (511.5, 511.5)
SIGN_FORWARD = 1
SIGN_LATERAL = 1


def crop_bev_occupancy(
    semantic_bev_image: np.ndarray,
    grid_spec: OccupancyGridSpec,
    meters_per_pixel: float = BEV_METERS_PER_PIXEL,
    origin_px=BEV_ORIGIN_PX,
    sign_forward: int = SIGN_FORWARD,
    sign_lateral: int = SIGN_LATERAL,
) -> np.ndarray:
    """Crop and remap a semantic BEV image into an ego-relative occupancy grid.

    Row axis of `semantic_bev_image` corresponds to forward/backward (ego X),
    column axis corresponds to left/right (ego Y) — confirmed empirically in
    docs/superpowers/specs/2026-07-29-synwoodscape-fisheye-bev-occupancy-design.md §4.3.
    """
    origin_col, origin_row = origin_px
    row_offsets_m = (np.arange(grid_spec.n_rows) + 0.5) * grid_spec.cell_m - grid_spec.rear_m
    col_offsets_m = (np.arange(grid_spec.n_cols) + 0.5) * grid_spec.cell_m - grid_spec.half_width_m

    rows = np.round(origin_row + sign_forward * row_offsets_m / meters_per_pixel).astype(int)
    cols = np.round(origin_col + sign_lateral * col_offsets_m / meters_per_pixel).astype(int)

    height, width = semantic_bev_image.shape
    rows = np.clip(rows, 0, height - 1)
    cols = np.clip(cols, 0, width - 1)

    sampled_labels = semantic_bev_image[np.ix_(rows, cols)]
    return remap_semantic_to_occupancy(sampled_labels)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/bev_gt/test_bev_crop.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Task 8 결과 반영**

Task 8 실행 결과가 `BEV_METERS_PER_PIXEL`/`SIGN_FORWARD`/`SIGN_LATERAL` 기본값과 다르면 이 파일의 상수를 갱신한다.

- [ ] **Step 6: 커밋**

```bash
git add projects/bev_gt/bev_crop.py tests/bev_gt/test_bev_crop.py
git commit -m "feat: add BEV image cropping into ego-relative occupancy grids"
```

---

### Task 10: Occupancy GT 배치 생성 스크립트

**Files:**
- Create: `tools/build_occupancy_gt.py`

**Interfaces:**
- Consumes: `projects.bev_gt.bev_crop.crop_bev_occupancy`, `projects.bev_gt.grid.ROBOT_GRID_SPEC` (Task 6, 9)

**동작**: 지정한 샘플들에 대해 occupancy 그리드를 만들어 `outputs/occupancy_gt/`에 `.npy`로 저장하고, 시각화 PNG(drivable=흰색, non-drivable=검은색)를 같이 저장한다.

- [ ] **Step 1: 스크립트 작성**

`tools/build_occupancy_gt.py`:

```python
"""Phase 2 실행: semantic_annotations의 _BEV.png로부터 로봇 기준 occupancy GT를 생성한다.

Run: python tools/build_occupancy_gt.py --samples 00000 00001 00002
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from projects.bev_gt.bev_crop import crop_bev_occupancy
from projects.bev_gt.grid import ROBOT_GRID_SPEC

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
OUTPUT_DIR = Path("outputs/occupancy_gt")


def build_sample(sample_idx: str) -> None:
    semantic_bev = np.array(
        Image.open(DATASET_ROOT / "semantic_annotations/gtLabels" / f"{sample_idx}_BEV.png")
    )
    occupancy = crop_bev_occupancy(semantic_bev, ROBOT_GRID_SPEC)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / f"{sample_idx}_occupancy.npy", occupancy)

    visualization = (occupancy * 255).astype(np.uint8)
    Image.fromarray(visualization).save(OUTPUT_DIR / f"{sample_idx}_occupancy.png")
    print(f"[{sample_idx}] drivable_fraction={occupancy.mean():.3f} -> {OUTPUT_DIR}/{sample_idx}_occupancy.npy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=["00000", "00001", "00002"])
    args = parser.parse_args()

    for sample_idx in args.samples:
        build_sample(sample_idx)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행 및 육안 확인**

Run: `python tools/build_occupancy_gt.py --samples 00000 00001 00002`
Expected: `outputs/occupancy_gt/`에 `.npy`/`.png`가 생성되고, `drivable_fraction`이 0~1 사이 합리적인 값으로 출력된다. `*_occupancy.png`를 열어 흰색(주행 가능) 영역이 도로 모양과 대체로 일치하는지 확인한다.

- [ ] **Step 3: 커밋**

```bash
git add tools/build_occupancy_gt.py
git commit -m "feat: add Phase 2 occupancy GT batch generation script"
```

---

## 실행 순서 요약

Task 1→2(좌표계) → 3(fisheye) → 4(지표) → 5(Phase 1 실행) 은 순차 의존. Task 6(그리드 스펙)은 Task 1~5와 독립적으로 아무 때나 먼저 해도 된다. Task 7→8→9→10(Phase 2)은 Task 6 완료 후 순차 진행하며, Task 2의 `world_points_to_ego`를 Task 8에서 재사용한다.
