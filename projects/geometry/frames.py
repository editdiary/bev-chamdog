"""좌표계 변환 헬퍼 — SynWoodScape(CARLA 생성) 데이터셋 기준.

## 이 파일에서 쓰는 좌표계 (모두 SynWoodScape 자체 데이터로 검증됨)

- **CARLA 프레임** (world / vehicle): X=전방, Y=**우측**, Z=상방. UE4 기반의 **왼손 좌표계**.
  `vehicle_data/rgb_images/*.txt`의 `Transform(Location, Rotation(pitch,yaw,roll))`,
  `lidar_data/*.pkl`의 `transform`, `_unuserd/box_3d_annotations/*.pkl`의 코너 좌표가 모두 이 프레임이다.
- **ego 프레임** (이 프로젝트의 작업 프레임): X=전방, Y=**좌측**, Z=상방. **오른손 좌표계**.
  원점은 CARLA vehicle actor 원점과 같다(차량 중심, 지면 높이).
  `calibration_data/*.json`의 extrinsic이 카메라를 이 프레임으로 보내므로 여기에 맞춘다.
  (extrinsic quaternion은 det=+1인 정상 회전이고 radial_poly 카메라 프레임은 오른손계이므로,
  대상 프레임도 반드시 오른손계여야 한다 → X=전방·Z=상방이면 Y는 좌측이다.)

CARLA 프레임 ↔ ego 프레임은 Y 부호만 다르다 (`carla_frame_to_ego`).

## 검증 근거 (docs 아닌 실측)

- `lidar_data/*.pkl`의 `transform` 회전은 같은 샘플 `vehicle_data/rgb_images/*.txt`가 보고하는
  `Rotation(pitch,yaw,roll)`로 만든 CARLA `get_matrix()`와 최대 오차 1e-6 수준으로 일치하고,
  translation은 vehicle Location + (0,0,2.0)이다 → `transform`은 world_T_lidar이며
  LiDAR는 차량과 **같은 방향**으로 ego 기준 (0,0,2.0)에 장착돼 있다 (readme.txt와 일치).
- `_unuserd/box_3d_annotations/*.pkl`의 ego 자신의 박스(key 24)를 위 CARLA 행렬로 되돌리면
  x∈[-1.853,1.853], y∈[-0.894,0.895], z∈[0.009,1.556]로 **정확히 축정렬·원점중심**이 된다
  (여러 샘플에서 동일) → CARLA vehicle 프레임의 원점/축 의미 확인.
"""
import re
from pathlib import Path

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


# readme.txt: "The lidar sensor is attached to the ego vehicle at the position
# x=0.0, z=2.0, y=0.0 in the ego vehicle reference."  회전 오프셋은 없다(위 검증 근거 참고).
LIDAR_MOUNT_OFFSET_EGO = np.array([0.0, 0.0, 2.0])

# CARLA(왼손, Y=우측) ↔ ego(오른손, Y=좌측): Y 부호만 뒤집으면 된다.
_CARLA_EGO_AXIS_FLIP = np.array([1.0, -1.0, 1.0])


def carla_frame_to_ego(points_carla: np.ndarray) -> np.ndarray:
    """CARLA 규약(Y=우측, 왼손) 좌표를 ego 규약(Y=좌측, 오른손)으로 바꾼다.

    두 프레임은 원점·X·Z가 같고 Y 부호만 반대이므로 involution이다
    (같은 함수를 두 번 적용하면 원래대로 돌아온다).
    """
    return np.asarray(points_carla, dtype=np.float64) * _CARLA_EGO_AXIS_FLIP


def lidar_points_to_ego(points_lidar: np.ndarray) -> np.ndarray:
    """`lidar_data/*.pkl`의 `points`를 ego 프레임으로 옮긴다 (장착 오프셋만 더함).

    **`transform`(world_T_lidar)를 곱하지 않는다.** pkl의 `points`는 world 좌표가 아니라
    **이미 LiDAR 센서 로컬 좌표**이며, 그 축 규약도 CARLA(왼손)가 아니라 ego와 같은
    **오른손(Y=좌측)** 이다. 따라서 ego 프레임으로 가려면 장착 높이만 더하면 된다.

    근거 (SynWoodScape 자체 데이터 실측):

    1. `points`의 원점 기준 거리 중앙값이 5.65 m, 평균 좌표가 ≈(0.5, 1.7, 0.4)로 원점 주변에
       모여 있다 (world 좌표라면 차량 위치 ≈(-111, 81) 주변이어야 한다).
    2. `points`의 Y를 뒤집어 CARLA 프레임으로 보내면, vehicle 자세로 ego 프레임에 옮긴
       `_unuserd/box_3d_annotations`의 차량 3D 박스 안으로 vehicle 라벨(10/21) 포인트가
       들어간다. 뒤집지 않으면 거의 0개다 (예: 샘플 00000 → 29.4% vs 0.1%).
    3. 이 함수의 결과를 `projects.geometry.fisheye.load_camera`로 투영하면, 카메라 자신의
       depth map과 5% 이내로 일치하는(=실제로 보이는) 포인트에서 semantic 라벨 일치율이
       4대 카메라 모두 0.96~0.99가 된다.

    이전 구현은 `inv(world_T_lidar)`를 곱했는데, 이는 이미 로컬인 포인트를 차량의 world
    yaw(샘플마다 88°/-131°/82° 등)만큼 추가로 회전시키고 ~140 m 평행이동시켰다. 오차가
    샘플마다 다른 회전이라서 어떤 축 부호/순서 조합으로도 복구되지 않았다.
    """
    return np.asarray(points_lidar, dtype=np.float64) + LIDAR_MOUNT_OFFSET_EGO


_VEHICLE_POSE_RE = re.compile(
    r"Transform\(Location\(x=([-\d.]+), y=([-\d.]+), z=([-\d.]+)\), "
    r"Rotation\(pitch=([-\d.]+), yaw=([-\d.]+), roll=([-\d.]+)\)\)"
)


def carla_rotation_matrix(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """CARLA `Transform.get_matrix()`의 3x3 회전부 (도 단위, CARLA 왼손 규약).

    CARLA/UE4가 실제로 쓰는 성분 배치를 그대로 옮긴 것이다. 오른손 회전으로 다시 쓰면
    `Rz(yaw) @ Ry(-pitch) @ Rx(-roll)`과 같다 — 이 관계가 `fisheye.py`의 roll 부호 보정을
    유도하는 출발점이다.
    """
    cy, sy = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    cp, sp = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))
    cr, sr = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    return np.array(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ]
    )


def parse_vehicle_transform(vehicle_data_txt_path) -> np.ndarray:
    """vehicle_data/rgb_images/*.txt의 ego world pose를 4x4 CARLA 행렬(world_T_ego)로 읽는다."""
    text = Path(vehicle_data_txt_path).read_text()
    match = _VEHICLE_POSE_RE.search(text)
    if match is None:
        raise ValueError(f"could not parse a vehicle Transform out of {vehicle_data_txt_path}")
    location = np.array([float(match.group(i)) for i in (1, 2, 3)])
    pitch, yaw, roll = (float(match.group(i)) for i in (4, 5, 6))
    transform = np.eye(4)
    transform[:3, :3] = carla_rotation_matrix(pitch, yaw, roll)
    transform[:3, 3] = location
    return transform
