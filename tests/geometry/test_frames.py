import pickle
import re
from pathlib import Path

import numpy as np
import pytest

from projects.geometry.frames import (
    LIDAR_MOUNT_OFFSET_EGO,
    apply_4x4,
    carla_frame_to_ego,
    carla_rotation_matrix,
    invert_4x4,
    lidar_points_to_ego,
    parse_vehicle_transform,
)

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
requires_dataset = pytest.mark.skipif(
    not DATASET_ROOT.exists(), reason="SynWoodScape dataset not available locally"
)
BOX_3D_DIR = DATASET_ROOT / "_unuserd" / "box_3d_annotations"
requires_box_3d = pytest.mark.skipif(
    not BOX_3D_DIR.exists(), reason="SynWoodScape box_3d_annotations not available locally"
)
EGO_INSTANCE_ID = 24  # box_3d_annotations의 ego 차량 자신의 instance id


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


def test_lidar_points_to_ego_only_applies_the_mount_offset():
    """`points`는 이미 센서 로컬 좌표이므로 world_T_lidar를 곱하면 안 된다.

    과거 구현은 `inv(world_T_lidar)`를 곱해서 포인트 클라우드를 차량의 world yaw만큼
    추가 회전시키는 버그가 있었다. 이 테스트가 그 회귀를 막는다.
    """
    points_lidar = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, -1.0]])

    result = lidar_points_to_ego(points_lidar)

    np.testing.assert_allclose(result, points_lidar + LIDAR_MOUNT_OFFSET_EGO)


def test_lidar_sensor_origin_maps_to_the_mount_position():
    np.testing.assert_allclose(lidar_points_to_ego(np.zeros((1, 3))), [[0.0, 0.0, 2.0]])


def test_carla_frame_to_ego_flips_only_y():
    points = np.array([[1.0, 2.0, 3.0]])

    np.testing.assert_allclose(carla_frame_to_ego(points), [[1.0, -2.0, 3.0]])
    # 두 프레임은 Y 부호만 다르므로 involution이다.
    np.testing.assert_allclose(carla_frame_to_ego(carla_frame_to_ego(points)), points)


def test_carla_rotation_matrix_is_a_proper_rotation():
    rotation = carla_rotation_matrix(pitch_deg=7.0, yaw_deg=-40.0, roll_deg=3.0)

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_carla_rotation_matrix_yaw_only_matches_the_closed_form():
    rotation = carla_rotation_matrix(pitch_deg=0.0, yaw_deg=90.0, roll_deg=0.0)

    np.testing.assert_allclose(rotation, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)


@requires_dataset
@pytest.mark.parametrize("sample_idx", ["00000", "00001", "00002"])
def test_lidar_transform_equals_the_carla_matrix_of_the_reported_vehicle_pose(sample_idx):
    """pkl의 `transform`이 world_T_lidar임을, 데이터셋 자신의 vehicle pose로 확인한다.

    회전부는 vehicle_data가 보고하는 Rotation(pitch,yaw,roll)로 만든 CARLA 행렬과 같고,
    translation은 vehicle Location + (0,0,2.0)이다. 즉 LiDAR는 차량과 같은 방향으로
    ego 기준 (0,0,2.0)에 장착돼 있다.
    """
    world_T_ego = parse_vehicle_transform(DATASET_ROOT / "vehicle_data" / "rgb_images" / f"{sample_idx}.txt")
    with open(DATASET_ROOT / "lidar_data" / f"{sample_idx}.pkl", "rb") as f:
        lidar_record = pickle.load(f)
    world_T_lidar = np.asarray(lidar_record["transform"])

    np.testing.assert_allclose(world_T_lidar[:3, :3], world_T_ego[:3, :3], atol=1e-5)
    np.testing.assert_allclose(
        world_T_lidar[:3, 3] - world_T_ego[:3, 3], LIDAR_MOUNT_OFFSET_EGO, atol=0.06
    )


@requires_dataset
@pytest.mark.parametrize("sample_idx", ["00000", "00001", "00002"])
def test_stored_lidar_points_are_sensor_local_not_world(sample_idx):
    """`points`가 world 좌표가 아니라 센서 로컬 좌표임을 거리 분포로 확인한다.

    world 좌표라면 차량 위치(수십~수백 m 떨어진 값) 주변에 모여 있어야 하지만, 실제로는
    원점 주변에 모여 있다(거리 중앙값 한 자리 m).
    """
    with open(DATASET_ROOT / "lidar_data" / f"{sample_idx}.pkl", "rb") as f:
        points = np.asarray(pickle.load(f)["points"], dtype=np.float64)

    range_from_origin = np.linalg.norm(points, axis=1)

    assert np.median(range_from_origin) < 50.0
    assert range_from_origin.min() < 5.0


@requires_dataset
@requires_box_3d
@pytest.mark.parametrize("sample_idx", ["00000", "00001", "00002"])
def test_ego_3d_box_is_axis_aligned_in_the_carla_vehicle_frame(sample_idx):
    """CARLA vehicle 프레임의 원점·축 의미를 데이터셋 자신의 3D 박스로 확인한다.

    ego 차량 자신의 박스를 vehicle pose로 되돌리면 원점 중심으로 축정렬되어야 하고,
    바닥면이 z=0(지면)에 놓여야 한다.
    """
    world_T_ego = parse_vehicle_transform(DATASET_ROOT / "vehicle_data" / "rgb_images" / f"{sample_idx}.txt")
    with open(BOX_3D_DIR / f"{sample_idx}.pkl", "rb") as f:
        boxes = pickle.load(f)

    corners_in_ego = apply_4x4(invert_4x4(world_T_ego), np.asarray(boxes[EGO_INSTANCE_ID])[:, :3])

    # 좌우/전후 대칭 (원점 = 차량 중심), 바닥면 z ≈ 0 (원점 = 지면 높이)
    assert corners_in_ego[:, 0].min() == pytest.approx(-corners_in_ego[:, 0].max(), abs=0.02)
    assert corners_in_ego[:, 1].min() == pytest.approx(-corners_in_ego[:, 1].max(), abs=0.02)
    assert corners_in_ego[:, 2].min() == pytest.approx(0.0, abs=0.05)
    # 승용차 치수: 길이 ~3.7m, 폭 ~1.8m, 높이 ~1.55m
    extents = corners_in_ego.max(axis=0) - corners_in_ego.min(axis=0)
    assert 3.0 < extents[0] < 5.0
    assert 1.5 < extents[1] < 2.5
    assert 1.2 < extents[2] < 2.0


@requires_dataset
@requires_box_3d
def test_vehicle_labelled_lidar_points_fall_inside_annotated_3d_boxes():
    """LiDAR `points`의 축 규약(오른손, Y=좌측)을 카메라 없이 확인한다.

    vehicle 라벨 포인트를 ego 프레임으로 옮긴 뒤 CARLA 프레임으로 되돌리면 데이터셋의
    3D 차량 박스 안에 들어가야 한다. Y를 뒤집지 않으면(=`points`가 CARLA 규약이라고 가정하면)
    거의 하나도 들어가지 않는다.
    """
    sample_idx = "00000"
    world_T_ego = parse_vehicle_transform(DATASET_ROOT / "vehicle_data" / "rgb_images" / f"{sample_idx}.txt")
    with open(BOX_3D_DIR / f"{sample_idx}.pkl", "rb") as f:
        boxes = pickle.load(f)
    with open(DATASET_ROOT / "lidar_data" / f"{sample_idx}.pkl", "rb") as f:
        lidar = pickle.load(f)

    labels = np.asarray(lidar["labels"])
    is_vehicle = np.isin(labels, [10, 21])  # four-wheeler / two-wheeler
    points_ego = lidar_points_to_ego(lidar["points"])[is_vehicle]
    points_ego = points_ego[np.linalg.norm(points_ego, axis=1) < 60.0]

    def fraction_inside_boxes(points_carla):
        inside = np.zeros(len(points_carla), dtype=bool)
        for instance_id, corners in boxes.items():
            if instance_id == EGO_INSTANCE_ID:
                continue
            box = apply_4x4(invert_4x4(world_T_ego), np.asarray(corners)[:, :3])
            origin = box[0]
            axes = np.stack([box[1] - origin, box[3] - origin, box[4] - origin], axis=1)
            local = np.linalg.solve(axes, (points_carla - origin).T).T
            inside |= np.all((local > -0.05) & (local < 1.05), axis=1)
        return inside.mean()

    correct = fraction_inside_boxes(carla_frame_to_ego(points_ego))
    unflipped = fraction_inside_boxes(points_ego)

    assert correct > 0.2
    assert correct > 20 * max(unflipped, 1e-3)
