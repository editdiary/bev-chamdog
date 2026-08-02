import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as SciRot

from projects.common.metrics import class_consistency_rate
from projects.geometry.fisheye import (
    CAMERA_TO_EGO_BASE,
    ego_rotation_from_quaternion,
    ego_translation_from_calibration,
    load_camera,
)
from projects.geometry.frames import lidar_points_to_ego
from projects.geometry.reprojection import (
    project_points_to_image,
    unproject_depth_to_ego,
    visibility_mask,
)

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
CALIB_DIR = DATASET_ROOT / "calibration_data"
CAMERAS = ["FV", "MVL", "MVR", "RV"]
requires_dataset = pytest.mark.skipif(
    not CALIB_DIR.exists(), reason="SynWoodScape dataset not available locally"
)
requires_depth = pytest.mark.skipif(
    not (DATASET_ROOT / "depth_maps" / "raw_data").exists(),
    reason="SynWoodScape depth maps not available locally",
)
EGO_VEHICLE_CLASS = 24  # readme.txt semantic palette


def _mount_euler(rotation):
    """camera→ego 회전에서 차량 장착 각도(ZYX: yaw, pitch, roll)를 뽑는다."""
    return SciRot.from_matrix(rotation @ CAMERA_TO_EGO_BASE.T).as_euler("ZYX", degrees=True)


def test_camera_to_ego_base_is_a_proper_rotation():
    np.testing.assert_allclose(CAMERA_TO_EGO_BASE @ CAMERA_TO_EGO_BASE.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(CAMERA_TO_EGO_BASE) == pytest.approx(1.0)


def test_camera_to_ego_base_maps_the_optical_axis_forward():
    # OpenCV 카메라: x=오른쪽, y=아래, z=광축 → ego: x=전방, y=좌측, z=상방
    np.testing.assert_allclose(CAMERA_TO_EGO_BASE @ [0, 0, 1], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(CAMERA_TO_EGO_BASE @ [1, 0, 0], [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(CAMERA_TO_EGO_BASE @ [0, 1, 0], [0, 0, -1], atol=1e-12)


def test_ego_rotation_from_quaternion_negates_only_the_mount_roll():
    """§1 보정의 정의 그대로: yaw/pitch는 유지하고 roll 부호만 뒤집는다."""
    quaternion = SciRot.from_matrix(
        SciRot.from_euler("ZYX", [30.0, -12.0, 5.0], degrees=True).as_matrix() @ CAMERA_TO_EGO_BASE
    ).as_quat()

    yaw, pitch, roll = _mount_euler(ego_rotation_from_quaternion(quaternion))

    assert yaw == pytest.approx(30.0)
    assert pitch == pytest.approx(-12.0)
    assert roll == pytest.approx(-5.0)


def test_ego_rotation_from_quaternion_is_a_proper_rotation():
    quaternion = SciRot.from_euler("ZYX", [10.0, 20.0, 30.0], degrees=True).as_quat()

    rotation = ego_rotation_from_quaternion(quaternion)

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "camera_name, stored, expected",
    [
        # FV/RV: y=0이라 CARLA→ego Y 반전이 값에 드러나지 않는다.
        ("FV", [1.92, 0.0, 0.9], [1.92, 0.0, 0.9]),
        ("RV", [-1.92, 0.0, 0.9], [-1.92, 0.0, 0.9]),
        # 미러 카메라: y/z 전치 후 Y 반전 → MVL은 좌측(+y), MVR은 우측(-y), 둘 다 z=1.0
        ("MVL", [0.8, 1.0, -0.9], [0.8, 0.9, 1.0]),
        ("MVR", [0.8, 1.0, 0.9], [0.8, -0.9, 1.0]),
    ],
)
def test_ego_translation_from_calibration(camera_name, stored, expected):
    np.testing.assert_allclose(ego_translation_from_calibration(camera_name, stored), expected)


@requires_dataset
def test_load_camera_parses_fv_json():
    cam = load_camera(CALIB_DIR / "FV.json")

    assert cam.width == 1280
    assert cam.height == 966
    assert abs(cam.aspect_ratio - 1.00021) < 1e-6
    np.testing.assert_allclose(cam.translation, [1.92, 0.0, 0.9])


@requires_dataset
@pytest.mark.parametrize(
    "camera_name, expected_translation",
    [
        ("FV", [1.92, 0.0, 0.9]),
        ("MVL", [0.8, 0.9, 1.0]),
        ("MVR", [0.8, -0.9, 1.0]),
        ("RV", [-1.92, 0.0, 0.9]),
    ],
)
def test_load_camera_applies_the_translation_correction(camera_name, expected_translation):
    cam = load_camera(CALIB_DIR / f"{camera_name}.json")

    np.testing.assert_allclose(cam.translation, expected_translation)


@requires_dataset
@pytest.mark.parametrize("camera_name", CAMERAS)
def test_load_camera_flips_the_roll_stored_in_the_json(camera_name):
    with open(CALIB_DIR / f"{camera_name}.json") as f:
        quaternion = json.load(f)["extrinsic"]["quaternion"]
    stored_yaw, stored_pitch, stored_roll = _mount_euler(SciRot.from_quat(quaternion).as_matrix())

    yaw, pitch, roll = _mount_euler(load_camera(CALIB_DIR / f"{camera_name}.json").rotation)

    assert yaw == pytest.approx(stored_yaw, abs=1e-6)
    assert pitch == pytest.approx(stored_pitch, abs=1e-6)
    assert roll == pytest.approx(-stored_roll, abs=1e-6)


@requires_dataset
@pytest.mark.parametrize(
    "camera_name, axis, sign",
    [
        ("FV", 0, +1),   # 전방
        ("RV", 0, -1),   # 후방
        ("MVL", 1, +1),  # 좌측 (ego +y)
        ("MVR", 1, -1),  # 우측 (ego -y)
    ],
)
def test_camera_optical_axes_point_the_way_their_names_say(camera_name, axis, sign):
    """좌우 handedness 가드: 미러 카메라의 광축이 이름대로 좌/우를 향해야 한다."""
    cam = load_camera(CALIB_DIR / f"{camera_name}.json")

    optical_axis = cam.rotation @ np.array([0.0, 0.0, 1.0])

    assert sign * optical_axis[axis] > 0.05
    # 4대 모두 아래쪽으로 기울어 장착돼 있다.
    assert optical_axis[2] < 0.0


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


@requires_dataset
@requires_depth
@pytest.mark.parametrize("camera_name", CAMERAS)
def test_road_unprojects_onto_the_ground_plane(camera_name):
    """extrinsic 정확도 회귀 테스트 (데이터셋 자신의 depth map 사용).

    노면(class 6/7) 픽셀을 각 카메라의 depth로 역투영해 ego 프레임으로 옮기면 z ≈ 0인
    평면이 되어야 한다. roll 부호나 카메라 높이가 틀리면 이 테스트가 깨진다.
    """
    cv2 = pytest.importorskip("cv2")
    sample_idx = "00000"
    cam = load_camera(CALIB_DIR / f"{camera_name}.json")
    depth = np.load(DATASET_ROOT / "depth_maps" / "raw_data" / f"{sample_idx}_{camera_name}.npy")
    gt_labels = cv2.imread(
        str(DATASET_ROOT / "semantic_annotations" / "gtLabels" / f"{sample_idx}_{camera_name}.png"),
        cv2.IMREAD_GRAYSCALE,
    )

    is_road = np.isin(gt_labels, [6, 7]) & (depth < 15.0)
    rows, cols = np.nonzero(is_road)
    picked = np.random.default_rng(0).choice(len(rows), size=2000, replace=False)
    rows, cols = rows[picked], cols[picked]

    points_ego = unproject_depth_to_ego(cam, rows, cols, depth)

    # 노면 높이 z는 0 근처에 모여 있어야 한다 (노면 캠버/depth 양자화로 약간 퍼진다).
    assert abs(np.median(points_ego[:, 2])) < 0.10
    assert np.percentile(np.abs(points_ego[:, 2]), 90) < 0.25


# 여러 샘플에 걸친 end-to-end 지표용 샘플 (데이터셋 전체에 퍼뜨린다).
# 임계값은 42개 샘플 × 4대 = 168쌍 실측에서 유도했다 — 스펙 §3.5 참고.
METRIC_SAMPLES = ["00000", "00100", "00150", "00200", "00300", "00400"]

# 카메라별 median 기준. 168쌍 실측 median은 depth_agreement 0.663~0.773,
# visible rate 0.946~0.981이었다. METRIC_SAMPLES 6개에서의 실측은
# depth_agreement median 0.655~0.771, visible rate median 0.950~0.982 / 쌍별 최소 0.943.
MIN_MEDIAN_DEPTH_AGREEMENT = 0.60
MIN_MEDIAN_VISIBLE_RATE = 0.93
MIN_PAIR_VISIBLE_RATE = 0.85


@pytest.fixture(scope="module")
def reprojection_metrics():
    """METRIC_SAMPLES × 4대에 대해 depth 일치율과 가시 라벨 일치율을 한 번만 계산한다."""
    cv2 = pytest.importorskip("cv2")
    if not (DATASET_ROOT / "depth_maps" / "raw_data").exists():
        pytest.skip("SynWoodScape depth maps not available locally")

    metrics = {name: {"depth_agreement": [], "visible_rate": []} for name in CAMERAS}
    cameras = {name: load_camera(CALIB_DIR / f"{name}.json") for name in CAMERAS}
    for sample_idx in METRIC_SAMPLES:
        with open(DATASET_ROOT / "lidar_data" / f"{sample_idx}.pkl", "rb") as f:
            lidar = pickle.load(f)
        points_ego = lidar_points_to_ego(lidar["points"])
        labels = np.asarray(lidar["labels"])
        for name in CAMERAS:
            depth = np.load(DATASET_ROOT / "depth_maps" / "raw_data" / f"{sample_idx}_{name}.npy")
            gt_labels = cv2.imread(
                str(DATASET_ROOT / "semantic_annotations" / "gtLabels" / f"{sample_idx}_{name}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            projected = project_points_to_image(cameras[name], points_ego)
            visible = visibility_mask(projected, depth)
            metrics[name]["depth_agreement"].append(float(visible.mean()))
            metrics[name]["visible_rate"].append(
                class_consistency_rate(
                    labels[projected.index],
                    gt_labels[projected.row, projected.col],
                    valid_mask=visible,
                )
            )
    return metrics


@pytest.mark.parametrize("camera_name", CAMERAS)
def test_lidar_range_agrees_with_the_cameras_own_depth_map(camera_name, reprojection_metrics):
    """기하 정확도 회귀 테스트 — 여러 샘플의 **median**으로 판정한다.

    `depth_agreement`는 캘리브레이션 품질만이 아니라 장면 내용(그 시점에 카메라가 얼마나
    가려져 있는지)에도 좌우되므로, 쌍별 하한을 걸면 장면 난이도 기준이 되어버린다.
    168쌍 실측에서 쌍별 최소값은 0.433까지 내려가지만 카메라별 median은 0.663 이상이었다.
    """
    values = reprojection_metrics[camera_name]["depth_agreement"]

    assert np.median(values) > MIN_MEDIAN_DEPTH_AGREEMENT, (
        f"{camera_name}: depth_agreement median={np.median(values):.3f} — "
        f"LiDAR 거리가 카메라 depth map과 맞지 않는다 (extrinsic 오차 의심). values={values}"
    )


@pytest.mark.parametrize("camera_name", CAMERAS)
def test_visible_lidar_points_land_on_matching_semantic_labels(camera_name, reprojection_metrics):
    """가려지지 않은 포인트에서는 semantic 라벨이 일치해야 한다.

    이 지표는 가시성 필터의 자기선택 편향 때문에 기하 오차에 **둔감**하다(카메라를 0.5 m
    옮겨도 0.87~0.99를 유지한다). 세밀한 판별은
    `test_ego_translation_from_calibration`(값 고정)과
    `test_mirror_cameras_see_their_own_side_of_the_vehicle`(좌우 배치)가 담당한다.
    """
    values = reprojection_metrics[camera_name]["visible_rate"]

    assert np.median(values) > MIN_MEDIAN_VISIBLE_RATE, f"{camera_name}: median={np.median(values):.3f}"
    assert min(values) > MIN_PAIR_VISIBLE_RATE, f"{camera_name}: min={min(values):.3f}, values={values}"


@requires_dataset
@requires_depth
@pytest.mark.parametrize(
    "camera_name, y_low, y_high",
    [
        # 미러 카메라는 자기 쪽 차체를 본다. 반대쪽에 놓으면 y 부호가 뒤집힌다
        # (실측: MVL +0.756 → -1.044, MVR -0.757 → +1.043).
        # 상한 0.90은 차체 반폭(3D 박스 기준 0.894 m)이다 — y 크기가 과하게 커지는 것도 막는다.
        ("MVL", 0.30, 0.90),
        ("MVR", -0.90, -0.30),
        # FV/RV는 차량 중심선에 있으므로 차체가 좌우 대칭으로 보인다.
        ("FV", -0.25, 0.25),
        ("RV", -0.25, 0.25),
    ],
)
def test_mirror_cameras_see_their_own_side_of_the_vehicle(camera_name, y_low, y_high):
    """좌우(y) 배치 가드 — `depth_agreement`가 잡지 못하는 오차를 잡는다.

    `depth_agreement`는 카메라 **높이**에는 민감하지만 횡방향 오프셋에는 둔감하다.
    미러 카메라를 반대쪽에 놓아도 0.603/0.901까지 나와서 지표만으로는 통과해 버린다.

    대신 물리적 사실을 직접 검사한다: **왼쪽 미러 카메라는 차량의 왼쪽 면을 본다.**
    depth map에서 `ego-vehicle`(class 24) 픽셀을 역투영해 ego 프레임 y의 중앙값을 보면
    42개 샘플에 걸쳐 ±0.001 이내로 안정적이다.
    """
    cv2 = pytest.importorskip("cv2")
    sample_idx = "00000"
    cam = load_camera(CALIB_DIR / f"{camera_name}.json")
    depth = np.load(DATASET_ROOT / "depth_maps" / "raw_data" / f"{sample_idx}_{camera_name}.npy")
    gt_labels = cv2.imread(
        str(DATASET_ROOT / "semantic_annotations" / "gtLabels" / f"{sample_idx}_{camera_name}.png"),
        cv2.IMREAD_GRAYSCALE,
    )

    is_ego_body = (gt_labels == EGO_VEHICLE_CLASS) & (depth < 20.0)
    rows, cols = np.nonzero(is_ego_body)
    assert len(rows) > 1000, f"{camera_name}: ego-vehicle 픽셀이 너무 적다 ({len(rows)})"
    picked = np.random.default_rng(0).choice(len(rows), size=3000, replace=False)
    points_ego = unproject_depth_to_ego(cam, rows[picked], cols[picked], depth)

    median_y = float(np.median(points_ego[:, 1]))
    assert y_low < median_y < y_high, (
        f"{camera_name}: 차체가 ego y={median_y:.3f}에서 보인다 "
        f"(기대 {y_low}~{y_high}) — 카메라 좌우 배치가 틀렸을 수 있다"
    )
