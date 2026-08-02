"""SynWoodScape fisheye(radial_poly) 카메라 로더.

`calibration_data/*.json`의 intrinsic(radial_poly k1~k4, cx/cy_offset, aspect_ratio)은
그대로 쓸 수 있지만, **extrinsic은 그대로 쓰면 안 된다**. 아래 §1, §2의 두 가지 보정을
적용해야 SynWoodScape 자신의 depth map / semantic 라벨 / 3D 박스와 맞는다.

투영 대상 프레임은 `projects.geometry.frames`가 정의하는 **ego 프레임**
(X=전방, Y=좌측, Z=상방, 오른손계, 원점=차량 중심·지면 높이)이다.

--------------------------------------------------------------------------------
§1. extrinsic 회전: roll 부호가 뒤집혀 있다
--------------------------------------------------------------------------------
카메라(OpenCV 규약: x=오른쪽, y=아래, z=광축)를 ego 프레임으로 보내는 올바른 회전은
CARLA 장착 각도 (pitch p, yaw y, roll r)에 대해

    R_true = S · M_carla(p, y, r) · C            (S = diag(1,-1,1), C = OpenCV→CARLA센서)
           = Rz(-y) · Ry(-p) · Rx(+r) · BASE     (BASE = S·C = OpenCV→ego)

인데, JSON의 quaternion은 CARLA 각도 세 개를 **일괄 부호 반전**해 오른손 ZYX에 그대로 넣은
것에 해당한다:

    R_json = Rz(-y) · Ry(-p) · Rx(-r) · BASE

두 식의 차이는 **roll(X축) 성분의 부호 하나뿐**이다. 즉 `R_json @ BASE.T`를 ZYX로 분해해
roll만 부호를 뒤집으면 올바른 회전이 된다. (올바른 CARLA(왼손)→ego(오른손) 변환 `S·M·S`는
CARLA 파라미터 기준으로 yaw와 roll만 반전하고 pitch는 유지한다. 세 각을 일괄 반전한 것은
전형적인 handedness 변환 실수다.)

실측 확인 — 카메라 자신의 depth map에서 노면(class 6/7) 픽셀을 역투영해 평면을 맞추고,
그 법선을 extrinsic이 예측하는 법선(`R^T·(0,0,1)`)과 비교한 각도 오차:

    | cam | mount roll | 보정 전 | 보정 후 |
    |-----|-----------:|--------:|--------:|
    | FV  |     -2.37° |   4.46° |   0.46° |
    | MVL |    -10.18° |   4.70° |   0.18° |
    | MVR |     +8.38° |   3.99° |   0.17° |
    | RV  |     -0.12° |   0.18° |   0.16° |

roll이 거의 0인 RV만 원래도 맞았다는 점, 그리고 오차 크기가 각 카메라의 `2·|roll|`을
법선의 면내 성분에 투영한 값과 정량적으로 일치한다는 점이 이 진단의 근거다.

--------------------------------------------------------------------------------
§2. extrinsic translation: CARLA 규약 + 미러 카메라의 y/z 성분 전치
--------------------------------------------------------------------------------
JSON 키 이름이 `"translation used in CARLA (CARLA reference)"`인 대로 이 값은 **CARLA 규약
(Y=우측)** 이다. ego 프레임(Y=좌측)으로 쓰려면 Y 부호를 뒤집어야 한다.

추가로 미러 카메라(MVL/MVR)는 저장된 3-튜플의 **y와 z 성분이 서로 바뀌어** 있다:

    MVL: JSON (0.8, 1.0, -0.9) → 전치 (0.8, -0.9, 1.0) → Y 반전 (0.8, +0.9, 1.0)
    MVR: JSON (0.8, 1.0,  0.9) → 전치 (0.8,  0.9, 1.0) → Y 반전 (0.8, -0.9, 1.0)

전치를 빼면 MVL은 **지면 아래(z=-0.9) 이고 좌우도 반대(y=-1.0 → 우측)** 로 놓이고,
MVR은 좌우는 맞지만 높이가 10 cm 낮고 횡방향이 10 cm 넓어진다. FV/RV는 y=0이라 Y 반전이
값에 드러나지 않으며, y/z 전치는 **적용하면 안 된다**(전치하면 (1.92, 0.9, 0.0)처럼
지면 높이에 놓여 버린다 — 실측 카메라 높이 0.9 m와 맞지 않는다).

실측 확인 — LiDAR 포인트를 투영해 depth map과의 상대오차를 robust 손실로 두고 (R, t)를
자유롭게 재추정하면(6-DOF, 3개 샘플 합산):

    | cam | 위 규칙의 t          | 자유 추정 t                  | 회전 추가 보정 |
    |-----|----------------------|------------------------------|---------------:|
    | FV  | ( 1.92,  0.0, 0.9)   | ( 1.908,  0.001, 0.902)      |         0.66°  |
    | MVL | ( 0.8,   0.9, 1.0)   | ( 0.764,  0.869, 1.0005)     |         0.17°  |
    | MVR | ( 0.8,  -0.9, 1.0)   | ( 0.792, -0.903, 0.9998)     |         0.02°  |
    | RV  | (-1.92,  0.0, 0.9)   | (-1.918, -0.004, 0.900)      |         0.02°  |

네 카메라 모두 4 cm 이내로 일치한다. 자유 추정값은 데이터에 과적합될 수 있으므로
코드에는 넣지 않고, 위의 규칙(전치 + Y 반전)만 적용한다.
"""
import json
import sys
from pathlib import Path

import numpy as np

_WOODSCAPE_CALIB_DIR = (
    Path(__file__).resolve().parents[2] / "third_party/datasets/WoodScape/scripts/calibration"
)
if str(_WOODSCAPE_CALIB_DIR) not in sys.path:
    sys.path.insert(0, str(_WOODSCAPE_CALIB_DIR))

from projection import Camera, RadialPolyCamProjection  # noqa: E402  (WoodScape submodule; see docs/project_structure.md)
from scipy.spatial.transform import Rotation as SciRot  # noqa: E402

# OpenCV 카메라 프레임(x=오른쪽, y=아래, z=광축) → ego 프레임(x=전방, y=좌측, z=상방).
# 열이 카메라 축의 ego 프레임 표현이다: x_cam→-y_ego, y_cam→-z_ego, z_cam→+x_ego.
CAMERA_TO_EGO_BASE = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)

# translation 3-튜플의 y/z 성분이 전치되어 저장된 카메라 (§2)
_YZ_TRANSPOSED_TRANSLATION = frozenset({"MVL", "MVR"})

_TRANSLATION_KEYS = ("translation", "translation used in CARLA (CARLA reference)")


def ego_rotation_from_quaternion(quaternion) -> np.ndarray:
    """calibration JSON의 quaternion → 올바른 camera→ego 회전행렬 (roll 부호 보정 포함, §1)."""
    rotation = SciRot.from_quat(quaternion).as_matrix()
    yaw, pitch, roll = SciRot.from_matrix(rotation @ CAMERA_TO_EGO_BASE.T).as_euler(
        "ZYX", degrees=True
    )
    corrected_mount = SciRot.from_euler("ZYX", [yaw, pitch, -roll], degrees=True).as_matrix()
    return corrected_mount @ CAMERA_TO_EGO_BASE


def ego_translation_from_calibration(camera_name: str, translation) -> np.ndarray:
    """calibration JSON의 translation → ego 프레임에서의 카메라 위치 (§2)."""
    translation = np.asarray(translation, dtype=np.float64)
    if camera_name in _YZ_TRANSPOSED_TRANSLATION:
        translation = translation[[0, 2, 1]]
    return translation * np.array([1.0, -1.0, 1.0])  # CARLA(Y=우측) → ego(Y=좌측)


def load_camera(calibration_json_path) -> Camera:
    """SynWoodScape `calibration_data/*.json`에서 `Camera`를 만든다.

    반환된 `Camera.project_3d_to_2d`는 **ego 프레임**(`projects.geometry.frames` 참고)
    좌표를 받는다. extrinsic 보정은 위 모듈 docstring의 §1·§2를 따른다.
    """
    calibration_json_path = Path(calibration_json_path)
    with open(calibration_json_path) as f:
        config = json.load(f)

    intrinsic = config["intrinsic"]
    extrinsic = config["extrinsic"]
    camera_name = config.get("name") or calibration_json_path.stem

    translation = next((extrinsic[key] for key in _TRANSLATION_KEYS if key in extrinsic), None)
    if translation is None:
        raise KeyError(
            f"{calibration_json_path}: extrinsic에 translation 키가 없다 "
            f"(찾은 키: {sorted(extrinsic)}, 기대한 키 중 하나: {list(_TRANSLATION_KEYS)})"
        )

    return Camera(
        rotation=ego_rotation_from_quaternion(extrinsic["quaternion"]),
        translation=ego_translation_from_calibration(camera_name, translation),
        lens=RadialPolyCamProjection(
            [intrinsic["k1"], intrinsic["k2"], intrinsic["k3"], intrinsic["k4"]]
        ),
        size=(intrinsic["width"], intrinsic["height"]),
        principle_point=(intrinsic["cx_offset"], intrinsic["cy_offset"]),
        aspect_ratio=intrinsic["aspect_ratio"],
    )
