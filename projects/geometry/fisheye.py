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
