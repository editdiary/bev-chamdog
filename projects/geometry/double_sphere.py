"""자체 수집 리그(Double Sphere 어안 3-cam)의 카메라 모델과 ego extrinsic 체인.

SynWoodScape 쪽 대응물은 `projects/geometry/fisheye.py`(radial_poly)다. 두 데이터셋은
렌즈 모델과 캘리브레이션 포맷이 다르므로 로더를 분리하고, 이후 단계(`Vox_util` 래핑,
`Segnet` 입력 텐서)는 공통 규약(ego 프레임 = X 전방 / Y 좌측 / Z 상방, 원점은 지면)으로
맞춘다.

--------------------------------------------------------------------------------
§1. extrinsic 체인 — calib.yaml은 "front 카메라 기준"이다
--------------------------------------------------------------------------------
`calib.yaml`의 `extrinsics`는 전부 `T_<cam>_front`(front 프레임 점 -> cam 프레임)이고,
여기에 `T_front_lidar`(lidar -> front)가 하나 더 있다. 라벨 생성 파이프라인(`ipm.py`,
`slab_label.py`)이 **ego = body = LiDAR 프레임**(x 전방 / y 좌 / z 상)을 쓰므로:

    cam_T_lidar = T_cam_front @ T_front_lidar
    lidar_T_cam = inv(cam_T_lidar)

--------------------------------------------------------------------------------
§2. 원점을 LiDAR에서 지면으로 내린다
--------------------------------------------------------------------------------
`projects/datasets/simplebev_vox.py`의 ref 프레임은 원점이 **지면 높이**임을 전제한다
(`vox_bounds`가 ego z=0 기준 ±height_margin 슬래브를 잡고, Y=1이면 정확히 z=0 한 평면에서
이미지 특징을 샘플링한다). LiDAR 원점은 지면 위에 떠 있으므로 z 평행이동 하나가 필요하다:

    ego_T_cam = Translate(0, 0, -GROUND_Z_IN_LIDAR_M) @ lidar_T_cam

`GROUND_Z_IN_LIDAR_M = -0.87`인 근거: 라벨 파이프라인의 `slab_label.camera_observable`이
지면 판정 평면으로 `z_body = -0.87`을 쓴다. `ipm.py`는 같은 상수를 *카메라 렌즈* 높이로
해석해 지면을 `C_z - 0.87 = -0.8921`로 잡아 2.2 cm 차이가 나는데, 실제로 배포된
`common/self_mask.png`를 두 값으로 재투영해 대조하면 -0.87 쪽이 확연히 잘 맞는다
(IoU 0.606 vs 0.498). 라벨과 같은 규약을 따라야 하므로 -0.87을 쓴다.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_EPS = 1e-9

# 라벨 파이프라인이 지면 판정에 쓴 body(LiDAR) 프레임 z. 위 §2 참고.
GROUND_Z_IN_LIDAR_M = -0.87

# 자체 리그의 fine-tuning 카메라 구성. `ipm.py`/`slab_label.py`가 라벨을 만들 때 쓴 것과
# 같아야 한다(rear는 캘리브레이션에만 있고 라벨 생성에 쓰이지 않았다).
FINETUNE_CAMERA_NAMES = ("front", "left", "right")


@dataclass(frozen=True)
class DoubleSphereCamera:
    """Usenko et al. 2018 Double Sphere 모델. intrinsics 순서는 Kalibr/`calib.yaml`과 같다."""

    xi: float
    alpha: float
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    name: str = ""

    @classmethod
    def from_intrinsics(cls, intrinsics, resolution, name="") -> "DoubleSphereCamera":
        xi, alpha, fx, fy, cx, cy = (float(v) for v in intrinsics)
        w, h = resolution
        return cls(xi, alpha, fx, fy, cx, cy, int(w), int(h), name)

    @property
    def domain_cos_limit(self) -> float:
        """`z > -w2 * |P|` 판정에 쓰는 w2 — 모델 정의역(투영이 유효한 원뿔)의 경계."""
        w1 = self.alpha / (1.0 - self.alpha) if self.alpha <= 0.5 else (1.0 - self.alpha) / self.alpha
        return (w1 + self.xi) / np.sqrt(2.0 * w1 * self.xi + self.xi ** 2 + 1.0)

    def project(self, points_cam):
        """(..., 3) 카메라 프레임(OpenCV) 점 -> (u, v, valid).

        `valid=False`인 점은 모델 정의역 밖이라 (u, v)를 신뢰하면 안 된다.
        """
        P = np.asarray(points_cam, dtype=np.float64)
        x, y, z = P[..., 0], P[..., 1], P[..., 2]

        d1 = np.sqrt(x * x + y * y + z * z)
        k = self.xi * d1 + z
        d2 = np.sqrt(x * x + y * y + k * k)
        denom = self.alpha * d2 + (1.0 - self.alpha) * k

        valid = (z > -self.domain_cos_limit * d1) & (np.abs(denom) > _EPS)
        safe = np.where(np.abs(denom) > _EPS, denom, 1.0)
        return self.fx * x / safe + self.cx, self.fy * y / safe + self.cy, valid

    def unproject(self, u, v):
        """픽셀 좌표 -> (단위 방향벡터 (..., 3), valid). `project`의 역."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy
        r2 = mx * mx + my * my

        if self.alpha > 0.5:
            valid = r2 <= 1.0 / (2.0 * self.alpha - 1.0)
        else:
            valid = np.ones_like(r2, dtype=bool)
        r2c = np.where(valid, r2, 0.0)

        mz = (1.0 - self.alpha ** 2 * r2c) / (
            self.alpha * np.sqrt(np.maximum(1.0 - (2.0 * self.alpha - 1.0) * r2c, 0.0))
            + (1.0 - self.alpha)
        )
        num = mz * self.xi + np.sqrt(np.maximum(mz * mz + (1.0 - self.xi ** 2) * r2c, 0.0))
        coef = num / (mz * mz + r2c + _EPS)

        dirs = np.stack([coef * mx, coef * my, coef * mz - self.xi], axis=-1)
        return dirs / np.maximum(np.linalg.norm(dirs, axis=-1, keepdims=True), _EPS), valid


def se3_inverse(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def load_cameras(calib_path) -> dict:
    """`calib.yaml` -> {카메라 이름: DoubleSphereCamera}."""
    calib = yaml.safe_load(Path(calib_path).read_text())
    return {
        name: DoubleSphereCamera.from_intrinsics(c["intrinsics"], c["resolution"], name=name)
        for name, c in calib["cameras"].items()
    }


def load_ego_T_cams(calib_path, ground_z_in_lidar_m: float = GROUND_Z_IN_LIDAR_M) -> dict:
    """`calib.yaml` -> {카메라 이름: 4x4 ego_T_cam}. ego = 지면 원점, X 전방 / Y 좌 / Z 상.

    §1의 `T_<cam>_front` + `T_front_lidar` 체인을 뒤집고, §2의 z 평행이동을 얹는다.
    """
    calib = yaml.safe_load(Path(calib_path).read_text())
    extrinsics = calib["extrinsics"]
    front_T_lidar = np.array(extrinsics["T_front_lidar"], dtype=np.float64)

    lidar_to_ego = np.eye(4, dtype=np.float64)
    lidar_to_ego[2, 3] = -ground_z_in_lidar_m

    ego_T_cams = {}
    for key, matrix in extrinsics.items():
        if not (key.startswith("T_") and key.endswith("_front")):
            continue  # T_front_lidar 등 비-카메라 키
        name = key[len("T_"):-len("_front")]
        cam_T_lidar = np.array(matrix, dtype=np.float64) @ front_T_lidar
        ego_T_cams[name] = lidar_to_ego @ se3_inverse(cam_T_lidar)
    return ego_T_cams


def load_camera_index_names(orientation_path) -> dict:
    """`orientation.json` -> {0: "front", 1: "right", ...} (cam<idx> 파일명 매핑)."""
    orientation = json.loads(Path(orientation_path).read_text())
    return {int(key.replace("cam", "")): value for key, value in orientation.items()}
