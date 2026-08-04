"""SynWoodScape 캘리브레이션 -> Simple-BEV `Segnet.forward()` 입력 텐서 변환.

`Segnet`의 lifting(`utils.vox.unproject_image_to_mem`)은 핀홀 project(4x4 행렬곱 + 원근분할)를
전제하므로 SynWoodScape의 `radial_poly` 어안 캘리브레이션을 그대로 넣을 수 없다. 여기서는
`rho(theta) = k1*theta + k2*theta^2 + ...`가 `theta -> 0` 근방에서 `rho ≈ k1*theta`로
수렴하는 성질을 이용해 k1을 핀홀 focal length로 쓰는 **임시 근사**를 만든다 — 화각 중심부에서만
근사가 맞고 화각 가장자리(어안 왜곡이 큰 영역)에서는 어긋난다. 실제 어안 투영으로의 교체는
ROADMAP Phase 3.2에서 다룬다.

`ego_T_cam`은 근사가 아니라 `projects.geometry.fisheye`가 이미 검증한 카메라 pose
(`camera.rotation`, `camera.translation` — 둘 다 ego 프레임 기준)를 그대로 4x4로 옮긴 것이다.
"""
import numpy as np


def ego_T_cam_from_camera(camera) -> np.ndarray:
    """camera(ego 프레임 기준 pose)를 4x4 (camera -> ego) 동차행렬로 반환한다."""
    ego_T_cam = np.eye(4, dtype=np.float64)
    ego_T_cam[:3, :3] = camera.rotation
    ego_T_cam[:3, 3] = camera.translation
    return ego_T_cam


def pinhole_pix_T_cam_from_camera(camera, out_width: int, out_height: int) -> np.ndarray:
    """`theta -> 0` 1차 근사(k1)를 focal length로 쓰는 임시 핀홀 4x4 intrinsic.

    native(`camera.width` x `camera.height`) 해상도 기준으로 만든 뒤 `(out_width, out_height)`로
    스케일한다. `Segnet`은 자신의 인코더 다운샘플 비율(`Hf/H`, `Wf/W`)만 추가로 스케일하므로,
    여기서 넘기는 `(out_width, out_height)`는 실제로 모델에 넣는 리사이즈된 이미지 크기여야 한다.
    """
    k1 = float(camera.lens.coefficients[0])
    fx = k1
    fy = k1 * float(camera.aspect_ratio)
    x0 = float(camera.cx)
    y0 = float(camera.cy)

    sx = out_width / float(camera.width)
    sy = out_height / float(camera.height)

    pix_T_cam = np.eye(4, dtype=np.float64)
    pix_T_cam[0, 0] = fx * sx
    pix_T_cam[1, 1] = fy * sy
    pix_T_cam[0, 2] = x0 * sx
    pix_T_cam[1, 2] = y0 * sy
    return pix_T_cam
