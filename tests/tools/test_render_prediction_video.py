"""동영상 프레임 조립·프레임 탐색 테스트 (모델 없이).

레이아웃 계약을 못박는 이유: 상단 RGB / 하단 좌 IPM / 하단 우 예측이라는 배치가 사용자
요구사항이고, 좌우가 뒤바뀌면 그림을 잘못 읽게 되는데 눈으로는 알아채기 어렵다.
"""
import numpy as np
import pytest
from PIL import Image

from tools.render_prediction_video import (
    compose_frame,
    list_frames,
    resolve_camera_files,
)

CAMERA_NAMES = ("front", "left", "right")


def _solid(height, width, colour):
    return np.tile(np.array(colour, np.uint8), (height, width, 1))


def test_compose_frame_puts_ipm_left_and_prediction_right_below_the_cameras():
    cameras = [_solid(72, 128, c) for c in ((255, 0, 0), (0, 255, 0), (0, 0, 255))]
    ipm = _solid(10, 10, (255, 255, 0))          # 노랑 = IPM
    pred = _solid(10, 10, (0, 255, 255))         # 시안 = 예측

    canvas = compose_frame(cameras, CAMERA_NAMES, ipm, pred, "제목", bev_upscale=4,
                           camera_width=128)
    pixels = np.asarray(canvas)

    # 하단 두 패널의 중앙을 찍어 좌우가 뒤바뀌지 않았는지 본다.
    yellow = np.argwhere(np.all(pixels == (255, 255, 0), axis=-1))
    cyan = np.argwhere(np.all(pixels == (0, 255, 255), axis=-1))
    assert yellow.size and cyan.size
    assert yellow[:, 1].mean() < cyan[:, 1].mean(), "IPM이 예측보다 왼쪽이어야 한다"
    # 카메라 3장은 그 위에 있어야 한다.
    red = np.argwhere(np.all(pixels == (255, 0, 0), axis=-1))
    assert red[:, 0].max() < yellow[:, 0].min(), "RGB 패널이 BEV 패널보다 위여야 한다"
    # BEV는 정수배 확대만 한다 -- 보간이 섞이면 클래스 색 사이에 없는 색이 생긴다.
    assert len(yellow) == (10 * 4) ** 2


def test_list_frames_accepts_both_with_and_without_the_rgb_images_level(tmp_path):
    """라벨된 시퀀스는 `<seq>/rgb_images/<frame>/`이고, 새로 받은 영상 프레임은
    상위 폴더가 없을 수 있다. 둘 다 받아야 한다."""
    labelled = tmp_path / "seq"
    for name in ("sample_000001", "sample_000000"):
        (labelled / "rgb_images" / name).mkdir(parents=True)
    assert [p.name for p in list_frames(labelled)] == ["sample_000000", "sample_000001"]

    raw = tmp_path / "raw"
    for name in ("0002", "0001"):
        (raw / name).mkdir(parents=True)
    assert [p.name for p in list_frames(raw)] == ["0001", "0002"]


def test_list_frames_fails_loudly_when_there_are_no_frame_directories(tmp_path):
    """조용히 0프레임 동영상을 만들면 원인을 찾는 데 시간이 든다."""
    (tmp_path / "seq").mkdir()
    (tmp_path / "seq" / "cam_front.jpg").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="프레임 디렉터리"):
        list_frames(tmp_path / "seq")


def test_compose_frame_returns_a_pil_image_sized_to_fit_both_bev_panels():
    cameras = [_solid(36, 64, (10, 10, 10)) for _ in CAMERA_NAMES]
    canvas = compose_frame(cameras, CAMERA_NAMES, _solid(120, 120, (1, 2, 3)),
                           _solid(120, 120, (4, 5, 6)), "t", bev_upscale=2, camera_width=100)
    assert isinstance(canvas, Image.Image)
    assert canvas.width >= 2 * 120 * 2      # BEV 두 장이 나란히 들어가야 한다


def _orientation(tmp_path):
    path = tmp_path / "orientation.json"
    path.write_text('{"cam0": "front", "cam1": "right", "cam2": "rear", "cam3": "left"}')
    return path


def test_camera_files_resolve_by_name_when_that_convention_is_present(tmp_path):
    frame = tmp_path / "sample_000000"
    frame.mkdir()
    for name in CAMERA_NAMES:
        (frame / f"cam_{name}.jpg").write_bytes(b"")

    resolved = resolve_camera_files(frame, CAMERA_NAMES, _orientation(tmp_path))

    assert {k: v.name for k, v in resolved.items()} == {
        "front": "cam_front.jpg", "left": "cam_left.jpg", "right": "cam_right.jpg"
    }


def test_camera_files_fall_back_to_the_index_convention_via_orientation(tmp_path):
    """원본 전체 시퀀스는 `cam0..cam3.jpg`이고 매핑은 `orientation.json`이 정한다.

    **left가 cam3, right가 cam1이다** -- 인덱스 순서대로 이름을 붙이면 좌우가 뒤바뀐다.
    그 실수는 그림으로는 알아채기 어려우므로 여기서 못박는다.
    """
    frame = tmp_path / "frame_000000"
    frame.mkdir()
    for index in range(4):
        (frame / f"cam{index}.jpg").write_bytes(b"")

    resolved = resolve_camera_files(frame, CAMERA_NAMES, _orientation(tmp_path))

    assert resolved["front"].name == "cam0.jpg"
    assert resolved["right"].name == "cam1.jpg"
    assert resolved["left"].name == "cam3.jpg"      # cam2는 rear -- 쓰지 않는다


def test_missing_camera_files_name_the_paths_that_were_tried(tmp_path):
    frame = tmp_path / "frame_000000"
    frame.mkdir()
    (frame / "cam0.jpg").write_bytes(b"")           # front만 있고 좌우가 없다

    with pytest.raises(FileNotFoundError, match="cam<idx>"):
        resolve_camera_files(frame, CAMERA_NAMES, _orientation(tmp_path))
