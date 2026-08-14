import numpy as np

from tools.visualize_predictions import (
    COLOR_DRIVABLE,
    COLOR_OBSTACLE,
    COLOR_UNKNOWN,
    COLOR_VISIBLE,
    format_obstacle_iou,
    occupancy_to_image,
    visibility_to_image,
)


def test_format_obstacle_iou_marks_samples_without_obstacle_in_gt():
    valid = np.ones((1, 2), dtype=bool)
    # GT에 obstacle이 없으면 IoU는 예측과 무관하게 0이라 숫자를 보여주면 오해를 부른다.
    assert format_obstacle_iou(np.zeros((1, 2)), np.zeros((1, 2)), valid) == "n/a (GT obstacle 없음)"
    assert format_obstacle_iou(np.array([[1, 0]]), np.array([[1, 0]]), valid) == "1.000"


def test_visibility_to_image_marks_visible_and_invisible_cells():
    visible = np.array([[True, False]], dtype=bool)

    image = visibility_to_image(visible, upscale=1)
    pixels = np.asarray(image)

    assert tuple(pixels[0, 0]) == COLOR_VISIBLE
    assert tuple(pixels[0, 1]) == COLOR_UNKNOWN


def test_occupancy_to_image_can_use_predicted_visibility_mask():
    occupancy = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    visible = np.array([[True, True], [False, False]], dtype=bool)

    image = occupancy_to_image(occupancy, visible, upscale=1)
    pixels = np.asarray(image)

    assert tuple(pixels[0, 0]) == COLOR_DRIVABLE
    assert tuple(pixels[0, 1]) == COLOR_OBSTACLE
    assert tuple(pixels[1, 0]) == COLOR_UNKNOWN
    assert tuple(pixels[1, 1]) == COLOR_UNKNOWN
