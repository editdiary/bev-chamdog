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
