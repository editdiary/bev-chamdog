import numpy as np
import pytest
import torch

from projects.common.polar import RayIndex
from tools.analyze_error_structure import (
    azimuth_profile,
    decile_contrast,
    frame_rows,
    main,
    mirror_asymmetry,
    split_by_quality,
)


def _dummy_rays() -> RayIndex:
    """격자 안이 하나도 없는 광선 인덱스 -- range 값을 보지 않는 테스트용."""
    return RayIndex(
        rows=np.zeros((1, 1), dtype=np.int64), cols=np.zeros((1, 1), dtype=np.int64),
        radii_m=np.zeros(1), inside=np.zeros((1, 1), dtype=bool),
    )


def _mask(rows) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)[None, None]


def test_frame_rows_keeps_every_frame_separate_instead_of_averaging_the_batch():
    """배치 평균을 내면 '어느 프레임이 나쁜가'가 사라진다 -- 이 도구의 존재 이유가 그것이다."""
    # 프레임 0은 완벽, 프레임 1은 절반이 틀리다.
    pred = torch.cat([_mask([[1, 1], [1, 1]]), _mask([[1, 1], [1, 1]])])
    gt = torch.cat([_mask([[1, 1], [1, 1]]), _mask([[1, 1], [0, 0]])])
    valid = torch.ones_like(pred)

    rows = frame_rows(pred, gt, valid, _dummy_rays(),
                      [("raws1", "000"), ("raws1", "001")])

    assert [r["sample_id"] for r in rows] == ["000", "001"]
    assert rows[0]["iou_free"] == pytest.approx(1.0)
    assert rows[1]["iou_free"] == pytest.approx(0.5)


def test_over_share_is_one_when_the_model_only_over_predicts_free():
    """사용자가 관찰한 기제('애매한 통로를 지나갈 수 있다고 본다')는 fatal 쪽 오차다.
    이 값이 없으면 낮은 `iou_free`가 과잉 낙관인지 과잉 보수인지 구별할 수 없다."""
    pred = _mask([[1, 1], [1, 1]])
    gt = _mask([[1, 1], [0, 0]])
    valid = torch.ones_like(pred)

    row = frame_rows(pred, gt, valid, _dummy_rays(), [("raws1", "000")])[0]

    assert row["fatal_cells"] == 2 and row["miss_cells"] == 0
    assert row["over_share"] == pytest.approx(1.0)


def test_over_share_is_zero_when_the_model_only_misses_free():
    pred = _mask([[1, 1], [0, 0]])
    gt = _mask([[1, 1], [1, 1]])
    valid = torch.ones_like(pred)

    row = frame_rows(pred, gt, valid, _dummy_rays(), [("raws1", "000")])[0]

    assert row["fatal_cells"] == 0 and row["miss_cells"] == 2
    assert row["over_share"] == pytest.approx(0.0)


def test_azimuth_profile_bins_thetas_and_ignores_unpaired_rays():
    """짝이 안 맞은 광선(NaN)을 0으로 세면 오차가 실제보다 작아 보인다."""
    deltas = np.array([
        [0.1, 0.3, np.nan, np.nan],
        [0.1, 0.3, np.nan, np.nan],
    ])

    profile = azimuth_profile(deltas, n_bins=2)

    assert profile[0]["n"] == 4  # 앞 두 방위각 × 2프레임
    assert profile[0]["mae"] == pytest.approx(0.2)
    assert profile[1]["n"] == 0
    assert np.isnan(profile[1]["mae"])


def test_azimuth_profile_rejects_bin_counts_that_do_not_divide_the_rays():
    """나누어지지 않으면 마지막 구간만 좁아져 프로파일이 조용히 왜곡된다."""
    with pytest.raises(ValueError, match="n_bins"):
        azimuth_profile(np.zeros((1, 5)), n_bins=2)


def test_mirror_asymmetry_pairs_each_azimuth_with_its_mirror():
    """+θ와 −θ를 짝지어야 좌우 extrinsic 결함이 라벨 애매성과 갈라진다."""
    # 4구간: 0-90(전방 좌), 90-180, 180-270, 270-360(전방 우). 짝은 (1, 3)뿐이다.
    profile = [
        {"deg_from": 0.0, "mae": 0.1, "bias": 0.1},
        {"deg_from": 90.0, "mae": 0.5, "bias": 0.5},
        {"deg_from": 180.0, "mae": 0.2, "bias": 0.2},
        {"deg_from": 270.0, "mae": 0.1, "bias": -0.1},
    ]

    pairs = mirror_asymmetry(profile)

    assert len(pairs) == 1
    assert pairs[0]["deg"] == 90.0
    assert pairs[0]["mae_ccw"] == 0.5 and pairs[0]["mae_cw"] == 0.1
    assert pairs[0]["mae_diff"] == pytest.approx(0.4)


def test_azimuth_profile_normalizes_by_gt_range_so_distance_is_not_read_as_azimuth():
    """전방은 통로가 길어 `r_gt`가 크다. meter 오차만 보면 거리 효과가 방위각 구조로
    보인다 -- 같은 상대오차(10 %)인데 meter mae는 4배 차이가 나는 이 경우가 그 예다."""
    deltas = np.array([[0.4, 0.1]])
    gt_ranges = np.array([[4.0, 1.0]])

    profile = azimuth_profile(deltas, gt_ranges, n_bins=2)

    assert profile[0]["mae"] == pytest.approx(0.4)
    assert profile[1]["mae"] == pytest.approx(0.1)
    assert profile[0]["rel_mae"] == pytest.approx(0.1)
    assert profile[1]["rel_mae"] == pytest.approx(0.1)
    assert profile[0]["r_gt_mean"] == pytest.approx(4.0)


def test_azimuth_profile_rejects_gt_ranges_that_do_not_match_the_deltas():
    with pytest.raises(ValueError, match="gt_ranges"):
        azimuth_profile(np.zeros((2, 4)), np.zeros((1, 4)), n_bins=2)


def test_split_by_quality_separates_frames_at_the_iou_median():
    """전방 bias가 두 집단에 똑같이 있으면 기하 결함, 나쁜 쪽에만 몰리면 라벨 애매성이다 --
    이 분리가 틀리면 그 판정 자체가 무의미해진다."""
    rows = [{"iou_free": 0.9}, {"iou_free": 0.5}]
    deltas = np.array([[0.1, 0.1], [0.9, 0.9]])   # 좋은 프레임 0.1, 나쁜 프레임 0.9
    gt_ranges = np.ones_like(deltas)

    groups = split_by_quality(rows, deltas, gt_ranges, n_bins=1)

    assert groups["worst_half"]["n_frames"] == 1
    assert groups["worst_half"]["mean_iou"] == pytest.approx(0.5)
    assert groups["worst_half"]["profile"][0]["bias"] == pytest.approx(0.9)
    assert groups["best_half"]["mean_iou"] == pytest.approx(0.9)
    assert groups["best_half"]["profile"][0]["bias"] == pytest.approx(0.1)


def test_decile_contrast_reports_over_share_at_both_ends_of_the_iou_distribution():
    """상관계수 하나로 뭉개면 '나쁜 프레임이 어떤 종류로 나쁜가'에 답할 수 없다."""
    rows = [{"iou_free": 0.1 * i, "over_share": 1.0 if i < 5 else 0.0} for i in range(10)]

    contrast = decile_contrast(rows)

    assert contrast["k"] == 1
    assert contrast["worst_iou"] == pytest.approx(0.0)
    assert contrast["best_iou"] == pytest.approx(0.9)
    assert contrast["worst_over_share"] == pytest.approx(1.0)
    assert contrast["best_over_share"] == pytest.approx(0.0)


def test_main_rejects_an_unknown_formulation():
    """정식화를 틀리면 head 채널 수가 안 맞는다 -- 데이터를 읽기 전에 막는다."""
    with pytest.raises(ValueError, match="formulation"):
        main(checkpoint="/nonexistent.pth", formulation="binaryy")
