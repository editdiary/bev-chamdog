from pathlib import Path

import numpy as np
import pytest
import torch

from projects.bev_gt.grid import ROBOT_GRID_SPEC, OccupancyGridSpec
from projects.bev_gt.ipm import render_ipm
from projects.common.polar import RAY_CENSORED, RAY_NO_FREE, RAY_OK, build_ray_index
from projects.common.three_class_panels import (
    CLASS_COLOURS,
    ERROR_COLOURS,
    draw_range_profile,
    load_font,
    render_classes,
    render_errors,
)
from projects.datasets.robot_simplebev import split_samples_within_sequences
from projects.geometry.double_sphere import (
    FINETUNE_CAMERA_NAMES,
    load_cameras,
    load_ego_T_cams,
)
from tools.visualize_robot_predictions import _SORT_KEYS, _nan_to_inf, sample_scores


@pytest.fixture
def tiny_range_profile_fixture():
    grid_spec = OccupancyGridSpec(front_m=1.0, rear_m=1.0, half_width_m=1.0, cell_m=0.5)
    return grid_spec, build_ray_index(grid_spec, n_theta=4)


def _row(values):
    """`(1, 1, 1, N)` bool 텐서 -- 패널 렌더러가 요구하는 최소 형상."""
    return torch.tensor(values, dtype=torch.bool).view(1, 1, 1, -1)


def _parts(free, occupied, unknown):
    return {"free": _row(free), "occupied": _row(occupied), "unknown": _row(unknown)}


def _on_real_grid(values):
    """`(1, 1, 120, 120)`의 첫 행 앞쪽 N칸에만 값을 놓는다.

    `sample_scores`는 `range_error`를 통과하고 그 광선 인덱스가 `ROBOT_GRID_SPEC` 기준으로
    만들어지므로, 마스크가 실제 격자 크기여야 한다. 나머지 칸은 `valid=0`으로 두어 지표가
    이 N칸만 보게 만든다.
    """
    grid = torch.zeros(1, 1, ROBOT_GRID_SPEC.n_rows, ROBOT_GRID_SPEC.n_cols, dtype=torch.bool)
    grid[0, 0, 0, :len(values)] = torch.tensor(values, dtype=torch.bool)
    return grid


def _grid_parts(free, occupied, unknown):
    return {"free": _on_real_grid(free), "occupied": _on_real_grid(occupied),
            "unknown": _on_real_grid(unknown)}


def test_class_panel_uses_a_distinct_colour_per_class():
    parts = {
        "free": torch.tensor([[True, False, False]]),
        "occupied": torch.tensor([[False, True, False]]),
        "unknown": torch.tensor([[False, False, True]]),
    }
    valid = torch.tensor([[True, True, True]])

    panel = render_classes(parts, valid)

    assert panel.shape == (1, 3, 3)
    assert tuple(panel[0, 0]) == CLASS_COLOURS["free"]
    assert tuple(panel[0, 1]) == CLASS_COLOURS["occupied"]
    assert tuple(panel[0, 2]) == CLASS_COLOURS["unknown"]
    assert len({CLASS_COLOURS[name] for name in ("free", "occupied", "unknown")}) == 3


def test_invalid_cells_get_their_own_colour_not_unknown():
    """`valid=0`은 수집 아티팩트다. `unknown`과 같은 색으로 칠하면 검수에서 구분이 안 된다."""
    parts = {
        "free": torch.tensor([[False]]),
        "occupied": torch.tensor([[False]]),
        "unknown": torch.tensor([[False]]),
    }

    panel = render_classes(parts, torch.tensor([[False]]))

    assert tuple(panel[0, 0]) == CLASS_COLOURS["invalid"]
    assert CLASS_COLOURS["invalid"] != CLASS_COLOURS["unknown"]


def test_error_map_separates_fatal_from_miss_and_from_class_confusion():
    """오차 지도의 색이 지표 이름과 1:1로 대응해야 한다 -- 그림에서 본 색을 로그의 숫자로
    바로 연결하는 것이 이 패널의 목적이다.

    4칸: [fatal, miss, occupied<->unknown 혼동, 정답]
    """
    pred = {
        "free": torch.tensor([[True, False, False, True]]),
        "occupied": torch.tensor([[False, True, True, False]]),
        "unknown": torch.tensor([[False, False, False, False]]),
    }
    gt = {
        "free": torch.tensor([[False, True, False, True]]),
        "occupied": torch.tensor([[True, False, False, False]]),
        "unknown": torch.tensor([[False, False, True, False]]),
    }

    panel = render_errors(pred, gt, torch.tensor([[True, True, True, True]]))

    assert tuple(panel[0, 0]) == ERROR_COLOURS["fatal"]
    assert tuple(panel[0, 1]) == ERROR_COLOURS["miss"]
    assert tuple(panel[0, 2]) == ERROR_COLOURS["occ_unknown"]
    assert tuple(panel[0, 3]) == ERROR_COLOURS["correct"]
    assert len(set(map(tuple, panel[0]))) == 4


def test_every_prediction_and_gt_class_pair_gets_the_expected_error_colour():
    """(pred 클래스 × GT 클래스) 9가지 조합 전부의 색을 못박는다.

    세 오류 범주는 서로소여야 한다 -- 겹치면 지도가 칠하는 순서에 의존하게 되고, 같은 그림이
    구현을 바꿀 때마다 다른 색을 내놓는다. 조합 표를 여기 손으로 적어 두면 조건을 하나만
    바꿔도 어느 조합이 어떻게 옮겨갔는지 드러난다.

    기대표 (행 = pred, 열 = GT):
                     GT free   GT occupied   GT unknown
        pred free    correct   fatal         fatal
        pred occ     miss      correct       occ_unknown
        pred unknown miss      occ_unknown   correct
    """
    names = ("free", "occupied", "unknown")
    expected = {
        ("free", "free"): "correct", ("free", "occupied"): "fatal", ("free", "unknown"): "fatal",
        ("occupied", "free"): "miss", ("occupied", "occupied"): "correct",
        ("occupied", "unknown"): "occ_unknown",
        ("unknown", "free"): "miss", ("unknown", "occupied"): "occ_unknown",
        ("unknown", "unknown"): "correct",
    }

    for pred_name in names:
        for gt_name in names:
            pred = _parts(*([pred_name == name] for name in names))
            gt = _parts(*([gt_name == name] for name in names))
            panel = render_errors(pred, gt, _row([True]))
            assert tuple(panel[0, 0]) == ERROR_COLOURS[expected[(pred_name, gt_name)]], (
                pred_name, gt_name,
            )


def test_error_map_marks_invalid_cells_separately_from_correct_ones():
    """`valid=0`은 애초에 채점 대상이 아니다. `correct`로 칠하면 "맞았다"로 읽힌다."""
    pred = _parts([True], [False], [False])
    gt = _parts([False], [True], [False])

    panel = render_errors(pred, gt, _row([False]))

    assert tuple(panel[0, 0]) == ERROR_COLOURS["invalid"]
    assert ERROR_COLOURS["invalid"] != ERROR_COLOURS["correct"]


def test_range_profile_draws_only_ray_ok(tiny_range_profile_fixture):
    grid_spec, rays = tiny_range_profile_fixture
    panel = np.zeros((grid_spec.n_rows, grid_spec.n_cols, 3), np.uint8)
    colour = (17, 34, 51)

    draw_range_profile(
        panel, np.full(4, 0.5), np.array([RAY_OK, RAY_NO_FREE, RAY_CENSORED, RAY_NO_FREE]),
        rays, grid_spec, colour, RAY_OK,
    )

    assert tuple(panel[0, 2]) == colour
    assert np.count_nonzero(panel) == 3


def test_range_profile_theta_90_degrees_moves_to_vehicle_left(tiny_range_profile_fixture):
    grid_spec, rays = tiny_range_profile_fixture
    panel = np.zeros((grid_spec.n_rows, grid_spec.n_cols, 3), np.uint8)
    colour = (17, 34, 51)

    draw_range_profile(
        panel, np.full(4, 0.5), np.array([RAY_NO_FREE, RAY_OK, RAY_NO_FREE, RAY_NO_FREE]),
        rays, grid_spec, colour, RAY_OK,
    )

    assert tuple(panel[2, 0]) == colour
    assert np.count_nonzero(panel) == 3


def test_the_font_is_a_real_truetype_at_the_requested_size():
    """기본 PIL 폰트는 약 11 px로 고정이라, 폭 1000 px가 넘는 패널에서 글자가 읽히지 않았다.
    그게 이 패널을 다시 만든 이유 중 하나이므로 크기가 실제로 반영되는지 고정한다."""
    small, large = load_font(12), load_font(30)

    assert large.getbbox("iou_free")[2] > small.getbbox("iou_free")[2] * 2
    assert large.getbbox("M")[3] >= 20


def test_sample_scores_come_from_the_training_metric_functions():
    """시각화가 자기만의 지표를 다시 구현하면 그림과 로그가 다른 말을 한다.

    4칸, valid=1:
        gt_free   = [T, T, F, F]
        pred_free = [T, F, T, F]
    손계산:
        iou_free  = 교집합 {0} / 합집합 {0,1,2} = 1/3
        fatal     = pred free & ~gt free = {2} / |pred free| {0,2} = 1/2
        free_miss = ~pred free & gt free = {1} / |gt free| {0,1} = 1/2
    같은 값을 원시 지표 함수로 직접 계산해 대조한다 -- 여기서 다른 정의를 쓰면 깨진다.
    """
    from projects.common.free_space_metrics import fatal_rate, free_miss_rate, iou_free

    pred = _grid_parts([True, False, True, False], [False, True, False, True], [False] * 4)
    gt = _grid_parts([True, True, False, False], [False, False, True, True], [False] * 4)
    valid = _on_real_grid([True] * 4)
    rays = build_ray_index(ROBOT_GRID_SPEC, n_theta=4)

    scores = sample_scores(pred, gt, valid, rays, ROBOT_GRID_SPEC.cell_m)

    assert scores["iou_free"] == pytest.approx(iou_free(pred["free"], gt["free"], valid)[0])
    assert scores["fatal_rate"] == pytest.approx(fatal_rate(pred["free"], gt["free"], valid)[0])
    assert scores["free_miss_rate"] == pytest.approx(
        free_miss_rate(pred["free"], gt["free"], valid)[0]
    )
    # 손계산과도 맞는지 -- 원시 함수까지 같이 틀린 경우를 잡는다.
    assert scores["iou_free"] == pytest.approx(1 / 3)
    assert scores["fatal_rate"] == pytest.approx(0.5)
    assert scores["free_miss_rate"] == pytest.approx(0.5)


def test_sort_keys_rank_worse_samples_first():
    """`--sort_by`는 `index`를 빼면 전부 "나쁜 순"이다. 부호가 뒤집히면 가장 좋은 샘플만
    보게 되고, 이 도구의 목적(평균 뒤에 숨은 실패를 먼저 만나기)이 사라진다."""
    good = {"iou_free": 0.9, "fatal_rate": 0.01, "free_miss_rate": 0.01,
            "f1_occupied": 0.9, "range_mae": 0.05}
    bad = {"iou_free": 0.2, "fatal_rate": 0.50, "free_miss_rate": 0.50,
           "f1_occupied": 0.2, "range_mae": 0.90}

    for name, key in _SORT_KEYS.items():
        if name == "index":
            continue
        assert key(bad) > key(good), name


def test_unmeasurable_samples_sink_to_the_bottom():
    """짝지어진 광선이 없어 `range_mae`가 NaN인 샘플은 순위를 왜곡하지 않아야 한다."""
    assert _nan_to_inf(float("nan")) == -np.inf
    assert _nan_to_inf(0.5) == 0.5
    assert _nan_to_inf(0.0) > _nan_to_inf(float("nan"))


COMMON_ROOT = Path("dataset/sj_datasets/common")
SEQUENCE_ROOT = Path("dataset/sj_datasets/raws1")
requires_dataset = pytest.mark.skipif(
    not (SEQUENCE_ROOT / "occupancy_npy").exists() or not COMMON_ROOT.exists(),
    reason="self-collected dataset not available locally",
)


@requires_dataset
def test_render_ipm_fills_the_camera_covered_area():
    from PIL import Image

    cameras = load_cameras(COMMON_ROOT / "calibration/calib.yaml")
    ego_T_cams = load_ego_T_cams(COMMON_ROOT / "calibration/calib.yaml")
    sample_id = sorted(p.stem for p in (SEQUENCE_ROOT / "occupancy_npy").glob("*.npy"))[0]
    images = {
        name: np.asarray(Image.open(
            SEQUENCE_ROOT / "rgb_images" / sample_id / f"cam_{name}.jpg").convert("RGB"))
        for name in FINETUNE_CAMERA_NAMES
    }

    ipm = render_ipm(images, cameras, ego_T_cams, ROBOT_GRID_SPEC)

    assert ipm.shape == (ROBOT_GRID_SPEC.n_rows, ROBOT_GRID_SPEC.n_cols, 3)
    filled = ipm.any(axis=2)
    # 화각 밖(원점 주변 원반)은 비고 나머지는 대부분 채워져야 한다.
    assert 0.85 < filled.mean() < 1.0
    # ego 원점은 그리드 중앙이 아니다 -- ROI가 전후 비대칭(전방 4 m / 후방 2 m)이라
    # x=0은 row (front_m / cell_m)에 온다.
    origin_row = int(ROBOT_GRID_SPEC.front_m / ROBOT_GRID_SPEC.cell_m)
    origin_col = int(ROBOT_GRID_SPEC.half_width_m / ROBOT_GRID_SPEC.cell_m)
    assert not filled[origin_row - 2:origin_row + 2, origin_col - 2:origin_col + 2].any()


@requires_dataset
def test_tail_split_is_contiguous_and_disjoint():
    train, val = split_samples_within_sequences([SEQUENCE_ROOT], tail_fraction=0.2)
    all_samples = sorted(p.stem for p in (SEQUENCE_ROOT / "occupancy_npy").glob("*.npy"))

    assert len(train) + len(val) == len(all_samples)
    assert not set(train) & set(val)
    # val은 시퀀스 끝의 연속 구간이어야 한다.
    assert [s for _, s in val] == all_samples[-len(val):]
    assert [s for _, s in train] == all_samples[:len(train)]


def test_tail_split_never_empties_training(tmp_path):
    (tmp_path / "occupancy_npy").mkdir(parents=True)
    for i in range(3):
        np.save(tmp_path / "occupancy_npy" / f"sample_{i:06d}.npy", np.zeros((2, 2), np.uint8))

    train, val = split_samples_within_sequences([tmp_path], tail_fraction=0.99)
    assert len(train) >= 1
    assert len(val) == len(train) + len(val) - len(train)

    with pytest.raises(ValueError):
        split_samples_within_sequences([tmp_path], tail_fraction=1.0)


def test_the_label_font_can_render_korean():
    """라벨이 한국어인데 DejaVu에는 한글 글리프가 없어 처음에는 전부 tofu(□□)로 나왔다.
    폰트 후보 순서가 바뀌면 그 상태로 조용히 돌아간다.

    "무언가 그려졌다"나 "라틴 문자열과 다르다"로는 부족하다 -- tofu도 사각형을 그리므로 둘 다
    통과한다(직접 확인함). **서로 다른 두 한글 글자를 그려 픽셀이 달라야** 실제 글리프가 있다는
    뜻이다. 글리프가 없으면 둘 다 같은 사각형이 되어 완전히 일치한다.
    """
    from PIL import Image, ImageDraw

    def rendered(font, text):
        image = Image.new("L", (120, 60), 0)
        ImageDraw.Draw(image).text((4, 4), text, fill=255, font=font)
        return np.asarray(image)

    for mono in (False, True):
        font = load_font(28, mono=mono)
        first, second = rendered(font, "가"), rendered(font, "밟")
        assert first.sum() > 0
        assert not np.array_equal(first, second), mono


def test_the_legend_wraps_instead_of_running_off_the_right_edge():
    """범례 항목이 늘어나면 한 줄로는 폭을 넘어 오른쪽이 조용히 잘린다."""
    from projects.common.three_class_panels import _legend_item_width, _wrap_legend

    font = load_font(19)
    legend = [(f"item {i} with a fairly long label", (0, 0, 0)) for i in range(8)]
    width = _legend_item_width(legend[0][0], font) * 3

    rows = _wrap_legend(legend, width, font)

    assert len(rows) > 1
    assert sum(len(row) for row in rows) == len(legend)
    for row in rows:
        assert sum(_legend_item_width(text, font) for text, _ in row) <= width or len(row) == 1


def _panel_with_legend(legend):
    from projects.common.three_class_panels import build_panel, upscale

    bev = upscale(np.zeros((20, 20, 3), np.uint8), 2)
    return build_panel(
        camera_images=[np.zeros((10, 10, 3), np.uint8)], camera_names=["front"],
        bev_grid=[[("a", bev), ("b", bev)]], headline="t",
        metric_lines=["m1", "m2"], legend=legend, cam_thumb_wh=(80, 45),
    )


def test_the_panel_grows_taller_when_the_legend_needs_more_rows():
    """`build_panel`이 실제로 줄바꿈 결과를 반영해 캔버스를 키우는지 본다.

    범례 항목이 늘어나도 높이가 그대로면 항목이 한 줄에 다 그려지고 오른쪽이 잘린다는 뜻이다
    -- `_wrap_legend`를 직접 호출하는 테스트만으로는 `build_panel`이 그 결과를 무시하도록
    바뀌어도 통과한다(직접 확인함).
    """
    entries = [(f"legend entry number {i}", (10 * i, 0, 0)) for i in range(12)]

    short = _panel_with_legend(entries[:2])
    long = _panel_with_legend(entries)

    assert long.height > short.height
    assert long.width == short.width, "범례가 캔버스를 옆으로 늘려서는 안 된다"


def test_the_wrapped_legend_is_not_cut_off_at_the_bottom():
    """줄바꿈된 범례가 캔버스 아래로 잘려 나가면 줄바꿈 자체가 무의미하다."""
    panel = _panel_with_legend([(f"legend entry number {i}", (10 * i, 0, 0)) for i in range(12)])
    bottom_strip = np.asarray(panel)[-4:]

    assert (bottom_strip == np.array([250, 250, 250], np.uint8)).all()
