"""라벨이 조용히 바뀌는 것을 잡는 gated 테스트.

여기 박힌 숫자는 2026-08-17에 154 프레임 전수로 실측한 값이다(스펙 §2.1). 라벨 파이프라인이
바뀌면 여기가 먼저 터져야 한다 -- 학습 지표가 이상해진 뒤에 원인을 찾는 것보다 훨씬 싸다.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from projects.datasets.robot_simplebev import (
    DEFAULT_COMMON_ROOT,
    build_bev_masks,
    list_sequence_samples,
    load_masked_labels,
)

DATASET_ROOT = Path("dataset/sj_datasets")
SEQUENCES = ("raws1", "raws2", "raws3", "rawos1")
requires_dataset = pytest.mark.skipif(
    not (DATASET_ROOT / "raws1" / "occupancy_npy").exists(),
    reason="self-collected dataset not available locally",
)


def _labels():
    permanent_blind, invalid = build_bev_masks(DEFAULT_COMMON_ROOT)
    for name in SEQUENCES:
        root = DATASET_ROOT / name
        if not (root / "occupancy_npy").exists():
            continue
        for sequence_root, sample_id in list_sequence_samples(root):
            yield name, load_masked_labels(sequence_root, sample_id, permanent_blind, invalid)


@requires_dataset
def test_occupied_is_a_one_cell_frontier_shell():
    """`occupied`가 표면이라는 것이 M3(거리로 잰다)의 전제다. 실측 0.998."""
    neighbourhood = np.ones((3, 3), bool)
    touching = total = 0
    for _, (occ, vis, valid) in _labels():
        occupied = (~occ) & vis & valid
        if not occupied.any():
            continue
        adjacent_to_unknown = ndimage.binary_dilation(~vis, neighbourhood)
        touching += int((occupied & adjacent_to_unknown).sum())
        total += int(occupied.sum())

    assert total > 0
    assert touching / total >= 0.99


@requires_dataset
def test_free_and_occupied_fractions_stay_in_the_measured_range():
    """실측(스펙 §2.1): free는 격자의 0.15~0.28, occupied는 평가 마스크의 0.04~0.08.

    154 프레임 전체 평균만 보면 시퀀스 하나가 통째로 어긋나도 나머지 세 개에 묻혀 통과할 수
    있다(2026-08-17 실측: 시퀀스별 평균이 이미 free 0.151(rawos1)~0.258(raws3),
    occupied 0.048(raws3)~0.074(rawos1)로 넓게 퍼져 있어, 시퀀스 하나가 그 범위 밖으로
    빠져도 전체 평균은 여전히 0.15~0.28 안에 남을 수 있다 -- 직접 확인함).
    그래서 전체 평균과는 별개로 시퀀스별 평균도 체크한다.

    시퀀스별 밴드는 절대 구간(예: 이전 버전의 free ∈ [0.10, 0.32])이 아니라 **그 시퀀스
    자신의 실측 평균 대비 상대 구간(±25 %)**으로 잡는다. 절대 구간은 시퀀스별로 헤드룸이
    달라 raws3(free 0.2576)처럼 평균이 이미 밴드 중앙 근처에 있는 시퀀스는 48 % 넘게
    무너져도(0.2576 → 0.1326) 여전히 통과하는 반면 rawos1(free 0.1508)은 밴드 경계에
    바짝 붙어 있어 훨씬 작은 변화에도 걸리는 식으로 시퀀스마다 민감도가 들쭉날쭉했다(직접
    주입해 확인함 -- 리포트 참고). 상대 구간을 쓰면 **모든 시퀀스가 자기 평균 대비 25 %
    이상 벗어나면 예외 없이 걸린다** -- 민감도가 시퀀스마다 균일해진다.
    """
    free_fractions, occupied_fractions = [], []
    per_sequence_free, per_sequence_occupied = {}, {}
    for name, (occ, vis, valid) in _labels():
        mask = vis & valid
        free = (occ & mask).sum() / valid.sum()
        occupied = ((~occ) & mask).sum() / max(mask.sum(), 1)
        free_fractions.append(free)
        occupied_fractions.append(occupied)
        per_sequence_free.setdefault(name, []).append(free)
        per_sequence_occupied.setdefault(name, []).append(occupied)

    assert 0.15 <= float(np.mean(free_fractions)) <= 0.28
    assert 0.04 <= float(np.mean(occupied_fractions)) <= 0.08

    # 2026-08-17 실측 시퀀스별 평균 (fix round 1). 시퀀스가 늘어나 이 값이 자연스럽게
    # 조금씩 옮겨갈 수는 있으나, 라벨 파이프라인이 바뀌어 어느 한 시퀀스가 자기 평균 대비
    # 25 %를 넘게 움직이면 이 테스트가 먼저 터져야 한다.
    measured_free = {"raws1": 0.1647, "raws2": 0.1757, "raws3": 0.2576, "rawos1": 0.1508}
    measured_occupied = {"raws1": 0.0648, "raws2": 0.0647, "raws3": 0.0484, "rawos1": 0.0739}
    relative_margin = 0.25
    for name in per_sequence_free:
        seq_free = float(np.mean(per_sequence_free[name]))
        seq_occupied = float(np.mean(per_sequence_occupied[name]))
        free_center, occupied_center = measured_free[name], measured_occupied[name]
        assert (1 - relative_margin) * free_center <= seq_free <= (1 + relative_margin) * free_center, (
            name, seq_free, free_center,
        )
        assert (
            (1 - relative_margin) * occupied_center
            <= seq_occupied
            <= (1 + relative_margin) * occupied_center
        ), (name, seq_occupied, occupied_center)


@requires_dataset
def test_all_four_sequences_are_present_and_frame_count_is_154():
    """`_labels()`는 `occupancy_npy`가 없는 시퀀스를 조용히 `continue`로 건너뛴다.

    시퀀스 4개 중 아무거나 하나가 통째로 빠져도(예: 마운트 실패, 경로 오타) 위 두 테스트는
    나머지 시퀀스만으로 여전히 자기 밴드 안에 들어 통과할 수 있다 -- 시퀀스가 빠진 것 자체는
    "라벨이 조용히 바뀌는 것"의 가장 노골적인 형태인데, 그걸 보는 테스트가 없었다. 발견된
    시퀀스 집합이 정확히 `SEQUENCES`와 같은지, 그리고 총 프레임 수가 실측값(154)과 같은지
    직접 확인한다.
    """
    discovered = {
        name for name in SEQUENCES if (DATASET_ROOT / name / "occupancy_npy").exists()
    }
    assert discovered == set(SEQUENCES), discovered

    total_frames = sum(1 for _ in _labels())
    assert total_frames == 154


@requires_dataset
def test_static_masks_have_the_measured_area_and_placement():
    """`permanent_blind`/`invalid`는 `build_bev_masks`가 **런타임에** 계산한다(스펙 §7) --
    캘리브레이션·카메라 대수가 바뀌면 라벨 파일은 그대로인데 이 두 마스크만 조용히
    바뀔 수 있다. 기존 두 테스트는 이 마스크가 통째로 빠지거나 뒤집혀도 잡지 못한다:
    `permanent_blind`가 `vis`를 줄이므로 그게 사라지면 `~vis`로 보는 frontier 비율은
    거의 그대로고(0.9978 → 0.9978, 직접 확인함), free 비율은 0.186 → 0.227로 위
    상대 밴드 안에 남는다.

    넓이만이 아니라 **위치**도 pin한다 -- `permanent_blind`는 좌우가 거의(99.12 %)
    대칭이라 넓이 검사만으로는 좌우 미러를 잡을 수 없다(직접 측정함). transpose와
    좌우 미러 둘 다 걸리는 구체적인 셀을 하나씩 못박는다.
    """
    permanent_blind, invalid = build_bev_masks(DEFAULT_COMMON_ROOT)
    grid_cells = permanent_blind.size

    # 넓이 (2026-08-17 실측, 스펙 §2.1의 5.6 %/2.7 %와 일치).
    pb_fraction = permanent_blind.sum() / grid_cells
    invalid_fraction = invalid.sum() / grid_cells
    assert 0.045 <= pb_fraction <= 0.07, pb_fraction
    assert 0.02 <= invalid_fraction <= 0.035, invalid_fraction

    # 위치 (transpose/미러 회귀를 잡기 위한 구체적 셀, 2026-08-17 실측).
    # ego 원점(약 0.5 m 반경 원반)은 항상 blind여야 한다.
    assert permanent_blind[80, 60]

    # transpose pin: (67, 60)은 blind인데 그 transpose 좌표 (60, 67)은 blind가 아니다.
    # 배열을 통째로 transpose하면 이 둘의 값이 뒤바뀌어 두 assert 모두 깨진다.
    assert permanent_blind[67, 60]
    assert not permanent_blind[60, 67]

    # 좌우 미러 pin: (67, 72)는 blind인데 그 좌우 대칭 좌표 (67, 47)은 blind가 아니다
    # (전체가 99.12 % 대칭이라 우연히 대칭인 셀을 고르면 미러를 통과시켜 버리므로, 실제로
    # 비대칭인 셀을 실측으로 골랐다). col을 좌우로 뒤집으면 이 둘의 값이 뒤바뀐다.
    assert permanent_blind[67, 72]
    assert not permanent_blind[67, 47]
