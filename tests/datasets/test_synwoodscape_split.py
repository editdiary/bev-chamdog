from pathlib import Path

import pytest

from projects.datasets.synwoodscape_split import (
    cluster_by_proximity,
    discover_all_sample_ids,
    train_val_split,
)

DATASET_ROOT = Path("dataset/synwoodscape/SynWoodScape_V0.1.0")
requires_dataset = pytest.mark.skipif(
    not (DATASET_ROOT / "vehicle_data").exists(), reason="SynWoodScape dataset not available locally"
)


@requires_dataset
def test_discover_all_sample_ids_finds_500_samples():
    sample_ids = discover_all_sample_ids(DATASET_ROOT)
    assert len(sample_ids) == 500
    assert sample_ids == sorted(sample_ids)


@requires_dataset
def test_clusters_partition_all_samples_without_overlap():
    sample_ids = discover_all_sample_ids(DATASET_ROOT)
    clusters = cluster_by_proximity(sample_ids, DATASET_ROOT)

    flattened = [sid for cluster in clusters for sid in cluster]
    assert sorted(flattened) == sample_ids
    assert len(set(flattened)) == len(flattened)


@requires_dataset
def test_clusters_are_mostly_small_bursts_not_one_long_sequence():
    """500장이 연속 시퀀스가 아니라 흩어진 짧은 버스트들이라는 실측을 회귀로 고정한다."""
    sample_ids = discover_all_sample_ids(DATASET_ROOT)
    clusters = cluster_by_proximity(sample_ids, DATASET_ROOT)

    assert len(clusters) > 50  # 하나의 긴 시퀀스라면 클러스터가 거의 1개여야 함
    assert max(len(c) for c in clusters) <= 10


@requires_dataset
def test_train_val_split_has_no_overlap_and_covers_all_samples():
    sample_ids = discover_all_sample_ids(DATASET_ROOT)
    train_ids, val_ids = train_val_split(sample_ids, DATASET_ROOT, val_fraction=0.1, seed=0)

    assert set(train_ids).isdisjoint(val_ids)
    assert sorted(train_ids + val_ids) == sample_ids
    assert 0.05 < len(val_ids) / len(sample_ids) < 0.2


@requires_dataset
def test_train_val_split_never_splits_a_cluster_across_both_sides():
    sample_ids = discover_all_sample_ids(DATASET_ROOT)
    clusters = cluster_by_proximity(sample_ids, DATASET_ROOT)
    train_ids, val_ids = train_val_split(sample_ids, DATASET_ROOT, val_fraction=0.1, seed=0)
    train_set, val_set = set(train_ids), set(val_ids)

    for cluster in clusters:
        in_train = any(sid in train_set for sid in cluster)
        in_val = any(sid in val_set for sid in cluster)
        assert not (in_train and in_val), f"cluster split across train/val: {cluster}"


@requires_dataset
def test_train_val_split_is_deterministic_given_seed():
    sample_ids = discover_all_sample_ids(DATASET_ROOT)
    a = train_val_split(sample_ids, DATASET_ROOT, seed=0)
    b = train_val_split(sample_ids, DATASET_ROOT, seed=0)
    assert a == b
