"""SynWoodScape 500 samples의 train/val 분할.

500장은 하나의 연속 주행 시퀀스가 **아니다** — `vehicle_data/rgb_images/*.txt`의
`Transform(Location(...))`을 인덱스 순서대로 훑어보면 인접 인덱스 간 거리가 중앙값 12.7m,
최대 258m로 CARLA 맵 전역에 흩어져 있고(각 파일의 `frame = 1`이 항상 같은 값인 것도 매
인덱스가 독립된 짧은 캡처라는 정황과 일치), 3m 이내로 붙어있는 "묶음"은 길이 1~5의 작은
버스트 64개뿐이다. 즉 실제 인접성/누수 위험은 인덱스 전체가 아니라 이 작은 버스트
내부에서만 존재한다.

따라서 여기서는 "앞/뒤 큰 구간을 통째로 나누는" 방식 대신, **위치 기반으로 버스트를
찾아 그 버스트를 통째로 train/val 중 한 쪽에만 배정**한다 — 버스트 하나가 쪼개지면 거의
같은 장면이 train과 val에 양쪽에 들어가는 누수가 생기기 때문이다. 버스트 사이는 서로
멀리 떨어져 있으므로 버스트 단위로는 무작위 배정이어도 누수가 없다.
"""
import random
import re
from pathlib import Path

_TRANSFORM_RE = re.compile(
    r"Transform\(Location\(x=([-\d.]+), y=([-\d.]+), z=([-\d.]+)\)"
)


def discover_all_sample_ids(dataset_root: Path) -> list:
    dataset_root = Path(dataset_root)
    return sorted(p.stem[: -len("_BEV")] for p in dataset_root.glob("rgb_images/*_BEV.png"))


def _ego_xy(dataset_root: Path, sample_id: str) -> tuple:
    text = (Path(dataset_root) / "vehicle_data/rgb_images" / f"{sample_id}.txt").read_text()
    x, y, _z = _TRANSFORM_RE.search(text).groups()
    return float(x), float(y)


def cluster_by_proximity(sample_ids: list, dataset_root: Path, threshold_m: float = 3.0) -> list:
    """정렬된 `sample_ids`를 훑으며, 이전 샘플과의 ego xy 거리가 `threshold_m` 미만인
    연속 구간을 하나의 버스트로 묶는다. 반환값: 버스트(각각 sample id 리스트)의 리스트.
    """
    sample_ids = sorted(sample_ids)
    positions = [_ego_xy(dataset_root, sid) for sid in sample_ids]

    clusters = []
    current = [sample_ids[0]]
    for i in range(1, len(sample_ids)):
        (x0, y0), (x1, y1) = positions[i - 1], positions[i]
        distance = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if distance < threshold_m:
            current.append(sample_ids[i])
        else:
            clusters.append(current)
            current = [sample_ids[i]]
    clusters.append(current)
    return clusters


def train_val_split(
    sample_ids: list,
    dataset_root: Path,
    val_fraction: float = 0.1,
    threshold_m: float = 3.0,
    seed: int = 0,
) -> tuple:
    """버스트 단위로 섞어 val_fraction 비율만큼 val에 배정한다. 반환: (train_ids, val_ids)."""
    clusters = cluster_by_proximity(sample_ids, dataset_root, threshold_m=threshold_m)
    rng = random.Random(seed)
    shuffled = clusters.copy()
    rng.shuffle(shuffled)

    target = round(val_fraction * len(sample_ids))
    val_ids, train_ids = [], []
    for cluster in shuffled:
        if len(val_ids) < target:
            val_ids.extend(cluster)
        else:
            train_ids.extend(cluster)
    return sorted(train_ids), sorted(val_ids)
