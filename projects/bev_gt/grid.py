from dataclasses import dataclass

import numpy as np

DRIVABLE_CLASS_IDS = frozenset({6, 7})  # road line(6), road(7) per SynWoodScape semantic palette


@dataclass(frozen=True)
class OccupancyGridSpec:
    front_m: float
    rear_m: float
    half_width_m: float
    cell_m: float

    @property
    def n_rows(self) -> int:
        return round((self.front_m + self.rear_m) / self.cell_m)

    @property
    def n_cols(self) -> int:
        return round((2 * self.half_width_m) / self.cell_m)


# 자체 로봇(소형 실내/보도 주행 플랫폼) 기준 그리드. 최종 fine-tuning 타깃이다.
ROBOT_GRID_SPEC = OccupancyGridSpec(front_m=3.0, rear_m=1.0, half_width_m=1.0, cell_m=0.05)

# SynWoodScape pretraining 전용 그리드.
#
# 왜 별도로 두는가: SynWoodScape의 ego는 **풀사이즈 승용차**(3D 박스 x∈[-1.85,1.85],
# y∈[-0.89,0.90])다. ROBOT_GRID_SPEC(전방3m/후방1m/좌우±1m = 4m×2m)에 이 차체를 얹으면
# 그리드 셀의 56%가 ego 차체 자신에 덮여 항상 non-drivable로 고정되고, 실측상 샘플 간에
# 값이 바뀌는 셀이 ~1.5%뿐이다 → 학습 신호가 거의 없는 상수 GT가 된다.
# 전방7m/후방3m/좌우±5m로 넓히면 샘플 간 변동 셀이 92% 이상으로 올라간다.
# ROBOT_GRID_SPEC은 그대로 남겨 둔다 — 실제 로봇 데이터 fine-tuning 단계의 타깃이다.
SYNWOODSCAPE_PRETRAIN_GRID_SPEC = OccupancyGridSpec(
    front_m=7.0, rear_m=3.0, half_width_m=5.0, cell_m=0.05
)


def remap_semantic_to_occupancy(semantic_labels: np.ndarray, drivable_class_ids=DRIVABLE_CLASS_IDS) -> np.ndarray:
    """Map semantic class ids to binary occupancy (1=drivable, 0=non-drivable)."""
    semantic_labels = np.asarray(semantic_labels)
    return np.isin(semantic_labels, list(drivable_class_ids)).astype(np.uint8)
