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
#
# 전방4m/후방2m/좌우±3m(=6m×6m, 120×120)로 결정됨 — 로봇의 진행 방향(전방) 관측이 더
# 중요하므로 SYNWOODSCAPE_PRETRAIN_GRID_SPEC과 같은 전방 편향 비대칭 형태를 따른다.
# (이전 값 front_m=3.0/rear_m=1.0/half_width_m=1.0=4m×2m는 2026-07-29 최초 설계 초안의
# placeholder였고, 이 6m×6m 결정 이후 갱신되지 않은 채 남아 있었다 — 코드에 실제로 반영한
# 적이 없었을 뿐, 로봇 물리적 크기와는 애초에 무관하다.)
ROBOT_GRID_SPEC = OccupancyGridSpec(front_m=4.0, rear_m=2.0, half_width_m=3.0, cell_m=0.05)

# SynWoodScape pretraining 전용 그리드.
#
# 왜 별도로 두는가: SynWoodScape의 ego는 **풀사이즈 승용차**(3D 박스 x∈[-1.85,1.85],
# y∈[-0.89,0.90])다. ROBOT_GRID_SPEC(전방3m/후방1m/좌우±1m = 4m×2m)에 이 차체를 얹으면
# 그리드 셀의 56%가 ego 차체 자신에 덮여 항상 non-drivable로 고정되고, 실측상 샘플 간에
# 값이 바뀌는 셀이 ~1.5%뿐이다 → 학습 신호가 거의 없는 상수 GT가 된다.
#
# 전방5m/후방3m/좌우±4m(=8m×8m)로 잡은 이유: 자체 로봇의 최종 fine-tuning 타깃이
# 0.05m/cell × 120×120(=6m×6m)으로 정해졌고, pretrain 거리 스케일을 그 목표와 같은
# 자릿수(근~중거리)로 맞추는 편이 원거리 위주 자율주행 분포와의 괴리를 줄인다.
# 다만 6m×6m에 그대로 맞추면 ego 차체 차지 비율이 다시 ~18%까지 올라가 GT 변별력이
# 줄어드므로, ego 비율을 ~10% 선으로 낮추면서도 로봇 목표와 가까운 거리대를 유지하는
# 절충점으로 8m×8m을 택했다. cell 수는 160×160(0.05m/cell 기준 8의 배수).
SYNWOODSCAPE_PRETRAIN_GRID_SPEC = OccupancyGridSpec(
    front_m=5.0, rear_m=3.0, half_width_m=4.0, cell_m=0.05
)


def remap_semantic_to_occupancy(semantic_labels: np.ndarray, drivable_class_ids=DRIVABLE_CLASS_IDS) -> np.ndarray:
    """Map semantic class ids to binary occupancy (1=drivable, 0=non-drivable)."""
    semantic_labels = np.asarray(semantic_labels)
    return np.isin(semantic_labels, list(drivable_class_ids)).astype(np.uint8)


def cell_centers_m(grid_spec: OccupancyGridSpec):
    """Row/col index -> ego-frame (forward_m, lateral_m) cell-center coordinates.

    Shared by `bev_crop.crop_bev_occupancy` and `visibility.compute_visible_mask` so the
    row/col <-> meters mapping lives in one place (previously duplicated and let the two
    drift apart). Orientation matches `crop_bev_occupancy`: row 0 = front-most, col 0 =
    vehicle's left.
    """
    forward_m = grid_spec.front_m - (np.arange(grid_spec.n_rows) + 0.5) * grid_spec.cell_m
    lateral_m = grid_spec.half_width_m - (np.arange(grid_spec.n_cols) + 0.5) * grid_spec.cell_m
    return forward_m, lateral_m
