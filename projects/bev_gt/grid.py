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

# SynWoodScape pretraining 전용 그리드. 전방8m/후방4m/좌우±6m(=12m×12m, 240×240).
#
# 왜 별도로 두는가: SynWoodScape의 ego는 **풀사이즈 승용차**(3D 박스 x∈[-1.85,1.85],
# y∈[-0.89,0.90])다. ROBOT_GRID_SPEC 정도의 좁은 그리드에 이 차체를 얹으면 그리드 셀의
# 절반 이상이 ego 차체 자신에 덮여 항상 non-drivable로 고정되고, 샘플 간에 값이 바뀌는
# 셀이 ~1.5%뿐이다 → 학습 신호가 거의 없는 상수 GT가 된다. 그래서 로봇보다 넓게 잡는다.
#
# 이 값은 수동 검수를 마친 라벨(`dataset/synwoodscape_2head_roi_8_4_6_h08`, ROI 8/4/±6,
# visibility H=0.8)의 ROI와 일치해야 한다 -- 라벨이 그 ROI로 잘려서 저장돼 있으므로
# 여기를 바꾸면 라벨을 다시 만들어야 한다.
#
# (이력) 2026-07-29 설계 초안의 전방5m/후방3m/좌우±4m(=8m×8m, 160×160)이 한동안 같은
# 이름으로 남아 있었으나, 실제 라벨링과 학습은 위 8/4/±6으로 진행됐다. 이름은 같고 값만
# 다른 상수가 둘 있어 혼동을 일으켜 하나로 합쳤다. 옛 8m×8m 스펙의 근거는
# `docs/dataset_analysis/synwoodscape_geometry_findings.md`에 기록으로 남아 있다.
SYNWOODSCAPE_PRETRAIN_GRID_SPEC = OccupancyGridSpec(
    front_m=8.0, rear_m=4.0, half_width_m=6.0, cell_m=0.05
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
