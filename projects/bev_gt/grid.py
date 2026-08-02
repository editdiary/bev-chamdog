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


ROBOT_GRID_SPEC = OccupancyGridSpec(front_m=3.0, rear_m=1.0, half_width_m=1.0, cell_m=0.05)


def remap_semantic_to_occupancy(semantic_labels: np.ndarray, drivable_class_ids=DRIVABLE_CLASS_IDS) -> np.ndarray:
    """Map semantic class ids to binary occupancy (1=drivable, 0=non-drivable)."""
    semantic_labels = np.asarray(semantic_labels)
    drivable_ids = np.fromiter(drivable_class_ids, dtype=semantic_labels.dtype)
    return np.isin(semantic_labels, drivable_ids).astype(np.uint8)
