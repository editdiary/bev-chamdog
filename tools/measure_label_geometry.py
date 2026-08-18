"""라벨 기하 실측 -- censored 광선 비율과 polar bin 수의 영향 (스펙 §12).

두 숫자가 M3의 설계 파라미터를 결정한다:
- censored 비율이 높으면 M3의 유효 표본이 줄어 격자 확장을 검토해야 한다
- 360 bin과 720 bin의 지표 차이가 작으면 360을 유지한다 (스펙 §5.2, 720으로 결정된 이후로도
  M3 자체의 유효성 확인용으로 남겨 둔다)

**"raw" roundtrip IoU와 "mask-excluded" roundtrip IoU는 다른 것을 잰다** (스펙 §12).
`reconstruct_free`는 원점(반지름 0)부터 채우므로 원점 부근의 `permanent_blind ∪ invalid`를
복원 결과에 되살린다 -- `free` 자체는 이미 그 마스크들을 제외하고 있으므로(§7) 이는 항상
거짓양성이 된다. raw 값은 이 효과를 포함하고(즉 `reconstruct_free`를 그대로 쓰는 임의의
호출자가 실제로 보는 값), mask-excluded 값은 `permanent_blind ∪ invalid`를 양쪽에서 제외해
**polar 표현 자체의 손실**만 남긴 값이다(§2.5가 재는 것과 같은 정의).

실행: python tools/measure_label_geometry.py
"""
import sys
from pathlib import Path

import numpy as np
from fire import Fire

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from projects.common.polar import (  # noqa: E402
    RAY_CENSORED,
    RAY_NO_FREE,
    RAY_OK,
    build_ray_index,
    first_free_range,
    reconstruct_free,
)
from projects.common.free_space import decompose  # noqa: E402
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    GRID_SPEC,
    build_bev_masks,
    list_sequence_samples,
    load_masked_labels,
)


def main(sequences="raws1,raws2,raws3,rawos1", dataset_root=DEFAULT_DATASET_ROOT):
    permanent_blind, invalid = build_bev_masks(DEFAULT_COMMON_ROOT)
    mask_out = permanent_blind | invalid
    frees = []
    for name in str(sequences).split(","):
        root = Path(dataset_root) / name.strip()
        if not (root / "occupancy_npy").exists():
            continue
        for sequence_root, sample_id in list_sequence_samples(root):
            frees.append(decompose(*load_masked_labels(
                sequence_root, sample_id, permanent_blind, invalid
            ))["free"])
    print(f"frames = {len(frees)}")
    print(
        f"permanent_blind ∪ invalid area = {mask_out.sum()}/{mask_out.size} "
        f"= {mask_out.sum() / mask_out.size:.4f}"
    )

    for n_theta in (180, 360, 720):
        rays = build_ray_index(GRID_SPEC, n_theta=n_theta)
        status_counts = np.zeros(3, dtype=np.int64)
        roundtrip, roundtrip_excl = [], []
        fp_in_mask, fp_other = [], []
        for free in frees:
            r_m, status = first_free_range(free, rays)
            for code in (RAY_OK, RAY_NO_FREE, RAY_CENSORED):
                status_counts[code] += int((status == code).sum())
            restored = reconstruct_free(r_m, status, rays, free.shape)
            union = (restored | free).sum()
            roundtrip.append((restored & free).sum() / union if union else np.nan)

            # mask-excluded: permanent_blind/invalid를 origin-fill이 되살리는 효과를
            # 양쪽에서 제거하고 비교한다 (§12) -- §2.5와 같은 정의.
            restored_excl, free_excl = restored & ~mask_out, free & ~mask_out
            union_excl = (restored_excl | free_excl).sum()
            roundtrip_excl.append(
                (restored_excl & free_excl).sum() / union_excl if union_excl else np.nan
            )

            false_positive = restored & ~free
            fp_in_mask.append(int((false_positive & mask_out).sum()))
            fp_other.append(int((false_positive & ~mask_out).sum()))
        total = status_counts.sum()
        total_fp = sum(fp_in_mask) + sum(fp_other)
        print(
            f"n_theta={n_theta:4d}  ok {status_counts[RAY_OK] / total:.3f}"
            f"  no_free {status_counts[RAY_NO_FREE] / total:.3f}"
            f"  censored {status_counts[RAY_CENSORED] / total:.3f}"
            f"  |  raw roundtrip IoU {np.nanmean(roundtrip):.4f}"
            f"  |  mask-excluded roundtrip IoU {np.nanmean(roundtrip_excl):.4f}"
            f"  |  FP in mask {100 * sum(fp_in_mask) / total_fp:.1f}%"
        )


if __name__ == "__main__":
    Fire(main)
