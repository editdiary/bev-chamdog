"""SynWoodScape(source)와 로봇(target)의 **라벨 prior**를 같은 정의로 비교한다 (CPU).

`02_projection_and_prior` 축 B의 근거다. 학습이 필요 없고 라벨 `.npy`만 읽는다.

**왜 필요한가.** `docs/archive/research/paper_experiment_compendium.md` §4는 사전학습이 해로운 이유를
"SynWoodScape는 격자의 83 %가 free인데 로봇은 20 %"라고 적었다. 그런데 그 대조는 **두 가지
다른 양**을 비교한 것이다 -- 로봇 쪽 숫자가 낮은 이유는 free/장애물 균형이 아니라
**관측된 셀이 격자의 28 %뿐**이라는 데 있다. 관측된 셀만 놓고 보면 로봇이 오히려 더 free하다.

그래서 도메인 격차의 정체는 "free가 많은 세계"가 아니라 **"어디든 답을 아는 세계"**다.
사전학습 모델이 물려받는 것은 free 편향이 아니라 격자 전체에 대해 확신하는 습관이고,
그것이 로봇 데이터에서는 감독되지 않는 4분의 3에 쏟아진다.

**라벨 규약.** 두 데이터셋 모두 `occ=1`이 free(drivable)다 -- 이름과 반대이므로 주의한다
(`projects/common/free_space.decompose`). `free = occ & vis`로 통일해 센다.

실행:
    python tools/measure_domain_prior.py
    python tools/measure_domain_prior.py --stride=5     # 표본만 빠르게
"""
import sys
from pathlib import Path

import numpy as np
from fire import Fire

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from projects.bev_gt.grid import (  # noqa: E402
    ROBOT_GRID_SPEC,
    SYNWOODSCAPE_PRETRAIN_GRID_SPEC,
)

_SYNWOODSCAPE = _REPO_ROOT / "dataset/synwoodscape_2head_roi_8_4_6_h08"
_ROBOT = _REPO_ROOT / "dataset/sj_datasets"


def _tally(occ_paths, vis_of):
    free = visible = cells = 0
    for path in occ_paths:
        occ = np.load(path).astype(bool)
        vis = np.load(vis_of(path)).astype(bool)
        free += int((occ & vis).sum())
        visible += int(vis.sum())
        cells += int(occ.size)
    return free, visible, cells


def _report(label, spec, occ_paths, vis_of):
    free, visible, cells = _tally(occ_paths, vis_of)
    print(f"  {label:<14} {len(occ_paths):>4}샘플 | {spec.n_rows}x{spec.n_cols}, "
          f"전 {spec.front_m:g} / 후 {spec.rear_m:g} / 횡 ±{spec.half_width_m:g} m, "
          f"{spec.cell_m*100:.0f} cm")
    print(f"      free / 격자 전체    {100*free/cells:>5.1f}%")
    print(f"      free / 관측된 셀    {100*free/visible:>5.1f}%")
    print(f"      관측된 셀 / 격자    {100*visible/cells:>5.1f}%")
    return free / cells, free / visible, visible / cells


def main(stride: int = 1):
    sw = sorted(_SYNWOODSCAPE.glob("*_occupancy.npy"))[::stride]
    rb = sorted(_ROBOT.glob("raw*/occupancy_npy/*.npy"))[::stride]
    if not sw or not rb:
        raise SystemExit(f"라벨을 못 찾았다: synwoodscape {len(sw)}개, robot {len(rb)}개")

    print("=== 두 도메인의 라벨 prior (free = occ & vis, occ=1이 free) ===")
    sw_stats = _report("SynWoodScape", SYNWOODSCAPE_PRETRAIN_GRID_SPEC, sw,
                       lambda p: p.with_name(p.name.replace("_occupancy.npy", "_visible.npy")))
    rb_stats = _report("로봇", ROBOT_GRID_SPEC, rb,
                       lambda p: p.parents[1] / "visibility_npy" / p.name)

    print("\n=== 읽는 법 ===")
    print(f"  격자 전체 기준으로는 {100*sw_stats[0]:.0f}% 대 {100*rb_stats[0]:.0f}%로 크게 다르다.")
    print(f"  그런데 **관측된 셀만** 보면 {100*sw_stats[1]:.1f}% 대 {100*rb_stats[1]:.1f}%로, "
          f"로봇이 오히려 더 free하다.")
    print(f"  차이는 전부 관측 비율에서 온다 -- {100*sw_stats[2]:.1f}% 대 {100*rb_stats[2]:.1f}%.")
    print("  즉 불일치의 정체는 'free가 많은 세계'가 아니라 **'어디든 답을 아는 세계'**다.")
    print("  합성 데이터는 시뮬레이터가 전지적이고, 로봇 라벨은 LiDAR가 훑은 곳에만 있다.")


if __name__ == "__main__":
    Fire(main)
