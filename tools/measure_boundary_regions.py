"""soft-boundary loss의 세 영역을 **라벨만으로** 실측한다.

무엇을 답하는가 (`docs/soft_boundary_loss_design.md` §6의 남은 항목):

1. `delta`별 `Omega_F` / `Omega_N` / `Omega_B`의 실제 셀 비율 -> §5.6의 셀당 가중치 배분
2. `Omega_F` 안에 라벨이 free가 **아닌** 셀이 몇 %인가 -> §4.1(라벨 목표 대 기하 목표)
3. 광선 상태 분포(`RAY_NO_FREE` 발생률) -> §4.2
4. **반경 정의와 유클리드 정의가 얼마나 다른가** -> §2.2의 주의가 실제로 문제인지

**이것은 하이퍼파라미터를 정하는 도구가 아니다.** `delta`와 `lambda_B`는 validation 스윕으로
정해야 한다(같은 문서 §7). 여기서 얻는 것은 "고른 값이 무엇을 뜻하는지"와 "정의가 의도대로
동작하는지"뿐이다.

실행:
    python tools/measure_boundary_regions.py
    python tools/measure_boundary_regions.py --sequences=raws1,rawos3
"""
import sys
from pathlib import Path

import numpy as np
from fire import Fire
from scipy.ndimage import distance_transform_edt

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from projects.common.free_space import decompose  # noqa: E402
from projects.common.polar import (  # noqa: E402
    RAY_CENSORED,
    RAY_NO_FREE,
    RAY_OK,
    build_cell_ray_map,
    build_ray_index,
    first_free_range,
    signed_boundary_distance,
)
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    GRID_SPEC,
    build_bev_masks,
    list_sequence_samples,
    load_masked_labels,
    parse_sequence_names,
)
from projects.geometry.double_sphere import FINETUNE_CAMERA_NAMES  # noqa: E402

DEFAULT_DELTAS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
_STATUS_NAMES = {RAY_OK: "RAY_OK", RAY_NO_FREE: "RAY_NO_FREE", RAY_CENSORED: "RAY_CENSORED"}


def euclidean_signed_distance(free, keep, cell_m) -> np.ndarray:
    """비교용 부호 있는 **유클리드** 거리 [m]. 진단 §26이 쓴 정의와 같다.

    `keep`이 False인 셀(`permanent_blind | invalid`)을 free에 합친다 -- ego 아래 원반은
    프레임마다 같은 정적 경계라 빼지 않으면 "경계 근방" 밴드가 그것으로 오염된다.
    """
    free_ext = free | ~keep
    return np.where(free_ext,
                    distance_transform_edt(free_ext, sampling=cell_m),
                    -distance_transform_edt(~free_ext, sampling=cell_m))


def region_counts(d, free, keep, deltas) -> dict:
    """`delta`별 세 영역의 셀 수와 각 영역의 라벨 오염도.

    `Omega_F`에서 세는 것은 "`d > delta`인데 라벨이 free가 아닌" 셀이고, `Omega_N`은 그
    반대다. 이 두 숫자가 §4.1의 결정(라벨 목표 대 기하 목표)을 가른다 -- 0에 가까우면 두
    선택이 사실상 같고, 크면 기하 목표가 라벨을 덮어쓴다는 뜻이다.
    """
    out = {}
    for delta in deltas:
        in_f, in_n = (d > delta) & keep, (d < -delta) & keep
        in_b = keep & ~in_f & ~in_n
        out[delta] = {
            "n_F": int(in_f.sum()), "n_N": int(in_n.sum()), "n_B": int(in_b.sum()),
            "F_not_free": int((in_f & ~free).sum()),
            "N_is_free": int((in_n & free).sum()),
            "B_is_free": int((in_b & free).sum()),
        }
    return out


# `P(label=free | d)` 곡선용 세밀 빈. soft target이 겨냥하는 곡선의 **실측 대응물**이다.
_CURVE_EDGES = np.arange(-0.30, 0.3001, 0.025)


def accumulate(sequences, dataset_root, common_root, deltas, n_theta) -> dict:
    dataset_root = Path(dataset_root)
    permanent_blind, invalid = build_bev_masks(common_root, GRID_SPEC, FINETUNE_CAMERA_NAMES)
    keep = ~(permanent_blind | invalid)
    rays = build_ray_index(GRID_SPEC) if n_theta is None else build_ray_index(GRID_SPEC, n_theta)
    cell_rays = build_cell_ray_map(GRID_SPEC, rays.rows.shape[0])

    totals = {d: {k: 0 for k in ("n_F", "n_N", "n_B", "F_not_free", "N_is_free", "B_is_free")}
              for d in deltas}
    eucl = {d: {k: 0 for k in ("n_F", "n_N", "n_B", "F_not_free", "N_is_free", "B_is_free")}
            for d in deltas}
    # 반경 정의와 유클리드 정의의 대역 소속 일치도 (delta는 대표값 하나로만 본다)
    agree = {"both": 0, "radial_only": 0, "eucl_only": 0, "neither": 0}
    status_counts = {name: 0 for name in _STATUS_NAMES.values()}
    n_bins = len(_CURVE_EDGES) - 1
    curve = {"n_rad": np.zeros(n_bins), "free_rad": np.zeros(n_bins),
             "n_euc": np.zeros(n_bins), "free_euc": np.zeros(n_bins)}
    n_frames = n_cells = 0

    for name in parse_sequence_names(sequences):
        for sequence_root, sample_id in list_sequence_samples(dataset_root / name):
            free = decompose(*load_masked_labels(
                sequence_root, sample_id, permanent_blind, invalid))["free"]
            d_rad = signed_boundary_distance(free, rays, cell_rays)
            d_euc = euclidean_signed_distance(free, keep, GRID_SPEC.cell_m)

            for delta, counts in region_counts(d_rad, free, keep, deltas).items():
                for key, value in counts.items():
                    totals[delta][key] += value
            for delta, counts in region_counts(d_euc, free, keep, deltas).items():
                for key, value in counts.items():
                    eucl[delta][key] += value

            b_rad = keep & (np.abs(d_rad) <= deltas[len(deltas) // 2])
            b_euc = keep & (np.abs(d_euc) <= deltas[len(deltas) // 2])
            agree["both"] += int((b_rad & b_euc).sum())
            agree["radial_only"] += int((b_rad & ~b_euc).sum())
            agree["eucl_only"] += int((b_euc & ~b_rad).sum())
            agree["neither"] += int((keep & ~b_rad & ~b_euc).sum())

            for tag, field in (("rad", d_rad), ("euc", d_euc)):
                idx = np.digitize(field, _CURVE_EDGES) - 1
                inside = keep & (idx >= 0) & (idx < n_bins)
                np.add.at(curve[f"n_{tag}"], idx[inside], 1.0)
                np.add.at(curve[f"free_{tag}"], idx[inside], free[inside].astype(float))

            _, status = first_free_range(free, rays)
            for code, sname in _STATUS_NAMES.items():
                status_counts[sname] += int((status == code).sum())
            n_frames += 1
            n_cells += int(keep.sum())

    return {"totals": totals, "eucl": eucl, "agree": agree, "status": status_counts,
            "curve": curve, "n_frames": n_frames, "n_cells": n_cells,
            "agree_delta": deltas[len(deltas) // 2]}


def format_report(result, deltas) -> str:
    n = result["n_cells"]
    lines = [f"프레임 {result['n_frames']}개 | 제외 후 셀 {n/1e6:.3f}M "
             f"(`permanent_blind | invalid` 제외)", ""]

    for tag, label in (("totals", "반경 (radial)"), ("eucl", "유클리드 (perpendicular)")):
        lines += [f"=== delta별 영역 비율 -- {label} ===",
                  f"{'delta[m]':>9s} {'Omega_F':>9s} {'Omega_N':>9s} {'Omega_B':>9s} "
                  f"{'| F의 non-free':>15s} {'N의 free':>10s} {'B의 free':>10s} "
                  f"{'| B 셀당 계수':>14s} {'B:N 배율':>10s}"]
        for delta in deltas:
            t = result[tag][delta]
            # 셀당 계수는 `lambda_B / |Omega_B|`를 `|V|` 단위로 읽은 것 (설계 문서 §5.6).
            coeff_b = 0.5 * n / max(t["n_B"], 1)
            coeff_n = 0.5 * n / max(t["n_N"], 1)
            lines.append(
                f"{delta:9.2f} {100*t['n_F']/n:8.2f}% {100*t['n_N']/n:8.2f}% "
                f"{100*t['n_B']/n:8.2f}% "
                f"{100*t['F_not_free']/max(t['n_F'],1):14.2f}% "
                f"{100*t['N_is_free']/max(t['n_N'],1):9.2f}% "
                f"{100*t['B_is_free']/max(t['n_B'],1):9.2f}% "
                f"{coeff_b:13.2f} {coeff_b/coeff_n:9.1f}x"
            )
        lines.append("")

    a, ad = result["agree"], result["agree_delta"]
    both, ro, eo = a["both"], a["radial_only"], a["eucl_only"]
    union = both + ro + eo
    lines += ["", f"=== 대역 소속 일치도 (delta={ad} m) ===",
              f"둘 다 대역: {100*both/n:.2f}%  | 반경만: {100*ro/n:.2f}%  "
              f"| 유클리드만: {100*eo/n:.2f}%",
              f"Jaccard = {both/max(union,1):.3f}  "
              f"(1.0이면 두 정의가 같은 셀을 고른다)"]

    total_rays = sum(result["status"].values())
    lines += ["", "=== 광선 상태 분포 ==="]
    for name, count in result["status"].items():
        lines.append(f"{name:>14s}: {100*count/total_rays:6.2f}%")

    c = result["curve"]
    lines += ["", "=== P(label=free | d) -- soft target이 겨냥하는 곡선의 실측 대응물 ===",
              "계단이 날카로우면 라벨 모호성이 아니라 이산화 잡음이다.",
              f"{'d 구간 [m]':>16s} {'반경: 셀 수':>12s} {'P(free)':>9s} "
              f"{'| 유클리드: 셀 수':>18s} {'P(free)':>9s}"]
    for i in range(len(_CURVE_EDGES) - 1):
        lo, hi = _CURVE_EDGES[i], _CURVE_EDGES[i + 1]
        nr, ne = c["n_rad"][i], c["n_euc"][i]
        pr = c["free_rad"][i] / nr if nr else float("nan")
        pe = c["free_euc"][i] / ne if ne else float("nan")
        lines.append(f"[{lo:+.3f},{hi:+.3f}) {nr:12.0f} {pr:9.3f} {ne:18.0f} {pe:9.3f}")
    return "\n".join(lines)


def main(sequences="raws2,raws3,rawos1,rawos2,rawos4,raws1,rawos3",
         dataset_root=DEFAULT_DATASET_ROOT, common_root=DEFAULT_COMMON_ROOT,
         deltas=DEFAULT_DELTAS, n_theta=None):
    deltas = tuple(float(d) for d in (deltas if isinstance(deltas, (tuple, list)) else [deltas]))
    result = accumulate(sequences, dataset_root, common_root, deltas, n_theta)
    print(format_report(result, deltas))


if __name__ == "__main__":
    Fire(main)
