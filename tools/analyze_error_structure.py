"""오차가 **어디에 몰려 있나**를 잰다 -- 프레임별 분포와 방위각별 프로파일.

요약 지표(`iou_free` 0.80)만으로는 두 가지 전혀 다른 세계를 구별할 수 없다.

1. **라벨 애매성 가설.** 대부분의 프레임은 잘 맞고, annotation이 애매했던 소수 프레임이
   점수를 끌어내린다. -> 프레임별 `iou_free`가 **꼬리가 긴 분포**가 되고, 나쁜 프레임은
   모델이 GT보다 free를 넓게 본 쪽(`fatal`)으로 치우친다.
2. **기하/캘리브레이션 결함 가설**(`docs/finetune_overfitting_diagnosis.md` §18).
   모든 프레임이 고르게 0.80이고 경계가 일정하게 몇 셀 밀려 있다. -> 프레임별 분포가
   **좁은 단봉형**이고, 방위각별 오차에 좌우 비대칭 같은 구조가 남는다.

두 가설은 라벨을 다시 만들지 않고도 이 두 분포로 갈린다. 그래서 이 도구가 필요하다.
`--out_csv`로 프레임별 원자료를 남기므로, 최악 프레임을 눈으로 확인하는 정성 검토
(`tools/visualize_robot_predictions.py`)로 바로 이어붙일 수 있다.

실행:
    CUDA_VISIBLE_DEVICES=0 python tools/analyze_error_structure.py \\
        --checkpoint=runs/robot_bev/ckpt/<run>/model_best-000000035.pth \\
        --formulation=binary --sequences=raws1,rawos3 \\
        --out_csv=runs/robot_bev/analysis/noos1_ep35_val_frames.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.common.binary_metrics import predicted_parts as binary_predicted_parts  # noqa: E402
from projects.common.free_space import decompose, decompose_from_class_index  # noqa: E402
from projects.common.free_space_metrics import (  # noqa: E402
    fatal_rate,
    free_miss_rate,
    iou_free,
)
from projects.common.polar import RAY_OK, build_ray_index, first_free_range  # noqa: E402
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    GRID_SPEC,
    RobotBEVDataset,
    list_sequence_samples,
    parse_sequence_names,
)
from projects.geometry.double_sphere import FINETUNE_CAMERA_NAMES  # noqa: E402
from projects.datasets.simplebev_vox import (  # noqa: E402
    height_config_for_ckpt_dirs,
    vox_dims,
)
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.pixel_grid import convention_for_checkpoints  # noqa: E402
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402
from tools.rescore_checkpoints import load_checkpoint_state_dict  # noqa: E402

_CSV_FIELDS = (
    "sequence", "sample_id", "iou_free", "fatal_rate", "free_miss_rate",
    "fatal_cells", "miss_cells", "over_share", "range_mae", "range_bias", "n_paired_rays",
)


def frame_rows(free_pred, free_gt, valid, rays, identities, deltas=None) -> list:
    """프레임 하나당 한 줄. 요약 지표를 배치로 평균내지 않고 **프레임 단위로** 남긴다.

    `over_share` = fatal 셀 / (fatal + miss 셀). 사용자가 관찰한 기제("애매한 통로를
    모델은 지나갈 수 있다고 본다")가 참이면 나쁜 프레임에서 이 값이 1에 가까워야 한다.
    반대로 모델이 과잉 보수적이어서 점수가 낮다면 0에 가까워진다 -- 두 방향은 비용의
    종류가 정반대이므로 절대 하나의 `iou_free`로 합쳐 읽지 않는다.

    셀 수를 비율과 함께 남기는 이유: 프레임의 free 면적이 작으면 비율이 몇 개 셀로
    출렁인다. CSV에서 그 프레임을 걸러낼 수 있어야 한다.
    """
    valid_b = valid.bool()
    pred = free_pred.bool() & valid_b
    gt = free_gt.bool() & valid_b
    rows = []
    for i, (sequence, sample_id) in enumerate(identities):
        one = slice(i, i + 1)
        fatal_cells = int((pred[one] & ~gt[one]).sum().item())
        miss_cells = int((~pred[one] & gt[one]).sum().item())
        disagree = fatal_cells + miss_cells
        # 광선 루프는 프레임당 수십 ms라 두 번 돌리지 않는다 -- 호출부가 이미 계산했다면
        # 그것을 그대로 쓴다.
        delta = _paired_delta(pred[i, 0], gt[i, 0], rays)[0] if deltas is None else deltas[i]
        rows.append({
            "sequence": sequence,
            "sample_id": sample_id,
            "iou_free": iou_free(free_pred[one], free_gt[one], valid[one])[0],
            "fatal_rate": fatal_rate(free_pred[one], free_gt[one], valid[one])[0],
            "free_miss_rate": free_miss_rate(free_pred[one], free_gt[one], valid[one])[0],
            "fatal_cells": fatal_cells,
            "miss_cells": miss_cells,
            "over_share": fatal_cells / disagree if disagree else float("nan"),
            "range_mae": float(np.abs(delta[np.isfinite(delta)]).mean())
                         if np.isfinite(delta).any() else float("nan"),
            "range_bias": float(delta[np.isfinite(delta)].mean())
                          if np.isfinite(delta).any() else float("nan"),
            "n_paired_rays": int(np.isfinite(delta).sum()),
        })
    return rows


def _paired_delta(pred_free_2d, gt_free_2d, rays):
    """방위각별 `(dr = r_pred - r_gt, r_gt)`. 짝이 안 맞는 광선은 NaN으로 **자리를 남긴다.**

    `free_space_metrics.range_error`는 짝지어진 값만 이어 붙이므로 어느 방위각의 값인지가
    사라진다. 방위각 프로파일을 보려면 (n_theta,) 자리를 유지해야 한다.

    `r_gt`를 함께 돌려주는 이유: 전방 광선은 통로가 길어 GT 거리 자체가 크므로 meter 단위
    오차도 같이 커진다. 이 거리 효과를 방위각 효과로 오독하지 않으려면 `dr/r_gt`도 봐야 한다.
    """
    pred_np = pred_free_2d.cpu().numpy()
    gt_np = gt_free_2d.cpu().numpy()
    r_gt, s_gt = first_free_range(gt_np, rays)
    r_pred, s_pred = first_free_range(pred_np, rays)
    paired = (s_gt == RAY_OK) & (s_pred == RAY_OK)
    delta = np.full(paired.shape, np.nan)
    delta[paired] = r_pred[paired] - r_gt[paired]
    gt_range = np.where(paired, r_gt, np.nan)
    return delta, gt_range


def azimuth_profile(deltas, gt_ranges=None, n_bins=36) -> list:
    """(n_frames, n_theta) 오차를 방위각 구간으로 묶는다. theta=0이 전방, 반시계 방향.

    프레임끼리 평균하기 전에 구간으로 묶는 이유: 광선 하나하나는 표본이 적어 출렁이지만
    구간 평균은 캘리브레이션 결함이 만드는 **매끄러운 구조**를 드러낸다.

    `gt_ranges`를 주면 `rel_mae`(= mean |dr| / r_gt)를 함께 낸다. meter 단위 `mae`만 보면
    "전방이 제일 틀린다"가 나오는데, 전방은 통로가 길어 `r_gt` 자체가 큰 방향이다. 두 값이
    다른 이야기를 하면 그건 방위각 효과가 아니라 **거리 효과**다.
    """
    stacked = np.asarray(deltas, dtype=float)
    if stacked.ndim != 2:
        raise ValueError(f"(n_frames, n_theta) 배열이어야 한다: {stacked.shape}")
    n_theta = stacked.shape[1]
    if n_theta % n_bins:
        raise ValueError(f"n_theta({n_theta})가 n_bins({n_bins})로 나누어지지 않는다")
    ranges = None if gt_ranges is None else np.asarray(gt_ranges, dtype=float)
    if ranges is not None and ranges.shape != stacked.shape:
        raise ValueError(f"gt_ranges 형상이 deltas와 다르다: {ranges.shape} vs {stacked.shape}")
    per_bin = n_theta // n_bins
    rows = []
    for b in range(n_bins):
        window = slice(b * per_bin, (b + 1) * per_bin)
        chunk = stacked[:, window]
        finite = np.isfinite(chunk)
        values = chunk[finite]
        row = {
            "deg_from": b * 360.0 / n_bins,
            "deg_to": (b + 1) * 360.0 / n_bins,
            "mae": float(np.abs(values).mean()) if values.size else float("nan"),
            "bias": float(values.mean()) if values.size else float("nan"),
            "n": int(values.size),
            "r_gt_mean": float("nan"),
            "rel_mae": float("nan"),
            "rel_bias": float("nan"),
        }
        if ranges is not None and values.size:
            r = ranges[:, window][finite]
            positive = r > 0
            row["r_gt_mean"] = float(r.mean())
            if positive.any():
                row["rel_mae"] = float(np.abs(values[positive] / r[positive]).mean())
                row["rel_bias"] = float((values[positive] / r[positive]).mean())
        rows.append(row)
    return rows


def mirror_asymmetry(profile) -> list:
    """좌우 짝(+θ vs −θ)의 차이. **§18의 기하 결함을 라벨 애매성과 갈라내는 열쇠다.**

    좌우 extrinsic이 바뀌었거나 sample 좌표가 한쪽으로 밀렸다면 오차가 좌우로 비대칭하다.
    양옆 잎에서 오는 annotation 애매성은 좌우 어느 쪽에도 특별히 몰릴 이유가 없다.
    """
    n = len(profile)
    pairs = []
    for b in range(1, n // 2):
        left, right = profile[b], profile[n - b]
        pairs.append({
            "deg": left["deg_from"],
            "mae_ccw": left["mae"], "mae_cw": right["mae"],
            "mae_diff": left["mae"] - right["mae"],
            "bias_ccw": left["bias"], "bias_cw": right["bias"],
        })
    return pairs


def decile_contrast(rows, key="over_share") -> dict:
    """`iou_free` 최하위 10 %와 최상위 10 %에서 `key`의 평균을 대조한다.

    상관계수 하나로 뭉개지 않는 이유: 우리가 답하려는 질문은 "나쁜 프레임이 **어떤 종류로**
    나쁜가"이고, 그건 분포의 양 끝을 직접 비교해야 보인다.
    """
    finite = [r for r in rows if np.isfinite(r["iou_free"]) and np.isfinite(r[key])]
    if len(finite) < 10:
        return {"n": len(finite)}
    ordered = sorted(finite, key=lambda r: r["iou_free"])
    k = max(1, len(ordered) // 10)
    return {
        "n": len(finite), "k": k,
        "worst_iou": float(np.mean([r["iou_free"] for r in ordered[:k]])),
        "best_iou": float(np.mean([r["iou_free"] for r in ordered[-k:]])),
        f"worst_{key}": float(np.mean([r[key] for r in ordered[:k]])),
        f"best_{key}": float(np.mean([r[key] for r in ordered[-k:]])),
    }


def split_by_quality(rows, deltas, gt_ranges, n_bins=36) -> dict:
    """프레임을 `iou_free` 중앙값으로 갈라 각각의 방위각 프로파일을 낸다.

    **이것이 두 가설을 가르는 결정적 대조다.** 전방 range bias가
    - 좋은 프레임에도 나쁜 프레임에도 **똑같이** 있으면 -> 모든 프레임에 공통인 체계적
      오차이므로 기하/캘리브레이션(§18) 쪽이다. 라벨이 애매했던 소수 프레임 때문일 수 없다.
    - 나쁜 프레임에만 몰려 있으면 -> 특정 장면의 라벨 애매성 쪽이다.
    """
    ious = np.array([r["iou_free"] for r in rows], dtype=float)
    order = np.argsort(np.where(np.isfinite(ious), ious, np.inf))
    half = len(order) // 2
    stacked_d = np.asarray(deltas, dtype=float)
    stacked_r = np.asarray(gt_ranges, dtype=float)
    groups = {"worst_half": order[:half], "best_half": order[half:]}
    return {
        name: {
            "n_frames": int(idx.size),
            "mean_iou": float(np.nanmean(ious[idx])) if idx.size else float("nan"),
            "profile": azimuth_profile(stacked_d[idx], stacked_r[idx], n_bins=n_bins),
        }
        for name, idx in groups.items()
    }


def _histogram(values, edges) -> list:
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=edges)
    return [(float(lo), float(hi), int(c))
            for lo, hi, c in zip(edges[:-1], edges[1:], counts)]


def _print_report(rows, deltas, gt_ranges, n_bins, worst_n):
    ious = [r["iou_free"] for r in rows if np.isfinite(r["iou_free"])]
    print(f"\n프레임 {len(rows)}개 | iou_free 유효 {len(ious)}개")
    quantiles = np.percentile(ious, [0, 10, 25, 50, 75, 90, 100])
    print("iou_free 분위: " + "  ".join(
        f"p{p}={v:.3f}" for p, v in zip((0, 10, 25, 50, 75, 90, 100), quantiles)
    ))
    print(f"평균 {np.mean(ious):.4f}  표준편차 {np.std(ious):.4f}"
          f"  IQR {quantiles[4] - quantiles[2]:.4f}")
    print("\n[프레임별 iou_free 히스토그램] 좁은 단봉형=기하 결함, 왼쪽 꼬리=라벨 애매성")
    edges = np.arange(0.0, 1.0001, 0.05)
    for lo, hi, count in _histogram(ious, edges):
        if count or 0.5 <= lo:
            print(f"  {lo:.2f}-{hi:.2f} | {'#' * min(count, 60)} {count}")

    contrast = decile_contrast(rows)
    if "worst_iou" in contrast:
        print(f"\n[최하위/최상위 10 % 대조] 각 {contrast['k']}프레임")
        print(f"  최하위: iou_free {contrast['worst_iou']:.3f}"
              f"  over_share {contrast['worst_over_share']:.3f}")
        print(f"  최상위: iou_free {contrast['best_iou']:.3f}"
              f"  over_share {contrast['best_over_share']:.3f}")
        print("  over_share가 최하위에서 1에 가까우면 '모델이 GT보다 free를 넓게 본다'"
              " = 라벨 애매성 가설의 방향이다")

    profile = azimuth_profile(deltas, gt_ranges, n_bins=n_bins)
    print(f"\n[방위각 프로파일] {n_bins}구간, theta=0이 전방·반시계 방향")
    print(f"  {'deg':>12} {'mae(m)':>9} {'bias(m)':>9} {'r_gt(m)':>9}"
          f" {'rel_mae':>9} {'rel_bias':>9} {'n':>8}")
    for row in profile:
        print(f"  {row['deg_from']:5.0f}-{row['deg_to']:5.0f} {row['mae']:9.3f}"
              f" {row['bias']:+9.3f} {row['r_gt_mean']:9.3f}"
              f" {row['rel_mae']:9.3f} {row['rel_bias']:+9.3f} {row['n']:8d}")

    print("\n[품질별 대조] 전방 bias가 두 집단에 똑같이 있으면 기하 결함,"
          " 나쁜 프레임에만 몰리면 라벨 애매성이다")
    groups = split_by_quality(rows, deltas, gt_ranges, n_bins=n_bins)
    print(f"  {'deg':>12} " + " ".join(
        f"{name+' bias':>18}" for name in ("worst_half", "best_half")
    ))
    for b in range(n_bins):
        cells = []
        for name in ("worst_half", "best_half"):
            row = groups[name]["profile"][b]
            cells.append(f"{row['bias']:+9.3f}({row['n']:5d})" if row["n"]
                         else f"{'-':>9}({row['n']:5d})")
        low = b * 360.0 / n_bins
        print(f"  {low:5.0f}-{low + 360.0 / n_bins:5.0f} " + " ".join(f"{c:>18}" for c in cells))
    for name in ("worst_half", "best_half"):
        print(f"  {name}: {groups[name]['n_frames']}프레임,"
              f" 평균 iou_free {groups[name]['mean_iou']:.3f}")

    print("\n[좌우 비대칭] 차이가 크고 부호가 일관되면 기하/extrinsic 결함 쪽이다")
    print(f"  {'deg':>6} {'mae_ccw':>9} {'mae_cw':>9} {'diff':>9}"
          f" {'bias_ccw':>9} {'bias_cw':>9}")
    diffs = []
    for pair in mirror_asymmetry(profile):
        diffs.append(pair["mae_diff"])
        print(f"  {pair['deg']:6.0f} {pair['mae_ccw']:9.3f} {pair['mae_cw']:9.3f}"
              f" {pair['mae_diff']:+9.3f} {pair['bias_ccw']:+9.3f} {pair['bias_cw']:+9.3f}")
    finite_diffs = [d for d in diffs if np.isfinite(d)]
    if finite_diffs:
        print(f"  |diff| 평균 {np.mean(np.abs(finite_diffs)):.3f} m,"
              f" 부호 합 {np.sum(np.sign(finite_diffs)):+.0f}/{len(finite_diffs)}")

    print(f"\n[최악 {worst_n}프레임] 눈으로 확인할 대상")
    ordered = sorted((r for r in rows if np.isfinite(r["iou_free"])),
                     key=lambda r: r["iou_free"])
    for row in ordered[:worst_n]:
        print(f"  {row['sequence']}/{row['sample_id']}  iou_free {row['iou_free']:.3f}"
              f"  over_share {row['over_share']:.3f}"
              f"  fatal {row['fatal_cells']:5d} miss {row['miss_cells']:5d}"
              f"  range_mae {row['range_mae']:.3f}")


def main(
    checkpoint,
    sequences="raws1,rawos3",
    dataset_root=DEFAULT_DATASET_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    encoder_type="res101",
    # 학습 때와 같은 값을 줘야 한다. 틀리면 head 채널 수가 안 맞아 즉시 실패한다.
    formulation="binary",
    batch_size=4,
    num_workers=4,
    n_theta=None,
    n_bins=36,
    worst_n=20,
    out_csv="",
    device="cuda",
):
    if formulation not in ("three_class", "binary"):
        raise ValueError(f"formulation은 three_class 또는 binary여야 한다: {formulation}")
    dataset_root = Path(dataset_root)
    samples = [
        s for name in parse_sequence_names(sequences)
        for s in list_sequence_samples(dataset_root / name)
    ]
    dataset = RobotBEVDataset(samples, common_root=common_root)
    # `shuffle=False`가 계약이다 -- 프레임 정체(sequence/sample_id)를 배치 순서로만
    # 되찾으므로, 섞으면 CSV의 이름과 숫자가 조용히 어긋난다.
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


    # 표본 규약은 **체크포인트 옆 config.json에서 되찾는다** -- 기본값을 쓰면 옛 런(legacy)을
    # 새 기하로 재채점해 조용히 다른 숫자가 나온다.
    convention, offset = convention_for_checkpoints([checkpoint])
    _height = height_config_for_ckpt_dirs([Path(checkpoint).parent])
    Z, Y, X = vox_dims(GRID_SPEC, _height["height_bins"])
    vox_util = build_double_sphere_vox_util(GRID_SPEC, dataset.cameras, device=device,
                                           height_bins=_height["height_bins"],
                                           height_min_m=_height["height_min_m"],
                                           height_max_m=_height["height_max_m"],
                                           pixel_convention=convention,
                                           pixel_offset=offset)
    model = ThreeClassSegnet(
        Z, Y, X, vox_util, use_radar=False, use_lidar=False,
        do_rgbcompress=True, encoder_type=encoder_type, rand_flip=False,
        num_classes=2 if formulation == "binary" else 3,
    ).to(device)
    load_checkpoint_state_dict(model, checkpoint, device)
    model.eval()

    rays = build_ray_index(GRID_SPEC) if n_theta is None else build_ray_index(GRID_SPEC, n_theta=n_theta)

    def decompose_pred(logits, valid):
        if formulation == "binary":
            return binary_predicted_parts(logits, valid, rays)
        return decompose_from_class_index(logits.argmax(dim=1, keepdim=True), valid)

    rows, deltas, gt_ranges = [], [], []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb_camXs"].to(device) - 0.5
            _, _, logits, _, _ = model(
                rgb, batch["pix_T_cams"].to(device), batch["cam0_T_camXs"].to(device), vox_util
            )
            valid = batch["valid_bev_g"].to(device)
            gt_parts = decompose(batch["seg_bev_g"].to(device),
                                 batch["vis_bev_g"].to(device), valid)
            pred_parts = decompose_pred(logits, valid)
            count = valid.shape[0]
            identities = [(Path(root).name, sample_id)
                          for root, sample_id in dataset.samples[offset:offset + count]]
            offset += count

            pred, gt = pred_parts["free"], gt_parts["free"]
            valid_b = valid.bool()
            batch_deltas = []
            for i in range(count):
                delta, gt_range = _paired_delta(pred[i, 0].bool() & valid_b[i, 0],
                                                gt[i, 0].bool() & valid_b[i, 0], rays)
                batch_deltas.append(delta)
                gt_ranges.append(gt_range)
            deltas.extend(batch_deltas)
            rows.extend(frame_rows(pred, gt, valid, rays, identities, deltas=batch_deltas))
            print(f"\r  {offset}/{len(dataset)} 프레임", end="", flush=True)

    _print_report(rows, deltas, gt_ranges, n_bins, worst_n)

    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_CSV_FIELDS))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n프레임별 원자료 -> {out_path}")


if __name__ == "__main__":
    Fire(main)
