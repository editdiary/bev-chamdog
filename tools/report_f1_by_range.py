"""`f1@τ`를 **거리별로** 분해한다 -- lifting 해상도가 지표의 바닥인지 확인한다 (2026-08-25).

## 무엇을 묻는가

`f1@10cm`(예측 경계가 실제 경계와 10 cm 안에서 일치하는 비율)이 0.55 근처에서 막혀 있고,
loss·정규화·데이터를 바꿔도 움직이지 않았다(진단 문서 §17.1·§19·§20.2).

한 가지 설명 후보가 새로 나왔다. **BEV 격자(5 cm)가 이미지 특징이 분해할 수 있는 거리보다
촘촘하다.** 실제 캘리브레이션으로 재면 특징맵 1픽셀이 BEV에서 갖는 크기가

    0~1 m  6 cm | 1~2 m  10.5 cm | 2~3 m  16 cm | 3~4 m  20~24 cm | 4~6 m  24~28 cm

다. 즉 **1.5 m를 넘으면 10 cm 정밀도는 입력에 없는 정보를 요구하는 것**이다. 그렇다면
`f1@10cm`은 근거리에서 높고 원거리에서 무너져야 하고, 전체 값 0.55는 그 평균일 뿐이다.

**반증도 준비해 둔다.** 이미 측정된 `range_mae`는 0~1.5 m에서 0.1987 m인데 그 구간의
해상도 바닥은 6~10 cm다 -- **오차가 바닥의 약 2배**다. 즉 해상도는 바닥이긴 하지만 근거리
오차를 다 설명하지 못한다. 그래서 이 도구가 답할 질문은 "해상도가 원인인가"가 아니라
**"해상도 바닥이 어느 거리부터 지배하는가"**다.

## 왜 `valid`에 링 마스크를 곱하면 안 되나

`tolerance_counts`가 하듯 `valid`를 링으로 자르면 **거리장이 링 안에서만 계산된다.** 그러면
1.49 m의 예측 경계와 1.52 m의 실제 경계가 서로 안 보여서 둘 다 실패로 세어진다 -- 링 경계에
가짜 오차가 생긴다. 그래서 여기서는 **거리장을 전체 맵에서 만들고, 셀을 셀 때만 링으로
제한한다.**

실행: python tools/report_f1_by_range.py --log_root=runs/ablation --cells=D_range
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.models\._utils")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.common.free_space import decompose  # noqa: E402
from projects.common.free_space_metrics import DEFAULT_RING_EDGES_M, build_ring_masks  # noqa: E402
from projects.common.occupied_metrics import (  # noqa: E402
    DEFAULT_TOLERANCES_M,
    _distance_field_m,
    derive_occupied,
    tolerance_key,
)
from projects.common.polar import build_ray_index  # noqa: E402
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    GRID_SPEC,
    RobotBEVDataset,
    parse_sequence_names,
    split_samples_by_sequence,
)
from projects.datasets.simplebev_vox import height_config_for_ckpt_dirs  # noqa: E402
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.pixel_grid import convention_for_run_dirs  # noqa: E402
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402

# `measure_warp` 실측값(§18.3 계열 작업). 표에 같이 찍어 지표와 바닥을 나란히 읽게 한다.
_RESOLUTION_CM = {"0.0-1.5m": "6~10", "1.5-3.0m": "10~16", "3.0-4.0m": "20~24"}


def _width(text) -> int:
    from unicodedata import east_asian_width
    return sum(2 if east_asian_width(ch) in "WF" else 1 for ch in str(text))


def _pad(text, width, align="<") -> str:
    fill = max(0, width - _width(text))
    return f"{text}{' ' * fill}" if align == "<" else f"{' ' * fill}{text}"


def _csv(value):
    if isinstance(value, (tuple, list)):
        return tuple(str(v).strip() for v in value)
    return tuple(v.strip() for v in str(value).split(",") if v.strip())


def main(
    log_root="runs/ablation",
    cells="A_ce,C_soft,D_range",
    seeds="0,1,2",
    tau=0.5,
    train_sequences="raws2,raws3,rawos1,rawos2,rawos4",
    val_sequences="raws1,rawos3",
    # 학습이 남긴 `split_val_samples.txt` 경로. 주면 위 두 인자를 무시하고 그 목록을 채점한다.
    split_file=None,
    dataset_root=DEFAULT_DATASET_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    encoder_type="res101",
    batch_size=4,
    num_workers=4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cells, seeds = _csv(cells), tuple(int(s) for s in _csv(seeds))

    root = Path(dataset_root)
    if split_file:
        # 프로브 런(프레임 단위 무작위 split)을 채점할 때 쓴다. 학습이 남긴
        # `split_val_samples.txt`를 그대로 읽으므로 **split을 다시 유도하지 않는다** --
        # 재유도하면 split_seed나 프레임 목록이 조금 달라져도 조용히 다른 집합을 채점한다.
        lines = [ln.strip() for ln in Path(split_file).read_text().splitlines() if ln.strip()]
        val_samples = [(root / ln.split("/")[0], ln.split("/")[1]) for ln in lines]
        print(f"[split] {split_file}에서 val {len(val_samples)}프레임을 읽었다")
    else:
        names = parse_sequence_names(train_sequences)
        val_names = parse_sequence_names(val_sequences)
        _, val_samples = split_samples_by_sequence(
            [root / n for n in names + val_names], val_names)
    dataset = RobotBEVDataset(val_samples, common_root=common_root, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # 표본 규약은 **런의 config.json에서 되찾는다** -- 기본값을 쓰면 `runs/ablation`(legacy)을
    # 새 기하로 재채점해 조용히 다른 숫자가 나온다.
    convention, offset = convention_for_run_dirs(
        [Path(log_root) / "logs" / f"{cell}_s{seed}" for cell in cells for seed in seeds])
    height = height_config_for_ckpt_dirs(
        [Path(f"{log_root}/ckpt/{cell}_s{seed}") for cell in cells for seed in seeds])
    vox_util = build_double_sphere_vox_util(GRID_SPEC, dataset.cameras, device=device,
                                           pixel_convention=convention, pixel_offset=offset,
                                           height_bins=height["height_bins"],
                                           height_min_m=height["height_min_m"],
                                           height_max_m=height["height_max_m"])
    rays = build_ray_index(GRID_SPEC)
    rings = build_ring_masks(GRID_SPEC, DEFAULT_RING_EDGES_M)
    cell_m = GRID_SPEC.cell_m

    labels = []
    for batch in loader:
        seg, vis, valid = (batch[k].to(device) for k in ("seg_bev_g", "vis_bev_g", "valid_bev_g"))
        labels.append((decompose(seg, vis, valid), valid))

    def counts_for(model):
        """(ring, τ) -> precision/recall 카운트. 거리장은 **전체 맵**에서 만든다."""
        acc = {(name, tolerance_key(t)): dict.fromkeys(
            ("n_pred", "n_gt", "hit_pred", "hit_gt"), 0)
            for name, _ in rings for t in DEFAULT_TOLERANCES_M}
        with torch.no_grad():
            for batch, (gt_parts, valid) in zip(loader, labels):
                _, _, logits, _, _ = model(batch["rgb_camXs"].to(device) - 0.5,
                                           batch["pix_T_cams"].to(device),
                                           batch["cam0_T_camXs"].to(device), vox_util)
                valid_b = valid.bool()
                pred_free = (torch.softmax(logits, dim=1)[:, 1:2] > tau) & valid_b
                pred_occ = (derive_occupied(pred_free, valid, rays) & valid_b).cpu().numpy()
                gt_occ = (gt_parts["occupied"].bool() & valid_b).cpu().numpy()
                for i in range(gt_occ.shape[0]):
                    pred, gt = pred_occ[i, 0], gt_occ[i, 0]
                    to_gt, to_pred = _distance_field_m(gt, cell_m), _distance_field_m(pred, cell_m)
                    for name, ring in rings:
                        p_ring, g_ring = pred & ring, gt & ring
                        n_pred, n_gt = int(p_ring.sum()), int(g_ring.sum())
                        for t in DEFAULT_TOLERANCES_M:
                            b = acc[(name, tolerance_key(t))]
                            b["n_pred"] += n_pred
                            b["n_gt"] += n_gt
                            if to_gt is not None:
                                b["hit_pred"] += int((p_ring & (to_gt <= t)).sum())
                            if to_pred is not None:
                                b["hit_gt"] += int((g_ring & (to_pred <= t)).sum())
        return acc

    def f1(bucket):
        p = bucket["hit_pred"] / bucket["n_pred"] if bucket["n_pred"] else float("nan")
        r = bucket["hit_gt"] / bucket["n_gt"] if bucket["n_gt"] else float("nan")
        return 2 * p * r / (p + r) if p + r else float("nan")

    per_cell = {}
    for cell in cells:
        runs = []
        for seed in seeds:
            found = sorted(Path(f"{log_root}/ckpt/{cell}_s{seed}").glob("model_best-*.pth"))
            if not found:
                continue
            # `Y`는 vox_util이 이미 알고 있다 -- 리터럴을 다시 쓰면 둘이 어긋날 수 있다.
            model = ThreeClassSegnet(GRID_SPEC.n_rows, vox_util.Y, GRID_SPEC.n_cols, vox_util,
                                     use_radar=False, use_lidar=False, do_rgbcompress=True,
                                     encoder_type=encoder_type, rand_flip=False,
                                     num_classes=2).to(device)
            state = torch.load(found[-1], map_location=device, weights_only=False)
            model.load_state_dict(state.get("model_state_dict", state), strict=True)
            model.eval()
            runs.append(counts_for(model))
            del model
            torch.cuda.empty_cache()
            print(f"  {cell}_s{seed}", flush=True)
        if runs:
            per_cell[cell] = runs

    if not per_cell:
        print("체크포인트를 찾지 못했다.")
        return

    print(f"\n=== `f1@τ`의 거리별 분해 (τ_decision={tau:.2f}, 시드 평균) ===")
    print("`해상도 바닥`은 그 거리에서 특징맵 1픽셀이 BEV에서 갖는 크기다(실제 캘리브레이션 실측).")
    print("**허용 오차가 해상도 바닥보다 작으면 입력에 없는 정밀도를 요구하는 것이다.**")
    print("`경계 셀 수`는 그 링에 있는 실제 경계 셀 수 -- 전체 `f1`에서 각 링의 가중치다.\n")

    tol_keys = [tolerance_key(t) for t in DEFAULT_TOLERANCES_M]
    head = [("cell", 9), ("거리", 10), ("해상도 바닥", 13)] + \
           [(f"f1@{k}", 9) for k in tol_keys] + [("경계 셀 수", 11)]
    print("  " + " ".join(_pad(h, w, ">") for h, w in head))
    for cell, runs in per_cell.items():
        for name, _ in rings:
            values = [float(np.mean([f1(r[(name, k)]) for r in runs])) for k in tol_keys]
            n_gt = int(np.mean([runs[i][(name, tol_keys[0])]["n_gt"] for i in range(len(runs))]))
            row = [cell, name, f"{_RESOLUTION_CM.get(name, '?')} cm"] + \
                  [f"{v:.4f}" for v in values] + [f"{n_gt}"]
            print("  " + " ".join(_pad(c, w, ">") for c, (_, w) in zip(row, head)))
        print()

    print("  판독: `f1@10cm`이 근거리에서 높고 원거리에서 무너지면 **해상도 바닥이 지배**한다는")
    print("  뜻이고, 그러면 전체 `f1@10cm` 하나로 품질을 판정하는 것이 오독이다. 반대로 링마다")
    print("  비슷하면 해상도는 원인이 아니고 다른 곳을 봐야 한다.")


if __name__ == "__main__":
    Fire(main)
