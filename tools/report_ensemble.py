"""체크포인트 앙상블 -- `experiment_history.md` §5 "다음 세션이 할 일" 4번.

## 무엇을 묻는가

§16.6은 "현재 modeling pipeline이 **공유하는** 오차"라는 표현을 쓴다. 그 근거는 시드 3개
앙상블이 `iou_free` 격차의 1.1 %만 닫는다는 실측(0.7977 -> 0.8000)인데, **세 시드는 초기화와
배치 순서만 다르다.** 그래서 그 실측은 loss 계열·아키텍처·데이터·split이 만드는 오차를
구분하지 못한다.

**loss 계열을 넘어 앙상블하면 한 단계 강해진다.** `A_ce`(가중 CE)와 `C_soft`/`D_range`
(soft-BCE)는 목적함수가 다르므로 오차 구조가 다를 여지가 있다. 9개를 섞어도 0.800 근처에서
멈추면 "loss 계열과 초기화를 바꿔도 남는 오차"가 된다.

**여전히 남는 교란: 같은 아키텍처·같은 데이터·같은 split이다.** 이 도구의 결과로
`task ceiling`이라고 부르면 안 된다(§16.6).

## 방법

`softmax` 확률의 **산술 평균**을 앙상블 확률로 쓰고, τ로 잘라 학습 루프와 **같은 지표
함수**로 채점한다(`free_metrics_from_masks`·`range_error`·`tolerance_counts`).
logit 평균이 아니라 확률 평균인 이유: §7.1의 기존 실측이 확률 평균이었고 숫자를 나란히
읽어야 한다.

forward 결과는 캐시한다 -- 조합을 바꿀 때마다 9개를 다시 돌리지 않는다.

실행:

    python tools/report_ensemble.py --log_root=runs/ablation --cells=A_ce,C_soft,D_range
"""
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.models\._utils")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.common.bev_occupancy_metrics import (  # noqa: E402
    append_free_metrics,
    summarize_free_metrics,
)
from projects.common.free_space import decompose  # noqa: E402
from projects.common.free_space_metrics import (  # noqa: E402
    free_metrics_from_masks,
    range_error,
    summarize_range_error,
)
from projects.common.occupied_metrics import (  # noqa: E402
    derive_occupied,
    summarize_tolerance_f1,
    tolerance_counts,
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


def _width(text) -> int:
    from unicodedata import east_asian_width
    return sum(2 if east_asian_width(ch) in "WF" else 1 for ch in str(text))


def _pad(text, width, align="<") -> str:
    fill = max(0, width - _width(text))
    return f"{text}{' ' * fill}" if align == "<" else f"{' ' * fill}{text}"


def _csv(value):
    """Fire는 `--cells=A_ce,C_soft`를 **tuple**로 파싱한다 -- 스칼라만 가정하면 조용히 깨진다."""
    if isinstance(value, (tuple, list)):
        return tuple(str(v).strip() for v in value)
    return tuple(v.strip() for v in str(value).split(",") if v.strip())


def _load(ckpt, vox, encoder_type, device):
    model = ThreeClassSegnet(GRID_SPEC.n_rows, vox.Y, GRID_SPEC.n_cols, vox, use_radar=False,
                             use_lidar=False, do_rgbcompress=True, encoder_type=encoder_type,
                             rand_flip=False, num_classes=2).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=True)
    model.eval()
    return model


def main(
    log_root="runs/ablation",
    cells="A_ce,C_soft,D_range",
    seeds="0,1,2",
    tau=0.5,
    train_sequences="raws2,raws3,rawos1,rawos2,rawos4",
    val_sequences="raws1,rawos3",
    dataset_root=DEFAULT_DATASET_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    encoder_type="res101",
    batch_size=8,
    num_workers=8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cells = _csv(cells)
    seeds = tuple(int(s) for s in _csv(seeds))

    root = Path(dataset_root)
    names = parse_sequence_names(train_sequences)
    val_names = parse_sequence_names(val_sequences)
    _, val_samples = split_samples_by_sequence([root / n for n in names + val_names], val_names)
    dataset = RobotBEVDataset(val_samples, common_root=common_root, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # 표본 규약은 **런의 config.json에서 되찾는다** -- 기본값을 쓰면 `runs/ablation`(legacy)을
    # 새 기하로 재채점해 조용히 다른 숫자가 나온다.
    convention, offset = convention_for_run_dirs(
        [Path(log_root) / "logs" / f"{cell}_s{seed}" for cell in cells for seed in seeds])
    _height = height_config_for_ckpt_dirs(
        [Path(log_root) / "ckpt" / f"{cell}_s{seed}" for cell in cells for seed in seeds])
    vox_util = build_double_sphere_vox_util(GRID_SPEC, dataset.cameras, device=device,
                                           height_bins=_height["height_bins"],
                                           height_min_m=_height["height_min_m"],
                                           height_max_m=_height["height_max_m"],
                                           pixel_convention=convention, pixel_offset=offset)
    rays = build_ray_index(GRID_SPEC)

    labels = []
    for batch in loader:
        seg, vis, valid = (batch[k].to(device) for k in ("seg_bev_g", "vis_bev_g", "valid_bev_g"))
        labels.append((decompose(seg, vis, valid), valid))

    def score(prob_batches):
        """캐시된 `p(free)`를 τ로 잘라 학습 루프와 같은 지표를 낸다."""
        free_d, range_d, tol_d = [], [], []
        for prob, (gt, valid) in zip(prob_batches, labels):
            valid_b = valid.bool()
            pred_free = (prob > tau) & valid_b
            pred = {"free": pred_free,
                    "occupied": derive_occupied(pred_free, valid, rays) & valid_b}
            pred["unknown"] = valid_b & ~pred["free"] & ~pred["occupied"]
            metrics = free_metrics_from_masks(pred, gt, valid)
            append_free_metrics(free_d, metrics)
            range_d.append(range_error(metrics["pred_free"], metrics["gt_free"], valid, rays))
            tol_d.append(tolerance_counts(metrics["pred_occupied"], metrics["gt_occupied"],
                                          valid, GRID_SPEC.cell_m))
        free = summarize_free_metrics(free_d)
        rng = summarize_range_error(range_d)
        tol = summarize_tolerance_f1(tol_d)
        return {"iou_free": free["iou_free"], "fatal": free["fatal_rate"],
                "free_miss": free["free_miss_rate"], "f1@10cm": tol["10cm"]["f1"],
                "range_mae": rng["mae"], "missed_obs": rng["missed_obstacle_rate"]}

    cache_dir = Path(log_root) / "analysis" / "probs"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 모델별 `p(free)`를 모은다. float16으로 캐시한다 -- τ=0.5 판정에 f16 정밀도(약 5e-4)는
    # 넉넉하고, 9개 모델 x 75프레임을 20 MB 안에 담는다.
    probs = {}
    for cell in cells:
        for seed in seeds:
            found = sorted(Path(f"{log_root}/ckpt/{cell}_s{seed}").glob("model_best-*.pth"))
            if not found:
                print(f"!! 체크포인트가 없어 건너뛴다: {log_root}/ckpt/{cell}_s{seed}")
                continue
            key = f"{cell}_s{seed}"
            cache = cache_dir / f"{key}_{found[-1].stem}.npy"
            if cache.exists():
                stacked = torch.from_numpy(np.load(cache)).to(device).float()
                print(f"  {key} (캐시)", flush=True)
            else:
                model = _load(found[-1], vox_util, encoder_type, device)
                chunks = []
                with torch.no_grad():
                    for batch in loader:
                        _, _, logits, _, _ = model(batch["rgb_camXs"].to(device) - 0.5,
                                                   batch["pix_T_cams"].to(device),
                                                   batch["cam0_T_camXs"].to(device), vox_util)
                        chunks.append(torch.softmax(logits, dim=1)[:, 1:2].clone())
                del model
                torch.cuda.empty_cache()
                stacked = torch.cat(chunks, dim=0)
                np.save(cache, stacked.half().cpu().numpy())
                print(f"  {key} ({found[-1].name})", flush=True)
            probs[key] = stacked

    if not probs:
        return

    batch_sizes = [gt["free"].shape[0] for gt, _ in labels]

    def as_batches(tensor):
        """(F, 1, Z, X) -> loader의 배치 경계로 다시 자른다."""
        out, start = [], 0
        for n in batch_sizes:
            out.append(tensor[start:start + n])
            start += n
        return out

    def ensemble(keys):
        return score(as_batches(torch.stack([probs[k] for k in keys]).mean(dim=0)))

    rows = []
    # 단일 모델 평균 -- 앙상블 이득의 기준선이다.
    for cell in cells:
        keys = [f"{cell}_s{s}" for s in seeds if f"{cell}_s{s}" in probs]
        if not keys:
            continue
        singles = [score(as_batches(probs[k])) for k in keys]
        rows.append((f"{cell} 단일 평균", len(keys),
                     {k: float(np.mean([s[k] for s in singles])) for k in singles[0]}))
        if len(keys) >= 2:
            rows.append((f"{cell} 시드 앙상블", len(keys), ensemble(keys)))

    # loss 계열 교차 -- 이 도구의 목적. 쌍도 같이 낸다(어느 계열 조합이 이득을 주는가).
    for pair in combinations(cells, 2):
        keys = [k for c in pair for s in seeds if (k := f"{c}_s{s}") in probs]
        if len(keys) >= 2:
            rows.append((f"{'+'.join(pair)} 교차", len(keys), ensemble(keys)))
    if len(cells) >= 3:
        keys = [k for c in cells for s in seeds if (k := f"{c}_s{s}") in probs]
        if len(keys) >= 2:
            rows.append((f"{'+'.join(cells)} 교차 (전체)", len(keys), ensemble(keys)))

    print(f"\n=== 체크포인트 앙상블 (softmax 확률 평균, τ={tau:.2f}) ===")
    print("**`task ceiling`이라고 부르지 않는다**(§16.6) -- 같은 아키텍처·데이터·split이 남아 있다.")
    print("`격차 닫음`은 `iou_free`가 1.0까지 남은 격차 중 몇 %를 닫았나다."
          " 기준은 각 행이 속한 계열의 단일 평균이 아니라\n"
          "**첫 행(단일 평균)**이므로 계열 간에 바로 비교된다.\n")

    head = [("구성", 26), ("n", 3), ("iou_free", 9), ("격차 닫음", 10), ("fatal", 8),
            ("free_miss", 10), ("f1@10cm", 9), ("range_mae", 10), ("missed_obs", 11)]
    print("  " + " ".join(_pad(h, w, ">") for h, w in head))
    ref = rows[0][2]["iou_free"] if rows else float("nan")
    for name, n, v in rows:
        closed = (v["iou_free"] - ref) / max(1.0 - ref, 1e-9) * 100.0
        row = [name, str(n), f"{v['iou_free']:.4f}", f"{closed:+.2f} %", f"{v['fatal']:.4f}",
               f"{v['free_miss']:.4f}", f"{v['f1@10cm']:.4f}", f"{v['range_mae']:.4f}",
               f"{v['missed_obs']:.4f}"]
        print("  " + " ".join(_pad(c, w, ">") for c, (_, w) in zip(row, head)))

    print("\n  판독: 교차 앙상블이 시드 앙상블보다 뚜렷하게 낫지 않으면 **loss 계열을 바꿔도")
    print("  같은 오차를 낸다**는 뜻이고, §16.6의 '공유 오차'가 한 단계 강해진다.")


if __name__ == "__main__":
    Fire(main)
