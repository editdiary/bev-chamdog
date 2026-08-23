"""결정 threshold τ를 쓸어 `(free_miss, fatal)` 교환 곡선을 그린다 -- 설계 문서 §16의 도구.

**왜 필요한가.** `fatal`↓ / `free_miss`↑는 **두 가지 서로 다른 원인에서 똑같이 나온다**:
representation이 좋아져도 나오고, 결정 threshold가 올라가도 나온다. 이 프로젝트는 모든 판정을
τ=0.5(=`argmax`) **한 점**에서만 읽어 왔으므로 둘을 구분할 방법이 없었고, 그래서 §15.5(3)이
"안전 개선"이라고 부른 것이 실제로는 동작점 이동이었다.

구분하는 방법은 하나다 -- **같은 `free_miss`에서 `fatal`을 비교한다.**

- 곡선이 겹치면 → **동작점 이동.** 대조군도 τ만 바꾸면 공짜로 같은 것을 얻는다
- 같은 `free_miss`에서 항상 낮으면 → **진짜 safety-utility 교환 곡선 개선**

**앞으로 안전 지표로 loss·config를 판정할 때는 반드시 이 표를 같이 낸다.** τ=0.5 한 점의
`fatal` 차이는 그 자체로는 아무것도 뜻하지 않는다.

실행:

    python tools/report_threshold_sweep.py --log_root=runs/ablation --cells=A_ce,C_soft,D_range

**forward는 모델당 한 번만 한다.** τ는 캐시된 확률에서 쓸므로 τ를 9개 늘려도 비용이 거의
같다. 지표 정의는 학습 루프와 **같은 함수**를 쓴다(`free_metrics_from_masks`·`range_error`·
`tolerance_counts`) -- 다른 경로로 재면 기존 표와 나란히 읽을 수 없다.
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
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402

# τ=0.5는 `argmax`와 같으므로 **기존 표의 값이 이 스윕의 한 점으로 재현되어야 한다** --
# 안 되면 평가 경로가 갈린 것이다. 양 끝(0.2/0.8)은 곡선의 모양을 보기 위한 것이고 운용
# 후보는 아니다.
DEFAULT_TAUS = (0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80)

# 같은 `free_miss`에서 비교할 지점. 세 칸의 곡선이 모두 덮는 구간 안이어야 한다 --
# 밖이면 `np.interp`가 끝값으로 **고정(clamp)**되어 비교가 조용히 무의미해진다.
DEFAULT_ANCHORS = (0.085, 0.090, 0.095, 0.099, 0.104)


def _width(text) -> int:
    """한글은 터미널에서 두 칸을 쓴다 -- f-string의 폭 지정은 글자 수를 세므로 표가 밀린다."""
    from unicodedata import east_asian_width
    return sum(2 if east_asian_width(ch) in "WF" else 1 for ch in str(text))


def _pad(text, width, align="<") -> str:
    fill = max(0, width - _width(text))
    return f"{text}{' ' * fill}" if align == "<" else f"{' ' * fill}{text}"


def _load(ckpt, vox, encoder_type, device):
    model = ThreeClassSegnet(GRID_SPEC.n_rows, 1, GRID_SPEC.n_cols, vox, use_radar=False,
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
    taus=DEFAULT_TAUS,
    anchors=DEFAULT_ANCHORS,
    train_sequences="raws2,raws3,rawos1,rawos2,rawos4",
    val_sequences="raws1,rawos3",
    dataset_root=DEFAULT_DATASET_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    encoder_type="res101",
    batch_size=8,
    num_workers=8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cells = tuple(c.strip() for c in str(cells).split(","))
    seeds = tuple(int(s) for s in str(seeds).split(","))
    taus, anchors = tuple(taus), tuple(anchors)

    root = Path(dataset_root)
    names = parse_sequence_names(train_sequences)
    val_names = parse_sequence_names(val_sequences)
    _, val_samples = split_samples_by_sequence(
        [root / n for n in names + val_names], val_names)
    dataset = RobotBEVDataset(val_samples, common_root=common_root, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    vox_util = build_double_sphere_vox_util(GRID_SPEC, dataset.cameras, device=device)
    rays = build_ray_index(GRID_SPEC)

    # 라벨 분해는 모델과 무관하므로 한 번만 한다 (모델 x τ 만큼 반복하면 CPU가 병목이 된다).
    labels = []
    for batch in loader:
        seg, vis, valid = (batch[k].to(device)
                           for k in ("seg_bev_g", "vis_bev_g", "valid_bev_g"))
        labels.append((decompose(seg, vis, valid), valid))

    def score(prob_batches, tau):
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
        return {"free_miss": free["free_miss_rate"], "fatal": free["fatal_rate"],
                "missed_obs": rng["missed_obstacle_rate"], "iou_free": free["iou_free"],
                "range_mae": rng["mae"], "f1@10cm": tol["10cm"]["f1"]}

    rows = {}
    for cell in cells:
        for seed in seeds:
            found = sorted(Path(f"{log_root}/ckpt/{cell}_s{seed}").glob("model_best-*.pth"))
            if not found:
                print(f"!! 체크포인트가 없어 건너뛴다: {log_root}/ckpt/{cell}_s{seed}")
                continue
            model = _load(found[-1], vox_util, encoder_type, device)
            probs = []
            with torch.no_grad():
                for batch in loader:
                    _, _, logits, _, _ = model(batch["rgb_camXs"].to(device) - 0.5,
                                               batch["pix_T_cams"].to(device),
                                               batch["cam0_T_camXs"].to(device), vox_util)
                    probs.append(torch.softmax(logits, dim=1)[:, 1:2].clone())
            del model
            torch.cuda.empty_cache()
            for tau in taus:
                rows.setdefault((cell, tau), []).append(score(probs, tau))
            del probs
            torch.cuda.empty_cache()
            print(f"  {cell}_s{seed} ({found[-1].name})", flush=True)

    keys = ("free_miss", "fatal", "missed_obs", "iou_free", "f1@10cm")
    mean = {k: {c: np.array([np.mean([r[k] for r in rows[(c, t)]]) for t in taus])
                for c in cells if (c, taus[0]) in rows} for k in keys}
    std = {k: {c: np.array([np.std([r[k] for r in rows[(c, t)]]) for t in taus])
               for c in cells if (c, taus[0]) in rows} for k in keys}
    live = [c for c in cells if (c, taus[0]) in rows]
    n_seeds = len(rows[(live[0], taus[0])]) if live else 0

    print(f"\n=== τ 스윕 (val {len(val_samples)}프레임, 셀당 시드 {n_seeds}개) ===")
    print("평균 ± 시드 간 표준편차. **τ=0.5가 argmax이므로 기존 표와 일치해야 한다.**")
    head = ["cell", "τ"] + list(keys)
    widths = [max(9, max(_width(c) for c in live)), 5] + [17] * len(keys)
    print("  " + " ".join(_pad(h, w, ">") for h, w in zip(head, widths)))
    for cell in live:
        print("  " + "-" * (sum(widths) + len(widths) - 1))
        for i, tau in enumerate(taus):
            cols = [_pad(cell, widths[0], ">"), _pad(f"{tau:.2f}", widths[1], ">")]
            cols += [_pad(f"{mean[k][cell][i]:.4f}±{std[k][cell][i]:.4f}", w, ">")
                     for k, w in zip(keys, widths[2:])]
            print("  " + " ".join(cols))

    print(f"\n=== 같은 free_miss에서의 비교 (곡선 선형보간) ===")
    print("**이것이 판정 표다.** 차이가 대조군의 시드 σ 안이면 '겹친다'이고, 그때 τ=0.5의")
    print("차이는 representation 개선이 아니라 동작점 이동이다.")
    for metric in ("fatal", "missed_obs"):
        print(f"\n  [{metric}]  free_miss  " + " ".join(_pad(c, 10, ">") for c in live)
              + " " + _pad(f"{live[0]}−{live[-1]}", 14, ">"))
        for anchor in anchors:
            vals = []
            for cell in live:
                xs, ys = mean["free_miss"][cell], mean[metric][cell]
                if not xs.min() <= anchor <= xs.max():
                    vals.append(float("nan"))     # clamp를 값으로 오독하지 않게 한다
                else:
                    vals.append(float(np.interp(anchor, xs, ys)))
            diff = vals[0] - vals[-1]
            print("  " + _pad("", 11) + _pad(f"{anchor:.4f}", 10, ">") + " "
                  + " ".join(_pad(f"{v:.4f}", 10, ">") for v in vals)
                  + " " + _pad(f"{diff:+.4f}", 14, ">"))
        print("  " + _pad("", 11) + "  (nan = 그 칸의 곡선이 이 free_miss를 덮지 않는다)")

    print("\n=== τ를 val에서 최적화했을 때의 최선 (낙관 편향, 모든 칸에 동일하게 걸린다) ===")
    print("  " + _pad("cell", 9, ">") + _pad("max iou_free", 14, ">") + _pad("@τ", 6, ">")
          + _pad("max f1@10cm", 14, ">") + _pad("@τ", 6, ">"))
    for cell in live:
        i, j = int(np.argmax(mean["iou_free"][cell])), int(np.argmax(mean["f1@10cm"][cell]))
        print("  " + _pad(cell, 9, ">") + _pad(f"{mean['iou_free'][cell][i]:.4f}", 14, ">")
              + _pad(f"{taus[i]:.2f}", 6, ">")
              + _pad(f"{mean['f1@10cm'][cell][j]:.4f}", 14, ">")
              + _pad(f"{taus[j]:.2f}", 6, ">"))

    print("\n=== 재현성이 동작점에 의존하는가 (fatal의 시드 σ) ===")
    print("  σ가 작은 것이 '곡선이 평평한 자리라서'인지 확인한다 -- d(fatal)/dτ를 같이 본다.")
    print("  " + _pad("cell", 9, ">") + " ".join(_pad(f"τ={t:.2f}", 9, ">") for t in taus)
          + _pad("|dfatal/dτ|@0.5", 17, ">") + _pad("σ_τ_eff", 9, ">"))
    for cell in live:
        i = taus.index(0.50) if 0.50 in taus else len(taus) // 2
        lo, hi = max(0, i - 1), min(len(taus) - 1, i + 1)
        slope = abs((mean["fatal"][cell][hi] - mean["fatal"][cell][lo])
                    / (taus[hi] - taus[lo])) if hi != lo else float("nan")
        eff = std["fatal"][cell][i] / slope if slope else float("nan")
        print("  " + _pad(cell, 9, ">")
              + " ".join(_pad(f"{std['fatal'][cell][k]:.4f}", 9, ">")
                         for k in range(len(taus)))
              + _pad(f"{slope:.3f}", 17, ">") + _pad(f"{eff:.3f}", 9, ">"))
    print("\n  σ_τ_eff = σ_fatal / |d fatal/dτ| -- '시드를 바꾸는 것이 τ를 얼마나 흔드는 것과")
    print("  같은가'다. 이 값이 작을수록 두 런이 경계 위치에 더 정확히 합의한다.")


if __name__ == "__main__":
    Fire(main)
