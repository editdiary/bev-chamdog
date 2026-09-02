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
from projects.datasets.simplebev_vox import (  # noqa: E402
    height_config_for_ckpt_dirs,
    vox_dims,
)
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.pixel_grid import convention_for_run_dirs  # noqa: E402
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402

# τ=0.5는 `argmax`와 같으므로 **기존 표의 값이 이 스윕의 한 점으로 재현되어야 한다** --
# 안 되면 평가 경로가 갈린 것이다. 양 끝(0.2/0.8)은 곡선의 모양을 보기 위한 것이고 운용
# 후보는 아니다.
DEFAULT_TAUS = (0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80)

# 같은 `free_miss`에서 비교할 지점. 모든 칸의 곡선이 **동시에 덮는** 구간 안이어야 한다 --
# 밖이면 `np.interp`가 끝값으로 **고정(clamp)**되어 비교가 조용히 무의미해진다.
#
# **기본은 `None`(자동 유도)이다.** 예전에는 `(0.085, ..., 0.104)`가 하드코딩돼 있었는데
# 그 값은 `runs/ablation`(Y=1, 옛 config)의 곡선에서 고른 것이다. 기하나 loss가 바뀌면
# 곡선이 통째로 옮겨 가므로 그 앵커가 구간 밖으로 나가고, 그러면 판정 표가 **전부 nan**이
# 되거나(가드가 있는 지금) 조용히 clamp된 값을 비교한다(가드가 없던 옛 코드). 실제로
# 자동 유도가 필요한 상황이 이번에 생겼다 -- Y=4에서 `free_miss` 대역이 달라졌다.
DEFAULT_ANCHORS = None

# 자동 유도할 때 몇 점을 찍나. 겹치는 구간을 균등 분할한다.
N_AUTO_ANCHORS = 5


def _auto_anchors(mean, live, n=N_AUTO_ANCHORS):
    """모든 칸의 `free_miss` 곡선이 동시에 덮는 구간을 균등 분할한다.

    `lo`는 각 칸 최솟값의 **최댓값**, `hi`는 각 칸 최댓값의 **최솟값**이다 -- 그래야 어느
    칸에서도 외삽이 아니다. 구간이 비면 빈 튜플을 주고 호출부가 그 사실을 찍는다.
    """
    curves = [mean["free_miss"][c] for c in live]
    lo = max(float(np.min(x)) for x in curves)
    hi = min(float(np.max(x)) for x in curves)
    if not lo < hi:
        return ()
    # 양 끝은 τ 격자의 끝점이라 곡선이 가장 덜 믿음직한 자리다. 1 %씩 안으로 넣는다.
    pad = 0.01 * (hi - lo)
    return tuple(np.linspace(lo + pad, hi - pad, n))


def _interp_monotone(x, xs, ys):
    """`np.interp`의 전제(`xs` 증가)를 **검사한 뒤** 보간한다.

    `np.interp`는 `xs`가 증가하지 않으면 경고 없이 틀린 값을 준다. `free_miss`는 τ에 대해
    단조 증가해야 하지만(문턱을 올리면 free 예측이 줄어든다) 시드 평균 곡선이 잡음으로
    한 칸 뒤집히는 일이 실제로 가능하므로, 뒤집히면 값을 주지 않고 nan을 준다.
    """
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if np.any(np.diff(xs) < 0):
        return float("nan")
    if not xs.min() <= x <= xs.max():
        return float("nan")     # clamp를 값으로 오독하지 않게 한다
    return float(np.interp(x, xs, ys))


def _csv(value):
    """`--cells=a,b`를 **Fire가 tuple로 파싱한다** -- `str(value).split(",")`로 받으면
    `"('a', 'b')"`를 쪼개게 되어 조용히 깨진 이름이 나온다(`seeds`에서는 `int('(0')`으로
    예외가 났다). 다른 도구들은 이미 이 helper를 쓰고 있었는데 여기만 빠져 있었다."""
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _width(text) -> int:
    """한글은 터미널에서 두 칸을 쓴다 -- f-string의 폭 지정은 글자 수를 세므로 표가 밀린다."""
    from unicodedata import east_asian_width
    return sum(2 if east_asian_width(ch) in "WF" else 1 for ch in str(text))


def _pad(text, width, align="<") -> str:
    fill = max(0, width - _width(text))
    return f"{text}{' ' * fill}" if align == "<" else f"{' ' * fill}{text}"


def _load(ckpt, vox, encoder_type, device):
    # `Y`는 vox_util이 이미 알고 있다 -- 리터럴을 다시 쓰면 둘이 어긋날 수 있다.
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
    taus=DEFAULT_TAUS,
    anchors=DEFAULT_ANCHORS,
    train_sequences="raws2,raws3,rawos1,rawos2,rawos4",
    val_sequences="raws1,rawos3",
    dataset_root=DEFAULT_DATASET_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    encoder_type="res101",
    batch_size=8,
    num_workers=8,
    # 셀x시드xτ의 **모든 지표**를 그대로 떨어뜨릴 CSV 경로. 표는 이것의 요약이다.
    csv_out=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cells = tuple(_csv(cells))
    seeds = tuple(int(s) for s in _csv(seeds))
    taus = tuple(taus)
    anchors = None if anchors is None else tuple(anchors)

    root = Path(dataset_root)
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
        [Path(log_root) / "ckpt" / f"{cell}_s{seed}" for cell in cells for seed in seeds])
    vox_util = build_double_sphere_vox_util(GRID_SPEC, dataset.cameras, device=device,
                                           pixel_convention=convention, pixel_offset=offset,
                                           height_bins=height["height_bins"],
                                           height_min_m=height["height_min_m"],
                                           height_max_m=height["height_max_m"])
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
    # **어느 시드의 행인지 남긴다.** `rows[(cell, tau)]`의 순서가 곧 이 목록의 순서다 --
    # 원시 CSV를 쓸 때 그 짝이 없으면 시드를 되찾을 수 없다.
    seed_order = {cell: [] for cell in cells}
    for cell in cells:
        for seed in seeds:
            found = sorted(Path(f"{log_root}/ckpt/{cell}_s{seed}").glob("model_best-*.pth"))
            if not found:
                print(f"!! 체크포인트가 없어 건너뛴다: {log_root}/ckpt/{cell}_s{seed}")
                continue
            seed_order[cell].append(seed)
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

    # === 원시 행을 CSV로 남긴다 =======================================================
    #
    # 아래 표들은 전부 이 행들의 요약(평균·표준편차·보간)이다. **표를 다시 만들 수 있어야
    # 하고, 표에 안 실린 지표(`range_mae`)도 남아야 한다** -- 그래서 `score()`가 낸 dict를
    # 통째로 쓴다. 셀·시드·τ가 키다.
    if csv_out:
        import csv as _csv_module
        Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
        all_keys = sorted({k for v in rows.values() for r in v for k in r})
        with open(csv_out, "w", newline="") as fh:
            writer = _csv_module.writer(fh)
            writer.writerow(("cell", "seed", "tau", *all_keys))
            for cell in cells:
                for tau_value in taus:
                    for seed, row in zip(seed_order[cell], rows.get((cell, tau_value), [])):
                        writer.writerow((cell, seed, tau_value,
                                         *(repr(float(row[k])) for k in all_keys)))
        print(f"  원시 행 -> {csv_out}")
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

    if anchors is None:
        anchors = _auto_anchors(mean, live)
        if anchors:
            print(f"\n  [앵커 자동 유도] 모든 칸이 덮는 free_miss 구간"
                  f" [{min(anchors):.4f}, {max(anchors):.4f}]에서 {len(anchors)}점.")
        else:
            print("\n  !! 칸들의 free_miss 구간이 겹치지 않는다 -- τ 격자를 넓혀야 한다"
                  " (`--taus=0.1,...,0.9`).")

    print(f"\n=== 같은 free_miss에서의 비교 (곡선 선형보간) ===")
    print("**이것이 판정 표다.** 차이가 대조군의 시드 σ 안이면 '겹친다'이고, 그때 τ=0.5의")
    print("차이는 representation 개선이 아니라 동작점 이동이다.")
    for metric in ("fatal", "missed_obs"):
        print(f"\n  [{metric}]  free_miss  " + " ".join(_pad(c, 10, ">") for c in live)
              + " " + _pad(f"{live[0]}−{live[-1]}", 14, ">"))
        for anchor in anchors:
            vals = []
            for cell in live:
                vals.append(_interp_monotone(anchor, mean["free_miss"][cell],
                                             mean[metric][cell]))
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

    # === 동작점 재현성: 시드마다 목표 free_miss를 맞추려면 τ가 얼마나 달라지나 ==========
    #
    # 위의 `σ_τ_eff`는 **평균 곡선의 기울기로 나눈 유도량**이라 "시드를 바꾸는 것이 τ를
    # 얼마나 흔드는 것과 같은가"의 **환산값**이다. 여기서는 그것을 직접 잰다 -- 시드마다
    # 자기 곡선에서 목표 `free_miss`를 만족하는 τ를 역보간하고, 그 τ들의 산포를 본다.
    # 두 숫자는 다른 것이다: 환산값은 곡선이 평평하면 커지고, 직접 잰 값은 곡선이 평평해도
    # 시드들이 같은 자리에 있으면 작다.
    #
    # **같은 자리에서 fatal도 다시 잰다.** 동작점을 맞춘 뒤에도 남는 fatal 차이만이
    # representation 차이이고, τ=0.5의 차이는 그렇지 않다(§16.2).
    if anchors:
        print("\n=== 목표 free_miss를 맞추는 τ의 시드 간 산포 (동작점 재현성) ===")
        print("  시드마다 자기 곡선에서 τ*(q)를 역보간한다. **σ(τ*)가 작을수록 같은 안전")
        print("  동작점을 만들기 위해 필요한 문턱이 재학습에 덜 흔들린다.**")
        print("  " + _pad("q(free_miss)", 14, ">")
              + " ".join(_pad(f"{c} τ*", 16, ">") for c in live)
              + " ".join(_pad(f"{c} fatal@τ*", 18, ">") for c in live))
        for anchor in anchors:
            tau_cols, fatal_cols = [], []
            for cell in live:
                n = len(rows[(cell, taus[0])])
                tstars, fstars = [], []
                for j in range(n):
                    fm = [rows[(cell, t)][j]["free_miss"] for t in taus]
                    ft = [rows[(cell, t)][j]["fatal"] for t in taus]
                    tstars.append(_interp_monotone(anchor, fm, taus))
                    fstars.append(_interp_monotone(anchor, fm, ft))
                tstars = [v for v in tstars if np.isfinite(v)]
                fstars = [v for v in fstars if np.isfinite(v)]
                # **모든 시드가 이 q를 덮어야 비교가 된다.** 일부만 덮으면 표본이 달라지고,
                # 그러면 σ가 작아진 것이 "안정적"인지 "표본이 줄어든 것"인지 갈리지 않는다.
                if len(tstars) < n or len(fstars) < n:
                    tau_cols.append("n/a"); fatal_cols.append("n/a"); continue
                tau_cols.append(f"{np.mean(tstars):.3f}±{np.std(tstars, ddof=1):.3f}"
                                if n > 1 else f"{np.mean(tstars):.3f}")
                fatal_cols.append(f"{np.mean(fstars):.4f}±{np.std(fstars, ddof=1):.4f}"
                                  if n > 1 else f"{np.mean(fstars):.4f}")
            print("  " + _pad(f"{anchor:.4f}", 14, ">")
                  + " ".join(_pad(c, 16, ">") for c in tau_cols)
                  + " ".join(_pad(c, 18, ">") for c in fatal_cols))
        print("  (n/a = 시드 중 일부의 곡선이 이 free_miss를 덮지 않는다)")


if __name__ == "__main__":
    Fire(main)
