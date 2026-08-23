"""런의 **수렴 상태**를 읽는다 -- 최종 성능표와는 다른 질문에 답한다.

왜 별도 도구인가: `summarize_repeats.py`는 "고정 epoch에서 얼마나 좋은가"를 집계한다.
그런데 이 프로젝트의 문제는 **val loss가 단조 상승하면서 지표는 평평하다**는 것이었고
(진단 §17, §26), 그건 최종 숫자만으로는 안 보인다. 여기서는 곡선의 모양을 읽는다.

**loss 값은 loss 종류를 넘어 비교할 수 없다.** soft-boundary loss에는 target 엔트로피가
상수로 들어 있어 0으로 내려가지 않는다(설계 문서 §5.4). 그래서 아래 표는 절대값 대신
**"최저점에서 얼마나 되올라갔나"** 를 주 숫자로 쓴다 -- 그것은 정의가 달라도 뜻이 같다.

지표(`iou_free`·`f1@10cm`·`fatal`·`free_miss`)는 정의가 완전히 같으므로 그대로 비교한다.

실행:
    python tools/report_convergence.py --log_root=runs/robot_bev_cv/loss_sweep/logs
    python tools/report_convergence.py --log_root=... --pattern='sb_*'
"""
import json
import sys
from pathlib import Path

from fire import Fire
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# 마지막 몇 epoch으로 기울기를 재는가. 10이면 40 epoch 런의 1/4이라 plateau 폭
# (실측 0.003~0.008, §24)에 묻히지 않고 추세가 나온다.
TAIL_EPOCHS = 10


def _series(acc, tag) -> dict:
    return {e.step: e.value for e in acc.Scalars(tag)} if tag in acc.Tags()["scalars"] else {}


def _tail_slope(series, tail=TAIL_EPOCHS):
    """마지막 `tail` epoch의 최소제곱 기울기 [단위/epoch]. **양수면 아직 발산 중이다.**"""
    steps = sorted(series)[-tail:]
    if len(steps) < 3:
        return float("nan")
    n = len(steps)
    mean_x = sum(steps) / n
    mean_y = sum(series[s] for s in steps) / n
    denom = sum((s - mean_x) ** 2 for s in steps)
    if denom == 0:
        return float("nan")
    return sum((s - mean_x) * (series[s] - mean_y) for s in steps) / denom


def read_run(run_dir) -> dict:
    run_dir = Path(run_dir)
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    val_loss = _series(acc, "val/loss_epoch")
    train_loss = _series(acc, "train/loss_epoch")
    iou = _series(acc, "val/iou_free_epoch")
    if not val_loss:
        return {"name": run_dir.name, "config": config, "epochs": 0}

    last = max(val_loss)
    min_epoch = min(val_loss, key=val_loss.get)
    f1 = _series(acc, "val/occupied_f1_10cm_epoch")
    best_f1_epoch = max(f1, key=f1.get) if f1 else None
    # `iou_free`의 최고점은 체크포인트 선택 기준과 같은 것이어야 한다.
    best_iou_epoch = max(iou, key=iou.get) if iou else None
    return {
        "name": run_dir.name,
        "config": config,
        "epochs": last,
        "val_loss_min": val_loss[min_epoch],
        "val_loss_min_epoch": min_epoch,
        "val_loss_last": val_loss[last],
        # **주 수렴 숫자.** 최저점 대비 상승분. 정의가 달라도 뜻이 같다.
        "rebound": val_loss[last] - val_loss[min_epoch],
        "rebound_pct": 100.0 * (val_loss[last] / val_loss[min_epoch] - 1.0),
        "val_slope": _tail_slope(val_loss),
        "train_loss_last": train_loss.get(last, float("nan")),
        "train_iou_last": _series(acc, "train/iou_free_epoch").get(last, float("nan")),
        "iou_last": iou.get(last, float("nan")),
        "iou_best": iou[best_iou_epoch] if best_iou_epoch else float("nan"),
        "iou_best_epoch": best_iou_epoch,
        "f1_10cm_last": f1.get(last, float("nan")),
        "f1_best": f1[best_f1_epoch] if best_f1_epoch else float("nan"),
        "f1_best_epoch": best_f1_epoch if best_f1_epoch else 0,
        "f1_20cm_last": _series(acc, "val/occupied_f1_20cm_epoch").get(last, float("nan")),
        "fatal_last": _series(acc, "val/fatal_rate_epoch").get(last, float("nan")),
        "miss_last": _series(acc, "val/free_miss_rate_epoch").get(last, float("nan")),
        "range_mae_last": _series(acc, "val/range_mae_epoch").get(last, float("nan")),
        # soft-boundary 전용. `kl_boundary`가 경계 항의 진짜 진행도다(설계 문서 §5.4).
        "kl_last": _series(acc, "val/kl_boundary_epoch").get(last, float("nan")),
        "kl_slope": _tail_slope(_series(acc, "val/kl_boundary_epoch")),
        "kl_train_last": _series(acc, "train/kl_boundary_epoch").get(last, float("nan")),
        # `L_range` 전용(§13). `arc_mae`는 dead zone 전의 순수 거리 오차 [m]이고,
        # `arc_bias`는 그 부호다 -- 양수면 자유공간 과대예측(= `fatal` 방향)이다.
        "arc_mae_last": _series(acc, "val/range_arc_mae_epoch").get(last, float("nan")),
        "arc_mae_train_last": _series(acc, "train/range_arc_mae_epoch").get(last, float("nan")),
        "arc_bias_last": _series(acc, "val/range_arc_bias_epoch").get(last, float("nan")),
        "share_range_last": _series(acc, "train/share_range_epoch").get(last, float("nan")),
    }


def label(run) -> str:
    """config에서 손잡이만 뽑아 한 칸에 담는다 -- 런 이름은 타임스탬프 때문에 길다."""
    config = run["config"]
    name = config.get("exp_name", run["name"].split("_")[0])
    if config.get("loss") != "soft_boundary":
        return f"{name} (CE)"
    target = config.get("soft_target", "?")
    sigma = config.get("sigma_m")
    suffix = f",σ{float(sigma):.3f}" if isinstance(sigma, (int, float)) else ""
    return (f"{name} (δ{float(config.get('delta_m', 0)):.2f}"
            f",λ{float(config.get('lambda_b', 0)):.2f},{target[:3]}{suffix})")


def _f(value, fmt=".4f", width=0):
    text = "n/a" if value is None else f"{value:{fmt}}"
    return f"{text:>{width}}" if width else text


def format_report(runs) -> str:
    done = [r for r in runs if r.get("epochs")]
    if not done:
        return "완료된 런이 없다."
    lines = ["", "=== 수렴 (val loss는 loss 종류를 넘어 비교하지 말 것 -- '되올림'만 비교한다) ===",
             f"{'런':>26s} {'ep':>3s} {'val최저':>9s} {'@ep':>4s} {'val마지막':>10s} "
             f"{'되올림':>9s} {'%':>7s} {'마지막10기울기':>14s} {'train_loss':>11s} {'train_iou':>10s}"]
    for r in done:
        lines.append(
            f"{label(r):>26s} {r['epochs']:3d} {_f(r['val_loss_min'], '.4f', 9)} "
            f"{r['val_loss_min_epoch']:4d} {_f(r['val_loss_last'], '.4f', 10)} "
            f"{_f(r['rebound'], '+.4f', 9)} {_f(r['rebound_pct'], '+.1f', 6)}% "
            f"{_f(r['val_slope'], '+.5f', 14)} {_f(r['train_loss_last'], '.4f', 11)} "
            f"{_f(r['train_iou_last'], '.4f', 10)}"
        )

    lines += ["", "=== 성능 (정의가 같으므로 그대로 비교한다. 모두 마지막 epoch) ===",
              f"{'런':>26s} {'iou_free':>9s} {'최고(ep)':>12s} {'f1@10cm':>9s} {'f1@20cm':>9s} "
              f"{'fatal':>8s} {'free_miss':>10s} {'range_mae':>10s}"]
    for r in done:
        best = f"{r['iou_best']:.4f}({r['iou_best_epoch']})"
        lines.append(
            f"{label(r):>26s} {_f(r['iou_last'], '.4f', 9)} {best:>12s} "
            f"{_f(r['f1_10cm_last'], '.4f', 9)} {_f(r['f1_20cm_last'], '.4f', 9)} "
            f"{_f(r['fatal_last'], '.4f', 8)} {_f(r['miss_last'], '.4f', 10)} "
            f"{_f(r['range_mae_last'], '.4f', 10)}"
        )

    soft = [r for r in done if r["config"].get("loss") == "soft_boundary"]
    if soft:
        # **배율(train/val KL 격차)이 §12.2가 정한 판정 숫자다.** 현재 28~42배이고, 보조항이
        # 프론티어를 옮겼다면 이것이 줄어야 한다. 스칼라 하나가 좋아지는 것보다 강한 근거다.
        lines += ["", "=== 경계 항 (kl_boundary가 진짜 진행도. loss_boundary는 하한이 있어 0으로 안 간다) ===",
                  f"{'런':>26s} {'val kl':>9s} {'train kl':>10s} {'배율':>7s} {'val kl 기울기':>13s}"]
        for r in soft:
            train_kl = r["kl_train_last"]
            ratio = (r["kl_last"] / train_kl) if train_kl and train_kl == train_kl and train_kl > 0 else float("nan")
            lines.append(f"{label(r):>26s} {_f(r['kl_last'], '.4f', 9)} "
                         f"{_f(r['kl_train_last'], '.4f', 10)} {_f(ratio, '.1f', 6)}x "
                         f"{_f(r['kl_slope'], '+.5f', 13)}")

    ranged = [r for r in soft if float(r["config"].get("lambda_r") or 0.0) > 0.0]
    if ranged:
        # `arc_mae`도 train/val 격차를 갖는다(구현 시점 실측 0.043 대 0.257, 6배). 그 격차가
        # 줄어드는지가 이 항이 암기를 실제로 억제했는지의 두 번째 독립 판독값이다.
        # `share_r`는 gradient 몫이 아니라 loss 기여 몫이지만, `λ_R`이 캘리브레이션 값에서
        # 벗어나 주 loss가 되어 버렸는지를 값싸게 확인해 준다(§13.3).
        lines += ["", "=== 보조항 L_range (arc_mae는 dead zone 전의 거리 오차 [m]) ===",
                  f"{'런':>26s} {'λ_R':>5s} {'δ_R':>5s} {'val arc_mae':>12s} "
                  f"{'train arc_mae':>14s} {'배율':>7s} {'arc_bias':>9s} {'share_r':>8s}"]
        for r in ranged:
            train_arc = r["arc_mae_train_last"]
            ratio = (r["arc_mae_last"] / train_arc) if train_arc and train_arc == train_arc and train_arc > 0 else float("nan")
            lines.append(
                f"{label(r):>26s} {float(r['config']['lambda_r']):5.2f} "
                f"{float(r['config'].get('delta_r_m', 0)):5.2f} "
                f"{_f(r['arc_mae_last'], '.4f', 12)} {_f(r['arc_mae_train_last'], '.4f', 14)} "
                f"{_f(ratio, '.1f', 6)}x {_f(r['arc_bias_last'], '+.4f', 9)} "
                f"{_f(r['share_range_last'], '.3f', 8)}"
            )
    return "\n".join(lines)


def knobs(run) -> dict:
    """손잡이만 뽑는다. `α`는 `σ/δ`이고 선형 target은 `α → ∞` 극한이다(설계 문서 §10)."""
    config = run["config"]
    if config.get("loss") != "soft_boundary":
        return {"shape": "-", "delta": None, "lambda_b": None,
                "lambda_r": 0.0, "delta_r": 0.0}
    alpha = config.get("sigma_alpha")
    sigma = config.get("sigma_m")
    if config.get("soft_target") == "linear":
        shape = "inf(lin)"
    elif alpha is not None:
        shape = f"{float(alpha):.3f}"
    elif sigma is not None:
        shape = f"σ{float(sigma):.4f}"
    else:
        shape = "?"
    return {"shape": shape, "delta": config.get("delta_m"),
            "lambda_b": config.get("lambda_b"),
            # `λ_R`과 `δ_R`이 없으면 §13 스윕의 런들이 표에서 구별되지 않는다 -- 이름만
            # 다르고 손잡이 칸이 전부 같게 찍힌다. 옛 런에는 config 키가 없으므로 0으로 읽는다.
            "lambda_r": float(config.get("lambda_r") or 0.0),
            "delta_r": float(config.get("delta_r_m") or 0.0)}


def format_frontier(runs) -> str:
    """**교환 곡선 표.** `f1@10cm`(성능)과 되올림(수렴)의 파레토 관계를 직접 보여준다.

    왜 필요한가: 손잡이(`δ`·`α`·`λ_B`)가 프론티어를 **옮기는지** 아니면 그 위를 **미끄러지는지**
    가 이 실험의 핵심 질문이고, 두 열을 나란히 정렬하지 않으면 그것이 보이지 않는다.
    지배되는 점(되올림이 더 나쁜데 f1도 더 낮은 점)은 표시한다 -- 그 점의 손잡이 조합은
    더 볼 필요가 없다는 뜻이다.
    """
    done = [r for r in runs if r.get("epochs")]
    if not done:
        return ""
    rows = sorted(done, key=lambda r: -r["f1_10cm_last"])
    lines = ["", "=== 교환 곡선: 성능(f1@10cm) 대 수렴(되올림) ===",
             f"{'런':>16s} {'α':>9s} {'δ':>5s} {'λ_B':>5s} {'λ_R':>5s} {'δ_R':>5s} | "
             f"{'되올림':>8s} {'f1@10':>7s} {'최고(ep)':>12s} {'fatal':>7s} | {'판정':>8s}"]
    for run in rows:
        k = knobs(run)
        # 되올림이 더 작으면서 f1이 더 높은 다른 런이 있으면 이 점은 지배된다.
        dominated = any(o["rebound_pct"] < run["rebound_pct"]
                        and o["f1_10cm_last"] > run["f1_10cm_last"] for o in rows)
        name = run["config"].get("exp_name", run["name"].split("_")[0])
        lines.append(
            f"{name:>16s} {k['shape']:>9s} "
            f"{('%.2f' % k['delta']) if k['delta'] is not None else '-':>5s} "
            f"{('%.2f' % k['lambda_b']) if k['lambda_b'] is not None else '-':>5s} "
            f"{('%.2f' % k['lambda_r']) if k['lambda_r'] else '-':>5s} "
            f"{('%.2f' % k['delta_r']) if k['lambda_r'] else '-':>5s} | "
            f"{run['rebound_pct']:+7.1f}% {run['f1_10cm_last']:7.4f} "
            f"{run['f1_best']:.4f}({run['f1_best_epoch']:2d}) {run['fatal_last']:7.4f} | "
            f"{'지배됨' if dominated else '프론티어':>8s}"
        )
    return "\n".join(lines)


def main(log_root, pattern="*"):
    log_root = Path(log_root)
    run_dirs = sorted(d for d in log_root.glob(pattern) if d.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"런 폴더가 없다: {log_root}/{pattern}")
    runs = [read_run(d) for d in run_dirs]
    print(format_report(runs))
    print(format_frontier(runs))


if __name__ == "__main__":
    Fire(main)
