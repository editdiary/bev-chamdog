"""`α` 스윕에서 **각 loss 항이 수렴하는가**를 읽는다 (2026-08-27).

## 무엇을 묻는가

`α = σ/δ`는 soft target의 모양만 정한다 -- 대역 폭은 `δ`가 정하고 둘은 직교한다(§6.2).
α가 작으면 target이 hard에 가까워지고(α=0.25에서 최근접 셀 0.909), 크면 평평해진다
(선형 극한에서 0.667). **"경계가 어디인지 모른다"를 얼마나 강하게 주장하는가**의 손잡이다.

`y4_s0`(α=0.5)에서 val 총 loss가 ep3에 최저 0.4956을 찍고 ep39에 0.5408로 오르는데,
가중 기여로 쪼개면 **경계 항 +0.0952 / 나머지 셋 전부 음수**다. 즉 발산의 유일한 원인이
경계 항이고, α는 그 항의 target confidence를 정하는 유일한 손잡이다.

그래서 이 도구가 묻는 것은 순위가 아니라 **곡선의 모양**이다: train이 내려가는 동안 val이
같이 내려가는가, 아니면 조기 최저를 지나 다시 오르는가, 그리고 그 상승폭이 α에 따라
어떻게 변하는가. `L_B` 말고 `L_F`·`L_N`·`L_range`도 같이 본다 -- 한 항을 평평하게 만드느라
다른 항이 나빠지면 그건 교환이지 개선이 아니다.

## 읽을 때의 함정 둘

1. **`loss_boundary`는 α끼리 비교하면 안 된다.** target 엔트로피 `H̄`가 상수로 들어 있고
   그 값이 α마다 다르다(평평할수록 크다). 비교 가능한 것은 `kl_boundary = loss_boundary − H̄`
   이고, 이 도구는 `H̄`를 따로 찍어 그 차이를 눈에 보이게 한다.
2. **`kl_boundary`조차 완전히 공정하지는 않다** -- 평평한 target은 본질적으로 맞히기 쉽다.
   그래서 판정은 **target에 의존하지 않는 지표**(`iou_free`·`fatal_rate`·`f1@10cm`)와
   같이 읽어야 하고, 시드가 1개이므로 그쪽의 작은 차이는 읽으면 안 된다
   (실측 σ_run: `f1@10cm` 0.0037, 설계 문서 §13.6.1).

실행: python tools/report_alpha_convergence.py --log_root=runs/alpha_y4
"""
import json
import sys
from pathlib import Path

from fire import Fire
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# 총 loss는 `0.5·L_F + 0.5·L_N + λ_B·L_B + λ_R·L_range`다(`soft_boundary.py:206,224`).
# **가중치를 곱해야 항끼리 기여를 비교할 수 있다** -- 원값은 각자 자기 집합 위의 평균이다.
_TERMS = (
    ("boundary", "loss_boundary_epoch", "lambda_b"),
    ("free", "loss_free_epoch", 0.5),
    ("not_free", "loss_not_free_epoch", 0.5),
    ("range", "loss_range_epoch", "lambda_r"),
)


def _width(text) -> int:
    from unicodedata import east_asian_width
    return sum(2 if east_asian_width(ch) in "WF" else 1 for ch in str(text))


def _pad(text, width, align=">") -> str:
    fill = max(0, width - _width(text))
    return f"{text}{' ' * fill}" if align == "<" else f"{' ' * fill}{text}"


def _row(cells, widths):
    return "  " + " ".join(_pad(c, w) for c, w in zip(cells, widths))


def _load(run_dir):
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags()["scalars"])
    curves = {t: [s.value for s in ea.Scalars(t)] for t in available}
    config = json.loads((run_dir / "config.json").read_text())
    return curves, config.get("config", config)


def _shape(values):
    """`(최저 epoch, 최저값, 마지막값, 상승폭)`. 상승폭 > 0이면 조기 최저 후 발산이다."""
    lo = min(range(len(values)), key=lambda i: values[i])
    return lo, values[lo], values[-1], values[-1] - values[lo]


def _label(config, varying=()):
    """**변하는 손잡이만 이름에 넣는다.** 한 표에 α 스윕과 κ 스윕이 섞여도 읽히게 하려는 것이고,
    안 변하는 축을 다 찍으면 폭만 먹고 대조가 안 보인다."""
    shape = ("linear" if config.get("soft_target") == "linear"
             else f"α{float(config['sigma_alpha']):g}")
    bits = []
    if "shape" in varying:
        bits.append(shape)
    if "delta_m" in varying:
        bits.append(f"δ{float(config['delta_m']):.2f}")
    if "band_kappa" in varying:
        bits.append(f"κ{float(config.get('band_kappa', 1.0)):.2f}")
    if "label_eps" in varying:
        bits.append(f"ε{float(config.get('label_eps', 0.0)):.2f}")
    return " ".join(bits) if bits else shape


def _varying(runs):
    """런들 사이에서 실제로 값이 다른 축의 집합. 하나뿐이면 그것만 이름에 남는다."""
    out = set()
    shapes = {(r[2].get("soft_target"), r[2].get("sigma_alpha")) for r in runs}
    if len(shapes) > 1:
        out.add("shape")
    for key, default in (("delta_m", None), ("band_kappa", 1.0), ("label_eps", 0.0)):
        if len({float(r[2].get(key, default)) for r in runs}) > 1:
            out.add(key)
    return out or {"shape"}


def _sort_key(config):
    """평평해지는 순서로 세운다 -- α는 선형이 극한이라 맨 뒤, κ는 낮을수록 평평하다."""
    a = (float("inf") if config.get("soft_target") == "linear"
         else float(config["sigma_alpha"]))
    return (float(config["delta_m"]), float(config.get("label_eps", 0.0)),
            -float(config.get("band_kappa", 1.0)), a)


def main(log_root="runs/alpha_y4", reference="runs/height_bins/logs/y4_s0"):
    # `--log_root`는 스윕의 `OUT_ROOT`(`runs/alpha_y4`)로도, 로그 디렉터리
    # (`runs/alpha_y4/logs`)로도 줄 수 있게 한다 -- 스윕 스크립트가 안내하는 것은 전자다.
    root = Path(log_root)
    if (root / "logs").is_dir():
        root = root / "logs"
    runs = []
    for d in sorted(root.glob("*/")):
        if not (d / "config.json").exists():
            continue
        curves, config = _load(d)
        # **진행 중인 런은 건너뛴다.** tfevents가 아직 flush되기 전이면 스칼라가 0개로
        # 읽히는데, 그대로 두면 KeyError로 죽어서 완료된 런의 집계까지 못 본다.
        if "val/loss_epoch" not in curves:
            print(f"  [건너뜀] {d.name}: 기록된 epoch이 없다 (진행 중이거나 실패)")
            continue
        runs.append((d.name, curves, config))
    if reference and Path(reference).exists():
        curves, config = _load(Path(reference))
        runs.append((f"{Path(reference).name}(참조)", curves, config))
    if not runs:
        print(f"런을 찾지 못했다: {log_root}")
        return
    runs.sort(key=lambda r: _sort_key(r[2]))
    varying = _varying(runs)

    print(f"=== `α` 스윕 수렴 판독 ({log_root}, 시드 {runs[0][2].get('seed')}, "
          f"Y={runs[0][2].get('height_bins')}) ===")

    print("\n[0] 상수 하한 `H̄` -- **`loss_boundary`를 α끼리 직접 비교하면 안 되는 이유**")
    print("  `L_B = H̄ + KL`이고 `H̄`는 target만으로 정해지는 상수라 gradient가 0이다.")
    print("  평평한 target일수록 `H̄`가 크므로 `loss_boundary`도 그만큼 그냥 커진다.")
    w = (20, 12, 14, 14, 16)
    print(_row(("config", "H̄", "λ_B·H̄", "val L_B(끝)", "그중 H̄ 몫"), w))
    for name, c, config in runs:
        h = c["val/entropy_boundary_epoch"][-1]
        lb = c["val/loss_boundary_epoch"][-1]
        # **`δ → 0`이면 `Ω_B`가 비고 `_masked_mean`이 0을 준다** -- 경계 구조 제거 대조군이
        # 정확히 그 경우다. 항이 없는 것이지 0으로 수렴한 것이 아니므로 따로 표시한다.
        share = f"{h / lb * 100:.1f} %" if lb > 1e-9 else "Ω_B 없음"
        print(_row((_label(config, varying), f"{h:.4f}", f"{float(config['lambda_b']) * h:.4f}",
                    f"{lb:.4f}", share), w))

    print("\n[1] **각 항이 수렴하는가** -- val 곡선의 모양")
    print("  `최저@ep`가 이르고 `상승폭`이 크면 그 항의 target이 일반화되지 않는다는 뜻이다.")
    print("  train은 대조군이다 -- 거기서도 안 내려가면 학습 자체가 안 된 것이다.")
    for term, tag, weight in (("총 loss", "loss_epoch", 1.0),) + _TERMS:
        # 경계 항은 `H̄`를 뺀 `kl_boundary`로 본다 -- 상수는 수렴 판정에 뜻이 없다.
        use_tag = "kl_boundary_epoch" if term == "boundary" else tag
        shown = "boundary(KL)" if term == "boundary" else term
        print(f"\n  -- {shown}")
        w = (20, 12, 12, 10, 12, 12, 12)
        print(_row(("config", "train 시작", "train 끝", "최저@ep", "val 최저",
                    "val 끝", "상승폭"), w))
        for name, c, config in runs:
            tr = c.get(f"train/{use_tag}")
            va = c.get(f"val/{use_tag}")
            if va is None:
                continue
            lo, lo_v, end, rise = _shape(va)
            mark = "" if rise <= 0 else ("  <-발산" if rise > 0.02 else "  <-미세상승")
            print(_row((_label(config, varying),
                        f"{tr[0]:.4f}" if tr else "-", f"{tr[-1]:.4f}" if tr else "-",
                        f"{lo}", f"{lo_v:.4f}", f"{end:.4f}", f"{rise:+.4f}{mark}"), w))

    print("\n[2] val 총 loss의 발산을 **항별 가중 기여**로 쪼갠다")
    print("  총 loss = 0.5·L_F + 0.5·L_N + λ_B·L_B + λ_R·L_range. 최저 epoch 대비 변화를")
    print("  가중치까지 곱해 더하면 총합의 변화와 일치한다 -- 어느 항이 범인인지가 바로 보인다.")
    w = (20, 10, 12, 12, 12, 12, 12)
    print(_row(("config", "최저@ep", "총 상승", "boundary", "free", "not_free", "range"), w))
    for name, c, config in runs:
        total = c["val/loss_epoch"]
        lo, _, _, rise = _shape(total)
        cells = [_label(config, varying), f"{lo}", f"{rise:+.4f}"]
        for term, tag, weight in _TERMS:
            k = float(config[weight]) if isinstance(weight, str) else weight
            v = c[f"val/{tag}"]
            cells.append(f"{k * (v[-1] - v[lo]):+.4f}")
        print(_row(cells, w))

    print("\n[3] target에 의존하지 않는 품질 지표 -- **판정은 여기서 한다**")
    print("  **시드 1개이므로 작은 차이는 읽으면 안 된다.** 실측 σ_run: `f1@10cm` 0.0037,")
    print("  `iou_free`도 같은 규모다(설계 문서 §13.6.1). 여기서 읽을 수 있는 것은 '망가졌는가'다.")
    w = (20, 14, 12, 12, 12, 12, 12)
    print(_row(("config", "iou_free 최고", "끝", "fatal", "free_miss", "f1@10cm",
                "arc_mae"), w))
    for name, c, config in runs:
        iou = c["val/iou_free_epoch"]
        f1 = c.get("val/occupied_f1_10cm_epoch")
        print(_row((_label(config, varying), f"{max(iou):.4f}@{iou.index(max(iou))}", f"{iou[-1]:.4f}",
                    f"{c['val/fatal_rate_epoch'][-1]:.4f}",
                    f"{c['val/free_miss_rate_epoch'][-1]:.4f}",
                    f"{f1[-1]:.4f}" if f1 else "-",
                    f"{c['val/range_arc_mae_epoch'][-1]:.4f}"), w))

    print("\n판독:")
    print("  [1]의 boundary(KL) 상승폭이 α에 따라 단조로 줄면, **경계 target이 너무 확신에")
    print("  차 있어서 발산한다**는 가설이 지지된다. 그때 [3]이 같이 나빠지지 않아야 개선이고,")
    print("  같이 나빠지면 '경계를 덜 배우니 val loss도 덜 오른다'는 동어반복이다.")
    print("  [2]에서 boundary 말고 다른 항이 커지기 시작하면 그건 교환이다.")


if __name__ == "__main__":
    Fire(main)
