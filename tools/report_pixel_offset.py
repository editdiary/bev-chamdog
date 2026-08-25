"""`configs/sweep_pixel_offset.sh`의 집계 -- 진단 문서 §18.3의 잔여 오프셋을 판정한다.

## 왜 `report_ablation.py`를 그대로 쓰지 않나

두 가지가 안 맞는다. (1) 셀 순서를 알파벳으로 잡아 `off_0.00`이 `off_m1.00` 앞에 오는데,
스윕은 **오프셋 순서로 읽어야** 곡선의 모양이 보인다. (2) config 표가 loss 손잡이만 찍어서
"무엇을 바꾼 런인지 표에서 읽을 수 없다" -- `ablation_loss.sh`가 새로 돌게 된 바로 그 이유다.

그래서 **판정 방법론(노이즈 바닥, 2σ 문턱, 지표 목록)은 `report_ablation.py`에서 그대로
import하고** 표의 축만 바꾼다. §15·§16의 숫자를 낸 도구는 건드리지 않는다.

## 무엇을 판정하나

두 축이 있고 섞으면 안 된다.

1. **규약 수정의 효과** -- `legacy`(대조군) 대 `off_0.00`. 여기서 바뀌는 것은 정규화 규약
   하나이고, 그건 정답이 있는 correctness 문제다(§18.3.1).
2. **잔여 오프셋의 효과** -- `pixel_center` 셀들끼리. `off_0.00`을 기준으로 본다.
   유도가 예측하는 최적 구간은 `[-0.5375, -0.0375]` 특징픽셀이고(§18.3.3), 그래서
   **뜻이 있는 대조는 `-0.5` 대 `0`이다.** 최적점이 `-1.0`이나 `+0.5`에서 나오면 유도가
   틀렸다는 뜻이므로 그 결과는 그대로 채택하지 않는다.

**config가 기하 말고도 다르면 즉시 멈춘다** -- 그러면 스윕이 재는 것이 기하가 아니게 된다.

실행: python tools/report_pixel_offset.py
"""
import sys
from pathlib import Path

from fire import Fire

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from tools.report_ablation import (  # noqa: E402
    DEFAULT_SIGMA_RUNS,
    QUALITY,
    STABILITY,
    Z_THRESHOLD,
    _mean_std,
    _pad,
    _pooled_std,
    read_run,
    sigma_run,
)

# 기하 외에 셀마다 달라지면 안 되는 config 키. 하나라도 갈리면 스윕이 무의미하다.
_MUST_MATCH = ("loss", "soft_target", "delta_m", "lambda_b", "sigma_alpha", "lambda_r",
               "delta_r_m", "delta_r_over_m", "huber_beta_m", "formulation", "encoder_type",
               "augment", "num_epochs", "train_sequences", "val_sequences", "lr",
               "weight_decay", "init_checkpoint", "flip_augment", "freeze_encoder")


def _geometry(config) -> tuple:
    return (config.get("pixel_convention", "legacy_index"),
            float(config.get("pixel_offset", 0.0) or 0.0))


def main(log_root="runs/pixel_offset/logs", pattern="*", sigma_from=DEFAULT_SIGMA_RUNS):
    root = Path(log_root)
    if not root.exists():
        raise FileNotFoundError(f"로그 폴더가 없다: {root}")
    runs = [r for r in (read_run(d) for d in sorted(root.glob(pattern)) if d.is_dir()) if r]

    cells, pending = {}, []
    for run in runs:
        if run.get("incomplete"):
            pending.append(f"{run['cell']} ({run['incomplete']})")
            continue
        cells.setdefault(run["cell"], []).append(run)
    if not cells:
        print("완료된 런이 없다." + (f" 진행 중: {', '.join(pending)}" if pending else ""))
        return

    # **기하 말고 다른 것이 갈렸으면 여기서 멈춘다.**
    reference = next(iter(cells.values()))[0]["config"]
    for cell, members in cells.items():
        for run in members:
            differing = {k: (reference.get(k), run["config"].get(k))
                         for k in _MUST_MATCH if reference.get(k) != run["config"].get(k)}
            if differing:
                raise SystemExit(
                    f"config가 기하 외에도 다르다 -- 스윕이 기하를 재지 않는다.\n"
                    f"  {cell}_s{run['seed']}: {differing}")

    # 대조군(legacy)을 맨 앞에, 나머지는 **오프셋 순서**로.
    def sort_key(cell):
        convention, offset = _geometry(cells[cell][0]["config"])
        return (0 if convention == "legacy_index" else 1, offset)

    order = sorted(cells, key=sort_key)
    legacy = [c for c in order if _geometry(cells[c][0]["config"])[0] == "legacy_index"]
    fixed = [c for c in order if c not in legacy]
    # 잔여 오프셋의 기준점은 `pixel_center` + offset 0 이다.
    zero = next((c for c in fixed if _geometry(cells[c][0]["config"])[1] == 0.0), None)

    lines = []
    if pending:
        lines += ["", f"!! 아직 도는 중이라 표에서 제외했다: {', '.join(pending)}"]

    lines += ["", "=== 실행한 설정 (전부 config.json에서 읽었다) ===",
              "기하 외의 손잡이는 전부 같은 것을 확인했다 (다르면 이 스크립트가 멈춘다).",
              f"  {'셀':<12s} {'규약':<14s} {'오프셋[특징px]':>15s} {'[native px]':>12s}  시드"]
    for cell in order:
        convention, offset = _geometry(cells[cell][0]["config"])
        seeds = ",".join(sorted(r["seed"] for r in cells[cell]))
        note = "  <- 대조군 (규약 불일치 그대로)" if convention == "legacy_index" else ""
        lines.append(f"  {cell:<12s} {convention:<14s} {offset:>15.2f} {offset * 20:>12.1f}"
                     f"  s={seeds}{note}")
    lines.append(f"  공통: loss={reference.get('loss')} λ_B={reference.get('lambda_b')}"
                 f" λ_R={reference.get('lambda_r')} δ_R={reference.get('delta_r_m')}"
                 f" {reference.get('num_epochs')} epoch {reference.get('encoder_type')}")

    sigma_ref = sigma_run(sigma_from) if sigma_from else {}
    name_w = max(len(n) for _, n, _ in QUALITY + STABILITY) + 2

    def digits_for(name):
        return 1 if "%" in name or name.endswith("ep") else 4

    def series(name):
        return {c: [r["values"][name] for r in cells[c] if name in r["values"]] for c in order}

    lines += ["", "=== 셀별 값 | 평균 ± 표준편차 ===",
              _pad("지표", name_w) + "   " + "".join(_pad(c, 18, ">") for c in order)]
    for _, name, direction in QUALITY + STABILITY:
        values = series(name)
        arrow = {1: "↑", -1: "↓", 0: "·"}[direction]
        row = _pad(name, name_w) + _pad(arrow, 3, ">")
        for cell in order:
            xs = values[cell]
            if not xs:
                row += _pad("-", 18, ">")
                continue
            mean, sd, _ = _mean_std(xs)
            d = digits_for(name)
            row += _pad(f"{mean:.{d}f}±{sd:.{d}f}" if len(xs) >= 2 else f"{mean:.{d}f}(n=1)",
                        18, ">")
        lines.append(row)

    df = sum(max(0, len(v) - 1) for v in cells.values())
    source = ([f"셀 내부 pooled 표준편차(df={df})"] if df else []) + \
             ([f"σ_run 실측(n={sigma_ref.get('__n__', '?')})"] if sigma_ref else [])
    noise_note = (f"노이즈 바닥은 {' 와 '.join(source) or '없음'} 중 **큰 쪽**을 쓴다. "
                  f"그 {Z_THRESHOLD}배를 넘으면 판정하고, 아니면 `노이즈`다. "
                  "괄호 안은 노이즈 바닥의 몇 배(σ)다.\n"
                  "**σ_run 참조 런은 λ_R=0이므로 보수적 하한으로만 쓴다**(설계 문서 §15.4).")

    comparisons = []
    if legacy and zero:
        comparisons.append((f"규약 수정의 효과 -- {legacy[0]}(대조군) → {zero}",
                            [(legacy[0], zero)]))
    others = [c for c in fixed if c != zero]
    if zero and others:
        comparisons.append((f"잔여 오프셋의 효과 -- {zero} 기준",
                            [(zero, c) for c in others]))

    for title, pairs in comparisons:
        lines += ["", f"=== {title} ===", noise_note,
                  _pad("지표", name_w) + "   "
                  + "".join(_pad(f"→{b}", 20, ">") for _, b in pairs)]
        for _, name, direction in QUALITY + STABILITY:
            values = series(name)
            candidates = [s for s in (_pooled_std(values.values()), sigma_ref.get(name))
                          if s and s == s]
            sigma = max(candidates) if candidates else float("nan")
            arrow = {1: "↑", -1: "↓", 0: "·"}[direction]
            row = _pad(name, name_w) + _pad(arrow, 3, ">")
            for a, b in pairs:
                lo, hi = values[a], values[b]
                if not lo or not hi:
                    row += _pad("-", 20, ">")
                    continue
                delta = sum(hi) / len(hi) - sum(lo) / len(lo)
                d = digits_for(name)
                se = sigma * (1.0 / len(lo) + 1.0 / len(hi)) ** 0.5
                if se != se or se == 0:
                    verdict, z = "?", None
                else:
                    z = abs(delta) / se
                    verdict = ({1: "좋아짐" if delta > 0 else "나빠짐",
                                -1: "좋아짐" if delta < 0 else "나빠짐",
                                0: "유의"}[direction] if z >= Z_THRESHOLD else "노이즈")
                zt = f"({z:.0f}σ)" if z is not None else ""
                row += _pad(f"{delta:+.{d}f} {verdict}{zt}", 20, ">")
            lines.append(row)

        # **선택 epoch 혼동을 경고한다.** `train−val iou 격차`·`되올림`은 얼마나 오래
        # 학습했는지에 직접 붙는 양이고, 체크포인트 선택 epoch은 시드 간 σ가 7~13이라
        # 그 자체가 노이즈다. 두 셀이 다른 epoch에서 뽑혔으면 이 지표들의 차이는 처치
        # 때문이 아니라 epoch 때문일 수 있다. (실제로 `legacy_s0`이 ep 9에서 뽑혀
        # `train−val 격차`가 5σ 차이로 찍혔다.)
        ep = series("선택 ep")
        warn = []
        for a, b in pairs:
            if ep[a] and ep[b]:
                gap = sum(ep[b]) / len(ep[b]) - sum(ep[a]) / len(ep[a])
                if abs(gap) >= 8.0:
                    warn.append(f"{b} (Δ선택ep {gap:+.0f})")
        if warn:
            lines += ["  !! 선택 epoch이 크게 다르다: " + ", ".join(warn),
                      "     `train−val iou 격차`·`되올림`은 학습을 얼마나 오래 했는지에 붙는"
                      " 양이므로,",
                      "     이 차이는 기하 때문이 아니라 **뽑힌 epoch 때문**일 수 있다."
                      " 다른 지표부터 읽는다."]

    lines += ["", "=== 판독 규칙 (§18.3.3) ===",
              "  유도가 예측하는 최적 구간은 [-0.5375, -0.0375] 특징픽셀이고, 그 안에서",
              "  뜻이 있는 대조는 **-0.5 대 0**이다. 최적점이 -1.0이나 +0.5에서 나오면",
              "  유도 어딘가가 틀렸다는 신호이므로 그 값을 그대로 채택하지 않는다.",
              "  전부 `노이즈`로 나오면 결론은 **'기하 오프셋은 이 task에서 무시할 크기'**이고,",
              "  그때도 규약 수정 자체는 correctness 문제로 남는다 -- 성능이 근거가 아니다."]
    print("\n".join(lines) + "\n")


if __name__ == "__main__":
    Fire(main)
