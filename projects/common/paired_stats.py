"""짝지은 비교의 유의성 판정 -- **캠페인 v3 판정 규칙의 정본**(2026-10-02 사용자 결정).

**규칙.** 같은 시드끼리 짝지은 차이 `Δ_s = 팔_s − 대조_s`(n = 5)에 **양측 t-검정**을 하고,
**95 % 신뢰구간이 0을 포함하지 않을 때만 "유의미한 차이"**다. 자유도 n − 1 = 4에서 이는
`|Δ̄| ≥ t_{0.975,4} · SE = 2.776 · SE`(SE = sd(Δ)/√n)와 같다. 그 밖에는 "유의한 차이 없음"이며
"차이가 없다"고 쓰지 않는다 -- 신뢰구간으로 범위를 말한다.

**바뀐 것.** 2026-09까지의 규칙은 `|Δ̄| > σ_seed` 그리고 `|Δ̄| ≥ 2·SE`였다. 2·SE는 자유도 4에서
약 88 % 신뢰도(p ≈ 0.12)라 95 %보다 느슨했다. σ_seed는 이제 판정에 쓰지 않고 참고값으로만 남긴다.

**비모수 검정을 쓰지 않는 이유.** n = 5에서는 부호 검정·Wilcoxon의 최소 양측 p가 0.0625라 95 %에
도달할 수 없다.

**논문 표.** 지표 mean ± SD만 싣고 유의한 칸에 `*`. 캡션에 비교 대상 · 검정(5시드 짝지은 양측 t,
p < 0.05) · "지표별 검정, 다중 비교 보정 없음"을 적는다. LOSO fold 간 비교에는 쓰지 않는다
(fold가 학습 데이터를 공유해 독립이 아니다).
"""
import math
import statistics

from scipy import stats as _stats

CONFIDENCE = 0.95
SIGNIFICANCE_MARK = "*"


def paired_test(diffs, confidence=CONFIDENCE) -> dict:
    """짝지은 차이 목록 -> 판정에 필요한 값 전부. 결측(None/NaN)은 뺀다.

    반환: `n, mean, sd, se, t, df, p, ci_low, ci_high, significant, sign_agreement`.
    n < 2이거나 차이가 전부 같으면(sd = 0) 검정이 정의되지 않으므로 `significant=False`이고
    p·t·CI는 NaN이다 -- 그런 경우를 유의로 세지 않는다.
    """
    clean = [float(d) for d in diffs if d is not None and math.isfinite(float(d))]
    n = len(clean)
    nan = float("nan")
    out = {"n": n, "mean": nan, "sd": nan, "se": nan, "t": nan, "df": max(n - 1, 0),
           "p": nan, "ci_low": nan, "ci_high": nan, "significant": False,
           "sign_agreement": ""}
    if n == 0:
        return out
    mean = statistics.fmean(clean)
    positive = sum(1 for d in clean if d > 0)
    out.update(mean=mean, sign_agreement=f"{max(positive, n - positive)}/{n}")
    if n < 2:
        return out
    sd = statistics.stdev(clean)
    se = sd / math.sqrt(n)
    out.update(sd=sd, se=se)
    if not se > 0:
        return out
    t = mean / se
    df = n - 1
    crit = float(_stats.t.ppf(0.5 + confidence / 2.0, df))
    out.update(t=t, p=float(2.0 * _stats.t.sf(abs(t), df)),
               ci_low=mean - crit * se, ci_high=mean + crit * se)
    out["significant"] = bool(out["ci_low"] > 0.0 or out["ci_high"] < 0.0)
    return out


def verdict(test: dict, higher_is_better) -> str:
    """사람이 읽는 판정 한 줄. 표의 `*`는 `test["significant"]`만 본다."""
    if not math.isfinite(test.get("mean", float("nan"))):
        return "-"
    if not test["significant"]:
        return "no significant difference (95% CI includes 0)"
    if higher_is_better is None:
        return "significant (no preferred direction)"
    return "arm better" if (test["mean"] > 0) == bool(higher_is_better) else "control better"


def _mean_sd(values):
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return float("nan"), float("nan")
    return statistics.fmean(clean), (statistics.stdev(clean) if len(clean) > 1 else float("nan"))


def paper_cell(mean, sd, significant=False, places=3) -> str:
    """`0.815 ± 0.001*` -- 논문 표의 한 칸. 결측이면 빈 문자열."""
    if mean is None or not math.isfinite(mean):
        return ""
    cell = f"{mean:.{places}f}" + (f" ± {sd:.{places}f}" if sd is not None and math.isfinite(sd) else "")
    return cell + (SIGNIFICANCE_MARK if significant else "")


def paper_metric_table(groups, control, metrics, seeds, labels=None, places=3):
    """논문 표 -- 행 = 조건(대조군 먼저), 열 = 지표, 칸 = `mean ± SD`, 대조군과 유의하면 `*`.

    `groups`는 `{조건: {seed: {지표: 값}}}`, `metrics`는 지표 이름 목록이다. 유의성은 같은 시드끼리
    짝지은 `paired_test`로 지표마다 따로 낸다(다중 비교 보정 없음 -- 캡션에 적는다).
    반환: `(header, rows)`.
    """
    labels = labels or {}
    header = ["condition"] + list(metrics)
    rows = []
    for name in [control] + [g for g in groups if g != control]:
        row = [labels.get(name, name)]
        for m in metrics:
            mean, sd = _mean_sd([groups[name][s].get(m) for s in seeds])
            sig = False
            if name != control:
                sig = paired_test([
                    (groups[name][s][m] - groups[control][s][m])
                    if groups[name][s].get(m) is not None and groups[control][s].get(m) is not None
                    else None for s in seeds])["significant"]
            row.append(paper_cell(mean, sd, sig, places))
        rows.append(row)
    return header, rows


#: 논문 표 캡션에 그대로 쓰는 문장(영문). 생성기가 표 옆 README에 적는다.
CAPTION_NOTE = ("Values are mean ± SD over 5 seeds at the fixed final epoch. "
                "* marks a significant difference from the control (two-sided paired t-test over "
                "seeds, 95% confidence interval excluding 0; each metric tested separately, "
                "no correction for multiple comparisons).")
