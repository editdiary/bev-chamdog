"""loss 항 ablation을 **계단별 차이**로 읽는다. 실험 스크립트는 `configs/ablation_loss.sh`.

이 도구가 `report_convergence.py`와 다른 점은 질문이다. 저쪽은 "이 런의 곡선이 어떤
모양인가"를 읽고, 여기는 **"이 항을 추가하니 무엇이 얼마나 변했나"** 만 읽는다. 그래서

- 셀을 사다리 순서(`A_ce -> B_perset -> C_soft -> D_range`)로 고정하고 **인접 쌍의 Δ**를 낸다.
  인접한 두 셀은 손잡이 하나만 다르므로 그 Δ가 그 항의 효과다.
- 시드 3개의 **평균 ± 표준편차**로 보고한다. σ_run(같은 config·같은 시드 반복)이 f1@10cm에서
  0.0037이라(설계 문서 §13.6.1) 단일 런 차이는 노이즈와 구별되지 않는다.
- Δ를 **셀 내부 산포에서 유도한 문턱**과 비교해 `유의`/`노이즈`를 찍는다. 셀마다 n=3이라
  각각의 표준편차는 못 믿으므로 네 셀을 합친 pooled 표준편차(df=8)를 쓴다.

**지표를 어느 epoch에서 읽나 -- 체크포인트 선택 epoch이다**(`val/iou_free`의 최고점). 그것이
`model_best`로 저장되는 epoch이고, "이 loss로 학습하면 어떤 모델을 얻나"에 답하는 자리다.
수렴 지표(되올림, KL 배율)만 곡선 전체에서 읽는다 -- 그건 애초에 모양에 대한 질문이다.

실행:
    python tools/report_ablation.py
    python tools/report_ablation.py --log_root=runs/ablation/logs
"""
import json
from pathlib import Path

from unicodedata import east_asian_width

from fire import Fire
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 사다리 순서. 여기 순서가 곧 Δ를 매기는 순서다.
LADDER = ("A_ce", "B_perset", "C_hard", "C_soft", "D_range")

# 계단마다 무엇이 추가되는지 -- 표에 그대로 찍어서 사람이 대조하지 않게 한다.
STEP_MEANING = {
    # **`A→B`는 손잡이가 둘이다** -- 집계 방식(역빈도 가중 -> per-set 평균)과 경계 대역
    # 감독 제거가 동시에 일어난다. 그래서 이 계단의 Δ는 어느 쪽에도 귀속시킬 수 없다.
    # `C_hard`가 그 분해를 위해 있다(아래 `EXTRA_PAIRS`).
    ("A_ce", "B_perset"): "per-set 평균 + 경계 대역 감독 제거 (손잡이 둘)",
    ("B_perset", "C_hard"): "경계 대역에 hard 감독 추가 λ_B·L_B^hard",
    ("C_hard", "C_soft"): "**대역 target만 hard -> soft** (나머지 전부 동일)",
    ("C_soft", "D_range"): "보조항 λ_R·L_range",
}

# 인접하지 않지만 **한 손잡이만 다른** 쌍. 사다리 순서로는 안 나오는데 원인 귀속에 꼭 필요하다.
#   A_ce  vs C_hard  -- 둘 다 hard target으로 모든 셀을 감독한다. 다른 것은 집계 방식뿐이다.
EXTRA_PAIRS = (("A_ce", "C_hard"),)

# (태그, 표시 이름, 방향). 방향 `-1`은 낮을수록 좋다는 뜻이다.
QUALITY = (
    ("val/fatal_rate_epoch", "fatal_rate", -1),
    ("val/range_missed_obstacle_rate_epoch", "missed_obstacle", -1),
    ("val/free_miss_rate_epoch", "free_miss_rate", -1),
    ("val/iou_free_epoch", "iou_free", +1),
    ("val/occupied_f1_10cm_epoch", "f1@10cm", +1),
    ("val/occupied_f1_20cm_epoch", "f1@20cm", +1),
    ("val/range_mae_epoch", "range_mae [m]", -1),
    ("val/range_bias_epoch", "range_bias [m]", 0),
)

# **안정성 축.** 여기가 이 실험의 주 질문이다 -- 라벨이 부정확하므로 정확도 개선에는
# 상한이 있고(설계 문서 §14), 대신 "학습이 얼마나 얌전한가"는 loss 설계가 실제로 바꿀 수 있다.
#
# 네 가지 서로 다른 질문을 잰다. 하나가 좋아지고 다른 하나가 나빠지는 일이 있으므로 묶어서
# 읽어야 한다.
#
#   (a) 과적합으로 되올라가나          -- 되올림, val최저 ep
#   (b) train을 외우고 있나            -- train−val 격차 (모든 loss에서 정의된다)
#   (c) 곡선이 흔들리나                -- 마지막 10 epoch의 표준편차
#   (d) 언제 멈춰야 할지 정해지나      -- 선택 ep과 그 시드 간 산포
#
# `train/val KL 배율`은 `λ_B = 0`이면 경계 항이 일을 하지 않아 뜻이 없다. 그래서
# **모든 셀에서 정의되는 `train−val iou 격차`를 같이 낸다** -- 그쪽이 계단마다 판정된다.
# **loss 기반 지표는 보조로 내려 둔다.** 아래 순서가 그 우선순위다 -- 위의 두 줄은 loss가
# 등장하지 않으므로 loss 종류를 넘어 안전하고, `되올림`은 그렇지 않다(주석은 `read_run`).
STABILITY = (
    ("iou_drop", "iou 최고→끝 하락", -1),
    ("fatal_jitter", "val fatal 흔들림", -1),
    ("iou_jitter", "val iou 흔들림", -1),
    ("iou_gap", "train−val iou 격차", -1),
    ("rebound_reducible", "되올림(하한제거) [%]", -1),
    ("rebound_pct", "되올림(생) [%]", -1),
    ("val_loss_min_epoch", "val최저 ep", +1),
    ("selected_epoch", "선택 ep", 0),
    # **val loss를 모델 선택 신호로 쓸 수 있나.** 우리의 실제 선택 규칙은 `iou_free`이므로
    # (`select_checkpoint_score`) 아래 둘은 "만약 val loss로 골랐다면 얼마나 손해였나"다.
    # `Δep`은 두 최적점의 거리, `regret`은 그때 실제로 잃는 품질이다. **`regret`이 0에
    # 가까우면 val loss가 쓸 만한 선택 신호**라는 뜻이고, 그건 loss 종류를 넘어 비교된다
    # (품질 지표 자체는 정의가 같으므로).
    ("dep_iou", "Δep(loss↔iou)", -1),
    ("regret_iou", "regret iou_free", -1),
    ("regret_f1", "regret f1@10cm", -1),
    ("kl_ratio", "train/val KL 배율", -1),
)

# 곡선의 흔들림을 마지막 몇 epoch에서 재나. 10이면 40 epoch 런의 1/4이라 추세가 아니라
# 진동을 잡는다.
JITTER_EPOCHS = 10

# **노이즈 바닥을 재는 런들.** config가 같고 **시드까지 같은** 4런이다 -- 그런데도 결과가
# 갈리는 원인은 `grid_sample` backward의 `atomicAdd`다. 네 런 모두 `C_soft`와 정확히 같은
# 설정(`soft_boundary`, δ=0.15, α=0.5, λ_B=0.5, λ_R=0, 40 epoch, seed 0)이므로 이 사다리의
# 노이즈 바닥으로 그대로 쓸 수 있다. `sigma_run`이 config 일치를 검사한다.
# **[2026-09-01] 경로가 `runs/archive/`로 옮겨졌다**(runs 정리). 이 런들은 `Y=1` 기하이고
# 옛 loss config(δ=0.15, α=0.5)이므로, **새 실험에서는 쓰지 않는다** -- 다른 기하의 노이즈
# 바닥을 가져오는 것이 되고 그건 "옛 숫자와 한 표에 세우지 않는다"는 방침에 어긋난다.
# 셀마다 시드가 여러 개면 그 실험 자신의 pooled 표준편차가 옳은 바닥이다.
# 새 실험에서는 `--sigma_from=None`으로 끈다(`configs/loss_effect_analysis.sh`가 그렇게 부른다).
DEFAULT_SIGMA_RUNS = ("runs/archive/robot_bev_cv/run_noise/logs/noise_r*,"
                      "runs/archive/robot_bev_cv/loss_sweep/logs/sb_a050_re_*")

# `F(df, df)`의 상위 5 % 분위수. scipy를 부르지 않으려고 표로 둔다 -- 이 도구는
# TensorBoard만 읽는 가벼운 집계이고 무거운 의존을 새로 들이지 않는다.
# 값은 표준 F 분포표(단측 0.05)에서 왔다.
_F_CRIT_95 = {1: 161.4, 2: 19.00, 3: 9.28, 4: 6.39, 5: 5.05, 6: 4.28, 7: 3.79,
              8: 3.44, 9: 3.18, 10: 2.98}

# Δ 판정 문턱. 노이즈 바닥의 몇 배를 넘어야 `유의`로 찍나.
# 2.0은 n=3 두 셀 비교에서 대략 95 % 수준이고, 이 프로젝트가 σ_run에 대해 써 온 기준과 같다.
Z_THRESHOLD = 2.0


def _series(acc, tag) -> dict:
    return {e.step: e.value for e in acc.Scalars(tag)} if tag in acc.Tags()["scalars"] else {}


def read_run(run_dir) -> dict:
    """한 런 -> `{cell, seed, config, 지표 dict}`. 값이 없는 지표는 키가 없다."""
    run_dir = Path(run_dir)
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    val_loss = _series(acc, "val/loss_epoch")
    iou = _series(acc, "val/iou_free_epoch")
    if not val_loss or not iou:
        return {}

    # **아직 도는 중인 런을 표에 넣으면 안 된다.** 절반쯤 학습된 곡선에서는 되올림도
    # 흔들림도 KL 배율도 전부 다른 뜻이 되고, 그런데 값이 있으니 표에는 조용히 찍힌다
    # (실제로 겪었다 -- 도는 중인 셀의 `흔들림`이 완료된 셀의 30배로 나왔다).
    epochs = max(val_loss)
    want = int(config.get("num_epochs", 0) or 0)
    if want and epochs < want:
        return {"cell": Path(run_dir).name, "incomplete": f"{epochs}/{want} epoch"}

    # **체크포인트 선택 규칙과 같은 epoch.** 여기서 모든 품질 지표를 읽는다.
    selected = max(iou, key=iou.get)
    min_epoch = min(val_loss, key=val_loss.get)
    last = max(val_loss)

    values = {}
    for tag, name, _ in QUALITY:
        series = _series(acc, tag)
        if selected in series:
            values[name] = series[selected]

    # **`되올림`을 loss 종류를 넘어 비교하려면 하한을 빼야 한다.** soft-boundary loss의
    # 경계 항에는 target 엔트로피 `λ_B·H̄`가 상수로 들어 있다(설계 문서 §5.4). val loss를
    # `c + r(t)`로 쓰면 되올림 = `(r_last − r_min)/(c + r_min)`이므로 **`c`가 클수록 비가
    # 기계적으로 작아진다.** 실측에서 `C_soft`가 19.9 % -> 29.0 %로 바뀌었다 -- 즉 생 되올림은
    # soft loss에 유리하게 45 % 부풀려져 있었다. 두 값을 다 낸다.
    entropy = _series(acc, "val/entropy_boundary_epoch")
    lambda_b = float(config.get("lambda_b", 0) or 0)
    floor_last = lambda_b * entropy.get(last, 0.0)
    floor_min = lambda_b * entropy.get(min_epoch, 0.0)
    values["되올림(생) [%]"] = 100.0 * (val_loss[last] / val_loss[min_epoch] - 1.0)
    if val_loss[min_epoch] - floor_min > 0:
        values["되올림(하한제거) [%]"] = 100.0 * (
            (val_loss[last] - floor_last) / (val_loss[min_epoch] - floor_min) - 1.0)
    # **loss가 등장하지 않는 안정성 지표.** `iou_free`는 정의가 loss 종류와 무관하므로
    # "최고점에서 끝까지 얼마나 떨어지나"는 네 셀에서 같은 뜻이다. 실무에서 "안정적 수렴"의
    # 뜻이 사실 이것이다 -- 어디서 멈춰도 같은 모델을 얻는가.
    values["iou 최고→끝 하락"] = iou[max(iou, key=iou.get)] - iou[last]
    values["val최저 ep"] = float(min_epoch)
    values["선택 ep"] = float(selected)
    # **`regret`은 "val loss로 골랐을 때 잃는 품질"이다.** `iou_free`의 최적 epoch은
    # 정의상 `selected`이므로 `Δep`은 그것과 val loss 최저점의 거리다. `f1@10cm`은 최적
    # epoch이 또 다르므로 자기 최고점을 기준으로 따로 잰다.
    values["Δep(loss↔iou)"] = float(abs(min_epoch - selected))
    if min_epoch in iou:
        values["regret iou_free"] = iou[selected] - iou[min_epoch]
    f1 = _series(acc, "val/occupied_f1_10cm_epoch")
    if f1 and min_epoch in f1:
        values["regret f1@10cm"] = f1[max(f1, key=f1.get)] - f1[min_epoch]
    # **train을 외우고 있나.** `iou_free`는 정의가 loss 종류와 무관하므로 이 격차는 네 셀
    # 전부에서 같은 뜻이다 -- `kl_boundary`가 `λ_B = 0`에서 정의되지 않는 것을 메운다.
    # 선택 epoch에서 읽는다(그 시점의 모델이 저장되는 모델이다).
    train_iou = _series(acc, "train/iou_free_epoch")
    if selected in train_iou:
        values["train−val iou 격차"] = train_iou[selected] - iou[selected]
    # **곡선이 흔들리나.** 마지막 `JITTER_EPOCHS`의 표준편차. 추세가 아니라 진동을 잡는다.
    for tag, name in (("val/iou_free_epoch", "val iou 흔들림"),
                      ("val/fatal_rate_epoch", "val fatal 흔들림")):
        series = _series(acc, tag)
        tail = [series[s] for s in sorted(series)[-JITTER_EPOCHS:]]
        if len(tail) >= 3:
            values[name] = _mean_std(tail)[1]
    # 경계 항의 train/val 격차. **암기 계량기다**(설계 문서 §12.2) -- `λ_B = 0`이면 항이
    # 일을 하지 않으므로 이 숫자는 뜻이 없고, 그래서 그때는 내지 않는다.
    train_kl = _series(acc, "train/kl_boundary_epoch").get(last)
    val_kl = _series(acc, "val/kl_boundary_epoch").get(last)
    if train_kl and val_kl and float(config.get("lambda_b", 0)) > 0:
        values["train/val KL 배율"] = val_kl / train_kl

    name = run_dir.name
    cell, _, seed = name.rpartition("_s")
    return {"cell": cell or name, "seed": seed, "config": config, "values": values}


def _mean_std(xs):
    n = len(xs)
    mean = sum(xs) / n
    if n < 2:
        return mean, 0.0, n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, var ** 0.5, n


def _pooled_std(groups):
    """셀별 표본에서 합동 표준편차. `groups`는 값 리스트의 리스트다.

    셀마다 n=3이면 각 셀의 표준편차는 자유도 2로 못 믿는다. 네 셀의 분산을 합치면 df=8이
    되어 문턱이 훨씬 안정된다 -- **셀 간 분산이 같다는 가정**이고, 같은 데이터·같은 스케줄에
    loss 항만 다른 런들이므로 받아들일 만하다.
    """
    num = den = 0.0
    for xs in groups:
        if len(xs) < 2:
            continue
        mean = sum(xs) / len(xs)
        num += sum((x - mean) ** 2 for x in xs)
        den += len(xs) - 1
    return (num / den) ** 0.5 if den > 0 else float("nan")


def sigma_run(patterns) -> dict:
    """**노이즈 바닥**을 실측 런에서 읽는다. `patterns`는 런 디렉터리 glob의 콤마 목록이다.

    여기 들어가는 런들은 **config가 같고 시드까지 같아야 한다.** 그런데도 결과가 갈리는데,
    원인은 `grid_sample` backward의 `atomicAdd`다(`cudnn.benchmark`는 꺼져 있다). 즉 이것은
    시드 분산이 아니라 **같은 실험을 다시 돌렸을 때의 재현 한계**이고, 그보다 작은 차이는
    어떤 loss 항의 효과라고 주장할 수 없다.

    셀마다 n=1일 때 이 값이 유일한 판정 근거가 된다. 셀 안에서 반복이 있으면 pooled
    표준편차와 비교해 **큰 쪽**을 쓴다 -- 둘 중 하나가 노이즈를 과소평가하면 없는 효과를
    주장하게 되므로 보수적인 쪽으로 붙는다.
    """
    runs = []
    for pattern in str(patterns).split(","):
        pattern = pattern.strip()
        if not pattern:
            continue
        for path in sorted(Path().glob(pattern)):
            if path.is_dir():
                run = read_run(path)
                if run and not run.get("incomplete"):
                    runs.append(run)
    if len(runs) < 2:
        raise ValueError(f"σ_run에는 같은 config의 런이 2개 이상 필요하다: {patterns}")
    keys = {"loss", "lambda_b", "lambda_r", "delta_m", "sigma_alpha", "soft_target",
            "delta_r_m", "huber_beta_m", "num_epochs", "seed",
            "train_sequences", "val_sequences", "augment"}
    signatures = {json.dumps({k: r["config"].get(k) for k in sorted(keys)}) for r in runs}
    if len(signatures) > 1:
        raise ValueError("σ_run 런들의 config가 서로 다르다 -- 그러면 노이즈가 아니라 "
                         "설정 차이를 재게 된다")
    names = {name for r in runs for name in r["values"]}
    out = {"__n__": len(runs)}
    for name in names:
        xs = [r["values"][name] for r in runs if name in r["values"]]
        if len(xs) >= 2:
            out[name] = _mean_std(xs)[1]
    return out


def _width(text) -> int:
    """터미널에서 차지하는 칸 수. **한글·CJK는 두 칸이다.**

    `f"{text:<18s}"`는 문자 수로 세므로 한글이 섞인 표가 통째로 어긋난다. 실제로 그래서
    이전 스윕 표가 읽기 어려웠다.
    """
    return sum(2 if east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text, width, align="<") -> str:
    fill = max(0, width - _width(text))
    return text + " " * fill if align == "<" else " " * fill + text


def _fmt_config(runs) -> str:
    """셀의 손잡이 한 줄. **시드 사이에 config가 다르면 그 사실을 찍는다.**"""
    config = runs[0]["config"]
    loss = config.get("loss", "?")
    if loss != "soft_boundary":
        knobs = f"{loss:<14s} {'-':>5s} {'-':>15s} {'-':>5s} {'-':>5s}"
    else:
        target = f"{config.get('soft_target', '?')} α{config.get('sigma_alpha')}"
        lambda_r = float(config.get("lambda_r", 0))
        delta_r = f"{float(config.get('delta_r_m', 0)):.2f}" if lambda_r > 0 else "-"
        knobs = (f"{loss:<14s} {float(config.get('lambda_b', 0)):5.2f} {target:>15s} "
                 f"{lambda_r:5.2f} {delta_r:>5s}")
    seeds = ",".join(r["seed"] for r in runs)
    mismatch = ""
    keys = ("loss", "lambda_b", "lambda_r", "delta_m", "sigma_alpha", "delta_r_m",
            "huber_beta_m", "num_epochs", "train_sequences", "val_sequences")
    for key in keys:
        if len({json.dumps(r["config"].get(key)) for r in runs}) > 1:
            mismatch += f" !{key}"
    return f"{knobs}  s={seeds}{mismatch}"


def format_report(runs, sigma_ref=None) -> str:
    sigma_ref = sigma_ref or {}
    cells, pending = {}, []
    for run in runs:
        if run.get("incomplete"):
            pending.append(f"{run['cell']} ({run['incomplete']})")
            continue
        cells.setdefault(run["cell"], []).append(run)
    order = [c for c in LADDER if c in cells] + sorted(set(cells) - set(LADDER))
    if not order:
        return "완료된 런이 없다." + (f" 진행 중: {', '.join(pending)}" if pending else "")

    lines = []
    if pending:
        # **표에서 빠졌다는 사실을 반드시 찍는다.** 조용히 빼면 n이 셀마다 다른 표를
        # 완결된 것으로 읽게 된다.
        lines += ["", f"!! 아직 도는 중이라 표에서 제외했다: {', '.join(pending)}"]
    lines += ["", "=== 실행한 설정 (전부 config.json에서 읽었다) ===",
             f"{'셀':<10s} {'loss':<14s} {'λ_B':>5s} {'soft target':>15s} {'λ_R':>5s} "
             f"{'δ_R':>5s}  시드"]
    for cell in order:
        lines.append(f"{cell:<10s} {_fmt_config(sorted(cells[cell], key=lambda r: r['seed']))}")

    def digits_for(name):
        return 1 if "%" in name or name.endswith("ep") else 4

    name_w, cell_w = 20, 18
    header = _pad("지표", name_w) + "   " + "".join(_pad(c, cell_w, ">") for c in order)
    for title, metrics in (
        ("품질 -- 어떤 모델을 얻나 (체크포인트 선택 epoch = val iou_free 최고점)", QUALITY),
        ("안정성 -- 학습이 얼마나 얌전한가", STABILITY),
    ):
        n_max = max(len(v) for v in cells.values())
        spread = f"평균 ± 표준편차 (셀당 n={n_max})" if n_max > 1 else f"셀당 n={n_max}"
        lines += ["", f"=== {title} | {spread} ===", header]
        for _, name, direction in metrics:
            arrow = {1: "↑", -1: "↓", 0: "·"}[direction]
            row = _pad(name, name_w) + _pad(arrow, 3, ">")
            for cell in order:
                xs = [r["values"][name] for r in cells[cell] if name in r["values"]]
                if not xs:
                    row += _pad("-", cell_w, ">")
                    continue
                mean, std, n = _mean_std(xs)
                d = digits_for(name)
                celltext = f"{mean:.{d}f}±{std:.{d}f}" if n > 1 else f"{mean:.{d}f}"
                row += _pad(celltext, cell_w, ">")
            lines.append(row)

    # **셋째 표: 산포 자체를 비교한다.** 위 두 표는 평균이 옮겨졌나를 묻는데, 이 실험에서
    # 가장 크고 일관된 효과는 **평균이 아니라 산포**에 나타났다 -- CE는 같은 설정으로 다시
    # 돌리면 `fatal_rate`가 ±0.0063 흔들리는데 `D_range`는 ±0.0007이다. 실무에서 "학습이
    # 안정적이다"의 첫째 뜻이 이것이므로(같은 실험이 같은 답을 낸다) 따로 낸다.
    if max(len(v) for v in cells.values()) >= 2:
        # **F 임계값은 시드 수에서 나온다** -- 예전에는 n=3 기준(F(2,2)=19.0)이 상수로
        # 박혀 있어서 n=5로 돌려도 4.36배를 요구했다. 그러면 실재하는 산포 차이를
        # `노이즈`로 찍는다. `df = n-1`의 양측 5 % 상위 분위수를 쓴다.
        n_min = min(len(v) for v in cells.values())
        df = max(n_min - 1, 1)
        f_crit = _F_CRIT_95.get(df, _F_CRIT_95[max(_F_CRIT_95)])
        lines += ["", "=== 재현성 -- 시드를 바꿨을 때 결과가 얼마나 흔들리나 ===",
                  f"값은 시드 간 표준편차이고 작을수록 좋다. 마지막 열은 {order[0]} 대비 "
                  f"{order[-1]}의 비이고, F({df},{df}) 상위 5 % 임계값 {f_crit:.2f}"
                  f"(=σ비 {f_crit ** 0.5:.2f}배)를 넘으면 `유의`다.",
                  f"**셀당 n={n_min}이라 분산비 검정은 여전히 약하다** -- 여러 지표가 같은"
                  " 방향인 것이 근거다.",
                  _pad("지표", name_w) + "   " + "".join(_pad(c, cell_w, ">") for c in order)
                  + _pad(f"{order[0][0]}/{order[-1][0]} 배", 12, ">")]
        for _, name, _d in QUALITY + STABILITY:
            stds = {}
            for cell in order:
                xs = [r["values"][name] for r in cells[cell] if name in r["values"]]
                stds[cell] = _mean_std(xs)[1] if len(xs) >= 2 else float("nan")
            if any(s != s for s in stds.values()):
                continue
            d = digits_for(name)
            row = _pad(name, name_w) + _pad("", 3)
            row += "".join(_pad(f"{stds[c]:.{d}f}", cell_w, ">") for c in order)
            first, last_ = stds[order[0]], stds[order[-1]]
            if last_ > 0:
                ratio = first / last_
                mark = "유의" if ratio ** 2 > f_crit else ""
                row += _pad(f"{ratio:.1f}x {mark}", 12, ">")
            lines.append(row)

    # 두 가지 비교를 다 낸다. **인접 계단**은 "이 항이 무엇을 하나"에 답하고, **대조군 대비**는
    # "그래서 CE 대신 이걸 쓰면 무엇이 달라지나"에 답한다. 사다리 중간 셀(`B_perset`)은
    # 목적지가 아니라 분해용이므로 인접 표만 보면 최종 판정을 놓친다.
    # 셋째 원소는 "계단의 뜻을 같이 찍나"다. 대조군 대비 표에서는 `A→B`만 우연히 인접해
    # 한 줄이 나오는데, 그게 나머지 열에도 해당하는 것처럼 읽힌다.
    ladders = [("계단별 차이 -- 이 항을 추가하니 무엇이 얼마나 변했나",
                [(order[i], order[i + 1]) for i in range(len(order) - 1)], True),
               (f"대조군({order[0]}) 대비 -- 그래서 CE 대신 쓰면 무엇이 달라지나",
                [(order[0], c) for c in order[1:]], False)]
    # **인접하지 않지만 손잡이 하나만 다른 쌍**을 따로 낸다. 사다리 표에는 안 나오는데
    # 원인 귀속에는 이쪽이 결정적이다(`EXTRA_PAIRS`의 주석).
    extra = [(a, b) for a, b in EXTRA_PAIRS if a in cells and b in cells]
    if extra:
        ladders.append(("한 손잡이만 다른 쌍 -- 원인을 가르는 비교", extra, True))
    df = sum(max(0, len(v) - 1) for v in cells.values())
    source = []
    if df:
        source.append(f"셀 내부 pooled 표준편차(df={df})")
    if sigma_ref:
        source.append(f"σ_run 실측(같은 config·같은 시드 n={sigma_ref.get('__n__', '?')})")
    noise_note = (f"노이즈 바닥은 {' 와 '.join(source) or '없음'} 중 **큰 쪽**을 쓴다. "
                  f"그 {Z_THRESHOLD}배를 넘으면 판정하고, 아니면 `노이즈`다. "
                  "괄호 안은 노이즈 바닥의 몇 배(σ)다.")
    for title, pairs, explain in ladders:
        lines += ["", f"=== {title} ===", noise_note]
        for a, b in (pairs if explain else ()):
            meaning = STEP_MEANING.get((a, b))
            if meaning:
                lines.append(f"  {a[0]}→{b[0]}  {a} → {b}: {meaning}")
        lines.append(_pad("지표", name_w) + "   "
                     + "".join(_pad(f"{a[0]}→{b[0]}", 26, ">") for a, b in pairs))
        for _, name, direction in QUALITY + STABILITY:
            values = {c: [r["values"][name] for r in cells[c] if name in r["values"]]
                      for c in order}
            # **둘 중 큰 쪽을 노이즈 바닥으로 쓴다.** 하나가 노이즈를 과소평가하면 없는
            # 효과를 주장하게 되므로 보수적인 쪽에 붙는다.
            candidates = [s for s in (_pooled_std(values.values()), sigma_ref.get(name))
                          if s and s == s]
            sigma = max(candidates) if candidates else float("nan")
            arrow = {1: "↑", -1: "↓", 0: "·"}[direction]
            row = _pad(name, name_w) + _pad(arrow, 3, ">")
            for a, b in pairs:
                lo, hi = values[a], values[b]
                if not lo or not hi:
                    row += _pad("-", 26, ">")
                    continue
                delta = sum(hi) / len(hi) - sum(lo) / len(lo)
                d = digits_for(name)
                se = sigma * (1.0 / len(lo) + 1.0 / len(hi)) ** 0.5
                if se != se or se == 0:
                    verdict, z = "?", None
                else:
                    z = abs(delta) / se
                    if z >= Z_THRESHOLD:
                        # 방향이 정의된 지표에서는 좋아졌는지 나빠졌는지도 찍는다.
                        verdict = {1: "좋아짐" if delta > 0 else "나빠짐",
                                   -1: "좋아짐" if delta < 0 else "나빠짐",
                                   0: "유의"}[direction]
                    else:
                        verdict = "노이즈"
                zt = f"({z:.0f}σ)" if z is not None else ""
                row += _pad(f"{delta:+.{d}f} {verdict}{zt}", 26, ">")
            lines.append(row)
    return "\n".join(lines) + "\n"


def main(log_root="runs/ablation/logs", pattern="*", sigma_from=DEFAULT_SIGMA_RUNS):
    """`sigma_from`은 노이즈 바닥을 잴 런들의 glob 콤마 목록이다. `None`이면 쓰지 않는다."""
    root = Path(log_root)
    if not root.exists():
        raise FileNotFoundError(f"로그 폴더가 없다: {root}")
    runs = [r for r in (read_run(d) for d in sorted(root.glob(pattern)) if d.is_dir()) if r]
    sigma_ref = sigma_run(sigma_from) if sigma_from else None
    print(format_report(runs, sigma_ref))


if __name__ == "__main__":
    Fire(main)
