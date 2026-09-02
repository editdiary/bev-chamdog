"""흩어진 분석 산출물을 **결과 파일 하나**로 묶는다 -- `RESULTS.json` + `RESULTS.csv`.

## 왜 필요한가

`runs/*/analysis/`에는 도구마다 다른 형식의 파일이 열몇 개 쌓인다(`.txt` 표, `.json` 요약,
`.csv` 원시 행, `.npz` 배열). 각각은 만든 도구를 알아야 읽히고, "그래서 결론이 뭐였지"에
답하려면 여러 파일을 오가야 한다. **이 도구는 그 위에 한 겹을 얹는다** -- 핵심 결과를 한
파일에 모으고, 각 숫자마다 **어느 파일에서 왔는지**(`source`)를 같이 적는다.

원본을 대체하지 않는다. 원본이 정본이고 이 파일은 색인이자 요약이다.

## 무엇이 나오나

**`RESULTS.json`** -- 자기 설명적(self-describing) 단일 파일.

    _about       이 파일이 무엇인지, 어떻게 읽는지 (사람이 읽는 글)
    _schema      각 절의 키가 무슨 뜻인지
    environment  python/numpy/torch/conda 환경 -- **어느 환경에서 나온 숫자인가**
    experiment   런 목록, 공통 config, 칸마다 다른 손잡이
    integrity    확률맵이 학습 로그를 재현하는지 검사한 결과
    runs         런 하나당 한 행 (선택 epoch, 지표, 되올림 ...)
    per_cell     칸별 평균 ± 시드 간 표준편차
    axes         계획서의 다섯 축 각각의 핵심 숫자
    headline     계획서 §12의 여덟 질문에 대한 답과 근거 숫자

**`RESULTS.csv`** -- 같은 내용의 긴 형식(long format) 표. 스프레드시트·pandas용.

    scope,cell,seed,metric,value,source

    scope=run       런 하나의 값 (seed가 채워진다)
    scope=cell_mean 칸 평균          scope=cell_sd 칸의 시드 간 표준편차
    scope=axis      축별 요약 숫자 (seed 비어 있음)

## 무엇을 읽나

전부 `{root}/analysis/` 아래의 기존 산출물이다. 없으면 그 절만 비우고 `_missing`에 적는다.

    scalars.csv                  런 x 지표 x epoch 전부 (1차 원본)
    configs.json                 런마다 config.json 전체
    verify_predictions.json      무결성 검사
    seed_jitter.json             광선 산포
    decision_disagreement*.json  셀 단위 결정 불일치
    threshold_sweep_rows*.csv    셀 x 시드 x τ 의 모든 지표
    per_sequence.json            시퀀스별·시드별 지표

실행:
    python tools/build_results_bundle.py --root=runs/loss_effect
"""
import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np
from fire import Fire

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# 사다리 순서. 표와 JSON의 키 순서가 이걸 따른다.
LADDER = ("A_ce", "B_perset", "C_hard", "C_soft", "D_range")

# (TensorBoard 태그, 짧은 이름, 높을수록 좋은가)
METRICS = (
    ("val/iou_free_epoch", "iou_free", True),
    ("val/occupied_f1_10cm_epoch", "f1@10cm", True),
    ("val/occupied_f1_20cm_epoch", "f1@20cm", True),
    ("val/fatal_rate_epoch", "fatal_rate", False),
    ("val/free_miss_rate_epoch", "free_miss_rate", False),
    ("val/range_mae_epoch", "range_mae", False),
    ("val/range_bias_epoch", "range_bias", None),
    ("val/range_missed_obstacle_rate_epoch", "missed_obstacle", False),
    ("val/loss_epoch", "val_loss", False),
    ("train/iou_free_epoch", "train_iou_free", True),
)

# plateau 구간 -- epoch 간 변동을 재는 자리. 초기 급강하를 뺀다.
PLATEAU = (10, 40)


# `analysis/` 안의 파일 하나하나가 무엇인지. **폴더를 처음 여는 사람이 길을 잃지 않게**
# `README.md`로 떨어뜨린다 -- 파일 이름만으로는 무엇이 정본이고 무엇이 파생인지 안 보인다.
FILE_GUIDE = (
    ("RESULTS.json", "★ **여기서 시작한다.** 핵심 결과 전부를 담은 단일 파일. 환경·config·"
                     "무결성·축별 요약·칸별 집계·런별 행. 숫자마다 출처가 적혀 있다"),
    ("RESULTS.csv", "★ 같은 내용의 긴 형식 표(스프레드시트·pandas용). "
                    "`scope`가 run/cell_mean/cell_sd/axis로 갈린다"),
    ("README.md", "이 파일. 폴더 안내"),
    ("scalars.csv", "**1차 원본.** 런 × 전 지표 × 전 epoch. 태그를 고르지 않았다"),
    ("configs.json", "**1차 원본.** 런마다 `config.json` 전체(하이퍼파라미터 46개)"),
    ("manifest.json", "런마다 epoch 수·태그 목록·split 줄 수·완주 여부"),
    ("predictions/", "**1차 원본.** val 확률맵(`p(free)`, τ 적용 전) + GT 라벨 + 부호 거리장. "
                     "체크포인트 없이 어떤 τ·영역·지표든 다시 계산할 수 있다"),
    ("verify_predictions.json", "확률맵이 학습 로그를 재현하는지 대조한 결과(무결성)"),
    ("seed_jitter/", "광선별 예측 거리 `R̂`와 `RAY_OK` 캐시(런 × 프레임 × 720광선)"),
    ("seed_jitter.json", "광선 산포 요약 -- 전역/광선별/잔차 σ와 상태 불일치"),
    ("decision_disagreement.json", "셀 단위 free↔non-free 뒤집힘(선택 체크포인트)"),
    ("decision_disagreement_last.json", "같은 것, 고정 epoch 체크포인트"),
    ("threshold_sweep_rows.csv", "**1차 원본.** 셀 × 시드 × τ 의 모든 지표"),
    ("per_sequence.json", "시퀀스별·시드별 지표"),
    ("report_*.txt", "각 도구의 화면 출력 전문(사람이 읽는 표)"),
)


def _write_readme(analysis: Path, bundle: dict) -> Path:
    """`analysis/README.md` -- 폴더 안내. 있는 파일만 크기와 함께 적는다."""
    env = bundle["environment"]
    exp = bundle["experiment"]
    lines = [
        f"# `{analysis}` 안내",
        "",
        "이 폴더는 **자동 생성**된다(`configs/loss_effect_analysis.sh` →",
        "`tools/build_results_bundle.py`). 손으로 고치지 않는다.",
        "",
        "## 어디부터 보나",
        "",
        "1. **`RESULTS.json`** -- 결과 전부가 여기 있다. 맨 위 `_about`과 `_schema`가 읽는 법을",
        "   설명한다. 사람이 읽는 해석은 `docs/loss_effect_results.md`.",
        "2. 표를 직접 만지고 싶으면 **`RESULTS.csv`**.",
        "3. 더 파고들 때만 아래 원본으로 내려간다.",
        "",
        "## 이 결과가 나온 환경",
        "",
        f"- conda `{env.get('conda_env')}` | Python {env.get('python')} |"
        f" torch {env.get('torch')} | numpy {env.get('numpy')}",
        f"- 런 {exp.get('n_runs')}개 = 칸 {len(exp.get('cells', []))}"
        f" × 시드 {len(exp.get('seeds', []))}",
        f"- 체크포인트 선택 규칙: {exp.get('checkpoint_selection_rule')}",
        "",
        "## 파일",
        "",
        "| 파일 | 크기 | 무엇인가 |",
        "|---|---|---|",
    ]
    for name, what in FILE_GUIDE:
        target = analysis / name.rstrip("/")
        if name.endswith("/") and target.is_dir():
            size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
            count = sum(1 for p in target.rglob("*") if p.is_file())
            lines.append(f"| `{name}` | {size / 1e6:.1f} MB ({count}개) | {what} |")
        elif "*" in name:
            hits = sorted(analysis.glob(name))
            if hits:
                size = sum(p.stat().st_size for p in hits)
                lines.append(f"| `{name}` | {size / 1024:.0f} KB ({len(hits)}개) | {what} |")
        elif target.exists():
            size = target.stat().st_size
            unit = f"{size / 1e6:.1f} MB" if size > 1e6 else f"{size / 1024:.0f} KB"
            lines.append(f"| `{name}` | {unit} | {what} |")
    lines += ["", "## 원본과 파생",
              "",
              "**1차 원본은 셋뿐이다** -- `scalars.csv`(학습이 기록한 모든 epoch 지표),",
              "`predictions/`(예측 확률맵), `threshold_sweep_rows.csv`(τ별 지표).",
              "나머지 `.json`/`.txt`와 `RESULTS.*`는 전부 그 셋에서 유도된 것이라,",
              "**의심스러우면 원본에서 다시 계산해 대조한다.**", ""]
    path = analysis / "README.md"
    path.write_text("\n".join(lines))
    return path


def _load_json(path: Path, missing: list):
    if not path.exists():
        missing.append(str(path))
        return None
    return json.loads(path.read_text())


def _read_scalars(path: Path, missing: list):
    """`scalars.csv` -> `{(cell, seed, tag): {epoch: value}}`."""
    if not path.exists():
        missing.append(str(path))
        return {}
    series = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if not row["tag"].endswith("_epoch") or row["seed"] == "":
                continue
            key = (row["cell"], int(row["seed"]), row["tag"])
            series.setdefault(key, {})[int(row["epoch"])] = float(row["value"])
    return series


def _mean_sd(values):
    values = [v for v in values if v == v]
    if not values:
        return {"mean": None, "sd": None, "n": 0, "values": []}
    return {"mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "n": len(values), "values": [float(v) for v in values]}


def _environment() -> dict:
    """**어느 환경에서 나온 숫자인가.** 2026-09-01에 20런 전체를 의도하지 않은 conda
    환경(`base`, Python 3.14 + torch 2.13)에서 돌린 적이 있어서, 그 뒤로 결과 파일에
    환경을 박아 둔다."""
    import os
    # **환경 이름은 인터프리터 경로에서 유도한다.** `CONDA_DEFAULT_ENV`는 셸의 상태라,
    # 다른 환경의 python을 직접 호출하면(`envs/bev-chamdog/bin/python tools/...`) 실제
    # 인터프리터와 어긋난 이름이 박힌다 -- 그러면 이 파일이 거짓말을 한다.
    prefix = Path(sys.prefix)
    derived = prefix.name if prefix.parent.name == "envs" else "base"
    info = {"python": platform.python_version(),
            "conda_env": derived,
            "conda_default_env_var": os.environ.get("CONDA_DEFAULT_ENV"),
            "numpy": np.__version__, "executable": sys.executable}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["device"] = (torch.cuda.get_device_name(0)
                          if torch.cuda.is_available() else None)
    except Exception as exc:                                   # noqa: BLE001
        info["torch"] = f"불러오지 못함: {exc}"
    try:
        from projects.common.npsafe import elision_is_broken
        info["numpy_elision_broken"] = bool(elision_is_broken())
    except Exception:                                          # noqa: BLE001
        info["numpy_elision_broken"] = None
    return info


def _rebound(loss, entropy, lambda_b):
    """되올림 -- 최저점 대비 마지막 epoch의 상승률 [%].

    `excess`는 soft loss의 target 엔트로피 하한 `λ_B·H̄`를 뺀 것이다. **loss 종류를 넘어
    비교할 수 있는 것은 이쪽뿐이다** -- 생 되올림은 상수가 클수록 기계적으로 작아진다.
    """
    if not loss:
        return {"raw_pct": None, "excess_pct": None, "min_epoch": None}
    last, e_min = max(loss), min(loss, key=loss.get)
    raw = 100.0 * (loss[last] / loss[e_min] - 1.0)
    excess = None
    fl, fm = lambda_b * entropy.get(last, 0.0), lambda_b * entropy.get(e_min, 0.0)
    if loss[e_min] - fm > 0:
        excess = 100.0 * ((loss[last] - fl) / (loss[e_min] - fm) - 1.0)
    return {"raw_pct": raw, "excess_pct": excess, "min_epoch": e_min}


def _threshold_rows(path: Path, missing: list):
    """`threshold_sweep_rows.csv` -> `{(cell, seed): {tau: {metric: value}}}`."""
    if not path.exists():
        missing.append(str(path))
        return {}
    out = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            cell, seed, tau = row["cell"], int(row["seed"]), float(row["tau"])
            metrics = {k: float(v) for k, v in row.items()
                       if k not in ("cell", "seed", "tau")}
            out.setdefault((cell, seed), {})[tau] = metrics
    return out


def _tau_at(curve_x, curve_y, target):
    """`np.interp`의 전제(x 증가)를 검사한 뒤 역보간. 뒤집히거나 구간 밖이면 None."""
    x, y = np.asarray(curve_x, float), np.asarray(curve_y, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if np.any(np.diff(x) < 0) or not x.min() <= target <= x.max():
        return None
    return float(np.interp(target, x, y))


def main(root="runs/loss_effect", cells=LADDER, seeds=(0, 1, 2, 3, 4),
         fixed_epoch=40, out_dir=None):
    root = Path(root)
    analysis = Path(out_dir) if out_dir else root / "analysis"
    cells = [str(c) for c in (cells if isinstance(cells, (list, tuple))
                              else str(cells).split(","))]
    seeds = [int(s) for s in (seeds if isinstance(seeds, (list, tuple))
                              else str(seeds).split(","))]
    missing = []

    series = _read_scalars(analysis / "scalars.csv", missing)
    configs = _load_json(analysis / "configs.json", missing) or {}
    verify = _load_json(analysis / "verify_predictions.json", missing)
    jitter = _load_json(analysis / "seed_jitter.json", missing)
    disagree = _load_json(analysis / "decision_disagreement.json", missing)
    disagree_last = _load_json(analysis / "decision_disagreement_last.json", missing)
    per_seq = _load_json(analysis / "per_sequence.json", missing)
    tau_rows = _threshold_rows(analysis / "threshold_sweep_rows.csv", missing)

    # ── 런별 행 ────────────────────────────────────────────────────────────────
    runs, csv_rows = [], []
    for cell in cells:
        for seed in seeds:
            run = f"{cell}_s{seed}"
            iou = series.get((cell, seed, "val/iou_free_epoch"), {})
            if not iou:
                continue
            config = configs.get(run, {})
            lambda_b = float(config.get("lambda_b", 0) or 0)
            selected = max(iou, key=iou.get)          # 체크포인트 선택 규칙과 같다
            loss = series.get((cell, seed, "val/loss_epoch"), {})
            entropy = series.get((cell, seed, "val/entropy_boundary_epoch"), {})
            reb = _rebound(loss, entropy, lambda_b)

            at_sel, at_fixed = {}, {}
            for tag, name, _ in METRICS:
                s = series.get((cell, seed, tag), {})
                at_sel[name] = s.get(selected)
                at_fixed[name] = s.get(int(fixed_epoch))

            f1 = series.get((cell, seed, "val/occupied_f1_10cm_epoch"), {})
            entry = {
                "run": run, "cell": cell, "seed": seed,
                "selected_epoch": int(selected), "fixed_epoch": int(fixed_epoch),
                "val_loss_min_epoch": reb["min_epoch"],
                "rebound_excess_pct": reb["excess_pct"], "rebound_raw_pct": reb["raw_pct"],
                # "val loss로 골랐다면 얼마나 손해였나"
                "regret_iou_free": (iou[selected] - iou[reb["min_epoch"]]
                                    if reb["min_epoch"] in iou else None),
                "regret_f1@10cm": (f1[max(f1, key=f1.get)] - f1[reb["min_epoch"]]
                                   if f1 and reb["min_epoch"] in f1 else None),
                "delta_epoch_loss_vs_iou": (abs(selected - reb["min_epoch"])
                                            if reb["min_epoch"] is not None else None),
                "plateau_loss_fluctuation": (
                    float(np.mean([abs(loss[e] - loss[e - 1])
                                   for e in range(PLATEAU[0] + 1, PLATEAU[1] + 1)
                                   if e in loss and e - 1 in loss]))
                    if loss else None),
                "metrics_at_selected_epoch": at_sel,
                "metrics_at_fixed_epoch": at_fixed,
            }
            runs.append(entry)
            for name, value in at_sel.items():
                if value is not None:
                    csv_rows.append(("run", cell, seed, name, value, "scalars.csv@selected"))
            for name, value in at_fixed.items():
                if value is not None:
                    csv_rows.append(("run", cell, seed, f"{name}@ep{fixed_epoch}", value,
                                     "scalars.csv@fixed"))
            for key in ("rebound_excess_pct", "regret_iou_free", "regret_f1@10cm",
                        "delta_epoch_loss_vs_iou", "plateau_loss_fluctuation",
                        "selected_epoch", "val_loss_min_epoch"):
                if entry[key] is not None:
                    csv_rows.append(("run", cell, seed, key, entry[key], "scalars.csv"))

    # ── 칸별 집계 ──────────────────────────────────────────────────────────────
    per_cell = {}
    for cell in cells:
        rows = [r for r in runs if r["cell"] == cell]
        if not rows:
            continue
        block = {}
        for _, name, _d in METRICS:
            block[name] = _mean_sd([r["metrics_at_selected_epoch"][name] for r in rows])
            block[f"{name}@ep{fixed_epoch}"] = _mean_sd(
                [r["metrics_at_fixed_epoch"][name] for r in rows])
        for key in ("rebound_excess_pct", "rebound_raw_pct", "regret_iou_free",
                    "regret_f1@10cm", "delta_epoch_loss_vs_iou",
                    "plateau_loss_fluctuation", "selected_epoch", "val_loss_min_epoch"):
            block[key] = _mean_sd([r[key] for r in rows if r[key] is not None])
        per_cell[cell] = block
        for name, stat in block.items():
            if stat["mean"] is not None:
                csv_rows.append(("cell_mean", cell, "", name, stat["mean"], "집계"))
            if stat["sd"] is not None:
                csv_rows.append(("cell_sd", cell, "", name, stat["sd"], "집계"))

    # ── 축별 요약 ──────────────────────────────────────────────────────────────
    axes = {}

    axes["1_objective_convergence"] = {
        "_what": "val loss가 최저점 뒤에 얼마나 되올라가나, 그리고 그 최저점이 품질 정점과"
                 " 정렬되나. 계획서 §2.",
        "_source": "analysis/scalars.csv",
        "rebound_excess_pct": {c: per_cell[c]["rebound_excess_pct"] for c in per_cell},
        "plateau_loss_fluctuation": {c: per_cell[c]["plateau_loss_fluctuation"]
                                     for c in per_cell},
        "val_loss_min_epoch": {c: per_cell[c]["val_loss_min_epoch"] for c in per_cell},
        "regret_iou_free": {c: per_cell[c]["regret_iou_free"] for c in per_cell},
        "regret_f1@10cm": {c: per_cell[c]["regret_f1@10cm"] for c in per_cell},
    }

    axes["2_final_metric_reproducibility"] = {
        "_what": "같은 설정을 다시 학습했을 때 최종 지표가 얼마나 흔들리나. 계획서 §3.1."
                 " 판정은 시드 간 표준편차의 비이고, F(n-1,n-1) 상위 5 %를 넘어야 유의다.",
        "_source": "analysis/scalars.csv",
        "seed_sd": {c: {n: per_cell[c][n]["sd"] for _, n, _d in METRICS} for c in per_cell},
    }
    if len(cells) >= 2 and cells[0] in per_cell:
        base = cells[0]
        ratios = {}
        for c in cells[1:]:
            if c not in per_cell:
                continue
            ratios[f"{base}/{c}"] = {
                n: (per_cell[base][n]["sd"] / per_cell[c][n]["sd"])
                for _, n, _d in METRICS
                if per_cell[base][n]["sd"] and per_cell[c][n]["sd"]}
        axes["2_final_metric_reproducibility"]["seed_sd_ratio_vs_control"] = ratios
        n_min = min(len(v["values"]) for c in per_cell
                    for v in [per_cell[c]["iou_free"]])
        axes["2_final_metric_reproducibility"]["significant_if_ratio_above"] = {
            3: 4.36, 4: 3.05, 5: 2.53, 6: 2.25}.get(n_min)
        # **고정 epoch 기준을 따로 낸다 -- 재현성 주장의 1차 기준이 이쪽이다.**
        # 선택 epoch은 40 epoch에 대한 argmax라 그 자체가 확률변수이고, "시드가 만든 차이"와
        # "선택이 곡선의 어디에 떨어졌나"가 섞인다. 실측으로 A_ce는 런 안에서 epoch만 바꿔도
        # `fatal`이 0.0045 흔들리는데 고정 epoch의 시드 간 산포는 0.0022였다 -- 즉 선택
        # 위치가 시드보다 큰 변동원이다. 두 기준에서 **모두** 성립하는 것만 주장한다.
        ratios_fixed = {}
        for c in cells[1:]:
            if c not in per_cell:
                continue
            ratios_fixed[f"{base}/{c}"] = {
                n: (per_cell[base][f"{n}@ep{fixed_epoch}"]["sd"]
                    / per_cell[c][f"{n}@ep{fixed_epoch}"]["sd"])
                for _, n, _d in METRICS
                if per_cell[base][f"{n}@ep{fixed_epoch}"]["sd"]
                and per_cell[c][f"{n}@ep{fixed_epoch}"]["sd"]}
        axes["2_final_metric_reproducibility"]["seed_sd_ratio_at_fixed_epoch"] = ratios_fixed
        axes["2_final_metric_reproducibility"]["_caution"] = (
            "선택 epoch 기준 비율은 선택 규칙의 성질이 섞인다. 재현성 주장은"
            " `seed_sd_ratio_at_fixed_epoch`을 1차 기준으로 하고, 두 기준에서 모두"
            " 성립하는 것만 주장한다.")

    if jitter:
        axes["3_ray_geometry"] = {
            "_what": "광선별 예측 자유거리의 시드 간 표준편차 [cm]. `global`은 시드별 전역"
                     " 편향의 산포(계획서 §3.2의 V_global), `resid`는 그 편향을 뺀 국소 산포"
                     "(§4.2의 σ_local), `disagree`는 σ가 볼 수 없는 상태 불일치[%].",
            "_source": "analysis/seed_jitter.json",
            "unit": "cm (disagree/zero는 %)",
            "cells": jitter.get("cells"),
        }
        for c, v in (jitter.get("cells") or {}).items():
            for k in ("global", "median", "p90", "resid", "disagree"):
                if k in v:
                    csv_rows.append(("axis", c, "", f"ray_{k}", v[k], "seed_jitter.json"))

    if disagree:
        axes["4_decision_reproducibility"] = {
            "_what": "같은 셀의 free/non-free 판정이 시드마다 뒤집히는 비율. 계획서 §5."
                     " `_m`이 붙은 것은 **동작점을 맞춘 뒤**의 값이고 그쪽이 판정 표다"
                     " -- 고정 τ에서만 좋아지면 그건 확률 눈금 이동이지 결정 재현성이 아니다.",
            "_source": "analysis/decision_disagreement.json (+ _last)",
            "target_free_miss": disagree.get("target_free_miss"),
            "band_m": disagree.get("band_m"),
            "at_selected_checkpoint": disagree.get("cells"),
            "at_fixed_epoch": (disagree_last or {}).get("cells"),
        }
        for c, v in (disagree.get("cells") or {}).items():
            for k in ("all", "near", "far", "all_m", "near_m", "tau_sd"):
                if k in v and v[k] == v[k]:
                    csv_rows.append(("axis", c, "", f"decision_{k}", v[k],
                                     "decision_disagreement.json"))

    if tau_rows:
        # 모든 칸·시드가 덮는 free_miss 구간에서 목표 동작점을 잡는다.
        taus = sorted({t for v in tau_rows.values() for t in v})
        lo = max(min(v[t]["free_miss"] for t in v) for v in tau_rows.values())
        hi = min(max(v[t]["free_miss"] for t in v) for v in tau_rows.values())
        target = 0.5 * (lo + hi) if lo < hi else None
        op = {"_what": "목표 free_miss를 맞추는 문턱 τ*의 시드 간 산포와, 그 동작점에서의"
                       " fatal. 계획서 §6. **τ=0.5의 차이는 여기서 사라지면 동작점 이동이다.**",
              "_source": "analysis/threshold_sweep_rows.csv",
              "common_free_miss_range": [lo, hi], "target_free_miss": target,
              "cells": {}}
        if target is not None:
            for cell in cells:
                tau_star, fatal_star = [], []
                for seed in seeds:
                    v = tau_rows.get((cell, seed))
                    if not v:
                        continue
                    xs = [v[t]["free_miss"] for t in taus if t in v]
                    t_hat = _tau_at(xs, [t for t in taus if t in v], target)
                    f_hat = _tau_at(xs, [v[t]["fatal"] for t in taus if t in v], target)
                    if t_hat is not None:
                        tau_star.append(t_hat)
                    if f_hat is not None:
                        fatal_star.append(f_hat)
                op["cells"][cell] = {"tau_star": _mean_sd(tau_star),
                                     "fatal_at_matched_operating_point": _mean_sd(fatal_star)}
                for key, stat in op["cells"][cell].items():
                    if stat["mean"] is not None:
                        csv_rows.append(("axis", cell, "", f"{key}_mean", stat["mean"],
                                         "threshold_sweep_rows.csv"))
                    if stat["sd"] is not None:
                        csv_rows.append(("axis", cell, "", f"{key}_sd", stat["sd"],
                                         "threshold_sweep_rows.csv"))
        axes["5_operating_point"] = op

    # === 되올림이 경계에서 오는가 (계획 검토서 Priority 3) =============================
    #
    # 각 런의 `val/loss_epoch`은 **자기 loss의 값**이라 CE 런과 soft 런을 나란히 놓을 수
    # 없다. 그래서 학습 때 loss와 무관한 공통 눈금(hard CE)을 대역 안/밖으로 나눠 기록해
    # 뒀다(`binary_metrics.region_ce_diagnostics`). 여기서는 그 궤적의 **최저점 -> 마지막
    # epoch 변화**를 셀 수로 가중해 "증가분의 몇 %가 대역에서 오나"를 낸다.
    loc = {"_what": "val 오차 증가분이 경계 대역에 국소화되는가. `contribution`은 셀 수로"
                    " 가중한 기여분이고, `boundary_share`가 그 중 대역의 몫이다."
                    " 눈금은 loss 종류와 무관한 hard CE이고 대역은 `|d| <= 0.15 m`로 고정이다.",
           "_source": "analysis/scalars.csv (val/ce_boundary_epoch, val/ce_confident_epoch)",
           "band_m": 0.15, "cells": {}}
    for cell in cells:
        rows_c = [r for r in runs if r["cell"] == cell]
        if not rows_c:
            continue
        fracs, d_b, d_c, rel_b, rel_c = [], [], [], [], []
        for seed in seeds:
            b = series.get((cell, seed, "val/ce_boundary_epoch"), {})
            f = series.get((cell, seed, "val/ce_confident_epoch"), {})
            fr = series.get((cell, seed, "val/frac_ce_boundary_epoch"), {})
            if not b or not f:
                continue
            last = max(b)
            eb, ef = min(b, key=b.get), min(f, key=f.get)
            d_b.append(b[last] - b[eb]); d_c.append(f[last] - f[ef])
            rel_b.append(100.0 * (b[last] / b[eb] - 1.0))
            rel_c.append(100.0 * (f[last] / f[ef] - 1.0))
            if fr:
                fracs.append(fr[last])
        if not d_b:
            continue
        frac = float(np.mean(fracs)) if fracs else float("nan")
        contrib_b, contrib_c = frac * np.mean(d_b), (1.0 - frac) * np.mean(d_c)
        loc["cells"][cell] = {
            "frac_boundary_cells": frac,
            "boundary_ce_rise_pct": _mean_sd(rel_b),
            "confident_ce_rise_pct": _mean_sd(rel_c),
            "contribution_boundary": float(contrib_b),
            "contribution_confident": float(contrib_c),
            "boundary_share": float(contrib_b / (contrib_b + contrib_c))
            if (contrib_b + contrib_c) else None,
        }
        for k in ("boundary_share", "contribution_boundary", "contribution_confident"):
            if loc["cells"][cell][k] is not None:
                csv_rows.append(("axis", cell, "", f"rebound_{k}",
                                 loc["cells"][cell][k], "scalars.csv"))
        csv_rows.append(("axis", cell, "", "boundary_ce_rise_pct",
                         loc["cells"][cell]["boundary_ce_rise_pct"]["mean"], "scalars.csv"))
    if loc["cells"]:
        axes["7_boundary_localization"] = loc

    if per_seq:
        axes["6_per_sequence"] = {
            "_what": "시퀀스별 지표. 계획서 §7. **val 시퀀스가 둘뿐이라 '시퀀스 간 SD'는 두"
                     " 값의 차이일 뿐이고 일반화 지표로 읽으면 안 된다** -- 그 질문의 정본은"
                     " LOSO 7-fold다.",
            "_source": "analysis/per_sequence.json",
            "sequences": per_seq.get("sequences"),
            "frames_per_sequence": per_seq.get("frames_per_sequence"),
            "per_cell_sequence_mean": per_seq.get("per_cell_sequence_mean"),
        }

    # ── 무결성 ────────────────────────────────────────────────────────────────
    integrity = {"_what": "저장한 확률맵으로 지표를 처음부터 다시 계산해 학습 로그와 대조한"
                          " 결과. 전혀 다른 코드 경로로 잰 두 값이 맞으면 프레임 순서·기하·"
                          "체크포인트 선택이 전부 맞은 것이다.",
                 "_source": "analysis/verify_predictions.json"}
    if verify:
        diffs = [r["abs_diff"] for r in verify.get("rows", []) if r.get("abs_diff") == r.get("abs_diff")]
        integrity.update({
            "n_files": verify.get("n_files"), "n_checks": len(verify.get("rows", [])),
            "failures": verify.get("failures", []),
            "passed": not verify.get("failures"),
            "max_abs_diff": max(diffs) if diffs else None,
        })

    common = {}
    if configs:
        keys = set.intersection(*(set(c) for c in configs.values()))
        for key in sorted(keys):
            values = {json.dumps(c.get(key), ensure_ascii=False) for c in configs.values()}
            if len(values) == 1:
                common[key] = next(iter(configs.values()))[key]
    per_cell_config = {}
    for cell in cells:
        one = next((c for r, c in configs.items() if r.startswith(cell + "_s")), None)
        if one:
            per_cell_config[cell] = {k: v for k, v in one.items() if k not in common}

    bundle = {
        "_about": (
            "이 파일 하나가 실험의 핵심 결과 전부다. `runs/*/analysis/`의 개별 산출물을"
            " 모은 것이고, 숫자마다 `_source`로 어느 파일에서 왔는지 적혀 있다."
            " 사람이 읽는 해석은 `docs/loss_effect_results.md`에 있고 여기는 숫자만 있다."
            " 읽는 순서 제안: environment -> experiment -> integrity -> headline -> axes"
            " -> per_cell -> runs."),
        "_schema": {
            "environment": "어느 conda 환경/torch/numpy에서 나온 숫자인가. 환경이 다르면"
                           " 다른 실험이다.",
            "experiment": "런 목록과 config. `config_common`은 모든 런이 공유하는 값,"
                          " `config_per_cell`은 칸마다 다른 손잡이(= 이 실험이 조작한 것).",
            "integrity": "원시 데이터가 학습 로그를 재현하는지. `passed`가 false면 아래를"
                         " 믿으면 안 된다.",
            "runs": "런 하나당 한 행. `metrics_at_selected_epoch`는 체크포인트 선택 규칙"
                    "(val iou_free 최고점)의 epoch에서 읽은 값이라 **위로 편향돼 있고**,"
                    " `metrics_at_fixed_epoch`는 사전 선언한 고정 epoch이라 편향이 없다."
                    " 논문의 주 숫자는 후자여야 한다.",
            "per_cell": "칸별 `mean`/`sd`/`n`/`values`. `sd`는 **시드 간** 표준편차다.",
            "axes": "계획서의 축별 요약. 각 절의 `_what`이 그 축이 무엇을 재는지 설명한다.",
            "headline": "계획서 §12의 여덟 질문에 대한 답. `verdict`는 개선/불변/악화/미확인.",
        },
        "environment": _environment(),
        "experiment": {
            "root": str(root), "cells": cells, "seeds": seeds, "n_runs": len(runs),
            "fixed_epoch": int(fixed_epoch),
            "checkpoint_selection_rule": "val/iou_free 최고점 (select_checkpoint_score)",
            "config_common": common, "config_per_cell": per_cell_config,
        },
        "integrity": integrity,
        "axes": axes,
        "per_cell": per_cell,
        "runs": runs,
        "_missing": missing,
    }

    analysis.mkdir(parents=True, exist_ok=True)
    json_path = analysis / "RESULTS.json"
    json_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False,
                                    allow_nan=True))
    csv_path = analysis / "RESULTS.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(("scope", "cell", "seed", "metric", "value", "source"))
        writer.writerows(csv_rows)

    readme_path = _write_readme(analysis, bundle)

    print(f"런 {len(runs)}개, 칸 {len(per_cell)}개, 축 {len(axes)}개")
    print(f"  {json_path}  ({json_path.stat().st_size / 1024:.0f} KB)")
    print(f"  {csv_path}  ({len(csv_rows):,}행)")
    print(f"  {readme_path}")
    if missing:
        print(f"  !! 없어서 건너뛴 입력 {len(missing)}개: "
              + ", ".join(Path(m).name for m in missing))
    return 0


if __name__ == "__main__":
    sys.exit(Fire(main) or 0)
