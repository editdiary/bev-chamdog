"""캠페인 `02_projection_and_prior`의 학습 큐를 실행한다 (원장 §6).

`tools/run_paper_final_overall.py`와 **같은 관례**다 -- 설정을 단일 출처에서 만들고, manifest를
먼저 쓰고, 완료된 런은 건너뛰고, 불완전한 런은 덮어쓰지 않고 `blocked_incomplete`로 남긴다.

**팔(arm) 넷.**

| arm | 무엇 | 대조군 |
|---|---|---|
| `source_pretrain` | SynWoodScape binary+soft_boundary 사전학습 (시드 0~4) | (없음, 전처리 단계) |
| `source_prior` | 위 체크포인트에서 시작한 target 학습 | `01`의 `final_s*` |
| `pinhole120` | undistort -> 가상 핀홀 HFOV 120°로 학습 | `01`의 `final_s*` |
| `pinhole150` | 같은 것, HFOV 150° | `01`의 `final_s*` |

**대조군을 다시 돌리지 않는다.** `01_overall`의 고정 split 5런이 공통 대조군이고 시드가
1:1로 짝지어진다(paired difference). `source_prior` 시드 s는 `source_pretrain` 시드 s에서
시작한다 -- 그래야 5런이 공통 원인을 공유하지 않아 산포가 총 변동을 과소평가하지 않는다.

**전이하는 체크포인트는 `model-{epochs:09d}.pth`(고정 epoch)다.** `model_best`가 아니다 --
캠페인 규약이 val 기반 선택을 쓰지 않기로 했기 때문이다(원장 §3 "주 체크포인트 = 고정 epoch").

실행:
    python tools/run_paper_final_projection_prior.py --plan_only
    python tools/run_paper_final_projection_prior.py --arms=source_pretrain,source_prior
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# `01_overall`의 고정 split과 **같아야 한다** -- 대조군을 재사용하므로 split이 다르면
# paired difference가 성립하지 않는다.
FIXED_TRAIN = ("raws2", "raws3", "rawos1", "rawos2", "rawos4")
FIXED_VAL = ("raws1", "rawos3")

ARMS = ("source_pretrain", "source_prior", "pinhole120", "pinhole150")
PINHOLE_HFOV = {"pinhole120": 120.0, "pinhole150": 150.0}


@dataclass(frozen=True)
class RunSpec:
    arm: str
    run_name: str
    seed: int
    script: str
    environment: dict = field(default_factory=dict)


def _source_checkpoint(output_root: Path, seed: int, num_epochs: int) -> Path:
    return (Path(output_root) / "source_pretrain" / "ckpt"
            / f"swscape_binary_pretrain_s{seed}" / f"model-{num_epochs:09d}.pth")


from tools.paper_final_aux_loss import (  # noqa: E402
    DEFAULT_AUX_LOSS, add_aux_loss_argument, aux_loss_env,
)


def build_run_specs(arms, seeds, output_root, num_epochs: int, gpu: str,
                    range_loss_mode: str = DEFAULT_AUX_LOSS):
    """요청된 팔의 런 사양 전부. `source_pretrain`이 항상 앞에 온다 -- 뒤 팔이 그 산출물에 의존한다."""
    output_root = Path(output_root)
    # **사전학습에도 같은 보조항을 건다.** 예전에는 사전학습 셸이 값을 하드코딩하고 있어
    # 미세조정만 형태를 바꾸면 절반만 적용됐다(원장 §8 Phase 0).
    aux_env = aux_loss_env(range_loss_mode)
    specs = []
    for arm in [a for a in ARMS if a in arms]:
        for seed in seeds:
            if arm == "source_pretrain":
                root = output_root / "source_pretrain"
                specs.append(RunSpec(
                    arm, f"swscape_binary_pretrain_s{seed}", seed,
                    "configs/train_synwoodscape_binary_pretrain.sh",
                    {
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "SEED": str(seed),
                        "RUN_NAME": f"swscape_binary_pretrain_s{seed}",
                        "NUM_EPOCHS": str(num_epochs),
                        "OUT_ROOT": str(root),
                        **aux_env,
                    },
                ))
                continue

            root = output_root / arm
            env = {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "SEED": str(seed),
                "RUN_NAME": f"{arm}_s{seed}",
                "EXP_NAME": arm,
                "NUM_EPOCHS": str(num_epochs),
                "TRAIN_SEQUENCES": ",".join(FIXED_TRAIN),
                "VAL_SEQUENCES": ",".join(FIXED_VAL),
                "LOG_DIR": str(root / "logs"),
                "CKPT_DIR": str(root / "ckpt"),
                # --- 확정 설정 (원장 §3). `01_overall`과 글자 그대로 같아야 한다 ---
                "FORMULATION": "binary",
                "LOSS": "soft_boundary",
                "DELTA_M": "0.30",
                "SIGMA_M": "0.10",
                "LAMBDA_B": "0.5",
                "SOFT_TARGET": "gaussian",
                "BAND_KAPPA": "1.0",
                "LABEL_EPS": "0.0",
                **aux_env,
                "HEIGHT_BINS": "4",
                "HEIGHT_MIN_M": "-0.25",
                "HEIGHT_MAX_M": "1.75",
                "ENCODER_TYPE": "res101",
                "PIXEL_CONVENTION": "pixel_center",
                "PIXEL_OFFSET": "0.0",
                "AUGMENT": "True",
                "FLIP_AUGMENT": "False",
                "LR": "1e-4",
                "WEIGHT_DECAY": "1e-7",
                # 체크포인트 하나가 471 MB다. 주 결과는 고정 epoch이므로 마지막만 남긴다.
                "SAVE_FREQ_EPOCHS": str(num_epochs),
                "INIT_CHECKPOINT": "none",
            }
            if arm == "source_prior":
                env["INIT_CHECKPOINT"] = str(_source_checkpoint(output_root, seed, num_epochs))
            else:
                env["PROJECTION"] = "pinhole"
                env["PINHOLE_HFOV_DEG"] = f"{PINHOLE_HFOV[arm]:.1f}"
            specs.append(RunSpec(arm, f"{arm}_s{seed}", seed,
                                 "configs/train_robot_bev_finetune.sh", env))
    return specs


def run_state(spec: RunSpec, num_epochs: int) -> str:
    """`_run_state`(`run_paper_final_overall.py`)와 같은 판정이다."""
    env = spec.environment
    # 사전학습 셸은 `OUT_ROOT` 하나만 받고 그 아래 `logs/`·`ckpt/`를 스스로 만든다.
    # 팔 **이름**으로 구분하면 새 사전학습 팔을 추가할 때마다 여기를 고쳐야 하므로
    # 어떤 변수를 받았는지로 구분한다(`02b_native_source_prior`가 이 경로를 함께 쓴다).
    if "OUT_ROOT" in env and "LOG_DIR" not in env:
        root = Path(env["OUT_ROOT"])
        log_dir, ckpt_dir = root / "logs" / spec.run_name, root / "ckpt" / spec.run_name
    else:
        log_dir = Path(env["LOG_DIR"]) / spec.run_name
        ckpt_dir = Path(env["CKPT_DIR"]) / spec.run_name
    if not log_dir.exists() and not ckpt_dir.exists():
        return "fresh"
    if (ckpt_dir / f"model-{num_epochs:09d}.pth").exists():
        return "complete"
    return "incomplete"


def _blockers(spec: RunSpec) -> list:
    """실행 전에 반드시 존재해야 하는 입력. 없으면 이 런을 건드리지 않는다.

    팔 이름이 아니라 **`INIT_CHECKPOINT`가 실제 경로를 가리키는지**로 판단한다 -- 사전학습에서
    시작하는 팔이 늘어나도 여기를 고치지 않는다.
    """
    init = spec.environment.get("INIT_CHECKPOINT", "none")
    if str(init).strip().lower() in ("", "none", "no"):
        return []
    checkpoint = Path(init)
    return [] if checkpoint.exists() else [f"source 체크포인트가 없다: {checkpoint}"]


def _write_status(output_root: Path, status: dict) -> None:
    """`queue_status.json`을 **병합해서** 쓴다.

    팔을 나눠 여러 번 호출하면(예: 먼저 source 팔, 나중에 핀홀 팔) 통째로 덮어쓸 경우
    앞 호출의 기록이 사라진다. 그러면 나중에 "어떤 런이 왜 안 돌았나"를 파일에서 읽을 수
    없다 -- 2026-09-21에 실제로 `source_prior` 5런의 `blocked` 기록이 이렇게 사라졌다.
    """
    path = Path(output_root) / "queue_status.json"
    merged = {}
    if path.exists():
        try:
            merged = json.loads(path.read_text())
        except json.JSONDecodeError:
            merged = {}
    merged.update(status)
    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")


def run_queue(specs, output_root, num_epochs, plan_only=False, command_runner=subprocess.run):
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, check=True,
                                capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    # manifest도 **병합한다.** 팔을 나눠 돌리면 통째로 덮어쓸 경우 앞 호출의 런이 사라져,
    # 논문 provenance의 런 목록이 실제보다 짧아진다(2026-09-21에 20런이 10런으로 줄었다).
    manifest_path = output_root / "experiment_manifest.json"
    known = {}
    if manifest_path.exists():
        try:
            known = {r["run_name"]: r for r in json.loads(manifest_path.read_text())["runs"]}
        except (json.JSONDecodeError, KeyError, TypeError):
            known = {}
    known.update({s.run_name: asdict(s) for s in specs})
    ordered = sorted(known.values(),
                     key=lambda r: (ARMS.index(r["arm"]) if r["arm"] in ARMS else 99, r["seed"]))
    manifest = {
        "updated_at": datetime.now().astimezone().isoformat(),
        "git_commit": commit,
        "output_root": str(output_root),
        "n_runs": len(ordered),
        "n_runs_this_invocation": len(specs),
        "control_arm": "runs/paper_final/01_overall/fixed_split (재사용, 시드 1:1)",
        "runtime": {"python": platform.python_version(), "executable": sys.executable,
                    "conda_env": Path(sys.prefix).name},
        "runs": ordered,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    status = {}
    for spec in specs:
        state = run_state(spec, num_epochs)
        blockers = _blockers(spec)
        if state == "complete":
            decision = "skip_complete"
        elif state == "incomplete":
            decision = "blocked_incomplete"
        elif blockers:
            decision = "blocked_missing_input"
        else:
            decision = "run"
        status[spec.run_name] = {"arm": spec.arm, "state": state,
                                 "decision": decision, "blockers": blockers}
        print(f"  {spec.run_name:<32} {state:<11} -> {decision}"
              + (f"   ({blockers[0]})" if blockers else ""))

    _write_status(output_root, status)
    if plan_only:
        print(f"\n[plan_only] manifest와 queue_status만 썼다: {output_root}")
        return status

    for spec in specs:
        decision = status[spec.run_name]["decision"]
        # **`blocked_missing_input`은 여기서 반드시 다시 판정한다.** 계획 단계의 판정은
        # 큐를 시작하는 시점의 것이라, 같은 큐 안에서 앞 팔이 만들어 줄 입력은 그때 아직
        # 없다. 계획 판정을 그대로 믿고 건너뛰면 `source_prior` 5런이 통째로 실행되지
        # 않는데, 로그에는 `[skip]`조차 남지 않아 몇 시간 뒤에야 알아챈다
        # (2026-09-21에 실제로 그렇게 됐다).
        if decision not in ("run", "blocked_missing_input"):
            continue
        blockers = _blockers(spec)
        if blockers:
            status[spec.run_name].update(decision="blocked_missing_input", blockers=blockers)
            print(f"[skip] {spec.run_name}: {blockers[0]}", flush=True)
            continue
        status[spec.run_name].update(decision="run", blockers=[])
        env = dict(os.environ)
        env.update(spec.environment)
        print(f"\n{'='*70}\n[run] {spec.run_name}  ({spec.arm})\n{'='*70}", flush=True)
        result = command_runner(["bash", spec.script], cwd=_REPO_ROOT, env=env)
        code = getattr(result, "returncode", 0)
        status[spec.run_name]["returncode"] = code
        status[spec.run_name]["decision"] = "done" if code == 0 else "failed"
        _write_status(output_root, status)
        if code != 0:
            print(f"[fail] {spec.run_name} returncode={code} -- 큐를 멈춘다.", flush=True)
            break
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="all")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--output_root", default="runs/paper_final/02_projection_and_prior")
    parser.add_argument("--plan_only", action="store_true")
    add_aux_loss_argument(parser)
    args = parser.parse_args()

    arms = ARMS if args.arms == "all" else tuple(a.strip() for a in args.arms.split(","))
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"모르는 arm: {unknown}. 가능한 값: {list(ARMS)}")
    seeds = [int(s) for s in str(args.seeds).split(",")]
    specs = build_run_specs(arms, seeds, args.output_root, args.num_epochs, args.gpu,
                            args.range_loss_mode)
    print(f"arms={list(arms)} seeds={seeds} epochs={args.num_epochs} gpu={args.gpu}")
    print(f"런 {len(specs)}개\n")
    run_queue(specs, args.output_root, args.num_epochs, plan_only=args.plan_only)


if __name__ == "__main__":
    main()
