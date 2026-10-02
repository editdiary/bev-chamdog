"""논문 Overall Performance의 고정 split·LOSO 학습 큐를 실행한다."""
import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ALL_SEQUENCES = ("raws1", "raws2", "raws3", "rawos1", "rawos2", "rawos3", "rawos4")
FIXED_TRAIN = ("raws2", "raws3", "rawos1", "rawos2", "rawos4")
FIXED_VAL = ("raws1", "rawos3")


@dataclass(frozen=True)
class RunSpec:
    protocol: str
    run_name: str
    seed: int
    train_sequences: tuple[str, ...]
    val_sequences: tuple[str, ...]
    held_out: str | None = None


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.paper_final_run_command import make_command_runner  # noqa: E402
from tools.paper_final_epochs import DEFAULT_NUM_EPOCHS  # noqa: E402
from tools.paper_final_aux_loss import (  # noqa: E402
    DEFAULT_AUX_LOSS, add_aux_loss_argument, aux_loss_env,
)


def build_run_specs(protocol, seeds, folds=()):
    folds = tuple(folds) or ALL_SEQUENCES
    specs = []
    if protocol in ("fixed", "all"):
        specs.extend(
            RunSpec("fixed_split", f"final_s{seed}", seed, FIXED_TRAIN, FIXED_VAL)
            for seed in seeds
        )
    if protocol in ("loso", "all"):
        specs.extend(
            RunSpec(
                "loso",
                f"loso_{held}_s{seed}",
                seed,
                tuple(name for name in ALL_SEQUENCES if name != held),
                (held,),
                held,
            )
            for seed in seeds
            for held in folds
        )
    return specs


def training_environment(spec: RunSpec, output_root: Path, num_epochs: int, gpu: str,
                        range_loss_mode: str = DEFAULT_AUX_LOSS):
    """한 런이 사용하는 설정 전부를 문자열 환경변수로 반환한다."""
    output_root = Path(output_root)
    protocol_root = output_root / spec.protocol
    return {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "EXP_NAME": "final" if spec.protocol == "fixed_split" else f"loso_{spec.held_out}",
        "RUN_NAME": spec.run_name,
        "SEED": str(spec.seed),
        "NUM_EPOCHS": str(num_epochs),
        "LOSS": "soft_boundary",
        "SOFT_TARGET": "gaussian",
        "DELTA_M": "0.30",
        "SIGMA_M": "0.10",
        "SIGMA_ALPHA": "None",
        "LAMBDA_B": "0.5",
        "BAND_KAPPA": "1.0",
        "LABEL_EPS": "0.0",
        # 보조항은 `tools/paper_final_aux_loss.py`가 정본이다 -- 세 러너가 같은 값을 쓴다.
        **aux_loss_env(range_loss_mode),
        "DELTA_R_OVER_M": "None",
        "HEIGHT_BINS": "4",
        "HEIGHT_MIN_M": "-0.25",
        "HEIGHT_MAX_M": "1.75",
        "PIXEL_CONVENTION": "pixel_center",
        "PIXEL_OFFSET": "0.0",
        "FORMULATION": "binary",
        "ENCODER_TYPE": "res101",
        "AUGMENT": "True",
        "INIT_CHECKPOINT": "none",
        "LR": "1e-4",
        "WEIGHT_DECAY": "1e-7",
        "MAX_CLASS_WEIGHT": "20",
        "LABEL_SMOOTHING": "0.0",
        "FLIP_AUGMENT": "False",
        "FREEZE_ENCODER": "False",
        "SAVE_FREQ_EPOCHS": str(num_epochs),
        "TRAIN_SEQUENCES": ",".join(spec.train_sequences),
        "VAL_SEQUENCES": ",".join(spec.val_sequences),
        "LOG_DIR": str(protocol_root / "logs"),
        "CKPT_DIR": str(protocol_root / "ckpt"),
    }


def _run_state(spec: RunSpec, output_root: Path, num_epochs: int):
    protocol_root = Path(output_root) / spec.protocol
    log_dir = protocol_root / "logs" / spec.run_name
    ckpt_dir = protocol_root / "ckpt" / spec.run_name
    if not log_dir.exists() and not ckpt_dir.exists():
        return "fresh"
    config_path = log_dir / "config.json"
    final_checkpoint = ckpt_dir / f"model-{num_epochs:09d}.pth"
    if config_path.exists() and final_checkpoint.exists():
        config = json.loads(config_path.read_text())
        if int(config.get("num_epochs", -1)) == int(num_epochs):
            return "complete"
    return "incomplete"


def run_queue(
    specs,
    output_root,
    num_epochs,
    gpu,
    plan_only=False,
    command_runner=None,
    range_loss_mode=DEFAULT_AUX_LOSS,
):
    """실행 manifest를 먼저 쓰고, 요청된 경우 학습 큐를 순서대로 실행한다."""
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    runs = []
    for spec in specs:
        row = asdict(spec)
        row["environment"] = training_environment(spec, output_root, num_epochs, gpu, range_loss_mode)
        runs.append(row)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": commit,
        "output_root": str(output_root),
        "n_runs": len(runs),
        "runtime": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "conda_env": Path(sys.prefix).name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numpy": np.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "runs": runs,
    }
    (output_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(f"manifest: {len(specs)} runs -> {output_root / 'experiment_manifest.json'}", flush=True)
    if plan_only:
        print("plan-only: 학습을 시작하지 않는다.", flush=True)
        return 0

    statuses = []
    for spec in specs:
        state = _run_state(spec, output_root, num_epochs)
        if state == "complete":
            print(f"[{spec.run_name}] complete -- skip", flush=True)
            statuses.append({"run_name": spec.run_name, "status": "skipped_complete"})
            (output_root / "queue_status.json").write_text(
                json.dumps({"runs": statuses}, indent=2, ensure_ascii=False)
            )
            continue
        if state == "incomplete":
            print(f"[{spec.run_name}] incomplete artifacts -- preserve and block", flush=True)
            statuses.append({"run_name": spec.run_name, "status": "blocked_incomplete"})
            (output_root / "queue_status.json").write_text(
                json.dumps({"runs": statuses}, indent=2, ensure_ascii=False)
            )
            continue
        # **watchdog을 여기서 만든다** -- 멈춘 런을 프로세스 그룹째 죽여 큐가 서지 않게 한다.
        # 테스트는 가짜 `command_runner`를 넘기므로 그때는 만들지 않는다.
        if command_runner is None:
            command_runner = make_command_runner(output_root / "run_logs")
        env = os.environ.copy()
        env.update(training_environment(spec, output_root, num_epochs, gpu, range_loss_mode))
        print(f"[{spec.run_name}] start ({spec.protocol}, seed={spec.seed})", flush=True)
        result = command_runner(
            ["bash", "configs/train_robot_bev_finetune.sh"],
            cwd=repo_root,
            env=env,
            check=False,
        )
        rc = int(result.returncode)
        print(f"[{spec.run_name}] {'complete' if rc == 0 else f'failed rc={rc}'}", flush=True)
        statuses.append({"run_name": spec.run_name,
                         "status": "completed" if rc == 0 else "failed",
                         "returncode": rc})
        (output_root / "queue_status.json").write_text(
            json.dumps({"runs": statuses}, indent=2, ensure_ascii=False)
        )
    return 1 if any(row["status"] in ("failed", "blocked_incomplete")
                    for row in statuses) else 0


def _csv(value, cast=str):
    return tuple(cast(item.strip()) for item in str(value).split(",") if item.strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("fixed", "loso", "all"), default="all")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--folds", default=",".join(ALL_SEQUENCES))
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output_root", default="runs/paper_final/01_overall")
    parser.add_argument("--plan_only", action="store_true")
    add_aux_loss_argument(parser)
    args = parser.parse_args(argv)
    specs = build_run_specs(args.protocol, _csv(args.seeds, int), _csv(args.folds))
    return run_queue(specs, args.output_root, args.num_epochs, args.gpu,
                     args.plan_only, range_loss_mode=args.range_loss_mode)


if __name__ == "__main__":
    raise SystemExit(main())
