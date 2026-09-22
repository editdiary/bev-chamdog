"""캠페인 `02b_native_source_prior`의 학습 큐를 실행한다.

**묻는 것.** 지금까지의 사전학습은 SynWoodScape를 **타깃 과제에 맞춰 가공해서**(전방8/후방4/
횡±6 m, 셀 0.05 m로 타깃과 동일) 썼다. 그러면 재는 것은 "도메인 **내용**의 불일치"뿐이다.
남이 공개한 주행 BEV 모델을 그대로 가져다 쓰는 상황은 그렇지 않다 -- 범위도 해상도도 그쪽
관행을 따른다. 이 실험은 그 상황을 만든다.

| arm | source 격자 | 셀 | 라벨 | 대조군 |
|---|---|---|---|---|
| `source_pretrain_native` | 30 m × 30 m (200×200) | **0.15 m** | 자동 크롭 | (전처리 단계) |
| `source_prior_native` | 위 체크포인트에서 시작한 target 학습 | | | `01`의 `final_s*` |

**두 개의 기존 팔과 나란히 읽는다.**

- `01_overall/fixed_split` (**주 대조군**, scratch) -- 시드 1:1로 짝지어진다
- `02_projection_and_prior/source_prior` (가공한 사전학습) -- 차이에 **기하와 라벨 품질이
  섞여 있다**(보정본은 8/4/±6에만 있다). 보조 비교로만 쓴다

**대조군을 다시 돌리지 않는다.** `01_overall`의 고정 split 5런이 공통 대조군이다.
`source_prior_native` 시드 s는 `source_pretrain_native` 시드 s에서 시작한다.

**보조항은 `--range_loss_mode`로 정한다.** 기본값 `arc_huber`(λ_R=0.3)가 기존 런을 재현하고,
`cumulative_l1`(λ_R=0.15)이 논문에 싣는 형태다. 값은 `tools/paper_final_aux_loss.py`가
정본이며, **사전학습과 미세조정에 같은 값이 걸린다** -- 여기만 바꾸면 비교가 성립하지 않는다.

큐 기계장치(manifest 병합·상태 판정·완료 런 건너뛰기)는
`tools/run_paper_final_projection_prior.py`의 것을 **그대로 재사용한다** -- 같은 로직을 두 번
쓰면 한쪽만 고쳐졌을 때 조용히 갈린다. 거기서 고쳤던 두 버그(`blocked_missing_input` 재판정,
manifest 덮어쓰기)가 여기에도 그대로 적용된다.

실행:
    python tools/run_paper_final_native_prior.py --plan_only
    python tools/run_paper_final_native_prior.py --gpu=0
"""
import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from tools.run_paper_final_projection_prior import (  # noqa: E402
    FIXED_TRAIN,
    FIXED_VAL,
    RunSpec,
    run_queue,
)
from tools.paper_final_aux_loss import (  # noqa: E402
    DEFAULT_AUX_LOSS, add_aux_loss_argument, aux_loss_env,
)

ARMS = ("source_pretrain_native", "source_prior_native")
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "runs/paper_final/02b_native_source_prior"
NATIVE_LABEL_ROOT = _REPO_ROOT / "dataset/synwoodscape_native_roi_15_15_15_h08"


def _source_checkpoint(output_root: Path, seed: int, num_epochs: int) -> Path:
    return (Path(output_root) / "source_pretrain_native" / "ckpt"
            / f"swscape_native_pretrain_s{seed}" / f"model-{num_epochs:09d}.pth")


def build_run_specs(arms, seeds, output_root, num_epochs: int, gpu: str,
                    range_loss_mode: str = DEFAULT_AUX_LOSS):
    """요청된 팔의 런 사양 전부. 사전학습이 항상 앞에 온다 -- 뒤 팔이 그 산출물에 의존한다."""
    output_root = Path(output_root)
    # 사전학습과 미세조정에 **같은** 보조항을 건다(원장 §8 Phase 0).
    aux_env = aux_loss_env(range_loss_mode)
    specs = []
    for arm in [a for a in ARMS if a in arms]:
        for seed in seeds:
            if arm == "source_pretrain_native":
                specs.append(RunSpec(
                    arm, f"swscape_native_pretrain_s{seed}", seed,
                    "configs/train_synwoodscape_native_pretrain.sh",
                    {
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "SEED": str(seed),
                        "RUN_NAME": f"swscape_native_pretrain_s{seed}",
                        "NUM_EPOCHS": str(num_epochs),
                        "OUT_ROOT": str(output_root / "source_pretrain_native"),
                        **aux_env,
                    },
                ))
                continue

            root = output_root / arm
            specs.append(RunSpec(
                arm, f"{arm}_s{seed}", seed, "configs/train_robot_bev_finetune.sh",
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "SEED": str(seed),
                    "RUN_NAME": f"{arm}_s{seed}",
                    "EXP_NAME": arm,
                    "NUM_EPOCHS": str(num_epochs),
                    "TRAIN_SEQUENCES": ",".join(FIXED_TRAIN),
                    "VAL_SEQUENCES": ",".join(FIXED_VAL),
                    "LOG_DIR": str(root / "logs"),
                    "CKPT_DIR": str(root / "ckpt"),
                    # --- 확정 설정. `01_overall`·`02`의 `source_prior`와 **글자 그대로 같다** ---
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
                    "SAVE_FREQ_EPOCHS": str(num_epochs),
                    "INIT_CHECKPOINT": str(_source_checkpoint(output_root, seed, num_epochs)),
                },
            ))
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--plan_only", action="store_true")
    add_aux_loss_argument(parser)
    args = parser.parse_args()

    if not NATIVE_LABEL_ROOT.exists():
        raise SystemExit(
            f"원본 기하 라벨이 없다: {NATIVE_LABEL_ROOT}\n"
            f"  먼저: python tools/build_synwoodscape_native_labels.py")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"알 수 없는 arm {unknown}. 가능한 값: {list(ARMS)}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    specs = build_run_specs(arms, seeds, args.output_root, args.num_epochs, args.gpu,
                            args.range_loss_mode)
    print(f"큐 {len(specs)}런  (arms={arms}, seeds={seeds}, epochs={args.num_epochs})\n")
    run_queue(specs, args.output_root, args.num_epochs, plan_only=args.plan_only)


if __name__ == "__main__":
    main()
