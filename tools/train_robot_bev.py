"""자체 수집 데이터셋 -> 단일 head Simple-BEV fine-tuning (ROADMAP Phase 4).

정식화는 `--formulation`으로 고른다: `three_class`(free/occupied/unknown) 또는
`binary`(free/not-free). 갈리는 것은 출력 채널 수·loss·로그 항 이름뿐이고 데이터·지표·
체크포인트 선택 기준은 공유한다 -- 두 런을 한 표에 놓고 비교하기 위해서다
(`docs/finetune_overfitting_diagnosis.md` §15).

SynWoodScape pretrain(`tools/train_synwoodscape.py`)과 지표·로깅을 공유하며
(`projects/common/bev_occupancy_metrics.py`), 다른 것은 세 가지뿐이다:

1. 데이터셋: `RobotBEVDataset` (Double Sphere 3-cam, 시퀀스 단위 split)
2. lifting: `DoubleSphereVoxUtil`
3. 초기화: pretrain 체크포인트에서 시작한다 (`--init_checkpoint`)

2-head 정식화는 제거됐다 -- Phase 3 A/B에서 3-class로 확정했다
(`docs/archive/free_space_metric_migration.md` §8, §9).

실행 예:
    CUDA_VISIBLE_DEVICES=0 python tools/train_robot_bev.py \\
        --train_sequences=raws2,raws3,rawos1,rawos2,rawos4 --val_sequences=raws1,rawos3 \\
        --init_checkpoint=<pretrain best>.pth
"""
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.models\._utils")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

import saverloader  # noqa: E402  (simple_bev submodule; see docs/project_structure.md)
from projects.common.baselines import as_batch, constant_free_map  # noqa: E402
from projects.common import binary_metrics, three_class_metrics  # noqa: E402
from projects.common.free_space import decompose  # noqa: E402
from projects.common.free_space_metrics import (  # noqa: E402
    DEFAULT_RING_EDGES_M,
    build_ring_masks,
    iou_free,
)
from projects.common.polar import build_ray_index  # noqa: E402
from projects.common.range_loss import RayGather  # noqa: E402
from projects.common.bev_occupancy_metrics import (  # noqa: E402
    _Ansi,
    _c,
    _print_banner,
    append_free_metrics,
    empty_epoch_metrics,
    evaluate_split,
    format_epoch_log,
    mean_loss_parts,
    select_checkpoint_score,
    summarize_free_metrics,
    weighted_mean,
    write_epoch_scalars,
)
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    GRID_SPEC,
    RobotBEVDataset,
    build_bev_masks,
    load_masked_labels,
    parse_sequence_names,
    split_samples_by_sequence,
    split_samples_within_sequences,
)
from projects.geometry.double_sphere import FINETUNE_CAMERA_NAMES  # noqa: E402
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.simplebev_three_class import (  # noqa: E402
    ThreeClassSegnet,
    head_was_transferred,
    load_trunk_weights,
    unexpected_skips,
)


def from_scratch(init_checkpoint) -> bool:
    """`--init_checkpoint`가 "체크포인트 없음"을 뜻하는가.

    `None` 외에 문자열 `"none"`/`"no"`/`""`도 받는다. 셸 config에서 변수를 비우는 것으로는
    표현할 수 없기 때문이다(비우면 필수 검사에 걸린다). 이 분기가 없으면 `"none"`이 경로로
    취급되어 `load_trunk_weights`가 파일을 못 찾고 죽는다.
    """
    return init_checkpoint is None or str(init_checkpoint).strip().lower() in ("", "none", "no")


def compute_label_statistics(samples, permanent_blind, invalid) -> dict:
    """라벨 분포를 **마스킹이 적용된 셀**에서만 실측한다.

    3-class loss는 `valid` 안에서만 채점되고 `vis`가 unknown/관측 클래스를 가르므로,
    마스킹 전 라벨로 비율을 세면 실제 loss가 보는 분포와 어긋난다. 자체 데이터셋에서는 이
    차이가 크다 -- 그리드 전체 obstacle 비율은 23%인데 마스킹 후 관측 영역에서는 5%
    수준이다(온실 통로에서 raycast visibility가 장애물에 닿으며 멈춰 장애물 대부분이 경계
    바깥에 놓이기 때문).
    """
    free = occupied = supervised = total = 0
    for sequence_root, sample_id in samples:
        # 조합은 `free_space.decompose` 하나만 쓴다 -- 여기서 다시 쓰면 loss/지표가 보는
        # 클래스 정의와 이 통계가 세는 정의가 갈라질 수 있다.
        parts = decompose(*load_masked_labels(
            sequence_root, sample_id, permanent_blind, invalid
        ))
        observed = parts["free"] | parts["occupied"]        # == vis & valid
        free += int(parts["free"].sum())
        occupied += int(parts["occupied"].sum())
        supervised += int(observed.sum())
        total += observed.size
    return {
        "trivial_iou": free / max(supervised, 1),
        "supervised_fraction": supervised / max(total, 1),
        "obstacle_fraction": occupied / max(supervised, 1),
    }


def _baseline_iou_free(val_samples, permanent_blind, invalid, constant_map, device):
    """학습 split의 셀별 다수결 free map을 validation 라벨에 한 번만 채점한다."""
    if constant_map is None or not val_samples:
        return float("nan")
    values, counts = [], []
    for sequence_root, sample_id in val_samples:
        triple = load_masked_labels(sequence_root, sample_id, permanent_blind, invalid)
        occ, _, valid = triple
        free_gt = torch.from_numpy(decompose(*triple)["free"]).view(1, 1, *occ.shape).to(device)
        valid_t = torch.from_numpy(valid).view(1, 1, *valid.shape).to(device)
        value, count = iou_free(as_batch(constant_map, 1, device), free_gt, valid_t)
        values.append(value)
        counts.append(count)
    return weighted_mean(values, counts)


# 정식화별로 갈리는 것 전부. 이 dict 하나로 모아 두는 이유: 학습 루프 안에 `if binary`가
# 흩어지면 한쪽 경로만 조용히 다른 loss나 다른 가중치를 쓰게 되고, 그러면 두 런을 비교하는
# 것 자체가 무의미해진다. 갈리는 것은 **출력 채널 수·loss 함수·가중치 계산·로그 항 이름**뿐이고
# 데이터·지표·체크포인트 선택 기준(`iou_free`)은 완전히 공유한다.
_FORMULATIONS = {
    "three_class": {
        "num_classes": 3,
        "module": three_class_metrics,
        "weight_label": "unknown/free/occupied",
    },
    "binary": {
        "num_classes": 2,
        "module": binary_metrics,
        "weight_label": "not_free/free",
    },
}


# 반전할 텐서. `sample_id`처럼 텐서가 아닌 항목과, 반전해도 값이 같은 캘리브레이션 항목은
# 건드리지 않는다 -- 반전은 **vox util 쪽**에서 처리되므로 extrinsic은 그대로 넘어가는 것이 맞다.
# `d_bev_g`도 반드시 포함한다 -- 거리장은 라벨의 함수이므로 라벨을 뒤집으면 같이 뒤집혀야
# 한다. 빼먹으면 soft-boundary loss가 **반전된 예측을 반전되지 않은 경계로 채점**한다.
_MIRRORED_KEYS = ("rgb_camXs", "seg_bev_g", "vis_bev_g", "valid_bev_g", "d_bev_g")


def _maybe_mirror(batch, vox_util, mirror_vox_util, flip_augment):
    """확률 0.5로 배치 전체를 좌우 반전한다. **train에서만 호출한다.**

    샘플 단위가 아니라 배치 단위인 이유: lifting 기하가 vox util에 들어 있고 그것은 forward
    한 번에 하나만 쓸 수 있다. 배치가 8장이므로 epoch 전체로 보면 반전 비율은 여전히 절반이다.

    이미지와 BEV 라벨을 **같은 축**으로 뒤집는다(마지막 축 = 이미지 폭 = BEV 횡방향). 기하
    정합성은 `tests/models/test_double_sphere_vox.py`가 고정한다.
    """
    if not flip_augment or float(torch.rand(())) >= 0.5:
        return batch, vox_util
    mirrored = dict(batch)
    for key in _MIRRORED_KEYS:
        mirrored[key] = torch.flip(batch[key], dims=[-1])
    return mirrored, mirror_vox_util


def main(
    exp_name="robot_finetune",
    # 산출물 디렉터리 이름을 **그대로** 정한다. `None`이면 기존 규칙
    # (`{exp_name}_{encoder}_bs{bs}_lr{lr}_s{seed}_{타임스탬프}`)을 쓴다.
    #
    # **왜 이 손잡이가 있나.** 자동 이름은 스윕 폴더에서 사람이 읽을 수 없다 --
    # `sb_a050_re_res101_bs8_lr1e-04_s0_260823_150800`을 보고 무엇을 바꾼 런인지 알 수 없고,
    # 표를 만들 때마다 `config.json`을 열어 대조해야 했다(2026-08-23 사용자 지적).
    # ablation처럼 셀 구성이 미리 정해진 실험에서는 `A_ce_s0`처럼 직접 이름을 준다.
    #
    # 타임스탬프가 사라지므로 **같은 이름의 디렉터리가 이미 있으면 실패한다** -- 덮어쓰면
    # 두 런의 tfevents가 한 디렉터리에 섞여 곡선이 조용히 뒤엉킨다.
    run_name=None,
    train_sequences="raws2,raws3,rawos1,rawos2,rawos4",
    val_sequences="raws1,rawos3",
    val_tail_fraction=0.0,  # val 시퀀스가 없을 때만 쓰는 임시 holdout (시퀀스 뒤쪽 연속 구간)
    # `"none"`/`""`도 from scratch로 받는다 -- config가 이 변수를 필수로 만들었으므로
    # (죽은 기본값을 없애면서) "pretrain 없이"를 셸에서 표현할 방법이 필요하다. 문자열
    # `"none"`을 경로로 취급하면 `load_trunk_weights`가 파일을 못 찾고 죽는다(실제로 겪었다).
    init_checkpoint=None,
    num_epochs=60,
    batch_size=8,
    lr=1e-4,  # pretrain(3e-4)보다 낮게 -- 초기값을 크게 흔들지 않는 것이 fine-tuning의 요점
    weight_decay=1e-7,
    num_workers=8,
    encoder_type="res101",
    augment=False,  # 광도 증강. pretrain에서는 +0.006이었지만 적용 여부는 사용자가 결정한다
    val_freq_epochs=1,
    save_freq_epochs=10,
    dataset_root=DEFAULT_DATASET_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    log_dir="runs/robot_bev/logs",
    ckpt_dir="runs/robot_bev/ckpt",
    device="cuda",
    # 주기 저장분을 몇 개까지 남길지. **기본값 3은 유효 구간을 지운다** -- `save_freq_epochs=10`
    # 으로 60 epoch을 돌리면 10/20/30/40/50/60에 저장되는데 3개만 남아 40/50/60이 되고,
    # 과적합이 빠른 리그에서 정작 쓸 만한 초기 epoch이 통째로 사라진다(2026-08-18 fine-tuning
    # 진단에서 실제로 겪었다: `docs/finetune_overfitting_diagnosis.md` §5). 체크포인트 하나가
    # 약 487 MB이므로 `num_epochs / save_freq_epochs` 만큼 남기는 것을 기본으로 둔다.
    keep_checkpoints=6,
    # 역빈도 가중치의 상한. 기본값(20)의 근거와 실측 병리는
    # `docs/finetune_overfitting_diagnosis.md` §3, §12에 있다. `1`이면 가중치 없음.
    max_class_weight=None,
    n_theta=None,
    # 정식화. `three_class` = free/occupied/unknown (기존), `binary` = free/not-free (§15).
    # binary는 `occupied`를 예측하지 않고 예측 free의 경계에서 유도해 보고하므로 지표 집합은
    # 완전히 같다 -- 두 런을 한 표에 놓고 비교하는 것이 이 플래그의 목적이다.
    formulation="three_class",
    # --- 과적합 손잡이 (§16.3 (b)). 근거는 `docs/finetune_overfitting_diagnosis.md` §17 ---
    # encoder를 얼린다. 40.6 M 파라미터 중 37.0 M(91 %)이 encoder인데 train은 192장이다.
    # ImageNet 특징을 그대로 쓰고 BEV decoder만 학습하면 학습 가능한 파라미터가 1/12로 준다.
    freeze_encoder=False,
    # CE의 label smoothing. val loss가 오르는 이유가 "더 많이 틀려서"가 아니라 "확신이
    # 커져서"이므로(§17), 한 셀이 낼 수 있는 loss에 상한을 씌워 그 발산을 직접 막는다.
    label_smoothing=0.0,
    # 좌우 반전 증강(train 배치 단위, 확률 0.5). 광도 증강과 달리 **기하 다양성을 실제로
    # 늘리는** 유일한 수단이다 -- 리그 ROI가 좌우 대칭(±3 m)이라 성립한다.
    # 기하 정합성은 `tests/models/test_double_sphere_vox.py`가 실측으로 고정한다.
    flip_augment=False,
    # 난수 시드. **반복 실험의 전제다.** 2026-08-21까지 이 값이 0으로 하드코딩돼 있어
    # 같은 config를 두 번 돌리면 같은 결과가 나왔고, 그래서 **시드 분산을 한 번도 재지
    # 못했다.** 그 상태로 §16–§23의 판정 대부분이 "노이즈 대역(0.02) 안"이었는데 그 0.02는
    # 측정값이 아니라 프로젝트 규약이었다. `--seed`를 바꿔 같은 config를 여러 번 돌린 뒤
    # `tools/summarize_repeats.py`로 σ를 실측한다.
    #
    # 시드는 세 곳에 들어간다: (1) decoder 랜덤 초기화, (2) DataLoader 셔플 순서,
    # (3) 광도 증강 파라미터(worker 시드가 base 시드에서 파생된다). cuDNN 비결정성이
    # 남으므로 같은 시드라도 비트 단위로 같지는 않다 -- σ는 그 몫까지 포함한 값이다.
    seed=0,
    # loss 종류. `weighted_ce`(현행, 역빈도 가중 CE) 또는 `soft_boundary`.
    # **현행 경로를 지우지 않는 이유: 대조군이다.** soft-boundary가 이겼다는 판정은 같은
    # 시드·같은 split에서 두 loss를 나란히 돌려서만 나온다
    # (`docs/soft_boundary_loss_design.md` §6).
    loss="weighted_ce",
    # 이하 셋은 `--loss=soft_boundary`에서만 쓰인다.
    # `delta_m` -- 불확실 대역의 반폭 [m]. 0.15 = 3셀. **라벨에서 추정할 수 없다** --
    # `P(free | d)`가 완벽한 계단이라 적합할 분포가 없다(같은 문서 §7). 스윕으로 정한다.
    delta_m=0.15,
    # `lambda_b` -- 경계 항의 가중치. 0.5에서 `Ω_B` 셀은 `Ω_N` 셀의 7.6배를 받는다(§5.6).
    # **뜻이 `delta_m`에 걸려 있으므로** 둘을 같이 움직이면 해석이 섞인다.
    # `0.0`으로 두면 "soft 항이 실제로 일을 하는가"의 ablation이 된다.
    lambda_b=0.5,
    # `soft_target` -- `linear`(기본) 또는 `gaussian`. 선형은 `ε ~ Uniform(−δ, δ)`,
    # Gaussian은 `ε ~ N(0, σ²)`의 사후확률이고 둘 다 정확한 유도가 있다(§5.1–5.2).
    # **선형을 먼저 돌린다** -- 손잡이가 하나 적고 hard 영역과 연속이다.
    soft_target="linear",
    # `sigma_alpha` -- `gaussian`의 **모양 매개변수** `α = σ/δ`. `gaussian`일 때 이것이나
    # `sigma_m` 중 하나를 준다(둘 다 주면 실패한다).
    #
    # **α로 주는 것을 권한다.** 정규화된 target은 `u = d/δ`로 쓰면 `α`만의 함수라 `δ`가
    # 식에서 사라진다 -- 즉 `δ`는 폭(gradient 희석 → 수렴)만, `α`는 모양(경계 정밀도)만
    # 정해 두 손잡이가 직교한다. `σ`를 미터로 주면 `δ`를 바꿀 때 모양이 조용히 딸려간다.
    # `α → ∞`는 정확히 선형 target이고, `α ≲ 0.2`는 5 cm 격자에서 사실상 hard가 된다
    # (`docs/soft_boundary_loss_design.md` §10).
    sigma_alpha=None,
    # `sigma_m` -- `α` 대신 σ를 미터로 직접 줄 때. 대역 안에서 정규화한다(§5.3).
    sigma_m=None,
    # 이하 셋은 **방위각 자유거리 보조항** `L_range`다 (`docs/soft_boundary_loss_design.md` §13).
    #
    # `lambda_r` -- 보조항 가중치. **0.0이면 항이 계산조차 되지 않는다**(대조군 보호).
    # **이 값은 추측하지 말고 `tools/measure_range_gradient.py`로 캘리브레이션한다** --
    # `L_range`는 미터, BCE는 nats라 공통 스케일이 없어서 숫자만 보고는 뜻이 정해지지 않는다.
    # gradient 비 `G_R/G_B ≈ 0.1`에 맞추면 "gradient의 10 %"라는 뜻 있는 값이 된다.
    lambda_r=0.0,
    # `delta_r_m` -- 허용 반폭 [m]. **이 항의 요점이다** -- 이 안에서 loss가 평평해져
    # gradient가 정확히 0이 되고, 그래서 라벨의 반경 방향 오차를 외울 동기가 사라진다.
    # `0.0`으로 두면 "dead zone이 실제로 필요한가"의 ablation이 된다.
    delta_r_m=0.20,
    # `delta_r_over_m` -- **과대예측 쪽 관용만 따로 좁힌다** [m]. `None`이면 `delta_r_m`과
    # 같아 대칭이다. `arc_hat > arc_gt`는 없는 자유공간을 있다고 예측한 것이고 로봇에게
    # 그쪽이 치명 방향이다(`fatal_rate`·`missed_obstacle`). 실측 `arc_bias`가 전 런에서
    # +0.063~+0.086 m로 일관되게 그쪽이었다(§13.7) -- 즉 대칭 dead zone은 **판정 축을
    # 무료로 허용하고 있었다.** 보수 방향(`delta_r_m`)은 넓게 둔다(§4.5).
    delta_r_over_m=None,
    # `huber_beta_m` -- Huber 전환점 [m]. **미터로 둔다** -- 정규화된 스케일에서 주면 실효
    # 오차가 항상 β보다 작아져 순수 L2로 퇴화하고 outlier 강건성이 사라진다.
    huber_beta_m=0.10,
    # 이하 둘은 **특징맵 표본 좌표의 기하**다 (`docs/finetune_overfitting_diagnosis.md` §18.3,
    # `projects/models/pixel_grid.py`). loss와 무관하게 lifting 단계에서만 쓰인다.
    #
    # `pixel_convention` -- `pixel_center`(기본, 옳은 규약) 또는 `legacy_index`(옛 동작).
    # `legacy_index`는 `normalize_grid2d`(픽셀 인덱스 규약)와
    # `grid_sample(align_corners=False)`(픽셀 가장자리 규약)를 섞어 써서 표본 위치가
    # `x*W/(W-1) - 0.5`로 어긋난다 -- 중심에서 0이고 가장자리로 갈수록 커지는 계통 오차이며
    # 마지막 열은 값이 절반으로 죽는다.
    #
    # **2026-08-25에 기본값을 `pixel_center`로 옮겼다(사용자 승인).** 15런 스윕에서 품질·
    # 안전 지표는 전부 노이즈였다 -- **성능이 근거가 아니고 correctness가 근거다**
    # (진단 문서 §18.3.5). **이 시점 이후의 런은 `runs/ablation`·설계 문서 §15·§16의
    # 숫자와 기하가 다르므로**, 그 숫자와 비교할 때는 그 사실을 같이 적는다.
    # `legacy_index`는 옛 런 재현용으로 남긴다.
    pixel_convention="pixel_center",
    # `pixel_offset` -- 표본 좌표에 더하는 상수 [특징픽셀]. 규약을 고쳐도 남는 부분
    # (native -> 입력 리사이즈 규약, stride 8 수용영역 중심, `upsampling_layer` 보간)이다.
    # **스윕 결과 0으로 확정했다** -- 네 칸({-1, -0.5, 0, +0.5}) 전부 노이즈였고 최적점이
    # 없었다(§18.3.5). 0이 아닌 값을 넣으면 노이즈를 적합하는 것이다.
    pixel_offset=0.0,
):
    # **첫 문장이어야 한다** -- 이 지점의 `locals()`는 정확히 인자 목록이다. 해석된 config를
    # 로그 폴더에 남기면 반복 실험을 집계할 때 런 이름을 파싱하지 않아도 되고, 논문 실행의
    # 재현 정보가 tfevents와 같은 자리에 남는다.
    # 튜플/리스트를 콤마 문자열로 되돌린다. Fire가 `--val_sequences=raws1,rawos3`을 **tuple**로
    # 파싱하므로 스칼라만 통과시키면 시퀀스 목록이 통째로 빠지고, LOSO 집계의
    # `--group_by=val_sequences`가 조용히 전부 한 그룹으로 묶인다.
    resolved_config = {
        key: (",".join(map(str, value)) if isinstance(value, (tuple, list)) else value)
        for key, value in locals().items()
        if isinstance(value, (int, float, str, bool, tuple, list, type(None)))
    }
    torch.manual_seed(seed)
    np.random.seed(seed)

    if loss not in ("weighted_ce", "soft_boundary"):
        raise ValueError(f"loss는 weighted_ce 또는 soft_boundary여야 한다: {loss}")
    # soft-boundary는 free/not-free 두 클래스를 전제로 유도됐다(부호 있는 거리 하나로
    # 영역을 나눈다). 3-class에 붙이려면 정식화 자체를 다시 유도해야 하므로 여기서 막는다.
    if loss == "soft_boundary" and formulation != "binary":
        raise ValueError("soft_boundary loss는 --formulation=binary에서만 쓴다")
    if float(lambda_r) > 0.0 and loss != "soft_boundary":
        raise ValueError("lambda_r은 --loss=soft_boundary에서만 쓴다")
    if formulation not in _FORMULATIONS:
        raise ValueError(f"formulation은 {tuple(_FORMULATIONS)} 중 하나여야 한다: {formulation}")
    spec = _FORMULATIONS[formulation]

    dataset_root = Path(dataset_root)
    names = parse_sequence_names(train_sequences)
    val_names = parse_sequence_names(val_sequences)
    sequence_roots = [dataset_root / name for name in names + val_names]
    for root in sequence_roots:
        if not (root / "occupancy_npy").exists():
            raise FileNotFoundError(f"시퀀스를 찾을 수 없다: {root}")

    if val_names:
        train_samples, val_samples = split_samples_by_sequence(sequence_roots, val_names)
        split_note = f"시퀀스 단위 holdout: {','.join(val_names)}"
    elif val_tail_fraction > 0:
        train_samples, val_samples = split_samples_within_sequences(
            sequence_roots, val_tail_fraction
        )
        split_note = (f"시퀀스 뒤쪽 {100 * val_tail_fraction:.0f}% holdout"
                      " (임시 -- 경계가 인접해 낙관적인 숫자다)")
    else:
        train_samples, val_samples = split_samples_by_sequence(sequence_roots, [])
        split_note = "없음"
    permanent_blind, invalid = build_bev_masks(common_root, GRID_SPEC, FINETUNE_CAMERA_NAMES)
    stats = compute_label_statistics(train_samples, permanent_blind, invalid)
    val_stats = compute_label_statistics(val_samples, permanent_blind, invalid)
    train_free_masks = [
        decompose(*load_masked_labels(sequence_root, sample_id, permanent_blind, invalid))["free"]
        for sequence_root, sample_id in train_samples
    ]
    constant_map = constant_free_map(train_free_masks) if train_free_masks else None
    baseline_iou_free = _baseline_iou_free(
        val_samples, permanent_blind, invalid, constant_map, device
    )
    rays = build_ray_index(GRID_SPEC) if n_theta is None else build_ray_index(GRID_SPEC, n_theta=n_theta)
    ring_masks = build_ring_masks(GRID_SPEC)
    class_weights = spec["module"].class_weights_from_labels(
        (load_masked_labels(sequence_root, sample_id, permanent_blind, invalid)
         for sequence_root, sample_id in train_samples),
        max_class_weight=max_class_weight,
    )

    _print_banner([
        f" robot dataset -> Simple-BEV {formulation} fine-tuning",
        f" exp_name={exp_name} | encoder={encoder_type} | cameras={','.join(FINETUNE_CAMERA_NAMES)}",
        f" batch_size={batch_size} | lr={lr:.0e} | epochs={num_epochs}",
        f" train sequences={','.join(names) or '-'} ({len(train_samples)} samples)",
        f" val   split={split_note} ({len(val_samples)} samples)",
        f" init_checkpoint="
        f"{'none (from scratch)' if from_scratch(init_checkpoint) else init_checkpoint}",
        f" masks: vis=0 on {permanent_blind.sum()} cells | valid=0 on {invalid.sum()} cells",
        f" observed (vis&valid) covers {100 * stats['supervised_fraction']:.2f}% of cells"
        f" (obstacle {100 * stats['obstacle_fraction']:.2f}% inside it)",
        # val 분포를 같이 찍는다 -- 한 시퀀스 안에서도 구간마다 관측 면적과 장애물 비율이
        # 몇 배씩 차이 나므로(raws1은 앞 30장 11.7%/7.4% vs 뒤 8장 36.5%/3.1%), 이게 안
        # 보이면 "val이 안 오른다"의 원인이 모델인지 분포 불일치인지 구분할 수 없다.
        f" val   distribution: covers {100 * val_stats['supervised_fraction']:.2f}%"
        f" (obstacle {100 * val_stats['obstacle_fraction']:.2f}% inside it)"
        + ("  <- train과 크게 다르다" if val_samples and (
            abs(val_stats["supervised_fraction"] - stats["supervised_fraction"]) > 0.05
            or abs(val_stats["obstacle_fraction"] - stats["obstacle_fraction"]) > 0.02
        ) else ""),
        f" loss = {loss}"
        + (f" | δ={float(delta_m):.3f} m | λ_B={float(lambda_b):.2f}"
           f" | target={soft_target}"
           + (f" | α={float(sigma_alpha):.3f} (σ={float(sigma_alpha)*float(delta_m):.4f} m)"
              if sigma_alpha is not None else "")
           + (f" | σ={float(sigma_m):.4f} m" if sigma_m is not None else "")
           + ("  <- λ_B=0: soft 항 ablation" if float(lambda_b) == 0.0 else "")
           if loss == "soft_boundary" else "  (역빈도 가중 CE -- 대조군)"),
        # `L_range` 보조항. 별도 줄로 두는 이유: `λ_R`은 gradient 비로 캘리브레이션한 값이라
        # (§13.3) 다른 손잡이와 성격이 다르고, 꺼져 있을 때는 줄 자체가 없어야 한다.
        *([f" loss += λ_R={float(lambda_r):.4f} · L_range"
           f" | δ_R⁻={float(delta_r_m):.3f} m"
           f" | δ_R⁺={float(delta_r_m if delta_r_over_m is None else delta_r_over_m):.3f} m"
           f" | β={float(huber_beta_m):.3f} m"
           + ("  <- δ_R=0: dead zone ablation" if float(delta_r_m) == 0.0 else "")
           + ("  <- 비대칭: 과대예측(fatal 방향)만 좁혔다"
              if delta_r_over_m is not None
              and float(delta_r_over_m) != float(delta_r_m) else "")]
          if float(lambda_r) > 0.0 else []),
        f" class weights ({spec['weight_label']}) = {class_weights.tolist()}"
        + ("  <- soft_boundary에서는 쓰이지 않는다 (per-set 평균이 대체)"
           if loss == "soft_boundary" else
           "  <- occupied는 예측하지 않고 free 경계에서 유도한다 (§15)"
           if formulation == "binary" else ""),
        f" photometric augment (train only) = {bool(augment)}"
        f" | flip augment = {bool(flip_augment)}"
        f" | freeze_encoder = {bool(freeze_encoder)}"
        f" | label_smoothing = {float(label_smoothing)}",
        f" feature sampling: convention = {pixel_convention}"
        f" | pixel_offset = {float(pixel_offset):+.3f} feat px"
        f" ({float(pixel_offset) * 20.0:+.1f} native px)"
        + ("  <- **옛 규약이다. 규약 불일치가 남아 있다** (§18.3)"
           if pixel_convention == "legacy_index" else ""),
        f" trivial 'always drivable' baseline IoU = {stats['trivial_iou']:.3f}  <- compare against this",
        f" constant-map baseline iou_free = {baseline_iou_free:.3f}  <- compare against this",
    ])
    if not val_samples:
        print(_c(_Ansi.YELLOW + _Ansi.BOLD,
                 " [warning] val 시퀀스가 없다 -- 체크포인트 선택 없이 마지막 epoch만 남는다."))

    train_ds = RobotBEVDataset(train_samples, common_root=common_root, augment=augment)
    # 마지막 배치가 1개일 때만 버린다(BatchNorm이 배치 1에서 죽는다). pretrain 쪽은
    # 그냥 drop_last=True인데, 여기서는 시퀀스 하나가 수십 장뿐이라 그러면 한 epoch에서
    # 샘플의 10~20%가 통째로 빠진다.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=len(train_samples) % batch_size == 1,
    )
    val_loader = DataLoader(
        RobotBEVDataset(val_samples, common_root=common_root),  # val은 항상 원본
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    Z, Y, X = GRID_SPEC.n_rows, 1, GRID_SPEC.n_cols
    vox_util = build_double_sphere_vox_util(
        GRID_SPEC, train_ds.cameras, device=device,
        pixel_convention=pixel_convention, pixel_offset=float(pixel_offset),
    )
    # 반전 배치는 lifting 기하가 달라지므로 vox util을 하나 더 둔다. 캘리브레이션은 같고
    # `mirror_x`만 다르다 -- 만드는 비용이 사실상 0이라 플래그와 무관하게 항상 준비해 둔다.
    mirror_vox_util = build_double_sphere_vox_util(
        GRID_SPEC, train_ds.cameras, device=device, mirror_x=True,
        pixel_convention=pixel_convention, pixel_offset=float(pixel_offset),
    )
    # rand_flip=False: 이 리그의 ROI는 전후 비대칭(전방 4 m / 후방 2 m)이라
    # Simple-BEV의 Z축 flip 증강이 물리적으로 성립하지 않는다.
    model = ThreeClassSegnet(
        Z, Y, X, vox_util,
        use_radar=False, use_lidar=False, do_rgbcompress=True,
        encoder_type=encoder_type, rand_flip=False, num_classes=spec["num_classes"],
    ).to(device)
    if from_scratch(init_checkpoint):
        print(_c(_Ansi.YELLOW, " weight transfer: 없음 -- ImageNet trunk + 랜덤 BEV decoder"))
    else:
        # shape가 맞는 키를 전부 복사한다. 2-head pretrain 체크포인트에서는 출력 head 6개가
        # 형상이 달라 skip되고(674/6), 3-class pretrain 체크포인트에서는 head까지 함께
        # 전이돼 skipped가 0이어야 한다 -- 그 숫자가 곧 "head가 전이됐는지"의 확인이다.
        report = load_trunk_weights(model, init_checkpoint, device)
        print(_c(
            _Ansi.CYAN,
            f" weight transfer: loaded {report['loaded']} tensors, skipped {len(report['skipped'])}"
            + ("  <- 출력 head까지 전이됨" if head_was_transferred(report["skipped"])
               else "  <- 출력 head는 랜덤 초기화"),
        ))
        unexpected = unexpected_skips(report["skipped"])
        if unexpected:
            raise RuntimeError(f"trunk keys were skipped: {unexpected[:5]}")

    if freeze_encoder:
        # optimizer에서도 빼 둔다 -- `requires_grad=False`만으로도 갱신은 안 되지만, 그러면
        # AdamW가 상태 텐서를 그대로 들고 있어 "얼렸다"가 로그로 확인되지 않는다.
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(_c(_Ansi.CYAN,
             f" trainable params {sum(p.numel() for p in trainable) / 1e6:.1f}M"
             f" / {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
             + ("  <- encoder 동결" if freeze_encoder else "")))
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, lr, num_epochs * steps_per_epoch + 10,
        pct_start=0.05, cycle_momentum=False, anneal_strategy="linear",
    )
    class_weights = class_weights.to(device)
    # binary는 예측 free의 경계에서 occupied를 유도하므로 `rays`가 필요하다(bs8에서 25 ms).
    # train에서도 매번 계산한다 -- val에서만 계산하면 두 곡선이 다른 것을 재게 된다.
    # soft-boundary는 `class_weights`를 받지 않는다 -- per-set 평균이 역빈도 가중치를
    # **대체**하므로 둘을 같이 걸면 클래스 보정이 두 번 들어간다(설계 문서 §3.1).
    blind_mask = torch.from_numpy(permanent_blind).view(1, 1, *permanent_blind.shape)
    if loss == "soft_boundary":
        # `L_range`의 광선 인덱스. **`rays`와 같은 객체에서 만든다** -- `n_theta`가 갈리면
        # loss와 M3 지표가 다른 광선 집합을 보게 된다. `λ_R = 0`이면 만들지 않는다.
        gather = (RayGather(rays, (GRID_SPEC.n_rows, GRID_SPEC.n_cols), device)
                  if float(lambda_r) > 0.0 else None)

        def step(batch, vox=vox_util):
            return binary_metrics.run_batch_soft_boundary(
                model, batch, vox, device, rays, blind_mask,
                delta=delta_m, lambda_b=lambda_b, target=soft_target,
                sigma=sigma_m, alpha=sigma_alpha,
                gather=gather, lambda_r=float(lambda_r),
                delta_r=float(delta_r_m), huber_beta=float(huber_beta_m),
                delta_r_over=None if delta_r_over_m is None else float(delta_r_over_m),
            )
        loss_part_names = (binary_metrics.SOFT_BOUNDARY_RANGE_LOSS_PART_NAMES
                           if float(lambda_r) > 0.0
                           else binary_metrics.SOFT_BOUNDARY_LOSS_PART_NAMES)
    else:
        step = (
            (lambda batch, vox=vox_util: three_class_metrics.run_batch(  # noqa: E731
                model, batch, vox, class_weights, device, label_smoothing))
            if formulation == "three_class" else
            (lambda batch, vox=vox_util: binary_metrics.run_batch(  # noqa: E731
                model, batch, vox, class_weights, device, rays, label_smoothing))
        )
        loss_part_names = spec["module"].LOSS_PART_NAMES

    # 시드를 이름에 넣는다 -- 반복 실험은 config가 같고 시드만 다르므로, 이름에 없으면
    # 타임스탬프만으로 구별해야 하고 표를 만들 때 사람이 대조해야 한다.
    if run_name is None:
        run_name = (f"{exp_name}_{encoder_type}_bs{batch_size}_lr{lr:.0e}_s{seed}"
                    f"_{datetime.now().strftime('%y%m%d_%H%M%S')}")
    else:
        run_name = str(run_name)
        existing = Path(log_dir) / run_name
        if existing.exists() and any(existing.iterdir()):
            raise FileExistsError(
                f"이미 있는 run_name이다 -- 덮어쓰면 두 런의 tfevents가 섞인다: {existing}"
            )
    log_path = Path(log_dir) / run_name
    writer = SummaryWriter(str(log_path))
    ckpt_path = Path(ckpt_dir) / run_name
    log_path.mkdir(parents=True, exist_ok=True)
    (log_path / "config.json").write_text(
        json.dumps(resolved_config, indent=2, ensure_ascii=False, sort_keys=True)
    )
    for tag, samples in (("train", train_samples), ("val", val_samples)):
        (log_path / f"split_{tag}_samples.txt").write_text(
            "".join(f"{root.name}/{sample_id}\n" for root, sample_id in samples)
        )

    global_step = 0
    best_val_score = 0.0
    interrupted = False
    try:
        for epoch in range(1, num_epochs + 1):
            model.train()
            epoch_start = time.time()
            losses, parts_dicts, free_dicts = [], [], []
            for batch in train_loader:
                optimizer.zero_grad()
                batch, batch_vox_util = _maybe_mirror(
                    batch, vox_util, mirror_vox_util, flip_augment
                )
                loss, parts, free_metrics = step(batch, batch_vox_util)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()

                losses.append(loss.item())
                parts_dicts.append({k: v.item() for k, v in parts.items()})
                append_free_metrics(free_dicts, free_metrics)
                writer.add_scalar("train/loss_step", loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                global_step += 1

            train = {
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "loss_parts": mean_loss_parts(parts_dicts),
                "free": summarize_free_metrics(free_dicts),
            }
            write_epoch_scalars(writer, "train", train, epoch)

            val = empty_epoch_metrics()
            if epoch % val_freq_epochs == 0 and len(val_loader) > 0:
                model.eval()
                val = evaluate_split(step, val_loader, device, rays, ring_masks,
                                     cell_m=GRID_SPEC.cell_m,
                                     range_edges_m=DEFAULT_RING_EDGES_M)
                write_epoch_scalars(writer, "val", val, epoch)

            val_score = select_checkpoint_score(val["free"])
            is_new_best = val_score > best_val_score  # NaN > x는 항상 False
            print(format_epoch_log(
                epoch=epoch, num_epochs=num_epochs, epoch_time=time.time() - epoch_start,
                train_loss=train["loss"], train_loss_parts=train["loss_parts"],
                train_free_metrics=train["free"],
                val_loss=val["loss"], val_loss_parts=val["loss_parts"],
                val_free_metrics=val["free"],
                val_range_metrics=val["range"], val_tolerance_metrics=val["tolerance"],
                baseline_iou_free=baseline_iou_free,
                val_score=val_score, best_val_score=best_val_score, is_new_best=is_new_best,
                loss_part_names=loss_part_names,
            ))

            if epoch % save_freq_epochs == 0 or epoch == num_epochs:
                saverloader.save(str(ckpt_path), optimizer, model, epoch,
                                 keep_latest=keep_checkpoints)
            if is_new_best:
                best_val_score = val_score
                saverloader.save(str(ckpt_path), optimizer, model, epoch,
                                 keep_latest=1, model_name="model_best")
    except KeyboardInterrupt:
        interrupted = True
        print("\n" + _c(_Ansi.YELLOW + _Ansi.BOLD,
                        "[interrupted] Ctrl+C. 저장된 체크포인트/로그는 안전하다."))
    finally:
        writer.close()

    if not interrupted:
        _print_banner([" done.", f" logs:  {log_path}", f" ckpts: {ckpt_path}"])


if __name__ == "__main__":
    Fire(main)
