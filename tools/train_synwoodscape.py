"""SynWoodScape -> 3-class 단일 head Simple-BEV pretrain 스크립트 (ROADMAP Phase 3.3).

Simple-BEV 원본(`train_nuscenes.py`)의 관례를 따라 `Fire`로 `main(...)`의 키워드 인자를
CLI에서 받는다 — config 파일 체계 대신 실행 스크립트(`configs/train_synwoodscape_baseline.sh`)에
인자를 나열한다.

값을 보고 어떻게 튜닝할지는 `docs/training_pipeline_walkthrough.md`(코드 정독)와
`docs/BEV_loss_and_metrics_design.md` §2.8(지표 정의)을 본다.
`docs/archive/training_guide.md`는 2-head 시절 문서이므로 플래그·경로를 그대로 쓰면 안 된다.

`tools/train_robot_bev.py`(fine-tuning)와 지표·로깅·체크포인트 선택 기준을 공유한다
(`projects/common/bev_occupancy_metrics.py`). 그래야 pretrain과 fine-tune 숫자를 나란히
읽을 수 있다. 다른 것은 데이터셋과 lifting(어안 투영), 그리고 그리드 크기뿐이다.

2-head 정식화는 제거됐다 -- Phase 3 A/B에서 3-class로 확정했다
(`docs/archive/free_space_metric_migration.md` §8, §9). 이 스크립트가 만드는 체크포인트는 출력
head까지 3-class이므로, fine-tuning이 `load_trunk_weights`로 받을 때 head가 함께 전이된다
(`skipped`가 0으로 찍히는 것이 그 확인이다).

실행 예:
    python tools/train_synwoodscape.py --exp_name=baseline --num_epochs=60 --batch_size=4
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

# Segnet(Encoder_res101/50)이 torchvision.models.resnet*(pretrained=True)로 내부에서 호출하는
# deprecated 인자 경고 -- submodule 코드라 직접 못 고치므로 여기서 억제한다. 동작에는 영향 없음
# (실제로는 pretrained=True와 동일하게 ImageNet 가중치를 불러온다).
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.models\._utils")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

import saverloader  # noqa: E402  (simple_bev submodule; see docs/project_structure.md)
from projects.datasets.simplebev_vox import build_vox_util  # noqa: E402
from projects.datasets.synwoodscape_simplebev import (  # noqa: E402
    CAMERA_NAMES,
    DEFAULT_DATASET_ROOT,
    DEFAULT_OCCUPANCY_GT_ROOT,
    RESIZE_HEIGHT,
    RESIZE_WIDTH,
    SynWoodScapeSimpleBEVDataset,
    resolve_source_profile,
)
from projects.datasets.synwoodscape_split import discover_all_sample_ids, train_val_split  # noqa: E402
from projects.geometry.fisheye import load_camera  # noqa: E402
from projects.datasets.simplebev_vox import (  # noqa: E402
    DEFAULT_HEIGHT_MAX_M,
    DEFAULT_HEIGHT_MIN_M,
    save_height_config,
    vox_dims,
)
from projects.models.fisheye_vox import build_fisheye_vox_util  # noqa: E402
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402
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
from projects.common import binary_metrics, three_class_metrics  # noqa: E402
from projects.common.range_loss import RayGather  # noqa: E402
from projects.common.free_space import decompose  # noqa: E402
from projects.common.free_space_metrics import build_ring_masks  # noqa: E402
from projects.common.polar import build_ray_index  # noqa: E402
from projects.common.baselines import as_batch, constant_free_map  # noqa: E402
from projects.common.free_space_metrics import iou_free  # noqa: E402

# SynWoodScape pretrain 그리드는 전방 8 m / 후방 4 m / 좌우 ±6 m라 로봇 fine-tuning 그리드
# (전방 4 m)보다 두 배 넓다. `DEFAULT_RING_EDGES_M`(0/1.5/3/4)를 그대로 쓰면 그리드 바깥쪽
# 절반이 어느 링에도 들어가지 않는다. 그래서 이 스크립트는 자기 그리드에 맞는 경계를 쓴다.
PRETRAIN_RING_EDGES_M = (0.0, 2.0, 4.0, 6.0, 8.0)   # `roi_8_4_6` 프로파일의 값 (하위 호환)


def load_label_triples(sample_ids, occupancy_gt_root: Path):
    """`(occ, vis, valid)` 3-tuple을 하나씩 내놓는다 -- `class_weights_from_labels`의 입력 계약.

    SynWoodScape에는 리그 고정 마스크(`permanent_blind`/`rear_self_box`)가 없으므로 `valid`가
    전부 1이다. `SynWoodScapeSimpleBEVDataset.__getitem__`이 `valid_bev_g`에 넣는 값과 같아야
    한다 -- 어긋나면 클래스 가중치가 loss가 실제로 보는 분포와 다른 분포에서 계산된다.
    """
    for sample_id in sample_ids:
        occupancy = np.load(occupancy_gt_root / f"{sample_id}_occupancy.npy").astype(bool)
        visible = np.load(occupancy_gt_root / f"{sample_id}_visible.npy").astype(bool)
        yield occupancy, visible, np.ones_like(occupancy)


def compute_trivial_baseline_iou(sample_ids, occupancy_gt_root: Path) -> float:
    """"항상 drivable로 예측"의 IoU = drivable_fraction. 모델 IoU가 이 값 근처면 학습이 아니라
    다수 클래스를 그냥 외운 것일 수 있다.
    """
    free = observed = 0
    for triple in load_label_triples(sample_ids, occupancy_gt_root):
        parts = decompose(*triple)
        free += int(parts["free"].sum())
        observed += int((parts["free"] | parts["occupied"]).sum())
    return free / max(observed, 1)


def baseline_iou_free(val_ids, occupancy_gt_root: Path, constant_map, device) -> float:
    """학습 split의 셀별 다수결 free map을 validation 라벨에 채점한다.

    이미지를 한 픽셀도 보지 않는 예측기다. 이 값을 배너와 매 epoch 로그에 병기하는 이유는
    이 프로젝트의 출발점이 바로 "모델이 트리비얼 예측기에 지고 있었는데 아무도 몰랐다"였기
    때문이다(`docs/archive/free_space_metric_migration.md` §1). fine-tuning 쪽과 같은 장치다.
    """
    if constant_map is None or not val_ids:
        return float("nan")
    values, counts = [], []
    for triple in load_label_triples(val_ids, occupancy_gt_root):
        occ, _, valid = triple
        free_gt = torch.from_numpy(decompose(*triple)["free"]).view(1, 1, *occ.shape).to(device)
        valid_t = torch.from_numpy(valid).view(1, 1, *valid.shape).to(device)
        value, count = iou_free(as_batch(constant_map, 1, device), free_gt, valid_t)
        values.append(value)
        counts.append(count)
    return weighted_mean(values, counts)


def main(
    exp_name="debug",
    num_epochs=60,
    batch_size=4,
    lr=3e-4,
    weight_decay=1e-7,
    num_workers=8,
    val_fraction=0.1,
    split_seed=0,
    # 소스 격자와 라벨 루트를 한 쌍으로 고른다 (`SOURCE_PROFILES`). 기본값은 지금까지의
    # 사전학습과 bit 동일한 경로다.
    source_profile="roi_8_4_6",
    encoder_type="res101",
    use_fisheye=True,
    augment=False,
    val_freq_epochs=1,
    save_freq_epochs=10,  # fine-tuning(`tools/train_robot_bev.py`)과 같은 값으로 맞춰 둔다
    log_dir="work_dirs/logs_synwoodscape",
    ckpt_dir="work_dirs/checkpoints_synwoodscape",
    max_samples=None,
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
    # ---- 아래는 `02_projection_and_prior` 축 B를 위해 추가됐다 (2026-09-21) ----
    # **목적: 사전학습을 확정 설정(원장 §3)과 같은 정식화·같은 loss·같은 Y로 맞추는 것.**
    # 그래야 출력 head까지 형상이 맞아 네트워크 전체가 target으로 전이되고, "합성 도메인의
    # prior를 최대한 물려받았을 때 얼마나 해로운가"를 잰다. 기본값은 예전 그대로라
    # 옛 3-class 호출은 바뀌지 않는다.
    formulation="three_class",   # three_class | binary
    loss="weighted_ce",          # weighted_ce | soft_boundary (binary에서만)
    seed=0,                      # **예전에는 0으로 하드코딩돼 있었다.** 시드를 바꿔야
                                 # target 시드와 1:1로 짝지을 수 있다 (원장 §6.3)
    run_name=None,               # 비우면 자동 이름(타임스탬프 포함)
    # soft_boundary 파라미터. 의미와 확정값 근거는 `configs/train_robot_bev_finetune.sh`.
    delta_m=0.30,
    sigma_m=0.10,
    lambda_b=0.5,
    soft_target="gaussian",
    lambda_r=0.3,
    # 보조항의 형태. `arc_huber`는 endpoint arc + dead zone + Huber(기존 기본값),
    # `cumulative_l1`은 같은 soft target과의 누적 arc profile L1이다.
    # **미세조정 쪽(`tools/train_robot_bev.py`)과 같은 값이어야 한다** -- 사전학습과
    # 미세조정이 다른 보조항으로 학습되면 "이 loss로 캠페인을 돌렸다"가 성립하지 않는다.
    range_loss_mode="arc_huber",
    delta_r_m=0.15,
    huber_beta_m=0.15,
    band_kappa=1.0,
    label_eps=0.0,
):
    if formulation not in ("three_class", "binary"):
        raise ValueError(f"formulation은 three_class 또는 binary여야 한다: {formulation}")
    if loss not in ("weighted_ce", "soft_boundary"):
        raise ValueError(f"loss는 weighted_ce 또는 soft_boundary여야 한다: {loss}")
    if loss == "soft_boundary" and formulation != "binary":
        raise ValueError("soft_boundary loss는 --formulation=binary에서만 쓴다")
    # `train_robot_bev.py`와 **같은 문구로 막는다** -- 오타가 조용히 기본값으로 떨어지면
    # 사전학습만 다른 보조항으로 돌아간다.
    if range_loss_mode not in ("arc_huber", "cumulative_l1"):
        raise ValueError("range_loss_mode는 arc_huber 또는 cumulative_l1이어야 한다: "
                         f"{range_loss_mode}")
    formulation_spec = {
        "three_class": {"num_classes": 3, "module": three_class_metrics,
                        "weight_label": "unknown/free/occupied"},
        "binary": {"num_classes": 2, "module": binary_metrics,
                   "weight_label": "not_free/free"},
    }[formulation]

    torch.manual_seed(seed)
    np.random.seed(seed)

    profile = resolve_source_profile(source_profile)
    grid_spec = profile["grid_spec"]
    gt_root = profile["occupancy_gt_root"]
    ring_edges_m = profile["ring_edges_m"]
    if not gt_root.exists():
        raise SystemExit(
            f"라벨 루트가 없다: {gt_root}\n"
            f"  source_profile={source_profile!r}이면 먼저 라벨을 만들어야 한다.\n"
            f"  native: python tools/build_synwoodscape_native_labels.py")

    all_ids = discover_all_sample_ids(DEFAULT_DATASET_ROOT)
    train_ids, val_ids = train_val_split(
        all_ids, DEFAULT_DATASET_ROOT, val_fraction=val_fraction, seed=split_seed
    )
    if max_samples is not None:  # 빠른 smoke run 용 -- 실제 학습에는 쓰지 않는다
        train_ids, val_ids = train_ids[:max_samples], val_ids[: max(1, max_samples // 4)]

    trivial_iou = compute_trivial_baseline_iou(val_ids, gt_root)
    class_weights = formulation_spec["module"].class_weights_from_labels(
        load_label_triples(train_ids, gt_root),
        max_class_weight=max_class_weight,
    )
    train_free_masks = [
        decompose(*triple)["free"]
        for triple in load_label_triples(train_ids, gt_root)
    ]
    constant_map = constant_free_map(train_free_masks) if train_free_masks else None
    del train_free_masks  # 240x240 bool을 train split 전체만큼 들고 있을 이유가 없다
    constant_baseline = baseline_iou_free(
        val_ids, gt_root, constant_map, device
    )

    _print_banner([
        f" SynWoodScape -> Simple-BEV {formulation} pretrain",
        f" exp_name={exp_name} | encoder={encoder_type} | fisheye={use_fisheye}",
        f" batch_size={batch_size} | lr={lr:.0e} | epochs={num_epochs} | seed={seed}",
        f" formulation={formulation} | loss={loss}",
        f" source_profile={source_profile} | grid={grid_spec.n_rows}x{grid_spec.n_cols}"
        f" @ {grid_spec.cell_m} m/cell | labels={gt_root.name}",
        f" train={len(train_ids)} | val={len(val_ids)}",
        f" class weights ({formulation_spec['weight_label']}) = {class_weights.tolist()}",
        f" photometric augment (train only) = {bool(augment)}",
        f" trivial 'always predict drivable' baseline IoU on val = {trivial_iou:.3f}",
        f" constant-map baseline iou_free = {constant_baseline:.3f}  <- compare against this",
    ])

    train_ds = SynWoodScapeSimpleBEVDataset(
        train_ids, occupancy_gt_root=gt_root, grid_spec=grid_spec, augment=augment)
    val_ds = SynWoodScapeSimpleBEVDataset(
        val_ids, occupancy_gt_root=gt_root, grid_spec=grid_spec)  # val은 항상 원본 -- 증강하면 비교가 흔들린다
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 채택 기본값(`simplebev_vox.DEFAULT_HEIGHT_*`)을 따른다. pretrain은 §4에서 해롭다고
    # 판정돼 확정 경로가 아니지만, 돌아가는 상태는 유지한다.
    Z, Y, X = vox_dims(grid_spec)
    if use_fisheye:
        cameras = [
            load_camera(DEFAULT_DATASET_ROOT / "calibration_data" / f"{name}.json")
            for name in CAMERA_NAMES
        ]
        vox_util = build_fisheye_vox_util(grid_spec, cameras, device=device)
    else:
        vox_util = build_vox_util(grid_spec, device=device)

    # rand_flip=False로 고정한다: Simple-BEV의 forward/backward(Z축) flip 증강은 대칭 grid를
    # 전제하는데 SynWoodScape pretrain grid는 전후 비대칭이라 물리적으로 성립하지 않는다.
    model = ThreeClassSegnet(
        Z, Y, X, vox_util,
        use_radar=False, use_lidar=False, do_rgbcompress=True,
        encoder_type=encoder_type, rand_flip=False,
        num_classes=formulation_spec["num_classes"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, lr, num_epochs * steps_per_epoch + 10,
        pct_start=0.05, cycle_momentum=False, anneal_strategy="linear",
    )
    class_weights = class_weights.to(device)
    rays = (build_ray_index(grid_spec) if n_theta is None
            else build_ray_index(grid_spec, n_theta=n_theta))
    ring_masks = build_ring_masks(grid_spec, edges_m=ring_edges_m)
    # step 구성은 `tools/train_robot_bev.py`와 **같은 모양으로 유지한다** -- 두 스크립트가
    # 다른 경로로 같은 loss를 계산하기 시작하면 사전학습과 fine-tuning이 조용히 갈린다.
    if loss == "soft_boundary":
        # SynWoodScape에는 리그 고정 가림(`permanent_blind`)이 없으므로 0 마스크를 넘긴다.
        blind_mask = torch.zeros(1, 1, grid_spec.n_rows, grid_spec.n_cols)
        gather = (RayGather(rays, (grid_spec.n_rows, grid_spec.n_cols), device)
                  if float(lambda_r) > 0.0 else None)

        def step(batch):
            return binary_metrics.run_batch_soft_boundary(
                model, batch, vox_util, device, rays, blind_mask,
                delta=delta_m, lambda_b=lambda_b, target=soft_target,
                sigma=sigma_m, alpha=None,
                gather=gather, lambda_r=float(lambda_r),
                range_loss_mode=range_loss_mode,
                delta_r=float(delta_r_m), huber_beta=float(huber_beta_m),
                delta_r_over=None,
                kappa=float(band_kappa), eps=float(label_eps),
            )
    elif formulation == "binary":
        step = lambda batch: binary_metrics.run_batch(  # noqa: E731
            model, batch, vox_util, class_weights, device, rays)
    else:
        step = lambda batch: three_class_metrics.run_batch(  # noqa: E731
            model, batch, vox_util, class_weights, device)

    # 타임스탬프를 반드시 넣는다: 이게 없으면 같은 exp_name으로 재실행할 때 global_step(epoch)이
    # 1부터 다시 시작하면서 이전 실행의 체크포인트(model-000000001.pth 등)를 그대로 덮어쓰고,
    # keep_latest 정리 로직이 이전 실행분을 지워버린다 (model_best는 keep_latest=1이라
    # 새 실행의 첫 저장에서 즉시 삭제됨). tensorboard 로그는 파일 자체가 지워지진 않지만
    # 같은 폴더에 섞여 들어가 epoch 축이 겹쳐 보인다.
    if run_name in (None, "None", ""):
        run_name = (f"{exp_name}_{encoder_type}_bs{batch_size}_lr{lr:.0e}_s{seed}"
                    f"_{datetime.now().strftime('%y%m%d_%H%M%S')}")
    log_path = Path(log_dir) / run_name
    writer = SummaryWriter(str(log_path))
    ckpt_path = Path(ckpt_dir) / run_name

    # **이 런이 무엇이었는지 폴더만 보고 알 수 있어야 한다.** 예전에는 이 스크립트가
    # config를 아무것도 남기지 않아서, 사전학습 팔이 둘 이상이 되는 순간 구분이 불가능했다
    # (fine-tuning 쪽 `train_robot_bev.py`는 처음부터 남기고 있었다).
    log_path.mkdir(parents=True, exist_ok=True)
    (log_path / "config.json").write_text(json.dumps({
        "run_name": run_name, "exp_name": exp_name, "seed": seed,
        "source_profile": source_profile,
        "grid": {"front_m": grid_spec.front_m, "rear_m": grid_spec.rear_m,
                 "half_width_m": grid_spec.half_width_m, "cell_m": grid_spec.cell_m,
                 "shape": [grid_spec.n_rows, grid_spec.n_cols]},
        "occupancy_gt_root": str(gt_root),
        "manual_refinement": profile["manual_refinement"],
        "ring_edges_m": list(ring_edges_m),
        "formulation": formulation, "loss": loss,
        "delta_m": delta_m, "sigma_m": sigma_m, "lambda_b": lambda_b,
        "lambda_r": lambda_r,
        # [2026-09-22] 플래그로 노출된다(그 전 런은 전부 `arc_huber`였고 이 칸이
        # 하드코딩돼 있었다). `λ_R=0`이면 항 자체가 만들어지지 않으므로 이 값은 무의미하다.
        "range_loss_mode": range_loss_mode,
        "range_loss_active": bool(float(lambda_r) > 0.0 and loss == "soft_boundary"),
        "delta_r_m": delta_r_m, "huber_beta_m": huber_beta_m,
        "soft_target": soft_target, "band_kappa": band_kappa, "label_eps": label_eps,
        "encoder_type": encoder_type, "use_fisheye": use_fisheye, "augment": augment,
        "num_epochs": num_epochs, "batch_size": batch_size, "lr": lr,
        "weight_decay": weight_decay, "max_class_weight": max_class_weight,
        "height_bins": Y, "height_min_m": DEFAULT_HEIGHT_MIN_M,
        "height_max_m": DEFAULT_HEIGHT_MAX_M,
        "camera_names": list(CAMERA_NAMES), "resize_wh": [RESIZE_WIDTH, RESIZE_HEIGHT],
        "n_train": len(train_ids), "n_val": len(val_ids), "split_seed": split_seed,
        "val_fraction": val_fraction,
    }, indent=2, ensure_ascii=False) + "\n")

    # 이번 실행에 실제로 쓰인 train/val sample id를 파일로 남긴다 -- split은 폴더 구조가
    # 아니라 코드(synwoodscape_split.py)로 계산되므로, 이 파일이 없으면 어떤 이미지가
    # train/val인지 육안으로 확인할 방법이 없다.
    # log_path(텐서보드 폴더)에 쓴다 -- ckpt_path에 쓰면 saverloader.load()가 그 폴더의
    # 파일 전체를 "{model_name}-{step}.pth" 패턴으로 가정하고 os.listdir 하다가
    # split_*.txt에서 IndexError로 죽는다(재현 확인함).
    log_path.mkdir(parents=True, exist_ok=True)
    (log_path / "split_train_ids.txt").write_text("\n".join(train_ids) + "\n")
    (log_path / "split_val_ids.txt").write_text("\n".join(val_ids) + "\n")
    print(f"train/val sample id lists saved to: {log_path}/split_{{train,val}}_ids.txt")

    # **보조항이 실제로 무엇으로 켜졌는지 로그 한 줄로 못박는다.** config.json만으로는
    # 나중에 로그를 grep해서 확인할 수 없고, 사전학습이 미세조정과 다른 보조항으로
    # 돌아가는 사고가 조용히 지나간다(2026-09-22에 이 스크립트가 그 상태였다).
    if loss == "soft_boundary":
        print("[loss] " + (
            f"soft_boundary | delta={delta_m} sigma={sigma_m} lambda_b={lambda_b}"
            + (f" | L_range ON: mode={range_loss_mode} lambda_r={lambda_r}"
               + (f" delta_r={delta_r_m} beta={huber_beta_m}"
                  if range_loss_mode == "arc_huber" else "")
               if float(lambda_r) > 0.0 else " | L_range OFF (lambda_r=0)")))

    global_step = 0
    best_val_score = 0.0  # iou_free; free 영역을 과대/과소 예측한 퇴행 해를 벌한다
    interrupted = False
    try:
        for epoch in range(1, num_epochs + 1):
            model.train()
            epoch_start = time.time()
            train_losses, train_parts_dicts, train_free_metric_dicts = [], [], []
            for batch in train_loader:
                optimizer.zero_grad()
                loss, loss_parts, free_metrics = step(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()

                train_losses.append(loss.item())
                train_parts_dicts.append({k: v.item() for k, v in loss_parts.items()})
                append_free_metrics(train_free_metric_dicts, free_metrics)
                writer.add_scalar("train/loss_step", loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                global_step += 1

            train = {
                "loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "loss_parts": mean_loss_parts(train_parts_dicts),
                "free": summarize_free_metrics(train_free_metric_dicts),
            }
            write_epoch_scalars(writer, "train", train, epoch)

            val = empty_epoch_metrics()
            if epoch % val_freq_epochs == 0 and len(val_loader) > 0:
                model.eval()
                val = evaluate_split(step, val_loader, device, rays, ring_masks,
                                     cell_m=grid_spec.cell_m,
                                     range_edges_m=PRETRAIN_RING_EDGES_M)
                write_epoch_scalars(writer, "val", val, epoch)

            epoch_time = time.time() - epoch_start
            val_score = select_checkpoint_score(val["free"])
            is_new_best = val_score > best_val_score  # NaN > x is always False -- val을 안 돌린 epoch은 자동으로 제외됨

            print(format_epoch_log(
                epoch=epoch,
                num_epochs=num_epochs,
                epoch_time=epoch_time,
                train_loss=train["loss"],
                train_loss_parts=train["loss_parts"],
                train_free_metrics=train["free"],
                val_loss=val["loss"],
                val_loss_parts=val["loss_parts"],
                val_free_metrics=val["free"],
                val_range_metrics=val["range"],
                val_tolerance_metrics=val["tolerance"],
                baseline_iou_free=constant_baseline,
                val_score=val_score,
                best_val_score=best_val_score,
                is_new_best=is_new_best,
            ))

            if epoch % save_freq_epochs == 0 or epoch == num_epochs:
                save_height_config(ckpt_path, Y, DEFAULT_HEIGHT_MIN_M, DEFAULT_HEIGHT_MAX_M)
                saverloader.save(str(ckpt_path), optimizer, model, epoch,
                                 keep_latest=keep_checkpoints)
            if is_new_best:
                best_val_score = val_score
                saverloader.save(str(ckpt_path), optimizer, model, epoch, keep_latest=1, model_name="model_best")
    except KeyboardInterrupt:
        # Ctrl+C는 정상적인 중단 방법이다 -- 지금까지 저장된 체크포인트는 그대로 안전하게
        # 남아 있다(에폭이 끝난 시점에만 저장하므로 반쪽짜리 체크포인트는 생기지 않는다).
        # 다만 finally 없이 여기서 그냥 죽으면 writer.close()가 안 불려서 마지막 몇 개
        # tensorboard scalar가(기본 flush_secs만큼) 디스크에 안 쓰인 채 유실될 수 있다.
        interrupted = True
        print("\n" + _c(_Ansi.YELLOW + _Ansi.BOLD, "[interrupted] Training stopped by Ctrl+C. Checkpoints/logs saved so far are safe."))
    finally:
        writer.close()

    if not interrupted:
        _print_banner([
            " done.",
            f" best val iou_free = {best_val_score:.3f}",
            f" trivial 'always predict drivable' baseline was drivable IoU {trivial_iou:.3f}, obstacle IoU 0.0",
        ])
    else:
        print(f"best val iou_free so far = {best_val_score:.3f}")


if __name__ == "__main__":
    Fire(main)
