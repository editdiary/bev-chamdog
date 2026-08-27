"""`λ_R`을 추측하지 않고 **gradient 비로 캘리브레이션한다** (설계 문서 §13).

**왜 필요한가.** `L_range`는 미터 단위이고 `L_BCE`는 nats다. 두 항의 절대값에 공통 스케일이
없으므로 `λ_R`을 숫자로 고르면 **그 숫자의 뜻이 정해지지 않는다.** 게다가 세 개의 축소 인자가
곱셈으로 겹친다 -- dead zone이 오차 대부분을 없애고, Huber가 작은 오차를 이차로 줄이고,
`λ_R`이 다시 곱해진다. 눈으로 고른 `λ_R = 0.05`가 실제로는 gradient의 0.01 %일 수도 있고
그러면 "보조항이 효과 없었다"가 아니라 **"보조항이 사실상 꺼져 있었다"**로 끝난다.

그래서 재는 것은 하나다.

    G_R / G_B  =  ‖∂(λ_R·L_range)/∂logits‖ / ‖∂L_soft-boundary/∂logits‖

`logits`에서 재는 이유: 두 항이 만나는 **유일한 공통 지점**이고, 그 아래(encoder)로는
chain rule이 같은 인자를 곱하므로 비율이 보존된다. 파라미터 gradient에서 재면 optimizer
상태·layer별 스케일이 섞인다.

**λ_R = 1로 재고 나눈다.** `L_range`가 `λ_R`에 선형이므로 한 번 재면 원하는 목표비의
`λ_R`이 나눗셈 한 번으로 나온다.

    λ_R = 목표비 / 측정비(λ_R = 1)

`G_R/G_B ~ 0.05~0.2`면 보조항이라고 부를 수 있다. 그보다 크면 주 loss를 밀어내고, 훨씬
작으면 꺼져 있는 것과 같다.

실행:

    python tools/measure_range_gradient.py --init_checkpoint=none --num_batches=20

**초기화 상태에서 재는 것의 한계를 알아 둔다.** 학습이 진행되면 `L_BCE`의 gradient는 줄고
`L_range`는 dead zone 때문에 더 빨리 0에 붙는다. 즉 이 비율은 **학습 초기의 값**이다. 그래서
학습 중 실제 몫은 `share_r`(콘솔 표)로 계속 확인해야 한다 -- 이 도구는 `λ_R`의 자릿수를
정하는 데 쓴다.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.models\._utils")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.common.free_space import decompose  # noqa: E402
from projects.common.polar import build_ray_index  # noqa: E402
from projects.common.range_loss import (  # noqa: E402
    DEFAULT_HUBER_BETA_M,
    RayGather,
    compute_range_loss,
)
from projects.common.soft_boundary import compute_soft_boundary_loss  # noqa: E402
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    FINETUNE_CAMERA_NAMES,
    GRID_SPEC,
    RobotBEVDataset,
    build_bev_masks,
    parse_sequence_names,
    split_samples_by_sequence,
)
from projects.datasets.simplebev_vox import (  # noqa: E402
    DEFAULT_HEIGHT_BINS,
    DEFAULT_HEIGHT_MAX_M,
    DEFAULT_HEIGHT_MIN_M,
    height_config_for_ckpt_dirs,
    vox_dims,
)
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.simplebev_three_class import (  # noqa: E402
    ThreeClassSegnet,
    load_trunk_weights,
)


def _from_scratch(init_checkpoint) -> bool:
    """`tools/train_robot_bev.py:from_scratch`와 같은 규약. 두 곳에서 갈리면 안 된다."""
    return init_checkpoint is None or str(init_checkpoint).strip().lower() in ("", "none", "no")


def _grad_norm(loss, logits):
    """`‖∂loss/∂logits‖₂`. 그래프를 유지해 두 항을 같은 forward에서 잰다."""
    (grad,) = torch.autograd.grad(loss, logits, retain_graph=True, allow_unused=False)
    return float(grad.norm())


def main(
    train_sequences="raws2,raws3,rawos1,rawos2,rawos4",
    val_sequences="raws1,rawos3",
    dataset_root=DEFAULT_DATASET_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    init_checkpoint="none",
    encoder_type="res101",
    batch_size=8,
    num_batches=20,
    num_workers=8,
    # 재는 대상 config. 스윕의 베이스와 같아야 한다 -- 다른 `δ`/`α`에서 잰 비율을 쓰면
    # `λ_R`이 그만큼 어긋난다.
    delta_m=0.15,
    lambda_b=0.5,
    soft_target="gaussian",
    sigma_alpha=0.5,
    delta_r_m=0.20,
    huber_beta_m=DEFAULT_HUBER_BETA_M,
    target_ratio=0.1,
    seed=0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_root = Path(dataset_root)
    names = parse_sequence_names(train_sequences)
    val_names = parse_sequence_names(val_sequences)
    roots = [dataset_root / name for name in names + val_names]
    train_samples, _ = split_samples_by_sequence(roots, val_names)
    permanent_blind, invalid = build_bev_masks(common_root, GRID_SPEC, FINETUNE_CAMERA_NAMES)

    dataset = RobotBEVDataset(train_samples, common_root=common_root, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers)

    # 체크포인트가 주어지면 **그것이 학습된 높이 설정**을 따라야 한다. 없으면(=랜덤 초기화로
    # gradient 비만 보는 경우) 현재 채택 기본값을 쓴다.
    _height = ({"height_bins": DEFAULT_HEIGHT_BINS, "height_min_m": DEFAULT_HEIGHT_MIN_M,
                "height_max_m": DEFAULT_HEIGHT_MAX_M} if _from_scratch(init_checkpoint)
               else height_config_for_ckpt_dirs([Path(init_checkpoint).parent]))
    Z, Y, X = vox_dims(GRID_SPEC, _height["height_bins"])
    vox_util = build_double_sphere_vox_util(
        GRID_SPEC, dataset.cameras, device=device,
        height_bins=_height["height_bins"], height_min_m=_height["height_min_m"],
        height_max_m=_height["height_max_m"])
    model = ThreeClassSegnet(Z, Y, X, vox_util, use_radar=False, use_lidar=False,
                             do_rgbcompress=True, encoder_type=encoder_type,
                             rand_flip=False, num_classes=2).to(device)
    if not _from_scratch(init_checkpoint):
        load_trunk_weights(model, init_checkpoint, device)
    model.train()

    rays = build_ray_index(GRID_SPEC)
    gather = RayGather(rays, (GRID_SPEC.n_rows, GRID_SPEC.n_cols), device)
    blind = torch.from_numpy(permanent_blind).view(1, 1, *permanent_blind.shape).to(device)

    ratios, mae, bias, used = [], [], [], []
    for index, batch in enumerate(loader):
        if index >= num_batches:
            break
        rgb = batch["rgb_camXs"].to(device) - 0.5
        seg = batch["seg_bev_g"].to(device)
        vis = batch["vis_bev_g"].to(device)
        valid = batch["valid_bev_g"].to(device)
        d = batch["d_bev_g"].to(device)

        _, _, logits, _, _ = model(rgb, batch["pix_T_cams"].to(device),
                                   batch["cam0_T_camXs"].to(device), vox_util)
        logits.retain_grad()

        boundary_loss, _ = compute_soft_boundary_loss(
            logits, d, valid, blind, delta=delta_m, lambda_b=lambda_b,
            kind=soft_target, alpha=sigma_alpha,
        )
        range_loss, range_parts = compute_range_loss(
            torch.softmax(logits, dim=1)[:, 1:2],
            decompose(seg, vis, valid)["free"], valid, gather,
            delta_r=delta_r_m, beta=huber_beta_m,
        )
        g_b = _grad_norm(boundary_loss, logits)
        g_r = _grad_norm(range_loss, logits)          # λ_R = 1에서의 값
        ratios.append(g_r / g_b if g_b > 0 else float("nan"))
        mae.append(float(range_parts["range_arc_mae"].detach()))
        bias.append(float(range_parts["range_arc_bias"].detach()))
        used.append(float(range_parts["frac_rays_used"].detach()))

    ratio = float(np.mean(ratios))
    print(f"\n배치 {len(ratios)}개 / {device} / init={init_checkpoint}")
    print(f"  base config: δ={delta_m} λ_B={lambda_b} target={soft_target} α={sigma_alpha}")
    print(f"               δ_R={delta_r_m} m  β={huber_beta_m} m")
    print(f"\n  arc_mae        {np.mean(mae):.4f} m  (dead zone 전)")
    print(f"  arc_bias      {np.mean(bias):+.4f} m  (양수 = 자유공간 과대예측)")
    print(f"  frac_rays_used {np.mean(used):.4f}")
    print(f"\n  G_R/G_B (λ_R=1) {ratio:.4f}  (배치간 σ {np.std(ratios):.4f})")
    if ratio > 0:
        print(f"\n  => λ_R = {target_ratio} / {ratio:.4f} = "
              f"{target_ratio / ratio:.4f}   (목표비 {target_ratio})")
    else:
        print("\n  => G_R = 0이다. dead zone이 모든 광선을 덮었거나 δ_R이 너무 크다.")


if __name__ == "__main__":
    Fire(main)
