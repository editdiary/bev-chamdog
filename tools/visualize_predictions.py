"""학습된 SynWoodScape `ThreeClassSegnet` 체크포인트로 추론하고, occupancy/visibility를 시각화한다.

ROADMAP Phase 3.3 성공 기준("예측 BEV가 GT와 육안으로도 정합한다")을 실제로 확인하는
스크립트 -- 지금까지는 IoU 숫자만 봤고 육안 비교 코드는 없었다.

카메라 썸네일(4-cam) + GT/pred occupancy, GT/pred visibility, pred occupancy masked by
pred visibility를 나란히 붙인 PNG를 `<sample_id>_compare.png`로 저장한다. 개별 확인을 위해
pred occupancy/visibility/combined PNG도 함께 저장한다.

Run:
    python tools/visualize_predictions.py \\
        --ckpt_dir=work_dirs/checkpoints_synwoodscape/baseline_res101_bs4_lr3e-04_20260810_120000 \\
        --output_dir=work_dirs/viz_baseline
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from PIL import Image, ImageDraw

# Segnet(Encoder_res101/50)이 내부에서 쓰는 torchvision.models.resnet*(pretrained=True)의
# deprecated 인자 경고 -- submodule 코드라 직접 못 고치므로 여기서 억제한다(동작 영향 없음).
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.models\._utils")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

import saverloader  # noqa: E402  (simple_bev submodule; see docs/project_structure.md)
from projects.datasets.simplebev_vox import build_vox_util  # noqa: E402
from projects.datasets.synwoodscape_simplebev import (  # noqa: E402
    CAMERA_NAMES,
    DEFAULT_DATASET_ROOT,
    GRID_SPEC,
    SynWoodScapeSimpleBEVDataset,
)
from projects.datasets.synwoodscape_split import discover_all_sample_ids, train_val_split  # noqa: E402
from projects.geometry.fisheye import load_camera  # noqa: E402
from projects.datasets.simplebev_vox import (  # noqa: E402
    height_config_for_ckpt_dirs,
    vox_dims,
)
from projects.models.fisheye_vox import build_fisheye_vox_util  # noqa: E402
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402
from projects.common.free_space import decompose_from_class_index  # noqa: E402

from projects.common.bev_panels import (  # noqa: E402
    CELL_UPSCALE,
    COLOR_DRIVABLE,
    COLOR_OBSTACLE,
    COLOR_UNKNOWN,
    COLOR_VISIBLE,
    build_panel,
    compute_iou,
    format_obstacle_iou,
    occupancy_to_image,
    visibility_to_image,
)



def observed_error_rates(pred_observed, gt_observed, valid) -> dict:
    """모델이 "보인다"고 부른 영역의 오류율. 삭제된 2-head `visibility_error_rates`와 같은 정의다.

    `false_high`는 GT가 미관측인 셀을 관측이라 부른 비율(과신), `false_low`는 그 반대다.
    3-class에서 "관측"은 `free | occupied`이므로 argmax 결과에서 그대로 유도된다.
    """
    valid = valid.astype(bool)
    pred_observed = pred_observed.astype(bool)
    gt_observed = gt_observed.astype(bool)
    false_high_den = max(int(((~gt_observed) & valid).sum()), 1)
    false_low_den = max(int((gt_observed & valid).sum()), 1)
    return {
        "false_high": float((pred_observed & ~gt_observed & valid).sum()) / false_high_den,
        "false_low": float((~pred_observed & gt_observed & valid).sum()) / false_low_den,
    }


def main(
    ckpt_dir,
    model_name="model_best",
    step=0,
    sample_ids=None,
    num_samples=8,
    val_fraction=0.1,
    split_seed=0,
    encoder_type="res101",
    use_fisheye=True,
    output_dir="work_dirs/viz",
    device="cuda",
):
    """`ckpt_dir`은 `train_synwoodscape.py`가 만든 실행별 체크포인트 폴더(타임스탬프 포함).
    `encoder_type`/`use_fisheye`는 그 체크포인트를 만들 때 쓴 값과 반드시 같아야 한다
    (모델 구조가 달라지면 `load_state_dict`가 깨진다).
    """
    if sample_ids is None:
        # split_seed/val_fraction이 학습 때와 같으면 train_val_split은 결정적이라 항상
        # 같은 val 목록을 재현한다 -- 파일을 따로 안 읽어도 된다.
        all_ids = discover_all_sample_ids(DEFAULT_DATASET_ROOT)
        _, val_ids = train_val_split(all_ids, DEFAULT_DATASET_ROOT, val_fraction=val_fraction, seed=split_seed)
        sample_ids = val_ids[:num_samples]
    else:
        sample_ids = list(sample_ids)
    print(f"visualizing {len(sample_ids)} samples: {sample_ids}")

    _height = height_config_for_ckpt_dirs([Path(ckpt_dir)])
    Z, Y, X = vox_dims(GRID_SPEC, _height["height_bins"])
    _hkw = dict(height_bins=_height["height_bins"],
                height_min_m=_height["height_min_m"], height_max_m=_height["height_max_m"])
    if use_fisheye:
        cameras = [
            load_camera(DEFAULT_DATASET_ROOT / "calibration_data" / f"{name}.json")
            for name in CAMERA_NAMES
        ]
        vox_util = build_fisheye_vox_util(GRID_SPEC, cameras, device=device, **_hkw)
    else:
        vox_util = build_vox_util(GRID_SPEC, device=device, **_hkw)

    model = ThreeClassSegnet(
        Z, Y, X, vox_util,
        use_radar=False, use_lidar=False, do_rgbcompress=True,
        encoder_type=encoder_type, rand_flip=False,
    ).to(device)
    saverloader.load(str(ckpt_dir), model, step=step, model_name=model_name)
    model.eval()

    dataset = SynWoodScapeSimpleBEVDataset(sample_ids)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, sample_id in enumerate(sample_ids):
        item = dataset[i]
        rgb_camXs = item["rgb_camXs"].unsqueeze(0).to(device) - 0.5
        pix_T_cams = item["pix_T_cams"].unsqueeze(0).to(device)
        cam0_T_camXs = item["cam0_T_camXs"].unsqueeze(0).to(device)

        with torch.no_grad():
            _, _, logits, _, _ = model(rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util)
        valid_dev = item["valid_bev_g"].unsqueeze(0).to(device)
        pred_parts = decompose_from_class_index(logits.argmax(dim=1, keepdim=True), valid_dev)
        # 3-class 분해에서 옛 occupancy/visibility 패널이 기대하는 두 마스크를 꺼낸다.
        # 정의상 `vis = free | occupied`(= unknown이 아님)이고 관측 영역 안에서 `drivable = free`다.
        pred_vis_np = (pred_parts["free"] | pred_parts["occupied"])[0, 0].cpu().numpy().astype(bool)
        pred_occ_np = pred_parts["free"][0, 0].cpu().numpy().astype(np.uint8)

        occupancy_np = item["seg_bev_g"][0].numpy().astype(np.uint8)
        visible_np = item["vis_bev_g"][0].numpy().astype(bool)
        valid_np = item["valid_bev_g"][0].numpy().astype(bool)

        d_iou = compute_iou(pred_occ_np, occupancy_np, valid_np)
        o_iou_text = format_obstacle_iou(1 - pred_occ_np, 1 - occupancy_np, valid_np)
        vis_metrics = observed_error_rates(pred_vis_np, visible_np, valid_np)

        all_cells = np.ones_like(visible_np, dtype=bool)
        gt_occ_image = occupancy_to_image(occupancy_np, visible_np)
        pred_occ_image = occupancy_to_image(pred_occ_np, all_cells)
        gt_vis_image = visibility_to_image(visible_np)
        pred_vis_image = visibility_to_image(pred_vis_np)
        pred_combined_image = occupancy_to_image(pred_occ_np, pred_vis_np)

        pred_occ_image.save(output_dir / f"{sample_id}_pred_occupancy.png")
        pred_vis_image.save(output_dir / f"{sample_id}_pred_visibility.png")
        pred_combined_image.save(output_dir / f"{sample_id}_pred_combined.png")

        panel = build_panel(
            item["rgb_camXs"].numpy(),
            CAMERA_NAMES,
            [
                ("GT occupancy", gt_occ_image),
                ("pred occupancy", pred_occ_image),
                ("GT visibility", gt_vis_image),
                ("pred visibility", pred_vis_image),
                ("pred occ x vis", pred_combined_image),
            ],
            [
                f"{sample_id}   occ drivable IoU {d_iou:.3f}   obstacle IoU {o_iou_text}",
                f"visibility false_high {vis_metrics['false_high']:.3f}   false_low {vis_metrics['false_low']:.3f}",
            ],
        )
        panel.save(output_dir / f"{sample_id}_compare.png")
        print(
            f"{sample_id}: drivable IoU {d_iou:.3f} obstacle IoU {o_iou_text} "
            f"vis false_high {vis_metrics['false_high']:.3f} false_low {vis_metrics['false_low']:.3f} "
            f"-> {output_dir}/{sample_id}_compare.png"
        )


if __name__ == "__main__":
    Fire(main)
