"""학습된 SynWoodScape `Segnet` 체크포인트로 추론하고, GT occupancy와 나란히 시각화한다.

ROADMAP Phase 3.3 성공 기준("예측 BEV가 GT와 육안으로도 정합한다")을 실제로 확인하는
스크립트 -- 지금까지는 IoU 숫자만 봤고 육안 비교 코드는 없었다.

카메라 썸네일(4-cam) + [GT occupancy | 예측 occupancy] BEV 지도를 나란히 붙인 PNG를
`<sample_id>_compare.png`로 저장한다. 색은 `tools/build_hybrid_occupancy.py`의
`_combined.png`와 같은 팔레트를 쓴다(회색=미관측, 초록=drivable, 빨강=obstacle) --
`dataset/synwoodscape_occupancy_gt/<sample_id>_combined.png`와 바로 비교 가능하다.

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
from nets.segnet import Segnet  # noqa: E402

from projects.datasets.simplebev_vox import build_vox_util  # noqa: E402
from projects.datasets.synwoodscape_simplebev import (  # noqa: E402
    CAMERA_NAMES,
    DEFAULT_DATASET_ROOT,
    GRID_SPEC,
    SynWoodScapeSimpleBEVDataset,
)
from projects.datasets.synwoodscape_split import discover_all_sample_ids, train_val_split  # noqa: E402
from projects.geometry.fisheye import load_camera  # noqa: E402
from projects.models.fisheye_vox import build_fisheye_vox_util  # noqa: E402

# tools/build_hybrid_occupancy.py의 GT combined.png와 같은 팔레트 -- 나란히 비교하기 쉽게 유지.
COLOR_UNKNOWN = (128, 128, 128)
COLOR_DRIVABLE = (76, 175, 80)
COLOR_OBSTACLE = (192, 57, 43)

CELL_UPSCALE = 3  # 160x160 grid -> 480x480 (NEAREST라 셀 경계가 흐려지지 않음)
CAM_THUMB_W, CAM_THUMB_H = 220, 166


def occupancy_to_image(occupancy: np.ndarray, observed: np.ndarray) -> Image.Image:
    image = np.full(occupancy.shape + (3,), COLOR_UNKNOWN, dtype=np.uint8)
    image[observed & (occupancy == 1)] = COLOR_DRIVABLE
    image[observed & (occupancy == 0)] = COLOR_OBSTACLE
    h, w = occupancy.shape
    return Image.fromarray(image).resize((w * CELL_UPSCALE, h * CELL_UPSCALE), Image.NEAREST)


def compute_iou(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    intersection = (pred * target * valid).sum()
    union = np.clip(pred + target, 0, 1) * valid
    return float(intersection / max(union.sum(), 1e-4))


def build_panel(rgb_camXs_01: np.ndarray, camera_names, gt_image, pred_image, title_lines) -> Image.Image:
    cams_row = Image.new("RGB", (CAM_THUMB_W * len(camera_names), CAM_THUMB_H))
    draw_cams = ImageDraw.Draw(cams_row)
    for i, name in enumerate(camera_names):
        arr = (rgb_camXs_01[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        thumb = Image.fromarray(arr).resize((CAM_THUMB_W, CAM_THUMB_H))
        cams_row.paste(thumb, (i * CAM_THUMB_W, 0))
        draw_cams.text((i * CAM_THUMB_W + 5, 5), name, fill=(255, 255, 0))

    gap = 20
    bev_row_w = gt_image.width + pred_image.width + gap
    bev_row_h = max(gt_image.height, pred_image.height)
    bev_row = Image.new("RGB", (bev_row_w, bev_row_h), (255, 255, 255))
    bev_row.paste(gt_image, (0, 0))
    bev_row.paste(pred_image, (gt_image.width + gap, 0))
    draw_bev = ImageDraw.Draw(bev_row)
    draw_bev.text((5, 5), "GT", fill=(255, 255, 0))
    draw_bev.text((gt_image.width + gap + 5, 5), "prediction", fill=(255, 255, 0))

    text_h = 20 * (len(title_lines) + 1)
    total_w = max(cams_row.width, bev_row.width)
    total_h = cams_row.height + bev_row.height + text_h
    panel = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    panel.paste(cams_row, (0, 0))
    panel.paste(bev_row, (0, cams_row.height))
    draw = ImageDraw.Draw(panel)
    for i, line in enumerate(title_lines):
        draw.text((5, cams_row.height + bev_row.height + i * 20), line, fill=(0, 0, 0))
    return panel


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

    Z, Y, X = GRID_SPEC.n_rows, 1, GRID_SPEC.n_cols
    if use_fisheye:
        cameras = [
            load_camera(DEFAULT_DATASET_ROOT / "calibration_data" / f"{name}.json")
            for name in CAMERA_NAMES
        ]
        vox_util = build_fisheye_vox_util(GRID_SPEC, cameras, device=device)
    else:
        vox_util = build_vox_util(GRID_SPEC, device=device)

    model = Segnet(
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
            _, _, seg_bev_e, _, _ = model(rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util)
        pred_np = torch.sigmoid(seg_bev_e)[0, 0].round().cpu().numpy().astype(np.uint8)

        occupancy_np = item["seg_bev_g"][0].numpy().astype(np.uint8)
        visible_np = item["valid_bev_g"][0].numpy().astype(bool)

        d_iou = compute_iou(pred_np, occupancy_np, visible_np)
        o_iou = compute_iou(1 - pred_np, 1 - occupancy_np, visible_np)

        gt_image = occupancy_to_image(occupancy_np, visible_np)
        # 예측도 GT와 같은 관측 마스크로 그린다 -- 모델이 실제로 "관측된" 영역에서 얼마나
        # 맞는지를 GT와 동일 조건으로 비교하기 위함(미관측 영역 예측은 회색으로 가림).
        pred_image = occupancy_to_image(pred_np, visible_np)

        panel = build_panel(
            item["rgb_camXs"].numpy(), CAMERA_NAMES, gt_image, pred_image,
            [f"{sample_id}   drivable IoU {d_iou:.3f}   obstacle IoU {o_iou:.3f}"],
        )
        panel.save(output_dir / f"{sample_id}_compare.png")
        print(f"{sample_id}: drivable IoU {d_iou:.3f} obstacle IoU {o_iou:.3f} -> {output_dir}/{sample_id}_compare.png")


if __name__ == "__main__":
    Fire(main)
