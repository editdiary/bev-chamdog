"""자체 수집 데이터셋의 two-head BEV 예측을 패널 이미지로 저장한다.

SynWoodScape 쪽(`tools/visualize_predictions.py`)과 같은 팔레트·배치를 쓰되
(`projects/common/bev_panels.py`), 두 가지가 다르다:

1. **IPM 패널을 함께 그린다.** 자체 데이터셋은 val 샘플이 매우 적어 정량 지표의 노이즈가
   크므로, 실제 판정은 눈으로 하게 된다. 같은 좌표계에 장면을 깔아주면 예측이 통로를
   따라가는지 바로 보인다.
2. **최악 샘플부터 골라볼 수 있다** (`--sort_by`). 전체를 다 볼 수 없을 때 평균 뒤에 숨은
   실패 유형을 먼저 만난다.

실행 예:
    CUDA_VISIBLE_DEVICES=1 python tools/visualize_robot_predictions.py \\
        --ckpt=runs/robot_bev/ckpt/<run>/model_best-000000030.pth \\
        --sort_by=obstacle_iou --limit=8
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from PIL import Image
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.models\._utils")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.bev_gt.grid import ROBOT_GRID_SPEC  # noqa: E402
from projects.bev_gt.ipm import render_ipm  # noqa: E402
from projects.common.bev_panels import (  # noqa: E402
    build_panel,
    compute_iou,
    format_obstacle_iou,
    occupancy_to_image,
    visibility_to_image,
)
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    RobotBEVDataset,
    list_sequence_samples,
    parse_sequence_names,
)
from projects.geometry.double_sphere import (  # noqa: E402
    FINETUNE_CAMERA_NAMES,
    load_cameras,
    load_ego_T_cams,
)
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.simplebev_two_head import TwoHeadSegnet, split_two_head_logits  # noqa: E402

CELL_UPSCALE = 4  # 120x120 -> 480x480. pretrain은 240x240이라 2를 썼다.
CAM_THUMB_WH = (240, 135)  # 16:9 (자체 리그는 1280x720)


def _ipm_to_image(ipm: np.ndarray, upscale: int = CELL_UPSCALE) -> Image.Image:
    h, w = ipm.shape[:2]
    return Image.fromarray(ipm).resize((w * upscale, h * upscale), Image.NEAREST)


def _sample_scores(pred_occ, gt_occ, mask) -> dict:
    """샘플 하나의 정렬 기준. 전부 "클수록 나쁨"으로 맞춰 정렬을 단순하게 둔다."""
    gt_obstacle = (gt_occ == 0) & mask
    pred_obstacle = (pred_occ == 0) & mask
    missed = int((gt_obstacle & ~pred_obstacle).sum())
    false_alarm = int((pred_obstacle & ~gt_obstacle).sum())
    obstacle_iou = compute_iou(pred_obstacle, gt_obstacle, mask) if gt_obstacle.any() else np.nan
    return {
        # GT에 obstacle이 없으면 IoU가 구조적으로 0이라 최악 순위를 독차지한다 -> 제외.
        "obstacle_iou": -obstacle_iou if gt_obstacle.any() else -np.inf,
        "missed_obstacle": missed / max(int(gt_obstacle.sum()), 1),
        "false_obstacle": false_alarm / max(int((~gt_obstacle & mask).sum()), 1),
        "_obstacle_iou_text": format_obstacle_iou(pred_obstacle, gt_obstacle, mask),
        "_missed": missed,
        "_false": false_alarm,
        "_gt_obstacle_cells": int(gt_obstacle.sum()),
    }


def main(
    ckpt,
    sequences="raws1",
    out_dir="runs/robot_bev/viz",
    sort_by="index",
    limit=8,
    batch_size=4,
    num_workers=4,
    encoder_type="res101",
    dataset_root=DEFAULT_DATASET_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    device="cuda",
):
    """`sort_by`: index | obstacle_iou | missed_obstacle | false_obstacle.

    `index`를 빼면 전부 "나쁜 순"이다 -- `obstacle_iou`는 IoU가 낮은 순, 나머지는 오류율이
    높은 순. GT에 obstacle이 없는 샘플은 `obstacle_iou` 정렬에서 제외한다(완벽히 맞혀도
    IoU가 0이라 순위를 독차지한다).
    """
    valid_sorts = {"index", "obstacle_iou", "missed_obstacle", "false_obstacle"}
    if sort_by not in valid_sorts:
        raise ValueError(f"sort_by는 {sorted(valid_sorts)} 중 하나여야 한다: {sort_by!r}")

    dataset_root = Path(dataset_root)
    samples = []
    for name in parse_sequence_names(sequences):
        samples.extend(list_sequence_samples(dataset_root / name))
    if not samples:
        raise FileNotFoundError(f"샘플이 없다: {sequences}")

    dataset = RobotBEVDataset(samples, common_root=common_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    vox_util = build_double_sphere_vox_util(ROBOT_GRID_SPEC, dataset.cameras, device=device)
    model = TwoHeadSegnet(
        ROBOT_GRID_SPEC.n_rows, 1, ROBOT_GRID_SPEC.n_cols, vox_util,
        use_radar=False, use_lidar=False, do_rgbcompress=True,
        encoder_type=encoder_type, rand_flip=False,
    ).to(device)
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    model.eval()

    records = []
    with torch.no_grad():
        for batch in loader:
            _, _, two_head_bev_e, _, _ = model(
                batch["rgb_camXs"].to(device), batch["pix_T_cams"].to(device),
                batch["cam0_T_camXs"].to(device), vox_util,
            )
            occ_logits, vis_logits = split_two_head_logits(two_head_bev_e)
            pred_occ = (torch.sigmoid(occ_logits) > 0.5).cpu().numpy()[:, 0].astype(np.uint8)
            pred_vis = (torch.sigmoid(vis_logits) > 0.5).cpu().numpy()[:, 0]
            for i in range(len(pred_occ)):
                gt_occ = batch["seg_bev_g"][i, 0].numpy().astype(np.uint8)
                gt_vis = batch["vis_bev_g"][i, 0].numpy().astype(bool)
                valid = batch["valid_bev_g"][i, 0].numpy().astype(bool)
                mask = gt_vis & valid
                records.append({
                    "sample_id": batch["sample_id"][i],
                    "rgb": batch["rgb_camXs"][i].numpy(),
                    "pred_occ": pred_occ[i], "pred_vis": pred_vis[i],
                    "gt_occ": gt_occ, "gt_vis": gt_vis, "valid": valid, "mask": mask,
                    **_sample_scores(pred_occ[i], gt_occ, mask),
                })

    if sort_by != "index":
        records.sort(key=lambda r: r[sort_by], reverse=True)
    selected = records[:limit] if limit else records

    cameras = load_cameras(Path(common_root) / "calibration/calib.yaml")
    ego_T_cams = load_ego_T_cams(Path(common_root) / "calibration/calib.yaml")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for rank, record in enumerate(selected):
        sequence_name, sample_id = record["sample_id"].split("/")
        images = {
            name: np.asarray(Image.open(
                dataset_root / sequence_name / "rgb_images" / sample_id / f"cam_{name}.jpg"
            ).convert("RGB"))
            for name in FINETUNE_CAMERA_NAMES
        }
        ipm = render_ipm(images, cameras, ego_T_cams, ROBOT_GRID_SPEC)

        drivable_iou = compute_iou(
            (record["pred_occ"] == 1), (record["gt_occ"] == 1), record["mask"]
        )
        panel = build_panel(
            record["rgb"], FINETUNE_CAMERA_NAMES,
            [
                ("IPM (actual scene)", _ipm_to_image(ipm)),
                ("GT occupancy", occupancy_to_image(record["gt_occ"], record["mask"], CELL_UPSCALE)),
                ("pred occupancy", occupancy_to_image(record["pred_occ"], record["mask"], CELL_UPSCALE)),
                ("GT visibility", visibility_to_image(record["gt_vis"], CELL_UPSCALE)),
                ("pred visibility", visibility_to_image(record["pred_vis"] & record["valid"], CELL_UPSCALE)),
            ],
            [
                f"{record['sample_id']}   (sort_by={sort_by}, rank {rank + 1}/{len(selected)})",
                f"obstacle IoU: {record['_obstacle_iou_text']}   drivable IoU: {drivable_iou:.3f}",
                f"GT obstacle cells: {record['_gt_obstacle_cells']}   "
                f"missed: {record['_missed']}   false alarm: {record['_false']}",
                "grey = excluded from loss (permanent blind vis=0 + collection artifact valid=0)",
            ],
            cam_thumb_wh=CAM_THUMB_WH,
        )
        name = f"{rank:02d}_{sequence_name}_{sample_id}.png" if sort_by != "index" \
            else f"{sequence_name}_{sample_id}.png"
        panel.save(out_path / name)

    print(f"{len(selected)} panels -> {out_path}")


if __name__ == "__main__":
    Fire(main)
