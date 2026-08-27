"""자체 수집 데이터셋의 3-class BEV 예측을 패널 이미지로 저장한다.

**샘플별 숫자는 학습 로그와 같은 함수로 계산한다** (`free_space_metrics`, `occupied_metrics`).
시각화가 자기만의 지표를 다시 구현하면 그림과 로그가 다른 말을 하게 되고, 그러면 어느 쪽을
믿어야 할지 알 수 없다. 그래서 여기서는 마스크를 만들어 넘기는 일만 한다.

패널 구성 (`projects/common/three_class_panels.py`):

    IPM (실제 장면)   |   GT 3-class
    pred 3-class      |   오차 지도

`occupancy`/`visibility`를 따로 그리지 않는다 -- 3-class 분해가 그 둘을 이미 담고 있어
(`vis = free | occupied`, 관측 영역 안에서 `drivable = free`) 같은 정보를 세 번 보는 셈이었다.

**IPM 패널을 함께 그리는 이유**: val 샘플이 75장뿐이라 정량 지표의 노이즈가 크고 실제 판정은
눈으로 하게 된다. 같은 좌표계에 장면을 깔아주면 예측이 통로를 따라가는지 바로 보인다.

**최악 샘플부터 골라볼 수 있다** (`--sort_by`). 전체를 다 볼 수 없을 때 평균 뒤에 숨은 실패
유형을 먼저 만난다.

실행 예:
    CUDA_VISIBLE_DEVICES=0 python tools/visualize_robot_predictions.py \\
        --ckpt=runs/robot_bev/ckpt/<run>/model_best-000000033.pth \\
        --sequences=raws1,rawos3 --sort_by=fatal_rate --limit=8
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
from projects.common import binary_metrics  # noqa: E402
from projects.common.free_space import decompose, decompose_from_class_index  # noqa: E402
from projects.common.free_space_metrics import (  # noqa: E402
    fatal_rate,
    free_miss_rate,
    iou_free,
    range_error,
)
from projects.common.occupied_metrics import (  # noqa: E402
    summarize_tolerance_f1,
    tolerance_counts,
)
from projects.common.polar import RAY_OK, build_ray_index, first_free_range  # noqa: E402
from projects.common.three_class_panels import (  # noqa: E402
    CLASS_COLOURS,
    ERROR_COLOURS,
    RANGE_COLOURS,
    build_panel,
    draw_range_profile,
    render_classes,
    render_errors,
    upscale,
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
from projects.datasets.simplebev_vox import height_config_for_ckpt_dirs  # noqa: E402
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.pixel_grid import convention_for_checkpoints  # noqa: E402
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402

CELL_UPSCALE = 5  # 120x120 -> 600x600
TOLERANCE_M = 0.20  # 패널에 적을 대표 tolerance. 전체 집합은 `occupied_metrics`가 정한다

# `--sort_by`: 값이 "클수록 나쁨"이 되도록 부호를 맞춰 두면 정렬이 한 줄로 끝난다.
_SORT_KEYS = {
    "index": lambda s: 0.0,
    "fatal_rate": lambda s: s["fatal_rate"],
    "free_miss_rate": lambda s: s["free_miss_rate"],
    "iou_free": lambda s: -s["iou_free"],
    "f1_occupied": lambda s: -s["f1_occupied"],
    "range_mae": lambda s: s["range_mae"],
}

LEGEND_CLASSES = [
    ("free", CLASS_COLOURS["free"]),
    ("occupied", CLASS_COLOURS["occupied"]),
    ("unknown (vis=0)", CLASS_COLOURS["unknown"]),
    ("invalid (valid=0, loss 제외)", CLASS_COLOURS["invalid"]),
]
LEGEND_ERRORS = [
    ("fatal: pred free, GT 아님", ERROR_COLOURS["fatal"]),
    ("miss: GT free, pred 아님", ERROR_COLOURS["miss"]),
    ("occupied<->unknown 혼동", ERROR_COLOURS["occ_unknown"]),
    ("correct", ERROR_COLOURS["correct"]),
]


def _nan_to_inf(value: float) -> float:
    """정렬에서 NaN이 임의의 위치로 가지 않게 한다 -- 잴 수 없는 샘플은 맨 뒤로 보낸다."""
    return -np.inf if np.isnan(value) else value


def sample_scores(pred_parts, gt_parts, valid, rays, cell_m) -> dict:
    """샘플 하나의 지표. **학습 로그와 같은 함수만 부른다.**"""
    pred_free, gt_free = pred_parts["free"], gt_parts["free"]
    iou, _ = iou_free(pred_free, gt_free, valid)
    fatal, _ = fatal_rate(pred_free, gt_free, valid)
    miss, _ = free_miss_rate(pred_free, gt_free, valid)
    tolerance = summarize_tolerance_f1([tolerance_counts(
        pred_parts["occupied"], gt_parts["occupied"], valid, cell_m,
        tolerances=(TOLERANCE_M,),
    )])
    ranges = range_error(pred_free, gt_free, valid, rays)
    f1 = next(iter(tolerance.values()))["f1"] if tolerance else float("nan")
    return {
        "iou_free": iou, "fatal_rate": fatal, "free_miss_rate": miss,
        "f1_occupied": f1,
        "range_mae": ranges["mae"],
        "missed_obstacle_rate": ranges["missed_obstacle_rate"],
        "n_paired_rays": ranges["n_paired_rays"],
    }


def _metric_lines(scores) -> list:
    def fmt(value, digits=3):
        return "  n/a" if np.isnan(value) else f"{value:.{digits}f}"

    tolerance_label = f"f1@{round(TOLERANCE_M * 100)}cm"
    return [
        f"iou_free {fmt(scores['iou_free'])}   fatal {fmt(scores['fatal_rate'])}"
        f"   free_miss {fmt(scores['free_miss_rate'])}",
        f"{tolerance_label} {fmt(scores['f1_occupied'])}   (경계 정밀도 -- 면적 IoU는"
        f" 두께 1셀 표면에서 무의미해 2026-08-21에 뺐다)",
        f"range_mae {fmt(scores['range_mae'])} m   missed_obstacle {fmt(scores['missed_obstacle_rate'])}"
        f"   paired rays {scores['n_paired_rays']}",
    ]


def main(
    ckpt,
    sequences="raws1,rawos3",
    out_dir="runs/robot_bev/viz",
    sort_by="index",
    limit=8,
    batch_size=4,
    num_workers=4,
    encoder_type="res101",
    # 체크포인트의 정식화. `binary`는 출력이 2채널이고 `occupied`를 예측 free의 경계에서
    # 유도한다 -- 학습 루프와 **같은 함수**(`binary_metrics.predicted_parts`)를 쓴다.
    formulation="three_class",
    dataset_root=DEFAULT_DATASET_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    device="cuda",
):
    """`sort_by`: index | fatal_rate | free_miss_rate | iou_free | f1_occupied | range_mae.

    `index`를 빼면 전부 **"나쁜 순"**이다. 잴 수 없는 샘플(예: GT에 짝지어진 광선이 없어
    `range_mae`가 NaN)은 순위를 왜곡하지 않도록 맨 뒤로 보낸다.
    """
    if sort_by not in _SORT_KEYS:
        raise ValueError(f"sort_by는 {sorted(_SORT_KEYS)} 중 하나여야 한다: {sort_by!r}")

    dataset_root = Path(dataset_root)
    samples = []
    for name in parse_sequence_names(sequences):
        samples.extend(list_sequence_samples(dataset_root / name))
    if not samples:
        raise FileNotFoundError(f"샘플이 없다: {sequences}")

    dataset = RobotBEVDataset(samples, common_root=common_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 표본 규약은 **체크포인트 옆 config.json에서 되찾는다** -- 기본값을 쓰면 옛 런(legacy)을
    # 새 기하로 재채점해 조용히 다른 숫자가 나온다.
    convention, offset = convention_for_checkpoints([ckpt])
    _height = height_config_for_ckpt_dirs([Path(ckpt).parent])
    vox_util = build_double_sphere_vox_util(ROBOT_GRID_SPEC, dataset.cameras, device=device,
                                           height_bins=_height["height_bins"],
                                           height_min_m=_height["height_min_m"],
                                           height_max_m=_height["height_max_m"],
                                           pixel_convention=convention,
                                           pixel_offset=offset)
    model = ThreeClassSegnet(
        ROBOT_GRID_SPEC.n_rows, vox_util.Y, ROBOT_GRID_SPEC.n_cols, vox_util,
        use_radar=False, use_lidar=False, do_rgbcompress=True,
        encoder_type=encoder_type, rand_flip=False,
        num_classes=2 if formulation == "binary" else 3,
    ).to(device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    # `strict=True`로 얹는다 -- 체크포인트 경로나 `encoder_type`이 어긋나면 일부가 랜덤
    # 초기화된 채로 그럴듯한 그림이 나오고, 그것을 보고 모델을 판정하게 된다.
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    rays = build_ray_index(ROBOT_GRID_SPEC)

    records = []
    with torch.no_grad():
        for batch in loader:
            valid = batch["valid_bev_g"].to(device)
            _, _, logits, _, _ = model(
                # **`- 0.5`를 빼먹으면 안 된다.** 학습(`*_metrics.run_batch`)·재채점
                # (`rescore_checkpoints`)·이미지 의존성 측정이 모두 이 정규화를 적용한다.
                # 2026-08-19까지 이 줄에 그것이 빠져 있어서 시각화가 학습과 **다른 입력**을
                # 모델에 넣고 있었다 -- train 샘플이 `iou_free` 0.731로 보이던 원인이고
                # (같은 샘플의 실제 값은 0.94), §9의 정성 판정이 그 위에서 내려졌다.
                batch["rgb_camXs"].to(device) - 0.5, batch["pix_T_cams"].to(device),
                batch["cam0_T_camXs"].to(device), vox_util,
            )
            pred_parts = (
                binary_metrics.predicted_parts(logits, valid, rays)
                if formulation == "binary"
                else decompose_from_class_index(logits.argmax(dim=1, keepdim=True), valid)
            )
            gt_parts = decompose(batch["seg_bev_g"].to(device), batch["vis_bev_g"].to(device), valid)
            for i in range(valid.shape[0]):
                one = slice(i, i + 1)
                pred_one = {k: v[one] for k, v in pred_parts.items()}
                gt_one = {k: v[one] for k, v in gt_parts.items()}
                records.append({
                    "sample_id": batch["sample_id"][i],
                    "camera_images": (
                        batch["rgb_camXs"][i].numpy().transpose(0, 2, 3, 1) * 255
                    ).clip(0, 255).astype(np.uint8),
                    "pred_parts": {k: v[i, 0].cpu().numpy() for k, v in pred_parts.items()},
                    "gt_parts": {k: v[i, 0].cpu().numpy() for k, v in gt_parts.items()},
                    "valid": valid[i, 0].cpu().numpy().astype(bool),
                    **sample_scores(pred_one, gt_one, valid[one], rays, ROBOT_GRID_SPEC.cell_m),
                })

    if sort_by != "index":
        records.sort(key=lambda r: _nan_to_inf(_SORT_KEYS[sort_by](r)), reverse=True)
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

        # range 프로파일은 pred 패널 위에 얹는다 -- 확대 **전에** 찍어야 한 셀이
        # upscale x upscale 블록이 되어 실제로 보인다.
        pred_image = render_classes(record["pred_parts"], record["valid"])
        for parts_key, colour_key in (("gt_parts", "gt"), ("pred_parts", "pred")):
            r_m, status = first_free_range(record[parts_key]["free"], rays)
            draw_range_profile(pred_image, r_m, status, rays, ROBOT_GRID_SPEC,
                               RANGE_COLOURS[colour_key], RAY_OK)

        panel = build_panel(
            camera_images=record["camera_images"],
            camera_names=FINETUNE_CAMERA_NAMES,
            bev_grid=[
                [("IPM (실제 장면)", upscale(ipm, CELL_UPSCALE)),
                 ("GT 3-class", upscale(render_classes(record["gt_parts"], record["valid"]),
                                       CELL_UPSCALE))],
                [("pred 3-class  (흰 점 = GT 거리, 노란 점 = pred 거리)",
                  upscale(pred_image, CELL_UPSCALE)),
                 ("오차 지도", upscale(render_errors(record["pred_parts"], record["gt_parts"],
                                                  record["valid"]), CELL_UPSCALE))],
            ],
            headline=f"{record['sample_id']}"
                     + (f"    [{sort_by} 나쁜 순 {rank + 1}/{len(selected)}]"
                        if sort_by != "index" else ""),
            metric_lines=_metric_lines(record),
            legend=LEGEND_CLASSES + LEGEND_ERRORS,
        )
        name = (f"{rank:02d}_{sequence_name}_{sample_id}.png" if sort_by != "index"
                else f"{sequence_name}_{sample_id}.png")
        panel.save(out_path / name)

    print(f"{len(selected)} panels -> {out_path}")


if __name__ == "__main__":
    Fire(main)
