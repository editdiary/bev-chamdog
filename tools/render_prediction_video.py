"""라벨 없는 시퀀스 전체를 추론해 동영상으로 만든다.

`visualize_robot_predictions.py`와 목적이 다르다. 그쪽은 **라벨이 있는** 샘플을 GT·오차 지도와
함께 정량 판정하는 도구이고, 이쪽은 **라벨이 없어도 되는** 정성 확인용이다. 그래서 GT 패널이
없고 프레임을 이어 붙여 시간축 일관성(통로가 흔들리는지, 유령 영역이 명멸하는지)을 본다.

레이아웃 (사용자 지정):

    상단        front | left | right 원본 RGB
    하단 좌측    IPM (지면 투영 -- 실제 장면)
    하단 우측    모델 예측

`valid`(수집 아티팩트 마스크)는 리그 고정이므로 라벨 없이도 계산된다. 예측 `occupied`는
binary 정식화에서 예측 free의 경계에서 유도한다 -- 학습·시각화와 **같은 함수**를 쓴다
(`binary_metrics.predicted_parts`).

실행 예:
    CUDA_VISIBLE_DEVICES=0 python tools/render_prediction_video.py \\
        --ckpt=runs/robot_bev/ckpt/<run>/model_best-000000019.pth --formulation=binary \\
        --sequence_root=dataset/sj_datasets/raws1 --out=runs/robot_bev/viz/raws1.mp4
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.models\._utils")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.bev_gt.grid import ROBOT_GRID_SPEC  # noqa: E402
from projects.bev_gt.ipm import render_ipm  # noqa: E402
from projects.common import binary_metrics  # noqa: E402
from projects.common.free_space import decompose_from_class_index  # noqa: E402
from projects.common.polar import build_ray_index  # noqa: E402
from projects.common.three_class_panels import (  # noqa: E402
    CLASS_COLOURS,
    load_font,
    render_classes,
)
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    RESIZE_HEIGHT,
    RESIZE_WIDTH,
    build_bev_masks,
)
from projects.datasets.simplebev_vox import ref_T_cam_from_ego_T_cam  # noqa: E402
from projects.geometry.double_sphere import (  # noqa: E402
    FINETUNE_CAMERA_NAMES,
    load_camera_index_names,
    load_cameras,
    load_ego_T_cams,
)
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.pixel_grid import convention_for_checkpoints  # noqa: E402
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402

_PAD = 12
_LABEL_H = 22
_BG = (250, 250, 250)
_FG = (20, 20, 20)


def list_frames(sequence_root: Path) -> list:
    """프레임 디렉터리 목록. `rgb_images/`가 있으면 그 아래, 없으면 인자 자체를 본다.

    두 배치를 다 받는 이유: 라벨된 시퀀스는 `<seq>/rgb_images/<sample>/cam_*.jpg` 구조인데,
    새로 받은 원본 영상 프레임은 그 상위 폴더가 없을 수 있다.
    """
    root = Path(sequence_root)
    base = root / "rgb_images" if (root / "rgb_images").is_dir() else root
    frames = sorted(p for p in base.iterdir() if p.is_dir())
    if not frames:
        raise FileNotFoundError(
            f"프레임 디렉터리를 찾지 못했다: {base}\n"
            f"각 프레임이 `cam_{{{','.join(FINETUNE_CAMERA_NAMES)}}}.jpg`를 담은 폴더여야 한다."
        )
    return frames


def resolve_camera_files(frame_dir: Path, camera_names, orientation_path) -> dict:
    """{카메라 이름: 파일 경로}. 두 가지 파일명 규약을 자동으로 가린다.

    - 라벨 시퀀스(`dataset/sj_datasets/<seq>/rgb_images/<sample>/`): `cam_front.jpg` 처럼
      **이름**이 파일명에 들어 있다.
    - 원본 전체 시퀀스(`dataset/sj_datasets_full-seq/.../frame_000000/`): `cam0.jpg` 처럼
      **인덱스**이고 매핑은 `orientation.json`이 정한다(cam0=front, cam1=right, cam2=rear,
      cam3=left). rear는 라벨 생성에 쓰이지 않았으므로 여기서도 읽지 않는다.

    추측하지 않고 실제 파일 존재로 판정한다 -- 규약을 잘못 고르면 좌우가 바뀐 채로 그럴듯한
    그림이 나오고, 그것은 눈으로 알아채기 어렵다.
    """
    by_name = {name: frame_dir / f"cam_{name}.jpg" for name in camera_names}
    if all(path.exists() for path in by_name.values()):
        return by_name

    index_by_name = {name: index for index, name
                     in load_camera_index_names(orientation_path).items()}
    by_index = {}
    for name in camera_names:
        if name not in index_by_name:
            raise FileNotFoundError(f"orientation.json에 '{name}' 카메라가 없다")
        by_index[name] = frame_dir / f"cam{index_by_name[name]}.jpg"
    missing = [str(path) for path in by_index.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"카메라 이미지를 찾지 못했다. `cam_<name>.jpg`도 `cam<idx>.jpg`도 없다.\n"
            f"  없는 파일: {missing[:3]}\n  프레임 디렉터리: {frame_dir}"
        )
    return by_index


def load_frame_images(frame_dir: Path, camera_names, orientation_path) -> dict:
    """{카메라 이름: (H, W, 3) uint8} 원본 해상도. IPM이 이 해상도를 그대로 쓴다."""
    return {
        name: np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        for name, path in resolve_camera_files(frame_dir, camera_names, orientation_path).items()
    }


def _model_input(images: dict, camera_names) -> torch.Tensor:
    """(S, 3, H, W) float. 학습과 같은 해상도로 줄이고 `[0,1]`로 만든다.

    `- 0.5`는 여기서 하지 않는다 -- 호출부가 forward 직전에 한 번만 적용하도록 두어,
    시각화가 그 정규화를 빠뜨렸던 사고(§18.1)를 한 곳에서만 관리한다.
    """
    stacked = np.stack([
        np.asarray(Image.fromarray(images[name]).resize(
            (RESIZE_WIDTH, RESIZE_HEIGHT), Image.BILINEAR), dtype=np.float32) / 255.0
        for name in camera_names
    ])
    return torch.from_numpy(stacked.transpose(0, 3, 1, 2))


def compose_frame(camera_images, camera_names, ipm_rgb, pred_rgb, title,
                  bev_upscale=4, camera_width=320) -> Image.Image:
    """상단 RGB 3장 / 하단 좌 IPM / 하단 우 예측. 순수 이미지 조립이라 모델과 무관하다."""
    bev_side = ipm_rgb.shape[0] * bev_upscale
    total_w = max(len(camera_names) * camera_width + (len(camera_names) - 1) * _PAD,
                  2 * bev_side + _PAD)
    cam_h = round(camera_width * camera_images[0].shape[0] / camera_images[0].shape[1])
    total_h = _LABEL_H + cam_h + _PAD + _LABEL_H + bev_side + _PAD

    canvas = Image.new("RGB", (total_w + 2 * _PAD, total_h + 2 * _PAD), _BG)
    draw = ImageDraw.Draw(canvas)
    title_font, label_font = load_font(16, bold=True), load_font(13)
    draw.text((_PAD, _PAD // 2), title, fill=_FG, font=title_font)

    y = _PAD + _LABEL_H
    for index, (image, name) in enumerate(zip(camera_images, camera_names)):
        x = _PAD + index * (camera_width + _PAD)
        canvas.paste(Image.fromarray(image).resize((camera_width, cam_h), Image.BILINEAR), (x, y))
        draw.text((x + 4, y + 2), name, fill=(255, 240, 60), font=label_font)

    y += cam_h + _PAD
    for index, (bev, label) in enumerate(((ipm_rgb, "IPM (실제 장면)"), (pred_rgb, "예측"))):
        x = _PAD + index * (bev_side + _PAD)
        draw.text((x, y), label, fill=_FG, font=label_font)
        canvas.paste(
            Image.fromarray(bev).resize((bev_side, bev_side), Image.NEAREST),
            (x, y + _LABEL_H),
        )
    return canvas


def main(
    ckpt,
    sequence_root,
    out="runs/robot_bev/viz/prediction.mp4",
    formulation="binary",
    encoder_type="res101",
    fps=5,
    limit=0,
    batch_size=4,
    bev_upscale=4,
    common_root=DEFAULT_COMMON_ROOT,
    device="cuda",
    save_frames=False,
):
    """`limit=0`이면 전체 프레임. `save_frames=True`면 PNG도 함께 남긴다."""
    import cv2  # 여기서만 쓴다 -- 임포트 실패가 다른 도구를 막지 않도록 지연 임포트한다.

    grid = ROBOT_GRID_SPEC
    frames = list_frames(sequence_root)
    if limit:
        frames = frames[:limit]

    calib = Path(common_root) / "calibration/calib.yaml"
    orientation = Path(common_root) / "calibration/orientation.json"
    cameras_by_name = load_cameras(calib)
    cameras = [cameras_by_name[name] for name in FINETUNE_CAMERA_NAMES]
    ego_T_cams = load_ego_T_cams(calib)
    cam0_T_camXs = torch.from_numpy(np.stack(
        [ref_T_cam_from_ego_T_cam(ego_T_cams[name]) for name in FINETUNE_CAMERA_NAMES]
    )).float()
    pix_T_cams = torch.eye(4).repeat(len(cameras), 1, 1)      # DS 경로에서 쓰이지 않는다

    permanent_blind, invalid = build_bev_masks(common_root, grid, FINETUNE_CAMERA_NAMES)
    # 라벨이 없으므로 `valid`는 리그 고정 마스크에서만 온다. `permanent_blind`는 `vis=0`이지
    # `valid=0`이 아니므로(§3) 여기서 제외하지 않는다 -- 학습과 같은 계약이다.
    valid_np = ~invalid

    # 표본 규약은 **체크포인트 옆 config.json에서 되찾는다** -- 기본값을 쓰면 옛 런(legacy)을
    # 새 기하로 재채점해 조용히 다른 숫자가 나온다.
    convention, offset = convention_for_checkpoints([ckpt])
    vox_util = build_double_sphere_vox_util(grid, cameras, device=device,
                                           pixel_convention=convention,
                                           pixel_offset=offset)
    model = ThreeClassSegnet(
        grid.n_rows, 1, grid.n_cols, vox_util, use_radar=False, use_lidar=False,
        do_rgbcompress=True, encoder_type=encoder_type, rand_flip=False,
        num_classes=2 if formulation == "binary" else 3,
    ).to(device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    rays = build_ray_index(grid)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = out_path.with_suffix("")
    if save_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)

    writer, written = None, 0
    valid_t = torch.from_numpy(valid_np).to(device)[None, None]
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            chunk = frames[start:start + batch_size]
            loaded = [load_frame_images(path, FINETUNE_CAMERA_NAMES, orientation)
                      for path in chunk]
            rgb = torch.stack([_model_input(one, FINETUNE_CAMERA_NAMES) for one in loaded])
            logits, _ = _forward(model, rgb, pix_T_cams, cam0_T_camXs, vox_util, device)

            for index, (path, images) in enumerate(zip(chunk, loaded)):
                one = logits[index:index + 1]
                parts = (
                    binary_metrics.predicted_parts(one, valid_t, rays)
                    if formulation == "binary"
                    else decompose_from_class_index(one.argmax(dim=1, keepdim=True), valid_t)
                )
                canvas = compose_frame(
                    [images[name] for name in FINETUNE_CAMERA_NAMES],
                    FINETUNE_CAMERA_NAMES,
                    # `render_ipm`은 이름으로 색인하는 dict를 받는다(모델 쪽은 순서 리스트다).
                    render_ipm(images, cameras_by_name, ego_T_cams, grid),
                    render_classes(parts, valid_t),
                    f"{Path(sequence_root).name}/{path.name}   frame {start + index + 1}"
                    f"/{len(frames)}",
                    bev_upscale=bev_upscale,
                )
                if save_frames:
                    canvas.save(frame_dir / f"{path.name}.png")
                bgr = np.asarray(canvas)[:, :, ::-1]
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps),
                        (bgr.shape[1], bgr.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"VideoWriter를 열 수 없다: {out_path}")
                writer.write(np.ascontiguousarray(bgr))
                written += 1
            print(f"  {written}/{len(frames)} 프레임", end="\r", flush=True)

    if writer is not None:
        writer.release()
    print(f"\n{written} 프레임 -> {out_path}"
          + (f" (+ PNG {frame_dir})" if save_frames else ""))


def _forward(model, rgb, pix_T_cams, cam0_T_camXs, vox_util, device):
    """`- 0.5` 정규화를 여기 한 곳에서만 한다 (§18.1의 사고 방지)."""
    batch = rgb.shape[0]
    _, _, logits, _, _ = model(
        rgb.to(device) - 0.5,
        pix_T_cams[None].repeat(batch, 1, 1, 1).to(device),
        cam0_T_camXs[None].repeat(batch, 1, 1, 1).to(device),
        vox_util,
    )
    return logits, batch


if __name__ == "__main__":
    Fire(main)
