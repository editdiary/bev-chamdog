"""모델이 **이미지를 실제로 쓰고 있는지**를 직접 재는 진단.

왜 필요한가: 과적합이 심할 때 두 가지 가능성이 구별되지 않는다.

1. 모델이 이미지를 보고 배우는데 학습 데이터가 적어 일반화가 막힌 것
2. 이미지 경로(어안 unprojection, 캘리브레이션, 리사이즈 스케일)가 어긋나 모델이 **이미지를
   무시하고 격자 레이아웃만 외운** 것 -- 이건 하이퍼파라미터로 고칠 수 없는 구조적 결함이다

방법: val 샘플의 GT·캘리브레이션은 그대로 두고 **입력 이미지만 다른 샘플의 것으로 바꿔치기**해
같은 지표를 다시 잰다. 이미지를 무시하는 모델은 예측이 거의 변하지 않는다.

읽는 법:

    shuffled iou_free ~= normal            -> 이미지를 안 쓴다 (구조적 결함 의심)
    shuffled iou_free ~= constant-map      -> 이미지에 의존하지 않는 성분이 곧 레이아웃 prior다
    agreement(normal, shuffled) 가 낮다     -> 예측이 이미지에 따라 실제로 바뀐다

`agreement`는 두 예측 free 마스크의 IoU다. GT와 무관하게 "예측이 입력에 반응하는가"만 본다.

실행:
    CUDA_VISIBLE_DEVICES=0 python tools/measure_image_dependence.py \\
        --ckpt=runs/robot_bev/ckpt/<run>/model_best-<step>.pth
"""
import sys
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.common.baselines import as_batch, constant_free_map  # noqa: E402
from projects.common.free_space import decompose, decompose_from_class_index  # noqa: E402
from projects.common.free_space_metrics import (  # noqa: E402
    fatal_rate,
    iou_free,
    iou_masked,
    weighted_mean,
)
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    GRID_SPEC,
    RobotBEVDataset,
    build_bev_masks,
    list_sequence_samples,
    load_masked_labels,
    parse_sequence_names,
)
from projects.geometry.double_sphere import FINETUNE_CAMERA_NAMES  # noqa: E402
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402


def shuffled_order(count: int) -> np.ndarray:
    """샘플 i가 샘플 `i + count//2`의 이미지를 받도록 하는 순서.

    **인접 프레임끼리 섞으면 안 된다.** 한 시퀀스는 연속 주행을 거리 기반으로 샘플링한 것이라
    이웃 프레임의 장면이 거의 같고, 그러면 "이미지를 바꿨는데 예측이 안 변했다"가 이미지를
    무시한다는 증거가 되지 못한다. 절반만큼 굴리면 대부분의 샘플이 다른 시퀀스의 이미지를 받는다.

    고정된 순열이라 재현된다 -- 난수를 쓰면 실행마다 숫자가 흔들려 비교가 안 된다.
    """
    if count < 2:
        raise ValueError(f"샘플이 2개 이상이어야 섞을 수 있다: {count}")
    return (np.arange(count) + max(1, count // 2)) % count


def _predict(model, vox_util, rgb, pix_T_cams, cam0_T_camXs, valid):
    _, _, logits, _, _ = model(rgb, pix_T_cams, cam0_T_camXs, vox_util)
    return decompose_from_class_index(logits.argmax(dim=1, keepdim=True), valid)


def main(
    ckpt,
    train_sequences="raws2,raws3,rawos1,rawos2,rawos4",
    val_sequences="raws1,rawos3",
    dataset_root=DEFAULT_DATASET_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    encoder_type="res101",
    num_workers=4,
    device="cuda",
):
    dataset_root = Path(dataset_root)
    val_samples = [s for name in parse_sequence_names(val_sequences)
                   for s in list_sequence_samples(dataset_root / name)]
    train_samples = [s for name in parse_sequence_names(train_sequences)
                     for s in list_sequence_samples(dataset_root / name)]

    dataset = RobotBEVDataset(val_samples, common_root=common_root)
    # 이미지를 샘플 간에 섞어야 하므로 val 전체를 한 번에 메모리에 올린다
    # (75장 x 3캠 x 288x512 float ~ 400 MB).
    batches = list(DataLoader(dataset, batch_size=len(dataset), num_workers=num_workers))
    batch = batches[0]

    vox_util = build_double_sphere_vox_util(GRID_SPEC, dataset.cameras, device=device)
    model = ThreeClassSegnet(
        GRID_SPEC.n_rows, 1, GRID_SPEC.n_cols, vox_util,
        use_radar=False, use_lidar=False, do_rgbcompress=True,
        encoder_type=encoder_type, rand_flip=False,
    ).to(device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()

    permanent_blind, invalid = build_bev_masks(common_root, GRID_SPEC, FINETUNE_CAMERA_NAMES)
    constant_map = constant_free_map([
        decompose(*load_masked_labels(root, sample_id, permanent_blind, invalid))["free"]
        for root, sample_id in train_samples
    ])

    order = shuffled_order(len(dataset))
    valid = batch["valid_bev_g"].to(device)
    gt = decompose(batch["seg_bev_g"].to(device), batch["vis_bev_g"].to(device), valid)
    rgb = batch["rgb_camXs"].to(device) - 0.5
    pix_T_cams, cam0_T_camXs = batch["pix_T_cams"].to(device), batch["cam0_T_camXs"].to(device)

    results = {}
    with torch.no_grad():
        # 배치를 쪼개 도는 이유: 75장을 한 번에 forward하면 메모리를 넘긴다. 캘리브레이션과
        # GT는 샘플에 붙어 있으므로 이미지만 `order`로 골라 넣는다.
        for name, index in (("normal", np.arange(len(dataset))), ("shuffled", order)):
            preds = []
            for start in range(0, len(dataset), 4):
                stop = min(start + 4, len(dataset))
                rows = slice(start, stop)
                preds.append(_predict(
                    model, vox_util, rgb[index[start:stop]],
                    pix_T_cams[rows], cam0_T_camXs[rows], valid[rows],
                )["free"])
            results[name] = torch.cat(preds)

    def score(pred):
        value, count = iou_free(pred, gt["free"], valid)
        fatal, denom = fatal_rate(pred, gt["free"], valid)
        return value, count, fatal, denom

    normal_iou, _, normal_fatal, _ = score(results["normal"])
    shuffled_iou, _, shuffled_fatal, _ = score(results["shuffled"])
    baseline_iou, _ = iou_free(
        as_batch(constant_map, len(dataset), device), gt["free"], valid
    )
    agreement, _ = iou_masked(results["shuffled"], results["normal"], valid)

    print("=" * 66)
    print(f" 이미지 의존도 진단  ({len(dataset)} val 샘플, 이미지를 {order[0] - 0}칸 굴려 교체)")
    print("=" * 66)
    print(f" normal    iou_free {normal_iou:.4f}   fatal {normal_fatal:.4f}")
    print(f" shuffled  iou_free {shuffled_iou:.4f}   fatal {shuffled_fatal:.4f}")
    print(f" constant-map baseline iou_free {baseline_iou:.4f}  (이미지를 한 픽셀도 안 본다)")
    print()
    print(f" 이미지 교체로 잃은 성능 = {normal_iou - shuffled_iou:+.4f}")
    print(f" 예측 일치도 agreement(normal, shuffled) = {agreement:.4f}")
    print()
    print(" 읽는 법: agreement가 1에 가까우면 예측이 입력 이미지에 반응하지 않는다는 뜻이고,")
    print("          그때 shuffled iou_free는 normal과 같아진다 -- 구조적 결함을 의심한다.")
    print("=" * 66)
    return {
        "normal_iou_free": normal_iou, "shuffled_iou_free": shuffled_iou,
        "baseline_iou_free": baseline_iou, "agreement": agreement,
        "normal_fatal_rate": normal_fatal, "shuffled_fatal_rate": shuffled_fatal,
    }


if __name__ == "__main__":
    Fire(main)
