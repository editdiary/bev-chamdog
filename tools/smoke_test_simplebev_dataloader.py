"""SynWoodScape 데이터로더 + Simple-BEV `Segnet.forward()` end-to-end smoke test.

파이프라인이 shape 에러 없이 forward+backward를 끝까지 도는지만 증명한다(성능 검증 아님).
Run: python tools/smoke_test_simplebev_dataloader.py
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))


from projects.datasets.simplebev_vox import build_vox_util  # noqa: E402
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402
from projects.common.free_space import decompose, to_class_index  # noqa: E402
from projects.common.three_class_metrics import compute_three_class_loss  # noqa: E402
from projects.datasets.synwoodscape_simplebev import (  # noqa: E402
    GRID_SPEC,
    SynWoodScapeSimpleBEVDataset,
)


def main():
    device = "cuda"
    sample_ids = ["00000", "00001", "00002", "00003"]
    dataset = SynWoodScapeSimpleBEVDataset(sample_ids)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    vox_util = build_vox_util(GRID_SPEC, device=device)
    Z, Y, X = GRID_SPEC.n_rows, 1, GRID_SPEC.n_cols

    model = ThreeClassSegnet(
        Z, Y, X, vox_util,
        use_radar=False, use_lidar=False, do_rgbcompress=True,
        encoder_type="res101",
    ).to(device)
    model.train()

    batch = next(iter(loader))
    rgb_camXs = batch["rgb_camXs"].to(device) - 0.5  # nuScenes 관례: [0,1] -> [-0.5,0.5]
    pix_T_cams = batch["pix_T_cams"].to(device)
    cam0_T_camXs = batch["cam0_T_camXs"].to(device)
    seg_bev_g = batch["seg_bev_g"].to(device)
    vis_bev_g = batch["vis_bev_g"].to(device)
    valid_bev_g = batch["valid_bev_g"].to(device)

    print("input  rgb_camXs   :", tuple(rgb_camXs.shape))
    print("input  pix_T_cams  :", tuple(pix_T_cams.shape))
    print("input  cam0_T_camXs:", tuple(cam0_T_camXs.shape))
    print("target seg_bev_g   :", tuple(seg_bev_g.shape))

    raw_e, feat_e, logits, center_e, offset_e = model(rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util)
    print("output logits      :", tuple(logits.shape))
    expected = (seg_bev_g.shape[0], 3, *seg_bev_g.shape[2:])
    assert tuple(logits.shape) == expected, f"shape mismatch: {tuple(logits.shape)} vs {expected}"

    class_index = to_class_index(decompose(seg_bev_g, vis_bev_g, valid_bev_g))
    # 가중치 1로 둔다 -- 이 스모크는 배선(forward/backward)만 보고 클래스 균형은 보지 않는다.
    loss, parts = compute_three_class_loss(
        logits, class_index, valid_bev_g, torch.ones(3, device=device)
    )
    print("loss               :", loss.item())
    for name in ("loss_unknown", "loss_free", "loss_occupied"):
        print(f"{name:19s}:", parts[name].item())

    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print("total grad norm    :", grad_norm)
    assert grad_norm > 0, "backward()가 실제로 gradient를 만들지 못했다"

    print("\nSMOKE TEST PASSED - pipeline runs end-to-end (forward+backward) with no shape errors.")


if __name__ == "__main__":
    main()
