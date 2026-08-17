"""SynWoodScape 데이터로더 + 실제 어안(radial_poly) 투영 + Simple-BEV `Segnet.forward()`
end-to-end smoke test (ROADMAP Phase 3.2).

Phase 3.1의 임시 핀홀 근사(`projects/datasets/simplebev_vox.Vox_util`)를
`projects/models/fisheye_vox.FisheyeVoxUtil`(실제 `radial_poly` 투영)로 교체해도 파이프라인이
forward+backward를 shape 에러 없이 도는지 확인한다. GPU 여유 메모리가 넉넉하지 않은 환경을
고려해 batch_size=1을 쓰고 최대 메모리 사용량을 출력한다.

Run: python tools/smoke_test_fisheye_segnet.py
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))


from projects.datasets.synwoodscape_simplebev import (  # noqa: E402
    CAMERA_NAMES,
    DEFAULT_DATASET_ROOT,
    GRID_SPEC,
    SynWoodScapeSimpleBEVDataset,
)
from projects.geometry.fisheye import load_camera  # noqa: E402
from projects.models.fisheye_vox import build_fisheye_vox_util  # noqa: E402
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402
from projects.common.free_space import decompose, to_class_index  # noqa: E402
from projects.common.three_class_metrics import compute_three_class_loss  # noqa: E402


def main():
    device = "cuda"
    torch.cuda.reset_peak_memory_stats(device)

    sample_ids = ["00000", "00001"]
    dataset = SynWoodScapeSimpleBEVDataset(sample_ids)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    cameras = [
        load_camera(DEFAULT_DATASET_ROOT / "calibration_data" / f"{name}.json")
        for name in CAMERA_NAMES
    ]
    vox_util = build_fisheye_vox_util(GRID_SPEC, cameras, device=device)
    Z, Y, X = GRID_SPEC.n_rows, 1, GRID_SPEC.n_cols

    model = ThreeClassSegnet(
        Z, Y, X, vox_util,
        use_radar=False, use_lidar=False, do_rgbcompress=True,
        encoder_type="res101",
    ).to(device)
    model.train()

    batch = next(iter(loader))
    rgb_camXs = batch["rgb_camXs"].to(device) - 0.5  # nuScenes 관례: [0,1] -> [-0.5,0.5]
    # Segnet.forward는 여전히 pix_T_cams를 인자로 받지만, FisheyeVoxUtil은 이를 무시하고
    # 자신이 들고 있는 실제 radial_poly 계수로 투영한다 (fisheye_vox.py 모듈 docstring 참고).
    pix_T_cams = batch["pix_T_cams"].to(device)
    cam0_T_camXs = batch["cam0_T_camXs"].to(device)
    seg_bev_g = batch["seg_bev_g"].to(device)
    vis_bev_g = batch["vis_bev_g"].to(device)
    valid_bev_g = batch["valid_bev_g"].to(device)

    print("input  rgb_camXs   :", tuple(rgb_camXs.shape))
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

    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    print(f"peak GPU memory    : {peak_mem_gb:.2f} GiB")

    print("\nSMOKE TEST PASSED - fisheye (radial_poly) lifting runs end-to-end (forward+backward).")


if __name__ == "__main__":
    main()
