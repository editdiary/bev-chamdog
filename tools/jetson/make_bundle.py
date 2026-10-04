"""Jetson 측정 꾸러미를 만든다 -- **이 폴더 하나만 보드로 옮기면 된다** (2026-10-04).

보드에는 저장소 코드·데이터셋·카메라 보정 파일이 필요 없다. 구조:

    jetson_bundle/
    ├── val_inputs.npz            검증 프레임 전부, uint8 (F, S, 3, H, W) + sample_ids     (모델 공통)
    ├── calib_inputs.npz          INT8 보정용 **학습** 프레임, uint8                        (모델 공통)
    ├── models/<run>/
    │   ├── encoder.onnx          인코더(정규화 포함). TensorRT 엔진의 재료
    │   ├── encoder_fp32.ts       같은 인코더의 TorchScript -- PyTorch fp32 기준선
    │   ├── bev_stage.ts          특징 집어 오기(투영은 fp32 상수) + 카메라 평균 + BEV 압축 + 디코더
    │   └── reference_prob_free.npz  서버 PyTorch fp32(배치 1)의 p(free) -- 보드 결과 대조용
    ├── build_engines.py, bench.py, README.md   보드에서 실행할 것
    └── manifest.json             출처·형상·대조 결과

보정은 학습 프레임으로 한다 -- 검증 프레임으로 보정하면 평가가 그 프레임에 맞춰진다.

**모델마다 검증 셋을 거친 뒤에만 쓴다.** (1) 분할 파이프라인 = 원래 `Segnet.forward`(같은 배치 1,
전 검증 프레임), (2) 캠페인이 저장한 확률맵과 같은 모델인가(배치 8이라 허용치가 넓다 -- 아래 주석),
(3) ONNX(onnxruntime)·TorchScript = PyTorch. 하나라도 어긋나면 종료 코드 1.

    python tools/jetson/make_bundle.py                     # 실험 1 고정 split 시드 0~4 -> runs/jetson_bundle/
    python tools/jetson/make_bundle.py --seeds=0           # 한 모델만
"""
import hashlib
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.models\._utils")
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    GRID_SPEC,
    RobotBEVDataset,
    parse_sequence_names,
    split_samples_by_sequence,
)
from projects.datasets.simplebev_vox import height_config_for_ckpt_dirs  # noqa: E402
from projects.deploy.split_model import split_model  # noqa: E402
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.pixel_grid import convention_for_run_dirs  # noqa: E402
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402
from projects.models.virtual_pinhole import projection_for_run_dirs  # noqa: E402

HERE = Path(__file__).resolve().parent
BOARD_FILES = ("build_engines.py", "bench.py", "README.md")

# 캠페인 확률맵은 **8프레임씩** 돌린 것이다. 같은 모델도 묶음 크기가 다르면 cuDNN이 다른 알고리즘을
# 골라 값이 조금 달라진다(실측 2026-10-04, 시드 0: 배치 8 재현은 fp16 저장값과 차이 0, 배치 1 대
# 배치 8은 최대 2.3e-3·평균 1.6e-5, τ=0.5 판정 108만 칸 중 26칸). 그래서 이 대조는 "같은 모델인가"만
# 거르고(5e-3), 정밀한 등가는 같은 배치끼리의 첫 대조가 맡는다.
LIMITS = {"split_vs_full_forward_max_abs": 1e-5, "split_vs_campaign_npz_max_abs": 5e-3,
          "torchscript_vs_split_max_abs": 1e-5}
ONNX_RELATIVE_LIMIT = 1e-3  # 특징 최대 크기에 곱한다


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _frames_uint8(dataset):
    """`[0, 1]` float 이미지 -> `uint8`. **무손실이어야 한다** -- 로더가 `uint8 / 255`로 만든다."""
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)
    images, ids = [], []
    for batch in loader:
        rgb = batch["rgb_camXs"]
        q = torch.round(rgb * 255.0).clamp(0, 255).to(torch.uint8)
        if not torch.equal(q.float() / 255.0, rgb):
            raise SystemExit("!! 이미지가 uint8/255로 정확히 표현되지 않는다 -- 무손실 저장 불가")
        images.append(q.numpy())
        ids.extend(batch["sample_id"])
    return np.concatenate(images), ids


def _export_run(run, root, checkpoint, out_models, val_set, val_u8, val_ids, device, opset):
    """한 모델을 내보내고 검증한다. 반환: 그 모델의 manifest 항목."""
    import onnx
    import onnxruntime as ort

    log_dir, ckpt_dir = root / "logs" / run, root / "ckpt" / run
    ckpt = ckpt_dir / checkpoint
    projection, _ = projection_for_run_dirs([log_dir], quiet=True)
    if projection != "ds_native":
        raise SystemExit(f"!! {run}: 어안 직접 투영 런만 지원한다 (런: {projection})")
    convention, offset = convention_for_run_dirs([log_dir], quiet=True)
    height = height_config_for_ckpt_dirs([ckpt_dir], quiet=True)
    W, H = val_set.resize_wh

    vox_util = build_double_sphere_vox_util(
        GRID_SPEC, val_set.cameras, device=device, pixel_convention=convention,
        pixel_offset=offset, height_bins=height["height_bins"],
        height_min_m=height["height_min_m"], height_max_m=height["height_max_m"])
    model = ThreeClassSegnet(GRID_SPEC.n_rows, vox_util.Y, GRID_SPEC.n_cols, vox_util,
                             use_radar=False, use_lidar=False, do_rgbcompress=True,
                             encoder_type="res101", rand_flip=False, num_classes=2).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=True)
    model.eval()
    pix_T_cams = val_set._pix_T_cams.to(device)
    cam0_T_camXs = val_set._cam0_T_camXs.to(device)
    enc, bev = split_model(model, vox_util, pix_T_cams, cam0_T_camXs, (H, W))

    out = out_models / run
    out.mkdir(parents=True, exist_ok=True)
    checks, split_probs, max_full = {}, [], 0.0
    with torch.no_grad():
        for i in range(len(val_u8)):
            rgb = torch.from_numpy(val_u8[i]).to(device).float() / 255.0      # (S, 3, H, W)
            got = bev(enc(rgb))
            _, _, logits, _, _ = model(rgb[None] - 0.5, pix_T_cams[None], cam0_T_camXs[None], vox_util)
            want = torch.softmax(logits, dim=1)[:, 1]
            max_full = max(max_full, (got - want).abs().max().item())
            split_probs.append(got[0].cpu().numpy())
    split_probs = np.stack(split_probs)
    checks["split_vs_full_forward_max_abs"] = max_full

    campaign_npz = root / "analysis" / "predictions" / f"{run}__last.npz"
    with np.load(campaign_npz, allow_pickle=False) as blob:
        ref = {k: np.array(blob[k]) for k in blob.files}
    if list(ref["sample_ids"]) != val_ids:
        raise SystemExit(f"!! {run}: 검증 프레임 순서가 캠페인 확률맵과 다르다")
    campaign = ref["prob_free"].astype(np.float32)
    checks["split_vs_campaign_npz_max_abs"] = float(np.abs(split_probs - campaign).max())
    checks["split_vs_campaign_npz_decision_flips"] = int(((split_probs > 0.5) != (campaign > 0.5)).sum())
    np.savez_compressed(out / "reference_prob_free.npz", prob_free=split_probs.astype(np.float32),
                        sample_ids=np.array(val_ids))

    enc_cpu, bev_cpu = enc.cpu(), bev.cpu()
    sample = torch.from_numpy(val_u8[0]).float() / 255.0
    with torch.no_grad():
        feat_cpu = enc_cpu(sample)
        torch.onnx.export(enc_cpu, sample, str(out / "encoder.onnx"), opset_version=opset,
                          input_names=["rgb"], output_names=["feat"], dynamo=False)
        torch.jit.trace(enc_cpu, sample).save(str(out / "encoder_fp32.ts"))
        torch.jit.trace(bev_cpu, feat_cpu).save(str(out / "bev_stage.ts"))

    onnx.checker.check_model(str(out / "encoder.onnx"))
    sess = ort.InferenceSession(str(out / "encoder.onnx"), providers=["CPUExecutionProvider"])
    onnx_feat = sess.run(None, {"rgb": sample.numpy()})[0]
    checks["onnx_vs_torch_feat_max_abs"] = float(np.abs(onnx_feat - feat_cpu.numpy()).max())
    checks["feat_abs_max"] = float(np.abs(feat_cpu.numpy()).max())
    enc_ts = torch.jit.load(str(out / "encoder_fp32.ts")).to(device).eval()
    bev_ts = torch.jit.load(str(out / "bev_stage.ts")).to(device).eval()
    ts_max = 0.0
    with torch.no_grad():
        for i in range(len(val_u8)):
            rgb = torch.from_numpy(val_u8[i]).to(device).float() / 255.0
            ts_max = max(ts_max, float(np.abs(bev_ts(enc_ts(rgb))[0].cpu().numpy() - split_probs[i]).max()))
    checks["torchscript_vs_split_max_abs"] = ts_max

    limits = dict(LIMITS, onnx_vs_torch_feat_max_abs=ONNX_RELATIVE_LIMIT * max(1.0, checks["feat_abs_max"]))
    failed = [k for k, lim in limits.items() if not checks[k] <= lim]
    print(f"  {run}: 분할=원본 {max_full:.1e} | 캠페인 npz {checks['split_vs_campaign_npz_max_abs']:.1e}"
          f" (뒤집힘 {checks['split_vs_campaign_npz_decision_flips']}칸) | ONNX {checks['onnx_vs_torch_feat_max_abs']:.1e}"
          f" | TorchScript {ts_max:.1e}" + (f"   !! 실패 {failed}" if failed else ""), flush=True)
    del model, enc, bev, enc_ts, bev_ts
    torch.cuda.empty_cache()
    return {"checkpoint": str(ckpt), "height_bins": height["height_bins"],
            "pixel_convention": convention, "pixel_offset": offset,
            "feat_shape": list(feat_cpu.shape), "checks": checks, "limits": limits, "failed": failed}


def main(
    root="runs/99_full_campaign/01_overall/fixed_split",
    seeds="0,1,2,3,4",
    run_prefix="final_s",
    checkpoint="model-000000100.pth",
    out_dir="runs/jetson_bundle",
    train_sequences="raws2,raws3,rawos1,rawos2,rawos4",
    val_sequences="raws1,rawos3",
    dataset_root=DEFAULT_DATASET_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    opset=17,
):
    import onnx
    import onnxruntime as ort

    device = "cuda"
    root, out = Path(root), Path(out_dir)
    if out.exists():
        raise SystemExit(f"!! {out}가 이미 있다 -- 옛 꾸러미와 섞이지 않게 지우거나 다른 경로를 준다")
    out.mkdir(parents=True)
    runs = [f"{run_prefix}{int(s)}" for s in str(seeds).split(",")]

    data_root = Path(dataset_root)
    train_names, val_names = parse_sequence_names(train_sequences), parse_sequence_names(val_sequences)
    train_samples, val_samples = split_samples_by_sequence(
        [data_root / n for n in train_names + val_names], val_names)
    val_set = RobotBEVDataset(val_samples, common_root=common_root, augment=False)
    train_set = RobotBEVDataset(train_samples, common_root=common_root, augment=False)
    if not (torch.equal(train_set._cam0_T_camXs, val_set._cam0_T_camXs)
            and torch.equal(train_set._pix_T_cams, val_set._pix_T_cams)):
        raise SystemExit("!! 학습·검증의 카메라 보정이 다르다 -- 투영을 상수로 박을 수 없다")
    W, H = val_set.resize_wh
    S = len(val_set.camera_names)

    val_u8, val_ids = _frames_uint8(val_set)
    train_u8, train_ids = _frames_uint8(train_set)
    np.savez_compressed(out / "val_inputs.npz", images=val_u8, sample_ids=np.array(val_ids))
    np.savez_compressed(out / "calib_inputs.npz", images=train_u8, sample_ids=np.array(train_ids))
    print(f"입력 {S}x3x{H}x{W} | 검증 {len(val_u8)}프레임 | 보정(학습) {len(train_u8)}프레임 | 모델 {runs}")

    models = {run: _export_run(run, root, checkpoint, out / "models", val_set, val_u8, val_ids,
                               device, opset) for run in runs}

    for name in BOARD_FILES:
        shutil.copy2(HERE / name, out / name)
    failed = {run: m["failed"] for run, m in models.items() if m["failed"]}
    manifest = {
        "created_by": "tools/jetson/make_bundle.py",
        "runs": runs, "root": str(root),
        "input_shape": [S, 3, H, W],
        "output_shape": [1, GRID_SPEC.n_rows, GRID_SPEC.n_cols],
        "camera_names": list(val_set.camera_names),
        "n_val_frames": int(len(val_u8)), "n_calib_frames": int(len(train_u8)),
        "val_sequences": val_sequences, "calib_sequences": train_sequences,
        "onnx_opset": opset,
        "server": {"torch": torch.__version__, "onnx": onnx.__version__,
                   "onnxruntime": ort.__version__},
        "models": models, "failed": failed,
        "sha256": {p.relative_to(out).as_posix(): _sha256(p) for p in sorted(out.rglob("*"))
                   if p.is_file() and p.name != "manifest.json"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\n꾸러미 {out} ({total / 1e6:.0f} MB)")
    if failed:
        print(f"!! 검증 실패 -- 이 꾸러미를 쓰지 않는다: {failed}")
        return 1
    print("검증 통과")
    return 0


if __name__ == "__main__":
    sys.exit(Fire(main))
