"""추론 지연·FPS·메모리를 잰다 -- **배포 대상(Jetson AGX Orin)에서 돌리는 것이 목적이다.**

## 왜 필요한가

배포는 Jetson AGX Orin이다. 그런데 이 저장소에는 **추론 비용을 한 번도 측정한 기록이 없다.**
그 상태로 "입력 해상도를 올려 BEV 표본 밀도를 높인다"를 실험하면 **배포할 수 없는 설정에
런을 쓰게 된다.** 그래서 해상도 작업의 첫 단계는 실험이 아니라 이 측정이다.

## 데이터셋이 필요 없다

지연은 **입력 값이 아니라 형상**만으로 정해지므로 난수 입력으로 잰다. 캘리브레이션도
없으면 합성값으로 만든다(연산 횟수가 같아서 지연이 같다). 즉 **저장소만 클론하면 Orin에서
바로 돌아간다** -- 데이터셋·체크포인트를 옮길 필요가 없다.

## 무엇을 재나

- **전체 forward 지연**과 그중 **encoder 몫**. 입력 해상도를 바꾸면 encoder만 커지고
  lifting·BEV decoder는 그대로이므로, 이 분해가 곧 "해상도를 올려도 되나"의 답이다.
- **peak GPU 메모리.**
- fp32와 fp16을 따로 낸다 -- Orin은 fp16에서 크게 유리하다.

## 읽을 때 주의

**이 숫자는 PyTorch eager 기준이고 실제 배포(TensorRT/ONNX)보다 느리다.** 보통 2~4배
차이가 난다. 그래서 이 측정의 용도는 **절대 FPS 확정이 아니라 설정 간 비교**다
(512x288 대 768x432, stride 8 대 4). 절대 FPS는 TensorRT로 변환한 뒤 다시 재야 한다.

전력·GPU 점유율은 이 스크립트가 못 본다. **같이 `tegrastats`를 띄워 두고 읽는다:**

    sudo tegrastats --interval 200 | tee tegrastats.log

## 실행 (Orin에서)

    # 현재 배포 후보 설정
    python tools/benchmark_inference.py

    # 해상도를 올리면 얼마나 느려지나
    python tools/benchmark_inference.py --resolutions=512x288,640x360,768x432,1024x576

    # 전력 모드를 바꿔 가며 (MAXN이 최대)
    sudo nvpmodel -m 0 && sudo jetson_clocks
    python tools/benchmark_inference.py
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from fire import Fire

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.models\._utils")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.bev_gt.grid import ROBOT_GRID_SPEC  # noqa: E402
from projects.geometry.double_sphere import FINETUNE_CAMERA_NAMES  # noqa: E402
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.simplebev_three_class import ThreeClassSegnet  # noqa: E402

_CALIB = _REPO_ROOT / "dataset/sj_datasets/common/calibration/calib.yaml"


class _SyntheticCamera:
    """캘리브레이션이 없는 환경(Orin)용. **지연은 값이 아니라 형상이 정한다.**"""

    def __init__(self, width=1280, height=720):
        self.xi, self.alpha = 0.0, 0.5
        self.fx = self.fy = width / 3.0
        self.cx, self.cy = width / 2.0, height / 2.0
        self.width, self.height = width, height
        self.domain_cos_limit = 0.0


def _cameras():
    """실제 캘리브레이션이 있으면 그것을, 없으면 합성값을 쓴다."""
    if _CALIB.exists():
        from projects.geometry.double_sphere import load_cameras
        loaded = load_cameras(_CALIB)
        return [loaded[n] for n in FINETUNE_CAMERA_NAMES], "실제 calib.yaml"
    return [_SyntheticCamera() for _ in FINETUNE_CAMERA_NAMES], "합성 캘리브레이션 (지연은 동일)"


def _parse_resolutions(value):
    if isinstance(value, (tuple, list)):
        items = [str(v) for v in value]
    else:
        items = [v for v in str(value).split(",") if v]
    out = []
    for item in items:
        w, _, h = item.strip().lower().partition("x")
        out.append((int(w), int(h)))
    return out


def _time_module(fn, iters, warmup, device):
    """`torch.cuda.synchronize()`로 감싸 정확히 잰다. warmup은 cuDNN autotune 때문에 필수다.

    **peak 메모리 통계는 warmup 뒤에 초기화한다** -- `cudnn.benchmark=True`가 autotune 중에
    여러 알고리즘의 workspace를 크게 잡아서, 그걸 같이 세면 정상 추론의 몇 배로 나온다.
    """
    for _ in range(warmup):
        fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return np.array(samples)


def main(
    resolutions="512x288",
    encoder_type="res101",
    batch_size=1,          # 배포는 1장이다 -- 학습 배치로 재면 FPS가 낙관적으로 나온다
    iters=30,
    warmup=10,
    precisions="fp32,fp16",
    device="cuda",
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("!! CUDA가 없다 -- CPU로 재면 배포 숫자로 쓸 수 없다.")
        device = "cpu"
    torch.backends.cudnn.benchmark = True

    cameras, calib_note = _cameras()
    n_cams = len(cameras)
    spec = ROBOT_GRID_SPEC
    Z, Y, X = spec.n_rows, 1, spec.n_cols

    print(f"장치: {torch.cuda.get_device_name(0) if device.startswith('cuda') else 'CPU'}")
    print(f"torch {torch.__version__} | 캘리브레이션: {calib_note}")
    print(f"BEV 격자 {Z}x{X} (셀 {spec.cell_m*100:.0f} cm) | 카메라 {n_cams}대 "
          f"| encoder {encoder_type} | batch {batch_size}")
    print("**PyTorch eager 기준이다 -- TensorRT 배포보다 2~4배 느리다. 설정 간 비교용으로만 읽는다.**\n")

    head = [("입력 해상도", 13), ("특징맵", 10), ("정밀도", 8), ("전체 ms", 10),
            ("FPS", 8), ("encoder ms", 12), ("encoder 몫", 11), ("peak MB", 10)]
    print("  " + " ".join(f"{h:>{w}}" for h, w in head))

    for (w, h) in _parse_resolutions(resolutions):
        for precision in [p.strip() for p in str(precisions).split(",") if p.strip()]:
            vox_util = build_double_sphere_vox_util(spec, cameras, device=device)
            model = ThreeClassSegnet(
                Z, Y, X, vox_util, use_radar=False, use_lidar=False, do_rgbcompress=True,
                encoder_type=encoder_type, rand_flip=False, num_classes=2,
            ).to(device).eval()

            rgb = torch.randn(batch_size, n_cams, 3, h, w, device=device)
            eye = torch.eye(4, device=device)
            mats = eye.repeat(batch_size, n_cams, 1, 1)

            # **`model.half()`를 쓰지 않는다.** `unproject_image_to_mem`이 유효 마스크를
            # `.float()`로 만들어 `values * valid_mem`에서 half가 float32로 승격되고,
            # 그 뒤 BEV decoder의 half 가중치와 dtype이 어긋나 그대로 실패한다.
            # autocast는 연산별로 dtype을 맞춰 주므로 모델 수정 없이 혼합정밀도를 잰다.
            # (실제 배포는 TensorRT fp16이고, autocast는 그 대리 측정이다.)
            amp = (torch.autocast("cuda", dtype=torch.float16) if precision == "fp16"
                   and device.startswith("cuda") else None)

            with torch.no_grad():
                def full():
                    if amp is None:
                        model(rgb, mats, mats, vox_util)
                    else:
                        with amp:
                            model(rgb, mats, mats, vox_util)

                # encoder만: `Segnet.forward`가 (B, S, ...)를 (B*S, ...)로 합친 뒤 넣는다.
                packed = rgb.reshape(batch_size * n_cams, 3, h, w)

                def enc():
                    if amp is None:
                        model.encoder(packed)
                    else:
                        with amp:
                            model.encoder(packed)

                try:
                    total = _time_module(full, iters, warmup, device)
                    encoder = _time_module(enc, iters, warmup, device)
                except RuntimeError as error:
                    print(f"  {w}x{h:<8} {'-':>10} {precision:>8}   실패: "
                          f"{str(error).splitlines()[0][:60]}")
                    del model, vox_util
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    continue

            peak = (torch.cuda.max_memory_allocated() / 1024 ** 2
                    if device.startswith("cuda") else float("nan"))
            feat = f"{w // 8}x{h // 8}"
            row = [f"{w}x{h}", feat, precision, f"{np.median(total):.1f}",
                   f"{1000.0 / np.median(total):.1f}", f"{np.median(encoder):.1f}",
                   f"{np.median(encoder) / np.median(total) * 100:.0f}%", f"{peak:.0f}"]
            print("  " + " ".join(f"{c:>{wd}}" for c, (_, wd) in zip(row, head)))

            del model, vox_util, rgb, mats, packed
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    print("\n  판독:")
    print("  - `encoder 몫`이 크면 **입력 해상도를 올리는 비용이 그만큼 그대로 붙는다.**")
    print("    작으면 해상도를 올릴 여지가 있고, lifting·BEV decoder가 병목이라는 뜻이다.")
    print("  - 해상도를 1.5배로 올리면 encoder 연산은 약 2.25배(면적 비)가 된다.")
    print("  - stride 8 -> 4는 **encoder backbone을 안 건드리고** head 한 단계만 4배 면적이 된다.")
    print("    그래서 해상도를 올리는 것보다 싸다 -- 그 가설을 이 표로 확인할 수 있다.")
    print("  - 전력·GPU 점유율은 `sudo tegrastats --interval 200`을 따로 띄워 읽는다.")


if __name__ == "__main__":
    Fire(main)
