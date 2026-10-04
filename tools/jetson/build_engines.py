"""[Jetson 보드에서 실행] 모델마다 `encoder.onnx` -> TensorRT 엔진 둘: fp16, INT8.

    python3 build_engines.py                 # 꾸러미 폴더 안에서. 모델 전부
    python3 build_engines.py --runs=final_s0 # 일부만

필요: TensorRT 파이썬 바인딩(JetPack 기본), torch(CUDA), numpy. 저장소 코드는 필요 없다.

- **fp16**: `BuilderFlag.FP16`.
- **INT8**: `BuilderFlag.INT8` + `FP16`(INT8로 못 가는 층은 fp16으로 남는다 -- TensorRT의 통상 INT8 빌드).
  보정은 `calib_inputs.npz`의 **학습 프레임**으로 한다(엔트로피 보정, 프레임 하나 = 카메라 S장이 한 배치).
  TensorRT 10.1부터 이 보정 API에 DeprecationWarning이 뜨지만 10.3에서 그대로 동작한다.

**이미 있는 엔진은 건너뛴다** -- 중간에 끊겨도 같은 명령을 다시 실행하면 이어서 만든다.
엔진은 **그 보드·그 TensorRT 버전 전용**이다. 층별 정밀도는 `models/<run>/engines/*_layers.json`에 남는다.
"""
import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

LOGGER = trt.Logger(trt.Logger.WARNING)


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """학습 프레임을 한 프레임(카메라 S장)씩 넘긴다. 입력은 `[0, 1]` float32 -- 엔진 입력과 같다."""

    def __init__(self, images_u8, cache_path: Path):
        super().__init__()
        self.images = images_u8
        self.cache_path = cache_path
        self.index = 0
        self.buffer = None

    def get_batch_size(self):
        return 1  # explicit batch 네트워크 -- 배치 차원은 입력 형상에 이미 들어 있다

    def get_batch(self, names):
        if self.index >= len(self.images):
            return None
        x = torch.from_numpy(self.images[self.index]).cuda().float().div_(255.0).contiguous()
        self.buffer = x  # 참조를 붙들어 둔다 -- TensorRT가 읽는 동안 메모리가 살아 있어야 한다
        self.index += 1
        if self.index % 64 == 0:
            print(f"    보정 {self.index}/{len(self.images)}", flush=True)
        return [int(x.data_ptr())]

    def read_calibration_cache(self):
        return self.cache_path.read_bytes() if self.cache_path.exists() else None

    def write_calibration_cache(self, cache):
        self.cache_path.write_bytes(bytes(cache))


def layer_precision_summary(layers):
    """합성곱 층이 어느 정밀도로 실행되는지 센다(출력 텐서 형식 기준)."""
    def kind(fmt):
        fmt = fmt.lower()
        return "int8" if "int8" in fmt else "fp16" if ("fp16" in fmt or fmt == "half") else \
            "fp32" if ("fp32" in fmt or fmt == "float") else "other"
    conv = collections.Counter()
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        # 층 종류 이름이 TensorRT 버전·GPU마다 다르다(10.3/Orin `CaskConvolution`, 10.16/Blackwell
        # `correlation`·`kgen`). 그래서 ONNX 노드 이름(`.../Conv`)으로 고르고 형식 변환 층은 뺀다.
        layer_type = str(layer.get("LayerType", ""))
        if layer_type != "Reformat" and "/Conv" in str(layer.get("Name", "")):
            conv[kind(layer["Outputs"][0].get("Format/Datatype", ""))] += 1
    return {"n_layers": sum(isinstance(x, dict) for x in layers), "conv_by_output_precision": dict(conv)}


def build(onnx_path: Path, precision: str, out_dir: Path, calib_images=None, workspace_gb=4):
    engine_path = out_dir / f"encoder_{precision}.engine"
    layers_path = out_dir / f"encoder_{precision}_layers.json"
    if engine_path.exists() and layers_path.exists():
        print(f"  {engine_path.parent.parent.name}/{engine_path.name}: 이미 있다 -- 건너뛴다")
        layers = json.loads(layers_path.read_text()).get("Layers", [])
        return {"engine": engine_path.name, "build_seconds": None, "skipped": True,
                "size_mb": round(engine_path.stat().st_size / 1e6, 2), **layer_precision_summary(layers)}

    builder = trt.Builder(LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise SystemExit("!! ONNX 파싱 실패:\n" + "\n".join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = EntropyCalibrator(calib_images, out_dir / "int8_calib.cache")

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit(f"!! {onnx_path.parent.name} {precision} 엔진 빌드 실패")
    seconds = time.time() - t0
    engine_path.write_bytes(bytes(serialized))

    engine = trt.Runtime(LOGGER).deserialize_cuda_engine(bytes(serialized))
    info = json.loads(engine.create_engine_inspector().get_engine_information(
        trt.LayerInformationFormat.JSON))
    layers_path.write_text(json.dumps(info, indent=1))
    summary = layer_precision_summary(info.get("Layers", []))
    print(f"  {onnx_path.parent.name}/{engine_path.name}: {engine_path.stat().st_size / 1e6:.1f} MB,"
          f" 빌드 {seconds:.0f} s, 합성곱 정밀도 {summary['conv_by_output_precision']}", flush=True)
    return {"engine": engine_path.name, "build_seconds": round(seconds, 1), "skipped": False,
            "size_mb": round(engine_path.stat().st_size / 1e6, 2), **summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--runs", default="", help="쉼표 구분. 비우면 models/ 아래 전부")
    ap.add_argument("--precisions", default="fp16,int8")
    ap.add_argument("--workspace_gb", type=int, default=4)
    args = ap.parse_args()

    bundle = Path(args.bundle)
    runs = ([r.strip() for r in args.runs.split(",") if r.strip()]
            or sorted(p.name for p in (bundle / "models").iterdir() if p.is_dir()))
    print(f"TensorRT {trt.__version__} | torch {torch.__version__} | 모델 {runs}")

    calib = None
    report = {"tensorrt": trt.__version__, "torch": torch.__version__,
              "time": time.strftime("%Y-%m-%d %H:%M:%S"), "models": {}}
    report_path = bundle / "build_report.json"
    if report_path.exists():
        report["models"] = json.loads(report_path.read_text()).get("models", {})
    for run in runs:
        out_dir = bundle / "models" / run / "engines"
        out_dir.mkdir(exist_ok=True)
        for precision in [p.strip() for p in args.precisions.split(",") if p.strip()]:
            if precision == "int8" and calib is None:
                with np.load(bundle / "calib_inputs.npz", allow_pickle=False) as blob:
                    calib = np.array(blob["images"])
            print(f"[{run} {precision}]", flush=True)
            report["models"].setdefault(run, {})[precision] = build(
                bundle / "models" / run / "encoder.onnx", precision, out_dir,
                calib_images=calib, workspace_gb=args.workspace_gb)
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"완료 -> {report_path}")


if __name__ == "__main__":
    main()
