"""[Jetson 보드에서 실행] 모델마다 세 구성의 추론 속도·정확도용 예측·전력 구간을 잰다.

    python3 bench.py                         # 꾸러미 폴더 안에서 (먼저 build_engines.py)

| 구성 | 인코더 | BEV 단계 |
|---|---|---|
| `pytorch_fp32` | TorchScript fp32 | TorchScript fp32 |
| `trt_fp16` | TensorRT fp16 엔진 | TorchScript fp32 |
| `trt_int8` | TensorRT INT8 엔진 | TorchScript fp32 |

순서와 무엇을 재나:

1. **대기** (`--idle_seconds`): 아무것도 안 돌리는 구간 -- 대기 전력 기준선.
2. **정확도용 예측**: 모델 × 구성마다 검증 프레임 전부 -> `results/<run>/<구성>_prob_free.npz`.
   채점은 서버가 논문과 같은 코드로 한다(`tools/jetson/score.py`).
3. **지연** (`--rounds` × 모델 × 구성, 번갈아): 한 프레임(카메라 S장)을 GPU에 올려 둔 상태에서
   인코더만 / BEV 단계만 / 전체를 CUDA 이벤트로 잰다. 매 호출 동기화(배치 1 실시간 조건).
   **번갈아 재는 이유**: 한 구성을 몰아서 재면 그사이 온도·클럭 변화가 구성 차이로 섞인다.
4. **연속 실행** (`--sustain_seconds`, 구성마다): 첫 모델로 전체 추론을 쉬지 않고 돌린다 --
   장시간 실행에서 느려지지 않는지(발열·클럭 저하)와 **구성별 정상 상태 전력**을 본다.

모든 구간의 시작·끝 시각을 `bench.json`의 `segments`에 남긴다. 서버가 tegrastats 기록을 이 시각으로
잘라 구성별 전력을 낸다. **디스크 읽기와 `uint8 -> float` 변환은 지연에 넣지 않는다.**

결과: `results/` 폴더 전체를 서버로 보낸다.
"""
import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

try:
    import tensorrt as trt
except ImportError:  # PyTorch 기준선만 잴 때
    trt = None

CONFIGS = ("pytorch_fp32", "trt_fp16", "trt_int8")


class TrtEncoder:
    """TensorRT 엔진을 torch 텐서 입출력으로 감싼다 -- 복사 없이 같은 GPU 메모리를 쓴다."""

    def __init__(self, engine_path: Path):
        logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inp = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        out_dtype = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}[
            self.engine.get_tensor_dtype(self.out)]
        self.output = torch.empty(tuple(self.engine.get_tensor_shape(self.out)), dtype=out_dtype,
                                  device="cuda")
        self.context.set_tensor_address(self.out, self.output.data_ptr())

    def __call__(self, x):
        x = x.contiguous()
        self.context.set_tensor_address(self.inp, x.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.output.float() if self.output.dtype != torch.float32 else self.output

    @property
    def device_memory_mb(self):
        size = getattr(self.engine, "device_memory_size_v2", None) or self.engine.device_memory_size
        return round(size / 1e6, 1)


def time_ms(fn, x, iters, warmup):
    """호출마다 CUDA 이벤트로 잰다(ms). 배치 1 실시간 조건이라 매 호출 동기화한다."""
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        torch.cuda.synchronize()
        out = []
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start.record()
            fn(x)
            end.record()
            end.synchronize()
            out.append(start.elapsed_time(end))
    return out


def summarize(samples):
    s = sorted(samples)
    pick = lambda q: s[min(len(s) - 1, int(round(q * (len(s) - 1))))]  # noqa: E731
    return {"mean": statistics.fmean(s), "median": statistics.median(s), "p90": pick(0.90),
            "p99": pick(0.99), "min": s[0], "max": s[-1], "n": len(s)}


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as exc:  # noqa: BLE001 -- 환경 기록은 실패해도 측정을 막지 않는다
        return f"(실패: {exc})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--runs", default="", help="쉼표 구분. 비우면 models/ 아래 전부")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--idle_seconds", type=float, default=20)
    ap.add_argument("--sustain_seconds", type=float, default=60)
    args = ap.parse_args()

    bundle = Path(args.bundle)
    results = bundle / "results"
    results.mkdir(exist_ok=True)
    runs = ([r.strip() for r in args.runs.split(",") if r.strip()]
            or sorted(p.name for p in (bundle / "models").iterdir() if p.is_dir()))
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    torch.backends.cudnn.benchmark = True
    segments = []

    def segment(kind, **meta):
        class _Seg:
            def __enter__(self_):
                torch.cuda.synchronize()
                self_.row = {"kind": kind, **meta, "start": time.time()}
                return self_

            def __exit__(self_, *exc):
                torch.cuda.synchronize()
                self_.row["end"] = time.time()
                segments.append(self_.row)
        return _Seg()

    with np.load(bundle / "val_inputs.npz", allow_pickle=False) as blob:
        val = np.array(blob["images"])
        ids = np.array(blob["sample_ids"])

    # === 모델 적재 ======================================================================
    stages = {}
    for run in runs:
        d = bundle / "models" / run
        bev = torch.jit.load(str(d / "bev_stage.ts")).cuda().eval()
        encs = {}
        for name in configs:
            if name == "pytorch_fp32":
                encs[name] = torch.jit.load(str(d / "encoder_fp32.ts")).cuda().eval()
            else:
                engine = d / "engines" / f"encoder_{name.split('_', 1)[1]}.engine"
                if trt is None or not engine.exists():
                    raise SystemExit(f"!! {engine}가 없다 -- 먼저 build_engines.py를 실행한다")
                encs[name] = TrtEncoder(engine)
        stages[run] = (encs, bev)
    print(f"모델 {runs} | 구성 {configs} | 프레임 입력 {tuple(val.shape[1:])} | 검증 {len(val)}프레임")

    # === 1. 대기 ========================================================================
    with segment("idle"):
        time.sleep(args.idle_seconds)

    # === 2. 정확도용 예측 ===============================================================
    accuracy = {}
    with torch.no_grad():
        for run, (encs, bev) in stages.items():
            with np.load(bundle / "models" / run / "reference_prob_free.npz", allow_pickle=False) as blob:
                reference = np.array(blob["prob_free"])
            (results / run).mkdir(exist_ok=True)
            for name, enc in encs.items():
                probs = []
                with segment("accuracy", run=run, config=name):
                    for i in range(len(val)):
                        x = torch.from_numpy(val[i]).cuda().float().div_(255.0)
                        probs.append(bev(enc(x))[0].float().cpu().numpy())
                probs = np.stack(probs)
                np.savez_compressed(results / run / f"{name}_prob_free.npz",
                                    prob_free=probs.astype(np.float32), sample_ids=ids)
                diff = np.abs(probs - reference)
                flips = float(((probs > 0.5) != (reference > 0.5)).mean())
                accuracy.setdefault(run, {})[name] = {
                    "max_abs_prob_diff_vs_server": float(diff.max()),
                    "mean_abs_prob_diff_vs_server": float(diff.mean()),
                    "decision_flip_rate_vs_server": flips}
                print(f"  {run} {name:13s} 서버 대비 확률 차이 최대 {diff.max():.2e} |"
                      f" 판정 뒤집힘 {100 * flips:.3f} %", flush=True)

    # === 3. 지연 (라운드 × 모델 × 구성, 번갈아) ========================================
    x = torch.from_numpy(val[0]).cuda().float().div_(255.0)
    timings = {run: {name: {"encoder": [], "bev_stage": [], "end_to_end": []} for name in configs}
               for run in runs}
    for r in range(args.rounds):
        for run, (encs, bev) in stages.items():
            with torch.no_grad():
                feat = encs[configs[0]](x).clone()
            for name, enc in encs.items():
                e2e = lambda t, enc=enc, bev=bev: bev(enc(t))  # noqa: E731
                for part, fn, arg in (("encoder", enc, x), ("bev_stage", bev, feat), ("end_to_end", e2e, x)):
                    with segment("latency", run=run, config=name, part=part, round=r):
                        timings[run][name][part].append(summarize(time_ms(fn, arg, args.iters, args.warmup)))
                print(f"  라운드 {r + 1}/{args.rounds} {run} {name:13s} 전체 중앙값"
                      f" {timings[run][name]['end_to_end'][-1]['median']:.2f} ms", flush=True)

    # === 4. 연속 실행 (첫 모델, 구성마다) ===============================================
    sustain = {}
    encs, bev = stages[runs[0]]
    for name, enc in encs.items():
        per_call, stamps = [], []
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with segment("sustain", run=runs[0], config=name), torch.no_grad():
            t_end = time.time() + args.sustain_seconds
            while time.time() < t_end:
                start.record()
                bev(enc(x))
                end.record()
                end.synchronize()
                per_call.append(start.elapsed_time(end))
                stamps.append(time.time())
        n = len(per_call)
        tenth = max(1, n // 10)
        sustain[name] = {"seconds": args.sustain_seconds, "calls": n,
                         "first_10pct": summarize(per_call[:tenth]),
                         "last_10pct": summarize(per_call[-tenth:]),
                         "all": summarize(per_call)}
        print(f"  연속 {name:13s} {n}회, 처음 10 % 중앙값 {sustain[name]['first_10pct']['median']:.2f} ms"
              f" -> 마지막 10 % {sustain[name]['last_10pct']['median']:.2f} ms", flush=True)

    # === 정리 ===========================================================================
    speed = {}
    for run in runs:
        for name in configs:
            parts = timings[run][name]
            row = {part: {"median_of_round_medians": statistics.median(s["median"] for s in rounds),
                          "rounds": rounds} for part, rounds in parts.items()}
            e2e_ms = row["end_to_end"]["median_of_round_medians"]
            row["fps"] = 1000.0 / e2e_ms
            row["encoder_share_of_end_to_end"] = row["encoder"]["median_of_round_medians"] / e2e_ms
            enc = stages[run][0][name]
            if isinstance(enc, TrtEncoder):
                row["engine_device_memory_mb"] = enc.device_memory_mb
            speed.setdefault(run, {})[name] = row

    env = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
           "tensorrt": trt.__version__ if trt else None, "python": platform.python_version(),
           "l4t": _cmd(["cat", "/etc/nv_tegra_release"]).splitlines()[:1],
           "nvpmodel": _cmd(["nvpmodel", "-q"]), "device": torch.cuda.get_device_name(0),
           "torch_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), "utc_offset_s": -time.timezone}
    out = {"runs": runs, "configs": configs, "iters": args.iters, "warmup": args.warmup,
           "rounds": args.rounds, "idle_seconds": args.idle_seconds,
           "sustain_seconds": args.sustain_seconds, "env": env, "speed": speed,
           "sustain": sustain, "accuracy_vs_server": accuracy, "segments": segments}
    (results / "bench.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n모델       구성            전체 ms    FPS   인코더 ms  BEV ms  인코더 몫")
    for run in runs:
        for name, row in speed[run].items():
            print(f"{run:10s} {name:13s} {row['end_to_end']['median_of_round_medians']:8.2f}"
                  f" {row['fps']:6.1f} {row['encoder']['median_of_round_medians']:9.2f}"
                  f" {row['bev_stage']['median_of_round_medians']:7.2f}"
                  f" {100 * row['encoder_share_of_end_to_end']:8.1f} %")
    print(f"\n결과 -> {results}  (이 폴더를 통째로 서버로 보낸다)")


if __name__ == "__main__":
    main()
