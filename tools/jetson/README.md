# Jetson AGX Orin 추론 속도 측정 꾸러미

이 폴더 하나로 보드에서 측정한다. 저장소 코드·데이터셋은 필요 없다.
서버에서 `python tools/jetson/make_bundle.py`로 만든다(`manifest.json`에 출처와 모델별 검증 결과).

## 무엇을 재나

모델 5개 = 실험 1 고정 split의 시드 0~4(`final_s0`~`final_s4`, epoch 100). 입력은 카메라 3장 × 512×288.

| 구성 | 인코더(ResNet-101) | BEV 단계(특징 집어 오기 + 디코더) |
|---|---|---|
| `pytorch_fp32` | PyTorch fp32 | PyTorch fp32 |
| `trt_fp16` | **TensorRT fp16** | PyTorch fp32 |
| `trt_int8` | **TensorRT INT8** (학습 프레임 192장으로 보정) | PyTorch fp32 |

`bench.py`가 하는 일: 대기 전력 20초 → 모델 × 구성마다 검증 75프레임 예측 → 지연(라운드 3 × 모델 5 ×
구성 3, 번갈아) → 구성마다 60초 연속 실행. 모든 구간의 시각이 기록돼 서버가 tegrastats를 구성별로 자른다.

## 보드에서 할 일

```bash
cd ~/Desktop/bev-chamdog && source .venv/bin/activate     # torch·tensorrt가 있는 환경
cd runs && tar -xf jetson_bundle.tar && cd jetson_bundle

# 0) 최대 성능 + 클럭 고정. **sudo가 성공했는지 꼭 본다** (지난번엔 비밀번호가 틀려 안 걸렸다)
sudo nvpmodel -m 0
sudo jetson_clocks
sudo jetson_clocks --show | grep -i gpc     # GPU 현재 클럭이 최대값과 같으면 정상
nvpmodel -q                                  # MAXN

# 1) 엔진 빌드 -- 모델 5개 × (fp16 + INT8). 약 30분. 끊기면 같은 명령을 다시 치면 이어서 한다
python3 build_engines.py

# 2) 측정 -- 약 25분. 그동안 보드를 다른 일에 쓰지 않는다
sudo tegrastats --interval 200 --logfile results_tegrastats.log --start
python3 bench.py
sudo tegrastats --stop
mv results_tegrastats.log results/
cp build_report.json results/
for d in models/*/; do mkdir -p results/engines/$(basename $d); cp $d/engines/*.json results/engines/$(basename $d)/; done
```

끝나면 **`results/` 폴더 하나**를 서버의 `runs/jetson_bundle/results/`로 보낸다(층 정밀도 json과
빌드 기록도 위 마지막 줄들이 그 안에 넣는다). 엔진 파일(`*.engine`)은 보내지 않는다.

## 함정

- **`pip install`로 torch를 건드리지 않는다.** JetPack용 wheel이 아니면 CUDA가 죽는다.
- 확인: `python3 -c "import torch, tensorrt; print(torch.cuda.is_available(), tensorrt.__version__)"` → `True 10.3.0`.
- 엔진은 그 보드·그 TensorRT 버전 전용이다. TensorRT를 바꾸면 다시 빌드한다(`models/*/engines/` 삭제).
- 측정 중에는 다른 GPU 프로그램(원격 데스크톱의 무거운 앱 포함)을 끈다.
- 빌드 중 `DeprecationWarning`(INT8 보정 API)과 `Missing scale and zero-point ... upsampling_layer` 경고는
  정상이다 -- 정규화 층 상수라 그 부분만 INT8 대신 fp16으로 돈다는 뜻이다.
