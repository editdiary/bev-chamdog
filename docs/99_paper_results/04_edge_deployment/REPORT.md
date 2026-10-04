# 실험 04 — Edge Deployment: Jetson AGX Orin 추론 속도와 TensorRT의 속도-정확도 교환

> 대상 실험 1 고정 split 모델 5개(`runs/99_full_campaign/01_overall/fixed_split`, 시드 0~4, epoch 100)
> · 측정 2026-10-04 16:38~17:02(보드) · 인용 수치는 같은 폴더 `data/*.csv`에 있고, 그 CSV는
> `make_package.py`가 보드 결과(`runs/jetson_bundle/results/`)에서 생성한 것이다.

---

## 0. 세 줄 요약

1. 학습한 모델을 그대로 PyTorch fp32로 돌리면 Jetson AGX Orin에서 한 프레임(카메라 3장)에
   **67.5 ms, 14.8 FPS**다. 그중 **81.8 %가 이미지 인코더**(ResNet-101)다.
2. 인코더만 TensorRT로 바꾸면 **fp16 23.6 ms(42.3 FPS, 2.86배)**, **INT8 18.6 ms(53.7 FPS,
   3.63배)**. fp16은 7개 지표 모두 유의한 차이가 없다.
3. INT8은 IoU_free **−0.0055**, precision **−0.0081**, BF@0.20 m **−0.0106** 등 유의하게 잃고,
   손실 크기가 모델마다 5배 넘게 다르다(IoU_free −0.0021 ~ −0.0115). **배포 권고는 fp16이다.**

---

## 1. 무엇을 물었고 어떻게 쟀는가

**질문 둘.** (1) 학습한 모델이 실제 운용 보드에서 얼마나 빠른가. (2) TensorRT로 낮은 정밀도를 쓰면
속도와 정확도를 어떻게 바꾸는가.

| 항목 | 값 |
|---|---|
| 보드 | Jetson AGX Orin, L4T R36.4(JetPack 6.1), **MAXN + `jetson_clocks`** (GPU 클럭 측정 내내 최대 근처 고정, 최고 온도 62 °C) |
| 소프트웨어 | PyTorch 2.8.0(CUDA 12.6), TensorRT 10.3 |
| 모델 | 실험 1 고정 split 시드 0~4, 마지막 epoch(100). 논문 대표 모델과 같다 |
| 입력 | 카메라 3장 × 512×288, 배치 1(실시간 처리 조건) |
| 정확도 평가 | 검증 75프레임, **논문과 같은 지표 함수**(`tools/jetson/score.py`) |
| 유의성 | 같은 시드끼리 짝지은 양측 t-검정의 95 % 신뢰구간, 대조 = 같은 모델의 PyTorch fp32 |

**세 구성.** 모델을 "인코더"와 "BEV 단계(특징 집어 오기 + 카메라 평균 + BEV 압축 + 디코더)"로 나누고
인코더만 바꾼다.

| 구성 | 인코더 | BEV 단계 |
|---|---|---|
| PyTorch fp32 | PyTorch fp32 | PyTorch fp32 |
| TensorRT fp16 | TensorRT fp16 | PyTorch fp32 |
| TensorRT INT8 | TensorRT INT8 (학습 프레임 192장으로 엔트로피 보정) | PyTorch fp32 |

**왜 인코더만인가** — 논문에 쓸 근거는 셋이다.
- **시간의 대부분이 인코더다.** PyTorch fp32에서 인코더가 전체의 **81.8 %**(55.2 / 67.5 ms)다(§3).
- **인코더는 표준 2D CNN이라 변환이 깔끔하다.** BEV 단계에는 어안 투영 좌표로 특징을 집어 오는
  3차원 `grid_sample`이 있어 TensorRT 변환이 버전에 따라 막힐 수 있고, 거기서 정밀도를 낮추면
  카메라 평균과 디코더 입력이 함께 거칠어진다.
- **투영은 상수다.** 리그가 고정이라 투영 좌표를 fp32로 한 번 계산해 박아 두었다 — 배포 단계에 어안
  투영 계산 자체가 없다(`projects/deploy/split_model.py`).

**잰 것.** 지연은 이미지가 GPU에 올라간 뒤 `p(free)` 맵이 나올 때까지다. **이미지 디코딩·리사이즈
(1280×720 → 512×288)는 넣지 않았다.** 어안을 펴는 단계(undistortion)는 이 방법에 아예 없다.
인코더 / BEV 단계 / 전체를 따로 재고, 라운드 3 × 모델 5 × 구성 3을 **번갈아** 쟀다(한 구성을 몰아
재면 온도·클럭 변화가 구성 차이로 섞인다). 각 측정은 워밍업 50회 + 300회, 보고값은 라운드별 중앙값의
중앙값을 다시 모델 5개로 평균한 것이다.

## 2. 결과가 믿을 만한지

| 검사 | 결과 |
|---|---|
| 분할 = 원래 모델 | 서버에서 모델 5개 × 검증 75프레임 전부 **비트 단위로 같다** (`tests/deploy`, `provenance/environment.json`) |
| 보드 PyTorch = 서버 | 보드의 PyTorch fp32 예측이 서버와 판정 0.001~0.002 %만 다르다 (`data/per_seed.csv` 끝 열) |
| 측정 반복성 | 라운드끼리 0.05 ms 안. 모델 간 SD도 0.04 ms 이하 (`table1_speed.csv`) |
| 장시간 실행 | 구성마다 60초 연속 실행에서 처음 10 %와 마지막 10 % 지연 비가 0.9994 / 1.0 / 0.9999 — **느려지지 않는다** (`sustained_run.csv`) |
| INT8 엔진 | 모델 5개 모두 합성곱 97개 중 **95개 INT8**, 2개 fp32(업샘플 블록의 두 번째 3×3 합성곱과 인코더 마지막 1×1 출력층 -- 정규화 상수에 INT8 범위가 없어 TensorRT가 남긴 곳) (`engine_layers.csv`) |

## 3. 결과 — 속도

원본 `data/table1_speed.csv`. 모델 5개의 평균(전체·인코더·BEV 단계 지연의 모델 간 SD는 0.04 ms 이하).

| 구성 | 전체 | FPS | 가속 | 인코더 | BEV 단계 | 인코더 몫 | p99 |
|---|---|---|---|---|---|---|---|
| PyTorch fp32 | 67.5 ms | 14.8 | 1.00× | 55.2 ms | 12.4 ms | 81.8 % | 68.0 ms |
| TensorRT fp16 | 23.6 ms | 42.3 | **2.86×** | 11.2 ms | 12.4 ms | 47.5 % | 23.8 ms |
| TensorRT INT8 | 18.6 ms | 53.7 | **3.63×** | 6.2 ms | 12.4 ms | 33.5 % | 18.7 ms |

**읽는 법.**
- 인코더 자체는 fp16에서 약 4.9배, INT8에서 약 8.9배 빨라진다. 전체 가속이 그보다 작은 것은
  **BEV 단계 12.4 ms가 그대로 남기 때문**이다.
- 그래서 최적화 뒤에는 **병목이 BEV 단계로 옮겨 간다** — fp16에서 전체의 절반 이상이다.
  이것이 다음 최적화 대상이다(§6 한계).
- 지연 꼬리(p99)가 중앙값보다 1 ms 안쪽이다. 실시간 제어 주기를 잡을 때 중앙값을 써도 된다.

## 4. 결과 — 정확도

원본 `data/table2_accuracy_paired.csv`(짝지은 차이와 신뢰구간), 논문 표 `data/table_paper_edge.csv`.

| 지표 | PyTorch fp32 | fp16 차이 | INT8 차이 | INT8 95 % CI |
|---|---|---|---|---|
| IoU_free | 0.816 | 유의하지 않음 | **−0.0055*** | [−0.0103, −0.0007] |
| IoU_non-free | 0.945 | 유의하지 않음 | **−0.0024*** | [−0.0045, −0.0003] |
| Precision | 0.883 | 유의하지 않음 | **−0.0081*** | [−0.0154, −0.0008] |
| Recall | 0.917 | 유의하지 않음 | +0.0021 | [−0.0000, +0.0043] |
| BF@0.10 m | 0.614 | 유의하지 않음 | −0.0122 | [−0.0279, +0.0034] |
| BF@0.20 m | 0.839 | 유의하지 않음 | **−0.0106*** | [−0.0200, −0.0011] |
| BF@0.30 m | 0.904 | 유의하지 않음 | **−0.0078*** | [−0.0143, −0.0014] |

(`*` = 95 % 신뢰구간이 0을 뺀다. INT8은 7개 지표 모두 5/5 시드가 같은 방향이다.)

**fp16은 정확도를 바꾸지 않는다.** 지표 차이가 모두 ±0.0012 안이고, 판정(τ = 0.5)이 바뀐 칸은
평균 0.013 %다.

**INT8은 "갈 수 있다"고 말하는 쪽으로 기운다.** precision이 떨어지고 recall은 오르는 방향이다.
막힌 칸을 비어 있다고 보는 오류가 늘어나는 쪽이라 **로봇에게는 위험한 방향**이다. 판정이 바뀐 칸은
평균 0.62 %다.

**INT8 손실은 모델마다 크게 다르다**(`data/per_seed_diff_vs_fp32.csv`).

| 모델 | IoU_free | Precision | BF@0.10 m | 판정 뒤집힘 |
|---|---|---|---|---|
| `final_s0` | −0.0115 | −0.0174 | −0.0306 | 1.05 % |
| `final_s1` | −0.0030 | −0.0044 | −0.0088 | 0.44 % |
| `final_s2` | −0.0072 | −0.0105 | −0.0190 | 0.70 % |
| `final_s3` | −0.0038 | −0.0049 | −0.0027 | 0.47 % |
| `final_s4` | −0.0021 | −0.0034 | −0.0000 | 0.47 % |

같은 학습 설정에서 시드만 다른 모델인데 INT8 손실이 5배 넘게 갈린다. BF@0.10 m가 평균으로는
크게 떨어지는데도 유의하지 않은 것은 이 산포 때문이다. 즉 **INT8은 "얼마나 잃는가"를 모델을 만들기
전에는 알 수 없고, 배포할 모델마다 다시 검증해야 한다.** 사전 측정(같은 `final_s0`, 엔진을 따로
빌드)에서도 판정 뒤집힘이 같은 1.05 %였다 — 손실은 보정의 우연이 아니라 그 모델의 성질이다.

## 5. 결과 — 전력

원본 `data/power.csv`. 구성마다 60초 연속 실행 구간의 tegrastats 중앙값(보드 세 레일의 합).

| 구성 | 전력 | 프레임당 에너지 | 대기 대비 추가 에너지 |
|---|---|---|---|
| 대기 | 11.0 W | — | — |
| PyTorch fp32 | 46.1 W | 3110 mJ | 2368 mJ |
| TensorRT fp16 | 48.4 W | 1143 mJ | 883 mJ |
| TensorRT INT8 | 44.6 W | 830 mJ | 625 mJ |

GPU 사용률이 세 구성 모두 98~99 %라 순간 전력은 비슷하다. **차이는 프레임당 에너지에서 난다** —
같은 일을 짧게 끝내므로 fp16이 fp32의 약 37 %, INT8이 약 27 %다. 다만 이 측정은 GPU를 쉬지 않고
돌린 값이다. 실제 운용에서 카메라 주기(예: 15~30 Hz)로 돌리면 남는 시간에 GPU가 쉬므로 평균 전력이
이보다 낮다.

## 6. 해석과 논문에 쓸 문장

**쓸 수 있는 주장.**
- 학습한 모델은 수정 없이 Orin에서 14.8 FPS로 돈다. 인코더만 TensorRT fp16으로 바꾸면 **정확도
  손실 없이 42.3 FPS**다.
- 어안을 펴지 않고 투영을 상수로 박으므로 배포 단계에 왜곡 보정·투영 계산이 없다.
- INT8은 추가로 1.27배 빨라지지만 정확도를 유의하게 잃고, 손실이 모델마다 크게 다르다.

**한계(본문에 정직하게).**
- **BEV 단계 12.4 ms는 최적화하지 않았다.** fp16 이후 전체의 절반 이상이 여기다. 이 단계까지
  TensorRT로 옮기면 더 빨라질 여지가 있다(3차원 `grid_sample` 변환이 관건).
- 지연에 이미지 디코딩·리사이즈는 들어 있지 않다.
- INT8은 **학습 후 양자화(PTQ, post-training quantization)**만 시험했다 -- 학습이 끝난 모델을 그대로 두고 학습 프레임 192장으로 층별 값 범위를 재어(엔트로피 보정) 8비트로 바꾼다. 학습 때부터 양자화를 흉내 내 모델이 적응하게 하는 **양자화 인지 학습(QAT)**은 범위 밖이고, 보통 손실이 더 작으므로 향후 과제로 쓸 수 있다.
- INT8 정확도는 TensorRT 버전에 따라 달라질 수 있다 -- 서버의 10.16은 같은 설정에서 합성곱 60/95만 INT8로 골랐다. 논문 값은 보드(10.3)에서 잰 것이다.
- 모델 크기·입력 해상도는 고정이다(512×288, ResNet-101).

**쓰면 안 되는 표현.** "INT8도 정확도가 거의 같다"(유의하게 다르다) · "fp16이 전력을 줄인다"(순간
전력은 오히려 조금 높다 — 줄어드는 것은 프레임당 에너지다) · "실시간 X FPS 달성"을 목표 주기 없이
쓰는 것(로봇의 요구 주기를 함께 적는다).

## 7. 논문 구성 제안

| 항목 | 내용 | 원본 |
|---|---|---|
| **Table** | 3행 × (지연, FPS, 7지표), 유의한 칸 `*` | `data/table_paper_edge.csv` + `table_paper_caption.txt` |
| **Figure 1** | (a) 구성별 지연 분해 (b) fp32 대비 지표 변화와 95 % CI | `figures/fig1_latency_accuracy.*` |
| 본문 숫자 | 인코더 몫 81.8 %, 가속 2.86× / 3.63×, 프레임당 에너지 | `table1_speed.csv`, `power.csv` |
| 부록 | 시드별 INT8 손실, 연속 실행, 층 정밀도 | `per_seed_diff_vs_fp32.csv`, `sustained_run.csv`, `engine_layers.csv` |

## 8. M&M 영문 초안

> **Edge deployment.** We measure inference on an NVIDIA Jetson AGX Orin (64 GB; JetPack 6.1, CUDA 12.6,
> PyTorch 2.8.0, TensorRT 10.3) in MAXN mode with clocks locked (`jetson_clocks`). The five fixed-split
> models of Section X are evaluated. Each model is split into an image encoder (ResNet-101 trunk with
> input normalisation) and a BEV stage (feature sampling, masked multi-camera averaging, BEV compression
> and decoder). Because the camera rig is fixed, the Double Sphere sampling grid and its validity mask are
> precomputed once in fp32 and stored as constants, so no projection or undistortion is computed at run
> time; the split model is bit-identical to the original. We compare three configurations: the whole model
> in PyTorch fp32, and the encoder replaced by a TensorRT fp16 or INT8 engine while the BEV stage stays in
> PyTorch fp32. The encoder is exported to ONNX (opset 17) and the engines are built on the device; the
> INT8 engine uses post-training quantization with entropy calibration on 192 training frames (fp16
> fallback for layers without INT8 scales), which executes 95 of its 97 convolutions in INT8. Latency is
> measured per frame (three 512×288 images, batch 1) from the GPU-resident input tensor to the free-space
> probability map, excluding image decoding and resizing, with CUDA events over 300 iterations after 50
> warm-up iterations. Three rounds are interleaved across models and configurations; we report the median
> over rounds of the per-round median, averaged over the five models. Accuracy is re-evaluated on the 75
> validation frames with the metrics of Section Y and compared with the fp32 model of the same seed by a
> two-sided paired t-test (95 % CI). Board power is read with `tegrastats` (200 ms) as the sum of the
> VDD_GPU_SOC, VDD_CPU_CV and VIN_SYS_5V0 rails during a 60 s continuous run per configuration; energy per
> frame is power × median latency.

## 9. 재생성과 재측정

```bash
python docs/99_paper_results/04_edge_deployment/make_package.py          # data/ provenance/
python docs/99_paper_results/04_edge_deployment/figures/make_latency_accuracy.py
```

보드 측정을 다시 하려면 `tools/jetson/README.md`(꾸러미 생성 → 엔진 빌드 → 측정 → 결과 회수).
사전 측정(시드 0 하나, 2026-10-04 15시)은 `runs/jetson_pilot_2026-10-04/`에 기록으로 남아 있다.
