# 현재 모델의 논문용 구조도

2026-09-16 작성·모식도 중심 개정. 구조의 근거는 코드 `d43bca3`와 실제 `D_range_s0` 실행 설정이다.
`dataset/figure/bev_pipeline_reference.html`을 참고하되 현행 코드와 대조했다.
이 문서는 그림의 설명·출처이며 모델이나 loss의 새 정본은 아니다.

## 파일

- **본문 그림:** [PDF](model_architecture.pdf), [SVG](model_architecture.svg), [PNG](model_architecture.png)
- **학습 목적함수 보조 그림:** [PDF](training_objective.pdf), [SVG](training_objective.svg), [PNG](training_objective.png)
- **브라우저 미리보기:** [model_architecture.html](model_architecture.html)
- **편집 원본:** [tools/render_model_architecture.py](../../tools/render_model_architecture.py)
- **실측 형상·체크포인트 출처:** [architecture_assets/provenance.json](architecture_assets/provenance.json)

PDF는 글꼴이 포함된 벡터 파일이고 SVG는 편집 가능한 텍스트를 유지한다. 사진과 실제 예측 지도만
내장 래스터다. PNG는 300 dpi이며 본문 그림 4320×1470, 보조 그림 4320×1530이다.
본문 그림은 2단 논문의 두 단 전체 폭에 배치하는 용도다. 한 단에 넣으면 세부 라벨이 작아진다.

```bash
conda run -n bev-chamdog python tools/render_model_architecture.py
```

기본 명령은 아키텍처 그림만 재생성한다. 목적함수 그림은 `--figure objective`를 명시할 때만
재생성하며, 모식도 중심 개정에서는 기존 목적함수 파일을 변경하지 않았다.
그림 재생성은 저장된 예제 이미지만 읽으므로 GPU·데이터셋·체크포인트가 필요 없다.
레이아웃과 라벨은 생성 스크립트에서 수정한다. SVG만 직접 수정하면 재생성 때 사라진다.

## 무엇을 표현했나

오른쪽으로 좁아지는 사다리꼴은 공유 인코더, 넓어지는 사다리꼴은 BEV 디코더다.
세워 쌓은 평면은 특징맵, 수평 격자 네 장은 높이별 표본, 한 장의 수평 격자는 압축된 BEV 특징이다.
사다리꼴은 기능을 나타내는 모식 표현이며, 실제 디코더가 단조롭게 업샘플링만 한다는 뜻은 아니다.
디코더 내부는 아래 표와 같은 U-Net이다. 내부 텐서 표와 상세 인코더·디코더 패널은 그림에서 빼고
이 설명 문서에 남겼다. 입력·출력 해상도, 인코더 stride, 표본 높이만 작게 보조 표기한다.

파란 도형은 학습 모듈과 그 특징, 황토색은 고정된 기하와 카메라 병합이다.
보조 그림의 보라색 점선은 학습에만 필요한 계산을 뜻한다.

| 단계 | 현행 구성과 텐서 크기 — 배치 B 생략 | 근거 |
|---|---|---|
| 입력 | front/left/right 3대, 각 RGB 288×512; ImageNet 정규화 | `projects/datasets/robot_simplebev.py::RobotBEVDataset`; `Segnet.forward` |
| 이미지 인코더 | 공유 ResNet-101; layer2의 512×36×64와 layer3의 1024×18×32를 upsample + concat; 3×3 conv 두 번과 1×1 투영 → 128×36×64 | `third_party/models/simple_bev/nets/segnet.py::Encoder_res101`, `UpsamplingConcat` |
| lifting | 120×4×120 질의점 → 카메라 좌표 변환 → Double Sphere 투영 → bilinear 특징 표본; 카메라별 128×120×4×120 | `projects/models/double_sphere_vox.py::DoubleSphereVoxUtil` |
| 카메라 병합 | 카메라 축의 masked mean → 128×120×4×120 | `Segnet.forward`, `reduce_masked_mean` |
| 높이 압축 | Y축을 C축으로 접어 512×120×120; 3×3 conv(512→128) + InstanceNorm + GELU | `Segnet.forward`, `bev_compressor` |
| BEV 디코더 | ResNet-18 기반 U-Net; 120→60→30→15→30→60→120; upsample 경로는 skip을 **더한다** | `projects/models/simplebev_three_class.py::ThreeClassDecoder` |
| segmentation head | 3×3 conv(128→128) + IN + ReLU + 1×1 conv(128→2) | `build_three_class_head(num_classes=2)` |
| 출력 | 2×120×120 logits, 순서 not-free/free. argmax → 이진 지도. softmax는 확률이 필요한 loss와 threshold 분석에서 사용 | `projects/common/binary_metrics.py` |

표본 높이는 지면 기준 **0, 0.5, 1.0, 1.5 m**이고 bin 범위는 −0.25~1.75 m다.
그림의 `Y`는 높이, `Z/X`는 BEV 배열의 두 축이다. ego 좌표의 축 이름과 다르다.
BEV 범위는 전방 4 m / 후방 2 m / 좌우 ±3 m로 전후 비대칭이다. 출력 지도의 화살표는
ego 위치(중앙 열, row 약 80)와 전방을 가리키는 표시다.

카메라 병합의 실제 마스크는 표본 특징의 `abs(feature)>0`이며 채널별로 적용된다.
DS 정의역·화면 범위를 벗어난 표본은 lifting에서 먼저 0이 된다. 이는 물체에 의한 가림을
추정하는 visibility head나 학습되는 attention이 아니다. 높이 축도 평균하거나 max-pool하지 않는다.
lifting에는 별도 depth 예측기가 없으며, 입력 어안 영상을 먼저 핀홀 영상으로 펴지 않고
원래 어안 특징맵에서 표본한다. 추론 입력은 이미지와 캘리브레이션이며 LiDAR 입력은 없다.

## Simple-BEV 계승 요소와 프로젝트 적용 요소

| 구분 | 내용 |
|---|---|
| 계승 | ResNet-101 특징 추출, voxel-wise 특징 표본, 카메라 masked mean, 높이-채널 접기와 BEV 압축, ResNet-18 U-Net과 additive skip |
| 프로젝트의 기하 적용 | 자체 front/left/right 3-cam 리그, Double Sphere 투영, 120×120 @ 5 cm 격자, `pixel_center` 표본 규약과 offset=0 |
| 프로젝트 내 채택 설정 | 기존 실험의 Y=1에서 Y=4로 변경; 높이 0/0.5/1.0/1.5 m 표본. **여러 높이를 쓰는 아이디어 자체를 이 프로젝트의 신규 발명으로 표현하지 않는다.** |
| 출력 변경 | 단일 2-logit binary head. 사용하지 않는 feature/instance-center/instance-offset heads 제거 |
| 학습 변경 | soft-boundary 감독과 radial free-length 보조항. 추론 시 추가 head나 GT 입력을 요구하지 않는다 |

`not-free`는 라벨의 `occupied ∪ unknown`이다. 그림에 세 클래스 출력이나 두 head를 넣지 않았다.
`ThreeClassSegnet`이라는 클래스명은 역사적 이름이며 실제 체크포인트의 출력은 두 채널이다.
encoder/decoder 전체가 새로 설계된 것처럼 해석하지 않도록 계승과 적용을 구별한다.

현재 런의 `INIT_CHECKPOINT=none`은 SynWoodScape 체크포인트를 가져오지 않는다는 뜻이다.
ResNet-101 trunk에는 여전히 ImageNet 초기 가중치가 들어간다. BEV 모듈은 랜덤 초기화한다.
확정 실험 설정은 `configs/loss_effect.sh`와 실행의 `config.json`을 기준으로 확인했다.
`train_robot_bev_finetune.sh`의 아무 인자 없는 기본값은 아직 `three_class`/`weighted_ce`이므로
그 기본값만 보고 현재 실험의 모델을 그리면 안 된다.

## 학습 보조 그림의 식

현재 `D_range` 구성은 다음과 같다.

\[
L_{\rm cell}=\tfrac12 L_F+\tfrac12 L_N+0.5L_B,\qquad
L=L_{\rm cell}+0.3L_{\rm range}.
\]

각 셀 항은 자신의 영역에서 평균한다. 경계 대역의 target은 ±δ로 절단한 Gaussian의
정규화 CDF이며 δ=0.30 m, σ=0.10 m다. δ는 반폭이다. 대역 밖은 hard target,
영구 사각 영역은 hard not-free target으로 감독한다. invalid 셀은 제외한다.

`V_θ`는 광선 θ의 유효 격자 표본이다. 보조항은 예측 free 확률과 GT free의
**광선 전체 누적 길이**를 같은 표본 연산자로 계산해 비교한다. 예측의 첫 장애물까지의 거리나
별도 거리 head의 출력이 아니다. 광선은 720개, 간격은 0.025 m이며, 유효한 GT 광선에
δ_R=0.15 m dead zone과 β=0.15 m Huber 항을 적용한다.
이 도식은 `D_range`에 해당하며, CE 대조군까지 같은 loss를 쓴다는 뜻은 아니다.

식의 근거: `projects/common/soft_boundary.py`, `projects/common/range_loss.py`,
`configs/loss_effect.sh`. 본 그림은 구조를 보여 주며 정확도·안전·재현성 개선 주장을 담지 않는다.

## 실제 예제와 검증

- 체크포인트: `runs/loss_effect/ckpt/D_range_s0/model_best-000000037.pth`.
- 프레임: 검증 시퀀스 `raws1/sample_000000`. 점수가 높은 사례를 검색하지 않고 첫 sample ID를 사용했다.
- 세 입력과 출력은 **동일 프레임**이다. `eval()`/`inference_mode()`로 기존 체크포인트를
  `strict=True` 로드했다. 라벨 valid 마스크 적용이나 형태학적 후처리 없는 raw argmax다.
- `bev-chamdog`, Python 3.11.15, torch 2.7.0+cu128에서 forward hook으로 형상을 확인했다.
  원본 입력은 `[1,3,3,288,512]`, lifting은 packed `[3,128,120,4,120]`,
  compressor 입력 `[1,512,120,120]`, 출력 logits `[1,2,120,120]`이었다.
- 생성기는 전체 페이지 밖으로 나가는 라벨을 검사한다. SVG·PNG를 시각 점검하고 PDF 글꼴 포함을 확인했다.

## 영문 캡션 초안

**Architecture figure.** Overview of the Simple-BEV-based fisheye free-space prediction network.
Three camera images are processed by a shared ResNet-101 encoder.
Using calibrated Double Sphere projection, image features are bilinearly sampled at four heights
(0, 0.5, 1.0, and 1.5 m above ground) and fused across cameras by masked averaging.
The height dimension is folded into the channel dimension and compressed into a 2D BEV feature map.
A ResNet-18-based U-Net with additive skip connections and a binary segmentation head predicts
free and not-free logits on a 120 × 120 grid at 5 cm resolution.
Narrowing and widening trapezoids denote encoding and decoding; stacked sheets depict intermediate
features, and horizontal grids depict the sampled height planes and the compressed BEV representation.
Blue denotes learned modules and features, while ochre denotes fixed geometric operations and fusion.
The camera images and output map show the same validation frame. Internal tensor dimensions are omitted
for clarity.

**Training figure.** Training objective for the soft-boundary model with radial free-length supervision.
Hard targets supervise cells outside the boundary band, while a truncated-Gaussian target
(δ = 0.30 m, σ = 0.10 m) supervises boundary cells. A radial auxiliary loss compares integrated
free-space probability with ground-truth free length using identical ray sampling and a dead-zone
Huber penalty (δ_R = β = 0.15 m). All supervision is applied during training; inference uses only
the image-to-BEV network and fixed camera calibration.

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/model_architecture.pdf}
  \caption{Overview of the Simple-BEV-based fisheye free-space prediction network.
  % 위의 architecture caption으로 교체한다.
  }
  \label{fig:architecture}
\end{figure*}
```
