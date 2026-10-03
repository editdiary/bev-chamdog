# 공통 설정 — 데이터·라벨·모델·환경

논문 최종 실험(`paper_final`) **전체가 공유하는 사항**이다. 실험별 문서는 여기를 가리키고
같은 내용을 반복하지 않는다. 실험 고유의 split·조건은 각 실험의 `REPORT.md`에 있다.

---

## 1. 과제와 출력

전방 주시 카메라 3대(`front`·`left`·`right`)로 로봇 주변의 **BEV(bird's-eye view) free
space**, 즉 위에서 내려다본 격자에서 "지금 갈 수 있는 칸"을 예측한다.

| 축 | 값 |
|---|---|
| 정식화 | binary `free` / `not-free` |
| 격자 | 120 × 120, 한 칸 **5 cm** |
| 범위 | 전방 4 m · 후방 2 m · 좌우 ±3 m |
| 입력 | 3-camera, 512 × 288, **Double Sphere 원본 투영**(undistort 하지 않음) |

격자 5 cm는 고정이다. 이 값은 다른 값과 바꿔 비교하는 축이 아니다.

## 2. 시퀀스 7개와 2×2 요인 구조

온실에서 촬영한 7개 시퀀스다. 조명(햇빛 차단막 유무)과 통로 폭이 요인이다.

| 시퀀스 | 조명 | 통로폭 | 프레임 |
|---|---|---|---|
| `raws1` | 햇빛 | 좁음 | 38 |
| `raws2` | 햇빛 | 좁음 | 41 |
| `raws3` | 햇빛 | **넓음** | 36 |
| `rawos1` | 가림막 | 좁음 | 39 |
| `rawos2` | 가림막 | 좁음 | 40 |
| `rawos3` | 가림막 | 넓음 | 37 |
| `rawos4` | 가림막 | 넓음 | 36 |
| | | | **합 267** |

**`raws3`가 (햇빛 × 넓은 통로) 칸의 유일한 시퀀스다.** 그래서 `raws3`를 빼고 학습하면 그
조합이 train에 아예 없다 — LOSO에서 이 fold만 **외삽**으로 따로 표시한다.

전체 267 프레임은 작은 데이터다. 논문에서 먼저 밝히는 편이 낫다.

## 3. 라벨 출처 — 반드시 본문에 명시할 것

라벨은 **LiDAR/LIO-SLAM으로 만든 뒤 사람이 수동 보정**한 것이다. 보정 기준은 저장소에
기록돼 있지 않다.

따라서 **절대적인 경계 오차는 측정할 수 없다.** BF@{0.10, 0.20, 0.30} m 같은 경계 지표는 전부
"라벨과의 일치도"이지 물리적 정확도가 아니다. 논문에서 그렇게 쓴다.

같은 이유로 **시퀀스별 라벨 품질 차이를 분리할 수 없다.** fold 간 성능 차이를 해석할 때
(i) 실제 장면 난이도, (ii) 표본 오차, (iii) 라벨 품질 셋이 섞여 있고 분리 불가능하다.

## 4. 동결 모델 설정

캠페인 v3(`runs/99_full_campaign`)의 모든 실험이 이 설정을 공유한다(실험 03은 loss 축만 바꾼다). 실험이 바꾸는 축만 각 `REPORT.md`에 적는다.

| 축 | 값 |
|---|---|
| 모델 | Simple-BEV, ResNet-101 encoder, stride 8, latent dim 128 |
| 초기화 | ImageNet encoder + 랜덤 BEV decoder (BEV 사전학습 없음) |
| lifting | `Y=4`, 높이 −0.25~1.75 m (표본 0 / 0.5 / 1.0 / 1.5 m) |
| 특징 표본 | `pixel_center`, offset 0 |
| loss | soft-boundary, gaussian target |
| boundary | `δ=0.30`, `σ=0.10`, `λ_B=0.5`, `κ=1.0`, `label_eps=0.0` |
| range 보조항 | **없음** (`λ_R=0`, 2026-10-02 사용자 결정). `config.json`에 남은 `range_loss_mode`는 쓰이지 않는다 |
| 최적화 | AdamW, lr 1e-4, weight decay 1e-7, batch 8, **100 epochs** (사전학습도 100) |
| 증강 | photometric on, flip off |
| 주 체크포인트 | **고정 epoch 100 = 마지막** (validation-best는 진단용으로만 병기) |
| 학습 시드 | 0, 1, 2, 3, 4 |

**용어 주의.** 논문에서 이 설정을 `scratch`로 줄여 부를 경우, 실제 의미는 완전 무작위
초기화가 아니라 **target-only training without BEV pretraining**이다. 그렇게 명시한다.

**고정 epoch을 쓰는 이유.** validation으로 epoch을 고르면 평가 대상에 맞춰 고르는
선택 편향이 생긴다. LOSO에서는 특히 그렇다(held-out 시퀀스가 곧 평가 대상이다).

## 5. 실행 환경

| | |
|---|---|
| conda 환경 | `bev-chamdog` |
| Python | 3.11.15 |
| torch / CUDA | 2.7.0+cu128 / 12.8 |
| numpy | 1.26.4 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (GPU 0) |

실험별 정확한 Git 커밋과 해석된 전체 설정은 각 실험의
`provenance/environment_and_config.json`에 있다.

## 6. 무결성 검사 방식

모든 실험이 같은 게이트를 통과해야 논문 숫자로 쓴다.

1. 모든 런이 epoch 100까지 완주했는가
2. 런 이름·시드·split을 제외한 설정이 전부 같은가
3. lifting 기하(`height.json`)가 단일 값인가
4. **학습 로그와 별도로 저장한 확률맵을 독립 경로로 재채점했을 때 일치하는가**
   (허용치 1e-3, 실패 0건이어야 함)

4번이 핵심이다. 표에 쓰는 숫자와 저장된 예측이 같은 것을 가리킨다는 확인이고, 이게
깨지면 결과를 쓰지 않는다. 결과는 각 실험의 `provenance/integrity.json`에 있다.

---

## 7. 학습 절차 상세 (optimizer · schedule · 증강 · 정밀도 · 환경 · 시간)

**→ [`training_details.md`](training_details.md)** 에 있다.

논문의 "Training and Implementation Details"를 쓸 때 필요한 수준의 값 전부와,
**`config.json`에 기록돼 있지만 실제로는 동작하지 않는 값 목록**, 그리고 고쳐 쓸 수 있는
영문 초안이 거기 있다. 여기 §4는 "무엇을 동결했나"의 요약표이고, 그쪽이 상세 정본이다.
