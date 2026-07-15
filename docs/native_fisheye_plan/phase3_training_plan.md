# Phase 3: 모델 학습(training) Plan (native 어안 / Simple-BEV)

## 0. 문서의 목적과 위치

- Phase 1(`phase1_projection_verification_plan.md`): radial_poly 투영 함수 검증 +
  Simple-BEV lifting에 native 투영 통합·검증.
- Phase 2(`phase2_alignment_verification_plan.md`): 세 좌표계 정렬 검증.
- **Phase 3(본 문서):** 위 검증 통과 전제 하에, Simple-BEV로 **native 어안** 학습을 돌릴 때
  신경 쓸 모든 사항.

전제 상태:
- native 투영 통합 완료(어안 원본을 그대로 입력, undistort 없음).
- 데이터 로더가 Simple-BEV 형식으로 (**어안 RGB** + calibration/extrinsic + BEV GT) 공급.
- extrinsic·native 투영 검증(Phase 2 검증 A), 격자 정렬(검증 B) 완료.
- Task: BEV semantic segmentation, `(C,H,W)` 출력(멀티라벨), 초기 클래스 = drivable/occupied.
- **첫 모델: Simple-BEV**(독립 레포, `third_party/models/simple_bev`).

### 0.1 native라서 Phase 3에서 바뀌는 점 (undistort 계획 대비)
- 입력이 **어안 원본** → undistort 빈 영역(검은 테두리) 처리 불필요. 대신 **화각(θ) valid
  마스크**(Phase 1 §5.1)가 그 역할.
- 어안 특유의 **주변부 왜곡을 백본이 그대로 본다** → §3.4의 백본 이슈가 새로 중요.
- 나머지(loss·불균형·pretrain·정규화·overfit·mIoU)는 **모델·투영 무관하게 유지**.

### 0.2 발전 경로
**Simple-BEV(증명·native 확립) → LSS(depth 감독으로 정확도) → BEVFormer(최고 정확도).**

---

## 1. 모델 출력 / Head

### 1.1 head 교체가 아니라 "출력 채널 맞추기"
- Simple-BEV는 이미 BEV semantic 출력을 낸다. detection 파이프라인 제거 불필요.
- 할 일: **출력 채널 수를 클래스 정의(drivable/occupied)에 맞추기.**
  - 독립 채널(멀티라벨) → 각 채널 **sigmoid**(softmax 아님).
  - 최종 출력 `(C_class, H_out, W_out)`.

### 1.2 출력 해상도
- Simple-BEV BEV 출력 해상도가 Phase 2에서 정렬한 GT 해상도와 일치하는지 확인.
- 불일치 시 출력단 upsample 또는 GT를 출력 해상도로 리샘플(nearest, Phase 2 규칙 준수).

### 1.3 확인
- Simple-BEV(`nets/segnet.py`) 출력 텐서 shape·좌표 규약(어느 축이 +x/전방)이 Phase 2
  정렬과 일치. 기본이 단일 클래스면 다채널 확장 확인.

---

## 2. Loss function

### 2.1 기본: 채널별 독립 loss
- 멀티라벨 → 각 채널 sigmoid → **채널별 Binary Cross-Entropy.**
- 채널마다 따로 계산 → "sim은 전 채널 / 실데이터 fine-tuning은 obstacle 채널만" 전략을
  채널별 loss on/off로 이어감.

### 2.2 핵심 문제: 클래스 불균형
- occupied(장애물)는 BEV 평면의 작은 일부. 대부분 셀은 background/drivable.
- 순수 BCE는 "전부 background" 예측으로도 loss가 낮아 장애물을 안 잡는 방향 수렴 위험.
- 대응: **Dice loss**(작은 전경에 강함), **Focal loss**(쉬운 배경 down). 권장 **BCE+Dice**,
  필요 시 Focal. **채널별 가중치**로 obstacle 채널 상향.
- ★ Simple-BEV 기본 loss 먼저 확인, 불균형 대응이 약하면 보강.

### 2.3 확인
- ignore 영역(unlabeled/sky/ego-vehicle) loss 제외(mask).
- loss 채널별 분리 로깅.

---

## 3. 입력 처리 / 초기화 (조용히 성능 좌우)

### 3.1 이미지 정규화 일치 (필수)
- Simple-BEV 백본도 특정 정규화(예: ImageNet mean/std) pretrain 가중치 사용.
- **어안 원본도 동일 정규화**를 거쳐야 pretrain 가중치가 의미. 레포 기본 정규화 확인·정합.

### 3.2 Pretrain 초기화 (500장 규모라 사실상 필수)
- 500장은 백본 scratch 학습에 매우 적음.
- 백본: ImageNet(또는 Simple-BEV 제공 nuScenes) pretrain 로드. 출력 채널 바꾼 head/마지막
  층만 새 초기화. shape mismatch 레이어 strict=False.

### 3.3 화각(θ) valid 마스크
- native 투영에서 화각 밖·behind-camera 격자점은 특징 샘플링 제외(Phase 1 §5.1).
- undistort의 "검은 테두리 crop"을 대체. 유효 특징만 BEV에 축적되게 함.

### 3.4 백본 선택 — ResNet baseline 확정 (distortion-aware 불필요)

**결론: 백본은 손대지 않는다. Simple-BEV 기본 ResNet을 그대로 쓴다.**

근거 — **기하와 외형의 분리:** native 경로에서 어안 왜곡의 *기하학적* 처리는 백본이 아니라
**투영(`project()`)** 이 담당한다. 백본은 "이 픽셀이 벽/바닥인가"의 **외형 특징**만 뽑고,
"그 특징이 3D 어디로 가는가"는 radial_poly 투영이 정확히 처리한다. 따라서 백본은 왜곡을
기하학적으로 이해할 필요가 없다. 보조 근거: (a) conv 국소성 — 어안 왜곡은 국소적으로는
완만한 스케일/warp라 augmentation 범위 내, (b) 왜곡이 카메라별 고정이라 학습 가능,
(c) ImageNet 저수준 특징(엣지·텍스처)은 왜곡 무관 + 고수준은 fine-tuning으로 적응.

**distortion-aware가 필요한 경우 = attention 계열뿐.** BEVFormer류의 spatial cross /
deformable attention은 이미지 위 reference point를 샘플링하는데, 그 오프셋이 직선 이미지를
가정해 학습되어 어안의 비균일 픽셀 밀도에서 **샘플링 패턴이 왜곡**된다 → 그래서
"distortion-aware spatial cross attention"으로 보정한다. **CNN(conv)은 고정 3×3 격자라 이
문제가 없다.** 즉 distortion-aware는 백본 일반의 문제가 아니라 **attention 샘플링 기하의
문제**다. → ResNet baseline에는 아무것도 추가하지 않는다.

**baseline = ResNet-50 (`Encoder_res50`, pretrained).** Simple-BEV 기본은 res101이나,
**RTX 3080 10GB + 어안 4-cam**에서 res101은 OOM 위험이 크다. res50으로 파이프라인을
증명하고, VRAM 여유·정확도 필요 시 res101로 상향. (§3.1의 feature-map stride 스케일링은
투영 통합에서 반드시 반영 — Phase 1 §5.1.)

**남는 리스크(백본 종류와 무관):** 어안 주변부는 스케일 변화가 극심 + sim-to-real이 겹쳐
**주변부 mIoU가 중심부보다 약할 수 있음.** 그러나 이는 백본 문제가 아니라 데이터/도메인
문제 → 백본 교체가 아니라 sim-to-real·4-cam 커버리지로 접근(§10). fisheye-aware conv 등은
주변부 mIoU가 실제 병목으로 확인될 때만 고급 옵션으로 검토.

---

## 4. Augmentation (정렬을 깨뜨릴 위험)

### 4.1 경고
BEV seg에서 flip/rotate augmentation은 **입력 이미지, extrinsic, BEV GT를 동시에 같은
변환으로** 바꿔야 한다. 이미지만 flip하고 GT/extrinsic을 안 바꾸면 Phase 2 정렬이 매 배치
깨진다. Simple-BEV는 투영에 extrinsic을 직접 쓰므로 더 치명적.

### 4.2 권장 순서
- **초기: augmentation 최소화(또는 off).** 파이프라인 정확성 먼저 검증.
- 이후 정렬 유지 augmentation만 단계적 추가. 추가 시마다 Phase 2 시각화로 정렬 유지 확인.

---

## 5. 데이터 규모 (500장) — 사실상 최대 제약

### 5.1 과적합
- 500장은 적음. **train/val split 필수**, val mIoU로 과적합 감시.
- **scene 단위 분리**로 leakage(연속 프레임이 train/val 혼입) 방지.

### 5.2 Overfit sanity check (강력한 디버깅)
- 5~10장에 **일부러 과적합** → loss≈0, 예측이 GT 재현되는지 확인.
- 재현 실패 → 학습 능력 이전 **파이프라인 버그**(GT 연결/좌표계/loss/출력 shape/native 투영).
- **정식 학습 전 반드시 통과.**

### 5.3 기대치
- 500장 모델은 목적 자체가 아님 → **파이프라인 검증 + 실데이터 fine-tuning용 pretrain 시드.**
- SOTA mIoU보다 "물리적으로 옳게 작동 + 수렴 + 시각적으로 그럴듯한 예측"이 목표.

---

## 6. 평가 / 모니터링

### 6.1 지표: mIoU
- **채널별 IoU의 평균.** Simple-BEV 기본 평가(IOU)를 다채널로 확장.

### 6.2 채널별 IoU 분리
- 평균만 보면 "drivable은 잘 되는데 obstacle은 바닥"이 가려짐. obstacle이 핵심 →
  채널별 분리 리포트.

### 6.3 예측 시각화 (학습 중 주기적)
- loss 곡선만 보지 말고 val 예측 BEV 맵을 주기적으로 저장. seg는 시각화가 정량지표보다
  문제를 빨리 드러냄(예: 특정 방향 전체가 빔 → 정렬/뷰 문제).
- ★ Simple-BEV `vis_nuscenes.py`/`vis_collage.py` 등 시각화 스크립트 활용.

---

## 7. 실무 함정 (환경/실행)

### 7.1 레포 의존성
- Simple-BEV는 특정 PyTorch/CUDA를 요구할 수 있음. nuScenes devkit 의존(데이터 로딩부)
  가능. 착수 시 README 확인.

### 7.2 데이터 로더 연결 (주 작업)
- mmdet3d info.pkl 대신 **Simple-BEV dataset 인터페이스에 SynWoodScape 연결**.
  - (a) Simple-BEV dataset 클래스를 상속/수정해 SynWoodScape(**어안 RGB** + calibration/
    extrinsic + BEV GT) 반환. **더 직관적, 권장.**
  - (b) SynWoodScape를 nuScenes 형식으로 감싸는 어댑터.
- Simple-BEV가 forward에 넘기는 dict/텐서(특히 intrinsic/extrinsic 표현) 파악 후 (a) 권장.
  - **주의:** Simple-BEV forward는 `pix_T_cams`(핀홀 K 4×4)를 기대할 수 있음. native에서는
    이 자리를 radial_poly calib으로 대체하거나, 투영 함수가 calib을 직접 받도록 인터페이스
    조정 필요(Phase 1 §5.1 통합과 연동).

### 7.3 Temporal
- Simple-BEV 기본 단일 프레임 → 초기 그대로. 추후 temporal 확장 시 `_prev` + ego motion
  (`vehicle_data`) 필요.

---

## 8. 학습 착수 체크리스트 (순서대로)

- [ ] **8.1** Simple-BEV 레포 의존성/데이터 형식/격자 파라미터/기본 loss·평가 파악.
- [ ] **8.2** 데이터 로더 연결(7.2 (a)): SynWoodScape(어안) → Simple-BEV 형식. calib 전달
      경로가 Phase 1 native 투영 통합과 일치하는지 확인.
- [ ] **8.3** 출력 채널을 drivable/occupied에 맞춤(sigmoid, 다채널). 출력 해상도 = GT 정렬 해상도.
- [ ] **8.4** loss: 채널별 BCE+Dice(+옵션 Focal), obstacle 가중, ignore mask, 채널별 로깅.
- [ ] **8.5** 이미지 정규화를 pretrain과 일치. θ valid 마스크 동작 확인.
- [ ] **8.6** pretrain 가중치 로드(백본), 출력층 새 초기화, strict=False.
- [ ] **8.7** augmentation 최소화/off로 시작.
- [ ] **8.8** train/val split(scene 단위).
- [ ] **8.9** **overfit sanity check (5~10장)** — loss≈0, 예측=GT. 실패 시 파이프라인 디버깅.
- [ ] **8.10** 평가를 채널별 mIoU로 확장, 예측 시각화 준비.
- [ ] **8.11** 소수 샘플 + 1 GPU + 단일 프레임으로 end-to-end 1 epoch 검증.
- [ ] **8.12** 전체 학습(train split), val mIoU·채널별 IoU·예측 시각화 모니터링.
- [ ] **8.13** (안정화 후) 정렬 유지 augmentation 등 단계적 고도화.

---

## 9. 우선순위 요약

Simple-BEV라서 줄어든 것:
1. head 교체 → **출력 채널 맞추기**로 축소.
2. temporal / mmdet3d config / depth 결정 → **없음.**
3. undistort 스캐폴드(remap/핀홀 뷰) → **없음**(native).

그래도 조용히 성능을 갉는 것:
4. **pretrain 초기화 + 이미지 정규화 일치**(500장 규모라 사실상 필수).
5. **백본의 어안 왜곡 특징 추출**(native 신규 이슈, §3.4).
6. **augmentation 좌표 동기화**(안 하면 정렬 매 배치 깨짐 → 초기 off).
7. **overfit check → train/val split → 채널별 mIoU + 예측 시각화.**

새로 생기는 주 작업:
8. **Simple-BEV dataset 인터페이스에 SynWoodScape(어안) 연결 + native calib 전달.**

---

## 10. 미해결 / 추후 결정 (다음 심화 논의 대상)

- ~~백본의 어안 왜곡 특징 추출 대응~~ → **해결: ResNet-50 baseline, distortion-aware 불필요
  (§3.4).** 주변부 mIoU가 병목으로 확인되면 그때 고급 옵션 검토.
- **Sim-to-real 도메인 갭**(SynWoodScape 도시 → 온실) 전이 유효성 검증 방법.
- Simple-BEV forward의 intrinsic/extrinsic 표현과 native calib 전달 인터페이스 상세.
- Simple-BEV 기본 loss·평가 로직, 격자 파라미터 기본값.
- BCE:Dice:Focal 가중, obstacle 채널 가중치 — 실험 튜닝.
- pretrain 소스(ImageNet vs Simple-BEV 제공 nuScenes).
- 500장 train/val 비율 및 scene 단위 split 기준.
- **(확장 task) multi-task = BEV seg + 3D OD:** 공유 BEV 인코더 + seg head + **detection
  head(CenterPoint식)** 의 additive 구조. seg-first가 막지 않음. sim(SynWoodScape)은
  `box_3d_annotations` 제공(라벨 공짜) → **standalone Simple-BEV에 억지 이식 금지**, mmdet3d
  기반(LSS/BEVFormer) 단계에서 detection head 네이티브 지원 활용해 sim으로 먼저 타진. 자체
  데이터 3D OD 라벨(인스턴스별 방향성 3D 박스)은 비싸므로 **downstream 필요(예: 사람 안전
  탐지) 확인 + 소수 이산 클래스 한정** 후 결정. (사람은 seg 클래스가 아니라 이 head의 몫.)
- (발전 경로) LSS 전환 시 depth 감독 + native 어안 프러스텀 재해석 — 별도 Phase.
- (발전 경로) 실데이터 fine-tuning 시 채널별 loss on/off + forgetting 방지 — 별도 Phase.
- (최종 병목) 자체 온실 데이터 BEV GT 생성 방법 — SLAM 맵 + 수동 라벨 vs LiDAR 점유 반자동.
