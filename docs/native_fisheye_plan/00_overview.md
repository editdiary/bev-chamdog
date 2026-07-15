# Native Fisheye BEV Plan — 개요 (Overview)

## 0. 이 폴더의 위치와 목적

이 폴더는 이전 undistort 기반 계획(구 `docs/tentative_plans/`, 폐기)을 **대체**한다.
핵심 변경: **어안 이미지를 undistort하지 않고, Simple-BEV의 투영 함수를 radial_poly
어안 모델로 교체하여 어안을 그대로(native) 입력**한다.

> 이전 계획은 "일단 파이프라인을 돌리기 위해 어안을 핀홀로 펴서(undistort) 넣는다"였다.
> 그러나 이 프로젝트의 **원래 의도는 "어안 이미지로부터 모델이 직접 2D↔3D 관계를
> 학습하는 것"**이며, undistort는 (a) 어안의 넓은 화각을 버리고 (b) 자체 데이터셋 단계에서
> 버려질 전처리 스캐폴드를 만든다. 따라서 native 경로로 전환한다.

---

## 1. 전환의 근거 — 코드로 확인된 사실

Simple-BEV의 lifting은 `utils/vox.py`의 `unproject_image_to_mem()` **단일 함수**에서
일어난다. 동작 순서:

1. BEV 3D 격자점 생성 → ego(ref) 좌표 `xyz_camA`
2. **extrinsic 적용** → 카메라 좌표 `xyz_camB = apply_4x4(camB_T_camA, xyz_camA)`
3. **핀홀 투영** → 픽셀 좌표
4. `F.grid_sample`으로 해당 픽셀의 이미지 특징 샘플링

이 중 "핀홀"을 가정하는 부분은 **정확히 아래 두 줄뿐**이다:

```python
xyz_pixB = utils.geom.apply_4x4(pixB_T_camA, xyz_camA)          # K 행렬(선형) 곱
xy_pixB  = xyz_pixB[:,:,:2] / torch.clamp(normalizer, min=EPS)  # perspective divide
```

나머지(extrinsic, 격자 생성, grid_sample, valid 마스크)는 투영 모델과 무관하다.
따라서 native 어안 전환은 위 두 줄을 아래로 교체하는 **국소 변경**이다:

```python
xyz_camB = apply_4x4(camB_T_camA, xyz_camA)          # ego→카메라 (이미 계산되어 있음)
uv = radial_poly_project(xyz_camB, fisheye_calib)     # ← Phase 1이 검증할 project()
# valid: z>0 대신 θ<θ_max(화각) 체크로 보강
```

**핵심 함의:**
- lifting은 **격자→픽셀 정방향(project)** 만 사용한다. radial_poly의 *쉬운 방향*이다.
  (어려운 역방향 unproject는 lifting에 불필요.)
- 이 `project()`는 어차피 Phase 1에서 만들고 검증할 함수 → native의 추가 비용이 거의 없다.
- 참고: LSS/BEVDepth는 픽셀을 depth 프러스텀으로 forward-lift 하므로 픽셀별 역투영 + 어안
  ray 위 depth 재해석이 필요해 native가 더 지저분하다. **Simple-BEV가 native 어안을 배우기
  가장 쉬운 모델**이며, 여기서 지금 확립해 두는 것이 모델 로드맵상 유리하다.

---

## 2. 이전(undistort) 계획 대비 유지 / 폐기

| 항목 | 상태 | 비고 |
|---|---|---|
| radial_poly `project()`/`unproject()` 구현·검증 (구 Layer A) | **유지** | native의 핵심. reproj<1px, depth vs LiDAR |
| undistort remap 테이블·핀홀 뷰 정의·뷰당 K/extrinsic 합성 (구 Layer B) | **폐기** | 버려질 스캐폴드였음 |
| Simple-BEV `unproject_image_to_mem` 투영 교체 | **신규** | 두 줄 교체 + 교체 검증 |
| 세 좌표계 정렬 검증 (Phase 2 검증 A/B) | **유지** | "핀홀 뷰" → "어안 원본+native 투영"으로 문구만 변경 |
| 학습 시 loss/클래스 불균형/overfit check/mIoU (Phase 3) | **유지** | 모델 무관 |
| 입력 정규화 / pretrain 초기화 | **유지** | native에서도 필수 |
| undistort 빈 영역(검은 테두리) 처리 | **폐기** | native는 대신 화각(θ) valid 마스크 |

---

## 3. 프로젝트 전제 (변동 없음)

- **최종 목표:** 자체 온실(greenhouse) 어안 4-cam 데이터셋으로 BEV 모델 학습, 로봇이
  perception만으로 근거리 주행.
- **Task:** BEV semantic segmentation, RGB-only. **v0 = 주행성(traversability) 중심 소수
  클래스**(`drivable` / `occupied`, 필요 시 `caution`; 독립 채널·멀티라벨). 세밀한 semantic
  구분과 3D detection(사람 등 이산 개체)은 **확장 task로 분리**. sim은 세분 학습 후 coarse
  remap(계층적 라벨).
- **모델 로드맵:** **Simple-BEV(파이프라인 증명·native 어안 확립) → LSS(depth 감독으로
  정확도) → BEVFormer(최고 정확도).**
- **프레임워크:** **v0는 standalone Simple-BEV(순수 PyTorch), mmdet3d 미사용.** mmdet3d는
  detection/multi-task(BEVFormer 등) 단계에서 등판(§5 프레임워크 결정 참조).
- **데이터:** SynWoodScape V0.1.0 (어안 4-cam, 500 samples, BEV GT 완비)로 pretrain →
  자체 실데이터로 fine-tuning.
- **하드웨어:** RTX 3080 10GB 단일 GPU.

---

## 4. Phase 구성

- **Phase 1** (`phase1_projection_verification_plan.md`): radial_poly 투영 함수 검증
  + Simple-BEV lifting에 native 투영 통합·검증.
- **Phase 2** (`phase2_alignment_verification_plan.md`): 학습 직전 세 좌표계
  (어안 이미지 ↔ ego 3D ↔ BEV 격자) 정렬 검증.
- **Phase 3** (`phase3_training_plan.md`): Simple-BEV native 어안 학습 실무.

원칙: **위험을 순서대로 격리한다.** 각 Phase는 한 종류의 실패만 검사하여, 나중에
"성능이 안 나오는데 원인 불명"을 방지한다.

---

## 5. 지금까지 확정 / 아직 논의 필요

**확정:**
- native 어안 직접 입력. undistort 폐기.
- project/unproject 검증은 유지(어차피 필요).
- 첫 모델 Simple-BEV, task=BEV seg(drivable/occupied), 데이터=SynWoodScape.

**논의 완료:**
- **백본이 왜곡 이미지에서 특징 추출** → **ResNet-50 baseline 확정, distortion-aware 불필요**
  (근거: 기하는 투영이, 외형만 백본이 담당 → 분리. distortion-aware는 attention 샘플링
  전용 이슈. Phase 3 §3.4). feature-map stride 스케일링은 투영 통합에 반영(Phase 1 §5.1).
- **task 범위** → **v0 = 주행성 중심 소수 클래스 BEV seg 확정.** 세밀 semantic·3D OD는
  확장. multi-task(BEV seg + 3D OD)는 **공유 인코더 + head 추가**의 additive 구조라 seg-first가
  막지 않음. sim은 box_3d 라벨 제공(공짜) → **mmdet3d 단계(LSS/BEVFormer)에서 detection head
  저비용 타진 → 자체데이터 3D OD 라벨은 downstream 필요 확인 후 결정.** 사람 등 이산 개체는
  seg 클래스가 아니라 이 detection head의 몫(seg는 인스턴스/방향 없음). Phase 3 §10.

- **카메라 모델** → **정리됨** (Phase 1 §2.3): sim=radial_poly 유지(변환 없음, WoodScape
  omnidet에 참고 코드), real 자체 캘리는 **표준 툴로 Double Sphere(1순위)·eUCM(후보)**,
  최종은 재투영 오차로 확정. sim↔real 투영 모델 통일 불필요. 개념 학습은
  `docs/study/camera_models_and_calibration.md`.
- **프레임워크(mmdet3d)** → **v0는 standalone Simple-BEV(순수 PyTorch), mmdet3d 미사용 확정.**
  근거: mmdet3d의 가치는 "표준 벤치마크 위 3D detection"(detection head/3D NMS/NDS 평가/
  model zoo)에 집중 → v0는 **seg + 자체데이터**라 그 가치가 대부분 안 걸림. Simple-BEV/LSS가
  mmdet3d를 안 쓰는 것도 같은 이유(seg 연구 코드). Simple-BEV 학습 = 평범한 PyTorch 루프
  (`train_nuscenes.py`: Dataset→DataLoader→Segnet→AdamW/OneCycleLR→loop; deps=torch/
  torchvision/efficientnet_pytorch/nuscenes-devkit/tensorboard, **mm 계열 불필요**). 학습
  루프가 노출돼 있어 파이프라인 학습·디버깅에 오히려 유리. 어안은 부차적 이유(mmdet3d
  파이프라인이 핀홀 가정을 깊게 내장 → native 주입이 더 번거로움). **mmdet3d는
  detection/multi-task(LSS/BEVFormer) 단계에서 등판**(Phase 0 세팅은 그때 자산). Phase 3 §7.

**다음에 깊게 논의할 열린 항목:**
1. **Sim-to-real 도메인 갭** — SynWoodScape(CARLA 도시 주행) → 온실. 전이가 실제로
   도움 되는지 검증 필요.
2. **화각(θ) valid 마스크 설계** — θ>90° 영역, behind-camera 처리, θ_max 결정.
3. **radial_poly 역함수 수치 안정성** — 넓은 θ에서 다항식 단조성(unproject는 검증·GT
   대조용으로 필요).
4. **자체 데이터 BEV GT 생성 방법** — 최종 병목. SLAM 맵 + 수동 라벨 vs LiDAR 점유 반자동.
5. **좌표계 규약** — CARLA reference 축, quaternion 순서(w-first/last), ego 원점.
6. **BEV 격자 파라미터** — Simple-BEV `Vox_util`(scene_centroid, bounds, Z/Y/X)를
   SynWoodScape BEV GT 커버 범위와 일치.
