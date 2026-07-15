# 카메라 모델과 캘리브레이션 — 학습 가이드

> 목적: 어안 4-cam BEV 프로젝트를 이해하는 데 필요한 **카메라 투영 모델**과
> **캘리브레이션**의 큰 개념을 원리 중심으로 정리하고, 더 깊이 파고들 때 무엇을 검색/독서
> 하면 되는지 안내한다. 공식 유도는 최소화하고 **직관 + 레퍼런스**에 집중한다.

---

## 0. 이 문서를 읽는 순서

1. §1 왜 카메라 "모델"이 필요한가 (핀홀의 한계)
2. §2 투영의 3요소: extrinsic / intrinsic / distortion
3. §3 어안 투영 모델 지도 (KB, UCM, eUCM, DS, radial_poly …)
4. §4 캘리브레이션이란 무엇인가 (파라미터를 어떻게 구하나)
5. §5 도구(툴) 지도
6. §6 이 프로젝트와의 연결
7. §7 검색 가이드 & 읽을거리 (가장 중요 — 여기서 깊게 파기)
8. §8 용어집

---

## 1. 왜 카메라 "모델"이 필요한가

카메라는 **3D 세계의 점을 2D 이미지 픽셀로 대응**시키는 장치다. 이 대응 규칙을 수식으로
쓴 것이 "카메라 모델(projection model)"이다. 모델이 있어야:
- (정방향, project) 3D 점이 이미지의 **어느 픽셀**에 찍히는지 계산하고,
- (역방향, unproject) 이미지 픽셀이 3D의 **어느 방향(광선)**에서 왔는지 복원한다.

### 1.1 핀홀(pinhole)과 그 한계
가장 단순한 모델. 3D 점 (X,Y,Z)를 `u = fx·X/Z + cx`, `v = fy·Y/Z + cy`로 투영(원근분할).
- 특징: **직선은 이미지에서도 직선**(rectilinear).
- 한계: 화각(FOV)이 넓어지면 Z/tan 때문에 발산 → **~120° 이상은 사실상 표현 불가.**
- 어안(≈180~200°)은 핀홀로 못 담는다. → 어안 전용 모델이 필요한 이유.

### 1.2 어안의 핵심 아이디어: "각도 → 반경"
어안 모델의 공통 직관: 광선이 광축과 이루는 **입사각 θ**가 커질수록, 이미지 중심에서의
**반경 r**이 커진다. 즉 `r = f(θ)`. 이 `f`를 어떻게 정하느냐가 모델의 차이다.
- 핀홀: r = f·tan(θ) (θ→90°에서 발산)
- 어안류: r을 θ에 대해 **완만하게** 두어 넓은 θ를 유한한 r에 담는다.

전통적 어안 "투영 방식" 4종(렌즈 설계 기준, 검색어로 유용):
- **equidistant** r = f·θ, **equisolid** r = 2f·sin(θ/2),
- **stereographic** r = 2f·tan(θ/2), **orthographic** r = f·sin(θ).
실제 렌즈는 이 이상형과 조금씩 다르므로, 아래 §3의 **파라미터화된 모델**로 근사·보정한다.

---

## 2. 투영의 3요소

어떤 카메라 모델이든 결국 세 부분의 합성이다:

1. **Extrinsic (외부 파라미터):** 카메라의 **위치·자세**(회전 R + 이동 t). 월드/ego 좌표를
   카메라 좌표로 바꾼다. 6 자유도. (멀티카메라·카메라-LiDAR 정합의 핵심.)
2. **Intrinsic (내부 파라미터):** 렌즈·센서 고유 성질. 초점거리(fx,fy), 주점(cx,cy), 그리고
   **왜곡/투영 계수**. 카메라 좌표 3D → 픽셀.
3. **Distortion / projection form:** intrinsic 안에서 특히 "r=f(θ)"를 어떤 함수족으로
   두느냐. 핀홀+방사왜곡(radtan)인지, 어안 다항식인지, 구 기반인지 등. **§3이 이 부분.**

> 캘리브레이션(§4)은 이 세 요소의 **숫자(파라미터)를 데이터로 추정**하는 일이다.

---

## 3. 어안 투영 모델 지도

크게 **(A) 각도-다항식 계열**과 **(B) 구(sphere) 기반 계열**로 나뉜다.

### 3.A 각도-다항식 계열 — "r = 다항식(θ)"
- **Kannala–Brandt (KB, "equidistant polynomial")**
  - r(θ) = θ(1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸) — **홀수 차수**.
  - **OpenCV `cv2.fisheye`가 바로 이 모델.** 가장 널리 쓰임. 표준 지원 풍부.
  - 역투영은 다항식 역함수(수치해) 필요.
- **WoodScape radial_poly (본 프로젝트 sim 데이터)**
  - r(θ) = k₁θ + k₂θ² + k₃θ³ + k₄θ⁴ — **연속 차수(짝수항 포함), order 4.**
  - KB와 같은 가족(θ→r 다항식). WoodScape 저자들이 자기 렌즈에 더 잘 맞아 채택.
  - **표준 툴은 이 형태를 직접 지원하지 않음.** 단, 투영 함수 구현은 쉬움(다항식+LUT 역함수).
    참고 코드: WoodScape omnidet `data/generate_luts.py`.
- **Scaramuzza / OcamCalib (Taylor 모델)**
  - 반경↔각도를 Taylor 다항식으로. MATLAB **OcamCalib** 툴박스로 유명(로보틱스에서 인기).

### 3.B 구(sphere) 기반 계열 — "3D를 구에 투영 후 핀홀"
- **UCM (Unified Camera Model, = Mei 모델)**
  - 3D 점 → 단위 구 투영 → 축 방향으로 **ξ(xi)** 만큼 옮긴 점에서 핀홀 투영.
  - 핀홀 + **파라미터 1개(ξ)**. 간단하지만 아주 강한 어안엔 정확도 한계.
- **eUCM (Enhanced UCM)**
  - 구 대신 **타원체**(파라미터 α, β) → 강어안 fit 개선. UCM의 상위호환.
- **Double Sphere (DS)**
  - 구를 **두 개** 사용(파라미터 ξ, α). 강어안에 정확하면서 **역투영이 닫힌 형태
    (closed-form)** → 계산 빠르고 안정적. 실무에서 인기 급상승.
- **FOV 모델**
  - 단일 파라미터 ω로 r=f(θ) 근사. 단순, 옛날 방식.

### 3.C 요약 비교 (직관)

| 모델 | 파라미터 수 | 강어안 정확도 | 역투영 | 표준 툴 |
|---|---|---|---|---|
| 핀홀(+radtan) | 적음 | ✗(넓은 FOV 불가) | 쉬움 | 모두 |
| KB (equidistant) | 4~ | 좋음 | 수치해 | OpenCV/Kalibr/Basalt |
| radial_poly (WoodScape) | 4 | 좋음 | 수치해(LUT) | **미지원**(직접 구현) |
| UCM | 1(+핀홀) | 보통 | 닫힌형 | Kalibr/OpenCV omnidir |
| eUCM | 2(+핀홀) | 좋음 | 닫힌형 | Kalibr/Basalt |
| **Double Sphere** | 2(+핀홀) | **좋음** | **닫힌형** | Kalibr/Basalt |

> **핵심 통찰:** 이들은 "같은 렌즈를 다르게 근사"하는 경쟁 모델이다. 어느 하나가 절대적으로
> 옳은 게 아니라, **내 렌즈에 재투영 오차가 작고 + 툴 지원이 되고 + 역투영이 편한** 것을
> 고르면 된다. (Double Sphere 논문이 이들을 한자리에서 비교하므로 최고의 출발점 — §7.)

---

## 4. 캘리브레이션이란 무엇인가

**캘리브레이션 = 위 모델의 파라미터(intrinsic/extrinsic/distortion) 숫자를 데이터로
추정하는 것.**

### 4.1 원리 (개념)
1. **알려진 기하의 타깃**을 여러 각도에서 촬영. (체커보드, 또는 AprilGrid/ChArUco.)
2. 타깃 위 점들의 **실제 3D 위치**는 안다(격자 간격). 이미지에서 그 점들의 **픽셀 위치**를
   검출한다.
3. 모델 파라미터를 미지수로 두고, "모델로 3D를 투영한 픽셀"과 "실제 검출된 픽셀"의 차이
   (**재투영 오차, reprojection error**)를 **최소화**한다. → 비선형 최적화(**bundle
   adjustment**, 보통 Levenberg–Marquardt).
4. 결과: 파라미터 + 평균 재투영 오차(px). **오차가 작을수록(예 <0.5px) 좋은 캘리.**

> 그래서 "어떤 모델이 내 렌즈에 맞나"는 **같은 데이터로 여러 모델을 캘리해 재투영 오차를
> 비교**하면 답이 나온다. (본 프로젝트: 자체 카메라를 DS·eUCM로 캘리 후 비교.)

### 4.2 무엇을 직접 안 해도 되나
- **비선형 최적화(BA) 자체를 구현할 필요 없음.** 툴이 해준다(§5).
- 우리가 직접 쓰는 것은 **project/unproject 함수**뿐(모델 forward/역). 이건 쉬움.

### 4.3 종류
- **Intrinsic 캘리:** 카메라 한 대의 내부 파라미터.
- **Extrinsic / 멀티카메라 캘리:** 카메라 간 상대 위치·자세(4-cam BEV에 필수).
- **카메라–IMU 캘리:** 시간 오프셋·상대자세(Kalibr의 대표 기능).
- **카메라–LiDAR 캘리:** 센서 융합·GT 생성 시 필요(추후 자체 데이터 단계).

---

## 5. 도구(툴) 지도

- **OpenCV**
  - `cv2.calibrateCamera` (핀홀+radtan), `cv2.fisheye` (KB), `cv2.omnidir` (contrib, Mei/UCM).
  - 접근성 최고, 파이썬. 어안이면 `cv2.fisheye`부터.
- **Kalibr** (로보틱스 표준)
  - 멀티카메라 + 카메라-IMU. 모델: pinhole-radtan, pinhole-equi(KB), omni(UCM), **eucm**,
    **ds(double sphere)**. AprilGrid 타깃.
- **Basalt**
  - UCM/eUCM/KB/DS 캘리 + VIO. Double Sphere 저자들 계열.
- **OcamCalib** (MATLAB)
  - Scaramuzza Taylor 모델. 로보틱스 전통.
- **COLMAP**
  - SfM. 캘리 전용은 아니나 카메라 모델 추정 포함.

---

## 6. 이 프로젝트와의 연결

- **sim (SynWoodScape):** `radial_poly` (파라미터 제공됨 → 캘리 불필요). project/unproject만
  구현. 참고: WoodScape omnidet `generate_luts.py`. → `native_fisheye_plan/phase1 §2`.
- **real (자체 4-cam, 추후):** 표준 툴로 캘리. 모델 **1순위 Double Sphere, 후보 eUCM**
  (강어안). 최종은 재투영 오차 비교로 확정. → `native_fisheye_plan/phase1 §2.3`.
- **sim↔real 모델 통일 불필요:** 투영은 카메라별 고정 기하 함수라 학습 가중치와 무관.
  각자 올바르면 pretrain→fine-tune 성립. **변환 안 함.**
- **좌표계 주의:** SynWoodScape extrinsic은 "CARLA reference" → 축 규약·quaternion 순서 확인
  필요(부호 오류 방지). → phase1 §2.2 / phase2.

---

## 7. 검색 가이드 & 읽을거리 (여기서 깊게 파기)

### 7.1 가장 먼저 (단일 최고 출발점)
- **논문: "The Double Sphere Camera Model" (Usenko, Demmel, Cremers, 3DV 2018).**
  - UCM/eUCM/KB/FOV/DS를 **한자리에서 비교**. 어안 모델 전반의 최고 입문. 검색:
    `"double sphere camera model" Usenko 3DV 2018 pdf`.

### 7.2 개별 모델 (검색어)
- `Kannala Brandt generic camera model fisheye 2006 PAMI`
- `Mei Rives unified camera model omnidirectional calibration ICRA 2007` (UCM)
- `enhanced unified camera model eUCM Khomutenko RA-L 2016`
- `Scaramuzza OCamCalib omnidirectional camera Taylor model`
- `fisheye projection equidistant equisolid stereographic orthographic` (렌즈 투영 방식 개념)
- `WoodScape ICCV 2019 fisheye dataset radial polynomial` (본 프로젝트 sim 데이터 원 논문)

### 7.3 캘리브레이션 실습 (검색어 / 문서)
- `OpenCV fisheye calibration tutorial python` (실습 1순위)
- `OpenCV omnidir calibration` (UCM/Mei)
- `Kalibr multiple camera IMU calibration wiki` (멀티캠·카메라-IMU)
- `AprilGrid vs checkerboard calibration target`
- `reprojection error camera calibration meaning good value`
- `bundle adjustment Levenberg-Marquardt intuition` (원리)

### 7.4 배경 이론 (교재)
- **Szeliski, "Computer Vision: Algorithms and Applications"** (무료 PDF) — 카메라 모델·캘리
  장. 입문~중급 최적.
- **Hartley & Zisserman, "Multiple View Geometry in Computer Vision"** — 사영기하 정석(심화).
- 검색: `Szeliski computer vision book pdf camera models`,
  `Hartley Zisserman multiple view geometry`.

### 7.5 좌표계·자세 표현 (부호 오류 방지)
- `quaternion wxyz vs xyzw convention` (성분 순서)
- `CARLA coordinate system left-handed x forward` (SynWoodScape extrinsic 규약)
- `camera coordinate vs world coordinate extrinsic R t`

---

## 8. 용어집 (빠른 참조)

- **project / unproject:** 3D→픽셀 / 픽셀→광선방향.
- **intrinsic / extrinsic:** 렌즈·센서 내부 / 카메라 위치·자세.
- **FOV:** 화각. 어안은 대략 180~200°.
- **rectilinear:** 직선이 직선으로 유지되는 투영(=핀홀). undistort의 목표 형태.
- **radtan (Brown–Conrady):** 핀홀용 방사+접선 왜곡 모델(k1,k2,p1,p2,…).
- **equidistant:** r=fθ. KB/OpenCV fisheye의 기반.
- **UCM/eUCM/DS:** 구·타원체·이중구 기반 어안 모델.
- **reprojection error:** 캘리 품질 지표(px). 작을수록 좋음.
- **bundle adjustment:** 재투영 오차를 최소화하는 비선형 최적화.
- **AprilGrid / ChArUco / checkerboard:** 캘리 타깃 종류.
