# WoodScape ICCV19 데이터셋 분석

> 목적: 자체 Fisheye 4-cam BEV 데이터셋을 설계하기 전에, 가장 유사한 공개 데이터셋인
> **WoodScape**(Valeo, ICCV 2019)가 실제로 어떻게 구축되어 있는지 — 폴더 구성, 파일명 규칙,
> 각 라벨 포맷 — 을 정리한다. 모든 내용은 로컬 데이터(`dataset/woodscape/`)와
> 공식 repo(`third_party/datasets/WoodScape/`)를 직접 열어 검증했다.

> 📌 **이 문서는 작성 당시(Phase 1) 기록이며, 프로젝트 task가 "BEV 3D 검출"이던 시점의 관점으로
> 쓰여 있다.** 이후 task는 **BEV occupancy map(drivable / non-drivable)** 으로 변경되었다(→ `ROADMAP.md`).
> 데이터셋 구조·포맷에 관한 사실 관계는 그대로 유효하나, "우리 dataset 함의"류의 서술은 3D 박스를
> 전제로 한 것이므로 현재 계획과 다르다. 현재 task 기준의 핵심 결론은 아래 §0의 첫 항목
> — **WoodScape 공개본은 2D 라벨만 제공하므로 학습에는 쓰지 않고, SynWoodScape를 쓴다** — 이다.

---

## 0. 한눈에 보기 (BEV 관점 결론부터)

- WoodScape 공개본은 **2D(이미지 평면) 라벨만** 제공한다: 2D 박스, semantic/instance/motion 세그멘테이션, dense polygon.
- **3D bounding box도, depth GT도 없다.** (자세한 이유는 §1)
- 따라서 우리가 WoodScape에서 **가져갈 것**은:
  1. **fisheye 캘리브레이션 모델·포맷** (4차 radial polynomial + quaternion/translation 외부파라미터)
  2. **폴더/파일명 구성 규칙** (`NNNNN_CAM`, per-frame 캘리브레이션)
  3. **2D/세그 라벨 포맷과 CAN(주행) 데이터 구조**
- 실제 **3D 박스 라벨 포맷**은 후속으로 분석할 **SynWoodScape**(합성 데이터, 3D 라벨 포함) 또는 nuScenes/KITTI를 참고해야 한다.

---

## 1. 논문의 "9 tasks" vs 실제 공개본 — 3D/depth 라벨은 왜 없나

원 논문(arXiv:1905.01489)은 WoodScape를 여러 task를 지원하는 데이터셋으로 소개하고, 여기에는 depth·3D box 같은 task도 언급된다. 그러나 이는 **네트워크(OmniDet)가 다루는 task 범위**이지, 모든 task의 GT가 배포된다는 뜻이 아니다.

공식 `README.md`는 명확히 적고 있다 — *"10K images with annotations for **7 tasks**"*. 실제 배포된 어노테이션은 **semantic seg · instance seg · motion seg · 2D box** 4종 + calibration + CAN(vehicle) + soiling 뿐이다.

두 task가 "없어 보이는" 이유:

| task | 상태 | 이유 |
|---|---|---|
| **Depth (거리 추정)** | GT 미배포 | WoodScape의 depth는 **self-supervised**. `previous_images` + `calibration`으로 프레임 간 움직임에서 깊이를 *학습*하므로 GT depth map 자체가 필요 없고 배포되지 않음. |
| **3D box detection** | 미배포 | GT 3D 박스는 README의 *"Coming Soon: Lidar and dGPS scenes"* 에 묶여 있었으나 끝내 미공개. LiDAR/dGPS 없이는 3D 박스 GT 생성 불가. |

추가로, GDPR/중국 데이터법 준수를 위해 원본의 **1/3(중국 촬영분)을 삭제**하고 나머지를 익명화했다(README 2021-03-05 업데이트). 이 때문에 공개본은 **샘플링된 부분집합**이다 (→ §3의 "왜 4장씩 안 묶이나"와 직결).

> **검증:** 6개 annotation zip 전체 목록을 확인했고, `depth`/`3d`/`lidar`/`dgps` 이름을 가진 파일은 데이터 트리 어디에도 없다.

---

## 2. 폴더 구조 (top-level)

루트: `dataset/woodscape/` (총 ~42 GB). 네이티브 이미지 크기 **1280×966** (soiling만 1280×960).

| 폴더 | 역할 | 내부 포맷 | 압축 여부 | 개수(train) | BEV 중요도 |
|---|---|---|---|---|---|
| `rgb_images/` | 입력 RGB (fisheye) | PNG | 압축 해제 (`rgb_images/rgb_images/`) | 8,234 (+test 1,766) | ★★★ |
| `previous_images/` | 각 프레임의 직전 프레임 (`_prev`) | PNG | 압축 해제 | 8,234 (+1,766) | ★★ (self-sup depth/motion용) |
| `box_2d_annotations/` | 2D 박스 (5 클래스) | `.txt` | 압축 해제 (`box_2d_annotations/`) | 8,234 | ★★ |
| `dense_polygon_annotations/` | 객체 dense 윤곽점 | `.txt` | 압축 해제 (`polygon_annotations/`) | 8,234 | ★ |
| `instance_annotations/` | **마스터** 인스턴스 폴리곤 (40+ 태그) | JSON | 압축 해제 (`instance_annotations/`) | 8,234 | ★★ |
| `semantic_annotations/` | semantic 세그 마스크 (10 클래스) | PNG (gt+rgb) | 압축 해제 (`semantic_annotations/`) | 8,235 | ★★ |
| `motion_annotations/` | motion(움직임) 세그 마스크 (19 클래스) | PNG (gt+rgb) | 압축 해제 (`motion_annotations/`) | 8,235 | ★ |
| `calibration_data/` | **프레임별** fisheye 캘리브레이션 | JSON | 압축 해제 (`calibration/`) | 8,235 (+1,766) | ★★★ |
| `vehicle_data/` | CAN/주행 데이터 (ego-motion) | JSON | 압축 해제 (`vehicle_info/`) | 8,234 | ★★ |
| `soiling_dataset/` | 렌즈 오염 세그 (4 클래스, 별도 트랙) | PNG | 압축 해제 | 4,000 (+test 1,000) | ✗ (무관) |
| `WoodScape License and Terms of Use.pdf` | 라이선스 | PDF | — | — | — |

- **모든 zip은 디스크에 압축 해제됨** (통일성 위해 전부 해제). 각 어노테이션 타입은 추출 결과
  **이중 폴더**(`<type>/<type>/...`) 형태다 — 예: `semantic_annotations/semantic_annotations/rgbLabels/`,
  `calibration_data/calibration/`. `vehicle_data`는 zip-in-zip이 풀려 `vehicle_info/{rgb_images,previous_images}/`.
- 각 타입 폴더에는 작은 메타 파일(`*_info.json` 또는 `readme.txt`)이 이중 폴더 밖(타입 폴더 바로 아래)에 함께 있다.
- 시각화 도구(`tools/woodscape_viz/`)는 이 **추출된 파일에서 직접** 읽는다 (§6).

---

## 3. 파일명 규칙 & "왜 4장씩 안 묶이나"

모든 파일은 **`NNNNN_CAM.ext`** 형식 (예: `02566_MVL.png`). `NNNNN`은 5자리 zero-pad id, `CAM`은 4개 카메라:

| 접미사 | 카메라 |
|---|---|
| `FV` | 전방 (Front View) |
| `RV` | 후방 (Rear View) |
| `MVL` | 좌측 미러 (Mirror View Left) |
| `MVR` | 우측 미러 (Mirror View Right) |

- `previous_images`는 확장자 앞에 `_prev` 추가 (`00000_FV_prev.png`).
- `soiling_dataset`만 예외적으로 **4자리** id (`0001_FV.png`).

**같은 `NNNNN`이 4개 카메라의 동시 촬영을 뜻하지 않는다.** GDPR 삭제·익명화로 공개본이 **비동기 샘플링된 부분집합**이 되었기 때문이다. 그 결과:
- 프레임 각각이 **독립적인 id**를 가지며, 4-tuple로 묶이지 않는다.
- **캘리브레이션도 프레임별**로 따로 제공된다 (한 카메라의 고정 파라미터를 공유하는 게 아니라 프레임마다 json 1개).
- 카메라별 장수도 제각각: **FV 2,037 / RV 1,986 / MVL 2,066 / MVR 2,145** (train).

> 우리 자체 데이터셋은 4-cam 동기 촬영이 가능하므로, 이 부분은 WoodScape보다 **더 나은 구조(동기화된 4-tuple + 카메라별 고정 캘리브레이션)** 를 가져갈 수 있다. WoodScape의 per-frame 방식은 "제약의 산물"임을 인지할 것.

---

## 4. 라벨 포맷 레퍼런스 (실제 예시 포함)

### 4.1 box_2d_annotations
- **5 클래스**: `vehicles(0)`, `person(1)`, `bicycle(2)`, `traffic_light(3)`, `traffic_sign(4)` (색상표는 `box_2d_annotation_info.json`).
- 파일: `box_2d_annotations/NNNNN_XX.txt`, **헤더 없는 CSV, 객체당 1줄, 6열**:
  ```
  class_name,class_index,x_min,y_min,x_max,y_max
  ```
  실제 예시 (`00000_FV.txt`):
  ```
  vehicles,0,89,379,135,442
  person,1,431,302,465,389
  traffic_light,3,379,243,398,277
  ```
  좌표는 정수 픽셀, 좌상단 원점.

### 4.2 dense_polygon_annotations
- box_2d와 **같은 5 클래스**. 파일: `polygon_annotations/NNNNN_XX.txt` (압축 해제됨).
- **가변 길이 CSV**: 앞 6열은 box_2d와 동일(class + bbox), 이후는 폴리곤 정점 x,y를 flat하게 나열:
  ```
  class_name,class_index,x_min,y_min,x_max,y_max,px1,py1,px2,py2,...
  ```
  (인스턴스 마스크 생성용 dense 윤곽점.)

### 4.3 instance_annotations — **마스터 어노테이션**
semantic/box/dense polygon은 모두 이 인스턴스 폴리곤에서 *생성*된 것이다. 클래스 어휘는 `class_info.json`의 **43개 태그**.
- 파일: `instance_annotations/NNNNN_XX.json`. **top-level은 파일명을 키로 갖는 dict**:
  ```json
  { "00000_FV.json": {
      "annotation": [ {
          "id": "<UUID>",
          "segmentation": [[193.0, 3.38], [246.7, 3.38], ...],  // float 픽셀 폴리곤 정점
          "states": {},
          "tags": ["ego_vehicle"],                              // class_info의 태그
          "z_order": 0                                          // 그리기 순서(작을수록 뒤)
        }, ... ],
      "annotation-tags": ["green_strip", "ego_vehicle", ...],
      "image_height": 966, "image_width": 1280, "image_channels": 3,
      "job_batch_id": 34314, "job_id": 212321544, "status": "finished"
  } }
  ```
  (예시 프레임엔 106개 인스턴스.)

### 4.4 semantic_annotations
- **10 클래스** (`seg_annotation_info.json`): `void(0)` `road(1)` `lanemarks(2)` `curb(3)` `person(4)` `rider(5)` `vehicles(6)` `bicycle(7)` `motorcycle(8)` `traffic_sign(9)`.
- 두 형태의 PNG (1280×966):
  - `gtLabels/NNNNN_XX.png` — **단일 채널 인덱스 마스크** (픽셀값 = 클래스 index 0–9). 학습용.
  - `rgbLabels/NNNNN_XX.png` — **3채널 색상 마스크** (info의 class_colors로 색칠). 시각화용.
- ⚠️ 이 semantic은 dense한 전체-scene 라벨이 **아니다.** 관심 10개만 라벨하고 나머지(건물·하늘·자연물 등)는 전부 `void(0)`다. info 첫 줄이 *"use semantic_map_generator.py for generating annotations for 40+ classes"* 라고 밝히듯, 공개본은 **43-class instance 마스터에서 뽑은 10-class 축약본**이다. instance보다 오히려 sparse해 보이는 이유는 §7.1에서 정리한다.

### 4.5 motion_annotations
- **19 클래스** (`motion_annotation_info.json`): animal, rider, person, bicycle, motorcycle, car, van, bus, truck, train_tram, ... (주의: info에 `no samples for dynamic_car, dynamic_van, moveable_objects`).
- PNG (1280×966), `gtLabels/` + `rgbLabels/`.
  - ⚠️ **주의:** motion `gtLabels`는 semantic과 달리 **3채널**로 저장되어 있고, 픽셀 인덱스 값과 info의 0-based 클래스 인덱스가 딱 맞아떨어지지 않는다. 따라서 **정확한 클래스 식별은 `rgbLabels`의 색상 ↔ info `class_colors` 매칭**으로 하는 편이 안전하다 (시각화 도구도 이 방식 사용).

### 4.6 calibration_data — **fisheye 캘리브레이션** (BEV 핵심)
- 파일: `calibration/NNNNN_XX.json` (test는 `calibration(test)/`). 스키마(실제값은 `05382_FV.json`):
  ```json
  {
    "extrinsic": {
      "quaternion": [qx, qy, qz, qw],   // camera→vehicle 회전 (scipy from_quat 순서)
      "translation": [tx, ty, tz]        // meters
    },
    "intrinsic": {
      "model": "radial_poly", "poly_order": 4,
      "k1": 339.749, "k2": -31.988, "k3": 48.275, "k4": -7.201,  // radial poly 계수
      "cx_offset": 3.942, "cy_offset": -3.093,   // principal point 오프셋(중심 기준, px)
      "aspect_ratio": 1.0,
      "width": 1280.0, "height": 966.0
    },
    "name": "FV"
  }
  ```
- **⚠️ "프레임별" 파일이지만 프레임마다 재추정한 값이 아니다.** calib 파일은 8,234개(프레임과 1:1)지만 **실제 unique 내용은 126개뿐**이다(카메라별로 예: FV 2,037파일 중 32종, RV 31, MVL 31, MVR 32). 같은 카메라의 인접 프레임(`00000_FV`↔`00001_FV`)은 내용이 md5까지 동일하다. 이유: WoodScape는 **한 대가 아니라 여러 차량·여러 캠페인**으로 수집했고, 카메라 모델이 같아도 개체·장착이 미세하게 달라 **차량(=recording) 단위로 한 번씩 체커보드 캘리브레이션**을 했다. 서로 다른 두 FV variant를 비교하면 intrinsic(k1 341.7 vs 339.7)도 extrinsic(translation x 4.009 vs 3.748)도 다르다 = 다른 물리 개체. 즉 실제 참조 구조는 `frame_id → vehicle_id → calibration`인데, vehicle_id 매핑을 공개하지 않아 **프레임마다 값을 복사해 박아넣은(denormalized) 형태**다. 사용자의 "보통 체커보드로 한 번 계산하는 것 아니냐"는 직관이 맞다 — 프레임 단위가 아니라 세션 단위 1회다.
  - **자체 데이터셋 시사점:** 차량·장비 세팅이 고정이면 calib 1세트(카메라 4개)면 충분하고 프레임마다 만들 필요 없다. 다만 여러 rig로 확장할 계획이면 WoodScape식 프레임별 파일 관례를 따르는 편이 로더 호환에 유리하다(nuScenes/KITTI 로더도 샘플마다 calib을 참조).
- **좌표계**: 외부파라미터는 ISO 8855 차량 좌표(원점=후축 중점 아래 지면; x 전방, y 좌, z 상)와 OpenCV 카메라 좌표 사이 변환. 회전은 quaternion, 이동은 meter.
- **fisheye 투영 모델** (4차 radial polynomial):
  ```
  ρ(θ) = k1·θ + k2·θ² + k3·θ³ + k4·θ⁴
  ```
  θ = 광축 대비 입사각, ρ = 이미지 중심으로부터의 거리(px). 3D→2D 투영식:
  ```
  χ = √(X² + Y²),  θ = atan2(χ, Z)
  u = ρ·X/χ + cx_offset + width/2  − 0.5
  v = ρ·Y/χ·aspect_ratio + cy_offset + height/2 − 0.5
  ```
  구현·역투영(np.roots)은 공식 `third_party/datasets/WoodScape/scripts/calibration/projection.py`에 그대로 있고, 우리 시각화 도구가 이를 재사용한다.
- **`calib` 시각화의 두 패널이 뜻하는 것** (`viz_calib.py`):
  - (좌) 차량 좌표 z=0 지면에 1m 격자 3D 점을 만들어 위 투영식으로 fisheye 이미지에 찍은 것 → **BEV 지면 좌표 ↔ 어안 픽셀 매핑**을 눈으로 확인.
  - (우) 원본 어안 모델과, 수평선을 편 원통형(cylindrical) 모델 사이의 remap 맵(`create_img_projection_maps`)을 만들어 `cv2.remap`으로 왜곡을 편 것.
- **⚠️ 자체 어안 카메라로 이만큼 깔끔한 변환이 안 나오는 이유** (로직이 아니라 입력·모델 품질 문제):
  - **(a) 투영 모델 종류.** WoodScape는 `ρ = Σ kₙ·θⁿ`로 **θ의 1~4차 전 차수**를 쓴다. OpenCV `fisheye`(Kannala-Brandt)는 홀수차만(`θ + k1θ³ + k2θ⁵ …`), OCamCalib/Scaramuzza는 반대로 **ρ→θ 다항식 + affine 행렬**이다. 렌즈에 맞지 않는 모델로 맞추면 가장자리 잔차가 커서 격자가 휜다.
  - **(b) extrinsic이 있어야 지면 격자가 나온다.** 좌측 격자는 **camera→vehicle 외부파라미터(회전·이동)가 정확히 있어야** 그릴 수 있다. 체커보드로 보통 얻는 것은 **intrinsic뿐**이라, 지면 z=0 격자 투영은 **별도의 extrinsic(지면 기준) 캘리브레이션**이 추가로 필요하다. WoodScape는 이 값이 데이터에 포함돼 있어 격자가 딱 맞는다.
  - **(c) 캘리 정확도.** 정밀 측정된 계수는 잔차가 작아 undistort 후에도 곡률이 거의 안 남는다. 손으로 몇 장 맞춘 계수는 미세 곡률이 남는다.

### 4.7 vehicle_data — CAN / ego-motion
- 압축 해제 후 `vehicle_info/` 아래 **두 폴더**: `rgb_images/`, `previous_images/` (원본은 zip-in-zip이었다). 각 프레임당 json 1개.
- 스키마(값은 문자열, 예 `rgb_images/00000_FV.json`):
  ```json
  {
    "frame_id": "00000_FV.png", "timestamp": "21996742",
    "ego_speed": "6.200000", "ego_acceleration": "62.000000",
    "ego_steering": "-11.440000", "ego_yawRate": "-6.530000",
    "ego_gear": "D", "ego_distance": "0.057384"
  }
  ```
  ego 속도/가속/조향/yaw rate/주행거리/기어/타임스탬프. **GPS(위경도)는 없음.** `previous_images` 카운터파트는 직전 시점의 ego 상태.

### 4.8 soiling_dataset (별도 트랙, BEV와 무관)
- **4 클래스**: `clean(0)` `transparent(1)` `semi_transparent(2)` `Opaque(3)` (`soiling_annotation_info.json`).
- `train/`(4,000) `test/`(1,000), 각각 `rgbImages/` + `gtLabels/`(인덱스) + `rgbLabels/`(색상). 4자리 id.

---

## 5. 데이터 규모 요약

| 구분 | 수량 |
|---|---|
| Core annotated set (train) | **8,234 samples** (rgb / previous / box_2d / instance / dense_polygon), semantic·motion·calibration은 8,235 |
| Test set (held-out) | **1,766 samples** (rgb / previous / calibration만) |
| Soiling (별도) | 5,000 (train 4,000 / test 1,000) |
| 클래스 수 | box 5 · semantic 10 · motion 19 · instance 43 태그 · soiling 4 |

---

## 6. 라벨 시각화 도구

`tools/woodscape_viz/` — 라벨을 RGB 이미지 위에 오버레이해 PNG로 저장한다. 추출된 데이터셋 파일에서 직접 읽는다(이중 폴더 레이아웃 기준). `bev-chamdog` conda 환경에서 실행 (numpy/opencv/scipy 필요).

```bash
# 한 샘플, 전체 라벨(+스택 패널)
python tools/woodscape_viz/visualize.py --sample 00000_FV --out outputs/woodscape_viz

# 특정 타입만
python tools/woodscape_viz/visualize.py --sample 01234_MVL --types box,semantic

# 갤러리: 카메라별 N개 랜덤 샘플, 전체 타입
python tools/woodscape_viz/visualize.py --gallery 3 --out outputs/woodscape_viz
```

지원 타입: `box`, `semantic`, `instance`, `motion`, `calib`.
- `box/semantic/instance/motion` — 라벨을 이미지에 오버레이 + 클래스 범례.
- `calib` — 공식 `projection.py`를 재사용해 (좌) 차량 좌표계 지면 격자(1m, z=0)를 fisheye 이미지에 투영, (우) fisheye→cylindrical 왜곡 보정을 나란히 보여준다. **BEV 좌표 ↔ fisheye 이미지 매핑**을 눈으로 확인하는 용도.

구성 파일:
- `ws_io.py` — 데이터셋 I/O (추출된 파일 읽기, 색상표 로딩, 오버레이/범례 유틸)
- `viz_box.py` / `viz_semantic.py` / `viz_instance.py` / `viz_motion.py` / `viz_calib.py` — 타입별 렌더러
- `visualize.py` — CLI 러너

---

## 7. semantic vs instance density, 그리고 BEV 라벨링 관례

라벨 시각화만 보면 "**semantic이 더 dense해야 하는데 오히려 instance가 더 빽빽하다**"는 인상을 받기 쉽다. 실제로 그게 맞고, 아래 두 가지가 이유다.

### 7.1 왜 instance가 semantic보다 빽빽한가 (직관과 반대)

두 라벨은 별개가 아니라 **instance가 원본(마스터), semantic이 그로부터 뽑은 축소판**이다.

| | 클래스 수 | 배경(건물/하늘/자연물) | 렌더링 방식 | 체감 밀도 |
|---|---|---|---|---|
| **semantic** | 10 (void 포함) | 전부 `void(0)`로 버려짐 | void=검정 픽셀은 blend에서 **스킵**(`ws_io.blend_nonzero`) → 배경 텅 빔 | sparse |
| **instance** | 43 태그 | `sky`·`structure`·`nature`·`construction`·`road_surface` 등 폴리곤 존재 | 모든 폴리곤을 `fillPoly`로 칠함(`viz_instance.py`) | dense |

- semantic이 비어 보이는 건 **버그가 아니라 의도된 축약본** + **void 미렌더링**의 조합이다.
- 진짜 dense한 40+ class semantic이 필요하면 instance 폴리곤에서 `semantic_map_generator.py`로 **재생성**해야 한다.
- BEV 3D 검출 관점에선 정보량이 많은 **instance 폴리곤 쪽이 활용 가치가 크다** (관심 객체 클래스만 추려 쓰면 됨).

### 7.2 BEV task는 왜 관심 객체만 라벨하나 (sky는 어디서도 불필요)

"BEV에서 sky 같은 클래스는 필요 없는데 왜 라벨돼 있나?"에 대한 답: **WoodScape의 semantic/instance는 이미지 평면(perspective) 라벨이지 BEV 라벨이 아니다.** BEV 라벨링 관례는 다르다.

| BEV 하위 task | 라벨 형태 | 대상 | sky |
|---|---|---|---|
| **3D object detection** (nuScenes/KITTI/Waymo) | 3D bounding box | **동적/관심 객체만** (car·truck·bus·person·cyclist 등) | 없음 (칠 박스 자체가 없음) |
| **BEV map seg / occupancy** (BEVFormer·Occ3D 등) | top-down 격자 / voxel semantic | 지면 요소 (drivable·lane·sidewalk / vegetation·manmade) | 없음 |

- **sky가 어느 BEV 표현에서도 빠지는 이유:** BEV는 지면(z≈0)을 내려다본 top-down 투영이라, 하늘은 **지면 발자국(ground footprint)이 없어** BEV 격자·voxel에 떨어질 좌표가 없다. 가장 dense한 3D voxel semantic인 Occ3D-nuScenes조차 `vegetation`/`manmade`(건물)는 있어도 **sky 클래스는 없다** (voxel이 지면에 앵커링되므로).
- 따라서 **"관심 객체만 라벨"은 BEV 3D detection의 정상 관례**이고, sky를 라벨하지 않는 것도 표준이다. 님 직관이 맞다.
- **우리 dataset 함의:** BEV 3D 검출이 목표라면 배경 클래스(sky·building)는 버리고, 관심 객체의 **3D 박스**만 필요하다. WoodScape 2D 폴리곤에서 관심 객체 클래스만 추리되, **3D 정보는 3D 라벨이 있는 SynWoodScape 등에서** 가져와야 한다(§8, §1).

---

## 8. 자체 BEV 데이터셋 구축에 주는 시사점

- **가져갈 것:** fisheye 캘리브레이션 JSON 스키마(radial_poly + quaternion/translation), per-frame 또는 per-camera 캘리브레이션 파일 분리, `NNNNN_CAM` 파일명 규칙, semantic gt=인덱스/rgb=색상 이원화, CAN(ego-motion) json 구조.
- **개선할 것:** 우리는 4-cam 동기 촬영이 가능하므로 **동기화된 4-tuple**과 **카메라별 고정 캘리브레이션**을 쓸 수 있다(WoodScape의 per-frame 비동기 구조는 GDPR 제약의 산물).
- **WoodScape에 없어 별도 참고 필요:** **3D bounding box 라벨 포맷**, **depth GT**. → 후속 **SynWoodScape**(3D 라벨 포함) 분석에서 다룬다.
