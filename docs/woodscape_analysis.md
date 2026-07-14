# WoodScape ICCV19 데이터셋 분석

> 목적: 자체 Fisheye 4-cam BEV 3D 검출 데이터셋을 설계하기 전에, 가장 유사한 공개 데이터셋인
> **WoodScape**(Valeo, ICCV 2019)가 실제로 어떻게 구축되어 있는지 — 폴더 구성, 파일명 규칙,
> 각 라벨 포맷 — 을 정리한다. 모든 내용은 로컬 데이터(`dataset/WoodScape_ICCV19/`)와
> 공식 repo(`WoodScape/`)를 직접 열어 검증했다.

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

루트: `dataset/WoodScape_ICCV19/` (총 ~42 GB). 네이티브 이미지 크기 **1280×966** (soiling만 1280×960).

| 폴더 | 역할 | 내부 포맷 | 압축 여부 | 개수(train) | BEV 중요도 |
|---|---|---|---|---|---|
| `rgb_images/` | 입력 RGB (fisheye) | PNG | 압축 해제 (`rgb_images/rgb_images/`) | 8,234 (+test 1,766) | ★★★ |
| `previous_images/` | 각 프레임의 직전 프레임 (`_prev`) | PNG | 압축 해제 | 8,234 (+1,766) | ★★ (self-sup depth/motion용) |
| `box_2d_annotations/` | 2D 박스 (5 클래스) | `.txt` | **zip** | 8,234 | ★★ |
| `dense_polygon_annotations/` | 객체 dense 윤곽점 | `.txt` | 압축 해제 (`polygon_annotations/`) | 8,234 | ★ |
| `instance_annotations/` | **마스터** 인스턴스 폴리곤 (40+ 태그) | JSON | **zip** | 8,234 | ★★ |
| `semantic_annotations/` | semantic 세그 마스크 (10 클래스) | PNG (gt+rgb) | **zip** | 8,235 | ★★ |
| `motion_annotations/` | motion(움직임) 세그 마스크 (19 클래스) | PNG (gt+rgb) | **zip** | 8,235 | ★ |
| `calibration_data/` | **프레임별** fisheye 캘리브레이션 | JSON | **zip** (+test zip) | 8,235 (+1,766) | ★★★ |
| `vehicle_data/` | CAN/주행 데이터 (ego-motion) | JSON | **zip-in-zip** | 8,234 | ★★ |
| `soiling_dataset/` | 렌즈 오염 세그 (4 클래스, 별도 트랙) | PNG | 압축 해제 | 4,000 (+test 1,000) | ✗ (무관) |
| `WoodScape License and Terms of Use.pdf` | 라이선스 | PDF | — | — | — |

- **압축 해제됨:** rgb_images, previous_images, dense_polygon_annotations, soiling_dataset
- **아직 zip:** box_2d, instance, semantic, motion, calibration, vehicle
- 각 zip 폴더에는 작은 메타 파일(`*_info.json` 또는 `readme.txt`)이 압축 밖에 함께 있다.
- 시각화 도구는 **zip을 풀지 않고 메모리에서 직접** 읽으므로 별도 압축 해제가 필요 없다 (§6).

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
  구현·역투영(np.roots)은 공식 `WoodScape/scripts/calibration/projection.py`에 그대로 있고, 우리 시각화 도구가 이를 재사용한다.

### 4.7 vehicle_data — CAN / ego-motion
- `vehicle_info.zip` 안에 **중첩 zip 2개**: `rgb_images.zip`, `previous_images.zip`. 각 프레임당 json 1개.
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

`tools/woodscape_viz/` — 라벨을 RGB 이미지 위에 오버레이해 PNG로 저장한다. zip을 디스크에 풀지 않고 메모리에서 직접 읽는다. `mmdet3d` conda 환경에서 실행 (numpy/opencv/scipy 필요).

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
- `ws_io.py` — 데이터셋 I/O (zip 스트리밍, 색상표 로딩, 오버레이/범례 유틸)
- `viz_box.py` / `viz_semantic.py` / `viz_instance.py` / `viz_motion.py` / `viz_calib.py` — 타입별 렌더러
- `visualize.py` — CLI 러너

---

## 7. 자체 BEV 데이터셋 구축에 주는 시사점

- **가져갈 것:** fisheye 캘리브레이션 JSON 스키마(radial_poly + quaternion/translation), per-frame 또는 per-camera 캘리브레이션 파일 분리, `NNNNN_CAM` 파일명 규칙, semantic gt=인덱스/rgb=색상 이원화, CAN(ego-motion) json 구조.
- **개선할 것:** 우리는 4-cam 동기 촬영이 가능하므로 **동기화된 4-tuple**과 **카메라별 고정 캘리브레이션**을 쓸 수 있다(WoodScape의 per-frame 비동기 구조는 GDPR 제약의 산물).
- **WoodScape에 없어 별도 참고 필요:** **3D bounding box 라벨 포맷**, **depth GT**. → 후속 **SynWoodScape**(3D 라벨 포함) 분석에서 다룬다.
