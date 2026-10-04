# projects/

이 프로젝트 고유의 **커스텀 파이썬 모듈**을 두는 패키지입니다. `third_party/`와 `mmdetection3d/`는 submodule이라 직접 수정하지 않고, 확장은 모두 여기서 합니다.

## 구성 (2026-10-04)

| 폴더 | 무엇 | 지금 쓰는 핵심 |
|---|---|---|
| `datasets/` | 데이터 로더 | `robot_simplebev.py`(자체 3-cam 데이터, 마스크), `simplebev_vox.py`(`vox_dims` = `Y`의 단일 출처), `photometric.py`(증강), `synwoodscape_*.py`(사전학습) |
| `geometry/` | 카메라 모델 | `double_sphere.py`(자체 리그), `fisheye.py`(SynWoodScape `radial_poly`) |
| `models/` | Simple-BEV 래퍼(submodule 무수정) | `double_sphere_vox.py`(어안 직접 lifting), `virtual_pinhole.py`(실험 02의 핀홀 대조), `pixel_grid.py` |
| `common/` | 손실·지표·통계 | `soft_boundary.py`(목적함수), `metric_spec.py`(**지표 이름·tag 정본**), `free_space_metrics.py`·`occupied_metrics.py`(지표), `binary_metrics.py`(학습 루프의 손실 분해), `paired_stats.py`(**유의성 판정 정본**), `npsafe.py` |
| `bev_gt/` | BEV 라벨 생성(SynWoodScape 시절) | 아래 표 |

옛 정식화(2-head·3-class)용 모듈(`three_class_*`, `segmentation_loss.py`, `simplebev_three_class.py`)과
광선 보조항(`range_loss.py`)은 코드·테스트가 남아 있지만 v3 경로에서는 쓰지 않는다.

## `bev_gt/` 모듈별 역할 (현재 vs 폐기)

시행착오를 거치며 여러 세대가 공존한다 — 자세한 이유는
`docs/dataset_analysis/synwoodscape_geometry_findings.md` §3 참고.

| 모듈 | 역할 | 상태 |
|---|---|---|
| `grid.py` | grid spec 정의, drivable class remap | 현재 |
| `bev_crop.py` | top-down `_BEV.png` → occupancy class 값 crop (`crop_bev_occupancy`) | 현재 |
| `raycast_occupancy.py` | `compute_observed_mask`(현재, observed/visible mask) + `build_raycast_occupancy`(폐기, 어안 라벨로 class까지 만듦) | 혼재 |
| `visibility.py` | grid cell을 z=0으로 투영해 가설검정하는 초기 visibility 시도 | 폐기 |
| `bev_calibration.py` | BEV 스케일/원점 보정용 헬퍼 (`tools/calibrate_bev_scale.py`가 사용) | 현재 |

## 사용 방식

Simple-BEV는 registry나 플러그인 체계가 없는 평범한 파이썬 코드이므로, 별도 등록 절차 없이 일반 패키지로 임포트합니다.

```python
import sys
sys.path.insert(0, "third_party/models/simple_bev")   # 설치형 패키지가 아니므로 경로 추가

from projects.<module> import ...
```

**실행은 저장소 루트에서** 수행합니다 (`projects`가 임포트 경로로 잡히도록).

자세한 내용은 `docs/project_structure.md` 참고.
