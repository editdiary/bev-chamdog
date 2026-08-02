# projects/

이 프로젝트 고유의 **커스텀 파이썬 모듈**을 두는 패키지입니다. `third_party/`와 `mmdetection3d/`는 submodule이라 직접 수정하지 않고, 확장은 모두 여기서 합니다.

## 예상 구성

- **데이터 로더** — SynWoodScape / 자체 데이터셋을 Simple-BEV 입력 형태로 변환 (아직 미착수)
- **BEV GT 변환** — `bev_gt/` 참고
- **어안 투영** — `geometry/` (`radial_poly` project / unproject, 캘리브레이션 규약은 `docs/study/camera_models_and_calibration.md` 참고)
- **모델 래퍼** — Simple-BEV의 lifting 투영을 어안으로 교체. **submodule을 고치지 않고 래핑한다.** (아직 미착수)

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
