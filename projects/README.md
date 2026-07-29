# projects/

이 프로젝트 고유의 **커스텀 파이썬 모듈**을 두는 패키지입니다. `third_party/`와 `mmdetection3d/`는 submodule이라 직접 수정하지 않고, 확장은 모두 여기서 합니다.

## 예상 구성

- **데이터 로더** — SynWoodScape / 자체 데이터셋을 Simple-BEV 입력 형태로 변환
- **BEV GT 변환** — semantic label → binary occupancy(drivable / non-drivable) remap
- **어안 투영** — `radial_poly` project / unproject (캘리브레이션 규약은 `docs/study/camera_models_and_calibration.md` 참고)
- **모델 래퍼** — Simple-BEV의 lifting 투영을 어안으로 교체. **submodule을 고치지 않고 래핑한다.**

## 사용 방식

Simple-BEV는 registry나 플러그인 체계가 없는 평범한 파이썬 코드이므로, 별도 등록 절차 없이 일반 패키지로 임포트합니다.

```python
import sys
sys.path.insert(0, "third_party/models/simple_bev")   # 설치형 패키지가 아니므로 경로 추가

from projects.<module> import ...
```

**실행은 저장소 루트에서** 수행합니다 (`projects`가 임포트 경로로 잡히도록).

자세한 내용은 `docs/project_structure.md` 참고.
