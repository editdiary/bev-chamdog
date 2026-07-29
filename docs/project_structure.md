# 프로젝트 폴더 구조

## 전체 트리

```
bev-chamdog/
├── README.md              # 프로젝트 개요
├── CLAUDE.md              # 개발 핵심 지침
├── ROADMAP.md             # 단계별 로드맵
├── constraints.txt        # 패키지 버전 고정 (numpy<2, opencv<5)
├── .gitignore
├── .gitmodules            # submodule 정의
├── docs/                  # 문서 모음
│   ├── setup_guide.md         # 환경 세팅 가이드 & 트러블슈팅
│   ├── git_workflow.md        # git 브랜치 전략
│   ├── project_structure.md   # (현재 문서)
│   ├── dataset_analysis/      # WoodScape/SynWoodScape 등 분석 노트
│   └── study/                 # 배경 지식 정리 (카메라 모델·캘리브레이션 등)
├── dataset/               # 데이터셋 (내용물은 git 미추적)
├── third_party/           # 외부 레포 (git submodule 모음)
│   ├── datasets/
│   │   └── WoodScape/         # WoodScape 공식 레포 (캘리브레이션 규약·참고 코드)
│   └── models/
│       ├── simple_bev/        # ★ 현재 baseline 모델
│       ├── lift-splat-shoot/  # 참고용 논문 구현체
│       ├── bevformer/         # 〃
│       └── BEVDet/            # 〃
├── mmdetection3d/         # MMDetection3D (submodule, v1.4.0 — 현재 미사용)
├── configs/               # 자체 실험 설정
├── tools/                 # 커스텀 스크립트
└── projects/              # 커스텀 파이썬 모듈 (로더·투영·모델 래퍼 등)
```

## 폴더별 역할

| 폴더 | 역할 |
|---|---|
| `docs/` | 모든 문서. 환경/워크플로/구조, 데이터셋 분석 노트, 배경 지식 정리. |
| `dataset/` | 원본·가공 데이터셋. 용량이 크므로 **내용물은 git으로 추적하지 않음** (`dataset/README.md`만 유지). |
| `third_party/datasets/` | 데이터셋 공식 레포(submodule). 코드에서 직접 참조하는 것만 둔다 (예: WoodScape 캘리브레이션 스크립트·규약 문서). |
| `third_party/models/` | BEV 논문 구현체(submodule). **`simple_bev/`는 현재 baseline으로 직접 사용**하며, 나머지는 참고·확장용. |
| `mmdetection3d/` | 학습 프레임워크. **현재 task(BEV occupancy)에서는 사용하지 않음.** 이후 3D 검출로 확장할 때를 위해 남겨둠. 실행 경로가 루트 기준으로 고정돼 있어 루트에 유지. |
| `configs/` | 자체 실험용 설정(하이퍼파라미터·경로·BEV 범위 등). → `configs/README.md` |
| `tools/` | 데이터 변환, 분석, 시각화, 검증 등 보조 스크립트. |
| `projects/` | 이 프로젝트 고유의 파이썬 코드(데이터 로더, BEV GT 변환, 어안 투영, 모델 래퍼 등). |

## submodule 취급

`third_party/*`와 `mmdetection3d/`는 모두 git submodule이다.

- **직접 수정 금지.** baseline(`simple_bev`)의 동작을 바꿔야 하더라도 submodule을 고치지 않고 `projects/`에서 래핑한다. submodule을 고치면 업스트림 갱신이 불가능해지고 변경 이력이 부모 저장소에 남지 않는다.
- **클론:** `git clone --recursive ...` 또는 클론 후 `git submodule update --init --recursive`.
- **버전 변경(신중히):**
  ```bash
  git -C <submodule-경로> fetch --tags
  git -C <submodule-경로> checkout <새-태그-또는-커밋>
  git add <submodule-경로>          # 부모 repo에 새 커밋 포인터 기록
  # merge/push는 사용자가 직접 (docs/git_workflow.md 규칙 준수)
  ```
- `mmdetection3d/`는 `v1.4.0`에 고정되어 있다.

## `projects/` 사용 방식

Simple-BEV는 registry나 플러그인 체계가 없는 평범한 파이썬 코드다. 따라서 `projects/`도 특별한 등록 절차 없이 **일반 파이썬 패키지**로 쓴다.

1. `projects/` 아래에 모듈을 작성한다. 예상 구성:
   - 데이터 로더 — SynWoodScape / 자체 데이터셋을 Simple-BEV 입력 형태로 변환
   - BEV GT 변환 — semantic label → binary occupancy(drivable / non-drivable) remap
   - 어안 투영 — `radial_poly` project / unproject
   - 모델 래퍼 — submodule을 건드리지 않고 Simple-BEV의 lifting 투영을 어안으로 교체
2. submodule 코드를 임포트할 때는 경로를 추가한다:
   ```python
   import sys
   sys.path.insert(0, "third_party/models/simple_bev")
   ```
   (`simple_bev`는 설치형 패키지가 아니라 스크립트 저장소라 이 방식이 필요하다.)
3. **실행은 저장소 루트에서** 수행한다 (`projects`가 임포트 경로로 잡히도록).
   ```bash
   python tools/<my_script>.py
   ```
