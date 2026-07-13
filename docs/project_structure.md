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
│   └── dataset_analysis/      # WoodScape/SynWoodScape 등 분석 노트
├── dataset/               # 데이터셋 (내용물은 git 미추적)
├── mmdetection3d/         # MMDetection3D (git submodule)
├── configs/               # 커스텀 학습/추론 config
├── tools/                 # 커스텀 스크립트
└── projects/              # 커스텀 mmdet3d 모듈 (registry 등록)
```

## 폴더별 역할

| 폴더 | 역할 |
|---|---|
| `docs/` | 모든 문서. 환경/워크플로/구조 및 데이터셋 분석 노트. |
| `dataset/` | 원본·가공 데이터셋. 용량이 크므로 **내용물은 git으로 추적하지 않음** (`dataset/README.md`만 유지). |
| `mmdetection3d/` | 학습 프레임워크. **submodule → 직접 수정 금지.** |
| `configs/` | 자체 실험용 학습/추론 config. mmdet3d config를 상속·오버라이드. |
| `tools/` | 데이터 변환, 분석, 시각화 등 보조 스크립트. |
| `projects/` | mmdet3d registry에 등록할 커스텀 코드(dataset 클래스, transform, model 등). |

## mmdetection3d submodule 취급

- **버전:** `v1.4.0`에 고정(pin)되어 있음.
- **직접 수정 금지.** 기능 확장이 필요하면 `projects/`에서 상속/등록으로 해결한다.
- **클론:** `git clone --recursive ...` 또는 클론 후 `git submodule update --init --recursive`.
- **업데이트(신중히):** 버전을 바꿔야 할 경우
  ```bash
  git -C mmdetection3d fetch --tags
  git -C mmdetection3d checkout <새-태그>
  git add mmdetection3d            # 부모 repo에 새 커밋 포인터 기록
  # merge/push는 사용자가 직접 (docs/git_workflow.md 규칙 준수)
  ```

## `projects/` 커스텀 모듈 등록 방식

mmdet3d는 registry 기반이라, 커스텀 클래스를 정의한 뒤 config에서 `custom_imports`로 불러오면 등록된다.

1. `projects/` 아래에 모듈 작성 (예: `projects/datasets/woodscape_dataset.py`), 데코레이터로 등록:
   ```python
   from mmdet3d.registry import DATASETS

   @DATASETS.register_module()
   class WoodScapeDataset(...):
       ...
   ```
2. config에서 등록 임포트를 지정:
   ```python
   custom_imports = dict(imports=['projects.datasets.woodscape_dataset'],
                         allow_failed_imports=False)
   ```
3. **실행은 저장소 루트에서** 수행한다 (`projects`가 임포트 경로로 잡히도록). 예:
   ```bash
   python mmdetection3d/tools/train.py configs/<my_config>.py
   ```
