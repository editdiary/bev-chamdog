# configs/

자체 실험용 학습/추론 config를 두는 폴더입니다.

- mmdet3d의 기본 config(`mmdetection3d/configs/...`)를 `_base_`로 상속하거나 필요한 값만 오버라이드합니다.
- 커스텀 모듈(`projects/`)을 쓰는 config에는 `custom_imports`를 명시합니다. (→ `docs/project_structure.md`)
- RTX 3080(VRAM 10GB) 단일 GPU 기준으로 `batch_size` 등을 조정합니다. (→ `docs/setup_guide.md`)
