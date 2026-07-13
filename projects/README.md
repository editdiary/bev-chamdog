# projects/

mmdet3d registry에 등록할 **커스텀 모듈**(dataset 클래스, transform, model 등)을 두는 파이썬 패키지입니다. `mmdetection3d/`는 submodule이라 직접 수정하지 않고, 확장은 모두 여기서 합니다.

## 사용 방식 (요약)

1. 모듈 작성 후 데코레이터로 등록:
   ```python
   from mmdet3d.registry import DATASETS

   @DATASETS.register_module()
   class WoodScapeDataset(...):
       ...
   ```
2. config에서 임포트 지정:
   ```python
   custom_imports = dict(imports=['projects.datasets.woodscape_dataset'],
                         allow_failed_imports=False)
   ```
3. **실행은 저장소 루트에서** (`projects`가 임포트 경로로 잡히도록).

자세한 내용은 `docs/project_structure.md` 참고.
