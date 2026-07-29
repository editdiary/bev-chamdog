# configs/

자체 실험 설정(하이퍼파라미터·데이터 경로·BEV 범위 등)을 두는 폴더입니다.

- baseline인 **Simple-BEV에는 config 파일 체계가 없습니다.** `train_nuscenes.py`가 [`fire`](https://github.com/google/python-fire)로 `main(...)`의 키워드 인자를 받고, 실행은 인자를 나열한 셸 스크립트(`train.sh` 형태)로 합니다.
- 따라서 여기에는 mmdet3d식 `_base_` 상속 config가 아니라, **실험별 설정을 담은 자체 파일**을 둡니다. 구체적인 형식(셸 스크립트 / YAML / 파이썬 dict)은 Phase 3에서 학습 스크립트를 작성하며 확정합니다. (→ `ROADMAP.md`)
- 설정에 담을 항목: 데이터 경로, BEV 범위·격자 해상도, drivable remap 규칙, batch size, 학습률, 에폭 수, 체크포인트 경로.
- RTX PRO 6000(VRAM 96GB) 단일 GPU 기준입니다. VRAM 제약이 없으므로 `batch_size` 축소·gradient accumulation을 전제할 필요가 없고, `num_workers`와 GPU 활용률을 함께 기록해 데이터 로딩 병목을 관리합니다. bf16 AMP는 실측 후 결정합니다. (→ `docs/setup_guide_pro6000.md`)
