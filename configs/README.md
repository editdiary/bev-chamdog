# configs/

자체 실험 설정(하이퍼파라미터·데이터 경로·BEV 범위 등)을 두는 폴더입니다.

- baseline인 **Simple-BEV에는 config 파일 체계가 없습니다.** `train_nuscenes.py`가 [`fire`](https://github.com/google/python-fire)로 `main(...)`의 키워드 인자를 받고, 실행은 인자를 나열한 셸 스크립트(`train.sh` 형태)로 합니다.
- **형식이 확정됐습니다 (Phase 3.3):** `tools/train_synwoodscape.py`도 같은 방식 — `Fire(main)`로 키워드 인자를 받고, `configs/train_synwoodscape_baseline.sh`처럼 실험별 셸 스크립트에 인자를 나열합니다. 새 실험을 만들 때는 이 파일을 복사해 `configs/train_synwoodscape_<실험명>.sh`로 남기는 걸 권장합니다.
- 설정에 담긴 항목: `exp_name`, `num_epochs`, `batch_size`, `lr`, `weight_decay`, `num_workers`, `val_fraction`/`split_seed`, `encoder_type`, `use_fisheye`, `pos_weight`, `log_dir`/`ckpt_dir`. BEV 범위·격자 해상도·drivable remap 규칙은 `projects/bev_gt/grid.py`에 이미 고정돼 있어 이 스크립트의 인자로 노출하지 않습니다.
- 각 값의 의미·튜닝 기준(무엇을 보고 무엇을 바꿀지)은 **`docs/training_guide.md`** 참고.
- RTX PRO 6000(VRAM 96GB) 단일 GPU 기준입니다. VRAM 제약이 없으므로 `batch_size` 축소·gradient accumulation을 전제할 필요가 없고, `num_workers`와 GPU 활용률을 함께 기록해 데이터 로딩 병목을 관리합니다. bf16 AMP는 실측 후 결정합니다. (→ `docs/setup_guide_pro6000.md`)
