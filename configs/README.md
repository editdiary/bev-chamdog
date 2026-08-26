# configs/

자체 실험 설정(하이퍼파라미터·데이터 경로·BEV 범위 등)을 두는 폴더입니다.

- baseline인 **Simple-BEV에는 config 파일 체계가 없습니다.** `train_nuscenes.py`가 [`fire`](https://github.com/google/python-fire)로 `main(...)`의 키워드 인자를 받고, 실행은 인자를 나열한 셸 스크립트(`train.sh` 형태)로 합니다.
- **형식이 확정됐습니다 (Phase 3.3):** `tools/train_synwoodscape.py`도 같은 방식 — `Fire(main)`로 키워드 인자를 받고, `configs/train_synwoodscape_baseline.sh`처럼 실험별 셸 스크립트에 인자를 나열합니다. 새 실험을 만들 때는 이 파일을 복사해 `configs/train_synwoodscape_<실험명>.sh`로 남기는 걸 권장합니다.
- 설정에 담긴 항목: `exp_name`, `num_epochs`, `batch_size`, `lr`, `weight_decay`, `num_workers`, `val_fraction`/`split_seed`, `encoder_type`, `use_fisheye`, `pos_weight`, `log_dir`/`ckpt_dir`. BEV 범위·격자 해상도·drivable remap 규칙은 `projects/bev_gt/grid.py`에 이미 고정돼 있어 이 스크립트의 인자로 노출하지 않습니다.
- 각 값의 의미·튜닝 기준(무엇을 보고 무엇을 바꿀지)은 **`docs/archive/training_guide.md`** 참고.
- RTX PRO 6000(VRAM 96GB) 단일 GPU 기준입니다. VRAM 제약이 없으므로 `batch_size` 축소·gradient accumulation을 전제할 필요가 없고, `num_workers`와 GPU 활용률을 함께 기록해 데이터 로딩 병목을 관리합니다. bf16 AMP는 실측 후 결정합니다. (→ `docs/setup_guide_pro6000.md`)

---

## 스크립트 색인 (2026-08-26)

**각 스크립트 맨 위 주석이 정본이다** -- 무엇을 묻는 실험인지, 대조군이 어디 있는지,
읽는 법이 무엇인지가 거기 적혀 있다. 아래는 찾아가기용 목차다.

| 스크립트 | 무엇 | 산출물 | 근거 문서 |
|---|---|---|---|
| `train_robot_bev_finetune.sh` | **모든 실험의 진입점.** 환경변수로 인자를 넘긴다. 다른 스크립트는 전부 이것을 부른다 | — | `docs/finetuning_guide.md` |
| `ablation_loss.sh` | **loss 사다리 4칸 × 시드 3 = 12런.** `A_ce → B_perset → C_soft → D_range` | `runs/ablation/` | 설계 §15·§16 |
| `sweep_soft_boundary.sh` / `sweep_soft_alpha.sh` | soft-boundary 1·2차 스윕(δ, α, λ_B) | `runs/robot_bev_cv/loss_sweep/` | 설계 §9~§11 |
| `sweep_range_loss.sh` / `sweep_range_asymmetric.sh` | `L_range` 스윕, 비대칭 dead zone | 같음 | 설계 §13.6·§13.8 |
| `measure_run_noise.sh` | **같은 config·같은 시드 5런** -- `σ_run` 실측 | `runs/robot_bev_cv/run_noise/` | 설계 §13.6.1 |
| `repeat_seeds.sh` | 대표 split 시드 반복 -- `σ_seed` 실측 | `runs/robot_bev_cv/seeds/` | 진단 §24 |
| `sweep_pixel_offset.sh` | 표본 좌표 규약·오프셋 15런 | `runs/pixel_offset/` | 진단 §18.3.5 |
| `probe_frame_split.sh` / `probe_frame_blocks.sh` | **같은 장면 일반화 프로브.** 프레임 단위 split(무작위 / 블록+gap). **성능으로 보고하지 않는다** | `runs/frame_split/`, `runs/frame_blocks/` | 진단 §28.4·§28.10 |
| `stride4_arms.sh` | **stride 8→4 사전 선언 실험** 6런(두 split × 시드 3) | `runs/stride4/` | 진단 §29(사전 선언)·§29.9(결과, **기각**) |
| `loso_folds.sh` | **LOSO 7-fold × 시드 3 = 21런.** config 동결 후 **한 번만** | `runs/robot_bev_cv/loso/` | 진단 §25.6·§25.7·§25.8·§31 |
| `train_synwoodscape_baseline.sh` / `..._threeclass_pretrain.sh` | SynWoodScape pretrain. **현행은 from scratch이므로 쓰지 않는다** | — | 진단 §11(pretrain 유해) |

**규약 넷.**

1. **이미 있는 런 폴더는 건너뛴다** -- 중단 후 재실행이 안전하다.
2. **런이 하나라도 떠 있으면 `configs/`를 편집하지 않는다.** bash가 파일을 게으르게 읽어
   바이트 오프셋이 어긋난 자리에서 재개한다(실제로 두 번 겪었다 -- 진단 §24.5·§28.9).
3. **`VAL_SEQUENCES=""`를 넘기지 않는다.** 셸의 `${VAR:-기본값}`이 빈 문자열도 unset으로
   취급해 기본값이 되살아나고, 같은 시퀀스가 두 번 들어간다(진단 §28.9).
4. **확정 config는 진단 §25.7이 동결 선언과 함께 적어 두었다.** 새 실험은 그 값에서
   **한 가지만** 바꾼다.
