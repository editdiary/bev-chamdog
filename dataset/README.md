# dataset/

학습 및 분석에 사용하는 데이터셋을 모아두는 폴더입니다.

> ⚠️ **이 폴더의 내용물은 git으로 추적하지 않습니다** (용량 큼). `.gitignore`에서 `/dataset/*`를 무시하고 이 `README.md`만 유지합니다. 데이터는 각자 환경에서 내려받아 배치하세요.

> 📌 현재 서버에서는 실제 데이터를 **`/data`(3.7T) 파티션**에 두고 이 폴더에 symlink를 겁니다.
> 루트 파티션에 넣으면 자체 데이터셋·체크포인트가 쌓일 때 좁아집니다.
> ```bash
> ln -s /data/datasets/synwoodscape dataset/synwoodscape
> ```

## 권장 하위 구조

```
dataset/
├── synwoodscape/              # ★ SynWoodScape 원본 다운로드 (Phase 3 입력)
├── synwoodscape_occupancy_gt/ # ★ 우리가 생성한 BEV occupancy GT (학습 입력)
├── woodscape/        # WoodScape 벤치마크 (분석 완료, 참고용)
├── nuscenes_mini/    # nuScenes v1.0-mini (포맷 참고용)
└── custom/           # 자체 구축 BEV 데이터셋 (Fisheye 4-cam, Phase 4에서 배치)
```

## 현재 보유

- **`synwoodscape/`** (약 61GB) — SynWoodScape `V0.1.0` 원본 다운로드, **500 samples**. Phase 3 학습 입력.
  - 샘플당 어안 4-cam(`FV`/`RV`/`MVL`/`MVR`) + BEV 1대의 RGB·semantic·depth, 그리고 LiDAR·3D 박스
  - `semantic_annotations/gtLabels/*_BEV.png` → **BEV occupancy GT의 원본** (drivable remap 대상)
  - `calibration_data/`에는 어안 4대의 json만 있고 **BEV 카메라 json은 없음** — 스케일은 이미
    `15/512 m/px`로 확정됨(→ `docs/dataset_analysis/synwoodscape_geometry_findings.md` §2)
  - 클래스 팔레트·센서 배치는 `synwoodscape/readme.txt` 참고
- **`synwoodscape_occupancy_gt/`** (약 34MB) — 위 원본으로부터 `tools/build_hybrid_occupancy.py`가
  생성한 BEV occupancy GT, **500 samples**. 원본 다운로드 트리와 분리해 우리가 만든 산출물만
  담는다. 샘플당 3개 파일:
  - `{sample}_occupancy.npy` — `(160,160) uint8`, `{0,1}` (1=drivable) — 학습 target
  - `{sample}_visible.npy` — `(160,160) bool` (True=관측됨=loss 포함, False=unknown=ignore) — 학습 mask
  - `{sample}_combined.png` — 초록/빨강/회색 3색 시각화 (사람이 눈으로 검증용, 학습에는 안 씀)
  - 설계 근거·시행착오는 `docs/dataset_analysis/synwoodscape_geometry_findings.md` §3 참고
- **`woodscape/`** (약 43GB) — WoodScape 원본(ICCV19). 분석 완료(→ `docs/dataset_analysis/woodscape_analysis.md`). 공개본은 2D 라벨만 제공하고 3D·depth GT가 없어 학습에는 쓰지 않는다. **캘리브레이션 규약 문서와 참고 코드는 `third_party/datasets/WoodScape/` submodule에 있다.**
- **`nuscenes_mini/`** (약 5.1GB) — nuScenes `v1.0-mini` (6-cam + LiDAR + radar). 3D bounding box 라벨 포맷 참고용(→ `docs/dataset_analysis/woodscape_analysis.md` §8). 현재 task(BEV occupancy)에는 사용하지 않으며, 이후 3D 검출로 확장할 때 참고한다.
