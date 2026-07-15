# dataset/

학습 및 분석에 사용하는 데이터셋을 모아두는 폴더입니다.

> ⚠️ **이 폴더의 내용물은 git으로 추적하지 않습니다** (용량 큼). `.gitignore`에서 `/dataset/*`를 무시하고 이 `README.md`만 유지합니다. 데이터는 각자 환경에서 내려받아 배치하세요.

## 권장 하위 구조

```
dataset/
├── woodscape/        # WoodScape 벤치마크 (분석용)
├── synwoodscape/     # SynWoodScape 벤치마크 (분석용)
├── nuscenes_mini/    # nuScenes v1.0-mini (3D 박스 라벨 포맷 참고용)
└── custom/           # 자체 구축 BEV 데이터셋 (Fisheye 4-cam)
```

## 현재 보유

- `WoodScape_ICCV19.tar.gz` — WoodScape 원본 아카이브 (약 41GB). 압축 해제 후 `woodscape/` 아래에 배치 예정.
- `nuscenes_mini/` — nuScenes `v1.0-mini` (6-cam + LIDAR + radar, `v1.0-mini/` 아래 annotation JSON 포함). WoodScape에 없는 **3D bounding box 라벨 포맷** 참고용 (`docs/dataset_analysis/woodscape_analysis.md` §8 참고).
