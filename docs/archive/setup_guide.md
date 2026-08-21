# 환경 세팅 가이드 (RTX 3080 기준) — 📦 이력 문서

> 🚨 **이 문서는 더 이상 현재 환경이 아닙니다.**
> 서버를 RTX PRO 6000 Blackwell 환경으로 이전했습니다. **현재 정본은 [`../setup_guide_pro6000.md`](../setup_guide_pro6000.md)** 입니다.
>
> 특히 아래 §2-2의 `pip install ... --index-url .../whl/cu118` 명령을 **새 서버에서 그대로 쓰면 안 됩니다.**
> cu118 wheel에는 sm_90까지의 커널만 들어 있어 sm_120(Blackwell)에서
> `CUDA error: no kernel image is available for execution on the device`로 실패합니다.
> (실제로 이전 과정에서 이 함정을 밟았습니다 → `../setup_guide_pro6000.md` §4-1)
>
> 이 문서는 3080 시절의 구축 과정·트러블슈팅 이력으로만 보존합니다.

> 작성일: 2026-07-13
> 대상 서버: agtechresearch (GPU: NVIDIA GeForce RTX 3080, VRAM 10GB)
> 목적: 개발 환경 구축 과정, 트러블슈팅, 최종 검증 결과 기록

> ⚠️ **이 문서는 Phase 0에서 mmdet3d 스택을 구축한 과정의 기록입니다.**
> 현재 학습 경로인 **Simple-BEV는 mmdet3d에 의존하지 않으므로**, 아래에서 실제로 필요한 것은
> **Python 3.9 / PyTorch 2.1.0+cu118 / torchvision 0.16.0+cu118 / numpy<2 / opencv<5** 까지입니다
> (설치 단계 0~3, 8). mmengine·mmcv·mmdet·mmdet3d(단계 4~7)는 같은 conda 환경에 설치되어 있으나
> 현재 사용하지 않습니다. 이후 3D 검출로 확장할 때 다시 필요합니다. (→ `ROADMAP.md`)
>
> conda 환경 이름이 `mmdet3d`인 것도 이때 붙은 것이며, 현재 학습 경로와는 무관합니다.

---

## 1. 최종 확정 버전

| 패키지 | 버전 | 비고 |
|---|---|---|
| OS / GPU | Ubuntu, RTX 3080 | Driver 550.163.01, CUDA(드라이버 기준) 12.4 |
| Python | 3.9 | conda 환경 `mmdet3d` |
| PyTorch | 2.1.0+cu118 | pip 설치 |
| torchvision | 0.16.0+cu118 | pip 설치 |
| NumPy | 1.26.4 | **반드시 2.0 미만 고정 필요** |
| opencv-python | 4.11.0 | numpy<2와 호환되는 버전 |
| mmengine | 0.10.7 | |
| mmcv | 2.1.0 | prebuilt wheel 사용 |
| mmdet | 3.2.0 | mmdet3d 1.4.0과 호환되는 버전으로 고정 (3.3.0은 비호환) |
| mmdet3d | 1.4.0 | GitHub 소스 클론, editable 설치 |

설치 확인 명령:
```bash
python -c "
import torch, torchvision, mmcv, mmdet, mmdet3d, mmengine, numpy, cv2
print('torch       :', torch.__version__, '| cuda:', torch.version.cuda, '| available:', torch.cuda.is_available())
print('torchvision :', torchvision.__version__)
print('numpy       :', numpy.__version__)
print('opencv      :', cv2.__version__)
print('mmengine    :', mmengine.__version__)
print('mmcv        :', mmcv.__version__)
print('mmdet       :', mmdet.__version__)
print('mmdet3d     :', mmdet3d.__version__)
"
```

---

## 2. 설치 단계

### 0) 드라이버/GPU 확인
```bash
nvidia-smi
```

### 1) conda 가상환경 생성
```bash
conda create -n mmdet3d python=3.9 -y
conda activate mmdet3d
```

### 2) PyTorch + torchvision 설치
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```
- `--index-url`(전용 지정)을 사용해 PyPI와 혼동 없이 정확한 wheel을 받도록 함.
- PyTorch wheel은 자체 CUDA 런타임을 내장하므로, 시스템에 해당 CUDA Toolkit을 별도로 설치할 필요 없음. GPU 드라이버가 해당 버전 이상을 지원하기만 하면 됨.

### 3) NumPy 버전 고정
```bash
pip install "numpy<2"
```
- PyTorch 2.1.0은 NumPy 1.x ABI 기준으로 빌드되어 있어 2.x와 충돌 발생. mmcv/mmdet 계열 대부분이 아직 NumPy 2.x 완전 지원 전이라 반드시 선행 고정 필요.

### 4) openmim + mmengine
```bash
pip install -U openmim
mim install mmengine
```

### 5) mmcv 설치 (prebuilt wheel 확인)
```bash
mim install "mmcv==2.1.0"
```
- 설치 로그에서 `.whl`로 받아지는지 확인 (`.tar.gz`면 소스 컴파일 필요, 실패 가능성 높음).

### 6) mmdetection 설치 (버전 고정)
```bash
pip install "mmdet==3.2.0"
```
- mmdet 최신(3.3.0)은 mmdet3d 1.4.0과 `mmdet<3.3.0` 요건 충돌 → 3.2.0으로 고정.

### 7) mmdetection3d 설치
```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.4.0
pip install -v -e .
```

### 8) 의존성 재점검 (numpy/opencv 재충돌 방지)
```bash
cat > constraints.txt << 'EOF'
numpy<2
opencv-python<5
opencv-python-headless<5
EOF
pip install -r constraints.txt --force-reinstall
pip check
```
- mmcv/mmdet/mmdet3d 설치 과정에서 의존성으로 numpy/opencv가 다시 2.x로 끌어올려지는 경우가 반복적으로 발생. constraints 파일로 강제 고정하여 재발 방지.

---

## 3. 트러블슈팅 기록

### 3-1) NumPy 2.x 호환성 경고
- **증상**: `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.2...`
- **원인**: PyTorch 2.1.0이 NumPy 1.x ABI로 빌드됨.
- **해결**: `pip install "numpy<2"`로 고정. 이후 mmcv/mmdet 설치 때마다 재확인 필요(반복 발생).

### 3-2) mmcv/mmdet 버전 상호 비호환
- **증상**: `mim install "mmcv>=2.0.0rc4"` 실행 시 mmcv 2.2.0이 설치되었으나, 이후 mmdet 3.3.0이 `mmcv<2.2.0` 요구하여 충돌.
- **원인**: 버전 범위를 열어두면(`>=x.x.x`) pip/mim이 그때그때 최신 버전을 선택해 서로 호환 안 되는 조합이 만들어질 수 있음.
- **해결**: mmcv 2.1.0, mmdet 3.2.0으로 명시적 고정.

### 3-3) mmdet 버전 변경 시 numpy가 다시 2.x로 자동 업그레이드
- **증상**: `pip install "mmdet==3.2.0"` 재실행 시 numpy가 2.0.2로 다시 변경됨.
- **원인**: mmdet의 하위 의존성이 조용히 numpy 최신 버전을 재요구.
- **해결**: `constraints.txt`(numpy<2, opencv-python<5 등)를 만들어 `pip install -c constraints.txt`로 모든 설치에 제약 강제 적용.

---

## 4. 최종 검증 결과

| 검증 항목 | 방법 | 결과 |
|---|---|---|
| 패키지 의존성 정합성 | `pip check` | ✅ No broken requirements found |
| GPU 인식 | `torch.cuda.is_available()` | ✅ True (RTX 3080) |
| 추론 파이프라인 | `demo/pcd_demo.py` 실행 | ✅ `outputs/preds/000008.json` 정상 생성 (차량 10개 검출) |
| CUDA 커스텀 연산 | `mmcv.ops.Voxelization` 직접 호출 | ✅ GPU 위에서 정상 실행 (voxel shape 확인) |
| 모델 빌드 | Config → `MODELS.build()` (VoxelNet) | ✅ 정상 빌드 |
| GPU 사양 확인 | `torch.cuda.get_device_properties` | RTX 3080, VRAM 약 10.5GB |

### 검증에 사용한 핵심 명령어
```bash
# 1) 의존성 확인
pip check

# 2) CUDA 연산 테스트
python -c "
import torch
from mmcv.ops import Voxelization
points = torch.rand(1000, 4).cuda()
voxel_layer = Voxelization(voxel_size=[0.16, 0.16, 4], point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1], max_num_points=32, max_voxels=16000)
voxels, coors, num_points = voxel_layer(points)
print('Voxelization on GPU: OK, voxel shape =', voxels.shape)
"

# 3) 모델 빌드 테스트 (register_all_modules() 필수)
python -c "
from mmengine import Config
from mmdet3d.utils import register_all_modules
from mmdet3d.registry import MODELS
register_all_modules()
cfg = Config.fromfile('pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py')
model = MODELS.build(cfg.model)
print('Model build from config: OK')
print(model.__class__.__name__)
"

# 4) 추론 데모
mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest .
python demo/pcd_demo.py demo/data/kitti/000008.bin pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth
```

---

## 6. 향후 참고 사항

- **GPU 메모리 제약**: 3080은 VRAM 10GB로, 공개 BEV 모델의 기본 batch size는 단일 소비자용 GPU에 맞지 않음. Simple-BEV `train_nuscenes.py`의 기본값은 `batch_size=8`, `grad_acc=5`(유효 배치 40)이고 저장소의 `train.sh` 예시는 `batch_size=1`이다. mmdet3d 기본 config도 대개 다중 GPU 기준. 단일 3080으로 학습 시 `batch_size`를 낮추고 `grad_acc`를 키우는 조합이 필요하며, **gradient accumulation은 Simple-BEV에 이미 내장되어 있어 따로 구현할 필요가 없다.** 첫 학습 실행 시 OOM(Out of Memory) 발생 가능성 있음.
- **버전 관리 원칙**: mmcv/mmdet/mmdet3d/numpy/opencv는 버전 범위(`>=`)보다 정확한 버전을 명시적으로 고정하는 것을 권장. 새 패키지 설치/업데이트 후에는 항상 `pip check`로 재검증.
- **재현성**: 다른 서버에 동일 환경을 구축할 경우, 본 문서의 "1. 최종 확정 버전" 표와 "3. 설치 단계"를 그대로 따르면 동일하게 재현 가능.