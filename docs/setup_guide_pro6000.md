# 환경 세팅 가이드 (RTX PRO 6000 Blackwell 기준)

> 작성일: 2026-07-29
> 대상 서버: ubuntu (GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition)
> 목적: 현재 사용 중인 환경의 정본 세팅 문서. 실제 설치·검증을 마친 결과만 기록.

> 📌 이전 서버(RTX 3080) 기준 문서는 [`setup_guide.md`](setup_guide.md)에 이력으로 남겨두었다.
> 두 문서의 가장 큰 차이는 **PyTorch CUDA 빌드(cu118 → cu128)** 와 **mmcv 계열 미설치**다.

---

## 1. 하드웨어

| 항목 | 값 | 학습에 주는 영향 |
|---|---|---|
| GPU | RTX PRO 6000 Blackwell Max-Q | **compute capability 12.0 (sm_120)** |
| VRAM | 97,887 MiB (약 96GB) | 3080(10GB)의 9.5배 → **batch size 제약 사실상 해소** |
| 드라이버 | 595.84 | CUDA 12.8 / 13.x wheel 모두 수용 |
| CPU / RAM | 32코어 / 250GB | DataLoader `num_workers` 여유 |
| `/dev/shm` | 126GB | worker 공유메모리 부족 걱정 없음 |
| 디스크 | `/` 937G, **`/data` 3.7T** | 데이터셋은 `/data`에 두고 symlink |

확인 명령:
```bash
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
nproc && free -g && df -h
```

---

## 2. 최종 확정 버전

| 패키지 | 버전 | 비고 |
|---|---|---|
| conda 환경 | **`bev-chamdog`** | 이전 서버의 `mmdet3d` 라는 이름을 버렸다 |
| Python | 3.11 | 신규 torch가 3.9를 지원하지 않는 방향. 3.11이 wheel 커버리지 최대 |
| PyTorch | **2.7.0+cu128** | `torch.version.cuda == 12.8` |
| torchvision | 2.7.0 대응 cu128 빌드 | 정확한 값은 `docs/env/pip-freeze-*.txt` 참조 |
| NumPy | 1.26.4 | **2.0 미만 유지** (아래 §2-1) |
| opencv-python | 4.11.0.86 | numpy<2 제약의 결과 (아래 §2-1) |
| scipy | 1.17.1 | |
| efficientnet_pytorch | 0.7.1 | `nets/segnet.py` 최상단 import → 필수 |
| einops | 0.8.2 | |
| fire | 0.7.1 | |
| tensorboard / tensorboardX | 2.21.0 / 2.6.5 | protobuf 7.35.1로 동작 (런타임 검증 완료) |
| matplotlib | 3.11.1 | |
| scikit-image / scikit-learn | 0.26.0 / 1.9.0 | |
| imageio | 2.37.4 | |
| **mmcv / mmdet / mmdet3d** | **미설치** | 아래 §2-2 |

설치 목록의 근거와 제외 항목은 [`../requirements.txt`](../requirements.txt) 주석에 정리되어 있다.

### 2-1. NumPy를 계속 `<2`로 두는 이유

`simple_bev` 코드를 전수 검사한 결과 numpy 1.x 전용 API(`np.float`, `np.bool` 등) 사용은 **0건**이다. 즉 numpy 2도 기술적으로 가능하다. 그래도 1.26.4를 유지하는 이유는, `third_party/`는 수정 금지 규칙이라 문제가 생기면 우회밖에 못 하기 때문이다. **하드 제약이 아니라 기본값**이며, 훗날 어떤 패키지가 numpy 2를 요구하면 풀어도 된다.

이 제약의 실제 비용은 opencv다. opencv-python 4.12+가 `numpy>=2`를 요구하므로 pip가 **4.11.0.86**까지 내려온다. 이전 3080 서버와 같은 버전이라 재현성 면에서는 오히려 유리하다.

### 2-2. mmcv / mmdet / mmdet3d를 설치하지 않는 이유

- mmcv 2.1.0의 prebuilt wheel은 **torch 2.1까지만** 존재한다. torch 2.7 + CUDA 12.8에서는 소스 컴파일이 필요하고 sm_120 대응도 보장되지 않는다.
- 현재 학습 경로(Simple-BEV)는 mmdet3d에 의존하지 않으므로 애초에 불필요하다.
- `mmdetection3d/` submodule은 코드로만 남겨두고 설치하지 않는다. 이후 3D 검출로 확장할 때는 **별도 conda 환경으로 분리**한다.

---

## 3. 설치 절차

### 0) 저장소 클론

```bash
git clone --recursive https://github.com/editdiary/bev-chamdog.git
cd bev-chamdog && git checkout develop
git submodule update --init --recursive
```

`--recursive` 없이 클론하면 `third_party/`와 `mmdetection3d/`가 빈 폴더로 받아진다.

### 1) conda 환경

```bash
conda create -n bev-chamdog python=3.11 -y
conda activate bev-chamdog
```

### 2) PyTorch (🔴 반드시 cu128)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

설치 로그의 파일명이 **`torch-2.7.0+cu128-...whl`** 인지 눈으로 확인한다. 여기서 `cu118`이 보이면 `~/.config/pip/pip.conf`나 `PIP_INDEX_URL` 환경변수가 index를 덮어쓰고 있는 것이므로 먼저 치운다.

### 3) 🔴 즉시 GPU 스모크 테스트 (실패하면 이후 단계 무의미)

```bash
python - <<'EOF'
import torch
print('torch      :', torch.__version__, '| cuda:', torch.version.cuda)
print('arch list  :', torch.cuda.get_arch_list())      # 'sm_120'이 있어야 함
print('device     :', torch.cuda.get_device_name(0))
print('capability :', torch.cuda.get_device_capability(0))
x = torch.randn(4096, 4096, device='cuda')
print('matmul     :', (x @ x).sum().item())            # sm_120 커널 실제 실행
import torch.nn as nn
conv = nn.Conv2d(3, 64, 3, padding=1).cuda()
print('conv       :', conv(torch.randn(2,3,224,224,device='cuda')).shape)
with torch.autocast('cuda', dtype=torch.bfloat16):
    print('bf16 amp   :', (x @ x).dtype)
EOF
```

통과 기준: `cuda: 12.8` / `arch list`에 `sm_120` 포함 / 호환성 경고 없음 / 세 연산 모두 성공.

### 4) 의존성 설치

```bash
pip install -c constraints.txt -r requirements.txt
pip check                                   # → No broken requirements found
mkdir -p docs/env && pip freeze > docs/env/pip-freeze-pro6000-$(date +%Y%m%d).txt
```

### 5) Simple-BEV 통합 검증

```bash
python - <<'EOF'
import sys, os, tempfile, warnings, numpy as np, torchvision
warnings.simplefilter('always')

# (a) 레거시 pretrained= 인자 + ImageNet 가중치 다운로드
m = torchvision.models.resnet101(pretrained=True)
print('resnet101 params :', sum(p.numel() for p in m.parameters()))   # 44,549,160

# (b) tensorboardX + protobuf 7.x 런타임
from tensorboardX import SummaryWriter
d = tempfile.mkdtemp()
w = SummaryWriter(d, max_queue=10, flush_secs=1)
w.add_scalar('test/loss', 0.5, 1)
w.add_image('test/img', np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8), 1)
w.close()
print('tensorboardX OK  :', os.listdir(d))

# (c) simple_bev 모듈 import (skimage.color + matplotlib 경로 포함)
sys.path.insert(0, 'third_party/models/simple_bev')
import nets.segnet, utils.improc, utils.geom, utils.vox, utils.basic
print('simple_bev import OK')
EOF
```

### 6) 데이터셋 배치

데이터셋은 루트 파티션이 아니라 **`/data`(3.7T)** 에 두고 symlink를 건다. 자체 데이터셋과 체크포인트가 쌓일 것을 감안한 배치다.

```bash
sudo mkdir -p /data/datasets && sudo chown $USER:$USER /data/datasets
ln -s /data/datasets/synwoodscape dataset/synwoodscape
```

이전 서버에서 전송 (61GB, **tmux 안에서** 실행):
```bash
rsync -avhP --partial \
  <이전서버>:/home/leedh/home/bev-chamdog/dataset/synwoodscape/ \
  /data/datasets/synwoodscape/
```

전송 후 양쪽에서 아래를 실행해 숫자를 비교한다:
```bash
find /path/to/synwoodscape -type f | wc -l && du -sb /path/to/synwoodscape
```

**옮기지 않는 것:**

| 대상 | 용량 | 판단 |
|---|---|---|
| `dataset/woodscape/` | 43G | 분석 완료·학습 미사용. 캘리브레이션 규약과 코드는 `third_party/datasets/WoodScape/` submodule에 있음 |
| `dataset/nuscenes_mini/` | 5.1G | 포맷 참고용. 필요해지면 재다운로드 |
| 이전 conda 환경 | 7.1G | 새로 만든다 |

---

## 4. 트러블슈팅 기록

### 4-1) 🔴 cu118 wheel을 받아 sm_120에서 커널 실행 실패

- **증상**
  ```
  torch : 2.7.0+cu118
  UserWarning: ... with CUDA capability sm_120 is not compatible with the current PyTorch installation.
  The current PyTorch install supports CUDA capabilities ... sm_90
  RuntimeError: CUDA error: no kernel image is available for execution on the device
  ```
- **원인**: PyTorch 버전(2.7.0)은 맞았으나 **CUDA 빌드가 cu118**이었다. cu118 wheel에는 sm_90까지의 커널만 포함된다. 이전 서버 문서의 `--index-url .../whl/cu118` 명령을 그대로 쓴 것이 원인.
- **함정**: `torch.cuda.is_available()`이 **True**로 나오고 `get_device_name()`·`get_device_capability()`도 정상 출력된다. 실패는 **실제 커널 실행 시점**에만 드러난다.
- **해결**
  ```bash
  pip uninstall -y torch torchvision
  pip freeze | grep -i "^nvidia-.*-cu11" | xargs -r pip uninstall -y   # CUDA 11 런타임 잔여물 제거
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  ```
- **교훈**: 스모크 테스트에 `torch.cuda.get_arch_list()`를 넣으면 연산을 실행해 죽기 전에 알 수 있다. 그리고 검증은 `is_available()`이 아니라 **device 위 matmul/conv 실행**까지 해야 한다.

### 4-2) 원본 `simple_bev/requirements.txt` 설치 불가

- **증상**: `matplotlib==3.5.1`, `scikit-learn==1.1.2`, `protobuf==3.19.4` 등이 Python 3.11에서 wheel 없음.
- **원인**: 해당 파일은 Python 3.7~3.8 / torch 1.12 시절 기준.
- **해결**: 실제 import를 추적해 [`../requirements.txt`](../requirements.txt)로 재구성. 최신 버전으로 올려도 코드 수정이 필요 없는지 아래를 전수 확인했다.
  - matplotlib 3.9에서 제거된 `cm.get_cmap` 계열 → **사용 없음**
  - scikit-image → `skimage.color`만 사용 (안정 API)
  - imageio → 이미 `imageio.v2` 명시적 사용
  - numpy 레거시 API → **0건**

### 4-3) `numpy` 다운그레이드 시 opencv 버전이 내려감

- **증상**: `opencv-python<5` 설치 시 pip가 4.14 → 4.11.0.86까지 backtrack.
- **원인**: opencv-python 4.12+가 `numpy>=2`를 요구.
- **판단**: 정상 동작. 이전 서버와 동일 버전이라 그대로 수용한다.

---

## 5. 신규 torch에서 알고 있어야 할 호환 지점

`third_party/`는 수정 금지이므로, 문제가 생기면 `projects/`의 래퍼에서 처리한다.

| 지점 | 위치 | 현재 상태 |
|---|---|---|
| `torchvision.models.resnet101(pretrained=True)` — 0.13에서 deprecated된 인자 | `nets/segnet.py:163` (기본 `encoder_type='res101'`) | **경고만 뜨고 동작.** `weights=ResNet101_Weights.IMAGENET1K_V1`과 동일하게 처리됨 |
| ImageNet 사전학습 가중치 다운로드 | `~/.cache/torch/hub/checkpoints/resnet101-63fe2227.pth` | **정상 다운로드 확인.** 외부망이 막힌 환경이라면 미리 받아 배치해야 함 |
| `torch.load`의 `weights_only=True` 기본값 (torch 2.6+) | `saverloader.py:47,62` | **위험 낮음.** `save()`가 저장하는 건 optimizer/model/scheduler state_dict 뿐이고 내용물이 tensor·dict·list·int·float 이라 허용 타입에 들어간다. 다만 **실제 체크포인트를 로딩하는 시점에 확인 필요** |
| `torch.meshgrid`의 `indexing` 인자 미지정 | `nets/bevformernet*.py`, `nets/tiimnet.py` | 경고만. 해당 모델은 현재 미사용 |

---

## 6. 학습 설정에 주는 영향 (10GB → 96GB)

VRAM이 9.5배가 되면서 이전 서버의 제약 대응이 대부분 불필요해진다.

- **batch size 축소 / gradient accumulation 회피 전략이 불필요하다.** Simple-BEV `train_nuscenes.py` 기본값은 `batch_size=8`, `grad_acc=5`(유효 배치 40), `res_scale=2`(→ 448×800), `encoder_type='res101'`이다. 96GB라면 `grad_acc=1`로 두고 `batch_size`를 키워 같은 유효 배치를 한 번에 처리하는 쪽이 빠르다.
  단, **최대 batch size는 추측하지 말고 실측으로 정한다.**
- **병목이 GPU에서 데이터 로딩으로 이동한다.** 32코어이므로 `num_workers=8~16`, `pin_memory=True`, `persistent_workers=True`부터 시작한다. `/dev/shm`이 126GB라 worker 공유메모리는 문제되지 않는다.
- **bf16 AMP를 검토할 가치가 생긴다.** 3080에서는 실익이 적었으나 Blackwell에서는 다르다. 스모크 테스트에서 `torch.autocast('cuda', dtype=torch.bfloat16)` 동작을 확인했다.
- **OOM 걱정보다 GPU 활용률 저하를 경계해야 한다.** 첫 학습에서는 `nvidia-smi`로 GPU-Util을 관찰하고, 낮다면 batch size·`num_workers`를 올린다.

---

## 7. 검증 체크리스트

| 항목 | 방법 | 상태 |
|---|---|---|
| GPU 인식 | `nvidia-smi`, `get_device_capability()` | ✅ sm_120 (12, 0) |
| CUDA 빌드 정합 | `torch.version.cuda`, `get_arch_list()` | ✅ 12.8 / sm_120 포함 |
| sm_120 커널 실행 | device 위 matmul + conv | ✅ |
| bf16 autocast | `torch.autocast` | ✅ |
| 의존성 정합성 | `pip check` | ✅ No broken requirements found |
| 사전학습 encoder | `resnet101(pretrained=True)` | ✅ 44,549,160 params |
| 로깅 스택 | `tensorboardX` + protobuf 7.x 실제 기록 | ✅ event file 생성 |
| simple_bev import | `nets.segnet`, `utils.*` | ✅ |
| 데이터셋 배치 | 파일 수·바이트 대조 | ⬜ 전송 진행 중 |
| 학습 1 iteration | Simple-BEV 실제 실행 | ⬜ 미실행 |
