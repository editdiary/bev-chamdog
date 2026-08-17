# BEV Free-space Task 재정의 구현 계획 (Phase 0–3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** BEV 학습의 평가 계층을 `free`(= occupancy ∧ visibility) 중심으로 교체하고, 그 지표로 현행 2-head 기준선을 고정한 뒤 3-class 단일 head로 전환한다.

**Architecture:** 새 지표는 head 구조와 데이터셋에 무관한 신규 모듈 3개(`free_space.py`, `polar.py`, `free_space_metrics.py`)에 둔다. 기존 `two_head_metrics.py`는 2-head 전용 `run_batch`·로깅을 계속 담당하며, 신규 모듈을 호출해 지표만 확장한다. 3-class head는 Simple-BEV의 최종 1×1 conv만 3채널로 바꾸고 encoder·lifting·decoder trunk는 그대로 재사용한다.

**Tech Stack:** Python 3.11 / PyTorch 2.7.0+cu128 / numpy<2 / scipy / pytest / tensorboardX / Fire. 모델은 `third_party/models/simple_bev`의 `Segnet` (submodule, **수정 금지**).

**설계 정본:** [`docs/superpowers/specs/2026-08-17-bev-free-space-task-redefinition-design.md`](../specs/2026-08-17-bev-free-space-task-redefinition-design.md). 이 계획의 §·M 번호는 전부 그 스펙을 가리킨다.

## Global Constraints

- conda 환경 **`bev-chamdog`** (Python 3.11). 모든 명령은 `conda activate bev-chamdog` 이후 저장소 루트에서 실행한다.
- **`third_party/`와 `mmdetection3d/`는 git submodule — 직접 수정 금지.** Simple-BEV 동작을 바꿔야 하면 `projects/`에서 래핑한다.
- **mmcv / mmdet / mmdet3d는 설치하지 않는다.**
- 커스텀 코드는 `projects/`(모듈), `configs/`(config), `tools/`(스크립트)에만 둔다.
- **커밋 메시지는 영어.** 코드 주석·문서는 한국어(주변 코드 스타일을 따른다 — 기존 `projects/common/*.py`는 한국어 docstring에 "왜"를 적는다).
- **`merge`와 `push`는 절대 수행하지 않는다.** 작업 브랜치는 **`feat/bev-free-space-task`**.
- 전체 테스트는 `python -m pytest tests/ -q` 로 돌린다. 계획 시작 시점 기준선은 **162 passed**.
- GPU 학습은 `CUDA_VISIBLE_DEVICES=1` (RTX PRO 6000, 96GB). batch size 축소나 gradient accumulation은 불필요하다.
- 실측 없이 숫자를 적지 않는다. 게이트 태스크의 결과는 반드시 명령 출력을 문서에 붙여 남긴다.

## 스펙 대비 의도적 편차 (하나)

스펙 §10 Phase 0은 `projects/common/two_head_metrics.py`를 `bev_occupancy_metrics.py`로 **개칭**하라고 적었다. 이 계획은 **개칭을 Phase 3(Task 16)으로 미룬다.**

이유: Phase 2의 기준선은 2-head 구성 그대로 뽑아야 하므로 Phase 0–2 내내 두 정식화가 공존한다. 그 기간에 개칭하면 두 트레이너·두 테스트 파일에 걸친 큰 rename diff가 정작 중요한 지표 변경 diff를 덮는다. 그리고 새 지표는 어차피 신규 모듈에 들어가므로 개칭이 사주는 것이 없다. `two_head_metrics.py`라는 이름은 그 안에 남는 것(2-head `run_batch`와 2-head 로깅)에 대해 Phase 3까지 계속 정확하다.

---

# Phase 0 — 지표 계층 신설 (모델 변경 없음)

## File Structure (Phase 0)

| 파일 | 책임 |
|---|---|
| `projects/common/free_space.py` (신규) | `free`/`occupied`/`unknown` 4-way 분해와 분할 불변식. 데이터셋·head 무관 |
| `projects/common/polar.py` (신규) | 방위각 광선 인덱스 캐시와 `r(θ)` 추출. 격자 기하만 안다 |
| `projects/common/free_space_metrics.py` (신규) | M1 `iou_free`, M2 `fatal_rate`, M2b `free_miss_rate`, M3 `range_error`, M4 거리 링 분해 |
| `tests/common/test_free_space.py` (신규) | Task 1 |
| `tests/common/test_free_space_metrics.py` (신규) | Task 2, 4, 5 |
| `tests/common/test_polar.py` (신규) | Task 3 |
| `tests/common/test_label_integrity.py` (신규) | Task 6. 데이터셋이 있을 때만 도는 gated 테스트 |

분해를 `free_space.py`에, 기하를 `polar.py`에, 집계를 `free_space_metrics.py`에 나눈 이유: 분해는 학습 루프·시각화·재채점 도구가 전부 쓰고, 기하는 격자가 바뀔 때만 바뀌며, 집계는 지표가 추가될 때 자란다. 세 축이 서로 다른 이유로 변하므로 같은 파일에 두면 안 된다.

---

### Task 1: 4-way 분해와 분할 불변식

**Files:**
- Create: `projects/common/free_space.py`
- Test: `tests/common/test_free_space.py`

**Interfaces:**
- Consumes: 없음 (torch만 사용)
- Produces:
  - `FREE: int = 1`, `OCCUPIED: int = 2`, `UNKNOWN: int = 0` (3-class 라벨 값)
  - `decompose(occ, vis, valid) -> dict[str, torch.Tensor]` — 키 `"free"`, `"occupied"`, `"unknown"`, 전부 `(B, 1, H, W)` bool
  - `decompose_from_class_index(class_index, valid) -> dict[str, torch.Tensor]` — `class_index`는 `(B, 1, H, W)` long
  - `to_class_index(parts) -> torch.Tensor` — `(B, 1, H, W)` long, `valid` 밖은 `UNKNOWN`
  - `partition_defect_count(parts, valid) -> int` — 0이면 정확한 분할

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/common/test_free_space.py`:

```python
import pytest
import torch

from projects.common.free_space import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    decompose,
    decompose_from_class_index,
    partition_defect_count,
    to_class_index,
)


def _grid(values):
    """(H, W) 리스트 -> (1, 1, H, W) float 텐서."""
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def test_decompose_splits_valid_cells_into_exactly_three_parts():
    occ = _grid([[1, 1], [0, 0]])
    vis = _grid([[1, 0], [1, 0]])
    valid = _grid([[1, 1], [1, 1]])

    parts = decompose(occ, vis, valid)

    assert parts["free"].squeeze().tolist() == [[True, False], [False, False]]
    assert parts["occupied"].squeeze().tolist() == [[False, False], [True, False]]
    # unknown은 vis=0인 셀 전부 -- occ 값과 무관하다
    assert parts["unknown"].squeeze().tolist() == [[False, True], [False, True]]
    assert partition_defect_count(parts, valid.bool()) == 0


def test_invalid_cells_belong_to_no_part():
    """valid=0은 unknown이 아니다 -- 배포 때 존재하지 않는 수집 아티팩트라 gradient도 지표도 받지 않는다."""
    occ = _grid([[1, 1]])
    vis = _grid([[0, 0]])
    valid = _grid([[1, 0]])

    parts = decompose(occ, vis, valid)

    assert parts["unknown"].squeeze().tolist() == [True, False]
    assert partition_defect_count(parts, valid.bool()) == 0


def test_class_index_roundtrip_preserves_the_partition():
    occ = _grid([[1, 0], [1, 0]])
    vis = _grid([[1, 1], [0, 0]])
    valid = _grid([[1, 1], [1, 1]])
    parts = decompose(occ, vis, valid)

    index = to_class_index(parts)
    assert index.squeeze().tolist() == [[FREE, OCCUPIED], [UNKNOWN, UNKNOWN]]

    restored = decompose_from_class_index(index, valid.bool())
    for key in ("free", "occupied", "unknown"):
        assert torch.equal(restored[key], parts[key])


def test_partition_defect_count_catches_overlap():
    valid = torch.ones((1, 1, 1, 2), dtype=torch.bool)
    broken = {
        "free": torch.tensor([[[[True, False]]]]),
        "occupied": torch.tensor([[[[True, False]]]]),   # free와 겹친다
        "unknown": torch.tensor([[[[False, True]]]]),
    }

    assert partition_defect_count(broken, valid) == 1
```

- [ ] **Step 2: 실패를 확인한다**

Run: `python -m pytest tests/common/test_free_space.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'projects.common.free_space'`

- [ ] **Step 3: 최소 구현을 쓴다**

`projects/common/free_space.py`:

```python
"""BEV 라벨의 4-way 분해 -- `free` / `occupied` / `unknown`, 그리고 `valid` 밖.

왜 별도 모듈인가: 이 분해를 학습 루프·시각화·체크포인트 재채점 도구가 전부 쓰는데,
각자 `occ`/`vis`/`valid`를 조합하게 두면 정의가 조용히 갈라진다. 실제로 예전에
`pos_weight`가 마스크를 무시하고 원본 라벨을 세는 바람에 클래스 보정이 통째로 어긋난 적이
있다. 분해의 단일 출처를 여기 둔다.

정의 (`valid`가 1인 셀 안에서 완전 분할):
    free     = occ  &  vis      통과 가능하고 실제로 관측된 곳
    occupied = ~occ &  vis      광선이 멈춘 표면 (두께 1셀이 정상이다)
    unknown  = ~vis             그 너머
`valid=0`은 셋 중 어디에도 속하지 않는다 -- 수집 아티팩트(카트 손잡이·미는 사람)라
배포 때 존재하지 않으므로 `unknown`으로 두면 "후방은 항상 가려져 있다"를 학습한다.
"""
import torch

UNKNOWN = 0
FREE = 1
OCCUPIED = 2

PART_NAMES = ("free", "occupied", "unknown")


def _as_bool(tensor: torch.Tensor) -> torch.Tensor:
    """확률·0/1 float·bool을 모두 받는다 -- 예측(sigmoid 출력)과 GT가 같은 함수를 타야 한다."""
    return tensor if tensor.dtype == torch.bool else tensor > 0.5


def decompose(occ, vis, valid) -> dict:
    """`(B, 1, H, W)` occupancy / visibility / valid -> 세 bool 마스크."""
    occ_b, vis_b, valid_b = _as_bool(occ), _as_bool(vis), _as_bool(valid)
    return {
        "free": occ_b & vis_b & valid_b,
        "occupied": (~occ_b) & vis_b & valid_b,
        "unknown": (~vis_b) & valid_b,
    }


def to_class_index(parts: dict) -> torch.Tensor:
    """분해 -> `(B, 1, H, W)` long. `valid` 밖은 `UNKNOWN`으로 떨어진다(3-class CE에서 마스킹됨)."""
    index = torch.full_like(parts["free"], UNKNOWN, dtype=torch.long)
    index[parts["free"]] = FREE
    index[parts["occupied"]] = OCCUPIED
    return index


def decompose_from_class_index(class_index, valid) -> dict:
    """3-class 출력(argmax) -> 분해. `decompose`와 같은 계약을 돌려준다."""
    valid_b = _as_bool(valid)
    return {
        "free": (class_index == FREE) & valid_b,
        "occupied": (class_index == OCCUPIED) & valid_b,
        "unknown": (class_index == UNKNOWN) & valid_b,
    }


def partition_defect_count(parts: dict, valid: torch.Tensor) -> int:
    """분할이 깨진 셀 수. 0이 아니면 배선이 틀린 것이므로 학습 루프에서 assert한다."""
    stacked = torch.stack([parts[name] for name in PART_NAMES], dim=0).long()
    covered = stacked.sum(dim=0)
    return int((covered != _as_bool(valid).long()).sum().item())
```

- [ ] **Step 4: 통과를 확인한다**

Run: `python -m pytest tests/common/test_free_space.py -v`
Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add projects/common/free_space.py tests/common/test_free_space.py
git commit -m "Add the free/occupied/unknown decomposition as a single source"
```

---

### Task 2: 면적 지표 M1 / M2 / M2b

**Files:**
- Create: `projects/common/free_space_metrics.py`
- Test: `tests/common/test_free_space_metrics.py`

**Interfaces:**
- Consumes: `projects.common.free_space.partition_defect_count` (Task 1)
- Produces:
  - `iou_free(free_pred, free_gt, valid) -> tuple[float, int]` — (per-sample IoU 평균, 집계에 쓴 샘플 수). union이 0인 샘플은 제외한다
  - `fatal_rate(free_pred, free_gt, valid) -> tuple[float, int]` — (비율, 분모 셀 수)
  - `free_miss_rate(free_pred, free_gt, valid) -> tuple[float, int]`
  - `weighted_mean(values, weights) -> float` — batch 간 가중평균 (NaN·0-weight 항목 제외)
  - `free_metrics_from_masks(pred_free, gt_parts, valid) -> dict` — **2-head와 3-class가 공유하는 집계기.** 키: `iou_free`, `iou_free_count`, `fatal_rate`, `fatal_denom`, `free_miss_rate`, `free_miss_denom`, `partition_defects`, `pred_free`, `gt_free`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/common/test_free_space_metrics.py`:

```python
import math

import pytest
import torch

from projects.common.free_space_metrics import (
    fatal_rate,
    free_miss_rate,
    iou_free,
    weighted_mean,
)


def _mask(values):
    return torch.tensor(values, dtype=torch.bool).unsqueeze(0).unsqueeze(0)


def test_iou_free_matches_hand_computation():
    gt = _mask([[True, True], [False, False]])     # free 2칸
    pred = _mask([[True, False], [True, False]])   # free 2칸, 교집합 1칸
    valid = torch.ones_like(gt)

    value, count = iou_free(pred, gt, valid)

    assert value == pytest.approx(1 / 3)           # 교집합 1 / 합집합 3
    assert count == 1


def test_perfect_prediction_scores_one_and_zero_errors():
    gt = _mask([[True, False], [True, False]])
    valid = torch.ones_like(gt)

    assert iou_free(gt, gt, valid)[0] == pytest.approx(1.0)
    assert fatal_rate(gt, gt, valid)[0] == pytest.approx(0.0)
    assert free_miss_rate(gt, gt, valid)[0] == pytest.approx(0.0)


def test_all_free_prediction_is_penalised_by_iou_free():
    """'전부 free'는 iou_drivable을 이겼던 트리비얼 해다. iou_free는 그걸 벌해야 한다."""
    gt = _mask([[True, False, False, False]])      # 4칸 중 1칸만 free
    pred = torch.ones_like(gt)
    valid = torch.ones_like(gt)

    assert iou_free(pred, gt, valid)[0] == pytest.approx(0.25)
    assert fatal_rate(pred, gt, valid)[0] == pytest.approx(0.75)


def test_metrics_ignore_invalid_cells():
    gt = _mask([[True, False]])
    pred = _mask([[True, True]])
    valid = _mask([[True, False]])                 # 두 번째 칸은 집계에서 빠진다

    assert iou_free(pred, gt, valid)[0] == pytest.approx(1.0)
    assert fatal_rate(pred, gt, valid)[0] == pytest.approx(0.0)


def test_empty_denominator_yields_nan_and_zero_weight():
    """GT에도 예측에도 free가 없는 샘플은 IoU가 0/0이다. 0점으로 세면 지표가 왜곡된다."""
    gt = _mask([[False, False]])
    pred = _mask([[False, False]])
    valid = torch.ones_like(gt)

    value, count = iou_free(pred, gt, valid)
    assert math.isnan(value)
    assert count == 0

    rate, denom = fatal_rate(pred, gt, valid)
    assert math.isnan(rate)
    assert denom == 0


def test_weighted_mean_skips_zero_weight_entries():
    assert weighted_mean([1.0, float("nan")], [2, 0]) == pytest.approx(1.0)
    assert math.isnan(weighted_mean([1.0], [0]))
```

- [ ] **Step 2: 실패를 확인한다**

Run: `python -m pytest tests/common/test_free_space_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'projects.common.free_space_metrics'`

- [ ] **Step 3: 최소 구현을 쓴다**

`projects/common/free_space_metrics.py`:

```python
"""free-space 지표 -- M1 `iou_free`, M2 `fatal_rate`, M2b `free_miss_rate`.

왜 `iou_drivable`/`iou_obstacle`을 대신하는가: 자체 로봇 데이터에서 평가 마스크
(`vis & valid`) 안 drivable 비율이 93.5%라 "전부 drivable"이 `iou_drivable` 0.935를
받아 학습된 모델(0.891)을 이겼다. `iou_obstacle`은 두께 1셀 표면에 IoU를 적용한 것이라
한 칸 정렬 오차가 점수를 반토막 낸다. 두 head의 결합 결과인 `free`를 재면 트리비얼 해가
이기지 못한다 (모델 0.850 vs constant map 0.673 vs 전부 free ~0.17).

전부 `(B, 1, H, W)` bool 텐서만 받고 `(값, 가중치)`를 돌려준다 -- 가중치는 batch 간
가중평균용이며, 분모가 0인 batch를 0점으로 세지 않기 위해 필요하다.
"""
import math

import torch


def weighted_mean(values, weights) -> float:
    """weight가 0인 항목은 값이 NaN일 수 있으므로(= 셀 수 있는 샘플이 없던 batch) 건너뛴다."""
    total = sum(weights)
    if total <= 0:
        return float("nan")
    return sum(v * w for v, w in zip(values, weights) if w > 0) / total


def iou_free(free_pred, free_gt, valid):
    """M1. per-sample IoU의 평균과, 평균에 실제로 들어간 샘플 수.

    GT에도 예측에도 free가 없는 샘플은 union이 0이라 IoU가 정의되지 않는다. 0점으로 세면
    "free가 없는 장면을 완벽히 맞혔는데 0점"이 되어 지표가 왜곡되므로 평균에서 뺀다.
    """
    valid_b = valid.bool()
    pred, gt = free_pred.bool() & valid_b, free_gt.bool() & valid_b
    dims = list(range(1, pred.ndim))
    intersection = (pred & gt).sum(dim=dims).float()
    union = (pred | gt).sum(dim=dims).float()
    has_union = union > 0
    count = int(has_union.sum().item())
    if count == 0:
        return float("nan"), 0
    return float((intersection[has_union] / union[has_union]).mean().item()), count


def _rate(numerator_mask, denominator_mask):
    denominator = int(denominator_mask.sum().item())
    if denominator == 0:
        return float("nan"), 0
    return int(numerator_mask.sum().item()) / denominator, denominator


def fatal_rate(free_pred, free_gt, valid):
    """M2. "갈 수 있다고 믿은 곳 중 틀린 비율" = 1 - precision(free).

    분모를 `|free_pred|`로 잡는 이유: `|~free_gt|`로 잡으면 분모가 격자의 83%라 값이 항상
    작게 나와 변별력이 없다 (같은 체크포인트에서 0.0113 vs 0.0587). planner 관점에서도
    "내가 신뢰한 영역의 오류율"이 직접적인 위험량이다.
    """
    valid_b = valid.bool()
    pred, gt = free_pred.bool() & valid_b, free_gt.bool() & valid_b
    return _rate(pred & ~gt, pred)


def free_miss_rate(free_pred, free_gt, valid):
    """M2b. 보수성(정보 낭비). `fatal_rate`와 비용이 다르므로 절대 합치지 않는다."""
    valid_b = valid.bool()
    pred, gt = free_pred.bool() & valid_b, free_gt.bool() & valid_b
    return _rate(~pred & gt, gt)


def free_metrics_from_masks(pred_free, gt_parts, valid) -> dict:
    """M1·M2·M2b + 분할 불변식을 한 번에. **2-head와 3-class가 공유한다.**

    두 정식화는 `pred_free`를 만드는 방식만 다르고(두 sigmoid의 AND vs. softmax argmax)
    그 이후 집계는 완전히 같다. 집계를 각자 복사해 두면 한쪽만 고쳐지는 순간 A/B가
    무의미해지므로 여기 한 벌만 둔다.

    `pred_free`/`gt_free`를 함께 실어 보내는 이유: M3·M4는 광선 루프가 CPU numpy라 학습
    step마다 돌리면 병목이 된다(이 리그는 이미 데이터 로딩이 병목이다). val 경로가
    forward를 다시 하지 않고 이 마스크를 받아 따로 계산한다.
    """
    gt_free = gt_parts["free"]
    iou, iou_count = iou_free(pred_free, gt_free, valid)
    fatal, fatal_denom = fatal_rate(pred_free, gt_free, valid)
    miss, miss_denom = free_miss_rate(pred_free, gt_free, valid)
    return {
        "iou_free": iou, "iou_free_count": iou_count,
        "fatal_rate": fatal, "fatal_denom": fatal_denom,
        "free_miss_rate": miss, "free_miss_denom": miss_denom,
        # 배선이 틀리면 조용히 이상한 숫자가 나오는 대신 여기서 0이 아니게 된다.
        "partition_defects": partition_defect_count(gt_parts, valid.bool()),
        "pred_free": pred_free, "gt_free": gt_free,
    }
```

`free_metrics_from_masks`는 Task 9·15가 쓰므로 이 태스크에서 **함께 구현하고 테스트한다.** 위 `import` 블록 맨 위에 `from projects.common.free_space import partition_defect_count`를 넣는다 — 이 파일의 모든 import는 **파일 상단 한 곳에** 모은다(Task 4·5가 뒤에 함수를 덧붙이지만 import는 위로 올린다).

테스트도 함께 추가한다:

```python
def test_free_metrics_from_masks_is_the_shared_aggregator():
    """2-head와 3-class가 같은 집계기를 써야 A/B가 공정하다."""
    from projects.common.free_space import decompose
    from projects.common.free_space_metrics import free_metrics_from_masks

    occ = torch.tensor([[[[1.0, 0.0]]]])
    vis = torch.tensor([[[[1.0, 1.0]]]])
    valid = torch.ones_like(occ)
    gt_parts = decompose(occ, vis, valid)

    result = free_metrics_from_masks(gt_parts["free"], gt_parts, valid)

    assert result["iou_free"] == pytest.approx(1.0)
    assert result["partition_defects"] == 0
    assert torch.equal(result["gt_free"], gt_parts["free"])
```

- [ ] **Step 4: 통과를 확인한다**

Run: `python -m pytest tests/common/test_free_space_metrics.py -v`
Expected: 6 passed

- [ ] **Step 5: 커밋**

```bash
git add projects/common/free_space_metrics.py tests/common/test_free_space_metrics.py
git commit -m "Add free-space area metrics that trivial baselines cannot win"
```

---

### Task 3: 방위각 광선 인덱스와 `r(θ)` 추출

**Files:**
- Create: `projects/common/polar.py`
- Test: `tests/common/test_polar.py`

**Interfaces:**
- Consumes: `projects.bev_gt.grid.OccupancyGridSpec`
- Produces:
  - `RAY_OK = 0`, `RAY_NO_FREE = 1`, `RAY_CENSORED = 2`
  - `build_ray_index(grid_spec, n_theta=360, step_cells=0.5) -> RayIndex` — frozen dataclass, 필드 `rows`/`cols` `(n_theta, n_steps)` int64, `radii_m` `(n_steps,)` float64, `inside` `(n_theta, n_steps)` bool
  - `first_free_range(free, ray_index) -> tuple[np.ndarray, np.ndarray]` — `free`는 `(H, W)` bool numpy. 반환은 `(r_m (n_theta,) float64, status (n_theta,) int8)`
  - `reconstruct_free(r_m, status, ray_index, shape) -> np.ndarray` — polar 왕복 검증용 `(H, W)` bool

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/common/test_polar.py`:

```python
import numpy as np
import pytest

from projects.bev_gt.grid import OccupancyGridSpec
from projects.common.polar import (
    RAY_CENSORED,
    RAY_NO_FREE,
    RAY_OK,
    build_ray_index,
    first_free_range,
    reconstruct_free,
)

# 원점이 정확히 격자 중앙에 오는 대칭 스펙 -- 손계산이 가능하다.
SPEC = OccupancyGridSpec(front_m=1.0, rear_m=1.0, half_width_m=1.0, cell_m=0.05)


def test_ray_index_radii_are_monotone_and_start_at_zero():
    rays = build_ray_index(SPEC, n_theta=8)

    assert rays.rows.shape == rays.cols.shape == rays.inside.shape
    assert rays.rows.shape[0] == 8
    assert rays.radii_m[0] == pytest.approx(0.0)
    assert np.all(np.diff(rays.radii_m) > 0)


def test_first_free_range_measures_to_the_first_non_free_cell():
    """원점 주변 정적 사각(0.5 m 원반)을 건너뛴 뒤 첫 free부터 재야 한다."""
    free = np.zeros((SPEC.n_rows, SPEC.n_cols), bool)
    # 전방(row 감소 방향) 한 줄만 free로 만든다: ego z=0 행은 row 19/20 근처.
    free[8:19, 20] = True

    rays = build_ray_index(SPEC, n_theta=4)     # theta=0 이 정확히 전방
    r_m, status = first_free_range(free, rays)

    assert status[0] == RAY_OK
    # free가 row 8까지 이어지므로 첫 non-free는 row 7 = 원점에서 약 0.6 m
    assert r_m[0] == pytest.approx(0.6, abs=SPEC.cell_m)


def test_ray_with_no_free_cell_is_reported_as_undefined():
    free = np.zeros((SPEC.n_rows, SPEC.n_cols), bool)
    rays = build_ray_index(SPEC, n_theta=4)

    _, status = first_free_range(free, rays)

    assert np.all(status == RAY_NO_FREE)


def test_ray_free_all_the_way_out_is_censored_not_a_range():
    """격자 끝까지 free인 광선을 r_max로 회귀에 섞으면 통계가 그 값에 눌린다."""
    free = np.ones((SPEC.n_rows, SPEC.n_cols), bool)
    rays = build_ray_index(SPEC, n_theta=4)

    _, status = first_free_range(free, rays)

    assert np.all(status == RAY_CENSORED)


def test_star_convex_region_survives_the_polar_roundtrip():
    """free가 단일 원점 raycast 결과라 star-convex라는 성질의 회귀 테스트 (스펙 §2.5)."""
    rows, cols = np.mgrid[0:SPEC.n_rows, 0:SPEC.n_cols]
    origin_r, origin_c = SPEC.front_m / SPEC.cell_m - 0.5, SPEC.half_width_m / SPEC.cell_m - 0.5
    radius = np.hypot(rows - origin_r, cols - origin_c)
    free = radius < 12                                    # 원점 중심 원반 = star-convex

    rays = build_ray_index(SPEC, n_theta=720)
    r_m, status = first_free_range(free, rays)
    restored = reconstruct_free(r_m, status, rays, free.shape)

    intersection = (restored & free).sum()
    union = (restored | free).sum()
    assert intersection / union >= 0.99
```

- [ ] **Step 2: 실패를 확인한다**

Run: `python -m pytest tests/common/test_polar.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'projects.common.polar'`

- [ ] **Step 3: 최소 구현을 쓴다**

`projects/common/polar.py`:

```python
"""BEV 격자를 ego 원점 기준 방위각 광선으로 읽는다 -- M3(range error)의 기하 부분.

왜 필요한가: `occupied`는 면적이 아니라 **광선이 멈춘 표면**이라 IoU가 틀린 도구다.
방향별 "첫 장애물까지의 거리"로 재면 통로 폭·정지거리로 직결되고, planner가 실제로
쓰는 양과 같아진다.

격자가 고정이므로 광선별 셀 인덱스는 한 번만 만들어 캐시한다.
"""
from dataclasses import dataclass

import numpy as np

from projects.bev_gt.grid import OccupancyGridSpec

RAY_OK = 0          # 첫 free 이후 첫 non-free를 격자 안에서 만났다
RAY_NO_FREE = 1     # 광선 위에 free 셀이 하나도 없다 (정적 사각·후방 등) -> 지표에서 제외
RAY_CENSORED = 2    # 격자 끝까지 free -> 회귀 통계에 섞지 않고 따로 센다


@dataclass(frozen=True)
class RayIndex:
    rows: np.ndarray      # (n_theta, n_steps) int64
    cols: np.ndarray
    radii_m: np.ndarray   # (n_steps,) float64, 원점으로부터의 거리
    inside: np.ndarray    # (n_theta, n_steps) bool, 격자 안인가


def build_ray_index(grid_spec: OccupancyGridSpec, n_theta: int = 360,
                    step_cells: float = 0.5) -> RayIndex:
    """`n_theta`개 방위각 × 반지름 표본의 (row, col) 인덱스를 미리 계산한다.

    row는 전방이 **감소** 방향이다(`grid.cell_centers_m`과 같은 규약). 원점의 격자 좌표는
    `(front_m/cell - 0.5, half_width_m/cell - 0.5)`이다.
    """
    origin_row = grid_spec.front_m / grid_spec.cell_m - 0.5
    origin_col = grid_spec.half_width_m / grid_spec.cell_m - 0.5
    max_radius_cells = float(np.hypot(
        max(grid_spec.front_m, grid_spec.rear_m), grid_spec.half_width_m
    ) / grid_spec.cell_m)

    radii_cells = np.arange(0.0, max_radius_cells, step_cells)
    thetas = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    # theta=0은 전방(+x). row는 전방이 감소하므로 부호가 뒤집힌다.
    d_row = -np.cos(thetas)[:, None]
    d_col = -np.sin(thetas)[:, None]

    rows = np.round(origin_row + radii_cells[None, :] * d_row).astype(np.int64)
    cols = np.round(origin_col + radii_cells[None, :] * d_col).astype(np.int64)
    inside = (
        (rows >= 0) & (rows < grid_spec.n_rows) & (cols >= 0) & (cols < grid_spec.n_cols)
    )
    # 격자 밖 인덱스는 fancy indexing이 wrap하지 않도록 0으로 눌러 두고 `inside`로만 판단한다.
    return RayIndex(
        rows=np.where(inside, rows, 0),
        cols=np.where(inside, cols, 0),
        radii_m=radii_cells * grid_spec.cell_m,
        inside=inside,
    )


def first_free_range(free: np.ndarray, rays: RayIndex):
    """광선마다 `(첫 free 셀 이후 첫 non-free 셀까지의 거리, 상태)`.

    첫 free부터 재는 이유: ego 원점 주변은 `permanent_blind`(반경 약 0.5 m 원반)라 원점에서
    바로 쏘면 즉시 non-free를 만난다. 그 정적 영역을 건너뛰어야 실제 자유거리가 나온다.

    **격자 안 구간만 본다.** 광선이 격자를 벗어난 뒤의 `False`는 장애물이 아니라 ROI 밖이다.
    그걸 hit으로 세면 모든 광선이 "격자 경계에서 막혔다"가 되어 censored를 구분할 수 없다.
    원점에서 직선으로 나가므로 격자(볼록) 안 구간은 항상 연속이다.
    """
    sampled = free[rays.rows, rays.cols] & rays.inside
    n_theta = sampled.shape[0]
    r_m = np.full(n_theta, np.nan)
    status = np.full(n_theta, RAY_NO_FREE, dtype=np.int8)

    for i in range(n_theta):
        inside_steps = np.nonzero(rays.inside[i])[0]
        if inside_steps.size == 0:
            continue
        lo, hi = int(inside_steps[0]), int(inside_steps[-1]) + 1
        segment = sampled[i, lo:hi]
        if not segment.any():
            continue
        start = int(np.argmax(segment))
        blocked = np.nonzero(~segment[start:])[0]
        if blocked.size == 0:
            status[i] = RAY_CENSORED     # 격자 끝까지 free -- 회귀 통계에 섞지 않는다
            continue
        status[i] = RAY_OK
        r_m[i] = rays.radii_m[lo + start + int(blocked[0])]
    return r_m, status


def reconstruct_free(r_m, status, rays: RayIndex, shape) -> np.ndarray:
    """`r(θ)`만으로 free 영역을 복원한다 -- polar 표현의 정보 손실을 재는 회귀 테스트용."""
    restored = np.zeros(shape, bool)
    sampled_inside = rays.inside
    for i in range(rays.rows.shape[0]):
        if status[i] == RAY_NO_FREE:
            continue
        limit = rays.radii_m[-1] + 1.0 if status[i] == RAY_CENSORED else r_m[i]
        take = sampled_inside[i] & (rays.radii_m < limit)
        restored[rays.rows[i][take], rays.cols[i][take]] = True
    return restored
```

- [ ] **Step 4: 통과를 확인한다**

Run: `python -m pytest tests/common/test_polar.py -v`
Expected: 5 passed

`test_first_free_range_measures_to_the_first_non_free_cell`의 기대값이 어긋나면, 먼저 `build_ray_index`의 `origin_row`/`d_row` 부호를 의심한다 — `grid.cell_centers_m`은 `forward_m = front_m - (i+0.5)*cell`이므로 row가 늘면 전방 거리가 준다.

- [ ] **Step 5: 커밋**

```bash
git add projects/common/polar.py tests/common/test_polar.py
git commit -m "Add a cached polar ray index and free-range extraction"
```

---

### Task 4: M3 range error 집계

**Files:**
- Modify: `projects/common/free_space_metrics.py` (append)
- Modify: `tests/common/test_free_space_metrics.py` (append)

**Interfaces:**
- Consumes: `projects.common.polar.{RayIndex, RAY_OK, RAY_CENSORED, RAY_NO_FREE, first_free_range}`
- Produces:
  - `range_error(free_pred, free_gt, valid, rays) -> dict` — 키: `abs_p50`, `abs_p90`, `over_mean`, `under_mean`, `over_count`, `under_count`, `n_paired_rays`, `censored_gt`, `censored_pred`, `no_free_gt`
  - `summarize_range_error(dicts) -> dict` — 같은 키. `abs_p50`/`abs_p90`은 `n_paired_rays`로, **`over_mean`은 `over_count`로, `under_mean`은 `under_count`로** 가중평균한다 (아래 근거 참고). 카운트는 전부 합

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/common/test_free_space_metrics.py` 끝에 추가:

```python
import numpy as np

from projects.bev_gt.grid import OccupancyGridSpec
from projects.common.free_space_metrics import range_error, summarize_range_error
from projects.common.polar import build_ray_index

RANGE_SPEC = OccupancyGridSpec(front_m=1.0, rear_m=1.0, half_width_m=1.0, cell_m=0.05)


def _forward_corridor(free_cells_ahead):
    """전방 한 줄만 `free_cells_ahead` 칸 열린 (1, 1, H, W) bool 텐서."""
    grid = np.zeros((RANGE_SPEC.n_rows, RANGE_SPEC.n_cols), bool)
    grid[19 - free_cells_ahead:19, 20] = True
    return torch.from_numpy(grid).unsqueeze(0).unsqueeze(0)


def test_range_error_is_zero_when_prediction_matches():
    gt = _forward_corridor(10)
    valid = torch.ones_like(gt)
    rays = build_ray_index(RANGE_SPEC, n_theta=4)

    result = range_error(gt, gt, valid, rays)

    assert result["n_paired_rays"] >= 1
    assert result["abs_p50"] == pytest.approx(0.0)
    assert result["over_mean"] == pytest.approx(0.0)


def test_over_prediction_is_reported_separately_from_under_prediction():
    """과대(위험)와 과소(보수적)는 비용이 다르므로 한 숫자로 뭉개면 안 된다."""
    gt = _forward_corridor(6)
    pred = _forward_corridor(10)                 # 0.2 m 더 멀리 열려 있다고 예측
    valid = torch.ones_like(gt)
    rays = build_ray_index(RANGE_SPEC, n_theta=4)

    result = range_error(pred, gt, valid, rays)

    assert result["over_mean"] == pytest.approx(0.2, abs=RANGE_SPEC.cell_m)
    assert result["under_mean"] == pytest.approx(0.0)


def test_censored_rays_are_counted_but_excluded_from_the_regression():
    gt = torch.ones((1, 1, RANGE_SPEC.n_rows, RANGE_SPEC.n_cols), dtype=torch.bool)
    valid = torch.ones_like(gt)
    rays = build_ray_index(RANGE_SPEC, n_theta=4)

    result = range_error(gt, gt, valid, rays)

    assert result["censored_gt"] == 4
    assert result["n_paired_rays"] == 0
    assert math.isnan(result["abs_p50"])


def test_summarize_range_error_weights_by_paired_ray_count():
    a = {"abs_p50": 0.1, "abs_p90": 0.2, "over_mean": 0.0, "under_mean": 0.1,
         "n_paired_rays": 30, "censored_gt": 1, "censored_pred": 2, "no_free_gt": 3}
    b = {"abs_p50": 0.3, "abs_p90": 0.4, "over_mean": 0.0, "under_mean": 0.3,
         "n_paired_rays": 10, "censored_gt": 0, "censored_pred": 0, "no_free_gt": 1}

    merged = summarize_range_error([a, b])

    assert merged["abs_p50"] == pytest.approx((0.1 * 30 + 0.3 * 10) / 40)
    assert merged["censored_gt"] == 1
    assert merged["n_paired_rays"] == 40
```

- [ ] **Step 2: 실패를 확인한다**

Run: `python -m pytest tests/common/test_free_space_metrics.py -v -k range`
Expected: FAIL — `ImportError: cannot import name 'range_error'`

- [ ] **Step 3: 최소 구현을 쓴다**

`projects/common/free_space_metrics.py`에 추가한다. **import 두 줄은 파일 상단 import 블록으로 올린다** -- 파일 중간에 import를 끼우면 순환 참조가 생겼을 때 원인 추적이 어려워진다.

```python
# --- 파일 상단 import 블록에 추가 ---
import numpy as np

from projects.common.polar import RAY_CENSORED, RAY_NO_FREE, RAY_OK, first_free_range

# --- 이하 파일 끝에 추가 ---

_RANGE_ABS_KEYS = ("abs_p50", "abs_p90")          # 전체 delta 통계 -> n_paired_rays로 가중
# 부분집합 통계 -> 그 부분집합의 크기로 가중해야 한다. 아래 summarize_range_error 참고.
_RANGE_PARTITION_KEYS = {"over_mean": "over_count", "under_mean": "under_count"}
_RANGE_COUNTS = ("n_paired_rays", "over_count", "under_count",
                 "censored_gt", "censored_pred", "no_free_gt")


def range_error(free_pred, free_gt, valid, rays) -> dict:
    """M3. 방위각별 "첫 장애물까지의 거리" 오차 `dr = r_pred - r_gt`.

    GT와 예측이 **둘 다** `RAY_OK`인 광선만 회귀 통계에 넣는다. censored(격자 끝까지 free)를
    r_max로 대체해 섞으면 통계가 그 상수에 눌린다.

    numpy로 계산하므로 CPU로 내린다 -- 학습 루프의 매 step이 아니라 val에서만 부르는 것을
    전제로 한다(광선 루프가 batch당 수 ms 든다).
    """
    pred_np = (free_pred.bool() & valid.bool()).cpu().numpy()
    gt_np = (free_gt.bool() & valid.bool()).cpu().numpy()

    deltas, counts = [], dict.fromkeys(_RANGE_COUNTS, 0)
    for i in range(gt_np.shape[0]):
        r_gt, s_gt = first_free_range(gt_np[i, 0], rays)
        r_pred, s_pred = first_free_range(pred_np[i, 0], rays)
        counts["censored_gt"] += int((s_gt == RAY_CENSORED).sum())
        counts["censored_pred"] += int((s_pred == RAY_CENSORED).sum())
        counts["no_free_gt"] += int((s_gt == RAY_NO_FREE).sum())
        paired = (s_gt == RAY_OK) & (s_pred == RAY_OK)
        deltas.append(r_pred[paired] - r_gt[paired])

    delta = np.concatenate(deltas) if deltas else np.empty(0)
    counts["n_paired_rays"] = int(delta.size)
    over, under = delta[delta > 0], -delta[delta < 0]
    counts["over_count"] = int(over.size)
    counts["under_count"] = int(under.size)
    if delta.size == 0:
        return {**dict.fromkeys(_RANGE_ABS_KEYS, float("nan")),
                **dict.fromkeys(_RANGE_PARTITION_KEYS, float("nan")), **counts}
    return {
        "abs_p50": float(np.percentile(np.abs(delta), 50)),
        "abs_p90": float(np.percentile(np.abs(delta), 90)),
        # 빈 부분집합은 0.0이 아니라 0.0 그대로 둔다: over/under는 강한 부등호로 나눈
        # 진짜 분할이라 원소가 하나라도 있으면 평균이 0일 수 없다. 따라서 batch 안에서
        # 0.0은 "over 사건이 없었다"만 뜻하고 "상쇄돼 0"과 헷갈리지 않는다.
        "over_mean": float(over.mean()) if over.size else 0.0,
        "under_mean": float(under.mean()) if under.size else 0.0,
        **counts,
    }


def summarize_range_error(dicts) -> dict:
    """batch 단위 결과를 합친다.

    **가중치가 통계마다 다르다.** `abs_p50`/`abs_p90`은 delta 전체에 대한 통계라
    `n_paired_rays`로 가중하는 것이 맞지만, `over_mean`은 **over 사건들만**의 평균이므로
    `over_count`로 가중해야 한다. `n_paired_rays`로 가중하면 다음처럼 크게 틀어진다:

        batch A: paired 100, over 1건 x 1.00 m -> over_mean 1.00
        batch B: paired 100, over 99건 x 0.01 m -> over_mean 0.01
        올바른 값 (1*1.00 + 99*0.01)/100 = 0.02
        n_paired_rays 가중 (1.00*100 + 0.01*100)/200 = 0.505   <- 25배 과대

    over-prediction은 "통로가 실제보다 길다"는 안전 실패라 이 숫자가 그대로 배포 판단과
    논문에 들어간다. 카운트는 **합**으로 둔다 -- batch당 평균으로 나누면 batch 크기가
    바뀔 때 숫자가 따라 움직여 run 간 비교가 안 된다.
    """
    if not dicts:
        return {**dict.fromkeys(_RANGE_ABS_KEYS, float("nan")),
                **dict.fromkeys(_RANGE_PARTITION_KEYS, float("nan")),
                **dict.fromkeys(_RANGE_COUNTS, 0)}
    paired = [d["n_paired_rays"] for d in dicts]
    result = {key: weighted_mean([d[key] for d in dicts], paired) for key in _RANGE_ABS_KEYS}
    for key, count_key in _RANGE_PARTITION_KEYS.items():
        result[key] = weighted_mean([d[key] for d in dicts], [d[count_key] for d in dicts])
    for key in _RANGE_COUNTS:
        result[key] = sum(d[key] for d in dicts)
    return result
```

- [ ] **Step 4: 통과를 확인한다**

Run: `python -m pytest tests/common/test_free_space_metrics.py -v`
Expected: 12 passed (기존 8 + 신규 4)

- [ ] **Step 5: 커밋**

```bash
git add projects/common/free_space_metrics.py tests/common/test_free_space_metrics.py
git commit -m "Measure occupied as per-azimuth range error instead of IoU"
```

---

### Task 5: M4 거리 링 분해

**Files:**
- Modify: `projects/common/free_space_metrics.py` (append)
- Modify: `tests/common/test_free_space_metrics.py` (append)

**Interfaces:**
- Consumes: `projects.bev_gt.grid.OccupancyGridSpec`
- Produces:
  - `DEFAULT_RING_EDGES_M = (0.0, 1.5, 3.0, 4.0)`
  - `build_ring_masks(grid_spec, edges_m=DEFAULT_RING_EDGES_M) -> list[tuple[str, np.ndarray]]` — 이름은 `"0.0-1.5m"` 형식, 마스크는 `(H, W)` bool
  - `metrics_per_ring(free_pred, free_gt, valid, ring_masks) -> dict[str, dict]` — 링 이름 -> `{"iou_free", "iou_free_count", "fatal_rate", "fatal_denom"}`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/common/test_free_space_metrics.py` 끝에 추가:

```python
from projects.common.free_space_metrics import build_ring_masks, metrics_per_ring


def test_ring_masks_partition_the_grid_by_distance_from_the_origin():
    rings = build_ring_masks(RANGE_SPEC, edges_m=(0.0, 0.5, 1.0))

    assert [name for name, _ in rings] == ["0.0-0.5m", "0.5-1.0m"]
    inner, outer = rings[0][1], rings[1][1]
    assert not (inner & outer).any()                      # 겹치지 않는다
    assert inner[19, 20] and not outer[19, 20]            # 원점 바로 앞은 안쪽 링


def test_metrics_per_ring_isolates_far_field_failure():
    """근거리의 쉬운 성능이 원거리 실패를 가리는 것을 막는 것이 M4의 목적이다."""
    gt = torch.ones((1, 1, RANGE_SPEC.n_rows, RANGE_SPEC.n_cols), dtype=torch.bool)
    rings = build_ring_masks(RANGE_SPEC, edges_m=(0.0, 0.5, 1.0))
    outer = torch.from_numpy(rings[1][1])
    pred = gt.clone()
    pred[0, 0][outer] = False                             # 바깥 링만 전부 틀린다
    valid = torch.ones_like(gt)

    per_ring = metrics_per_ring(pred, gt, valid, rings)

    assert per_ring["0.0-0.5m"]["iou_free"] == pytest.approx(1.0)
    assert per_ring["0.5-1.0m"]["iou_free"] == pytest.approx(0.0)
```

- [ ] **Step 2: 실패를 확인한다**

Run: `python -m pytest tests/common/test_free_space_metrics.py -v -k ring`
Expected: FAIL — `ImportError: cannot import name 'build_ring_masks'`

- [ ] **Step 3: 최소 구현을 쓴다**

`projects/common/free_space_metrics.py` 끝에 추가:

```python
DEFAULT_RING_EDGES_M = (0.0, 1.5, 3.0, 4.0)


def build_ring_masks(grid_spec, edges_m=DEFAULT_RING_EDGES_M):
    """원점으로부터의 거리 링 마스크. 격자가 전후 비대칭이라 바깥 링은 격자에서 잘린다 --
    그래도 "근거리는 맞는데 원거리에서 무너진다"를 보는 목적에는 충분하다."""
    origin_row = grid_spec.front_m / grid_spec.cell_m - 0.5
    origin_col = grid_spec.half_width_m / grid_spec.cell_m - 0.5
    rows, cols = np.mgrid[0:grid_spec.n_rows, 0:grid_spec.n_cols]
    distance_m = np.hypot(rows - origin_row, cols - origin_col) * grid_spec.cell_m
    return [
        (f"{lo}-{hi}m", (distance_m >= lo) & (distance_m < hi))
        for lo, hi in zip(edges_m[:-1], edges_m[1:])
    ]


def metrics_per_ring(free_pred, free_gt, valid, ring_masks) -> dict:
    """링마다 M1·M2를 다시 잰다. `valid`에 링 마스크를 곱해 같은 함수를 재사용한다."""
    result = {}
    for name, mask in ring_masks:
        ring = torch.from_numpy(mask).to(valid.device).view(1, 1, *mask.shape)
        ring_valid = valid.bool() & ring
        iou, iou_count = iou_free(free_pred, free_gt, ring_valid)
        rate, denom = fatal_rate(free_pred, free_gt, ring_valid)
        result[name] = {
            "iou_free": iou, "iou_free_count": iou_count,
            "fatal_rate": rate, "fatal_denom": denom,
        }
    return result
```

- [ ] **Step 4: 통과를 확인한다**

Run: `python -m pytest tests/common/test_free_space_metrics.py -v`
Expected: 12 passed

- [ ] **Step 5: 커밋**

```bash
git add projects/common/free_space_metrics.py tests/common/test_free_space_metrics.py
git commit -m "Decompose free-space metrics into distance rings"
```

---

### Task 6: 라벨 무결성 테스트와 미측정 값 확정

스펙 §9(B5, B6)와 §12의 두 미해결 항목(censored 광선 비율, 360 vs 720 bin)을 여기서 닫는다.

**Files:**
- Create: `tests/common/test_label_integrity.py`
- Create: `tools/measure_label_geometry.py`
- Modify: `docs/superpowers/specs/2026-08-17-bev-free-space-task-redefinition-design.md` (§12 갱신)

**Interfaces:**
- Consumes: `projects.datasets.robot_simplebev.{build_bev_masks, load_masked_labels, list_sequence_samples, GRID_SPEC, DEFAULT_COMMON_ROOT}`, `projects.common.polar`
- Produces: 없음 (검증 산출물)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/common/test_label_integrity.py`:

```python
"""라벨이 조용히 바뀌는 것을 잡는 gated 테스트.

여기 박힌 숫자는 2026-08-17에 154 프레임 전수로 실측한 값이다(스펙 §2.1). 라벨 파이프라인이
바뀌면 여기가 먼저 터져야 한다 -- 학습 지표가 이상해진 뒤에 원인을 찾는 것보다 훨씬 싸다.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from projects.datasets.robot_simplebev import (
    DEFAULT_COMMON_ROOT,
    build_bev_masks,
    list_sequence_samples,
    load_masked_labels,
)

DATASET_ROOT = Path("dataset/sj_datasets")
SEQUENCES = ("raws1", "raws2", "raws3", "rawos1")
requires_dataset = pytest.mark.skipif(
    not (DATASET_ROOT / "raws1" / "occupancy_npy").exists(),
    reason="self-collected dataset not available locally",
)


def _labels():
    permanent_blind, invalid = build_bev_masks(DEFAULT_COMMON_ROOT)
    for name in SEQUENCES:
        root = DATASET_ROOT / name
        if not (root / "occupancy_npy").exists():
            continue
        for sequence_root, sample_id in list_sequence_samples(root):
            yield name, load_masked_labels(sequence_root, sample_id, permanent_blind, invalid)


@requires_dataset
def test_occupied_is_a_one_cell_frontier_shell():
    """`occupied`가 표면이라는 것이 M3(거리로 잰다)의 전제다. 실측 0.998."""
    neighbourhood = np.ones((3, 3), bool)
    touching = total = 0
    for _, (occ, vis, valid) in _labels():
        occupied = (~occ) & vis & valid
        if not occupied.any():
            continue
        adjacent_to_unknown = ndimage.binary_dilation(~vis, neighbourhood)
        touching += int((occupied & adjacent_to_unknown).sum())
        total += int(occupied.sum())

    assert total > 0
    assert touching / total >= 0.99


@requires_dataset
def test_free_and_occupied_fractions_stay_in_the_measured_range():
    """실측(스펙 §2.1): free는 격자의 0.15~0.28, occupied는 평가 마스크의 0.04~0.08."""
    free_fractions, occupied_fractions = [], []
    for _, (occ, vis, valid) in _labels():
        mask = vis & valid
        free_fractions.append((occ & mask).sum() / valid.sum())
        occupied_fractions.append(((~occ) & mask).sum() / max(mask.sum(), 1))

    assert 0.15 <= float(np.mean(free_fractions)) <= 0.28
    assert 0.04 <= float(np.mean(occupied_fractions)) <= 0.08
```

- [ ] **Step 2: 실패를 확인한다**

Run: `python -m pytest tests/common/test_label_integrity.py -v`
Expected: FAIL — `ModuleNotFoundError` 또는 collection error (파일이 아직 없으므로 Step 1 직후에는 통과할 수도 있다. 그때는 임계값을 일부러 `>= 0.999999`로 바꿔 한 번 FAIL을 확인한 뒤 되돌린다 — 테스트가 실제로 무언가를 재고 있는지 확인하는 절차다.)

- [ ] **Step 3: 측정 도구를 쓴다**

`tools/measure_label_geometry.py`:

```python
"""라벨 기하 실측 -- censored 광선 비율과 polar bin 수의 영향 (스펙 §12).

두 숫자가 M3의 설계 파라미터를 결정한다:
- censored 비율이 높으면 M3의 유효 표본이 줄어 격자 확장을 검토해야 한다
- 360 bin과 720 bin의 지표 차이가 작으면 360을 유지한다 (스펙 §5.2)

실행: python tools/measure_label_geometry.py
"""
import sys
from pathlib import Path

import numpy as np
from fire import Fire

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from projects.common.polar import (  # noqa: E402
    RAY_CENSORED,
    RAY_NO_FREE,
    RAY_OK,
    build_ray_index,
    first_free_range,
    reconstruct_free,
)
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    GRID_SPEC,
    build_bev_masks,
    list_sequence_samples,
    load_masked_labels,
)


def main(sequences="raws1,raws2,raws3,rawos1", dataset_root=DEFAULT_DATASET_ROOT):
    permanent_blind, invalid = build_bev_masks(DEFAULT_COMMON_ROOT)
    frees = []
    for name in str(sequences).split(","):
        root = Path(dataset_root) / name.strip()
        if not (root / "occupancy_npy").exists():
            continue
        for sequence_root, sample_id in list_sequence_samples(root):
            occ, vis, valid = load_masked_labels(
                sequence_root, sample_id, permanent_blind, invalid
            )
            frees.append(occ & vis & valid)
    print(f"frames = {len(frees)}")

    for n_theta in (360, 720):
        rays = build_ray_index(GRID_SPEC, n_theta=n_theta)
        status_counts = np.zeros(3, dtype=np.int64)
        roundtrip = []
        for free in frees:
            r_m, status = first_free_range(free, rays)
            for code in (RAY_OK, RAY_NO_FREE, RAY_CENSORED):
                status_counts[code] += int((status == code).sum())
            restored = reconstruct_free(r_m, status, rays, free.shape)
            union = (restored | free).sum()
            roundtrip.append((restored & free).sum() / union if union else np.nan)
        total = status_counts.sum()
        print(
            f"n_theta={n_theta:4d}  ok {status_counts[RAY_OK] / total:.3f}"
            f"  no_free {status_counts[RAY_NO_FREE] / total:.3f}"
            f"  censored {status_counts[RAY_CENSORED] / total:.3f}"
            f"  |  roundtrip IoU {np.nanmean(roundtrip):.4f}"
        )


if __name__ == "__main__":
    Fire(main)
```

- [ ] **Step 4: 측정을 실행하고 테스트를 통과시킨다**

```bash
conda activate bev-chamdog
python -m pytest tests/common/test_label_integrity.py -v
python tools/measure_label_geometry.py
```

Expected: 테스트 2 passed. 측정 도구는 `n_theta=360`과 `720` 두 줄을 출력한다.

- [ ] **Step 5: 측정 결과를 스펙 §12에 반영한다**

스펙 §12의 마지막 문단("M3의 censored 광선 비율을 아직 측정하지 않았다")을 실측값으로 교체하고, 360 유지 여부에 대한 결론을 한 줄로 적는다. **판단 기준**: censored 비율이 0.3을 넘으면 격자 확장을 후속 과제로 §11에 추가한다. 360과 720의 roundtrip IoU 차이가 0.02 미만이면 360을 유지한다.

- [ ] **Step 6: 커밋**

```bash
git add tests/common/test_label_integrity.py tools/measure_label_geometry.py \
        docs/superpowers/specs/2026-08-17-bev-free-space-task-redefinition-design.md
git commit -m "Pin label geometry with regression tests and close the open measurements"
```

---

## Phase 0 게이트

```bash
conda activate bev-chamdog
python -m pytest tests/ -q
```

Expected: **162 + 신규 테스트 전부 passed, 실패 0.** 통과하지 못하면 Phase 1로 넘어가지 않는다.

---

# Phase 1 — 기존 체크포인트 재채점 (재학습 없음)

## File Structure (Phase 1)

| 파일 | 책임 |
|---|---|
| `tools/rescore_checkpoints.py` (신규) | 체크포인트 하나를 새 지표로 재채점하고, 같은 split의 트리비얼 baseline을 함께 출력 |
| `projects/common/baselines.py` (신규) | 이미지를 보지 않는 baseline 예측기 (constant map, all-free). 학습 스크립트도 Phase 2에서 재사용한다 |
| `tests/common/test_baselines.py` (신규) | Task 7 |
| `docs/free_space_metric_migration.md` (신규) | Task 8의 재채점 결과표 |

---

### Task 7: 트리비얼 baseline 예측기

**Files:**
- Create: `projects/common/baselines.py`
- Test: `tests/common/test_baselines.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `constant_free_map(free_masks) -> np.ndarray` — `(H, W)` bool. 셀별 다수결
  - `all_free_map(shape) -> np.ndarray`
  - `as_batch(map_2d, batch_size, device) -> torch.Tensor` — `(B, 1, H, W)` bool

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/common/test_baselines.py`:

```python
import numpy as np
import torch

from projects.common.baselines import all_free_map, as_batch, constant_free_map


def test_constant_free_map_takes_the_per_cell_majority():
    masks = [
        np.array([[True, True], [False, False]]),
        np.array([[True, False], [False, False]]),
        np.array([[True, False], [True, False]]),
    ]

    result = constant_free_map(masks)

    # 셀별 free 비율: 1.0, 0.33, 0.33, 0.0 -> 0.5 초과만 True
    assert result.tolist() == [[True, False], [False, False]]


def test_all_free_map_is_free_everywhere():
    assert all_free_map((2, 3)).all()


def test_as_batch_broadcasts_a_single_map_over_the_batch():
    batched = as_batch(np.array([[True, False]]), batch_size=4, device="cpu")

    assert batched.shape == (4, 1, 1, 2)
    assert batched.dtype == torch.bool
    assert batched[0, 0].tolist() == [[True, False]]
```

- [ ] **Step 2: 실패를 확인한다**

Run: `python -m pytest tests/common/test_baselines.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'projects.common.baselines'`

- [ ] **Step 3: 최소 구현을 쓴다**

`projects/common/baselines.py`:

```python
"""이미지를 전혀 보지 않는 baseline 예측기.

왜 필요한가: 지표를 고치는 것만으로는 재발을 막지 못한다. 이전 지표가 깨진 것을 6주 동안
못 본 이유가 정확히 "트리비얼 해와 비교하지 않은 것"이라, 기준선을 매 run 눈앞에 강제로
두는 장치가 있어야 한다. 실측(val=raws2): constant map의 iou_free 0.673, 학습된 모델 0.850.
"""
import numpy as np
import torch


def constant_free_map(free_masks) -> np.ndarray:
    """train split의 셀별 free 다수결 = "장면을 안 보고 레이아웃만 외운" 예측기."""
    stacked = np.stack([np.asarray(mask, dtype=np.float32) for mask in free_masks])
    return stacked.mean(axis=0) > 0.5


def all_free_map(shape) -> np.ndarray:
    """"전부 통과 가능"이라는 극단. `iou_drivable`은 이걸 못 이겼고 `iou_free`는 이겨야 한다."""
    return np.ones(shape, dtype=bool)


def as_batch(map_2d, batch_size: int, device) -> torch.Tensor:
    """`(H, W)` -> `(B, 1, H, W)`. 지표 함수가 batch 텐서만 받으므로 형상을 맞춘다."""
    tensor = torch.from_numpy(np.asarray(map_2d, dtype=bool)).to(device)
    return tensor.view(1, 1, *tensor.shape).expand(batch_size, 1, *tensor.shape)
```

- [ ] **Step 4: 통과를 확인한다**

Run: `python -m pytest tests/common/test_baselines.py -v`
Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add projects/common/baselines.py tests/common/test_baselines.py
git commit -m "Add image-blind baseline predictors for metric sanity checks"
```

---

### Task 8: 재채점 도구와 마이그레이션 결과표

**Files:**
- Create: `tools/rescore_checkpoints.py`
- Create: `docs/free_space_metric_migration.md`
- Test: `tests/tools/test_rescore_checkpoints.py`

**Interfaces:**
- Consumes: Task 1–7의 전부, `projects.models.simplebev_two_head.{TwoHeadSegnet, split_two_head_logits}`, `projects.datasets.robot_simplebev.RobotBEVDataset`, `projects.models.double_sphere_vox.build_double_sphere_vox_util`
- Produces:
  - `score_split(model, loader, vox_util, rays, ring_masks, device, constant_map) -> dict` — 키 `iou_free`, `fatal_rate`, `free_miss_rate`, `range`, `rings`, `baseline_iou_free`, `baseline_fatal_rate`, `all_free_iou_free`, 그리고 하위호환용 `iou_drivable`, `iou_obstacle`
  - `format_markdown_table(rows) -> str`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/tools/test_rescore_checkpoints.py`:

```python
from tools.rescore_checkpoints import format_markdown_table


def test_markdown_table_puts_the_baseline_next_to_every_model_row():
    """baseline이 같은 표에 없으면 숫자를 혼자 읽게 되고, 그게 이번 결함의 원인이었다."""
    rows = [
        {"name": "robot_finetune", "split": "raws2", "iou_free": 0.850,
         "baseline_iou_free": 0.673, "all_free_iou_free": 0.17,
         "fatal_rate": 0.0587, "baseline_fatal_rate": 0.1678,
         "iou_drivable": 0.891, "iou_obstacle": 0.312},
    ]

    text = format_markdown_table(rows)

    assert "iou_free" in text
    assert "baseline" in text
    assert "0.850" in text and "0.673" in text
    # 옛 지표도 남겨야 과거 실험 기록을 해석할 수 있다
    assert "0.891" in text
```

- [ ] **Step 2: 실패를 확인한다**

Run: `python -m pytest tests/tools/test_rescore_checkpoints.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.rescore_checkpoints'`

- [ ] **Step 3: 도구를 구현한다**

`tools/rescore_checkpoints.py`:

```python
"""기존 체크포인트를 새 free-space 지표로 재채점한다 (스펙 Phase 1).

재학습하지 않고 지표만 바꿔 다시 재는 것이 요점이다. 새 지표가 옛 지표를 포함·확장하는지,
그리고 트리비얼 baseline과의 순서가 맞는지를 여기서 확인한 뒤에야 학습 스크립트를 건드린다.

실행:
    CUDA_VISIBLE_DEVICES=1 python tools/rescore_checkpoints.py \\
        --checkpoint=runs/robot_bev/ckpt/<run>/model_best-000000046.pth \\
        --train_sequences=raws1,raws3,rawos1 --val_sequences=raws2
"""
import sys
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.common.baselines import all_free_map, as_batch, constant_free_map  # noqa: E402
from projects.common.free_space import decompose  # noqa: E402
from projects.common.free_space_metrics import (  # noqa: E402
    build_ring_masks,
    fatal_rate,
    free_miss_rate,
    iou_free,
    metrics_per_ring,
    range_error,
    summarize_range_error,
    weighted_mean,
)
from projects.common.polar import build_ray_index  # noqa: E402
from projects.common.two_head_metrics import compute_drivable_and_obstacle_iou  # noqa: E402
from projects.datasets.robot_simplebev import (  # noqa: E402
    DEFAULT_COMMON_ROOT,
    DEFAULT_DATASET_ROOT,
    GRID_SPEC,
    RobotBEVDataset,
    build_bev_masks,
    list_sequence_samples,
    load_masked_labels,
    parse_sequence_names,
)
from projects.geometry.double_sphere import FINETUNE_CAMERA_NAMES  # noqa: E402
from projects.models.double_sphere_vox import build_double_sphere_vox_util  # noqa: E402
from projects.models.simplebev_two_head import TwoHeadSegnet, split_two_head_logits  # noqa: E402

_COLUMNS = (
    ("name", "체크포인트"), ("split", "val"),
    ("iou_free", "iou_free↑"), ("baseline_iou_free", "baseline iou_free"),
    ("all_free_iou_free", "all-free iou_free"),
    ("fatal_rate", "fatal↓"), ("baseline_fatal_rate", "baseline fatal"),
    ("iou_drivable", "(참고) iou_drivable"), ("iou_obstacle", "(참고) iou_obstacle"),
)


def format_markdown_table(rows) -> str:
    """baseline을 모델 값 바로 옆에 붙인다 -- 숫자를 혼자 읽게 두는 것이 이번 결함의 원인이었다."""
    header = "| " + " | ".join(label for _, label in _COLUMNS) + " |"
    rule = "|" + "---|" * len(_COLUMNS)
    lines = [header, rule]
    for row in rows:
        cells = []
        for key, _ in _COLUMNS:
            value = row.get(key, float("nan"))
            cells.append(value if isinstance(value, str) else f"{value:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _collect_free_masks(samples, permanent_blind, invalid):
    masks = []
    for sequence_root, sample_id in samples:
        occ, vis, valid = load_masked_labels(sequence_root, sample_id, permanent_blind, invalid)
        masks.append(occ & vis & valid)
    return masks


def score_split(model, loader, vox_util, rays, ring_masks, device, constant_map) -> dict:
    ious, iou_counts, fatals, fatal_denoms, misses, miss_denoms = [], [], [], [], [], []
    base_ious, base_counts, base_fatals, base_denoms = [], [], [], []
    allfree_ious, allfree_counts = [], []
    d_ious, o_ious, o_counts, range_dicts, ring_dicts = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb_camXs"].to(device) - 0.5
            _, _, two_head, _, _ = model(
                rgb, batch["pix_T_cams"].to(device), batch["cam0_T_camXs"].to(device), vox_util
            )
            occ_logits, vis_logits = split_two_head_logits(two_head)
            seg_g = batch["seg_bev_g"].to(device)
            vis_g = batch["vis_bev_g"].to(device)
            valid = batch["valid_bev_g"].to(device)

            gt = decompose(seg_g, vis_g, valid)["free"]
            pred = decompose(torch.sigmoid(occ_logits), torch.sigmoid(vis_logits), valid)["free"]
            batch_size = gt.shape[0]
            base = as_batch(constant_map, batch_size, device)
            allfree = as_batch(all_free_map(constant_map.shape), batch_size, device)

            for values, counts, fn, arg in (
                (ious, iou_counts, iou_free, pred),
                (base_ious, base_counts, iou_free, base),
                (allfree_ious, allfree_counts, iou_free, allfree),
            ):
                value, count = fn(arg, gt, valid)
                values.append(value)
                counts.append(count)
            for values, denoms, fn, arg in (
                (fatals, fatal_denoms, fatal_rate, pred),
                (base_fatals, base_denoms, fatal_rate, base),
                (misses, miss_denoms, free_miss_rate, pred),
            ):
                value, denom = fn(arg, gt, valid)
                values.append(value)
                denoms.append(denom)

            d_iou, o_iou, o_count = compute_drivable_and_obstacle_iou(
                occ_logits, seg_g, vis_g * valid
            )
            d_ious.append(float(d_iou.item()))
            o_ious.append(float(o_iou.item()))
            o_counts.append(o_count)
            range_dicts.append(range_error(pred, gt, valid, rays))
            ring_dicts.append(metrics_per_ring(pred, gt, valid, ring_masks))

    rings = {}
    for name, _ in ring_masks:
        rings[name] = {
            "iou_free": weighted_mean([d[name]["iou_free"] for d in ring_dicts],
                                      [d[name]["iou_free_count"] for d in ring_dicts]),
            "fatal_rate": weighted_mean([d[name]["fatal_rate"] for d in ring_dicts],
                                        [d[name]["fatal_denom"] for d in ring_dicts]),
        }
    return {
        "iou_free": weighted_mean(ious, iou_counts),
        "baseline_iou_free": weighted_mean(base_ious, base_counts),
        "all_free_iou_free": weighted_mean(allfree_ious, allfree_counts),
        "fatal_rate": weighted_mean(fatals, fatal_denoms),
        "baseline_fatal_rate": weighted_mean(base_fatals, base_denoms),
        "free_miss_rate": weighted_mean(misses, miss_denoms),
        "iou_drivable": float(np.mean(d_ious)) if d_ious else float("nan"),
        "iou_obstacle": weighted_mean(o_ious, o_counts),
        "range": summarize_range_error(range_dicts),
        "rings": rings,
    }


def main(
    checkpoint,
    train_sequences="raws1,raws3,rawos1",
    val_sequences="raws2",
    dataset_root=DEFAULT_DATASET_ROOT,
    common_root=DEFAULT_COMMON_ROOT,
    encoder_type="res101",
    batch_size=4,
    num_workers=4,
    n_theta=360,
    device="cuda",
):
    dataset_root = Path(dataset_root)
    train_samples = [
        s for name in parse_sequence_names(train_sequences)
        for s in list_sequence_samples(dataset_root / name)
    ]
    val_samples = [
        s for name in parse_sequence_names(val_sequences)
        for s in list_sequence_samples(dataset_root / name)
    ]
    permanent_blind, invalid = build_bev_masks(common_root, GRID_SPEC, FINETUNE_CAMERA_NAMES)
    constant_map = constant_free_map(
        _collect_free_masks(train_samples, permanent_blind, invalid)
    )

    val_ds = RobotBEVDataset(val_samples, common_root=common_root)
    loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    Z, Y, X = GRID_SPEC.n_rows, 1, GRID_SPEC.n_cols
    vox_util = build_double_sphere_vox_util(GRID_SPEC, val_ds.cameras, device=device)
    model = TwoHeadSegnet(
        Z, Y, X, vox_util, use_radar=False, use_lidar=False,
        do_rgbcompress=True, encoder_type=encoder_type, rand_flip=False,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=False)
    model.eval()

    scores = score_split(
        model, loader, vox_util,
        build_ray_index(GRID_SPEC, n_theta=n_theta),
        build_ring_masks(GRID_SPEC), device, constant_map,
    )
    row = {"name": Path(checkpoint).parent.name, "split": val_sequences, **scores}
    print(format_markdown_table([row]))
    print()
    print(f"range: p50 {scores['range']['abs_p50']:.3f} m | p90 {scores['range']['abs_p90']:.3f} m"
          f" | over {scores['range']['over_mean']:.3f} m | under {scores['range']['under_mean']:.3f} m"
          f" | paired rays {scores['range']['n_paired_rays']}")
    for name, values in scores["rings"].items():
        print(f"  ring {name:10s} iou_free {values['iou_free']:.3f}"
              f"  fatal {values['fatal_rate']:.3f}")


if __name__ == "__main__":
    Fire(main)
```

- [ ] **Step 4: 단위 테스트를 통과시킨다**

Run: `python -m pytest tests/tools/test_rescore_checkpoints.py -v`
Expected: 1 passed

- [ ] **Step 5: 실제 체크포인트를 재채점한다**

```bash
conda activate bev-chamdog
CUDA_VISIBLE_DEVICES=1 python tools/rescore_checkpoints.py \
    --checkpoint=runs/robot_bev/ckpt/robot_finetune_res101_bs8_lr1e-04_260817_122504/model_best-000000046.pth \
    --train_sequences=raws1,raws3,rawos1 --val_sequences=raws2
```

Expected (2026-08-17 실측과 대조 — 이것이 **게이트 C7–C9**):

| 항목 | 기대값 | 허용 |
|---|---|---|
| `iou_free` | 0.850 | ±0.01 |
| `baseline_iou_free` | 0.673 | ±0.01 |
| `fatal_rate` | 0.059 | ±0.005 |
| `baseline_fatal_rate` | 0.168 | ±0.01 |
| `iou_drivable` | 0.891 | ±0.01 |
| `iou_obstacle` | 0.312 | ±0.01 |

그리고 **순서가 반드시 성립해야 한다**: `iou_free(모델) > iou_free(constant map) > iou_free(all-free)`. 어긋나면 배선이 틀린 것이므로 Phase 2로 넘어가지 않는다.

- [ ] **Step 6: 결과를 문서로 남긴다**

`docs/free_space_metric_migration.md`를 새로 만들고 다음을 담는다:
- 위 명령과 그 표준출력 전체를 그대로 붙인다
- 스펙 §2.4의 대비표(SynWoodScape에서는 옛 지표가 정상, 로봇에서는 깨짐)를 재확인한 결과
- "옛 지표를 삭제하지 않고 진단으로 남긴다"는 결정과 그 이유 한 줄

- [ ] **Step 7: 커밋**

```bash
git add tools/rescore_checkpoints.py tests/tools/test_rescore_checkpoints.py \
        docs/free_space_metric_migration.md
git commit -m "Rescore existing checkpoints with the free-space metrics"
```

---

## Phase 1 게이트

`docs/free_space_metric_migration.md`의 표에서 세 조건이 전부 성립해야 한다.

1. `iou_free(모델) > iou_free(constant) > iou_free(all-free)`
2. 재채점한 `iou_drivable`/`iou_obstacle`이 2026-08-17 실측값(0.891 / 0.312)과 ±0.01 안
3. `fatal_rate(모델) < fatal_rate(constant)`

하나라도 어긋나면 **원인을 찾을 때까지 Phase 2로 넘어가지 않는다.** 2번이 깨지면 데이터 로딩이나 체크포인트 경로 문제이므로, 지표가 아니라 배선을 먼저 의심한다.

---

# Phase 2 — 학습 스크립트 지표 교체 (2-head 유지)

## File Structure (Phase 2)

| 파일 | 책임 |
|---|---|
| `projects/common/two_head_metrics.py` (수정) | `run_batch`가 free-space 지표를 함께 돌려주고, 로그 행에 `iou_free`를 넣는다 |
| `tools/train_robot_bev.py` (수정) | baseline 계산, `val_score = iou_free`, TensorBoard 기록 |
| `tools/train_synwoodscape.py` (수정) | 같은 지표를 쓰도록 최소 배선 (pretrain 숫자와 나란히 읽기 위해) |
| `tools/visualize_robot_predictions.py` (수정) | 3-class 팔레트 + polar overlay |
| `tests/tools/test_train_synwoodscape_logging.py` (수정) | 로그 포맷 회귀 |

---

### Task 9: `run_batch`에 free-space 지표를 배선한다

**Files:**
- Modify: `projects/common/two_head_metrics.py`
- Test: `tests/tools/test_train_synwoodscape_logging.py` (추가)

**Interfaces:**
- Consumes: Task 1, 2, 4, 5
- Produces:
  - `compute_free_metrics(occ_logits, vis_logits, seg_g, vis_g, valid_g) -> dict` — 키 `iou_free`, `iou_free_count`, `fatal_rate`, `fatal_denom`, `free_miss_rate`, `free_miss_denom`, `partition_defects`, 그리고 val 경로가 M3·M4를 계산할 때 재사용할 `pred_free` / `gt_free` (둘 다 `(B, 1, H, W)` bool)
  - `run_batch(...)`의 반환에 그 딕셔너리 하나 추가 — 기존 8-튜플 뒤에 `free_metrics`가 붙어 **9-튜플**이 된다
  - `summarize_free_metrics(dicts) -> dict` — 키 `iou_free`, `fatal_rate`, `free_miss_rate`, `partition_defects`. `pred_free`/`gt_free`는 집계하지 않는다(텐서라 epoch 단위로 모으면 메모리가 샌다)

`range_error`와 `metrics_per_ring`은 **`run_batch`에 넣지 않는다.** 광선 루프가 CPU numpy라 매 학습 step마다 돌리면 병목이 된다(이 리그는 이미 데이터 로딩이 병목이다). val 경로에서만 따로 부른다 — Task 10에서 배선한다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/tools/test_train_synwoodscape_logging.py` 끝에 추가:

```python
from projects.common.two_head_metrics import summarize_free_metrics


def test_summarize_free_metrics_weights_iou_by_sample_count():
    dicts = [
        {"iou_free": 0.8, "iou_free_count": 3, "fatal_rate": 0.1, "fatal_denom": 100,
         "free_miss_rate": 0.2, "free_miss_denom": 50, "partition_defects": 0},
        {"iou_free": 0.4, "iou_free_count": 1, "fatal_rate": 0.5, "fatal_denom": 100,
         "free_miss_rate": 0.6, "free_miss_denom": 50, "partition_defects": 0},
    ]

    merged = summarize_free_metrics(dicts)

    assert merged["iou_free"] == pytest.approx((0.8 * 3 + 0.4 * 1) / 4)
    assert merged["fatal_rate"] == pytest.approx(0.3)


def test_summarize_free_metrics_of_an_empty_epoch_is_nan():
    merged = summarize_free_metrics([])

    assert math.isnan(merged["iou_free"])
```

- [ ] **Step 2: 실패를 확인한다**

Run: `python -m pytest tests/tools/test_train_synwoodscape_logging.py -v -k free_metrics`
Expected: FAIL — `ImportError: cannot import name 'summarize_free_metrics'`

- [ ] **Step 3: 구현한다**

`projects/common/two_head_metrics.py` 상단 import에 추가:

```python
from projects.common.free_space import decompose  # noqa: E402
from projects.common.free_space_metrics import free_metrics_from_masks  # noqa: E402
```

같은 파일에 함수 추가:

```python
def compute_free_metrics(occ_logits, vis_logits, seg_g, vis_g, valid_g) -> dict:
    """2-head 출력 -> free-space 지표.

    `free`는 두 head의 **결합** 결과다. 지금까지의 지표는 두 head를 따로 채점해서, 정작
    로봇이 쓰는 결합 결과를 아무도 보지 않았다.

    집계 자체는 `free_metrics_from_masks`에 있다 -- 3-class 경로와 같은 집계기를 써야
    Phase 3의 A/B가 공정하다. 여기서 하는 일은 **예측을 free 마스크로 바꾸는 것뿐**이다.
    """
    gt = decompose(seg_g, vis_g, valid_g)
    pred = decompose(torch.sigmoid(occ_logits), torch.sigmoid(vis_logits), valid_g)
    return free_metrics_from_masks(pred["free"], gt, valid_g)


def summarize_free_metrics(dicts) -> dict:
    if not dicts:
        return {"iou_free": float("nan"), "fatal_rate": float("nan"),
                "free_miss_rate": float("nan"), "partition_defects": 0}
    return {
        "iou_free": weighted_mean([d["iou_free"] for d in dicts],
                                  [d["iou_free_count"] for d in dicts]),
        "fatal_rate": weighted_mean([d["fatal_rate"] for d in dicts],
                                    [d["fatal_denom"] for d in dicts]),
        "free_miss_rate": weighted_mean([d["free_miss_rate"] for d in dicts],
                                        [d["free_miss_denom"] for d in dicts]),
        "partition_defects": sum(d["partition_defects"] for d in dicts),
    }
```

`run_batch`의 `return` 직전에 한 줄, 반환 튜플 끝에 한 항목을 추가한다:

```python
    free_metrics = compute_free_metrics(occ_bev_e, vis_bev_e, seg_bev_g, vis_bev_g, valid_bev_g)
    return (
        loss,
        loss_parts,
        drivable_iou,
        obstacle_iou,
        obstacle_count,
        vis_metrics,
        occ_metrics,
        deploy_metrics,
        free_metrics,
    )
```

- [ ] **Step 4: 호출부를 9-튜플로 맞춘다**

`run_batch`를 언패킹하는 곳은 네 군데다 — `tools/train_robot_bev.py`의 `_evaluate`와 학습 루프, `tools/train_synwoodscape.py`의 같은 두 곳. 네 곳 모두 같은 패턴으로 고친다.

`tools/train_robot_bev.py` `_evaluate`:

```python
def _evaluate(model, loader, vox_util, pos_weight_tensor, device, lambda_vis, vis_neg_weight):
    losses, occ_losses, vis_losses = [], [], []
    d_ious, o_ious, o_counts, false_highs, false_lows = [], [], [], [], []
    occ_dicts, deploy_dicts, free_dicts = [], [], []
    with torch.no_grad():
        for batch in loader:
            (loss, parts, d_iou, o_iou, o_count, vis_metrics, occ_metrics, deploy,
             free_metrics) = run_batch(
                model, batch, vox_util, pos_weight_tensor, device, lambda_vis, vis_neg_weight
            )
            ...                                    # 기존 append 들은 그대로
            free_dicts.append(free_metrics)
    ...
    return {
        ...,                                       # 기존 키는 그대로
        "free": summarize_free_metrics(free_dicts),
    }
```

학습 루프도 같다 — `free_dicts = []`를 epoch 시작 시 만들고, 언패킹에 `free_metrics`를 받아 `append`하고, `train` 딕셔너리에 `"free": summarize_free_metrics(free_dicts)`를 넣는다. val을 돌지 않은 epoch용 기본값 딕셔너리에도 `"free": summarize_free_metrics([])`를 추가한다.

`tools/train_synwoodscape.py`는 지표 이름이 같으므로 동일하게 고친다. import에 `summarize_free_metrics`를 추가한다.

- [ ] **Step 4b: 전체 테스트를 돌린다**

Run: `python -m pytest tests/ -q`
Expected: 전부 passed (신규 2개 포함). `TypeError: cannot unpack` 이 나면 네 호출부 중 하나를 놓친 것이다 — `grep -n "run_batch(" tools/*.py` 로 확인한다.

- [ ] **Step 5: 커밋**

```bash
git add projects/common/two_head_metrics.py tools/train_robot_bev.py \
        tools/train_synwoodscape.py tests/tools/test_train_synwoodscape_logging.py
git commit -m "Report free-space metrics from the shared two-head batch step"
```

---

### Task 10: 로그·배너·TensorBoard에 `iou_free`와 baseline

**Files:**
- Modify: `projects/common/two_head_metrics.py` (`format_epoch_log`, `_format_metric_row`)
- Modify: `tools/train_robot_bev.py` (배너, val 경로의 range/ring, TensorBoard)
- Test: `tests/tools/test_train_synwoodscape_logging.py`

**Interfaces:**
- Consumes: Task 4, 5, 7, 9
- Produces: `format_epoch_log(...)`에 키워드 인자 4개 추가 — `train_free_metrics`, `val_free_metrics`, `val_range_metrics=None`, `baseline_iou_free=None`. 전부 기본값이 있어 기존 호출부가 깨지지 않는다

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/tools/test_train_synwoodscape_logging.py` 끝에 추가:

```python
def test_epoch_log_shows_iou_free_with_the_baseline_delta():
    """baseline이 같은 줄에 없으면 숫자를 혼자 읽게 된다 -- 그게 이번 결함의 재발 경로다."""
    text = format_epoch_log(
        epoch=46, num_epochs=60, epoch_time=41.0,
        train_loss=0.03, train_occ_loss=0.003, train_vis_loss=0.06,
        train_d_iou=0.98, train_o_iou=0.78, train_v_false_high=0.001, train_v_false_low=0.02,
        val_loss=0.27, val_occ_loss=0.20, val_vis_loss=0.14,
        val_d_iou=0.89, val_o_iou=0.31, val_v_false_high=0.016, val_v_false_low=0.10,
        train_free_metrics={"iou_free": 0.97, "fatal_rate": 0.01,
                            "free_miss_rate": 0.02, "partition_defects": 0},
        val_free_metrics={"iou_free": 0.850, "fatal_rate": 0.0587,
                          "free_miss_rate": 0.0998, "partition_defects": 0},
        baseline_iou_free=0.673,
        val_score=0.850, best_val_score=0.840, is_new_best=True,
    )

    assert "iou_free" in text
    assert "0.850" in text
    assert "+0.177" in text          # baseline 대비 델타
    assert "fatal" in text


def test_epoch_log_flags_a_broken_partition_loudly():
    text = format_epoch_log(
        epoch=1, num_epochs=60, epoch_time=1.0,
        train_loss=0.1, train_occ_loss=0.1, train_vis_loss=0.1,
        train_d_iou=0.1, train_o_iou=0.1, train_v_false_high=0.1, train_v_false_low=0.1,
        val_loss=0.1, val_occ_loss=0.1, val_vis_loss=0.1,
        val_d_iou=0.1, val_o_iou=0.1, val_v_false_high=0.1, val_v_false_low=0.1,
        train_free_metrics={"iou_free": 0.1, "fatal_rate": 0.1,
                            "free_miss_rate": 0.1, "partition_defects": 7},
        val_free_metrics={"iou_free": 0.1, "fatal_rate": 0.1,
                          "free_miss_rate": 0.1, "partition_defects": 0},
        val_score=0.1, best_val_score=0.0, is_new_best=True,
    )

    assert "partition" in text.lower()
```

- [ ] **Step 2: 실패를 확인한다**

Run: `python -m pytest tests/tools/test_train_synwoodscape_logging.py -v -k "iou_free or partition"`
Expected: FAIL — `TypeError: format_epoch_log() got an unexpected keyword argument 'train_free_metrics'`

- [ ] **Step 3: 로그 포맷을 구현한다**

`projects/common/two_head_metrics.py`. `_format_metric_row`의 시그니처에 `free_metrics`를 추가하고 `fields` 맨 앞에 `iou_free`를 굵게 넣는다. `iou_drivable`/`iou_obstacle`은 `emphasis`를 뺀다 — 이제 참고 지표다:

```python
def _format_metric_row(tag, tag_color, loss, occ_loss, vis_loss, d_iou, o_iou,
                       v_false_high, v_false_low, occ_metrics, free_metrics=None):
    """train/val 한 줄. `iou_free`만 굵게 -- 나머지는 그걸 해석하기 위한 보조 지표다.

    `iou_drivable`/`iou_obstacle`은 삭제하지 않고 남긴다. pretrain run과의 숫자 연속성이
    끊기면 `docs/synwoodscape_pretrain_experiment_log.md`를 해석할 수 없다. 다만 강조는 뺀다
    -- 자체 로봇 데이터에서 이 둘은 트리비얼 해에 지는 지표다(스펙 §2.2).
    """
    fields = []
    if free_metrics is not None:
        fields += [
            _field("iou_free↑", free_metrics["iou_free"], emphasis=_Ansi.BOLD),
            _field("fatal↓", free_metrics["fatal_rate"]),
            _field("free_miss↓", free_metrics["free_miss_rate"]),
        ]
    fields += [
        _field("loss_total↓", loss, ".4f"),
        _field("loss_occ↓", occ_loss, ".4f"),
        _field("loss_vis↓", vis_loss, ".4f"),
        _c(_Ansi.DIM, "(ref)") + " " + _field("iou_drivable↑", d_iou),
        _field("iou_obstacle↑", o_iou),
        _field("vis_false_high↓", v_false_high),
        _field("vis_false_low↓", v_false_low),
    ]
    if occ_metrics is not None:
        fields += [
            _field("obst_frac", occ_metrics["obstacle_frac"]),
            _field("false_obstacle↓", occ_metrics["false_obstacle"]),
            _field("missed_obstacle↓", occ_metrics["missed_obstacle"]),
        ]
    return _format_row(tag, tag_color, fields)


def _format_range_line(metrics):
    """M3는 광선 통계라 train/val 행에 끼우면 줄이 터진다 -- 별도 줄로 뺀다."""
    if metrics is None:
        return []
    fields = [
        _field("abs_p50↓", metrics["abs_p50"], emphasis=_Ansi.BOLD),
        _field("abs_p90↓", metrics["abs_p90"]),
        _field("over↓", metrics["over_mean"]),
        _field("under", metrics["under_mean"]),
        _c(_Ansi.DIM, f"rays {metrics['n_paired_rays']} censored {metrics['censored_gt']}"),
    ]
    return [_format_row("range", _Ansi.BLUE, fields)]


def _format_partition_warning(train_free, val_free):
    """분할이 깨진 셀이 있으면 지표가 조용히 거짓말한다 -- 눈에 걸리게 만든다."""
    defects = (train_free or {}).get("partition_defects", 0) + \
              (val_free or {}).get("partition_defects", 0)
    if defects == 0:
        return []
    return [_c(_Ansi.BOLD + _Ansi.RED,
               f"  [BUG] partition defect on {defects} cells"
               " -- free/occupied/unknown이 valid를 정확히 덮지 않는다")]
```

`format_epoch_log`의 시그니처에 `train_free_metrics=None`, `val_free_metrics=None`, `val_range_metrics=None`, `baseline_iou_free=None`을 추가하고 헤더 줄과 행 조립을 바꾼다:

```python
    baseline_delta = ""
    if baseline_iou_free is not None and not math.isnan(val_score):
        gap = val_score - baseline_iou_free
        baseline_delta = " " + _c(
            _Ansi.GREEN if gap > 0 else _Ansi.RED, f"({gap:+.3f} vs baseline)"
        )
    return "\n".join([
        (
            _c(_Ansi.BOLD + _Ansi.CYAN, f"epoch {epoch:03d}/{num_epochs}") + _sep()
            + _c(_Ansi.DIM, f"time {epoch_time:6.1f}s") + _sep()
            + _field("val_iou_free↑", val_score, emphasis=_Ansi.BOLD) + delta
            + baseline_delta + _sep()
            + _field("best_val_iou_free↑", displayed_best) + _sep()
            + (_c(_Ansi.BOLD + _Ansi.GREEN, "checkpoint: new best") if is_new_best
               else _c(_Ansi.DIM, "checkpoint: -"))
        ),
        *_format_partition_warning(train_free_metrics, val_free_metrics),
        *_format_obstacle_bin_summary(val_occ_metrics),
        _format_metric_row("train", _Ansi.YELLOW, train_loss, train_occ_loss, train_vis_loss,
                           train_d_iou, train_o_iou, train_v_false_high, train_v_false_low,
                           train_occ_metrics, train_free_metrics),
        _format_metric_row("val", _Ansi.CYAN, val_loss, val_occ_loss, val_vis_loss,
                           val_d_iou, val_o_iou, val_v_false_high, val_v_false_low,
                           val_occ_metrics, val_free_metrics),
        *_format_range_line(val_range_metrics),
        *_format_deployment_line(val_deploy_metrics),
    ])
```

- [ ] **Step 3b: 트레이너에 baseline·range·TensorBoard를 배선한다**

`tools/train_robot_bev.py`. import에 다음을 추가한다:

```python
from projects.common.baselines import as_batch, constant_free_map  # noqa: E402
from projects.common.free_space import decompose  # noqa: E402
from projects.common.free_space_metrics import (  # noqa: E402
    build_ring_masks, iou_free, metrics_per_ring, range_error, summarize_range_error,
)
from projects.common.polar import build_ray_index  # noqa: E402
```

`main` 안, 배너를 찍기 전에 baseline을 만든다:

```python
    # train split의 셀별 free 다수결 = "이미지를 안 보고 레이아웃만 외운" 예측기.
    # 이 숫자를 배너와 매 epoch 줄에 강제로 띄운다 -- 지표를 고치는 것만으로는 재발을
    # 막지 못한다. 이전 결함을 오래 못 본 이유가 정확히 트리비얼 해와 비교하지 않은 것이다.
    train_free_masks = []
    for sequence_root, sample_id in train_samples:
        occ, vis, valid = load_masked_labels(
            sequence_root, sample_id, permanent_blind, invalid
        )
        train_free_masks.append(occ & vis & valid)
    constant_map = constant_free_map(train_free_masks) if train_free_masks else None
    rays = build_ray_index(GRID_SPEC, n_theta=n_theta)
    ring_masks = build_ring_masks(GRID_SPEC)
```

`main`의 인자 목록에 `n_theta=360`을 추가한다. 배너 리스트에 한 줄 추가한다:

```python
        f" constant-map baseline iou_free = {baseline_iou_free:.3f}  <- compare against this",
```

`baseline_iou_free`는 val split에 대해 한 번만 계산한다 (`val_loader`를 한 바퀴 돌면서 GT만 읽으면 되므로 모델이 필요 없다):

```python
def _baseline_iou_free(val_samples, permanent_blind, invalid, constant_map, device):
    """constant map baseline의 `iou_free`. 모델이 필요 없으므로 학습 시작 전에 한 번 잰다."""
    if constant_map is None or not val_samples:
        return float("nan")
    values, counts = [], []
    for sequence_root, sample_id in val_samples:
        occ, vis, valid = load_masked_labels(
            sequence_root, sample_id, permanent_blind, invalid
        )
        free_gt = torch.from_numpy(occ & vis & valid).view(1, 1, *occ.shape).to(device)
        valid_t = torch.from_numpy(valid).view(1, 1, *valid.shape).to(device)
        value, count = iou_free(as_batch(constant_map, 1, device), free_gt, valid_t)
        values.append(value)
        counts.append(count)
    return weighted_mean(values, counts)
```

**M3·M4는 val 경로에서만 계산한다** — 광선 루프가 CPU numpy라 학습 step마다 돌리면 병목이 된다. `run_batch`가 이미 만든 마스크를 재사용하므로 forward를 두 번 하지 않는다. `_evaluate`에 `rays`/`ring_masks` 인자를 추가하고 배치마다:

```python
            valid = batch["valid_bev_g"].to(device)
            range_dicts.append(range_error(
                free_metrics["pred_free"], free_metrics["gt_free"], valid, rays
            ))
            ring_dicts.append(metrics_per_ring(
                free_metrics["pred_free"], free_metrics["gt_free"], valid, ring_masks
            ))
```

`_evaluate`의 반환 딕셔너리에 `"range": summarize_range_error(range_dicts)`와, Task 8의 `score_split`과 같은 방식으로 링을 접은 `"rings"`를 추가한다. val을 돌지 않은 epoch용 기본값에도 `"range": summarize_range_error([])`, `"rings": {}`를 넣는다.

TensorBoard 기록을 추가한다:

```python
            for key, tb in (("iou_free", "iou_free_epoch"),
                            ("fatal_rate", "fatal_rate_epoch"),
                            ("free_miss_rate", "free_miss_rate_epoch")):
                writer.add_scalar(f"{split}/{tb}", summary["free"][key], epoch)
            # val 전용
            for key, tb in (("abs_p50", "range_abs_p50_epoch"), ("abs_p90", "range_abs_p90_epoch"),
                            ("over_mean", "range_over_epoch"), ("under_mean", "range_under_epoch")):
                writer.add_scalar(f"val/{tb}", val["range"][key], epoch)
            for name, values in val["rings"].items():
                writer.add_scalar(f"val/ring_{name}_iou_free_epoch", values["iou_free"], epoch)
```

`format_epoch_log` 호출에 `train_free_metrics=train["free"]`, `val_free_metrics=val["free"]`, `val_range_metrics=val["range"]`, `baseline_iou_free=baseline_iou_free`를 넘긴다.

- [ ] **Step 4: 통과를 확인한다**

Run: `python -m pytest tests/ -q`
Expected: 전부 passed

- [ ] **Step 5: 커밋**

```bash
git add projects/common/two_head_metrics.py tools/train_robot_bev.py \
        tests/tools/test_train_synwoodscape_logging.py
git commit -m "Put iou_free and its trivial baseline in the epoch log"
```

---

### Task 11: checkpoint 선택 기준을 `iou_free`로 교체

**Files:**
- Modify: `tools/train_robot_bev.py:~336` (`val_score = 0.5 * (val["d_iou"] + val["o_iou"])`)
- Modify: `tools/train_synwoodscape.py:292` (같은 식)
- Modify: `docs/training_guide.md`

**Interfaces:**
- Consumes: Task 9의 `summarize_free_metrics`
- Produces: 없음 (동작 변경)

- [ ] **Step 1: 두 트레이너의 `val_score`를 바꾼다**

`tools/train_robot_bev.py`:

```python
            # `iou_free`는 free를 넓게 불러도 좁게 불러도 벌점이라 축퇴 해가 없다.
            # 이전 기준 (d_iou + o_iou)/2 는 "obstacle을 아예 안 찍는 모델"에 유리했다
            # -- 실측으로 '전부 drivable'(0.935)이 학습된 모델(0.891)을 이겼다.
            val_score = val["free"]["iou_free"]
```

`tools/train_synwoodscape.py`도 같은 값으로 바꾸고, 마무리 배너의 `best val (drivable+obstacle)/2 IoU = ...` 문구를 `best val iou_free = ...`로 고친다.

- [ ] **Step 2: 문서를 맞춘다**

`docs/training_guide.md`의 epoch 로그 예시와 지표 설명을 새 형태로 갱신한다. `val_iou_mean` → `val_iou_free`, baseline 델타 설명, `iou_drivable`/`iou_obstacle`이 참고 지표로 강등됐다는 한 줄을 넣는다.

- [ ] **Step 3: 스모크 테스트로 한 epoch만 돌려본다**

```bash
conda activate bev-chamdog
CUDA_VISIBLE_DEVICES=1 python tools/train_robot_bev.py \
    --exp_name=metric_smoke --train_sequences=raws1,raws3,rawos1 --val_sequences=raws2 \
    --num_epochs=1 --batch_size=8 --num_workers=8 --init_checkpoint=None
```

Expected: 배너에 `constant map baseline iou_free` 줄이 보이고, epoch 줄에 `val_iou_free`와 baseline 델타가 보이며, `partition` 경고가 **없어야** 한다.

- [ ] **Step 4: 전체 테스트**

Run: `python -m pytest tests/ -q`
Expected: 전부 passed

- [ ] **Step 5: 커밋**

```bash
git add tools/train_robot_bev.py tools/train_synwoodscape.py docs/training_guide.md
git commit -m "Select checkpoints on iou_free instead of the degenerate IoU mean"
```

---

### Task 12: 시각화 — 3-class 팔레트와 polar overlay

**Files:**
- Modify: `tools/visualize_robot_predictions.py`
- Test: `tests/tools/test_visualize_robot_predictions.py`

**Interfaces:**
- Consumes: Task 1, 3
- Produces:
  - `FREE_SPACE_PALETTE: dict[str, tuple[int, int, int]]` — 키 `"free"`, `"occupied"`, `"unknown"`, `"invalid"`
  - `render_free_space_panel(parts, valid) -> np.ndarray` — `(H, W, 3)` uint8
  - `draw_range_profile(panel, r_m, status, rays, grid_spec, colour) -> np.ndarray`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/tools/test_visualize_robot_predictions.py` 끝에 추가:

```python
import numpy as np
import torch

from tools.visualize_robot_predictions import FREE_SPACE_PALETTE, render_free_space_panel


def test_free_space_panel_uses_a_distinct_colour_per_class():
    parts = {
        "free": torch.tensor([[True, False, False]]),
        "occupied": torch.tensor([[False, True, False]]),
        "unknown": torch.tensor([[False, False, True]]),
    }
    valid = torch.tensor([[True, True, True]])

    panel = render_free_space_panel(parts, valid)

    assert panel.shape == (1, 3, 3)
    assert tuple(panel[0, 0]) == FREE_SPACE_PALETTE["free"]
    assert tuple(panel[0, 1]) == FREE_SPACE_PALETTE["occupied"]
    assert tuple(panel[0, 2]) == FREE_SPACE_PALETTE["unknown"]


def test_invalid_cells_get_their_own_colour_not_unknown():
    """`valid=0`은 수집 아티팩트다. `unknown`과 같은 색으로 칠하면 검수에서 구분이 안 된다."""
    parts = {
        "free": torch.tensor([[False]]),
        "occupied": torch.tensor([[False]]),
        "unknown": torch.tensor([[False]]),
    }
    valid = torch.tensor([[False]])

    panel = render_free_space_panel(parts, valid)

    assert tuple(panel[0, 0]) == FREE_SPACE_PALETTE["invalid"]
    assert FREE_SPACE_PALETTE["invalid"] != FREE_SPACE_PALETTE["unknown"]
```

- [ ] **Step 2: 실패를 확인한다**

Run: `python -m pytest tests/tools/test_visualize_robot_predictions.py -v -k free_space`
Expected: FAIL — `ImportError: cannot import name 'FREE_SPACE_PALETTE'`

- [ ] **Step 3: 구현한다**

`tools/visualize_robot_predictions.py`에 추가:

```python
# GT와 pred를 같은 팔레트로 그려야 눈으로 뺄셈이 된다. `invalid`는 `unknown`과 반드시
# 달라야 한다 -- 하나는 배포 때 사라지는 수집 아티팩트고 다른 하나는 진짜 미관측이다.
FREE_SPACE_PALETTE = {
    "free": (60, 200, 90),
    "occupied": (220, 60, 60),
    "unknown": (25, 25, 30),
    "invalid": (150, 60, 190),
}


def render_free_space_panel(parts, valid) -> np.ndarray:
    """분해 -> (H, W, 3) uint8. 입력은 `(H, W)` 또는 `(1, 1, H, W)` bool 텐서."""
    def squeeze(tensor):
        array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        return array.reshape(array.shape[-2], array.shape[-1])

    valid_2d = squeeze(valid)
    panel = np.zeros((*valid_2d.shape, 3), np.uint8)
    panel[...] = FREE_SPACE_PALETTE["invalid"]
    panel[valid_2d] = FREE_SPACE_PALETTE["unknown"]
    for name in ("unknown", "occupied", "free"):
        panel[squeeze(parts[name])] = FREE_SPACE_PALETTE[name]
    return panel
```

그리고 `r(θ)`를 격자 좌표로 되돌려 찍는 함수를 추가한다. GT는 흰색, pred는 노란색으로 **같은 패널에** 겹쳐야 두 선의 벌어짐이 곧 range error로 읽힌다:

```python
RANGE_PROFILE_COLOURS = {"gt": (255, 255, 255), "pred": (250, 220, 60)}


def draw_range_profile(panel, r_m, status, rays, grid_spec, colour) -> np.ndarray:
    """`r(θ)`를 격자 위 점렬로 찍는다 -- M3 오차를 눈으로 보게 하는 것이 목적이다.

    `RAY_OK`가 아닌 광선은 건너뛴다. censored(격자 끝까지 free)를 격자 경계에 찍으면
    "저 방향은 경계에서 막혔다"로 오독된다.
    """
    from projects.common.polar import RAY_OK

    origin_row = grid_spec.front_m / grid_spec.cell_m - 0.5
    origin_col = grid_spec.half_width_m / grid_spec.cell_m - 0.5
    thetas = np.linspace(0.0, 2 * np.pi, len(r_m), endpoint=False)
    for i, theta in enumerate(thetas):
        if status[i] != RAY_OK:
            continue
        radius_cells = r_m[i] / grid_spec.cell_m
        row = int(round(origin_row - radius_cells * np.cos(theta)))
        col = int(round(origin_col - radius_cells * np.sin(theta)))
        if 0 <= row < panel.shape[0] and 0 <= col < panel.shape[1]:
            panel[row, col] = colour
    return panel
```

기존 패널 레이아웃에 `GT free-space`와 `pred free-space` 두 장을 추가하고, `pred free-space` 패널 위에 두 프로파일을 겹쳐 그린다.

- [ ] **Step 4: 통과 확인 + 실제 이미지 생성**

```bash
python -m pytest tests/tools/test_visualize_robot_predictions.py -v
CUDA_VISIBLE_DEVICES=1 python tools/visualize_robot_predictions.py \
    --checkpoint=runs/robot_bev/ckpt/robot_finetune_res101_bs8_lr1e-04_260817_122504/model_best-000000046.pth \
    --sequences=raws2 --limit=4
```

Expected: 테스트 통과. 생성된 PNG에서 `free`(초록) 영역의 GT/pred가 나란히 보이고, 노란 선(pred)이 흰 선(GT) 근처를 따라간다.

- [ ] **Step 5: 커밋**

```bash
git add tools/visualize_robot_predictions.py tests/tools/test_visualize_robot_predictions.py
git commit -m "Visualise free/occupied/unknown with the range profile overlaid"
```

---

### Task 13: 2-head 기준선 재학습 (게이트)

**Files:** 없음 (실행과 기록만)

- [ ] **Step 1: 현행 설정 그대로 재학습한다**

```bash
conda activate bev-chamdog
CUDA_VISIBLE_DEVICES=1 EXP_NAME=twohead_baseline_iou_free \
    TRAIN_SEQUENCES=raws1,raws3,rawos1 VAL_SEQUENCES=raws2 \
    bash configs/train_robot_bev_finetune.sh 2>&1 | tee runs/twohead_baseline.log
```

`configs/train_robot_bev_finetune.sh`의 `VAL_SEQUENCES` 기본값이 비어 있으므로 위처럼 환경변수로 넘긴다. `VAL_TAIL_FRACTION`은 `VAL_SEQUENCES`가 있으면 무시된다.

- [ ] **Step 2: 게이트를 확인한다**

재학습으로 나온 best `iou_free`가 **Phase 1 재채점 값(0.850) ±0.03 안**에 들어와야 한다. 같은 데이터·같은 설정·같은 지표이므로 크게 벗어나면 지표 배선이나 split이 달라진 것이다.

`partition_defects`는 **모든 epoch에서 0**이어야 한다. 하나라도 0이 아니면 즉시 멈추고 Task 1의 분해를 다시 본다.

- [ ] **Step 3: 결과를 문서에 남긴다**

`docs/free_space_metric_migration.md`에 "Phase 2 기준선" 절을 추가하고, 마지막 epoch의 로그 줄과 best 값을 붙인다. 이 숫자가 Phase 3 A/B의 비교 대상이다.

- [ ] **Step 4: 커밋**

```bash
git add docs/free_space_metric_migration.md
git commit -m "Record the two-head baseline under the free-space metrics"
```

---

## Phase 2 게이트

1. `python -m pytest tests/ -q` 전부 통과
2. 재학습 best `iou_free`가 Phase 1 재채점 값 ±0.03
3. 모든 epoch에서 `partition_defects == 0`
4. `docs/free_space_metric_migration.md`에 기준선이 기록됨

---

# Phase 3 — 3-class 단일 head 전환

## File Structure (Phase 3)

| 파일 | 책임 |
|---|---|
| `projects/models/simplebev_three_class.py` (신규) | `ThreeClassSegnet` — 최종 1×1 conv만 3채널. trunk는 Simple-BEV 그대로 |
| `projects/common/three_class_metrics.py` (신규) | 3-class용 `run_batch`와 loss. `two_head_metrics.py`의 형제 |
| `tools/train_robot_bev.py` (수정) | `--head=two_head|three_class` 스위치 |
| `tests/models/test_simplebev_three_class.py` (신규) | Task 14 |
| `tests/common/test_three_class_metrics.py` (신규) | Task 15 |

Task 16에서 `projects/common/two_head_metrics.py`를 `bev_occupancy_metrics.py`로 개칭한다 — 그때는 두 정식화가 한 파일 쌍으로 대칭이 되므로 이름이 의미를 갖는다.

---

### Task 14: `ThreeClassSegnet`

**Files:**
- Create: `projects/models/simplebev_three_class.py`
- Test: `tests/models/test_simplebev_three_class.py`

**Interfaces:**
- Consumes: `nets.segnet.{Decoder, Segnet}` (submodule), `projects.common.free_space.{FREE, OCCUPIED, UNKNOWN}`
- Produces:
  - `ThreeClassDecoder(in_channels, predict_future_flow=False)` — `segmentation_head`가 `(B, 3, H, W)` logits를 낸다. `forward`는 `out["three_class"]`를 추가로 담는다
  - `ThreeClassSegnet(Z, Y, X, vox_util, **kwargs)` — `TwoHeadSegnet`과 동일한 생성자
  - `load_trunk_weights(model, checkpoint_path, device) -> dict` — 2-head 체크포인트에서 trunk만 로드. 반환 키 `loaded`(int), `skipped`(list[str])

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/models/test_simplebev_three_class.py`. 기존 `tests/models/test_simplebev_two_head.py`와 같이 **Decoder만 직접 테스트한다** — `Segnet` 전체를 만들면 `vox_util`과 ResNet 가중치 다운로드가 필요해 단위 테스트가 무거워진다:

```python
import torch

from projects.common.free_space import FREE, OCCUPIED, UNKNOWN
from projects.models.simplebev_three_class import ThreeClassDecoder, load_trunk_weights
from projects.models.simplebev_two_head import TwoHeadDecoder


def test_class_ids_are_distinct():
    assert len({FREE, OCCUPIED, UNKNOWN}) == 3


def test_decoder_emits_three_logit_channels():
    decoder = ThreeClassDecoder(in_channels=8)
    x = torch.randn(2, 8, 16, 16)

    out = decoder(x)

    assert out["three_class"].shape == (2, 3, 16, 16)


def test_trunk_weights_transfer_from_a_two_head_checkpoint(tmp_path):
    """pretrain 자산은 trunk다. 최종 1x1 conv만 새로 배운다 -- 그래서 Phase 4(pretrain
    재학습)를 마지막에 둘 수 있다."""
    source = TwoHeadDecoder(in_channels=8)
    path = tmp_path / "two_head.pth"
    torch.save({"model_state_dict": source.state_dict()}, path)

    target = ThreeClassDecoder(in_channels=8)
    report = load_trunk_weights(target, path, device="cpu")

    assert report["loaded"] > 0
    # 건너뛴 것은 세그멘테이션 head의 파라미터뿐이어야 한다
    assert report["skipped"]
    assert all("segmentation_head" in name for name in report["skipped"])


def test_trunk_transfer_actually_changes_the_weights(tmp_path):
    """`strict=False` 로드는 형상이 안 맞는 키를 조용히 넘겨서 "pretrain이 안 먹었다"를
    못 찾게 만든다. 실제로 값이 옮겨졌는지 확인한다."""
    source = TwoHeadDecoder(in_channels=8)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.5)
    path = tmp_path / "two_head.pth"
    torch.save({"model_state_dict": source.state_dict()}, path)

    target = ThreeClassDecoder(in_channels=8)
    load_trunk_weights(target, path, device="cpu")

    shared = target.state_dict()["conv3d.weight"] if "conv3d.weight" in target.state_dict() \
        else next(iter(target.state_dict().values()))
    assert torch.allclose(shared, torch.full_like(shared, 0.5))
```

마지막 테스트의 `shared` 선택이 `Decoder` 내부 이름에 의존하므로, 구현 후 실제 `state_dict()` 키를 확인해 **trunk에 확실히 속하는 키 하나**로 바꿔 적는다(`print(list(target.state_dict())[:5])`).

- [ ] **Step 2: 실패를 확인한다**

Run: `python -m pytest tests/models/test_simplebev_three_class.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'projects.models.simplebev_three_class'`

- [ ] **Step 3: 구현한다**

`projects/models/simplebev_three_class.py`:

```python
"""3-class(`free`/`occupied`/`unknown`) Simple-BEV wrapper.

왜 2-head를 대신하는가:
- 2-head의 loss는 `free`를 한 번도 직접 보지 않고 두 프록시(occ, vis)를 따로 최적화했다.
  `free = occ & vis`이므로 어느 한쪽 오차가 곧 `free` 오차인데 loss는 그 결합을 모른다.
- `sigmoid(occ)>0.5`이면서 `sigmoid(vis)<0.5`인 "보이지도 않는데 주행 가능"한 모순 출력이
  가능했고 사후에 `&`로 뭉갰다. softmax에는 그런 상태가 없고, 세 확률이 제대로 된 분포라
  로그오즈 시간축 융합(rolling local map)에 그대로 쓸 수 있다.
- occ head는 `~vis` 영역에서 gradient를 못 받으면서 그 영역에도 출력을 냈다 -- 검증된 적
  없는 외삽에 capacity를 쓴 것이다.

upstream submodule은 손대지 않는다. encoder·lifting·BEV compressor·decoder trunk를 그대로
재사용하고 최종 세그멘테이션 projection만 3채널로 바꾼다.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

_SIMPLE_BEV_DIR = Path(__file__).resolve().parents[2] / "third_party/models/simple_bev"
if str(_SIMPLE_BEV_DIR) not in sys.path:
    sys.path.insert(0, str(_SIMPLE_BEV_DIR))

from nets.segnet import Decoder, Segnet  # noqa: E402

NUM_CLASSES = 3


def build_three_class_head(channels: int) -> nn.Sequential:
    """`TwoHeadSegmentationHead`와 같은 몸통에 출력만 3채널. `nn.Sequential`로 두는 이유는
    upstream `Decoder.segmentation_head`와 형태를 맞춰 state_dict 키가 예측 가능해지기 때문이다."""
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm2d(channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels, NUM_CLASSES, kernel_size=1, padding=0),
    )


class ThreeClassDecoder(Decoder):
    def __init__(self, in_channels: int, predict_future_flow: bool = False):
        super().__init__(in_channels=in_channels, n_classes=1,
                         predict_future_flow=predict_future_flow)
        self.segmentation_head = build_three_class_head(in_channels)

    def forward(self, x, bev_flip_indices=None):
        out = super().forward(x, bev_flip_indices=bev_flip_indices)
        out["three_class"] = out["segmentation"]
        return out


class ThreeClassSegnet(Segnet):
    """세그멘테이션 출력이 `(B, 3, H, W)` logits인 `Segnet` 변종."""

    def __init__(self, *args, **kwargs):
        latent_dim = kwargs.get("latent_dim", 128)
        super().__init__(*args, **kwargs)
        self.decoder = ThreeClassDecoder(in_channels=latent_dim, predict_future_flow=False)


def load_trunk_weights(model, checkpoint_path, device) -> dict:
    """2-head 체크포인트에서 trunk만 옮긴다. 형상이 맞는 키만 **명시적으로** 고른다.

    `load_state_dict(..., strict=False)`를 쓰면 안 되는 이유: 형상이 안 맞는 키를 조용히
    넘기므로 나중에 "왜 pretrain이 안 먹었지"를 추적할 수 없다. 여기서는 건너뛴 키를
    돌려주고 배너에 찍어, 세그멘테이션 head 말고 다른 것이 빠지면 바로 보이게 한다.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model_state_dict", checkpoint)
    target = model.state_dict()
    transfer, skipped = {}, []
    for name, tensor in source.items():
        if name in target and target[name].shape == tensor.shape:
            transfer[name] = tensor
        else:
            skipped.append(name)
    target.update(transfer)
    model.load_state_dict(target)
    model.to(device)
    return {"loaded": len(transfer), "skipped": skipped}
```

- [ ] **Step 4: 통과를 확인한다**

Run: `python -m pytest tests/models/test_simplebev_three_class.py -v`
Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add projects/models/simplebev_three_class.py tests/models/test_simplebev_three_class.py
git commit -m "Add a three-class Simple-BEV head that reuses the pretrained trunk"
```

---

### Task 15: 3-class weighted cross-entropy와 `run_batch`

**Files:**
- Create: `projects/common/three_class_metrics.py`
- Test: `tests/common/test_three_class_metrics.py`

**Interfaces:**
- Consumes: Task 1, 2, 14
- Produces:
  - `CLASS_ORDER = (UNKNOWN, FREE, OCCUPIED)` — `cross_entropy` 채널 인덱스 순서
  - `compute_three_class_loss(logits, class_index, valid, class_weights) -> tuple[torch.Tensor, dict]` — `loss_parts` 키는 `loss_free`, `loss_occupied`, `loss_unknown`
  - `default_class_weights(samples, permanent_blind, invalid, load_labels) -> torch.Tensor` — `(3,)` 역빈도, `MAX_CLASS_WEIGHT=20.0`으로 clip. `load_labels`는 `(sequence_root, sample_id, permanent_blind, invalid) -> (occ, vis, valid)` 서명을 가진 함수(데이터셋별로 다르므로 주입받는다)
  - `compute_free_metrics(logits, seg_g, vis_g, valid_g) -> dict` — `two_head_metrics.compute_free_metrics`와 **동일한 키**
  - `run_batch(model, batch, vox_util, class_weights, device) -> tuple` — `(loss, loss_parts, free_metrics)` 3-튜플

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/common/test_three_class_metrics.py`:

```python
import math

import pytest
import torch

from projects.common.free_space import FREE, OCCUPIED, UNKNOWN
from projects.common.three_class_metrics import compute_three_class_loss


def _logits_favouring(class_id, shape=(1, 3, 2, 2)):
    logits = torch.full(shape, -5.0)
    logits[:, class_id] = 5.0
    return logits


def test_loss_is_near_zero_when_the_prediction_is_confidently_right():
    target = torch.full((1, 1, 2, 2), FREE, dtype=torch.long)
    valid = torch.ones((1, 1, 2, 2), dtype=torch.bool)

    loss, parts = compute_three_class_loss(
        _logits_favouring(FREE), target, valid, torch.ones(3)
    )

    assert float(loss) < 0.01
    assert set(parts) == {"loss_free", "loss_occupied", "loss_unknown"}


def test_invalid_cells_contribute_no_gradient():
    """`valid=0`은 수집 아티팩트다. 학습되면 "후방은 항상 unknown"을 외운다."""
    target = torch.full((1, 1, 2, 2), FREE, dtype=torch.long)
    valid = torch.zeros((1, 1, 2, 2), dtype=torch.bool)

    loss, _ = compute_three_class_loss(
        _logits_favouring(OCCUPIED), target, valid, torch.ones(3)
    )

    assert float(loss) == pytest.approx(0.0)


def test_class_weights_scale_the_occupied_term():
    """`occupied`는 두께 1셀 껍질(평가 마스크의 6.5%)이라 가중치 없이는 무시된다."""
    target = torch.full((1, 1, 2, 2), OCCUPIED, dtype=torch.long)
    valid = torch.ones((1, 1, 2, 2), dtype=torch.bool)
    logits = _logits_favouring(FREE)

    light, _ = compute_three_class_loss(logits, target, valid, torch.ones(3))
    heavy, _ = compute_three_class_loss(
        logits, target, valid, torch.tensor([1.0, 1.0, 10.0])
    )

    assert float(heavy) > float(light) * 5
```

`torch.tensor([1.0, 1.0, 10.0])`의 인덱스는 `UNKNOWN=0`, `FREE=1`, `OCCUPIED=2` 순서다.

- [ ] **Step 2: 실패를 확인한다**

Run: `python -m pytest tests/common/test_three_class_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'projects.common.three_class_metrics'`

- [ ] **Step 3: 구현한다**

`projects/common/three_class_metrics.py`:

```python
"""3-class(`free`/`occupied`/`unknown`) 학습의 loss와 batch step.

`two_head_metrics.py`의 형제 모듈이다. 두 모듈이 **같은 `free_metrics` 딕셔너리**를
돌려주는 것이 핵심 -- 그래야 로깅·checkpoint 선택·TensorBoard 코드를 분기하지 않고
2-head와 3-class를 같은 지표로 A/B 할 수 있다.
"""
import torch
import torch.nn.functional as F

from projects.common.free_space import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    decompose,
    decompose_from_class_index,
    to_class_index,
)
from projects.common.free_space_metrics import free_metrics_from_masks

CLASS_ORDER = (UNKNOWN, FREE, OCCUPIED)          # cross_entropy의 채널 인덱스와 같은 순서
_PART_BY_CLASS = {UNKNOWN: "unknown", FREE: "free", OCCUPIED: "occupied"}
MAX_CLASS_WEIGHT = 20.0


def compute_three_class_loss(logits, class_index, valid, class_weights):
    """`valid` 안에서만 학습하는 가중 cross-entropy.

    `valid.sum()`으로 정규화하는 이유: 유효 셀 수가 프레임마다 크게 다르다(통로 중앙 vs
    행 끝). `.mean()`을 쓰면 batch 내 프레임 기여도가 불균등해져 학습이 흔들린다
    (`docs/BEV_loss_and_metrics_design.md` §1.3).

    `loss_parts`는 클래스별 평균을 따로 담는다 -- `occupied`가 두께 1셀 껍질이라 어느
    클래스가 학습을 지배하는지 보이지 않으면 클래스 가중치를 맞출 수 없다.
    """
    valid_f = valid.float()
    per_cell = F.cross_entropy(
        logits, class_index.squeeze(1), weight=class_weights.to(logits.device),
        reduction="none",
    ).unsqueeze(1)
    masked = per_cell * valid_f
    total = masked.sum() / (valid_f.sum() + 1e-6)

    parts = {}
    for class_id in CLASS_ORDER:
        selector = ((class_index == class_id) & valid.bool()).float()
        parts[f"loss_{_PART_BY_CLASS[class_id]}"] = (
            (per_cell * selector).sum() / (selector.sum() + 1e-6)
        )
    return total, parts


def default_class_weights(samples, permanent_blind, invalid, load_labels) -> torch.Tensor:
    """train split 실측 역빈도. `MAX_CLASS_WEIGHT`로 clip한다.

    로봇 데이터의 `occupied` 비율은 격자의 1%대라 역빈도를 그대로 쓰면 100배가 넘고,
    loss가 그 항 하나에 눌려 `free` 학습이 망가진다.
    """
    counts = torch.zeros(3, dtype=torch.float64)
    for sequence_root, sample_id in samples:
        occ, vis, valid = load_labels(sequence_root, sample_id, permanent_blind, invalid)
        counts[UNKNOWN] += float(((~vis) & valid).sum())
        counts[FREE] += float((occ & vis & valid).sum())
        counts[OCCUPIED] += float(((~occ) & vis & valid).sum())
    weights = counts.sum() / counts.clamp(min=1.0)
    weights = weights / weights.min()
    return weights.clamp(max=MAX_CLASS_WEIGHT).float()


def compute_free_metrics(logits, seg_g, vis_g, valid_g) -> dict:
    """3-class 출력 -> free-space 지표.

    `two_head_metrics.compute_free_metrics`와 **같은 집계기**(`free_metrics_from_masks`)를
    쓴다. 다른 것은 예측을 free 마스크로 바꾸는 방식뿐 -- 두 sigmoid의 AND 대신 argmax다.
    """
    gt = decompose(seg_g, vis_g, valid_g)
    pred = decompose_from_class_index(logits.argmax(dim=1, keepdim=True), valid_g)
    return free_metrics_from_masks(pred["free"], gt, valid_g)


def run_batch(model, batch, vox_util, class_weights, device):
    """`two_head_metrics.run_batch`의 3-class 대응물."""
    rgb_camXs = batch["rgb_camXs"].to(device) - 0.5   # nuScenes 관례: [0,1] -> [-0.5,0.5]
    seg_bev_g = batch["seg_bev_g"].to(device)
    vis_bev_g = batch["vis_bev_g"].to(device)
    valid_bev_g = batch["valid_bev_g"].to(device)

    _, _, logits, _, _ = model(
        rgb_camXs, batch["pix_T_cams"].to(device), batch["cam0_T_camXs"].to(device), vox_util
    )
    class_index = to_class_index(decompose(seg_bev_g, vis_bev_g, valid_bev_g))
    loss, loss_parts = compute_three_class_loss(
        logits, class_index, valid_bev_g, class_weights
    )
    return loss, loss_parts, compute_free_metrics(logits, seg_bev_g, vis_bev_g, valid_bev_g)
```

`default_class_weights`가 `load_labels`를 인자로 받는 이유: `robot_simplebev.load_masked_labels`와 SynWoodScape 쪽 대응물이 서명은 같지만 모듈이 다르다. 데이터셋 모듈을 import하면 이 모듈이 데이터셋에 묶여 Phase 4에서 재사용할 수 없다.

- [ ] **Step 4: 통과를 확인한다**

Run: `python -m pytest tests/common/test_three_class_metrics.py -v`
Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add projects/common/three_class_metrics.py tests/common/test_three_class_metrics.py
git commit -m "Add masked weighted cross-entropy for the three-class head"
```

---

### Task 16: 트레이너에 `--head` 스위치와 모듈 개칭

**Files:**
- Modify: `tools/train_robot_bev.py`
- Rename: `projects/common/two_head_metrics.py` → `projects/common/bev_occupancy_metrics.py`
- Modify: `tools/train_synwoodscape.py`, `tests/tools/test_train_synwoodscape_logging.py` (import 경로)

**Interfaces:**
- Consumes: Task 14, 15
- Produces: `main(..., head="two_head")` — `"two_head"` 또는 `"three_class"`

- [ ] **Step 1: 개칭을 `git mv`로 수행하고 import를 고친다**

```bash
git mv projects/common/two_head_metrics.py projects/common/bev_occupancy_metrics.py
grep -rl "two_head_metrics" --include="*.py" projects tools tests | xargs sed -i 's/two_head_metrics/bev_occupancy_metrics/g'
python -m pytest tests/ -q
```

Expected: 전부 passed. 개칭만 했으므로 동작은 그대로여야 한다.

- [ ] **Step 2: `--head` 스위치를 넣는다**

`tools/train_robot_bev.py`의 `main`에 `head="two_head"` 인자를 추가한다. 두 경로가 **같은 `free_metrics` 딕셔너리**를 돌려주므로 로깅·checkpoint 선택·TensorBoard 코드는 분기하지 않는다 — 이것이 A/B가 공정한 근거다. 분기는 세 곳뿐이다:

```python
    if head not in ("two_head", "three_class"):
        raise ValueError(f"head는 'two_head' 또는 'three_class'여야 한다: {head}")

    # (1) 모델
    model_cls = TwoHeadSegnet if head == "two_head" else ThreeClassSegnet
    model = model_cls(
        Z, Y, X, vox_util, use_radar=False, use_lidar=False,
        do_rgbcompress=True, encoder_type=encoder_type, rand_flip=False,
    ).to(device)

    # (2) 초기화. 3-class는 최종 1x1 conv의 형상이 달라 그 키만 건너뛴다.
    if init_checkpoint:
        if head == "two_head":
            load_initial_weights(model, init_checkpoint, device)
        else:
            report = load_trunk_weights(model, init_checkpoint, device)
            print(_c(_Ansi.CYAN,
                     f" trunk transfer: loaded {report['loaded']} tensors,"
                     f" skipped {len(report['skipped'])}"))
            # 세그멘테이션 head 말고 다른 것이 빠지면 pretrain이 조용히 안 먹은 것이다.
            unexpected = [n for n in report["skipped"] if "segmentation_head" not in n]
            if unexpected:
                raise RuntimeError(f"trunk 키가 빠졌다: {unexpected[:5]}")

    # (3) batch step
    if head == "two_head":
        step = lambda batch: two_head_run_batch(  # noqa: E731
            model, batch, vox_util, pos_weight_tensor, device, lambda_vis, vis_neg_weight
        )
    else:
        step = lambda batch: three_class_run_batch(  # noqa: E731
            model, batch, vox_util, class_weights, device
        )
```

`two_head_run_batch`는 9-튜플, `three_class_run_batch`는 3-튜플을 돌려준다. 학습 루프와 `_evaluate`가 두 형태를 모두 받도록 얇은 어댑터를 하나 둔다:

```python
def _normalise_step_output(head, output):
    """두 formulation의 반환을 `(loss, loss_parts, free_metrics, legacy)`로 통일한다.

    `legacy`는 2-head 전용 진단(`iou_drivable` 등)이며 3-class에서는 None이다. 로깅
    함수들이 None을 이미 받아들이므로(`occ_metrics=None`, `free_metrics=None`) 호출부는
    분기하지 않는다.
    """
    if head == "two_head":
        loss, parts, d_iou, o_iou, o_count, vis_m, occ_m, deploy, free_m = output
        return loss, parts, free_m, {
            "d_iou": float(d_iou.item()), "o_iou": float(o_iou.item()),
            "o_count": o_count, "vis": vis_m, "occ": occ_m, "deploy": deploy,
        }
    loss, parts, free_m = output
    return loss, parts, free_m, None
```

`class_weights`는 `main`에서 한 번 계산한다:

```python
    class_weights = default_class_weights(
        train_samples, permanent_blind, invalid, load_masked_labels
    )
```

그리고 배너에 `f" class weights (unknown/free/occupied) = {class_weights.tolist()}"`를 추가한다.

- [ ] **Step 3: 두 경로가 같은 지표를 내는지 확인한다**

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train_robot_bev.py --head=two_head \
    --exp_name=head_switch_smoke --train_sequences=raws1 --val_sequences=raws2 --num_epochs=1
CUDA_VISIBLE_DEVICES=1 python tools/train_robot_bev.py --head=three_class \
    --exp_name=head_switch_smoke3 --train_sequences=raws1 --val_sequences=raws2 --num_epochs=1
```

Expected: 두 실행 모두 같은 필드(`val_iou_free`, `fatal`, baseline 델타)를 찍고 `partition` 경고가 없다. 값 자체는 달라도 된다.

- [ ] **Step 4: 전체 테스트**

Run: `python -m pytest tests/ -q`
Expected: 전부 passed

- [ ] **Step 5: 커밋**

```bash
git add -A projects/common tools/train_robot_bev.py tools/train_synwoodscape.py tests/
git commit -m "Switch between the two-head and three-class formulations at runtime"
```

---

### Task 17: overfit 테스트 (게이트 D10)

**Files:** 없음 (실행과 기록만)

데이터가 154장뿐인 상황에서 **배선이 맞는지 검증하는 가장 강력한 수단**이다. 4샘플을 외우지 못하면 loss나 라벨 배선이 틀린 것이지 데이터가 부족한 것이 아니다.

- [ ] **Step 1: 4샘플만으로 200 epoch 돌린다**

```bash
conda activate bev-chamdog
mkdir -p /tmp/overfit4/seq/occupancy_npy /tmp/overfit4/seq/visibility_npy /tmp/overfit4/seq/rgb_images
python - <<'PY'
import shutil
from pathlib import Path
src, dst = Path("dataset/sj_datasets/raws1"), Path("/tmp/overfit4/seq")
for sample_id in sorted(p.stem for p in (src / "occupancy_npy").glob("*.npy"))[:4]:
    for sub in ("occupancy_npy", "visibility_npy"):
        shutil.copy(src / sub / f"{sample_id}.npy", dst / sub / f"{sample_id}.npy")
    shutil.copytree(src / "rgb_images" / sample_id, dst / "rgb_images" / sample_id,
                    dirs_exist_ok=True)
print("copied 4 samples")
PY

CUDA_VISIBLE_DEVICES=1 python tools/train_robot_bev.py --head=three_class \
    --exp_name=overfit4 --dataset_root=/tmp/overfit4 --train_sequences=seq \
    --val_sequences="" --val_tail_fraction=0.0 \
    --num_epochs=200 --batch_size=4 --lr=1e-4 --num_workers=2 --init_checkpoint=None
```

`val_sequences`가 비어 있으므로 val split이 없다. **train `iou_free`** 를 본다.

- [ ] **Step 2: 게이트를 확인한다**

train `iou_free`가 200 epoch 안에 **≥ 0.98**에 도달해야 한다. 도달하지 못하면:
- `partition_defects`가 0인지 먼저 본다 (0이 아니면 Task 1)
- `loss_parts`에서 어느 클래스가 지배하는지 본다 (`occupied`가 0에 붙어 있으면 Task 15의 클래스 가중치)
- 그래도 안 되면 `load_trunk_weights`가 아니라 **from scratch**로 다시 돌려본다. pretrain 가중치가 문제인지 배선이 문제인지 갈린다

**여기를 통과하지 못하면 Task 18로 넘어가지 않는다.**

- [ ] **Step 3: 결과를 기록하고 커밋한다**

`docs/free_space_metric_migration.md`에 "3-class overfit 검증" 절을 추가하고 도달한 epoch과 최종 train `iou_free`를 적는다.

```bash
git add docs/free_space_metric_migration.md
git commit -m "Record the three-class overfit sanity check"
```

---

### Task 18: 3-class vs 2-head A/B (게이트)

**Files:** 없음 (실행과 기록만)

- [ ] **Step 1: Phase 2 기준선과 같은 조건으로 3-class를 학습한다**

```bash
conda activate bev-chamdog
CUDA_VISIBLE_DEVICES=1 python tools/train_robot_bev.py --head=three_class \
    --exp_name=threeclass_ab --train_sequences=raws1,raws3,rawos1 --val_sequences=raws2 \
    --num_epochs=60 --batch_size=8 --lr=1e-4 --weight_decay=1e-7 --num_workers=8 \
    --encoder_type=res101 --augment=False --val_freq_epochs=1 --save_freq_epochs=10 \
    --init_checkpoint=runs/synwoodscape_twohead/ckpt/twohead_pretrain_photo_aug_res101_bs16_lr3e-04_260814_150556/model_best-000000046.pth \
    2>&1 | tee runs/threeclass_ab.log
```

Task 13의 기준선과 **바뀐 것은 `--head`와 loss뿐**이다. lr·epoch·batch·augment·split을 전부 동일하게 유지한다.

- [ ] **Step 2: 게이트를 판정한다**

| 조건 | 판정 |
|---|---|
| 3-class best `iou_free` ≥ 2-head 기준선 | **통과** — Phase 4(pretrain 재학습) 스펙으로 넘어간다 |
| 기준선 미만이지만 `fatal_rate`가 뚜렷이 낮음 | **보류** — 두 지표의 trade-off를 문서화하고 사용자 판단을 받는다 |
| 둘 다 나쁨 | **정지** — Phase 4로 넘어가지 않는다. Task 15의 클래스 가중치와 `load_trunk_weights`의 `skipped` 목록부터 다시 본다 |

**주의**: val이 41 프레임뿐이라 `iou_free` 차이가 0.02 이내면 노이즈로 취급한다. 스펙 §12대로 **이 판정은 데이터가 늘어난 뒤 재확인해야 한다** — 문서에 그 문장을 반드시 남긴다.

- [ ] **Step 3: 결과를 기록하고 커밋한다**

`docs/free_space_metric_migration.md`에 "Phase 3 A/B" 절을 추가한다. 두 run의 best `iou_free` · `fatal_rate` · `range_abs_p50` · 링별 `iou_free`를 한 표에 넣고, 판정과 그 근거, 그리고 "val 41 프레임 기준이므로 데이터 확장 후 재확인 필요"를 적는다.

```bash
git add docs/free_space_metric_migration.md
git commit -m "Record the three-class versus two-head comparison"
```

---

## Phase 3 게이트

1. `python -m pytest tests/ -q` 전부 통과
2. Task 17의 overfit이 train `iou_free` ≥ 0.98
3. Task 18의 A/B 결과와 판정이 `docs/free_space_metric_migration.md`에 기록됨
4. Phase 4(SynWoodScape 3-class pretrain 재학습)는 **별도 스펙으로 분리해 진행한다** — 이 계획의 범위 밖이다

---

## 부록: 스펙 대비 커버리지

| 스펙 항목 | 태스크 |
|---|---|
| §4 4-way 분해 단일 출처 | Task 1 |
| §5.1 M1 `iou_free` | Task 2 |
| §5.1 M2 `fatal_rate` / M2b `free_miss_rate` | Task 2 |
| §5.1 M3 `range_err` | Task 3, 4 |
| §5.1 M4 거리 링 분해 | Task 5 |
| §5.2 M3 구현 메모 (첫 free부터, censored 분리, 캐시) | Task 3, 4 |
| §5.3 옛 지표를 진단으로 강등 | Task 10 |
| §5.4 `val_score = iou_free` | Task 11 |
| §6 trivial baseline 자동 병기 | Task 7, 10 |
| §7 라벨 불변 (파생만) | Task 1, 6 — 새 라벨 파일을 만드는 태스크가 없다는 것 자체가 준수의 증거 |
| §8 3-class 전환 | Task 14, 15, 16 |
| §8.1 polar head 보류 | 태스크 없음 (의도적) |
| §9 (A) 지표 단위 테스트 | Task 1–5 |
| §9 (B) 라벨 무결성 | Task 6 |
| §9 (C) 기존 체크포인트 교차 검증 | Task 8 |
| §9 (D) 3-class 검증 | Task 14, 16, 17 |
| §9 (E) 궤적 기반 검증 | 범위 밖 (스펙 §11) |
| §10.1 시각화 | Task 12 |
| §12 censored 비율 / 360 vs 720 bin | Task 6 |
