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


def _as_bool(tensor):
    """확률·0/1 float·bool을 모두 받는다 -- 예측(sigmoid 출력)과 GT가 같은 함수를 타야 한다.

    **torch 텐서와 numpy 배열을 모두 받는다.** 학습 루프는 텐서를 넘기지만, 베이스라인·클래스
    가중치를 실측하는 코드는 `.npy`를 읽은 직후의 numpy 배열을 넘긴다. 둘 다 같은 정의를
    타야 하므로 여기서 dtype만 보고 처리한다(`tests/common/test_free_space.py`가 고정한다).
    """
    return tensor if tensor.dtype == torch.bool else tensor > 0.5


def decompose(occ, vis, valid) -> dict:
    """occupancy / visibility / valid -> 세 bool 마스크.

    입력은 `(B, 1, H, W)` torch 텐서이거나 `(H, W)` numpy 배열이며, 출력은 입력과 같은 종류다.
    **`occ=1`이 free(drivable)이고 `occ=0`이 장애물이다** -- 이름과 반대이므로 주의한다.
    """
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
