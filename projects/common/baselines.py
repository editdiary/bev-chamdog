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
    """`(H, W)` -> `(B, 1, H, W)`. 지표 함수가 batch 텐서만 받으므로 형상을 맞춘다.

    주의: expand()는 batch 차원에서 비연속적 뷰(non-contiguous view)를 반환하므로
    메모리를 공유한다 — 이는 의도된 것이다(baseline이 모든 샘플에서 동일하므로 B개 복사는
    낭비). 따라서 반환값을 제자리에서 쓰면 안 된다.
    """
    tensor = torch.from_numpy(np.asarray(map_2d, dtype=bool)).to(device)
    return tensor.view(1, 1, *tensor.shape).expand(batch_size, 1, *tensor.shape)
