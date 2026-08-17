import numpy as np
import pytest
import torch
import torch.nn as nn

from projects.common.free_space import FREE, OCCUPIED, UNKNOWN
from projects.common.free_space_metrics import (
    fatal_rate,
    free_miss_rate,
    iou_free,
    weighted_mean,
)
from projects.common.polar import RayIndex
from tools.rescore_checkpoints import (
    format_markdown_table,
    load_checkpoint_state_dict,
    main,
    score_split,
)


def test_markdown_table_puts_the_baseline_next_to_every_model_row():
    """baseline이 같은 표에 없으면 숫자를 혼자 읽게 되고, 그게 이번 결함의 원인이었다."""
    rows = [
        {"name": "threeclass_ab", "split": "rawos3", "iou_free": 0.774,
         "baseline_iou_free": 0.398, "all_free_iou_free": 0.17,
         "fatal_rate": 0.135, "baseline_fatal_rate": 0.1678,
         "free_miss_rate": 0.115},
    ]

    text = format_markdown_table(rows)

    assert "iou_free" in text
    assert "baseline" in text
    assert "0.774" in text and "0.398" in text
    # `fatal_rate`는 Phase 3에서 유일하게 후퇴한 지표라 표에서 빠지면 안 된다.
    assert "0.135" in text
    # 열 이름과 값의 순서가 어긋나면(예: iou_free/baseline_iou_free 값이 바뀌어도) 위 assert들은
    # 전부 통과한다 -- 두 값이 나란히 올바른 순서로 붙어 있는지 문자열 그대로 고정한다.
    assert "| 0.774 | 0.398 |" in text


class _ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.foo = nn.Linear(4, 4)


class _ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.bar = nn.Linear(4, 4)


def test_load_checkpoint_state_dict_rejects_mismatched_architecture(tmp_path):
    """`strict=False`가 안 맞는 키를 조용히 버리고 나머지만 얹으면, 체크포인트 경로가
    틀리거나 `encoder_type`이 학습 때와 달라도 실행이 끝까지 가서 그럴듯한(그러나 일부
    무작위 초기화인) 숫자를 뱉는다. 여기서 즉시 실패해야 그 배선 실수를 잡는다."""
    ckpt_path = tmp_path / "mismatched.pth"
    torch.save({"model_state_dict": _ModelB().state_dict()}, ckpt_path)

    with pytest.raises(RuntimeError):
        load_checkpoint_state_dict(_ModelA(), ckpt_path, "cpu")


def test_load_checkpoint_state_dict_accepts_matching_architecture(tmp_path):
    """예외가 없는 것만으로는 부족하다 -- 실제로 저장한 값이 얹혔는지까지 확인한다."""
    ckpt_path = tmp_path / "matching.pth"
    reference = _ModelA()
    torch.nn.init.constant_(reference.foo.weight, 3.5)
    torch.save({"model_state_dict": reference.state_dict()}, ckpt_path)

    model = _ModelA()
    load_checkpoint_state_dict(model, ckpt_path, "cpu")  # 예외 없이 통과해야 한다

    assert torch.equal(model.foo.weight, reference.foo.weight)


def test_load_checkpoint_state_dict_accepts_bare_state_dict_without_wrapper(tmp_path):
    """`{"model_state_dict": ...}`로 감싸지 않은 옛 형식(맨 state_dict)도 받아야 한다 --
    `state.get("model_state_dict", state)`의 fallback 분기다."""
    ckpt_path = tmp_path / "bare.pth"
    reference = _ModelA()
    torch.nn.init.constant_(reference.foo.bias, 1.25)
    torch.save(reference.state_dict(), ckpt_path)  # wrapper 없이 그대로 저장

    model = _ModelA()
    load_checkpoint_state_dict(model, ckpt_path, "cpu")

    assert torch.equal(model.foo.bias, reference.foo.bias)


# --------------------------------------------------------------------------------
# score_split 배선 테스트 -- 실제 인코더/데이터셋/GPU 없이 가짜 모델과 loader로 검증한다.
# --------------------------------------------------------------------------------

def _grid(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).view(1, 1, 4, 4)


def _class_logits(class_grid) -> torch.Tensor:
    """(4,4) 클래스 인덱스 격자 -> (1, 3, 4, 4) logits. argmax가 그 클래스를 고르게 만든다."""
    index = torch.tensor(class_grid, dtype=torch.long)
    logits = torch.full((1, 3, 4, 4), -5.0)
    for row in range(4):
        for col in range(4):
            logits[0, int(index[row, col]), row, col] = 5.0
    return logits


def _batch(seg_gt, vis_gt, valid_gt) -> dict:
    dummy = torch.zeros(1, 1, 1, 1, 1)  # rgb/pix/cam 값은 stub 모델이 보지 않으므로 형상만 채운다
    return {
        "rgb_camXs": dummy, "pix_T_cams": dummy, "cam0_T_camXs": dummy,
        "seg_bev_g": _grid(seg_gt), "vis_bev_g": _grid(vis_gt), "valid_bev_g": _grid(valid_gt),
    }


def _dummy_rays() -> RayIndex:
    """range_error가 요구하는 형상만 채운 광선 인덱스 -- 이 테스트는 range/rings 값을
    보지 않으므로 아무 것도 격자 안(inside)이 아니게 둬 계산 없이 통과시킨다."""
    return RayIndex(
        rows=np.zeros((1, 1), dtype=np.int64), cols=np.zeros((1, 1), dtype=np.int64),
        radii_m=np.zeros(1), inside=np.zeros((1, 1), dtype=bool),
    )


class _StubThreeClassModel(nn.Module):
    """forward가 인자를 무시하고 미리 정해둔 3-class logits를 loader 순서대로 돌려준다 --
    실제 인코더·리프팅 없이 `score_split`의 집계 배선만 검증하기 위한 가짜 모델이다."""

    def __init__(self, logits_sequence):
        super().__init__()
        self._logits = list(logits_sequence)
        self._next = 0

    def forward(self, rgb, pix_T_cams, cam0_T_camXs, vox_util):
        logits = self._logits[self._next]
        self._next += 1
        return None, None, logits, None, None


_ALL_ONES = [[1] * 4] * 4

# sample1 -- 네 종류의 셀이 한 번씩 등장하도록 짰다.
#   (0,1) GT occupied / pred occupied  -- 맞게 막았다고 부른 셀
#   (1,0) GT occupied / pred FREE      -- fatal 1건 (planner를 위험하게 하는 오류)
#   (2,0) GT free     / pred UNKNOWN   -- free_miss 1건 (보수적 오류)
#   (0,2) valid=0                      -- 마스킹이 빠지면 여기서 허위 fatal이 하나 더 생긴다
_SAMPLE1_PRED = [[FREE, OCCUPIED, FREE, FREE],
                 [FREE, FREE, FREE, FREE],
                 [UNKNOWN, FREE, FREE, FREE],
                 [FREE, FREE, FREE, FREE]]
_SAMPLE1_SEG = [[1, 0, 0, 1],
                [0, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]]
_SAMPLE1_VISGT = _ALL_ONES
_SAMPLE1_VALID = [[1, 1, 0, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1]]

# score_split이 argmax로 만들어야 하는 free 마스크를 테스트가 **독립적으로** 손으로 적는다.
# 이걸 logits에서 다시 유도하면 score_split의 유도 방식이 틀렸을 때 기대값도 같이 틀려
# 테스트가 통과해 버린다.
_SAMPLE1_PRED_FREE = _grid([[1, 0, 1, 1],
                            [1, 1, 1, 1],
                            [0, 1, 1, 1],
                            [1, 1, 1, 1]]).bool()


def test_score_split_derives_free_from_argmax_and_applies_the_valid_mask():
    """`score_split`이 3-class logits에서 free 마스크를 어떻게 뽑는지 고정한다.

    기대값은 이미 단위 테스트가 있는 원시 지표(`iou_free`/`fatal_rate`/`free_miss_rate`)에
    **손으로 적은** `pred_free`를 넣어 만든다. 따라서 이 테스트가 보는 것은 원시 지표의
    정확성이 아니라 score_split이 (a) `argmax == FREE`로 예측을 만들고 (b) `valid`로
    마스킹하고 (c) 배치 간 가중평균을 올바른 weight로 내는지다. `argmax`의 dim을 바꾸거나
    FREE 대신 OCCUPIED를 고르거나 valid 마스킹을 빼면 깨진다.
    """
    batch1 = _batch(_SAMPLE1_SEG, _SAMPLE1_VISGT, _SAMPLE1_VALID)
    batch2 = _batch(_ALL_ONES, _ALL_ONES, _ALL_ONES)
    all_free = _grid(_ALL_ONES).bool()

    model = _StubThreeClassModel([
        _class_logits(_SAMPLE1_PRED),
        _class_logits([[FREE] * 4] * 4),
    ])
    result = score_split(
        model, [batch1, batch2], vox_util=None, rays=_dummy_rays(),
        ring_masks=[], device="cpu", constant_map=np.zeros((4, 4), dtype=bool),
    )

    cases = [
        (_SAMPLE1_PRED_FREE, _grid(_SAMPLE1_SEG), _grid(_SAMPLE1_VISGT), _grid(_SAMPLE1_VALID)),
        (all_free, _grid(_ALL_ONES), _grid(_ALL_ONES), _grid(_ALL_ONES)),
    ]
    ious, iou_counts = [], []
    fatals, fatal_denoms, misses, miss_denoms = [], [], [], []
    for pred_free, seg, vis, valid in cases:
        gt_free = seg.bool() & vis.bool() & valid.bool()
        value, count = iou_free(pred_free, gt_free, valid)
        ious.append(value)
        iou_counts.append(count)
        value, denom = fatal_rate(pred_free, gt_free, valid)
        fatals.append(value)
        fatal_denoms.append(denom)
        value, denom = free_miss_rate(pred_free, gt_free, valid)
        misses.append(value)
        miss_denoms.append(denom)

    assert result["iou_free"] == pytest.approx(weighted_mean(ious, iou_counts), abs=1e-6)
    assert result["fatal_rate"] == pytest.approx(weighted_mean(fatals, fatal_denoms), abs=1e-6)
    assert result["free_miss_rate"] == pytest.approx(
        weighted_mean(misses, miss_denoms), abs=1e-6
    )
    # 위 세 값이 서로 우연히 같아 배선 오류를 덮지 않도록, 실제 숫자가 자명하지 않은지 본다.
    assert 0.0 < result["fatal_rate"] < 1.0
    assert 0.0 < result["free_miss_rate"] < 1.0
    assert 0.0 < result["iou_free"] < 1.0


def test_score_split_free_prediction_responds_to_the_predicted_class():
    """예측 클래스를 전부 FREE / 전부 UNKNOWN으로 뒤집으면 `iou_free`가 1.0과 0.0으로 갈려야 한다.

    logits를 아예 보지 않는(예: GT를 그대로 예측으로 쓰는) 배선 오류라면 둘이 같아진다.
    """
    def run(class_id):
        model = _StubThreeClassModel([_class_logits([[class_id] * 4] * 4)])
        return score_split(
            model, [_batch(_ALL_ONES, _ALL_ONES, _ALL_ONES)], vox_util=None,
            rays=_dummy_rays(), ring_masks=[], device="cpu",
            constant_map=np.zeros((4, 4), dtype=bool),
        )

    assert run(FREE)["iou_free"] == pytest.approx(1.0, abs=1e-9)
    assert run(UNKNOWN)["iou_free"] == pytest.approx(0.0, abs=1e-9)
    # OCCUPIED로 예측해도 free는 하나도 없으므로 0이어야 한다 (FREE와 OCCUPIED를 혼동하면 1.0이 된다).
    assert run(OCCUPIED)["iou_free"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------------
# main()이 실제로 load_checkpoint_state_dict를 호출하는지 -- 데이터셋/GPU 없이.
# --------------------------------------------------------------------------------

class _TinyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.p = nn.Linear(1, 1)


class _EmptyDataset:
    """빈 val split -- score_split의 for 루프가 0번 돈다. RobotBEVDataset 자리를 대신한다."""

    def __init__(self, *args, **kwargs):
        self.cameras = []

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError(index)


def test_main_calls_load_checkpoint_state_dict_and_propagates_its_error(tmp_path, monkeypatch):
    """`load_checkpoint_state_dict`를 다시 인라인 `strict=False` 코드로 되돌리는 변형
    (mutation 2)이면 이 테스트가 깨져야 한다. 데이터셋 로딩·인코더·GPU는 전부 가짜로
    바꿔치기하고, 체크포인트 로딩 호출부만 실제 `main()` 코드 경로를 그대로 태운다."""
    import tools.rescore_checkpoints as tool

    calls = []

    def _fake_load_checkpoint_state_dict(model, checkpoint_path, device):
        calls.append((model, checkpoint_path, device))
        raise RuntimeError("stub-load-checkpoint-called")

    monkeypatch.setattr(tool, "list_sequence_samples", lambda root: [])
    monkeypatch.setattr(tool, "build_bev_masks", lambda *a, **kw: (None, None))
    monkeypatch.setattr(tool, "constant_free_map", lambda masks: np.zeros((1, 1), dtype=bool))
    monkeypatch.setattr(tool, "RobotBEVDataset", _EmptyDataset)
    monkeypatch.setattr(tool, "build_double_sphere_vox_util", lambda *a, **kw: None)
    monkeypatch.setattr(tool, "ThreeClassSegnet", _TinyModel)
    monkeypatch.setattr(tool, "load_checkpoint_state_dict", _fake_load_checkpoint_state_dict)

    # 실제로 로드 가능한 작은 체크포인트 -- 변형 2가 적용돼 인라인 코드로 되돌아가면
    # (키가 안 맞아도 예외를 던지지 않으므로) 이 파일이 조용히 "로드"되고 끝까지 실행된다.
    ckpt_path = tmp_path / "tiny.pth"
    torch.save({"model_state_dict": {"unrelated.weight": torch.zeros(1)}}, ckpt_path)

    with pytest.raises(RuntimeError, match="stub-load-checkpoint-called"):
        tool.main(
            checkpoint=str(ckpt_path), train_sequences="fake", val_sequences="fake",
            batch_size=1, num_workers=0, device="cpu",
        )

    assert len(calls) == 1  # main()이 정확히 이 이름으로, 한 번 호출했는지 확인
