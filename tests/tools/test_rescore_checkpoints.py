import numpy as np
import pytest
import torch
import torch.nn as nn

from projects.common.free_space_metrics import weighted_mean
from projects.common.polar import RayIndex
from projects.common.bev_occupancy_metrics import compute_drivable_and_obstacle_iou
from tools.rescore_checkpoints import (
    format_markdown_table,
    load_checkpoint_state_dict,
    main,
    score_split,
)


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
    # 열 이름과 값의 순서가 어긋나면(예: iou_free/baseline_iou_free 값이 바뀌어도) 위 assert들은
    # 전부 통과한다 -- 두 값이 나란히 올바른 순서로 붙어 있는지 문자열 그대로 고정한다.
    assert "| 0.850 | 0.673 |" in text


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


def _two_head(occ_grid, vis_grid) -> torch.Tensor:
    return torch.cat([_grid(occ_grid), _grid(vis_grid)], dim=1)


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


class _StubTwoHeadModel(nn.Module):
    """forward가 인자를 무시하고 미리 정해둔 두 head logits를 loader 순서대로 돌려준다 --
    실제 인코더·리프팅 없이 `score_split`의 집계 배선만 검증하기 위한 가짜 모델이다."""

    def __init__(self, two_head_logits_sequence):
        super().__init__()
        self._logits = list(two_head_logits_sequence)
        self._next = 0

    def forward(self, rgb, pix_T_cams, cam0_T_camXs, vox_util):
        logits = self._logits[self._next]
        self._next += 1
        return None, None, logits, None, None


_ORD_LOGIT = [[3.0] * 4] * 4   # sigmoid든 raw든 threshold 0.5를 확실히 넘는 값 -- "배경" 셀
_ORD_ONES = [[1] * 4] * 4

# sample1: 특수 셀 4개
#   (0,0) occ_logit=0.2 -- sigmoid(0.2)=0.550>0.5(True)지만 raw 0.2>0.5는 False.
#     `torch.sigmoid`를 빼는 변형(mutation 1)이 정확히 이 셀에서만 갈린다.
#   (0,1) occ 높음(3.0)/vis 낮음(-3.0) -- occ만 보고 free를 판단하면(AND 대신 OR 등) 틀리는 셀.
#   (0,2) valid=0 -- masking이 빠지면 예측 free(occ·vis 둘 다 높음)와 GT obstacle이 어긋나
#     허위 fatal이 하나 생기므로 masking이 빠졌는지 드러난다.
#   (1,0) GT obstacle인데 occ·vis 예측은 둘 다 높음 -- sigmoid 유무와 무관한 "진짜" fatal 1건.
_SAMPLE1_OCC = [[0.2, 3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0, 3.0]]
_SAMPLE1_VIS = [[3.0, -3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0, 3.0]]
_SAMPLE1_SEG = [[1, 1, 0, 1],
                [0, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]]
_SAMPLE1_VISGT = [[1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1]]
_SAMPLE1_VALID = [[1, 1, 0, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1]]


def test_score_split_applies_sigmoid_and_masking_before_computing_free_metrics():
    """`torch.sigmoid`를 빼고 raw logit을 0.5로 자르는 변형(mutation 1)이면 이 테스트가
    깨져야 한다. 기대값은 손으로 뺀 것이다 (이 함수 docstring과 리포트에 유도 과정을 남긴다):

    sample1(16셀 중 특수 4셀, 위 주석 참고): intersection=13, union=15 -> IoU=13/15.
    |pred|=14, fatal(오탐)=1건(=(1,0)) -> fatal_rate=1/14. |gt|=14, miss=1건(=(0,1)) -> 1/14.
    sample2(전부 배경 16셀): IoU=1.0, fatal=0/16, miss=0/16.
    배치 가중평균(각 배치=샘플 1개, count=1과 |pred|/|gt|로 가중):
      iou_free = (13/15 + 1.0) / 2
      fatal_rate = (1/14*14 + 0*16) / (14+16) = 1/30
      free_miss_rate = (1/14*14 + 0*16) / (14+16) = 1/30
    """
    logits1 = _two_head(_SAMPLE1_OCC, _SAMPLE1_VIS)
    logits2 = _two_head(_ORD_LOGIT, _ORD_LOGIT)
    batch1 = _batch(_SAMPLE1_SEG, _SAMPLE1_VISGT, _SAMPLE1_VALID)
    batch2 = _batch(_ORD_ONES, _ORD_ONES, _ORD_ONES)

    model = _StubTwoHeadModel([logits1, logits2])
    result = score_split(
        model, [batch1, batch2], vox_util=None, rays=_dummy_rays(),
        ring_masks=[], device="cpu", constant_map=np.zeros((4, 4), dtype=bool),
    )

    assert result["iou_free"] == pytest.approx((13 / 15 + 1.0) / 2, abs=1e-5)
    assert result["fatal_rate"] == pytest.approx(1 / 30, abs=1e-5)
    assert result["free_miss_rate"] == pytest.approx(1 / 30, abs=1e-5)

    # iou_drivable/iou_obstacle은 이미 신뢰된 compute_drivable_and_obstacle_iou를 같은 인자로
    # 다시 불러 기대값을 만든다 -- 이 함수 자체의 정확성이 아니라 score_split이 인자·마스크를
    # 바꿔치기하지 않았는지만 본다.
    mask1 = _grid(_SAMPLE1_VISGT) * _grid(_SAMPLE1_VALID)
    mask2 = _grid(_ORD_ONES) * _grid(_ORD_ONES)
    d1, o1, c1 = compute_drivable_and_obstacle_iou(_grid(_SAMPLE1_OCC), _grid(_SAMPLE1_SEG), mask1)
    d2, o2, c2 = compute_drivable_and_obstacle_iou(_grid(_ORD_LOGIT), _grid(_ORD_ONES), mask2)
    expected_d = float(np.mean([d1.item(), d2.item()]))
    expected_o = weighted_mean([o1.item(), o2.item()], [c1, c2])
    assert result["iou_drivable"] == pytest.approx(expected_d, abs=1e-6)
    assert result["iou_obstacle"] == pytest.approx(expected_o, abs=1e-6)


def test_score_split_iou_drivable_ignores_predicted_visibility():
    """`iou_drivable`/`iou_obstacle`은 예측 vis head가 아니라 GT `vis_g * valid`로 마스킹된다
    (`compute_drivable_and_obstacle_iou`에 넘기는 세 번째 인자). occ/GT를 고정하고
    vis_logit만 뒤집어도 두 값이 그대로인지로 이를 확인한다 -- 반대로 `iou_free`는 vis
    예측을 그대로 반영해 달라져야 한다(둘 다 안 바뀌면 애초에 vis가 안 쓰인 것이다)."""

    def run(vis_logit_grid):
        model = _StubTwoHeadModel([_two_head(_ORD_LOGIT, vis_logit_grid)])
        batch = _batch(_ORD_ONES, _ORD_ONES, _ORD_ONES)
        return score_split(
            model, [batch], vox_util=None, rays=_dummy_rays(),
            ring_masks=[], device="cpu", constant_map=np.zeros((4, 4), dtype=bool),
        )

    low_vis = run([[-3.0] * 4] * 4)   # 전부 안 보인다고 예측
    high_vis = run(_ORD_LOGIT)        # 전부 보인다고 예측

    assert low_vis["iou_drivable"] == pytest.approx(high_vis["iou_drivable"], abs=1e-9)
    # 대조: iou_free는 vis 예측에 실제로 반응해야 한다 (occ만 보는 결함이면 둘 다 안 바뀐다).
    assert low_vis["iou_free"] == pytest.approx(0.0, abs=1e-9)
    assert high_vis["iou_free"] == pytest.approx(1.0, abs=1e-9)


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
    monkeypatch.setattr(tool, "TwoHeadSegnet", _TinyModel)
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
