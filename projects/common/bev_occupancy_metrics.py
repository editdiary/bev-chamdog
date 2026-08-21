"""3-class BEV 학습의 데이터셋 비의존 지표·로깅 -- SynWoodScape pretrain과 자체 데이터셋
fine-tuning이 공유한다.

원래는 `tools/train_synwoodscape.py` 안에 있었다. 자체 데이터셋 fine-tuning 스크립트가
같은 지표를 그대로 써야 하는데(그래야 pretrain과 fine-tune 숫자를 나란히 읽을 수 있다),
스크립트끼리 import하면 pretrain 스크립트를 손댈 때 fine-tune이 같이 깨진다. 여기 옮겨서
양쪽이 대등하게 의존하도록 한다.

여기 있는 것은 전부 텐서나 지표 dict만 받는다 -- 데이터셋 경로나 라벨 파일 규약에 의존하는
것은 각 스크립트에 남는다. loss와 batch step 자체는 `three_class_metrics.py`에 있다.

**2-head 정식화는 제거됐다.** Phase 3 A/B(`docs/archive/free_space_metric_migration.md` §8)에서
3-class로 확정한 뒤, 두 정식화를 병행 유지하는 비용이 사라졌기 때문이다. 그래서 이 모듈에
남은 것은 두 학습 경로가 **공유**하는 것뿐이다: 터미널 출력 헬퍼, epoch 로그 포매터,
체크포인트 선택 기준, free-space 지표의 epoch 집계.
"""
import math
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "third_party/models/simple_bev") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "third_party/models/simple_bev"))

from projects.common.free_space_metrics import (  # noqa: E402
    metrics_per_ring,
    range_error,
    summarize_range_error,
    summarize_range_error_by_gt_range,
    weighted_mean,
)
from projects.common.occupied_metrics import (  # noqa: E402
    DEFAULT_TOLERANCES_M,
    summarize_tolerance_f1,
    tolerance_counts,
    tolerance_key,
)


class _Ansi:
    """터미널 색 강조용 최소 ANSI 코드 -- 별도 의존성(rich/colorama) 없이 표준 출력에 직접 씀."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def _color_enabled() -> bool:
    """색은 **터미널로 직접 출력할 때만** 켠다.

    `... > train.log`나 `| tee`로 로그를 남기는 경우가 많은데, 그 파일에 escape sequence가
    섞이면 나중에 grep/diff가 지저분해진다. 관례대로 `NO_COLOR`로 끄고 `FORCE_COLOR`로
    강제할 수 있다(파이프로 넘기면서 색을 보고 싶을 때: `FORCE_COLOR=1 ... | less -R`).
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def _c(color: str, text: str) -> str:
    if not _color_enabled():
        return text
    return f"{color}{text}{_Ansi.RESET}"


def _print_banner(lines) -> None:
    """제목과 `<-`로 표시된 줄만 강조한다 -- 전부 같은 색으로 칠하면 강조가 아니라 벽이 된다.

    `<-`는 두 학습 스크립트 모두 "이 숫자와 비교하라"는 뜻으로 쓰는 표시다.
    """
    rule = "=" * 60
    print(_c(_Ansi.BOLD + _Ansi.CYAN, rule))
    for i, line in enumerate(lines):
        if i == 0:
            print(_c(_Ansi.BOLD + _Ansi.CYAN, line))
        elif "<-" in line:
            print(_c(_Ansi.BOLD + _Ansi.YELLOW, line))
        else:
            print(line)
    print(_c(_Ansi.BOLD + _Ansi.CYAN, rule))


def _sep() -> str:
    return _c(_Ansi.DIM, " | ")


def _field(label: str, value, fmt: str = ".3f", *, emphasis: str = "") -> str:
    """`라벨(흐리게) 값(또렷하게)`. 라벨은 매 epoch 똑같이 반복되는 부분이라 눈이 값만
    따라가면 되도록 흐리게 깐다. NaN은 숫자처럼 안 보이게 `-`로 눕힌다."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return _c(_Ansi.DIM, f"{label} -")
    return f"{_c(_Ansi.DIM, label)} {_c(emphasis, format(value, fmt)) if emphasis else format(value, fmt)}"


def _format_row(tag: str, tag_color: str, fields) -> str:
    return f"  {_c(_Ansi.BOLD + tag_color, tag.ljust(6))}{_c(_Ansi.DIM, '|')} " + _sep().join(fields)




# loss 파트 표시 순서. CE의 클래스별 항이고, loss 함수가 돌려주는 키와 이름이 같아야 한다 --
# 옛 2-head 시절에는 이 자리를 loss_occ/loss_vis 두 칸이 차지하고 있어서, 3-class의 세 항 중
# 하나(unknown)가 로그에서 아예 보이지 않았다.
#
# **정식화마다 항의 집합이 다르므로 호출부가 이름을 넘긴다**(3-class는 unknown/free/occupied,
# binary는 not_free/free). 여기 하드코딩하면 binary의 항이 로그에서 통째로 사라진다. 기본값을
# 3-class로 두는 이유는 pretrain 스크립트와 기존 테스트가 그 형태를 전제하기 때문이다.
DEFAULT_LOSS_PART_NAMES = ("unknown", "free", "occupied")

# 몫 칸의 헤더에 쓰는 약자. 이름이 길어 `share not_free/free`로 쓰면 줄이 넘친다.
_PART_INITIALS = {"unknown": "u", "free": "f", "occupied": "o", "not_free": "nf",
                  "boundary": "b"}


def _format_share_field(loss_parts, part_names):
    """각 클래스가 총 loss에 기여하는 몫을 한 칸에 담는다.

    합이 1이라 숫자를 붙여 놓는 것이 가장 읽기 쉽다. 이것을 로그에 넣는 이유: 클래스별
    평균(`loss_*`)만으로는 "occupied가 셀의 1.1 %인데 총 loss의 67 %"라는 사실이 보이지 않아
    실측에서 셀 비율을 손으로 곱해 봐야 알 수 있었다
    (`docs/finetune_overfitting_diagnosis.md` §12).
    """
    values = [(loss_parts or {}).get(f"share_{name}") for name in part_names]
    if any(value is None or (isinstance(value, float) and math.isnan(value)) for value in values):
        return None
    label = "share " + "/".join(_PART_INITIALS.get(name, name) for name in part_names)
    return _field(label, "/".join(f"{value:.2f}" for value in values), fmt="")


def _format_metric_row(tag, tag_color, loss, loss_parts, free_metrics=None,
                       part_names=DEFAULT_LOSS_PART_NAMES):
    """train/val 한 줄. `iou_free`가 주 지표이고 나머지는 그것을 해석하기 위한 것이다."""
    fields = []
    if free_metrics is not None:
        fields += [
            _field("iou_free↑", free_metrics["iou_free"], emphasis=_Ansi.BOLD),
            _field("fatal↓", free_metrics["fatal_rate"]),
            _field("free_miss↓", free_metrics["free_miss_rate"]),
        ]
    fields.append(_field("loss_total↓", loss, ".4f"))
    parts = loss_parts or {}
    fields += [_field(f"loss_{name}↓", parts.get(f"loss_{name}"), ".4f") for name in part_names]
    # 경계 항의 **진짜 진행도**. `loss_boundary`는 target 엔트로피가 하한이라 0으로 가지
    # 않으므로, 그것만 보면 하한에 붙어 평평한 것을 "수렴 실패"로 오독한다
    # (`docs/soft_boundary_loss_design.md` §5.4). soft-boundary loss일 때만 존재한다.
    if "kl_boundary" in parts:
        fields.append(_field("kl_bnd↓", parts["kl_boundary"], ".4f"))
        # 엔트로피 하한을 뺀 **축소 가능한** loss 중 경계 항의 몫. `share b`는 상수인 하한을
        # 포함하므로 경계 항의 영향력을 과대평가한다(같은 문서 §5.6).
        fields.append(_field("bnd_kl몫", parts.get("share_boundary_kl"), ".2f"))
    share_field = _format_share_field(parts, part_names)
    if share_field is not None:
        fields.append(share_field)
    return _format_row(tag, tag_color, fields)


def _format_range_line(metrics):
    """광선 통계는 train/val 행을 과도하게 넓히지 않도록 별도 줄에 표시한다.

    `missed↓`를 같은 줄에 붙여 둔다: 거리 통계는 GT·예측이 둘 다 `RAY_OK`인 광선만 쓰므로
    장애물을 완전히 놓친 광선이 표본에서 빠진다. 두 숫자를 떨어뜨려 놓으면 "mae가 좋아졌다"를
    혼자 읽게 되고, 그것이 바로 이 프로젝트가 이미 한 번 밟은 함정이다.
    """
    if metrics is None:
        return []
    fields = [
        _field("mae↓", metrics["mae"], emphasis=_Ansi.BOLD),
        _field("p50↓", metrics["abs_p50"]),
        _field("p90↓", metrics["abs_p90"]),
        _field("bias", metrics["bias"], "+.3f"),
        _field("over↓", metrics["over_mean"]),
        _field("under", metrics["under_mean"]),
        _field("missed↓", metrics["missed_obstacle_rate"]),
        _c(_Ansi.DIM, f"rays {metrics['n_paired_rays']}"),
    ]
    return [_format_row("range", _Ansi.BLUE, fields)]


def _format_occupied_line(tolerance_metrics):
    """경계 정밀도 줄 -- tolerance F1만 둔다.

    면적 `iou_occupied`를 이 줄에서 뺐다(2026-08-21): 두께 1셀 표면에 면적 IoU를 씌운 값은
    한 칸 밀리면 반토막 나서 binary에서 val 0.071까지 떨어졌고, "몇 셀 밀렸나"는 τ가 다른
    F1 셋이 훨씬 직접적으로 답한다. `f1@10cm`(2셀)이 낮고 `f1@40cm`(8셀)이 높으면 밀린
    것이고, 셋이 같이 낮으면 장애물을 못 찾은 것이다.
    """
    if not tolerance_metrics:
        return []
    fields = [
        _field(f"f1@{tolerance_key(t)}↑", (tolerance_metrics or {}).get(tolerance_key(t), {}).get("f1"),
               emphasis=_Ansi.BOLD if t == 0.20 else "")
        for t in DEFAULT_TOLERANCES_M
    ]
    return [_format_row("occ", _Ansi.MAGENTA, fields)]


def _format_partition_warning(train_free, val_free):
    """free/occupied/unknown 분할이 valid를 덮지 못하면 epoch 로그에서 즉시 경고한다."""
    defects = (train_free or {}).get("partition_defects", 0) + \
              (val_free or {}).get("partition_defects", 0)
    if defects == 0:
        return []
    return [_c(_Ansi.BOLD + _Ansi.RED,
               f"  [BUG] partition defect on {defects} cells"
               " -- free/occupied/unknown does not cover valid exactly")]


def format_epoch_log(
    *,
    epoch,
    num_epochs,
    epoch_time,
    train_loss,
    train_loss_parts=None,
    train_free_metrics=None,
    val_loss,
    val_loss_parts=None,
    val_free_metrics=None,
    val_range_metrics=None,
    val_tolerance_metrics=None,
    baseline_iou_free=None,
    val_score,
    best_val_score,
    is_new_best,
    loss_part_names=DEFAULT_LOSS_PART_NAMES,
):
    displayed_best = val_score if is_new_best else best_val_score
    # 직전 best 대비 증감. "이번 epoch이 나아졌나"를 숫자 두 개를 눈으로 빼지 않고 보려는 것.
    # best의 초기값이 0.0이라 첫 val epoch에서는 델타가 곧 점수라 무의미하므로 생략한다.
    delta = ""
    if best_val_score > 0 and not math.isnan(val_score):
        diff = val_score - best_val_score
        delta = " " + _c(_Ansi.GREEN if diff > 0 else _Ansi.RED, f"({diff:+.3f})")
    baseline_delta = ""
    val_free_iou = (val_free_metrics or {}).get("iou_free")
    if baseline_iou_free is not None and val_free_iou is not None and not math.isnan(val_free_iou):
        gap = val_free_iou - baseline_iou_free
        baseline_delta = " " + _c(
            _Ansi.GREEN if gap > 0 else _Ansi.RED, f"({gap:+.3f} vs baseline)"
        )
    return "\n".join([
        (
            _c(_Ansi.BOLD + _Ansi.CYAN, f"epoch {epoch:03d}/{num_epochs}") + _sep()
            + _c(_Ansi.DIM, f"time {epoch_time:6.1f}s") + _sep()
            + _field("val_iou_free↑", val_score, emphasis=_Ansi.BOLD) + delta + baseline_delta + _sep()
            + _field("best_val_iou_free↑", displayed_best) + _sep()
            + (_c(_Ansi.BOLD + _Ansi.GREEN, "checkpoint: new best") if is_new_best
               else _c(_Ansi.DIM, "checkpoint: -"))
        ),
        *_format_partition_warning(train_free_metrics, val_free_metrics),
        _format_metric_row("train", _Ansi.YELLOW, train_loss, train_loss_parts,
                           train_free_metrics, loss_part_names),
        _format_metric_row("val", _Ansi.CYAN, val_loss, val_loss_parts,
                           val_free_metrics, loss_part_names),
        *_format_range_line(val_range_metrics),
        *_format_occupied_line(val_tolerance_metrics),
    ])


def _add_scalar_if_finite(writer, tag, value, step) -> None:
    if value is not None and math.isfinite(float(value)):
        writer.add_scalar(tag, value, step)


# TensorBoard tag는 한 번 쓰기 시작하면 바꿀 수 없다(옛 run과 축이 갈린다). 그래서 지표 키와
# tag를 여기 한 곳에 나란히 적어 둔다.
_FREE_SCALAR_TAGS = (
    ("iou_free", "iou_free_epoch"),
    ("fatal_rate", "fatal_rate_epoch"),
    ("free_miss_rate", "free_miss_rate_epoch"),
)
# `abs_p50`은 표본 반지름 격자(0.5셀 = 2.5 cm)에 양자화돼 epoch이 바뀌어도 값이 거의
# 안 움직였다(60 epoch 동안 0.1500에 고정). `over_mean`/`under_mean`은 `bias`를 두 방향으로
# 쪼갠 것이라 셋 중 둘만 있으면 나머지가 정해진다. 계산은 `_delta_stats`에 그대로 남아
# 있으므로 필요해지면 재채점으로 되살릴 수 있다.
_RANGE_SCALAR_TAGS = (
    ("mae", "range_mae_epoch"),
    ("abs_p90", "range_abs_p90_epoch"),
    ("bias", "range_bias_epoch"),
    ("missed_obstacle_rate", "range_missed_obstacle_rate_epoch"),
)


def write_epoch_scalars(writer, split, metrics, epoch) -> None:
    """epoch scalar 전부를 TensorBoard에 쓴다. **두 학습 스크립트가 공유한다.**

    원래 두 스크립트가 각자 거의 같은 함수를 갖고 있었다. 지표를 하나 추가할 때 한쪽에만
    반영되면 pretrain과 fine-tune 곡선을 나란히 볼 수 없고, 그 비교가 이 프로젝트의 학습
    순서 전체의 근거다. 지표 집합이 커진 지금은 그 위험이 더 크므로 한 벌만 둔다.

    train split은 `range`/`rings`/`tolerance`가 없는 dict를 넘긴다 -- 광선·거리변환은 val에서만
    돌린다. 그래서 전부 `.get()`으로 읽고, 없으면 조용히 건너뛴다.
    """
    _add_scalar_if_finite(writer, f"{split}/loss_epoch", metrics.get("loss"), epoch)
    for key, value in (metrics.get("loss_parts") or {}).items():
        _add_scalar_if_finite(writer, f"{split}/{key}_epoch", value, epoch)

    free_metrics = metrics.get("free") or {}
    for key, tag in _FREE_SCALAR_TAGS:
        _add_scalar_if_finite(writer, f"{split}/{tag}", free_metrics.get(key), epoch)

    range_metrics = metrics.get("range") or {}
    for key, tag in _RANGE_SCALAR_TAGS:
        _add_scalar_if_finite(writer, f"{split}/{tag}", range_metrics.get(key), epoch)

    # `range_bins`(GT 거리별 `range_mae`)는 링별 `iou_free`와 같은 질문("원거리에서 무너지나")에
    # 답하므로 **선택 기준인 `iou_free`를 층화하는 쪽만** 남긴다. 계산은 그대로 있고
    # `rescore_checkpoints.py`가 여전히 출력한다.
    for name, values in (metrics.get("rings") or {}).items():
        _add_scalar_if_finite(writer, f"{split}/ring_{name}_iou_free_epoch",
                              values["iou_free"], epoch)
    # `precision`/`recall`은 `f1`과 같은 이야기를 τ마다 세 줄로 반복한다. 비대칭 비용은
    # `fatal_rate`/`free_miss_rate`와 `range_bias`가 이미 방향까지 갖고 보여준다.
    for name, values in (metrics.get("tolerance") or {}).items():
        _add_scalar_if_finite(writer, f"{split}/occupied_f1_{name}_epoch", values["f1"], epoch)


def select_checkpoint_score(free_metrics):
    """체크포인트 순위 지표. `iou_free` 하나다.

    한 줄짜리 함수를 남겨 두는 이유: 두 학습 스크립트가 각자 `val_free_metrics["iou_free"]`를
    직접 읽으면 한쪽만 조용히 다른 기준으로 바뀔 수 있다. 이름 붙은 이음매가 있으면 테스트가
    그 지점을 고정할 수 있다. 옛 `0.5 * (d_iou + o_iou)` 평균으로 되돌아가는 것을 막는 것이
    애초의 목적이었다(`docs/archive/free_space_metric_migration.md` §2.2).
    """
    return free_metrics["iou_free"]


def compute_iou_per_sample(pred, target, valid_g):
    """`pred`/`target`은 이미 0/1(같은 class를 가리키는 binary mask)이어야 한다."""
    intersection = (pred * target * valid_g).sum(dim=[1, 2, 3])
    union = ((pred + target) * valid_g).clamp(0, 1).sum(dim=[1, 2, 3])
    return intersection / (1e-4 + union)


def compute_iou(pred, target, valid_g):
    return compute_iou_per_sample(pred, target, valid_g).mean()


_FREE_METRIC_SCALAR_KEYS = (
    "iou_free", "iou_free_count",
    "fatal_rate", "fatal_denom",
    "free_miss_rate", "free_miss_denom",
    "partition_defects",
)

# `(집계 키, count 키)`. IoU는 "평균에 들어간 샘플 수"로 가중해야 한다 -- union이 0인 샘플이
# 빠지므로, batch 수로 평균하면 그 차이가 무시된다.
_IOU_KEYS = (
    ("iou_free", "iou_free_count"),
)


def summarize_ring_metrics(ring_dicts, ring_masks) -> dict:
    """링별 M1·M2의 epoch 집계. 링마다 셀 수가 달라 반드시 count로 가중해야 한다."""
    return {
        name: {
            "iou_free": weighted_mean([d[name]["iou_free"] for d in ring_dicts],
                                      [d[name]["iou_free_count"] for d in ring_dicts]),
            "fatal_rate": weighted_mean([d[name]["fatal_rate"] for d in ring_dicts],
                                        [d[name]["fatal_denom"] for d in ring_dicts]),
        }
        for name, _ in ring_masks
    }


def evaluate_split(step, loader, device, rays, ring_masks, *,
                   cell_m, range_edges_m) -> dict:
    """validation 한 바퀴. **pretrain과 fine-tuning이 공유한다.**

    두 학습 스크립트가 각자 이 루프를 갖고 있으면 한쪽만 고쳐지는 순간 pretrain과 fine-tune
    숫자를 나란히 읽을 수 없게 된다 -- 그 비교가 이 프로젝트의 학습 순서(SynWoodScape ->
    자체 데이터셋) 전체의 근거이므로 한 벌만 둔다.

    M3(range)·M4(ring)·F1@τ는 여기서만 계산한다. train에서는 계산하지 않는다 -- 광선 추출과
    거리변환이 배치마다 비싸고, 학습 중에 볼 값이 아니다.

    `cell_m`/`range_edges_m`을 키워드 필수로 둔 이유: 기본값을 주면 격자 해상도가 다른
    데이터셋에서 호출부가 조용히 틀린 스케일로 τ를 재고, 그 결과가 그럴듯한 숫자로 나온다.
    """
    losses, parts_dicts = [], []
    free_dicts, range_dicts, ring_dicts, tolerance_dicts = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            loss, parts, free_metrics = step(batch)
            losses.append(loss.item())
            parts_dicts.append({k: v.item() for k, v in parts.items()})
            valid = batch["valid_bev_g"].to(device)
            range_dicts.append(range_error(
                free_metrics["pred_free"], free_metrics["gt_free"], valid, rays
            ))
            ring_dicts.append(metrics_per_ring(
                free_metrics["pred_free"], free_metrics["gt_free"], valid, ring_masks
            ))
            tolerance_dicts.append(tolerance_counts(
                free_metrics["pred_occupied"], free_metrics["gt_occupied"], valid, cell_m
            ))
            append_free_metrics(free_dicts, free_metrics)
    return {
        "loss": (sum(losses) / len(losses)) if losses else float("nan"),
        "loss_parts": mean_loss_parts(parts_dicts),
        "free": summarize_free_metrics(free_dicts),
        "range": summarize_range_error(range_dicts),
        "range_bins": summarize_range_error_by_gt_range(range_dicts, range_edges_m),
        "rings": summarize_ring_metrics(ring_dicts, ring_masks),
        "tolerance": summarize_tolerance_f1(tolerance_dicts),
    }


def empty_epoch_metrics() -> dict:
    """val을 돌리지 않은 epoch의 자리표시자. `evaluate_split`과 키가 같아야 한다."""
    return {
        "loss": float("nan"),
        "loss_parts": {},
        "free": summarize_free_metrics([]),
        "range": summarize_range_error([]),
        "range_bins": {},
        "rings": {},
        "tolerance": {},
    }


def mean_loss_parts(parts_dicts) -> dict:
    """클래스별 loss 항의 epoch 평균.

    첫 배치의 키만 순회한다 -- 3-class loss는 배치마다 항이 세 개로 고정이므로 키 합집합을
    쓸 이유가 없고, 오히려 키가 배치마다 달라지면 그것이 버그다. 총 loss만 로그에 남기면
    unknown이 압도적 다수라 occupied 항이 언제 0으로 죽었는지 보이지 않으므로 따로 낸다.
    """
    if not parts_dicts:
        return {}
    return {
        key: float(sum(float(d[key]) for d in parts_dicts) / len(parts_dicts))
        for key in parts_dicts[0]
    }


def append_free_metrics(metric_dicts, free_metrics) -> None:
    """epoch 집계에는 스칼라만 보관한다; free mask는 val 배치에서 즉시 소비한다."""
    metric_dicts.append({key: free_metrics[key] for key in _FREE_METRIC_SCALAR_KEYS})


def summarize_free_metrics(dicts) -> dict:
    if not dicts:
        return {**{key: float("nan") for key, _ in _IOU_KEYS},
                "fatal_rate": float("nan"),
                "free_miss_rate": float("nan"), "partition_defects": 0}
    return {
        **{key: weighted_mean([d[key] for d in dicts], [d[count] for d in dicts])
           for key, count in _IOU_KEYS},
        "fatal_rate": weighted_mean([d["fatal_rate"] for d in dicts],
                                    [d["fatal_denom"] for d in dicts]),
        "free_miss_rate": weighted_mean([d["free_miss_rate"] for d in dicts],
                                        [d["free_miss_denom"] for d in dicts]),
        "partition_defects": sum(d["partition_defects"] for d in dicts),
    }

