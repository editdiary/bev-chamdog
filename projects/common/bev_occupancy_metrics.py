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
    weighted_mean,
)
from projects.common.metric_spec import (  # noqa: E402
    BF_TOLERANCES_M,
    FREE_METRICS,
    epoch_tag,
    tolerance_key,
)
from projects.common.occupied_metrics import (  # noqa: E402
    boundary_f_scores,
    summarize_boundary_f,
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
                  "boundary": "b", "range": "r"}


def _format_share_field(loss_parts, part_names):
    """각 클래스가 총 loss에 기여하는 몫을 한 칸에 담는다.

    합이 1이라 숫자를 붙여 놓는 것이 가장 읽기 쉽다. 이것을 로그에 넣는 이유: 클래스별
    평균(`loss_*`)만으로는 "occupied가 셀의 1.1 %인데 총 loss의 67 %"라는 사실이 보이지 않아
    실측에서 셀 비율을 손으로 곱해 봐야 알 수 있었다
    (`docs/archive/research/finetune_overfitting_diagnosis.md` §12).
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
            _field("iou_nf↑", free_metrics["iou_non_free"]),
            _field("P↑", free_metrics["precision"]),
            _field("R↑", free_metrics["recall"]),
        ]
    fields.append(_field("loss_total↓", loss, ".4f"))
    parts = loss_parts or {}
    fields += [_field(f"loss_{name}↓", parts.get(f"loss_{name}"), ".4f") for name in part_names]
    # 논문용 공통 눈금(`metric_spec.PAPER_LOSS_TERMS`) 중 경계 하나만 화면에 둔다 -- 손실 종류와
    # 무관한 BCE라 두 손실의 화면 로그를 나란히 읽게 한다.
    if "bce_boundary" in parts:
        fields.append(_field("bce_bnd↓", parts["bce_boundary"], ".4f"))
    # 2026-10-02 이전 로그 형식(soft-boundary 전용 항). 옛 dict를 다시 찍을 때만 걸린다.
    if "kl_boundary" in parts:
        fields.append(_field("kl_bnd↓", parts["kl_boundary"], ".4f"))
        # 엔트로피 하한을 뺀 **축소 가능한** loss 중 경계 항의 몫. `share b`는 상수인 하한을
        # 포함하므로 경계 항의 영향력을 과대평가한다(같은 문서 §5.6).
        fields.append(_field("bnd_kl몫", parts.get("share_boundary_kl"), ".2f"))
    # 방위각 자유거리 보조항(§13). `loss_range`는 dead zone과 Huber를 거친 값이라 미터로
    # 읽히지 않으므로, **dead zone 전의 순수 오차와 그 부호**를 같이 낸다. 부호가 필요한
    # 이유: dead zone이 `|e|`에 대칭이라 과대예측(= `fatal` 방향)이 무벌점으로 늘어날 수
    # 있고, 그것을 막는 것이 §4.5의 제약이다.
    if "range_arc_mae" in parts:
        fields.append(_field("arc_mae↓", parts["range_arc_mae"], ".4f"))
        fields.append(_field("arc_bias", parts.get("range_arc_bias"), "+.4f"))
    share_field = _format_share_field(parts, part_names)
    if share_field is not None:
        fields.append(share_field)
    return _format_row(tag, tag_color, fields)


def _format_boundary_line(boundary_metrics):
    """경계 줄 -- `BF@τ`만 둔다.

    `bf@10cm`(2셀)이 낮고 `bf@30cm`(6셀)이 높으면 경계가 밀린 것이고, 셋이 같이 낮으면 장애물을
    못 찾은 것이다. 방위각 거리 줄(`range`)은 2026-10-02에 지표와 함께 뺐다.
    """
    if not boundary_metrics:
        return []
    fields = [
        _field(f"bf@{tolerance_key(t)}↑", boundary_metrics.get(tolerance_key(t), {}).get("bf"),
               emphasis=_Ansi.BOLD if t == 0.20 else "")
        for t in BF_TOLERANCES_M
    ]
    return [_format_row("bnd", _Ansi.MAGENTA, fields)]


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
    val_boundary_metrics=None,
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
        *_format_boundary_line(val_boundary_metrics),
    ])


def _add_scalar_if_finite(writer, tag, value, step) -> None:
    if value is not None and math.isfinite(float(value)):
        writer.add_scalar(tag, value, step)


# TensorBoard tag의 이름 규칙은 `metric_spec.epoch_tag` 한 곳에 있다 -- 읽는 쪽(보고 도구·
# 패키지 생성기)이 같은 함수로 tag를 만든다. **2026-10-02에 tag 집합이 바뀌었다**: 그 이전
# 런은 `fatal_rate`/`free_miss_rate`/`range_*`/`occupied_f1_*`로 기록돼 있다.


def write_epoch_scalars(writer, split, metrics, epoch) -> None:
    """epoch scalar 전부를 TensorBoard에 쓴다. **두 학습 스크립트가 공유한다.**

    원래 두 스크립트가 각자 거의 같은 함수를 갖고 있었다. 지표를 하나 추가할 때 한쪽에만
    반영되면 pretrain과 fine-tune 곡선을 나란히 볼 수 없고, 그 비교가 이 프로젝트의 학습
    순서 전체의 근거다. 지표 집합이 커진 지금은 그 위험이 더 크므로 한 벌만 둔다.

    train split은 `rings`/`boundary`가 없는 dict를 넘긴다 -- 광선·거리변환은 val에서만
    돌린다. 그래서 전부 `.get()`으로 읽고, 없으면 조용히 건너뛴다.

    쓰는 tag (`{split}/..._epoch`): `metric_spec.FREE_METRICS` 넷, `bf_{τ}`와 그
    `_precision`/`_recall`, `ring_{링}_{FREE_METRICS}`.
    """
    # 총 손실은 `loss_total`이다(2026-10-02 이름 변경, 그 전에는 `loss_epoch`) -- 영역별 몫
    # `loss_free`·`loss_non_free`·`loss_boundary`와 한 묶음으로 읽히게.
    _add_scalar_if_finite(writer, f"{split}/loss_total_epoch", metrics.get("loss"), epoch)
    for key, value in (metrics.get("loss_parts") or {}).items():
        if key.startswith("_"):          # 집계용 내부 값(`decompose_loss`)은 로그에 안 낸다
            continue
        _add_scalar_if_finite(writer, f"{split}/{key}_epoch", value, epoch)

    free_metrics = metrics.get("free") or {}
    for key in FREE_METRICS:
        _add_scalar_if_finite(writer, epoch_tag(split, key), free_metrics.get(key), epoch)

    for ring, values in (metrics.get("rings") or {}).items():
        for key in FREE_METRICS:
            _add_scalar_if_finite(writer, epoch_tag(split, f"ring_{ring}_{key}"),
                                  values.get(key), epoch)
    # BF의 precision/recall도 쓴다. 예전에는 "f1과 같은 이야기"라며 버렸는데, 그 근거였던
    # `fatal_rate`/`range_bias`가 빠진 지금은 경계가 **어느 쪽으로** 틀렸는지 볼 곳이 여기뿐이다.
    for key, values in (metrics.get("boundary") or {}).items():
        bf = f"bf_{key}"
        _add_scalar_if_finite(writer, epoch_tag(split, bf), values["bf"], epoch)
        _add_scalar_if_finite(writer, epoch_tag(split, f"{bf}_precision"),
                              values["precision"], epoch)
        _add_scalar_if_finite(writer, epoch_tag(split, f"{bf}_recall"), values["recall"], epoch)


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


# `(집계 키, count 키)`. 전부 프레임 macro이므로 "평균에 들어간 프레임 수"로 가중하면
# epoch 전체의 정확한 프레임 평균이 된다 -- 빠진 프레임(예측·GT 둘 다 빈)이 batch마다 달라서
# batch 수로 평균하면 틀린다.
_MACRO_KEYS = tuple((name, f"{name}_count") for name in FREE_METRICS)

_FREE_METRIC_SCALAR_KEYS = tuple(k for pair in _MACRO_KEYS for k in pair) + ("partition_defects",)


def _summarize_macro(dicts) -> dict:
    return {key: weighted_mean([d[key] for d in dicts], [d[count] for d in dicts])
            for key, count in _MACRO_KEYS}


def summarize_ring_metrics(ring_dicts, ring_masks) -> dict:
    """링별 epoch 집계. 링마다 빠지는 프레임이 달라 반드시 count로 가중해야 한다."""
    return {name: _summarize_macro([d[name] for d in ring_dicts]) for name, _ in ring_masks}


def evaluate_split(step, loader, device, rays, ring_masks, *, cell_m) -> dict:
    """validation 한 바퀴. **pretrain과 fine-tuning이 공유한다.**

    두 학습 스크립트가 각자 이 루프를 갖고 있으면 한쪽만 고쳐지는 순간 pretrain과 fine-tune
    숫자를 나란히 읽을 수 없게 된다 -- 그 비교가 이 프로젝트의 학습 순서(SynWoodScape ->
    자체 데이터셋) 전체의 근거이므로 한 벌만 둔다.

    ring 지표와 BF@τ는 여기서만 계산한다. train에서는 계산하지 않는다 -- 거리변환이 배치마다
    비싸고, 학습 중에 볼 값이 아니다. (`rays`는 BF의 예측 경계를 유도하는 step 안에서 이미
    쓰였다. 방위각 거리 지표 M3는 2026-10-02에 뺐다.)

    `cell_m`을 키워드 필수로 둔 이유: 기본값을 주면 격자 해상도가 다른 데이터셋에서 호출부가
    조용히 틀린 스케일로 τ를 재고, 그 결과가 그럴듯한 숫자로 나온다.
    """
    losses, parts_dicts = [], []
    free_dicts, ring_dicts, boundary_dicts = [], [], []
    with torch.no_grad():
        for batch in loader:
            loss, parts, free_metrics = step(batch)
            # 프레임 수로 가중한다 -- 마지막 배치가 작아도(val 75 = 9 x 8 + 3) 같은 무게를 받지
            # 않게. 유효 셀 마스크가 프레임마다 같으므로 이것이 곧 epoch 전체의 셀 평균이다.
            losses.append((loss.item(), int(batch["valid_bev_g"].shape[0])))
            parts_dicts.append({k: v.item() for k, v in parts.items()})
            valid = batch["valid_bev_g"].to(device)
            ring_dicts.append(metrics_per_ring(
                free_metrics["pred_free"], free_metrics["gt_free"], valid, ring_masks
            ))
            boundary_dicts.append(boundary_f_scores(
                free_metrics["pred_occupied"], free_metrics["gt_occupied"], valid, cell_m
            ))
            append_free_metrics(free_dicts, free_metrics)
    merged = mean_loss_parts(parts_dicts)
    return {
        "loss": epoch_loss(losses, merged),
        "loss_parts": merged,
        "free": summarize_free_metrics(free_dicts),
        "rings": summarize_ring_metrics(ring_dicts, ring_masks),
        "boundary": summarize_boundary_f(boundary_dicts),
    }


def weighted_frame_mean(loss_and_frames) -> float:
    """`[(배치 손실, 프레임 수), ...]` -> 프레임 수로 가중한 epoch 손실. 비면 NaN.

    배치마다 같은 무게로 평균하면 작은 마지막 배치가 과대 대표된다. 2026-10-02 스모크에서 그
    때문에 가중 BCE의 val `loss`가 공통 손실 분해(`Σ frac·loss_r`)와 4 % 어긋났다.
    """
    total = sum(n for _, n in loss_and_frames)
    if not total:
        return float("nan")
    return sum(v * n for v, n in loss_and_frames) / total


def empty_epoch_metrics() -> dict:
    """val을 돌리지 않은 epoch의 자리표시자. `evaluate_split`과 키가 같아야 한다."""
    return {
        "loss": float("nan"),
        "loss_parts": {},
        "free": summarize_free_metrics([]),
        "rings": {},
        "boundary": {},
    }


def mean_loss_parts(parts_dicts) -> dict:
    """클래스별 loss 항의 epoch 평균.

    첫 배치의 키만 순회한다 -- 3-class loss는 배치마다 항이 세 개로 고정이므로 키 합집합을
    쓸 이유가 없고, 오히려 키가 배치마다 달라지면 그것이 버그다. 총 loss만 로그에 남기면
    unknown이 압도적 다수라 occupied 항이 언제 0으로 죽었는지 보이지 않으므로 따로 낸다.
    """
    if not parts_dicts:
        return {}
    out = {
        key: float(sum(float(d[key]) for d in parts_dicts) / len(parts_dicts))
        for key in parts_dicts[0] if not key.startswith("_")
    }
    # **공통 손실 분해는 셀 수로 가중해 모은다**(`binary_metrics.decompose_loss`). 영역 평균은
    # 그 영역의 셀 수로, `ce_all`은 유효 셀 수로 가중하면 epoch 전체에서 그 집합 모든 셀의
    # 평균이 된다 -- 배치를 어떻게 자르든 같은 값이다. 영역 비율도 셀 총수의 비로 다시 낸다.
    if "_n_valid" in parts_dicts[0]:
        n = [float(d["_n_valid"]) for d in parts_dicts]
        total = sum(n) or 1.0
        regions = ("free", "non_free", "boundary")
        for region in regions:
            w = [float(d[f"_frac_{region}"]) * nv for d, nv in zip(parts_dicts, n)]
            sw = sum(w) or 1.0
            keys = [f"loss_{region}", f"bce_{region}"] + (
                ["_floor_boundary"] if region == "boundary" else [])
            for key in keys:
                out[key] = sum(float(d[key]) * wi for d, wi in zip(parts_dicts, w)) / sw
            out[f"_frac_{region}"] = sum(w) / total
        # epoch 총 손실을 영역 평균으로 다시 짓는다(`epoch_loss`가 읽는다). 계수는 배치마다 같다.
        mode = float(parts_dicts[0].get("_objective_mode", -1.0))
        if mode == 0.0:
            out["_objective_set"] = sum(out[f"_frac_{r}"] * out[f"loss_{r}"] for r in regions)
        elif mode == 1.0:
            out["_objective_set"] = sum(float(parts_dicts[0][f"_objective_c_{r}"]) * out[f"loss_{r}"]
                                        for r in regions)
    return out


def epoch_loss(loss_and_frames, merged_parts) -> float:
    """epoch 총 손실 = **그 epoch의 모든 셀을 한 집합으로 본 목적함수**(공통 손실 분해가 있으면).

    가중 BCE는 셀 평균이고 soft-BCE는 영역별 평균의 가중합이라, 배치 값을 평균하면 soft-BCE에서
    `½·F + ½·N + λ_B·B`가 epoch 단위로 6~7 % 어긋났다(배치마다 영역 크기가 달라서, 2026-10-02
    스모크). 집합 전체로 정의하면 두 손실 모두 분해가 정확히 맞고 배치 크기와 무관하다.
    분해가 없으면(옛 경로·보조항) 프레임 가중 평균으로 돌아간다.
    """
    value = (merged_parts or {}).get("_objective_set")
    return float(value) if value is not None else weighted_frame_mean(loss_and_frames)


def label_constants(merged_parts) -> dict:
    """epoch 집계에서 **라벨만의 함수인 값**을 꺼낸다 -- `metric_spec.LABEL_CONSTANTS`.

    영역 셀 비율과 soft target의 경계 하한은 학습 중 변하지 않으므로 곡선으로 내지 않고
    `config.json`에 한 번 적는다(`record_label_constants`). 분해가 없으면 빈 dict.
    """
    parts = merged_parts or {}
    if "_frac_free" not in parts:
        return {}
    return {"frac_free": parts["_frac_free"], "frac_non_free": parts["_frac_non_free"],
            "frac_boundary": parts["_frac_boundary"],
            "loss_boundary_floor": parts.get("_floor_boundary", 0.0)}


def record_label_constants(config_path, split, merged_parts) -> bool:
    """`config.json`의 `label_constants.{split}`이 비어 있으면 채운다. 채웠으면 True.

    **첫 epoch에 한 번만 쓴다** -- 이후 epoch의 값은 같다(라벨·영역이 고정). train은 증강이 있어도
    영역 비율이 바뀌지 않는다(좌우 반전은 꺼져 있다, `FLIP_AUGMENT=False`).
    """
    import json
    values = label_constants(merged_parts)
    path = Path(config_path)
    if not values or not path.exists():
        return False
    config = json.loads(path.read_text())
    block = config.setdefault("label_constants", {})
    if split in block:
        return False
    block[split] = {k: round(float(v), 6) for k, v in values.items()}
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    return True


def append_free_metrics(metric_dicts, free_metrics) -> None:
    """epoch 집계에는 스칼라만 보관한다; free mask는 val 배치에서 즉시 소비한다."""
    metric_dicts.append({key: free_metrics[key] for key in _FREE_METRIC_SCALAR_KEYS})


def summarize_free_metrics(dicts) -> dict:
    """epoch 집계. 넷 다 프레임 수로 가중한 평균 = epoch 전체의 정확한 프레임 macro."""
    if not dicts:
        return {**{key: float("nan") for key, _ in _MACRO_KEYS}, "partition_defects": 0}
    return {**_summarize_macro(dicts),
            "partition_defects": sum(d["partition_defects"] for d in dicts)}

