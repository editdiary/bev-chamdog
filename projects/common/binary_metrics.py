"""(D) binary 정식화 -- `free` / `not-free` 한 장만 예측한다.

**왜 클래스를 둘로 줄이나.** 3-class에서 `occupied`는 셀의 1.1 %인데 val loss의 67 %를
만들었고, 어떤 클래스 가중치도 그것을 고치지 못했다 -- CE는 면적 loss인데 GT `occupied`는
두께 1셀 표면이라 셀 단위 정확도가 본질적으로 달성 불가능하기 때문이다
(`docs/finetune_overfitting_diagnosis.md` §12-§13).

**무엇을 잃나 -- 실측으로 거의 없다(§15).** 라벨에서 `occupied = ~occ & vis`이고 `vis`가 ego
원점 raycast이므로 GT `occupied`는 free 영역의 ego 기준 경계다. GT free를 광선에 되돌리면
`f1@20cm` 0.984로 복원된다. 그래서 예측에서도 `occupied`를 **유도**해 보고하고
(`occupied_metrics.derive_occupied`), 지표 집합은 3-class와 완전히 같게 유지한다.

`unknown`은 사라지지 않는다 -- `not_free` 안에 들어간다. 즉 모델은 여전히 "어디까지 보이는가"를
배워야 하고, 바뀐 것은 **"왜 못 가는가"를 loss가 더 이상 묻지 않는다**는 것뿐이다.
"""
import torch

from projects.common.free_space import FREE, decompose
from projects.common.free_space_metrics import free_metrics_from_masks
from projects.common.metric_spec import COMMON_LOSS_PARTS, DECOMPOSITION_DELTA_M  # noqa: F401
from projects.common.occupied_metrics import derive_occupied
from projects.common.segmentation_loss import (
    DEFAULT_MAX_CLASS_WEIGHT,
    inverse_frequency_weights,
    masked_weighted_ce,
)
from projects.common.range_loss import (
    DEFAULT_DELTA_R_M,
    DEFAULT_HUBER_BETA_M,
    compute_cumulative_range_loss,
    compute_range_loss,
)
from projects.common.soft_boundary import (
    DEFAULT_DELTA_M,
    DEFAULT_EPS,
    DEFAULT_KAPPA,
    DEFAULT_LAMBDA_B,
    TARGET_LINEAR,
    build_soft_boundary_target,
    compute_soft_boundary_loss,
    region_masks,
    target_entropy,
)

# `FREE`는 3-class와 같은 인덱스 1을 쓴다 -- 두 정식화의 logits를 같은 시각화·재채점 코드가
# 다룰 때 "1번 채널이 free"라는 규약이 갈리지 않도록 한다.
NOT_FREE = 0
CLASS_ORDER = (NOT_FREE, FREE)
_PART_BY_CLASS = {NOT_FREE: "not_free", FREE: "free"}
# 콘솔 표의 손실 칸 이름(`LOSS_PART_NAMES` 등)은 아래 "공통 손실 분해" 절에 있다.

# 3-class와 같은 기본 상한을 쓰지만 **실제로는 걸리지 않는다** -- 로봇 train split에서
# 순수 역빈도가 free 3.94 / not_free 1.00이다. 상한이 loss 균형을 결정하던 3-class의
# 상황(occupied 67.9 -> 20)이 정식화 자체로 사라진 것이 (D)의 요점 중 하나다.
MAX_CLASS_WEIGHT = DEFAULT_MAX_CLASS_WEIGHT


def to_class_index(parts: dict) -> torch.Tensor:
    """분해 -> `(B, 1, H, W)` long. `free`가 1, 나머지 전부가 0이다.

    `valid` 밖은 `NOT_FREE`로 떨어지지만 loss에서 마스킹되므로 값은 의미가 없다
    (3-class의 `free_space.to_class_index`가 `UNKNOWN`으로 떨어뜨리는 것과 같은 이유).
    """
    return parts["free"].long()


def compute_binary_loss(logits, class_index, valid, class_weights, label_smoothing=0.0):
    """`valid` 셀에 한정한 가중 2-class cross-entropy.

    구현은 3-class와 **같은 함수**(`segmentation_loss.masked_weighted_ce`)다. 정식화만 바꾸고
    loss의 정규화·집계는 그대로 두어야 두 런의 loss 곡선을 나란히 읽을 수 있다.
    """
    return masked_weighted_ce(logits, class_index, valid, class_weights, _PART_BY_CLASS,
                              label_smoothing)


def class_weights_from_labels(label_triples, max_class_weight=None) -> torch.Tensor:
    """`(occ, vis, valid)` 묶음에서 역빈도 가중치 `[not_free, free]`.

    `not_free`는 `occupied ∪ unknown`이다. 분해는 3-class와 같은
    `free_space.decompose` 하나만 쓴다 -- 여기서 마스크를 다시 조합하면 loss가 보는 클래스
    정의와 가중치가 세는 정의가 갈라진다(예전에 `pos_weight`가 마스크를 무시해 클래스 보정이
    통째로 어긋난 적이 있다).
    """
    counts = torch.zeros(2, dtype=torch.float64)
    for occ, vis, valid in label_triples:
        parts = decompose(occ, vis, valid)
        free_count = float(parts["free"].sum())
        counts[FREE] += free_count
        counts[NOT_FREE] += float(parts["occupied"].sum()) + float(parts["unknown"].sum())
    return inverse_frequency_weights(
        counts, MAX_CLASS_WEIGHT if max_class_weight is None else max_class_weight
    )


def predicted_parts(logits, valid, rays) -> dict:
    """binary logits -> `free`/`occupied`/`unknown` 세 마스크 (3-class와 같은 계약).

    `occupied`는 예측하지 않고 예측 free의 ego 기준 경계에서 유도하며, `unknown`은 나머지다.
    지표와 시각화가 **같은 함수**를 타야 그림과 로그가 다른 말을 하지 않는다.
    """
    valid_b = valid.bool()
    pred_free = (logits.argmax(dim=1, keepdim=True) == FREE) & valid_b
    pred_occupied = derive_occupied(pred_free, valid, rays) & valid_b
    return {
        "free": pred_free,
        "occupied": pred_occupied,
        "unknown": valid_b & ~pred_free & ~pred_occupied,
    }


def compute_free_metrics(logits, seg_g, vis_g, valid_g, rays) -> dict:
    """binary logits -> 3-class와 **같은 지표 dict**.

    분할을 복원해 두면 `free_metrics_from_masks`·`f1@τ`·M3·ring이 전부 그대로 돌아가고,
    3-class 런과 지표를 한 표에 놓을 수 있다. 유도 비용은 bs8에서 25 ms이므로 train에서도
    매 배치 계산한다 -- val에서만 계산하면 train/val 곡선이 다른 것을 재게 된다.
    """
    return free_metrics_from_masks(
        predicted_parts(logits, valid_g, rays), decompose(seg_g, vis_g, valid_g), valid_g
    )


# === 공통 손실 분해 (2026-10-02) =========================================================
#
# **어떤 손실로 학습하든 같은 이름·같은 영역으로 손실을 기록한다** -- 그래야 "가중 BCE는 경계
# 손실이 처음부터 오르고 soft-BCE는 수렴한다" 같은 진술을 두 곡선으로 할 수 있다(사용자 요청).
#
# 영역은 soft-boundary 손실의 세 집합 그대로다(`soft_boundary.region_masks`):
#     Ω_F  경계에서 δ보다 먼 free        Ω_N  경계에서 δ보다 먼 non-free (+ 영구 사각지대)
#     Ω_B  경계 대역 |d| ≤ δ
# 셋은 겹치지 않고 `valid`를 정확히 덮는다. **δ는 런의 손실 설정과 무관한 상수 0.30 m로 고정한다**
# -- 가중 BCE 런에는 δ가 없고, 손잡이가 다른 런끼리도 같은 셀 집합을 재야 한다. 논문 설정의
# δ와 같으므로 soft-BCE 런에서는 손실의 세 항과 정확히 같은 영역이다. BF 최대 허용오차와도 같다.

#: 모든 손실이 내는 손실 항 이름 -- TensorBoard tag가 `{split}/{이름}_epoch`이다. 두 학습 경로
#: (`run_batch`, `run_batch_soft_boundary`)가 **정확히 이 집합**을 낸다(테스트로 고정).
#:
#:   loss_free / loss_not_free / loss_boundary   그 런이 **실제로 최적화한 손실**을 셀마다 낸 값의
#:                                               영역 평균. 가중 BCE는 클래스 가중 hard CE, soft-BCE는
#:                                               대역에서 soft target에 대한 CE다
#:   loss_boundary_floor   loss_boundary의 줄일 수 없는 하한(target 엔트로피). hard target이면 0.
#:                         soft-BCE의 loss_boundary는 0이 아니라 이 값으로 수렴한다
#:   ce_free / ce_not_free / ce_boundary / ce_all   **모든 런에 같은 눈금**: 가중치 없는 hard 0/1
#:                         CE. 손실 함수가 달라도 그대로 비교된다
#:   frac_free / frac_not_free / frac_boundary   영역의 셀 비율(라벨만의 함수)
_REGIONS = (("omega_f", "free"), ("omega_n", "not_free"), ("omega_b", "boundary"))

# 콘솔 표의 손실 칸. 두 손실이 같은 칸을 쓴다.
LOSS_PART_NAMES = ("free", "not_free", "boundary")
SOFT_BOUNDARY_LOSS_PART_NAMES = LOSS_PART_NAMES
SOFT_BOUNDARY_RANGE_LOSS_PART_NAMES = LOSS_PART_NAMES + ("range",)   # legacy(보조항)


def decompose_loss(per_cell_loss, floor_per_cell, logits, gt_free, valid, d, permanent_blind,
                   delta=DECOMPOSITION_DELTA_M, set_coefficients=None) -> dict:
    """`COMMON_LOSS_PARTS` dict + `_n_valid`. **기록 전용**이라 gradient를 흘리지 않는다.

    `per_cell_loss`는 그 런이 최적화한 손실의 셀별 값 `(B, 1, H, W)`, `floor_per_cell`은 그
    셀의 줄일 수 없는 하한(hard target이면 `None` = 0)이다.

    `_n_valid`(배치의 유효 셀 수)는 epoch 집계용이다 -- `bev_occupancy_metrics.mean_loss_parts`가
    영역 평균을 **그 영역의 셀 수로 가중해** 모아 "epoch 전체에서 그 영역 모든 셀의 평균"을 만든다.
    배치마다 같은 무게로 평균하면 영역 셀 수가 배치마다 달라 가중 BCE에서 `Σ frac·loss_r =
    loss`가 epoch 단위로 5 % 어긋났다(2026-10-02 스모크). 밑줄로 시작하는 키는 로그에 안 나간다.

    `set_coefficients`는 **epoch 총 손실을 영역 평균으로 다시 짓는 법**이다(`_objective_*` 키).
    `None`이면 셀 평균 목적함수(가중 BCE: `Σ frac_r · loss_r`), `(c_F, c_N, c_B)`면 영역별 평균의
    합(soft-BCE: `½·F + ½·N + λ_B·B`), `"batch"`면 짓지 않는다(보조항을 켠 옛 설정). epoch 총 손실을
    "그 epoch의 모든 셀을 한 집합으로 본 목적함수"로 정의하면 두 손실 모두 분해가 정확히 맞고
    배치 크기와 무관해진다(`bev_occupancy_metrics.epoch_loss`).
    """
    with torch.no_grad():
        log_probs = torch.log_softmax(logits, dim=1)
        gt = gt_free.to(log_probs.dtype)
        # hard CE. 목표는 라벨이 정한 0/1이고 어떤 손실도 이 식을 바꾸지 않는다.
        hard = -(gt * log_probs[:, 1:2] + (1.0 - gt) * log_probs[:, 0:1])
        valid_f = valid.to(log_probs.dtype)
        n_valid = valid_f.sum() + 1e-6
        blind = permanent_blind.to(valid.device) if permanent_blind is not None \
            else torch.zeros_like(valid)
        regions = region_masks(d, valid, blind, delta)
        per_cell = per_cell_loss.detach().to(log_probs.dtype)
        floor = (torch.zeros_like(per_cell) if floor_per_cell is None
                 else floor_per_cell.detach().to(log_probs.dtype))

        def mean(x, m):
            return (x * m).sum() / (m.sum() + 1e-6)

        out = {}
        for key, name in _REGIONS:
            m = regions[key].to(log_probs.dtype)
            out[f"loss_{name}"] = mean(per_cell, m)
            out[f"ce_{name}"] = mean(hard, m)
            out[f"frac_{name}"] = m.sum() / n_valid
        out["loss_boundary_floor"] = mean(floor, regions["omega_b"].to(log_probs.dtype))
        out["ce_all"] = mean(hard, valid_f)
        hidden = {"_n_valid": valid_f.sum()}
        if set_coefficients is None:
            hidden["_objective_mode"] = 0.0                       # 셀 평균
        elif set_coefficients == "batch":
            hidden["_objective_mode"] = -1.0                      # 짓지 않는다
        else:
            hidden["_objective_mode"] = 1.0                       # 영역별 평균의 가중합
            for (_, name), c in zip(_REGIONS, set_coefficients):
                hidden[f"_objective_c_{name}"] = float(c)
        # 학습 루프가 모든 항에 `.item()`을 부르므로 **전부 텐서로** 낸다(파이썬 float을 넣었다가
        # 첫 배치에서 죽었다 -- 2026-10-02 스모크).
        hidden = {k: torch.as_tensor(v, dtype=log_probs.dtype, device=log_probs.device)
                  for k, v in hidden.items()}
        return {**{k: out[k] for k in COMMON_LOSS_PARTS}, **hidden}


def run_batch(model, batch, vox_util, class_weights, device, rays, label_smoothing=0.0,
              permanent_blind=None):
    """가중 BCE 한 배치. 손실 로그는 `decompose_loss`의 공통 항이다(soft-BCE와 같은 이름).

    `permanent_blind`는 영역 분할에만 쓴다(사각지대를 `Ω_N`에 넣는다). 없으면 0 마스크다 --
    SynWoodScape에는 리그 고정 가림이 없다.
    """
    rgb_camXs = batch["rgb_camXs"].to(device) - 0.5
    pix_T_cams = batch["pix_T_cams"].to(device)
    cam0_T_camXs = batch["cam0_T_camXs"].to(device)
    seg_bev_g = batch["seg_bev_g"].to(device)
    vis_bev_g = batch["vis_bev_g"].to(device)
    valid_bev_g = batch["valid_bev_g"].to(device)

    _, _, logits, _, _ = model(rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util)
    class_index = to_class_index(decompose(seg_bev_g, vis_bev_g, valid_bev_g))
    loss, _ = compute_binary_loss(
        logits, class_index, valid_bev_g, class_weights, label_smoothing
    )
    # 최적화한 손실의 **셀별 값** -- `masked_weighted_ce`와 같은 식(클래스 가중 + smoothing).
    # 영역 평균을 내면 `loss = Σ frac_r · loss_r`이 정확히 성립한다(세 영역이 valid를 덮는다).
    per_cell = torch.nn.functional.cross_entropy(
        logits, class_index.squeeze(1), weight=class_weights.to(logits.device),
        reduction="none", label_smoothing=label_smoothing).unsqueeze(1)
    # hard target이라 하한은 0이다. (label smoothing > 0이면 하한이 0이 아니지만 캠페인은 0이다.)
    if "d_bev_g" not in batch:
        raise KeyError("공통 손실 분해에 `d_bev_g`(경계까지의 부호 거리)가 필요하다 -- "
                       "로봇·SynWoodScape 데이터셋은 둘 다 만든다")
    loss_parts = decompose_loss(
        per_cell, None, logits, decompose(seg_bev_g, vis_bev_g, valid_bev_g)["free"],
        valid_bev_g, batch["d_bev_g"].to(device), permanent_blind)
    return loss, loss_parts, compute_free_metrics(
        logits, seg_bev_g, vis_bev_g, valid_bev_g, rays
    )


def run_batch_soft_boundary(model, batch, vox_util, device, rays, permanent_blind,
                            delta=DEFAULT_DELTA_M, lambda_b=DEFAULT_LAMBDA_B,
                            target=TARGET_LINEAR, sigma=None, alpha=None,
                            gather=None, lambda_r=0.0, delta_r=DEFAULT_DELTA_R_M,
                            delta_r_over=None, huber_beta=DEFAULT_HUBER_BETA_M,
                            kappa=DEFAULT_KAPPA, eps=DEFAULT_EPS,
                            range_loss_mode="arc_huber"):
    """soft-boundary loss로 한 배치. 설계는 `docs/soft_boundary_loss_design.md`.

    `run_batch`와 **지표 계산은 완전히 같다** -- 다른 것은 loss 하나뿐이다. 그래야 두 loss의
    런을 한 표에 놓을 수 있고, 그 비교가 이 변경의 판정 근거다.

    `class_weights`를 받지 않는다. per-set 평균(`½L_F + ½L_N`)이 역빈도 가중치를 **대체**하기
    때문이다(설계 문서 §3.1) -- 둘을 같이 걸면 클래스 보정이 두 번 들어간다.

    `permanent_blind`는 배치가 아니라 정적 마스크로 받는다. 프레임마다 같은 값이므로 배치에
    실어 보내면 데이터 전송만 늘어난다. `(1, 1, H, W)`로 broadcast된다.

    `gather`(`range_loss.RayGather`)가 있고 `lambda_r > 0`이면 방위각 자유거리 보조항이
    붙는다(설계 문서 §13). **둘 중 하나라도 없으면 항이 계산조차 되지 않는다** -- 대조군과
    기존 스윕 런이 새 코드 경로를 타지 않아야 한다.
    """
    if range_loss_mode not in ("arc_huber", "cumulative_l1"):
        raise ValueError("range_loss_mode는 arc_huber 또는 cumulative_l1이어야 한다: "
                         f"{range_loss_mode}")

    rgb_camXs = batch["rgb_camXs"].to(device) - 0.5
    pix_T_cams = batch["pix_T_cams"].to(device)
    cam0_T_camXs = batch["cam0_T_camXs"].to(device)
    seg_bev_g = batch["seg_bev_g"].to(device)
    vis_bev_g = batch["vis_bev_g"].to(device)
    valid_bev_g = batch["valid_bev_g"].to(device)
    d_bev_g = batch["d_bev_g"].to(device)

    _, _, logits, _, _ = model(rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util)

    blind = permanent_blind.to(device)
    range_term = None
    if gather is not None and lambda_r > 0.0:
        # `p(free)`는 loss가 쓰는 것과 같은 softmax에서 나와야 한다 -- 여기서 따로 sigmoid를
        # 쓰면 두 항이 다른 확률을 보게 된다.
        prob_free = torch.softmax(logits, dim=1)[:, 1:2]
        free_gt = decompose(seg_bev_g, vis_bev_g, valid_bev_g)["free"]
        if range_loss_mode == "arc_huber":
            range_term = compute_range_loss(prob_free, free_gt, valid_bev_g, gather,
                                            delta_r=delta_r, delta_r_over=delta_r_over,
                                            beta=huber_beta)
        else:
            target_free = build_soft_boundary_target(
                d_bev_g, valid_bev_g, blind, delta=delta, kind=target,
                sigma=sigma, alpha=alpha, kappa=kappa, eps=eps)
            range_term = compute_cumulative_range_loss(
                prob_free, target_free, free_gt, valid_bev_g, gather)

    loss, own_parts = compute_soft_boundary_loss(
        logits, d_bev_g, valid_bev_g, blind,
        delta=delta, lambda_b=lambda_b, kind=target, sigma=sigma, alpha=alpha,
        range_term=range_term, lambda_r=lambda_r, kappa=kappa, eps=eps,
    )
    # 최적화한 손실의 셀별 값: 모든 영역에서 target y에 대한 CE(`Ω_F`/`Ω_N`에서 y = 1/0이면
    # hard CE와 같다). 하한은 y의 엔트로피다. **가중 BCE 경로와 같은 분해를 탄다.**
    with torch.no_grad():
        y = build_soft_boundary_target(d_bev_g, valid_bev_g, blind, delta=delta, kind=target,
                                       sigma=sigma, alpha=alpha, kappa=kappa, eps=eps)
        log_probs = torch.log_softmax(logits, dim=1)
        per_cell = -(y * log_probs[:, 1:2] + (1.0 - y) * log_probs[:, 0:1])
    loss_parts = decompose_loss(
        per_cell, target_entropy(y), logits,
        decompose(seg_bev_g, vis_bev_g, valid_bev_g)["free"], valid_bev_g, d_bev_g, blind,
        delta=DECOMPOSITION_DELTA_M,
        set_coefficients=("batch" if range_term is not None or float(delta) != DECOMPOSITION_DELTA_M
                          else (0.5, 0.5, float(lambda_b))))
    if range_term is not None:      # legacy: 보조항을 켠 옛 설정만 그 항을 더 낸다
        loss_parts.update({k: v for k, v in own_parts.items() if "range" in k})
    return loss, loss_parts, compute_free_metrics(
        logits, seg_bev_g, vis_bev_g, valid_bev_g, rays
    )
