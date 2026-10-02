"""캠페인 세 러너가 공유하는 보조항(`L_range`) 설정.

**왜 한 곳에 두나.** `01_overall`·`02_projection_and_prior`·`02b_native_source_prior`가
각자 `LAMBDA_R`을 적어 두면, 보조항 형태를 바꿀 때 한 러너만 고쳐지고 나머지는 조용히
옛 값으로 남는다. 캠페인의 모든 런은 **같은 보조항**으로 학습돼야 짝지은 비교가 성립한다.

두 형태의 값은 `runs/loss_effect`의 `D_range_s*`·`E_cumulative_s*` config.json에서
글자 그대로 가져왔다 -- 실험 3의 사다리와 캠페인이 같은 목적함수를 쓰게 하려는 것이다.

`cumulative_l1`에서 `DELTA_R_M`/`HUBER_BETA_M`은 **쓰이지 않는다**(trainer가 무시한다).
그래도 넘기는 이유는 `E_cumulative`의 config.json에 그 값이 그대로 기록돼 있어서다 --
빼면 config diff에 의미 없는 차이가 생긴다.

**[2026-10-02 사용자 결정] 다음 캠페인은 광선 보조항을 아예 쓰지 않는다** -- 목적함수는
soft-boundary BCE(`½L_F + ½L_N + λ_B·L_B`)까지다. 그래서 기본값이 `none`이다. `λ_R = 0`이면
trainer가 항을 **만들지도 않는다**(`binary_metrics.run_batch_soft_boundary`), 그래서
`RANGE_LOSS_MODE` 등은 넘기지 않는다 -- config.json의 `range_loss_active: false`가 증거다.
`arc_huber`/`cumulative_l1`은 옛 두 캠페인을 재현하려고 남긴다.
"""
AUX_LOSS_ENV = {
    # 2026-10-02 이후의 기본값. 보조항 없음.
    "none": {
        "LAMBDA_R": "0.0",
    },
    # 2026-09-22 이전 캠페인(`runs/paper_final/`) 전체가 이 값으로 돌았다.
    "arc_huber": {
        "RANGE_LOSS_MODE": "arc_huber",
        "LAMBDA_R": "0.3",
        "DELTA_R_M": "0.15",
        "HUBER_BETA_M": "0.15",
    },
    # 2026-09-23 캠페인(`runs/paper_final_cumulative/`)의 형태(원장 §8). λ_R이 0.3이 아니라
    # **0.15**다 -- 형태가 다르면 gradient 크기도 달라서 같은 λ_R을 쓸 수 없다.
    "cumulative_l1": {
        "RANGE_LOSS_MODE": "cumulative_l1",
        "LAMBDA_R": "0.15",
        "DELTA_R_M": "0.15",
        "HUBER_BETA_M": "0.15",
    },
}
DEFAULT_AUX_LOSS = "none"


def aux_loss_env(mode: str) -> dict:
    """`mode`에 해당하는 환경변수 묶음. 호출자는 이 dict를 런 환경에 그대로 합친다."""
    if mode not in AUX_LOSS_ENV:
        raise SystemExit(f"알 수 없는 보조항 형태 {mode!r}. 가능한 값: {list(AUX_LOSS_ENV)}")
    return dict(AUX_LOSS_ENV[mode])


def add_aux_loss_argument(parser) -> None:
    parser.add_argument(
        "--range_loss_mode", default=DEFAULT_AUX_LOSS, choices=sorted(AUX_LOSS_ENV),
        help="보조항 L_range의 형태. 캠페인 전체가 같은 값이어야 한다 "
             "(기본 none = 보조항 없음, 2026-10-02 결정)")
