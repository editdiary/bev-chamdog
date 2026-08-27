"""`projects/bev_gt` grid_spec -> Simple-BEV `Vox_util`/`Segnet(Z, Y, X)` 설정.

Simple-BEV의 참조("ref"/"cam0") 프레임은 축 이름(Z/Y/X) 자체에 물리적 의미를 두지 않지만,
코드 관례상 두 가지는 고정돼 있다 (`nets/segnet.py`):
- `Y`축은 `bev_compressor`가 채널로 접어버리는 축이다 → 우리에게는 **높이**여야 한다
  (occupancy GT가 지면 2D뿐이므로 `Y=1`).
- `(Z, X)`가 decoder 출력의 (row, col) 두 공간축이 된다.

`projects/geometry`가 검증한 ego 프레임(X=전방, Y=좌측, Z=상방, 오른손계)은 이 축 이름과
다르므로, **회전만으로** 축을 맞바꾸는 고정 행렬 `EGO_TO_REF_ROTATION`을 둔다(평행이동 없음 —
두 프레임 모두 원점이 차량 중심·지면 높이로 같다). 부호는 `projects.bev_gt.grid`의 관례
(`cell_centers_m`: row 0 = 최전방, col 0 = 차량 좌측)가 `Vox_util`의 mem index 증가 방향
(각 축 MIN→MAX일 때 index 0→N-1)과 그대로 맞도록 다음처럼 정했다 — 그래서 occupancy GT
배열을 따로 뒤집을 필요가 없다:

    ref.x = -ego.y   (ego.y=+half_width(좌측)일 때 ref.x=XMIN → mem_x=0 = col 0 = 좌측)
    ref.y =  ego.z   (높이. Y=1 bin이라 방향 무관)
    ref.z = -ego.x   (ego.x=+front_m(최전방)일 때 ref.z=ZMIN → mem_z=0 = row 0 = 최전방)

수치 검증(`tests/datasets/test_simplebev_vox.py`)이 이 부호가 실제로 `cell_centers_m`과
일치함을 `Vox_util.Ref2Mem`으로 직접 확인한다.
"""
import pathlib
import sys
from pathlib import Path

import numpy as np
import torch

_SIMPLE_BEV_DIR = Path(__file__).resolve().parents[2] / "third_party/models/simple_bev"
if str(_SIMPLE_BEV_DIR) not in sys.path:
    sys.path.insert(0, str(_SIMPLE_BEV_DIR))

import utils.vox  # noqa: E402  (simple_bev submodule; see docs/project_structure.md)

from projects.bev_gt.grid import OccupancyGridSpec  # noqa: E402

EGO_TO_REF_ROTATION = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
    ]
)


def ref_T_ego_4x4() -> np.ndarray:
    ref_T_ego = np.eye(4, dtype=np.float64)
    ref_T_ego[:3, :3] = EGO_TO_REF_ROTATION
    return ref_T_ego


def ref_T_cam_from_ego_T_cam(ego_T_cam: np.ndarray) -> np.ndarray:
    """`ego_T_cam`(camera -> ego) -> `ref_T_cam`(camera -> Simple-BEV ref frame).

    `Segnet.forward`에 `cam0_T_camXs`로 그대로 넘긴다(변수명은 "cam0"이지만 실제로는 이
    ref 프레임 — Segnet은 그 이름에 특별한 의미를 두지 않는다).
    """
    return ref_T_ego_4x4() @ ego_T_cam


# 높이 축의 기본값. **`height_bins=1`이 여태까지의 전 실험 설정이고 기본값으로 남는다** --
# 이 값을 바꾸면 `bev_compressor`의 입력 채널이 바뀌어 옛 체크포인트와 호환되지 않는다.
DEFAULT_HEIGHT_BINS = 1
DEFAULT_HEIGHT_MARGIN_M = 0.25


def vox_dims(grid_spec: OccupancyGridSpec, height_bins: int = DEFAULT_HEIGHT_BINS) -> tuple:
    """(Z, Y, X) - `Segnet(Z, Y, X, ...)` 생성자에 그대로 넘긴다.

    `height_bins`가 곧 `Y`다. **여기가 `Y`의 단일 출처다** -- 예전에는 호출부 20곳이
    리터럴 `1`을 각자 들고 있었고, 그러면 학습과 평가가 어긋나도 알아채기 어렵다.
    """
    if height_bins < 1:
        raise ValueError(f"height_bins는 1 이상이어야 한다: {height_bins}")
    return grid_spec.n_rows, height_bins, grid_spec.n_cols


def vox_bounds(grid_spec: OccupancyGridSpec,
               height_margin_m: float = DEFAULT_HEIGHT_MARGIN_M,
               height_min_m=None, height_max_m=None) -> tuple:
    """(XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX) - `Vox_util(bounds=...)`에 그대로 넘긴다.

    높이(ref Y = ego Z)를 정하는 방법이 둘이다:

    - `height_margin_m`만 주면 **`z=0` 대칭** 슬래브 `[-margin, +margin]`. 기본값이고
      `Y=1`일 때 표본 평면이 정확히 `z=0`이 된다(`Vox_util`이 복셀 **중심**에서 표본하므로).
    - `height_min_m`/`height_max_m`를 주면 **비대칭 범위**. `Y>1`에서 지면 위쪽 대역을
      덮으려면 이쪽이 필요하다 -- 대칭 슬래브로는 지면 위 1.7 m를 잡을 수 없다.

    비대칭 범위를 쓸 때 **bin 중심이 어디 오는지 반드시 확인할 것.** mem index `i`의 높이는
    `height_min_m + size*(i+0.5)`이고 `size = (max-min)/Y`다. 예를 들어 `[-0.125, 1.875]`에
    `Y=8`이면 중심이 `0, 0.25, ..., 1.75`가 되어 **bin 0이 기존 `z=0` 평면과 정확히 일치**한다.
    반면 `[0, 1.7]`에 `Y=8`이면 최하단 중심이 0.106 m라 기존 평면이 사라진다.
    """
    if (height_min_m is None) != (height_max_m is None):
        raise ValueError("height_min_m과 height_max_m은 함께 주거나 함께 생략해야 한다")
    if height_min_m is None:
        height_min_m, height_max_m = -height_margin_m, height_margin_m
    if not height_max_m > height_min_m:
        raise ValueError(f"height_max_m > height_min_m 이어야 한다: {height_min_m}, {height_max_m}")
    return (
        -grid_spec.half_width_m, grid_spec.half_width_m,
        float(height_min_m), float(height_max_m),
        -grid_spec.front_m, grid_spec.rear_m,
    )


def height_bin_centers_m(bounds: tuple, height_bins: int):
    """`vox_bounds` 결과 -> 각 mem index가 실제로 표본하는 높이 [m], 지면 기준.

    `Vox_util.get_mem_T_ref`의 `center_T_ref[:,1,3] = -YMIN - vox_size_Y/2`와 같은 규약이다.
    `tests/datasets/test_simplebev_vox.py`가 이 함수를 `Vox_util`과 직접 대조한다.
    """
    y_min, y_max = bounds[2], bounds[3]
    size = (y_max - y_min) / float(height_bins)
    return [y_min + size * (i + 0.5) for i in range(height_bins)]


HEIGHT_CONFIG_NAME = "height.json"


def save_height_config(ckpt_dir, height_bins: int, height_min_m=None, height_max_m=None) -> None:
    """체크포인트 **옆에** 높이 설정을 남긴다.

    `saverloader.save`(third_party, 수정 금지)가 state_dict만 저장하므로, 평가 도구가
    `Y`와 높이 범위를 되찾을 방법이 없다. `Y`만은 `bev_compressor` 형상에서 되읽을 수
    있지만(`simplebev_three_class.height_bins_from_state_dict`) **범위는 형상에 안 남는다.**
    범위가 틀리면 표본 높이가 통째로 달라지는데 에러 없이 조용히 틀린 숫자가 나온다.
    """
    import json
    path = pathlib.Path(ckpt_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / HEIGHT_CONFIG_NAME).write_text(json.dumps(
        {"height_bins": int(height_bins),
         "height_min_m": height_min_m, "height_max_m": height_max_m},
        indent=2, sort_keys=True))


def load_height_config(ckpt_path) -> dict:
    """체크포인트 경로(파일 또는 디렉터리) -> 높이 설정 dict.

    파일이 없으면 **옛 기본값**(`Y=1`, `z=0` 대칭 슬래브)을 돌려준다 -- 이 손잡이가
    생기기 전에 만들어진 체크포인트가 전부 그 설정이므로 하위 호환이 유지된다.
    """
    import json
    path = pathlib.Path(ckpt_path)
    directory = path if path.is_dir() else path.parent
    config = directory / HEIGHT_CONFIG_NAME
    if not config.exists():
        return {"height_bins": DEFAULT_HEIGHT_BINS, "height_min_m": None, "height_max_m": None}
    return json.loads(config.read_text())


def height_config_for_ckpt_dirs(ckpt_dirs, *, quiet=False) -> dict:
    """여러 런의 체크포인트 폴더에서 학습 때 쓴 높이 설정을 되찾는다.

    `projects.models.pixel_grid.convention_for_run_dirs`와 같은 규율이다 -- **설정이 섞여
    있으면 `SystemExit`이다.** 표본 높이가 다른 런을 한 표에 세우면 서로 다른 기하의 숫자가
    한 열에 섞인다. 없는 폴더는 건너뛰고, 하나도 못 찾으면 옛 기본값(`Y=1`)을 쓴다.
    """
    found = {}
    for ckpt_dir in ckpt_dirs:
        path = pathlib.Path(ckpt_dir)
        if not (path if path.is_dir() else path.parent).exists():
            continue
        config = load_height_config(path)
        key = (config["height_bins"], config["height_min_m"], config["height_max_m"])
        found.setdefault(key, []).append(path.name)

    if not found:
        if not quiet:
            print(f"[표본 높이] height.json을 못 찾았다 -> 옛 기본값 Y={DEFAULT_HEIGHT_BINS}")
        return {"height_bins": DEFAULT_HEIGHT_BINS, "height_min_m": None, "height_max_m": None}

    if len(found) > 1:
        lines = [f"  Y={k[0]}, 범위 {k[1]}~{k[2]}: {', '.join(sorted(v))}"
                 for k, v in sorted(found.items(), key=lambda kv: str(kv[0]))]
        raise SystemExit(
            "학습 때 쓴 lifting 높이 설정이 런마다 다르다 -- 한 표에 세울 수 없다:\n"
            + "\n".join(lines))

    bins, low, high = next(iter(found))
    return {"height_bins": bins, "height_min_m": low, "height_max_m": high}


def build_vox_util(grid_spec: OccupancyGridSpec, height_margin_m: float = DEFAULT_HEIGHT_MARGIN_M,
                   device="cpu", height_bins: int = DEFAULT_HEIGHT_BINS,
                   height_min_m=None, height_max_m=None):
    Z, Y, X = vox_dims(grid_spec, height_bins)
    bounds = vox_bounds(grid_spec, height_margin_m, height_min_m, height_max_m)
    scene_centroid = torch.zeros(1, 3, dtype=torch.float32, device=device)
    return utils.vox.Vox_util(Z, Y, X, scene_centroid=scene_centroid, bounds=bounds, assert_cube=False)
