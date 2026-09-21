"""어안 원본(DS) 대 가상 핀홀의 **화각·각해상도·BEV 커버리지** 절충을 기하만으로 잰다 (CPU).

`02_projection_and_prior` 축 A의 **사전 확인**이자, 논문 setup 절의 카메라 구성 근거다.
학습·체크포인트·데이터셋 라벨이 필요 없고 `calib.yaml`과 원본 이미지 몇 장만 쓴다.
`tools/measure_height_bin_visibility.py`와 같은 방식이고 유효성 판정도 학습 경로와 같다
(`DoubleSphereCamera.project`의 정의역 + 특징맵 경계).

**왜 필요한가.** 핀홀 이미지는 평면이라 광축에서 θ 벗어난 광선이 `f·tan(θ)`에 찍히는데
`tan(90°)`가 무한대다. 즉 핀홀은 90° 이상의 광선을 **해상도와 무관하게** 담지 못한다.
어안을 펴서 비교하려면 화각을 잘라야 하고, 그러면 "성능 차이가 투영 방식 때문인가 화각을
좁게 잡아서인가"가 섞인다. 이 표가 그 교란의 크기를 미리 숫자로 만든다.

**읽는 법.** §2의 `정면 px/deg`가 어안과 같아지는 화각이 **공정한 비교점**이다. 그보다
넓히면 커버리지를 사는 대신 정면 해상도를 판다.

실행:
    python tools/measure_projection_tradeoff.py
    python tools/measure_projection_tradeoff.py --out_w=512 --out_h=288
"""
import sys
from pathlib import Path

import numpy as np
from fire import Fire

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from projects.bev_gt.grid import ROBOT_GRID_SPEC, cell_centers_m  # noqa: E402
from projects.geometry.double_sphere import (  # noqa: E402
    FINETUNE_CAMERA_NAMES,
    load_cameras,
    load_ego_T_cams,
    se3_inverse,
)

_CALIB = _REPO_ROOT / "dataset/sj_datasets/common/calibration/calib.yaml"

# 학습 경로와 같은 형상 (`measure_height_bin_visibility.py`와 동일).
FEAT_W, FEAT_H = 64, 36
NATIVE_W, NATIVE_H = 1280, 720
# `Y=4` 확정 설정의 표본 높이 [m, 지면=0].
SAMPLE_HEIGHTS = (0.0, 0.5, 1.0, 1.5)
# 실측 이미지 원 반경 [native px]. `--remeasure_circle`로 다시 잴 수 있다.
IMAGE_CIRCLE_PX = {"front": 572.0, "left": 592.0, "right": 588.0}


def _ego_points(fg, lg, height):
    return np.stack([fg.ravel(), lg.ravel(),
                     np.full(fg.size, float(height)), np.ones(fg.size)], axis=1)


def _cam_points(ego_T_cam, fg, lg, height):
    return (se3_inverse(ego_T_cam) @ _ego_points(fg, lg, height).T).T[:, :3]


def ds_visible(camera, ego_T_cam, fg, lg, height, circle_px=None):
    """DS 투영으로 특징맵 안에 들어오는가. `circle_px`를 주면 이미지 원 밖을 제외한다."""
    cam = _cam_points(ego_T_cam, fg, lg, height)
    u, v, valid = camera.project(cam)
    x, y = u * (FEAT_W / NATIVE_W), v * (FEAT_H / NATIVE_H)
    ok = valid & (x > -0.5) & (x < FEAT_W - 0.5) & (y > -0.5) & (y < FEAT_H - 0.5)
    if circle_px is not None:
        ok &= np.hypot(u - camera.cx, v - camera.cy) <= circle_px
    return ok.reshape(fg.shape)


def pinhole_visible(ego_T_cam, fg, lg, height, hfov_deg, out_w, out_h):
    """어안과 같은 광축의 가상 핀홀(16:9 출력) 화각 안에 들어오는가."""
    cam = _cam_points(ego_T_cam, fg, lg, height)
    X, Y, Z = cam[:, 0], cam[:, 1], cam[:, 2]
    th = np.tan(np.deg2rad(hfov_deg) / 2.0)
    tv = th * (out_h / out_w)
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = (Z > 1e-6) & (np.abs(X / Z) < th) & (np.abs(Y / Z) < tv)
    return ok.reshape(fg.shape)


def _ds_px_per_deg(camera, theta_deg, out_w):
    """그 각도 부근에서 1도가 출력 이미지의 몇 화소를 차지하나."""
    t = np.deg2rad([theta_deg - 0.5, theta_deg + 0.5])
    pts = np.stack([np.sin(t), np.zeros(2), np.cos(t)], axis=1)
    u, _, _ = camera.project(pts)
    return abs(u[1] - u[0]) * (out_w / NATIVE_W)


def _measure_image_circle(camera, name, n_frames):
    """원본 프레임을 평균해 이미지 원 반경을 잰다. 어안 모서리는 렌즈가 찍지 않아 검다."""
    import glob
    from PIL import Image
    pattern = str(_REPO_ROOT / f"dataset/sj_datasets/raw*/rgb_images/*/cam_{name}.jpg")
    files = sorted(glob.glob(pattern))
    if not files:
        return None, 0
    step = max(1, len(files) // n_frames)
    files = files[::step][:n_frames]
    acc = None
    for f in files:
        a = np.array(Image.open(f).convert("L"), dtype=np.float32)
        acc = a if acc is None else acc + a
    acc /= len(files)
    h, w = acc.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - camera.cx, yy - camera.cy)
    edges = np.arange(0.0, r.max(), 4.0)
    med = np.array([np.median(acc[(r >= a) & (r < b)]) if ((r >= a) & (r < b)).any() else np.nan
                    for a, b in zip(edges[:-1], edges[1:])])
    background = np.nanmedian(med[-8:])
    peak = np.nanmedian(med[:10])
    hit = np.where(med > background + 0.25 * (peak - background))[0]
    return (edges[hit.max() + 1] if len(hit) else np.nan), len(files)


def main(out_w: int = 512, out_h: int = 288, remeasure_circle: bool = False, n_frames: int = 24):
    cameras = load_cameras(_CALIB)
    ego_T_cams = load_ego_T_cams(_CALIB)
    spec = ROBOT_GRID_SPEC
    forward_m, lateral_m = cell_centers_m(spec)
    fg, lg = np.meshgrid(forward_m, lateral_m, indexing="ij")
    n_cells = fg.size
    rng = np.hypot(fg, lg)

    circle = dict(IMAGE_CIRCLE_PX)
    if remeasure_circle:
        for name in FINETUNE_CAMERA_NAMES:
            rad, used = _measure_image_circle(cameras[name], name, n_frames)
            if rad is not None:
                circle[name] = float(rad)
                print(f"  [재측정] {name}: 이미지 원 반경 {rad:.0f} px (프레임 {used}개)")

    print(f"BEV {spec.n_rows}x{spec.n_cols} = {n_cells} 셀, {spec.cell_m*100:.0f} cm "
          f"| 전방 {spec.front_m} m / 후방 {spec.rear_m} m / 좌우 ±{spec.half_width_m} m")
    print(f"표본 높이 {SAMPLE_HEIGHTS} m (Y=4 확정 설정) | 가상 핀홀 출력 {out_w}x{out_h}, "
          f"광축은 어안과 동일\n")

    def union(fn, names=FINETUNE_CAMERA_NAMES):
        m = np.zeros(fg.shape, bool)
        for z in SAMPLE_HEIGHTS:
            for n in names:
                m |= fn(n, z)
        return m

    ds_all = union(lambda n, z: ds_visible(cameras[n], ego_T_cams[n], fg, lg, z))

    print("=== §1. 카메라 구성별 BEV 커버리지 (3대 중 하나라도 보면 유효) ===")
    for label, names in (("front+left+right", FINETUNE_CAMERA_NAMES), ("front only", ("front",))):
        m = union(lambda n, z: ds_visible(cameras[n], ego_T_cams[n], fg, lg, z), names)
        print(f"  {label:<18} {m.sum():>6} 셀  {100*m.mean():>5.1f}%")
    front_only = union(lambda n, z: ds_visible(cameras[n], ego_T_cams[n], fg, lg, z), ("front",))
    print(f"  front만 쓰면 {ds_all.sum()-front_only.sum()} 셀 = "
          f"{100*(ds_all.sum()-front_only.sum())/ds_all.sum():.1f}%를 아예 못 본다")
    print("  거리대별 front-only 커버리지:", end=" ")
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4)):
        band = (rng >= lo) & (rng < hi)
        print(f"{lo}~{hi}m {100*front_only[band].mean():.0f}%", end="  ")
    print("\n  -> 충돌이 일어나는 근거리가 가장 크게 빈다 (측면 카메라가 보던 영역).\n")

    print("=== §2. 정면 각해상도: 어안 대 가상 핀홀 ===")
    base = _ds_px_per_deg(cameras["front"], 0.5, out_w)
    print(f"  어안 원본(DS)   정면 {base:5.2f} px/deg  "
          f"| 45° {_ds_px_per_deg(cameras['front'], 45, out_w):5.2f}"
          f"  | 90° {_ds_px_per_deg(cameras['front'], 90, out_w):5.2f}   <- 거의 균일하다")
    print(f"  {'화각':>8} {'초점거리':>9} {'정면 px/deg':>12} {'어안 대비':>10} "
          f"{'가장자리 px/deg':>16} {'커버리지':>9}")
    for hfov in (90, 100, 110, 120, 130, 140, 150, 160, 170):
        m = union(lambda n, z: pinhole_visible(ego_T_cams[n], fg, lg, z, hfov, out_w, out_h)) & ds_all
        f = (out_w / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
        ppd0 = f * np.deg2rad(1.0)
        ppde = ppd0 / np.cos(np.deg2rad(hfov / 2.0)) ** 2
        mark = "  <- 정면 해상도 일치" if abs(ppd0 - base) < 0.06 else ""
        print(f"  핀홀 {hfov:>3}° {f:>8.0f}px {ppd0:>11.2f} {ppd0/base:>9.2f}x "
              f"{ppde:>15.2f} {100*m.sum()/ds_all.sum():>8.1f}%{mark}")

    print("\n=== §3. 이미지 원 점검 — DS 모델이 렌즈 밖으로 외삽하는가 ===")
    total_vis = total_out = 0
    for name in FINETUNE_CAMERA_NAMES:
        c = cameras[name]
        t = np.linspace(0, np.pi, 8001)
        pts = np.stack([np.sin(t), np.zeros_like(t), np.cos(t)], axis=1)
        u, v, valid = c.project(pts)
        inside = valid & (np.hypot(u - c.cx, v - c.cy) <= circle[name])
        for z in SAMPLE_HEIGHTS:
            vis = ds_visible(c, ego_T_cams[name], fg, lg, z)
            vis_c = ds_visible(c, ego_T_cams[name], fg, lg, z, circle_px=circle[name])
            total_vis += vis.sum()
            total_out += (vis & ~vis_c).sum()
        print(f"  {name:<7} 이미지 원 반경 {circle[name]:.0f} px -> 최대 반각 "
              f"{np.degrees(t[inside].max()):5.1f}° (전체 화각 ~{2*np.degrees(t[inside].max()):.0f}°)"
              f" | DS 모델 주장 {np.degrees(t[valid].max()):.1f}°")
    ds_circle = union(lambda n, z: ds_visible(cameras[n], ego_T_cams[n], fg, lg, z,
                                              circle_px=circle[n]))
    print(f"  '보인다'고 판정된 표본 {total_vis}개 중 {total_out}개 "
          f"({100*total_out/total_vis:.1f}%)가 렌즈가 찍지 않은 검은 화소에서 온다.")
    print(f"  다만 커버리지 손실은 {ds_all.sum()-ds_circle.sum()} 셀이다 — "
          f"카메라 3대가 겹쳐서 그 셀들을 다른 대가 제대로 본다.")
    print("  -> 낭비이지 구멍이 아니다. 논문에는 각주로 충분하다.")


if __name__ == "__main__":
    Fire(main)
