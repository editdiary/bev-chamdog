#!/usr/bin/env python3
"""WoodScape ICCV19 label visualizer.

Overlays annotations on the RGB fisheye images and saves them as PNG files, so
you can inspect how each label type is actually built.

Examples:
    # one sample, all label types (+ a stacked panel)
    python tools/woodscape_viz/visualize.py --sample 00000_FV \
        --out outputs/woodscape_viz

    # only some types
    python tools/woodscape_viz/visualize.py --sample 01234_MVL \
        --types box,semantic --out outputs/woodscape_viz

    # gallery: N random samples per camera, all types
    python tools/woodscape_viz/visualize.py --gallery 3 \
        --out outputs/woodscape_viz

Run inside the `mmdet3d` conda env (needs numpy, opencv, scipy).
"""
import argparse
import random
from pathlib import Path

import cv2

import ws_io
import viz_box
import viz_semantic
import viz_instance
import viz_motion
import viz_calib

RENDERERS = {
    "box": viz_box.render,
    "semantic": viz_semantic.render,
    "instance": viz_instance.render,
    "motion": viz_motion.render,
    "calib": viz_calib.render,
}
ALL_TYPES = list(RENDERERS)

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "dataset" / "woodscape"


def montage(images, width=960):
    """Stack images vertically, each resized to a common width."""
    rows = []
    for im in images:
        h, w = im.shape[:2]
        rows.append(cv2.resize(im, (width, int(round(h * width / w)))))
    return cv2.vconcat(rows)


def render_sample(ws, sample, types, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    produced = []
    for t in types:
        try:
            img = RENDERERS[t](ws, sample)
        except Exception as e:  # keep going if one label type is missing/broken
            print(f"  [warn] {sample} {t}: {e}")
            continue
        path = out_dir / f"{sample}_{t}.png"
        cv2.imwrite(str(path), img)
        produced.append((t, img))
        print(f"  wrote {path}")
    # a combined panel (skip the wide calib image so the stack stays readable)
    panel_imgs = [img for t, img in produced if t != "calib"]
    if len(panel_imgs) > 1:
        panel = montage(panel_imgs)
        cv2.imwrite(str(out_dir / f"{sample}_panel.png"), panel)
        print(f"  wrote {out_dir / f'{sample}_panel.png'}")
    return produced


def gallery_samples(ws, per_cam, seed):
    rng = random.Random(seed)
    all_samples = ws.list_samples()
    by_cam = {c: [] for c in ws_io.CAMERAS}
    for s in all_samples:
        cam = s.split("_")[-1]
        if cam in by_cam:
            by_cam[cam].append(s)
    picked = []
    for cam in ws_io.CAMERAS:
        pool = by_cam[cam]
        picked.extend(rng.sample(pool, min(per_cam, len(pool))))
    return picked


def main():
    ap = argparse.ArgumentParser(description="WoodScape label visualizer")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="dataset root (default: repo dataset/woodscape)")
    ap.add_argument("--sample", help="sample id, e.g. 00000_FV")
    ap.add_argument("--gallery", type=int, metavar="N",
                    help="render N random samples per camera instead of --sample")
    ap.add_argument("--types", default=",".join(ALL_TYPES),
                    help=f"comma list of {ALL_TYPES}")
    ap.add_argument("--out", type=Path, default=Path("outputs/woodscape_viz"))
    ap.add_argument("--seed", type=int, default=0, help="gallery sampling seed")
    args = ap.parse_args()

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    bad = [t for t in types if t not in RENDERERS]
    if bad:
        ap.error(f"unknown types {bad}; choose from {ALL_TYPES}")

    ws = ws_io.WoodScape(args.root)

    if args.gallery:
        samples = gallery_samples(ws, args.gallery, args.seed)
    elif args.sample:
        samples = [args.sample]
    else:
        ap.error("provide --sample or --gallery N")

    for sample in samples:
        print(f"[{sample}]")
        render_sample(ws, sample, types, args.out / sample)


if __name__ == "__main__":
    main()
