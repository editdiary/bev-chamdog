"""Render instance segmentation polygons over the RGB image.

Source: instance_annotations (the master annotation — JSON polygons with 40+
class tags). Each instance is drawn as a translucent filled polygon (painted
back-to-front by z_order) plus an outline, colored deterministically per tag.
"""
import cv2
import numpy as np

import ws_io


def render(ws: "ws_io.WoodScape", sample: str, alpha: float = 0.4):
    img = ws.rgb(sample)
    data = ws.instance(sample)
    anns = sorted(data.get("annotation", []), key=lambda a: a.get("z_order", 0))

    overlay = img.copy()
    present = {}
    for a in anns:
        seg = a.get("segmentation")
        if not seg or len(seg) < 3:
            continue
        pts = np.round(np.asarray(seg, dtype=np.float32)).astype(np.int32)
        tag = a["tags"][0] if a.get("tags") else "void"
        color = ws_io.tag_color(tag)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)
        present.setdefault(tag, color)

    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    # legend can be long (40+ tags); cap to the classes present in this frame
    ws_io.draw_legend(out, list(present.items())[:20])
    return out
