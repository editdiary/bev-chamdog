"""Render 2D bounding boxes over the RGB image.

Source: box_2d_annotations (precomputed .txt, 5 classes). Colors come from
box_2d_annotation_info.json.
"""
import cv2

import ws_io


def render(ws: "ws_io.WoodScape", sample: str):
    img = ws.rgb(sample)
    names, colors = ws.box_classes()
    present = {}
    for name, idx, x0, y0, x1, y1 in ws.box2d(sample):
        color = ws_io.rgb_to_bgr(colors[idx]) if idx < len(colors) else (0, 255, 0)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(img, name, (x0, max(10, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        present[name] = color
    ws_io.draw_legend(img, list(present.items()))
    return img
