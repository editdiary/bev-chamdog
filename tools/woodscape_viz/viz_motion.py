"""Overlay the motion (moving-object) segmentation mask on the RGB image.

Motion labels ship as color-coded rgbLabels (19 classes). We blend them and
build the legend by matching mask colors against motion_annotation_info.json.
(The gtLabels index convention is ambiguous, so we key the legend off the
authoritative rgbLabels colors instead.)
"""
import ws_io


def render(ws: "ws_io.WoodScape", sample: str, alpha: float = 0.55):
    img = ws.rgb(sample)
    color_mask = ws.motion_rgb(sample)
    out = ws_io.blend_nonzero(img, color_mask, alpha)

    names, colors = ws.motion_classes()
    ws_io.draw_legend(out, ws_io.legend_from_color_mask(color_mask, names, colors))
    return out
