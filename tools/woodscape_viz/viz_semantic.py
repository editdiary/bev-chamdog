"""Overlay the semantic segmentation mask (10 classes) on the RGB image.

We blend the provided rgbLabels (already correctly colored) and build the
legend by matching mask colors against seg_annotation_info.json.
"""
import ws_io


def render(ws: "ws_io.WoodScape", sample: str, alpha: float = 0.5):
    img = ws.rgb(sample)
    color_mask = ws.semantic_rgb(sample)
    out = ws_io.blend_nonzero(img, color_mask, alpha)

    names, colors = ws.semantic_classes()
    ws_io.draw_legend(out, ws_io.legend_from_color_mask(color_mask, names, colors))
    return out
