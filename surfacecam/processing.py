"""Frame processing: IR visualisation, RGB+IR fusion, and a proximity map.

Note on "depth": the Surface Hello camera has no depth sensor, so none of
this is metric depth. `proximity_map` uses IR flood intensity as a rough
closeness proxy (brighter ~ nearer, because IR illumination falls off with
distance). It is clearly labelled as such in the UI.
"""
from __future__ import annotations

import cv2
import numpy as np


def normalize_gray(ir: np.ndarray, clip: float = 1.0) -> np.ndarray:
    """Stretch IR contrast to full 0-255 range."""
    if clip < 1.0:
        lo = np.percentile(ir, (1 - clip) * 50)
        hi = np.percentile(ir, 100 - (1 - clip) * 50)
        ir = np.clip((ir.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1) * 255
        return ir.astype(np.uint8)
    return cv2.normalize(ir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def ir_view(ir: np.ndarray) -> np.ndarray:
    g = normalize_gray(ir)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def ir_edges(ir: np.ndarray) -> np.ndarray:
    g = normalize_gray(ir)
    edges = cv2.Canny(g, 60, 160)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def proximity_map(ir: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """False-colored IR-intensity 'proximity' map (NOT metric depth)."""
    g = normalize_gray(ir, clip=0.98)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    heat = cv2.applyColorMap(g, colormap)
    return heat


class Aligner:
    """Rough manual alignment of the square IR frame onto the wide color frame.

    The two sensors sit side by side with different FOVs, so there is a fixed
    offset/scale. There's no factory calibration exposed, so we let the user
    nudge it live (arrow keys) and persist nothing - defaults look decent.
    """

    def __init__(self):
        self.scale = 1.30   # IR is zoomed vs color
        self.dx = 0
        self.dy = 0

    def nudge(self, dx=0, dy=0, dscale=0.0):
        self.dx += dx
        self.dy += dy
        self.scale = max(0.5, min(3.0, self.scale + dscale))

    def warp_ir_to_color(self, ir_gray: np.ndarray, color_shape) -> np.ndarray:
        """Return an IR gray image resampled into the color frame geometry."""
        ch, cw = color_shape[:2]
        ih, iw = ir_gray.shape[:2]
        # scale IR so its height ~ color height * scale factor, then center + offset
        target_h = int(ch * self.scale)
        target_w = int(iw * (target_h / ih))
        resized = cv2.resize(ir_gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((ch, cw), np.uint8)
        # top-left placement to center the IR image, plus manual offset
        x0 = (cw - target_w) // 2 + self.dx
        y0 = (ch - target_h) // 2 + self.dy
        # compute overlapping region
        sx0 = max(0, -x0); sy0 = max(0, -y0)
        dx0 = max(0, x0);  dy0 = max(0, y0)
        w = min(target_w - sx0, cw - dx0)
        h = min(target_h - sy0, ch - dy0)
        if w > 0 and h > 0:
            canvas[dy0:dy0 + h, dx0:dx0 + w] = resized[sy0:sy0 + h, sx0:sx0 + w]
        return canvas


def fuse(color_bgr: np.ndarray, ir_gray_aligned: np.ndarray,
         alpha: float = 0.5, tint: int = cv2.COLORMAP_BONE) -> np.ndarray:
    """Blend aligned IR (as a cyan-ish overlay) over the RGB image."""
    g = normalize_gray(ir_gray_aligned)
    ir_color = cv2.applyColorMap(g, tint)
    # only blend where IR has signal (avoid darkening black borders)
    mask = (ir_gray_aligned > 0).astype(np.float32)[..., None]
    blended = color_bgr.astype(np.float32) * (1 - alpha * mask) + \
        ir_color.astype(np.float32) * (alpha * mask)
    return blended.clip(0, 255).astype(np.uint8)


def edge_fuse(color_bgr: np.ndarray, ir_gray_aligned: np.ndarray) -> np.ndarray:
    """Overlay IR edges (green) on top of the RGB image."""
    g = normalize_gray(ir_gray_aligned)
    edges = cv2.Canny(g, 60, 160)
    out = color_bgr.copy()
    out[edges > 0] = (0, 255, 0)
    return out


def side_by_side(*imgs: np.ndarray, height: int = 480) -> np.ndarray:
    resized = []
    for im in imgs:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        h, w = im.shape[:2]
        resized.append(cv2.resize(im, (int(w * height / h), height)))
    return cv2.hconcat(resized)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 180), 1, cv2.LINE_AA)
    return out
