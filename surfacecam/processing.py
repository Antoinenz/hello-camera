"""Frame processing: IR visualisation, RGB+IR fusion, and a proximity map.

Note on "depth": the Surface Hello camera has no depth sensor, so none of
this is metric depth. `proximity_map` uses IR flood intensity as a rough
closeness proxy (brighter ~ nearer, because IR illumination falls off with
distance). It is clearly labelled as such in the UI.
"""
from __future__ import annotations

import json
import os

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


def _clahe(gray):
    return cv2.createCLAHE(2.0, (8, 8)).apply(
        cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))


def depth_confidence(color_bgr: np.ndarray) -> float:
    """0..1 estimate of whether stereo depth can work: it needs a RGB image
    with real contrast. In low light the color sensor goes near-black (the IR
    emitter still lights the scene, but stereo needs BOTH views), so contrast
    collapses and this returns ~0 - the cue for 'auto' to fall back to the
    IR-intensity proximity map."""
    g = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    std = float(g.std())          # dark scene ~1, well-lit ~50-80
    return float(np.clip((std - 10.0) / 30.0, 0.0, 1.0))


def _guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float):
    """Edge-aware smoothing of `src` steered by `guide` (both float32, same
    size). Makes the depth follow the guide image's edges (He et al. 2010)."""
    g = guide.astype(np.float32)
    p = src.astype(np.float32)
    box = (r, r)
    mean_g = cv2.boxFilter(g, -1, box)
    mean_p = cv2.boxFilter(p, -1, box)
    mean_gp = cv2.boxFilter(g * p, -1, box)
    cov_gp = mean_gp - mean_g * mean_p
    var_g = cv2.boxFilter(g * g, -1, box) - mean_g * mean_g
    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g
    mean_a = cv2.boxFilter(a, -1, box)
    mean_b = cv2.boxFilter(b, -1, box)
    return mean_a * g + mean_b


def _grad(gray):
    """Gradient-magnitude image: edges match across IR<->visible even though
    raw intensities don't, so flow tracks real structure instead of mush."""
    g = _clahe(gray)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    m = cv2.magnitude(gx, gy)
    return cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _disparity_field(color_bgr, ir_gray_aligned, work_w):
    """Shared core: estimate a signed disparity field (relative depth) from the
    RGB<->IR parallax. Returns (disp float32, mask bool, igray, ws) at working
    resolution. disp is guided-filter sharpened; larger = one end of depth."""
    ch, cw = color_bgr.shape[:2]
    s = work_w / cw
    ws = (work_w, max(1, int(ch * s)))
    cg = cv2.resize(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY), ws)
    igray = cv2.resize(ir_gray_aligned, ws)
    mask = cv2.resize(ir_gray_aligned, ws, interpolation=cv2.INTER_NEAREST) > 0

    # flow on gradient images (cross-modal robust), smaller window = sharper
    flow = cv2.calcOpticalFlowFarneback(
        _grad(cg), _grad(igray), None, pyr_scale=0.5, levels=4, winsize=13,
        iterations=4, poly_n=5, poly_sigma=1.1, flags=0)
    fx, fy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)

    gx = cv2.Sobel(cg, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(cg, cv2.CV_32F, 0, 1, 3)
    tex = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 3)
    if mask.any():
        good = mask & (tex > np.percentile(tex[mask], 60)) & \
            (mag < np.percentile(mag[mask], 98))
        vecs = np.stack([fx[good], fy[good]], 1)
        vecs = vecs[np.linalg.norm(vecs, axis=1) > 0.3]
    else:
        vecs = np.empty((0, 2), np.float32)
    if len(vecs) > 50:
        vv = vecs - vecs.mean(0)
        _, _, vt = np.linalg.svd(vv, full_matrices=False)
        axis = vt[0]
        if axis[0] < 0 or (axis[0] == 0 and axis[1] < 0):
            axis = -axis                          # stable sign (fixed baseline)
        disp = fx * axis[0] + fy * axis[1]        # signed projection = disparity
    else:
        disp = mag

    disp = disp.astype(np.float32)
    guide = _clahe(igray).astype(np.float32) / 255.0
    disp = _guided_filter(guide, disp, r=12, eps=1e-3)
    return disp, mask, igray, ws


def _colorize(disp, mask, cw, ch, colormap, pct=(5, 95)):
    if mask.any():
        lo, hi = np.percentile(disp[mask], pct)
    else:
        lo, hi = 0.0, 1.0
    norm = np.clip((disp - lo) / max(hi - lo, 1e-3), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)
    heat[~mask] = 0
    return cv2.resize(heat, (cw, ch), interpolation=cv2.INTER_NEAREST)


def stereo_depth(color_bgr: np.ndarray, ir_gray_aligned: np.ndarray,
                 work_w: int = 400, colormap: int = cv2.COLORMAP_TURBO,
                 return_conf: bool = False):
    """Relative depth from RGB<->IR parallax (dense flow on gradient images,
    projected onto the baseline axis, guided-filter sharpened). Relative, not
    metric. Returns a BGR depth image (and a 0..1 confidence if return_conf)."""
    ch, cw = color_bgr.shape[:2]
    disp, mask, _, _ = _disparity_field(color_bgr, ir_gray_aligned, work_w)
    heat = _colorize(disp, mask, cw, ch, colormap)
    if return_conf:
        return heat, depth_confidence(color_bgr)
    return heat


def portrait_depth(color_bgr: np.ndarray, ir_gray_aligned: np.ndarray,
                   work_w: int = 400, colormap: int = cv2.COLORMAP_TURBO):
    """Subject-aware depth for portraits: the IR image already segments a close
    subject cleanly (bright, from IR falloff), so use that as the subject mask
    and colour it by the stereo disparity (near/far *within* the subject),
    while pushing the background to the far end. Cleaner-looking than raw stereo
    for people/objects against a wall."""
    ch, cw = color_bgr.shape[:2]
    disp, mask, igray, ws = _disparity_field(color_bgr, ir_gray_aligned, work_w)

    # subject = bright IR region (Otsu on the in-FOV pixels)
    ig8 = cv2.normalize(igray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ig8 = cv2.GaussianBlur(ig8, (5, 5), 0)
    thr, _ = cv2.threshold(ig8[mask] if mask.any() else ig8, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    subject = mask & (ig8 >= thr)
    subject = cv2.morphologyEx(subject.astype(np.uint8), cv2.MORPH_OPEN,
                               np.ones((5, 5), np.uint8)).astype(bool)

    if subject.sum() > 200:
        lo, hi = np.percentile(disp[subject], (5, 95))
    else:
        lo, hi = float(disp[mask].min()) if mask.any() else 0.0, 1.0
    norm = np.clip((disp - lo) / max(hi - lo, 1e-3), 0, 1)
    norm[~subject] = 0.0                       # background -> far end
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)
    heat[~mask] = 0
    # dim the (non-subject) background so the subject pops
    bg = mask & ~subject
    heat[bg] = (heat[bg] * 0.35).astype(np.uint8)
    return cv2.resize(heat, (cw, ch), interpolation=cv2.INTER_NEAREST)


class Aligner:
    """Maps the IR frame onto the color frame with a 2x3 affine transform.

    The two sensors sit side by side with different FOVs, so the IR view must
    be scaled/rotated/translated to line up with the color view. The same
    matrix `M` is driven either manually (nudge) or automatically by matching
    shared features between the two images (auto_align).
    """

    def __init__(self):
        self.M = None                 # 2x3 float32, maps IR px -> color px
        self._color_shape = None
        self.last_info = None         # stats from the most recent auto_align

    # -- geometry ----------------------------------------------------------
    def reset(self, ir_shape, color_shape):
        ih, iw = ir_shape[:2]
        ch, cw = color_shape[:2]
        s = ch / ih                   # fit IR height to color height
        tx = (cw - iw * s) / 2.0
        ty = (ch - ih * s) / 2.0
        self.M = np.array([[s, 0, tx], [0, s, ty]], np.float32)
        self._color_shape = color_shape

    def _ensure(self, ir_shape, color_shape):
        self._color_shape = color_shape
        if self.M is None:
            self.reset(ir_shape, color_shape)

    def warp_ir_to_color(self, ir_gray: np.ndarray, color_shape) -> np.ndarray:
        ch, cw = color_shape[:2]
        self._ensure(ir_gray.shape, color_shape)
        return cv2.warpAffine(ir_gray, self.M, (cw, ch), flags=cv2.INTER_LINEAR)

    def nudge(self, dx=0, dy=0, dscale=0.0):
        if self.M is None:
            return
        if dx or dy:
            self.M[0, 2] += dx
            self.M[1, 2] += dy
        if dscale and self._color_shape is not None:
            s = 1.0 + dscale
            ch, cw = self._color_shape[:2]
            c = np.array([cw / 2.0, ch / 2.0], np.float32)
            old_t = self.M[:, 2].copy()
            self.M[:, :2] *= s
            self.M[:, 2] = s * old_t + (1.0 - s) * c

    # -- automatic feature-based alignment --------------------------------
    @staticmethod
    def _prep(gray):
        """Equalize contrast so near-IR and visible gradients are comparable."""
        g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.createCLAHE(2.0, (8, 8)).apply(g)

    @staticmethod
    def _structure(gray):
        """Gradient-magnitude image. NIR and visible intensities don't match,
        but their *edges* do, so gradient domain is what ECC/scoring should use
        for cross-modal alignment."""
        g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _correspondences(self, ir_gray, color_bgr, ratio=0.8):
        """Return matched (src_ir, dst_color) point lists for one frame pair."""
        g_ir = self._prep(ir_gray)
        g_col = self._prep(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY))
        orb = cv2.ORB_create(2000, scaleFactor=1.2, nlevels=8)
        k1, d1 = orb.detectAndCompute(g_ir, None)
        k2, d2 = orb.detectAndCompute(g_col, None)
        if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
            return [], []
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        src, dst = [], []
        for pair in bf.knnMatch(d1, d2, k=2):
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                src.append(k1[pair[0].queryIdx].pt)
                dst.append(k2[pair[0].trainIdx].pt)
        return src, dst

    def _solve(self, src, dst, min_matches, min_inliers) -> dict:
        if len(src) < min_matches:
            return {"ok": False, "matches": len(src), "inliers": 0,
                    "reason": "too few matches"}
        s = np.float32(src).reshape(-1, 1, 2)
        d = np.float32(dst).reshape(-1, 1, 2)
        M, inl = cv2.estimateAffinePartial2D(
            s, d, method=cv2.RANSAC, ransacReprojThreshold=6.0,
            maxIters=5000, confidence=0.999)
        inliers = int(inl.sum()) if inl is not None else 0
        if M is None or inliers < min_inliers:
            return {"ok": False, "matches": len(src), "inliers": inliers,
                    "reason": "no consensus"}
        scale = float(np.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])))
        if not (0.3 < scale < 5.0):
            return {"ok": False, "matches": len(src), "inliers": inliers,
                    "scale": scale, "reason": "implausible scale"}
        self.M = M.astype(np.float32)
        return {"ok": True, "matches": len(src), "inliers": inliers, "scale": scale}

    def auto_align(self, ir_gray: np.ndarray, color_bgr: np.ndarray) -> dict:
        """Single-frame align (used by live mode). Marginal but cheap."""
        self._ensure(ir_gray.shape, color_bgr.shape)
        src, dst = self._correspondences(ir_gray, color_bgr)
        info = self._solve(src, dst, min_matches=12, min_inliers=8)
        self.last_info = info
        return info

    def auto_align_pooled(self, pairs) -> dict:
        """Pool correspondences across several frame pairs, then solve once.

        The IR<->RGB geometry is constant, so true matches accumulate across
        frames while wrong matches stay random - RANSAC locks on much more
        reliably than it can from any single (cross-modal, noisy) frame.
        """
        if not pairs:
            return {"ok": False, "matches": 0, "inliers": 0, "reason": "no frames"}
        self._ensure(pairs[0][0].shape, pairs[0][1].shape)
        src, dst = [], []
        for ir, color in pairs:
            s, d = self._correspondences(ir, color)
            src.extend(s)
            dst.extend(d)
        info = self._solve(src, dst, min_matches=20, min_inliers=15)
        info["frames"] = len(pairs)
        self.last_info = info
        return info

    # -- photometric refinement (ECC) -------------------------------------
    def alignment_score(self, ir_gray, color_bgr, M=None) -> float:
        """Normalised cross-correlation between the warped IR and the color
        image over the covered region. 1.0 = perfect, ~0 = unrelated."""
        M = self.M if M is None else M
        if M is None:
            return -1.0
        ch, cw = color_bgr.shape[:2]
        warped = cv2.warpAffine(ir_gray, M, (cw, ch))
        tmpl = self._structure(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY))
        inp = self._structure(warped)
        mask = warped > 0
        if int(mask.sum()) < 2000:
            return -1.0
        a = inp[mask].astype(np.float32)
        b = tmpl[mask].astype(np.float32)
        a -= a.mean()
        b -= b.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(a @ b / denom) if denom > 0 else -1.0

    def refine_ecc(self, ir_gray, color_bgr, motion="affine") -> dict:
        """Polish the current transform by directly maximising image overlap
        (ECC). Starts from the existing M (manual or auto), so it needs a
        roughly-correct starting point. Only keeps the result if it measurably
        improves the alignment score, so it can never make things worse."""
        self._ensure(ir_gray.shape, color_bgr.shape)
        base = self.alignment_score(ir_gray, color_bgr)
        ch, cw = color_bgr.shape[:2]
        tmpl = self._structure(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY))
        warped = cv2.warpAffine(ir_gray, self.M, (cw, ch))
        inp = self._structure(warped)
        mt = cv2.MOTION_AFFINE if motion == "affine" else cv2.MOTION_EUCLIDEAN
        warp = np.eye(2, 3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
        try:
            _, warp = cv2.findTransformECC(tmpl, inp, warp, mt, crit, None, 5)
        except cv2.error:
            return {"ok": False, "reason": "ECC diverged", "score": base}
        # ECC gives a color-space correction; direction is ambiguous, so try
        # both and keep whichever actually improves the measured score.
        A = np.vstack([self.M, [0, 0, 1]])
        W = np.vstack([warp, [0, 0, 1]])
        best_m, best = None, base
        for cand in (W @ A, np.linalg.inv(W) @ A):
            m = cand[:2].astype(np.float32)
            sc = self.alignment_score(ir_gray, color_bgr, m)
            if sc > best:
                best, best_m = sc, m
        if best_m is None:
            return {"ok": False, "reason": "no improvement", "score": base}
        self.M = best_m
        return {"ok": True, "score": best, "was": base}

    # -- persistence (the sensor geometry is fixed, so save it once) -------
    def save(self, path, ir_shape, color_shape) -> bool:
        if self.M is None:
            return False
        data = {
            "M": self.M.tolist(),
            "ir": [int(ir_shape[0]), int(ir_shape[1])],
            "color": [int(color_shape[0]), int(color_shape[1])],
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except OSError:
            return False

    def load(self, path, ir_shape, color_shape) -> bool:
        """Load a saved transform only if it matches the current resolutions."""
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, ValueError):
            return False
        if (d.get("ir") != [int(ir_shape[0]), int(ir_shape[1])] or
                d.get("color") != [int(color_shape[0]), int(color_shape[1])]):
            return False
        self.M = np.array(d["M"], np.float32)
        self._color_shape = color_shape
        return True


def fuse(color_bgr: np.ndarray, ir_gray_aligned: np.ndarray,
         alpha: float = 0.5, tint: int = cv2.COLORMAP_BONE) -> np.ndarray:
    """Blend aligned IR (as a cyan-ish overlay) over the RGB image.

    Uses OpenCV's SIMD uint8 addWeighted instead of full-frame float math,
    then restores the original color where the IR overlay has no coverage
    (via cv2.copyTo, not numpy boolean indexing) so the black warp borders
    don't darken the edges. ~30x faster than the float version.
    """
    g = normalize_gray(ir_gray_aligned)
    ir_color = cv2.applyColorMap(g, tint)
    blended = cv2.addWeighted(color_bgr, 1.0 - alpha, ir_color, alpha, 0.0)
    # 255 where the warped IR has no signal -> copy the original color back
    uncovered = cv2.threshold(ir_gray_aligned, 0, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.copyTo(color_bgr, uncovered, blended)
    return blended


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


def apply_aspect(frame: np.ndarray, out_w: int, out_h: int, mode: str = "fit",
                 bg: int = 0) -> np.ndarray:
    """Resize `frame` into an out_h x out_w canvas without distortion.

    mode:
      "fit"     - scale to fit inside, preserve aspect, letterbox (no crop)
      "fill"    - scale to cover, preserve aspect, center-crop the overflow
      "stretch" - scale to exactly out_w x out_h (distorts aspect)
    """
    if out_w <= 0 or out_h <= 0:
        return frame
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = frame.shape[:2]

    if mode == "stretch":
        return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    if mode == "fill":
        s = max(out_w / w, out_h / h)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x0 = (nw - out_w) // 2
        y0 = (nh - out_h) // 2
        return resized[y0:y0 + out_h, x0:x0 + out_w]

    # default: "fit" (letterbox)
    s = min(out_w / w, out_h / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((out_h, out_w, 3), bg, np.uint8)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 180), 1, cv2.LINE_AA)
    return out
