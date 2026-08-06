"""Monocular depth estimation with a small neural net (MiDaS) via ONNX Runtime.

A single RGB frame in -> a dense relative-depth map out. No calibration, no
cross-modal stereo matching - just a network trained on lots of images that
predicts depth from monocular cues (occlusion, perspective, familiar sizes).
This is by far the best-looking depth in this project.

Lightweight: ONNX Runtime (~50MB) + MiDaS-small (~66MB), CPU, ~30ms/frame.
The model isn't shipped in the repo; fetch it with scripts/download_model.py
(or it downloads on first use if the network allows).
"""
from __future__ import annotations

import os

import cv2
import numpy as np

MODEL_PATH = os.path.join("models", "midas_small.onnx")
MODEL_URL = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_SIZE = 256                                      # MiDaS-small input is 256x256

try:
    import onnxruntime as ort
    HAVE_ORT = True
except Exception:
    HAVE_ORT = False


def download_model(path: str = MODEL_PATH, url: str = MODEL_URL) -> str:
    import urllib.request
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


class MLDepth:
    """Lazy wrapper around the ONNX MiDaS model."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self._sess = None
        self._in = None
        self.error = None
        self.provider = None            # active execution provider once loaded
        if not HAVE_ORT:
            self.error = "onnxruntime not installed (pip install onnxruntime)"

    @staticmethod
    def ir_to_input(ir_gray: np.ndarray) -> np.ndarray:
        """Turn an IR frame into a 3-channel image the RGB-trained net accepts.
        The net keys on structure/perspective, not colour, so IR works - which
        gives depth in the dark, where the colour camera is black."""
        g = cv2.normalize(ir_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    @property
    def available(self) -> bool:
        return HAVE_ORT and os.path.exists(self.model_path)

    @property
    def device(self) -> str:
        return "GPU" if self.provider and self.provider != "CPUExecutionProvider" \
            else "CPU"

    def _ensure(self) -> bool:
        if self._sess is not None:
            return True
        if not HAVE_ORT:
            return False
        if not os.path.exists(self.model_path):
            self.error = f"model missing - run scripts/download_model.py"
            return False
        # prefer the GPU (DirectML: any Windows GPU) and fall back to CPU
        provs = ort.get_available_providers()
        order = [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider",
                             "CPUExecutionProvider") if p in provs]
        self._sess = ort.InferenceSession(self.model_path, providers=order)
        self._in = self._sess.get_inputs()[0].name
        self.provider = self._sess.get_providers()[0]
        return True

    def infer(self, color_bgr: np.ndarray) -> np.ndarray | None:
        """Return a float32 relative-depth map at the input resolution
        (larger = nearer), or None if the model can't run."""
        if not self._ensure():
            return None
        h, w = color_bgr.shape[:2]
        x = cv2.cvtColor(cv2.resize(color_bgr, (_SIZE, _SIZE)), cv2.COLOR_BGR2RGB)
        x = x.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        x = np.transpose(x, (2, 0, 1))[None]
        out = self._sess.run(None, {self._in: x})[0]
        d = out[0] if out.ndim == 3 else out
        return cv2.resize(d, (w, h))

    def depth_map(self, color_bgr: np.ndarray,
                  colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray | None:
        d = self.infer(color_bgr)
        if d is None:
            return None
        dn = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(dn, colormap)
