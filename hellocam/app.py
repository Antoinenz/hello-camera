"""Interactive multi-mode viewer for the Windows Hello RGB + IR cameras."""
from __future__ import annotations

import os
import time
from datetime import datetime

import cv2
import numpy as np

from .capture import HelloCameras
from . import processing as P

MODES = [
    ("ir",        "IR (raw grayscale)"),
    ("rgb",       "RGB webcam"),
    ("side",      "Side by side: RGB | IR"),
    ("fuse",      "RGB + IR fusion (overlay)"),
    ("edgefuse",  "RGB + IR edges"),
    ("proximity", "IR proximity map (NOT true depth)"),
    ("iredges",   "IR edges"),
]

HELP = [
    "1-7 switch mode   |   s snapshot   |   v toggle recording",
    "arrows move IR overlay   [ ] scale IR   - + overlay alpha   h hide help   ESC quit",
]


class ViewerApp:
    def __init__(self, out_dir: str = "captures"):
        self.cams = HelloCameras(color=True, ir=True)
        self.mode = 0
        self.alpha = 0.5
        self.aligner = P.Aligner()
        self.show_help = True
        self.out_dir = out_dir
        self.writer = None
        self.win = "HelloCam - RGB + IR"

    # ------------------------------------------------------------------
    def run(self):
        print("Opening cameras (this powers on the IR emitter)...", flush=True)
        self.cams.open()
        print(f"  color: {self.cams.color_size}   ir: {self.cams.ir_size}", flush=True)
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, 960, 640)

        last = time.time()
        fps = 0.0
        try:
            while True:
                color = self.cams.read_color()
                ir = self.cams.read_ir()
                frame = self.render(color, ir)
                if frame is None:
                    if cv2.waitKey(10) == 27:
                        break
                    continue

                now = time.time()
                dt = now - last
                last = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1 / dt)

                frame = P.label(frame, f"[{MODES[self.mode][0]}] {MODES[self.mode][1]}"
                                       f"   {fps:4.1f} fps")
                if self.show_help:
                    frame = self._draw_help(frame)
                if self.writer is not None:
                    self._write(frame)
                    cv2.circle(frame, (frame.shape[1] - 18, 40), 7, (0, 0, 255), -1)

                cv2.imshow(self.win, frame)
                if not self._handle_key(cv2.waitKey(1) & 0xFF, frame):
                    break
                if cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            if self.writer is not None:
                self.writer.release()
            self.cams.close()
            cv2.destroyAllWindows()
            print("Closed. IR emitter off.", flush=True)

    # ------------------------------------------------------------------
    def render(self, color, ir):
        name = MODES[self.mode][0]
        if name == "ir":
            return P.ir_view(ir) if ir is not None else None
        if name == "iredges":
            return P.ir_edges(ir) if ir is not None else None
        if name == "proximity":
            return P.proximity_map(ir) if ir is not None else None
        if name == "rgb":
            return color
        if name == "side":
            if color is None or ir is None:
                return None
            return P.side_by_side(color, P.proximity_map(ir))
        # fusion modes need both
        if color is None or ir is None:
            return color if color is not None else (
                P.ir_view(ir) if ir is not None else None)
        aligned = self.aligner.warp_ir_to_color(ir, color.shape)
        if name == "fuse":
            return P.fuse(color, aligned, alpha=self.alpha)
        if name == "edgefuse":
            return P.edge_fuse(color, aligned)
        return color

    # ------------------------------------------------------------------
    def _handle_key(self, key, frame) -> bool:
        if key in (27,):            # ESC
            return False
        if key == 255:
            return True
        if ord("1") <= key <= ord("7"):
            self.mode = key - ord("1")
        elif key == ord("h"):
            self.show_help = not self.show_help
        elif key == ord("s"):
            self._snapshot(frame)
        elif key == ord("v"):
            self._toggle_record(frame)
        elif key == ord("+") or key == ord("="):
            self.alpha = min(1.0, self.alpha + 0.05)
        elif key == ord("-") or key == ord("_"):
            self.alpha = max(0.0, self.alpha - 0.05)
        elif key == ord("["):
            self.aligner.nudge(dscale=-0.02)
        elif key == ord("]"):
            self.aligner.nudge(dscale=+0.02)
        elif key == 81:  # left
            self.aligner.nudge(dx=-4)
        elif key == 83:  # right
            self.aligner.nudge(dx=+4)
        elif key == 82:  # up
            self.aligner.nudge(dy=-4)
        elif key == 84:  # down
            self.aligner.nudge(dy=+4)
        return True

    def _draw_help(self, frame):
        y = frame.shape[0] - 40
        cv2.rectangle(frame, (0, y - 6), (frame.shape[1], frame.shape[0]),
                      (0, 0, 0), -1)
        for i, line in enumerate(HELP):
            cv2.putText(frame, line, (8, y + 12 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1,
                        cv2.LINE_AA)
        return frame

    # ------------------------------------------------------------------
    def _snapshot(self, frame):
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir,
                            f"snap_{MODES[self.mode][0]}_{_stamp()}.png")
        cv2.imwrite(path, frame)
        print(f"saved {path}", flush=True)

    def _toggle_record(self, frame):
        if self.writer is None:
            os.makedirs(self.out_dir, exist_ok=True)
            path = os.path.join(self.out_dir, f"rec_{_stamp()}.mp4")
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
            self._rec_size = (w, h)
            print(f"recording -> {path}", flush=True)
        else:
            self.writer.release()
            self.writer = None
            print("recording stopped", flush=True)

    def _write(self, frame):
        if frame.shape[1::-1] != self._rec_size:
            frame = cv2.resize(frame, self._rec_size)
        self.writer.write(frame)


def _stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def run():
    ViewerApp().run()
