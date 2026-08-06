"""Native-windowed viewer (Tkinter) for the Windows Hello RGB + IR cameras.

Uses a real Windows menu bar for all controls:
  View    -> Mode (IR / RGB / fusion / ...) and Aspect (Fit / Fill / Stretch)
  Capture -> Snapshot, Record, open output folder
  Overlay -> live IR<->RGB alignment for the fusion modes
  Help    -> controls & about

The camera/processing core (hellocam.capture / .processing) is reused
unchanged; only the front-end differs from the OpenCV viewer.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from .capture import HelloCameras
from . import processing as P
from .mldepth import (MLDepth, MODELS as ML_MODELS, DEFAULT_MODEL as ML_DEFAULT,
                      download_model as ml_download)

try:
    import pyvirtualcam
    HAVE_VCAM = True
except Exception:
    HAVE_VCAM = False

VCAM_W, VCAM_H, VCAM_FPS = 1280, 720, 30


def _asset(name: str) -> str:
    """Path to a bundled asset, working both from source and from a frozen
    PyInstaller build (where data is unpacked under sys._MEIPASS)."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(base, "hellocam", "assets", name)


def _set_app_id():
    """Give the process an explicit AppUserModelID (Windows only). Without this,
    running via python.exe makes the taskbar show Python's icon; setting our own
    ID detaches the window so Windows uses the icon we set instead. No-op if the
    call isn't available."""
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "HelloCam.WindowsHello.RGB.IR")
    except Exception:
        pass

MODES = [
    ("ir",        "IR (raw grayscale)"),
    ("rgb",       "RGB webcam"),
    ("side",      "Side by side: RGB | IR"),
    ("fuse",      "RGB + IR fusion (overlay)"),
    ("edgefuse",  "RGB + IR edges"),
    ("proximity", "Depth map (IR proximity / stereo)"),
    ("iredges",   "IR edges"),
]
ASPECTS = ["fit", "fill", "stretch"]

# which cameras each mode needs -> unused ones get powered down
MODE_SOURCES = {
    "ir": (False, True), "iredges": (False, True), "proximity": (False, True),
    "rgb": (True, False),
    "side": (True, True), "fuse": (True, True), "edgefuse": (True, True),
}

FREEZE_OFF_SECS = 10.0      # frozen this long -> power cameras down
MINIMIZE_OFF_SECS = 3.0     # minimised this long -> power cameras down
UNUSED_DISABLE_SECS = 4.0   # a mode's unused camera off after this long


class ViewerGUI:
    def __init__(self, out_dir: str = "captures", calib_path: str = "calibration.json"):
        self.out_dir = out_dir
        self.calib_path = calib_path
        self._calib_tried = False
        self.stereo_calib = P.load_stereo_calib("captures/stereo_calib.npz")
        self.mldepth = MLDepth(ML_DEFAULT)
        self.cams = HelloCameras(color=True, ir=True)
        self.aligner = P.Aligner()
        self.alpha = 0.5
        self.writer = None
        self._rec_size = None
        self._fps = 0.0
        self._last = time.time()
        self._ir_fps = 0.0              # rate of new illuminated (held) IR frames
        self._ir_last_cnt = 0
        self._ir_last_t = time.time()
        self._frame_i = 0
        self._last_ver = -1
        self._save_after_id = None      # debounce for auto-saving manual nudges
        self._photo = None  # keep a ref so Tk doesn't GC the image
        # power management state
        self.frozen = False
        self._pause_since = None        # when the current pause began
        self._cams_off = False          # cameras suspended for power saving
        self._last_frame = None         # last rendered frame (for freeze badge)
        self._color_unused_since = None
        self._ir_unused_since = None
        self._paused = False
        self._last_power_t = 0.0        # throttle power checks (root.state is slow)

        _set_app_id()                   # so the taskbar uses our icon, not python's
        self.root = tk.Tk()
        self.root.title("HelloCam - RGB + IR")
        self.root.geometry("980x680")
        self.root.minsize(320, 240)
        self.root.configure(bg="black")
        self._set_window_icon()

        # tk vars driving the menus
        self.mode_var = tk.StringVar(value=MODES[0][0])
        self.aspect_var = tk.StringVar(value="fit")
        self.statusbar_var = tk.BooleanVar(value=True)
        self.fps_var = tk.BooleanVar(value=True)
        self.mirror_var = tk.BooleanVar(value=False)
        self.autoalign_live_var = tk.BooleanVar(value=False)
        self.ir_source_var = tk.StringVar(value=self.cams.ir_mode)
        self.autopause_min_var = tk.BooleanVar(value=True)
        self.depth_method_var = tk.StringVar(value="auto")
        self.ml_model_var = tk.StringVar(value=ML_DEFAULT)
        self.depth_alpha = 1.0          # depth-over-original blend (1=depth only)
        self.vcam_var = tk.BooleanVar(value=False)
        self.vcam = None

        self._build_menu()
        self._build_widgets()
        self._bind_keys()

    def _set_window_icon(self):
        """Set the title-bar / taskbar icon. Never fatal if the asset is
        missing (e.g. running from an odd cwd)."""
        try:
            self.root.iconbitmap(default=_asset("icon.ico"))
        except Exception:
            try:                                    # fallback via PIL/PhotoImage
                self._icon_img = ImageTk.PhotoImage(
                    Image.open(_asset("icon.ico")))
                self.root.iconphoto(True, self._icon_img)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _build_menu(self):
        menubar = tk.Menu(self.root)

        # View
        view = tk.Menu(menubar, tearoff=0)
        mode_menu = tk.Menu(view, tearoff=0)
        for i, (key, desc) in enumerate(MODES):
            mode_menu.add_radiobutton(
                label=f"{desc}", value=key, variable=self.mode_var,
                accelerator=str(i + 1), command=self._on_mode_change)
        view.add_cascade(label="Mode", menu=mode_menu)

        aspect_menu = tk.Menu(view, tearoff=0)
        labels = {"fit": "Fit  (letterbox, no crop)",
                  "fill": "Fill  (crop to fill)",
                  "stretch": "Stretch  (ignore aspect)"}
        accel = {"fit": "F", "fill": "L", "stretch": "T"}
        for a in ASPECTS:
            aspect_menu.add_radiobutton(
                label=labels[a], value=a, variable=self.aspect_var,
                accelerator=accel[a])
        view.add_cascade(label="Aspect ratio", menu=aspect_menu)

        view.add_separator()
        view.add_checkbutton(label="Show FPS overlay", variable=self.fps_var)
        view.add_checkbutton(label="Mirror (flip horizontally)", accelerator="M",
                             variable=self.mirror_var)

        ir_menu = tk.Menu(view, tearoff=0)
        ir_menu.add_radiobutton(label="Active (emitter on, anti-flicker)",
                                value="active", variable=self.ir_source_var,
                                command=self._set_ir_source)
        ir_menu.add_radiobutton(label="Passive (ambient IR, no emitter)",
                                value="passive", variable=self.ir_source_var,
                                command=self._set_ir_source)
        ir_menu.add_radiobutton(label="Raw (unfiltered strobe)",
                                value="raw", variable=self.ir_source_var,
                                command=self._set_ir_source)
        view.add_cascade(label="IR source", menu=ir_menu)

        view.add_checkbutton(label="Show status bar", variable=self.statusbar_var,
                             command=self._toggle_statusbar)
        view.add_command(label="Reset IR alignment", command=self._reset_align)
        view.add_separator()
        view.add_command(label="Freeze / resume", accelerator="Space",
                         command=self._toggle_freeze)
        view.add_checkbutton(label="Auto-pause when minimized",
                             variable=self.autopause_min_var)
        view.add_separator()
        view.add_command(label="Exit", accelerator="Esc", command=self._on_close)
        menubar.add_cascade(label="View", menu=view)

        # Depth  (the "Depth map" mode's settings)
        depth = tk.Menu(menubar, tearoff=0)
        method_menu = tk.Menu(depth, tearoff=0)
        method_menu.add_radiobutton(
            label="ML monocular (neural net)" if self.mldepth.available
            else "ML monocular (run download_model.py first)",
            value="ml", variable=self.depth_method_var)
        method_menu.add_radiobutton(label="Auto (ML if available, else stereo/IR)",
                                    value="auto", variable=self.depth_method_var)
        method_menu.add_radiobutton(label="Stereo (RGB<->IR parallax)",
                                    value="stereo", variable=self.depth_method_var)
        method_menu.add_radiobutton(label="Portrait (subject cutout + stereo)",
                                    value="portrait", variable=self.depth_method_var)
        method_menu.add_radiobutton(
            label="Calibrated (checkerboard stereo)" if self.stereo_calib
            else "Calibrated (run stereo_calibrate.py first)",
            value="calibrated", variable=self.depth_method_var)
        method_menu.add_radiobutton(label="Proximity (IR intensity)",
                                    value="proximity", variable=self.depth_method_var)
        depth.add_cascade(label="Method", menu=method_menu)

        mlm = tk.Menu(depth, tearoff=0)
        for key, spec in ML_MODELS.items():
            mlm.add_radiobutton(label=spec["label"], value=key,
                                variable=self.ml_model_var, command=self._set_ml_model)
        depth.add_cascade(label="ML model", menu=mlm)
        depth.add_command(label="Download selected ML model (~100MB)",
                          command=self._download_ml_model)

        depth.add_separator()
        depth.add_command(label="More depth / less original", accelerator="+",
                          command=lambda: self._set_depth_alpha(0.1))
        depth.add_command(label="Less depth / more original", accelerator="-",
                          command=lambda: self._set_depth_alpha(-0.1))
        depth.add_command(label="Depth only",
                          command=lambda: self._set_depth_alpha(1.0, True))
        depth.add_command(label="Original only",
                          command=lambda: self._set_depth_alpha(0.0, True))
        menubar.add_cascade(label="Depth", menu=depth)

        # Capture
        cap = tk.Menu(menubar, tearoff=0)
        cap.add_command(label="Snapshot", accelerator="Ctrl+S", command=self._snapshot)
        self._rec_label = tk.StringVar(value="Start recording")
        cap.add_command(label="Start recording", accelerator="Ctrl+R",
                        command=self._toggle_record)
        self._cap_menu = cap
        cap.add_separator()
        state = "normal" if HAVE_VCAM else "disabled"
        cap.add_checkbutton(label="Send to virtual camera", variable=self.vcam_var,
                            command=self._toggle_vcam, state=state)
        cap.add_command(label="Open captures folder", command=self._open_folder)
        menubar.add_cascade(label="Capture", menu=cap)

        # Overlay (fusion alignment)
        ov = tk.Menu(menubar, tearoff=0)
        ov.add_command(label="Auto-align (find)", accelerator="A",
                       command=self._auto_align_once)
        ov.add_command(label="Refine alignment (ECC)", accelerator="E",
                       command=self._refine_once)
        ov.add_checkbutton(label="Auto-align (live)", variable=self.autoalign_live_var)
        ov.add_separator()
        ov.add_command(label="Save alignment", command=self._save_current_calib)
        ov.add_separator()
        ov.add_command(label="Nudge left", accelerator="Left",
                       command=lambda: self._nudge(dx=-4))
        ov.add_command(label="Nudge right", accelerator="Right",
                       command=lambda: self._nudge(dx=4))
        ov.add_command(label="Nudge up", accelerator="Up",
                       command=lambda: self._nudge(dy=-4))
        ov.add_command(label="Nudge down", accelerator="Down",
                       command=lambda: self._nudge(dy=4))
        ov.add_separator()
        ov.add_command(label="Zoom IR in", accelerator="]",
                       command=lambda: self._nudge(dscale=0.02))
        ov.add_command(label="Zoom IR out", accelerator="[",
                       command=lambda: self._nudge(dscale=-0.02))
        ov.add_separator()
        ov.add_command(label="Overlay more opaque", accelerator="+",
                       command=lambda: self._set_alpha(0.05))
        ov.add_command(label="Overlay more transparent", accelerator="-",
                       command=lambda: self._set_alpha(-0.05))
        ov.add_command(label="Reset alignment", command=self._reset_align)
        menubar.add_cascade(label="Overlay", menu=ov)

        # Help
        hlp = tk.Menu(menubar, tearoff=0)
        hlp.add_command(label="Controls...", command=self._show_controls)
        hlp.add_command(label="About...", command=self._show_about)
        menubar.add_cascade(label="Help", menu=hlp)

        self.root.config(menu=menubar)

    def _build_widgets(self):
        self.canvas = tk.Label(self.root, bg="black", bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.status = tk.Label(self.root, anchor="w", bg="#101010", fg="#7fffd4",
                               font=("Consolas", 9), padx=6)
        self.status.pack(fill="x", side="bottom")

    def _bind_keys(self):
        r = self.root
        for i in range(len(MODES)):
            r.bind(str(i + 1), lambda e, k=MODES[i][0]: self._set_mode(k))
        r.bind("f", lambda e: self.aspect_var.set("fit"))
        r.bind("l", lambda e: self.aspect_var.set("fill"))
        r.bind("t", lambda e: self.aspect_var.set("stretch"))
        r.bind("m", lambda e: self.mirror_var.set(not self.mirror_var.get()))
        r.bind("M", lambda e: self.mirror_var.set(not self.mirror_var.get()))
        r.bind("<Left>", lambda e: self._nudge(dx=-4))
        r.bind("<Right>", lambda e: self._nudge(dx=4))
        r.bind("<Up>", lambda e: self._nudge(dy=-4))
        r.bind("<Down>", lambda e: self._nudge(dy=4))
        r.bind("[", lambda e: self._nudge(dscale=-0.02))
        r.bind("]", lambda e: self._nudge(dscale=0.02))
        r.bind("<plus>", lambda e: self._set_alpha(0.05))
        r.bind("<equal>", lambda e: self._set_alpha(0.05))
        r.bind("<minus>", lambda e: self._set_alpha(-0.05))
        r.bind("<Control-s>", lambda e: self._snapshot())
        r.bind("<Control-r>", lambda e: self._toggle_record())
        r.bind("a", lambda e: self._auto_align_once())
        r.bind("A", lambda e: self._auto_align_once())
        r.bind("e", lambda e: self._refine_once())
        r.bind("E", lambda e: self._refine_once())
        r.bind("<space>", lambda e: self._toggle_freeze())
        r.bind("<Escape>", lambda e: self._on_close())
        r.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    def run(self):
        print("Opening cameras (this powers on the IR emitter)...", flush=True)
        self.cams.open()
        self.cams.start_pump()   # capture on a background thread at full sensor rate
        print(f"  color: {self.cams.color_size}   ir: {self.cams.ir_size}", flush=True)
        self.root.after(0, self._tick)
        self.root.mainloop()

    def _tick(self):
        try:
            # power checks call root.state() (slow), so run them at ~5Hz, not
            # every tick; freeze via spacebar applies instantly via _toggle_freeze
            now = time.time()
            if now - self._last_power_t >= 0.2:
                self._last_power_t = now
                self._paused = self._manage_power()
                if not self._paused:
                    self._manage_sources()
            if self._paused:
                return self._reschedule()   # paused: skip all rendering/processing
            # render only when the pump has a new frame; otherwise just poll
            ver = self.cams.frame_ver
            if ver != self._last_ver:
                self._last_ver = ver
                self._frame_i += 1
                color = self.cams.latest_color()
                ir = self.cams.latest_ir()
                # once, on the first good frame pair, load a saved calibration
                if not self._calib_tried and color is not None and ir is not None:
                    self._calib_tried = True
                    if self.aligner.load(self.calib_path, ir.shape, color.shape):
                        self.status.config(
                            text=f"loaded calibration from {self.calib_path}")
                if (self.autoalign_live_var.get() and color is not None
                        and ir is not None and self._frame_i % 8 == 0):
                    info = self.aligner.auto_align(ir, color)
                    if info.get("ok"):
                        self._save_calib(ir, color)
                        self.status.config(
                            text=f"auto-align (live): {info['inliers']}/"
                                 f"{info['matches']} inliers, scale "
                                 f"{info['scale']:.2f}  (saved)")
                frame = self._render(color, ir)
                if frame is not None:
                    self._last_frame = frame
                    self._update_fps()
                    if self.writer is not None:
                        self._write(frame)
                    self._show(frame)
                    self._update_status(frame)
        except Exception as e:  # keep the loop alive, surface once
            self.status.config(text=f"error: {e}")
        self._reschedule()

    def _reschedule(self):
        self.root.after(2, self._tick)

    # -- power management --------------------------------------------------
    def _is_minimized(self) -> bool:
        try:
            return self.root.state() == "iconic"
        except tk.TclError:
            return False

    def _pause_request(self):
        """Return (paused, grace_secs, reason) for the current state."""
        if self.frozen:
            return True, FREEZE_OFF_SECS, "frozen"
        if self.autopause_min_var.get() and self._is_minimized():
            return True, MINIMIZE_OFF_SECS, "minimized"
        return False, 0.0, ""

    def _manage_power(self) -> bool:
        """Handle freeze / minimize pausing and camera power-down.
        Returns True when paused (caller should skip rendering)."""
        paused, grace, reason = self._pause_request()
        now = time.time()
        self._paused = paused       # set before any _show_badge so it hides FPS
        if paused:
            if self._pause_since is None:       # just entered pause
                self._pause_since = now
                if reason == "frozen" and self._last_frame is not None:
                    self._show_badge("FROZEN")
            elif not self._cams_off and now - self._pause_since > grace:
                self.cams.suspend()             # power cameras down (silently)
                self._cams_off = True
            return True
        # not paused: resume if we were
        if self._pause_since is not None:
            self._pause_since = None
            if self._cams_off:
                self.cams.resume()
                self._cams_off = False
                self.status.config(text="resumed")
            self._last_ver = -1                 # force a fresh render
        return False

    def _manage_sources(self):
        """Power down whichever camera the current mode doesn't need (after a
        short grace, so flipping modes doesn't churn the hardware)."""
        mode = self.mode_var.get()
        need_c, need_i = MODE_SOURCES.get(mode, (True, True))
        # depth sub-methods have different sensor needs
        if mode == "proximity":
            dm = self.depth_method_var.get()
            uses_ml = dm == "ml" or (dm == "auto" and self.mldepth.available)
            if uses_ml:
                need_c = True                   # ML runs on the color image...
                col = self.cams.latest_color()  # ...but on IR when it's dark
                need_i = col is None or P.depth_confidence(col) < 0.5
            elif dm != "proximity":
                need_c = True                   # stereo/portrait/calibrated need both
        now = time.time()
        self._color_unused_since = None if need_c else (
            self._color_unused_since or now)
        self._ir_unused_since = None if need_i else (
            self._ir_unused_since or now)
        want_c = need_c or (now - self._color_unused_since < UNUSED_DISABLE_SECS)
        want_i = need_i or (now - self._ir_unused_since < UNUSED_DISABLE_SECS)
        self.cams.request_sources(want_c, want_i)

    def _toggle_freeze(self):
        self.frozen = not self.frozen
        self._last_power_t = time.time()
        self._paused = self._manage_power()     # apply immediately, don't wait for poll

    def _show_badge(self, text):
        if self._last_frame is None:
            return
        f = self._last_frame.copy()
        h, w = f.shape[:2]
        cv2.rectangle(f, (0, h // 2 - 34), (w, h // 2 + 20), (0, 0, 0), -1)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(f, text, ((w - tw) // 2, h // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 255, 180), 2, cv2.LINE_AA)
        self._show(f)

    # ------------------------------------------------------------------
    def _render(self, color, ir):
        name = self.mode_var.get()
        if name == "ir":
            return P.ir_view(ir) if ir is not None else None
        if name == "iredges":
            return P.ir_edges(ir) if ir is not None else None
        if name == "proximity":
            return self._render_depth(color, ir)
        if name == "rgb":
            return color
        if name == "side":
            if color is None or ir is None:
                return None
            return P.side_by_side(color, P.proximity_map(ir))
        if color is None or ir is None:
            return color if color is not None else (
                P.ir_view(ir) if ir is not None else None)
        aligned = self.aligner.warp_ir_to_color(ir, color.shape)
        if name == "fuse":
            return P.fuse(color, aligned, alpha=self.alpha)
        if name == "edgefuse":
            return P.edge_fuse(color, aligned)
        return color

    def _render_depth(self, color, ir):
        heat, base = self._depth_and_base(color, ir)
        if heat is None:
            return None
        if self.depth_alpha >= 0.999 or base is None:
            return heat
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        if base.shape[:2] != heat.shape[:2]:
            base = cv2.resize(base, (heat.shape[1], heat.shape[0]))
        return cv2.addWeighted(base, 1.0 - self.depth_alpha,
                               heat, self.depth_alpha, 0.0)

    def _depth_and_base(self, color, ir):
        """Return (depth_colormap, original_image) so the two can be blended.
        `base` is whatever image the depth was derived from (RGB or IR)."""
        method = self.depth_method_var.get()
        # ML monocular depth uses only the color image (no IR needed); auto
        # prefers it when the model is present.
        want_ml = method == "ml" or (method == "auto" and self.mldepth.available)
        if want_ml and self.mldepth.available:
            # use RGB in good light; in the dark feed the IR frame instead so
            # depth still works (the net keys on structure, not colour)
            dark = color is None or P.depth_confidence(color) < 0.5
            if dark and ir is not None:
                base = self.mldepth.ir_to_input(ir)
                out, src = self.mldepth.depth_map(base), "IR"
            elif color is not None:
                base = color
                out, src = self.mldepth.depth_map(color), "RGB"
            else:
                out, base = None, None
            if out is not None:
                self._depth_note(f"ML {src} ({self.mldepth.device})")
                return out, base
        if want_ml and method == "ml" and not self.mldepth.available:
            self._depth_note(f"ML unavailable ({self.mldepth.error or 'no model'})"
                             " - run scripts/download_model.py")
        if want_ml:
            method = "auto"                     # fall back until model present
        if ir is None:
            return None, None
        # need a color frame for stereo/portrait; fall back to proximity without
        if method == "proximity" or color is None:
            self._depth_note("proximity")
            return P.proximity_map(ir), P.ir_view(ir)
        if method == "auto" and P.depth_confidence(color) < 0.5:
            self._depth_note("auto: proximity (too dark for stereo)")
            return P.proximity_map(ir), P.ir_view(ir)
        if method == "calibrated":
            if self.stereo_calib is not None:
                self._depth_note("calibrated stereo")
                return P.calibrated_depth(color, ir, self.stereo_calib), color
            self._depth_note("calibrated: no calibration - run stereo_calibrate.py")
            method = "stereo"                   # fall back until calibrated
        aligned = self.aligner.warp_ir_to_color(ir, color.shape)
        if method == "portrait":
            self._depth_note("portrait")
            return P.portrait_depth(color, aligned), color
        self._depth_note("stereo" if method == "stereo" else "auto: stereo")
        return P.stereo_depth(color, aligned), color

    def _depth_note(self, note):
        self._depth_active = note        # shown in the status bar (see _update_status)

    def _set_ml_model(self):
        key = self.ml_model_var.get()
        self.mldepth.set_model(key)
        if self.mldepth.available:
            self.status.config(text=f"ML model: {ML_MODELS[key]['label']} (loading...)")
        else:
            self.status.config(
                text=f"ML model '{key}' not downloaded - "
                     f"Depth > Download selected ML model")

    def _download_ml_model(self):
        """Fetch the selected model on a background thread (keeps the UI live).
        This is the exe-friendly path: no need for the download_model.py script."""
        if getattr(self, "_ml_downloading", False):
            return
        key = self.ml_model_var.get()
        if self.mldepth.available and key == self.mldepth.key:
            self.status.config(text=f"ML model already downloaded: "
                                    f"{ML_MODELS[key]['label']}")
            return
        self._ml_downloading = True
        self.status.config(text=f"downloading {ML_MODELS[key]['label']}... "
                                f"(~100MB, this can take a minute)")

        def worker():
            try:
                ml_download(key)
                msg, ok = f"downloaded {ML_MODELS[key]['label']} - ready", True
            except Exception as e:                       # network/disk error
                msg, ok = f"download failed: {e}", False
            self.root.after(0, lambda: self._on_ml_downloaded(key, ok, msg))

        threading.Thread(target=worker, daemon=True).start()

    def _on_ml_downloaded(self, key, ok, msg):
        self._ml_downloading = False
        if ok and self.ml_model_var.get() == key:
            self.mldepth.set_model(key)                  # reload so it's picked up
        self.status.config(text=msg)

    def _show(self, frame):
        if self.mirror_var.get():
            frame = cv2.flip(frame, 1)          # horizontal flip (selfie view)
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        disp = P.apply_aspect(frame, cw, ch, self.aspect_var.get())
        if self.fps_var.get() and not self._paused:
            self._draw_fps(disp)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.config(image=self._photo)
        if self.vcam is not None:
            self._send_vcam(frame)

    def _send_vcam(self, frame):
        try:
            v = P.apply_aspect(frame, VCAM_W, VCAM_H, "fit")
            self.vcam.send(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
        except Exception as e:
            self.status.config(text=f"virtual camera error: {e}")
            self._close_vcam()
            self.vcam_var.set(False)

    def _toggle_vcam(self):
        if self.vcam_var.get():
            try:
                self.vcam = pyvirtualcam.Camera(
                    width=VCAM_W, height=VCAM_H, fps=VCAM_FPS,
                    fmt=pyvirtualcam.PixelFormat.RGB)
                self.status.config(
                    text=f"virtual camera on: '{self.vcam.device}' "
                         f"(select it in Zoom/Discord/OBS)")
            except Exception as e:
                self.vcam = None
                self.vcam_var.set(False)
                self.status.config(text=f"virtual camera unavailable: {e}")
        else:
            self._close_vcam()
            self.status.config(text="virtual camera off")

    def _close_vcam(self):
        if self.vcam is not None:
            try:
                self.vcam.close()
            except Exception:
                pass
            self.vcam = None

    def _draw_fps(self, disp):
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = [f"{self._fps:4.1f} FPS"]
        if self._holding_phase():
            lines.append(f"IR {self._ir_fps:4.1f} fps")
        y = 0
        for i, txt in enumerate(lines):
            (tw, th), _ = cv2.getTextSize(txt, font, 0.6, 2)
            if i == 0:
                y = th + 12
            x = disp.shape[1] - tw - 12
            cv2.putText(disp, txt, (x, y), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(disp, txt, (x, y), font, 0.6, (80, 255, 180), 1, cv2.LINE_AA)
            y += th + 8

    # ------------------------------------------------------------------
    def _update_fps(self):
        now = time.time()
        dt = now - self._last
        self._last = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        # illuminated (held) IR frame rate, measured over a ~0.5s window so the
        # bursty strobe averages out
        win = now - self._ir_last_t
        if win >= 0.5:
            self._ir_fps = (self.cams.ir_new_count - self._ir_last_cnt) / win
            self._ir_last_cnt = self.cams.ir_new_count
            self._ir_last_t = now

    def _update_status(self, frame):
        if not self.statusbar_var.get():
            return
        desc = dict(MODES)[self.mode_var.get()]
        if self.mode_var.get() == "proximity" and getattr(self, "_depth_active", None):
            desc = f"Depth [{self._depth_active}]"
        rec = "  ● REC" if self.writer is not None else ""
        h, w = frame.shape[:2]
        ir = f"   |   IR {self._ir_fps:4.1f} fps" if self._holding_phase() else ""
        self.status.config(
            text=f"{desc}   |   aspect: {self.aspect_var.get()}   |   "
                 f"src {w}x{h}   |   alpha {self.alpha:.2f}   |   "
                 f"{self._fps:4.1f} fps{ir}{rec}")

    # -- menu/key callbacks ------------------------------------------------
    def _on_mode_change(self):
        pass  # mode_var already updated by radiobutton

    def _set_mode(self, key):
        self.mode_var.set(key)

    def _nudge(self, dx=0, dy=0, dscale=0.0):
        self.aligner.nudge(dx=dx, dy=dy, dscale=dscale)
        self._schedule_save()   # persist hand alignment (debounced)

    def _schedule_save(self):
        if self._save_after_id is not None:
            self.root.after_cancel(self._save_after_id)
        self._save_after_id = self.root.after(1200, self._save_current_calib)

    def _save_current_calib(self):
        self._save_after_id = None
        ir = self.cams.latest_ir()
        color = self.cams.latest_color()
        if ir is not None and color is not None and self.aligner.M is not None:
            if self.aligner.save(self.calib_path, ir.shape, color.shape):
                self.status.config(text=f"alignment saved to {self.calib_path}")

    def _set_alpha(self, d):
        # +/- means "opacity of the overlay for the current mode": depth blend
        # in the Depth map mode, IR-fusion opacity in the fusion modes
        if self.mode_var.get() == "proximity":
            self._set_depth_alpha(d)
        else:
            self.alpha = float(min(1.0, max(0.0, self.alpha + d)))

    def _set_depth_alpha(self, d, absolute=False):
        self.depth_alpha = float(d if absolute else self.depth_alpha + d)
        self.depth_alpha = min(1.0, max(0.0, self.depth_alpha))
        self.status.config(text=f"depth opacity: {int(self.depth_alpha * 100)}% "
                                f"depth / {int((1 - self.depth_alpha) * 100)}% original")

    def _auto_align_once(self):
        # Cross-modal matching is marginal on any single frame, so collect a
        # handful of frame pairs and pool their correspondences into one RANSAC
        # solve. The transform is fixed hardware geometry, so one success is
        # enough - then persist it for next launch.
        self.status.config(text="auto-aligning...")
        self.root.update_idletasks()
        pairs = []
        for _ in range(15):
            color = self.cams.latest_color()
            ir = self.cams.latest_ir()
            if color is not None and ir is not None:
                pairs.append((ir, color))
            time.sleep(0.04)

        if not pairs:
            self.status.config(text="auto-align: need both cameras streaming")
            return
        info = self.aligner.auto_align_pooled(pairs)
        if info.get("ok"):
            # polish the sparse-feature solve with photometric ECC
            self.aligner.refine_ecc(pairs[-1][0], pairs[-1][1])
            self._save_calib(pairs[-1][0], pairs[-1][1])
            self.status.config(
                text=f"auto-align OK: {info['inliers']}/{info['matches']} inliers, "
                     f"scale {info['scale']:.2f}, refined + saved  "
                     f"(use a fusion mode to see it)")
        else:
            self.status.config(
                text=f"auto-align failed ({info.get('reason', '?')}) - it's marginal "
                     f"cross-modal; align roughly by hand (arrows/[ ]) then press E "
                     f"to refine")

    def _refine_once(self):
        ir = self.cams.latest_ir()
        color = self.cams.latest_color()
        if ir is None or color is None:
            self.status.config(text="refine: need both cameras streaming")
            return
        self.status.config(text="refining alignment...")
        self.root.update_idletasks()
        r = self.aligner.refine_ecc(ir, color)
        if r.get("ok"):
            self._save_calib(ir, color)
            self.status.config(
                text=f"alignment refined (overlap {r['was']:.3f} -> {r['score']:.3f}) "
                     f"and saved")
        else:
            self.status.config(
                text=f"couldn't refine ({r.get('reason', '?')}) - get it roughly "
                     f"aligned by hand first (arrows to move, [ ] to zoom)")

    def _save_calib(self, ir, color):
        try:
            self.aligner.save(self.calib_path, ir.shape, color.shape)
        except Exception:
            pass

    def _reset_align(self):
        self.aligner = P.Aligner()
        self.alpha = 0.5

    def _set_ir_source(self):
        mode = self.ir_source_var.get()
        self.cams.ir_mode = mode
        labels = {"active": "active (emitter, anti-flicker)",
                  "passive": "passive (ambient IR, emitter excluded)",
                  "raw": "raw (unfiltered strobe)"}
        self.status.config(text=f"IR source: {labels.get(mode, mode)}")

    def _holding_phase(self) -> bool:
        """True when IR is phase-gated (active/passive), so the 'IR fps' figure
        reflects a distinct held-frame rate worth showing."""
        return self.ir_source_var.get() != "raw"

    def _toggle_statusbar(self):
        if self.statusbar_var.get():
            # re-pack BEFORE the expanding canvas so it reclaims its space
            self.status.pack(fill="x", side="bottom", before=self.canvas)
        else:
            self.status.pack_forget()

    def _snapshot(self):
        frame = self._grab_current()
        if frame is None:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir,
                            f"snap_{self.mode_var.get()}_{_stamp()}.png")
        cv2.imwrite(path, frame)
        self.status.config(text=f"saved {path}")

    def _toggle_record(self):
        if self.writer is None:
            frame = self._grab_current()
            if frame is None:
                return
            os.makedirs(self.out_dir, exist_ok=True)
            path = os.path.join(self.out_dir, f"rec_{_stamp()}.mp4")
            h, w = frame.shape[:2]
            self._rec_size = (w, h)
            self.writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
            self._cap_menu.entryconfig(1, label="Stop recording")
            self.status.config(text=f"recording -> {path}")
        else:
            self.writer.release()
            self.writer = None
            self._cap_menu.entryconfig(1, label="Start recording")
            self.status.config(text="recording stopped")

    def _write(self, frame):
        if (frame.shape[1], frame.shape[0]) != self._rec_size:
            frame = cv2.resize(frame, self._rec_size)
        self.writer.write(frame)

    def _grab_current(self):
        color = self.cams.latest_color()
        ir = self.cams.latest_ir()
        return self._render(color, ir)

    def _open_folder(self):
        os.makedirs(self.out_dir, exist_ok=True)
        try:
            os.startfile(os.path.abspath(self.out_dir))  # noqa: SC200 (Windows)
        except Exception as e:
            messagebox.showinfo("Captures", f"{os.path.abspath(self.out_dir)}\n\n{e}")

    def _show_controls(self):
        messagebox.showinfo(
            "Controls",
            "Modes:        keys 1-7  (or View > Mode)\n"
            "Aspect:       F fit / L fill / T stretch  (or View > Aspect)\n"
            "Mirror:       M  (flip horizontally, selfie view)\n"
            "Align by hand: arrow keys move, [ ] zoom the IR overlay\n"
            "               (auto-saved; reloads next launch)\n"
            "Refine align: E  (ECC polish of the current alignment)\n"
            "Auto-align:   A  (best-effort feature match, then refine)\n"
            "Opacity:      + / -  (fusion IR overlay, or depth-vs-original\n"
            "               blend in the Depth map mode)\n"
            "Snapshot:     Ctrl+S\n"
            "Record:       Ctrl+R\n"
            "Freeze:       Space  (cameras power down if held >10s)\n"
            "Quit:         Esc\n\n"
            "Tip: the IR and RGB cameras are fixed, so align once and it "
            "sticks. If auto-align misses, nudge it close by hand and press E.")

    def _show_about(self):
        messagebox.showinfo(
            "About HelloCam",
            "HelloCam - RGB + IR viewer for the Windows Hello camera.\n\n"
            "Reads the COLOR and INFRARED sensors via the WinRT Media Frame\n"
            "Source API. No dedicated depth sensor, so depth is either an\n"
            "IR-intensity proximity proxy or RGB<->IR stereo parallax\n"
            "(relative, not metric). View > Depth method: auto/stereo/proximity.")

    def _on_close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self._close_vcam()
        try:
            self.cams.close()
        except Exception:
            pass
        self.root.destroy()
        print("Closed. IR emitter off.", flush=True)


def _stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def run():
    ViewerGUI().run()
