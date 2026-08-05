"""Native-windowed viewer (Tkinter) for the Surface RGB + IR cameras.

Uses a real Windows menu bar for all controls:
  View    -> Mode (IR / RGB / fusion / ...) and Aspect (Fit / Fill / Stretch)
  Capture -> Snapshot, Record, open output folder
  Overlay -> live IR<->RGB alignment for the fusion modes
  Help    -> controls & about

The camera/processing core (surfacecam.capture / .processing) is reused
unchanged; only the front-end differs from the OpenCV viewer.
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from .capture import SurfaceCameras
from . import processing as P

MODES = [
    ("ir",        "IR (raw grayscale)"),
    ("rgb",       "RGB webcam"),
    ("side",      "Side by side: RGB | IR"),
    ("fuse",      "RGB + IR fusion (overlay)"),
    ("edgefuse",  "RGB + IR edges"),
    ("proximity", "IR proximity map (not true depth)"),
    ("iredges",   "IR edges"),
]
ASPECTS = ["fit", "fill", "stretch"]


class ViewerGUI:
    def __init__(self, out_dir: str = "captures"):
        self.out_dir = out_dir
        self.cams = SurfaceCameras(color=True, ir=True)
        self.aligner = P.Aligner()
        self.alpha = 0.5
        self.writer = None
        self._rec_size = None
        self._fps = 0.0
        self._last = time.time()
        self._photo = None  # keep a ref so Tk doesn't GC the image

        self.root = tk.Tk()
        self.root.title("SurfaceCam - RGB + IR")
        self.root.geometry("980x680")
        self.root.minsize(320, 240)
        self.root.configure(bg="black")

        # tk vars driving the menus
        self.mode_var = tk.StringVar(value=MODES[0][0])
        self.aspect_var = tk.StringVar(value="fit")
        self.statusbar_var = tk.BooleanVar(value=True)

        self._build_menu()
        self._build_widgets()
        self._bind_keys()

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
        view.add_checkbutton(label="Show status bar", variable=self.statusbar_var,
                             command=self._toggle_statusbar)
        view.add_command(label="Reset IR alignment", command=self._reset_align)
        view.add_separator()
        view.add_command(label="Exit", accelerator="Esc", command=self._on_close)
        menubar.add_cascade(label="View", menu=view)

        # Capture
        cap = tk.Menu(menubar, tearoff=0)
        cap.add_command(label="Snapshot", accelerator="Ctrl+S", command=self._snapshot)
        self._rec_label = tk.StringVar(value="Start recording")
        cap.add_command(label="Start recording", accelerator="Ctrl+R",
                        command=self._toggle_record)
        self._cap_menu = cap
        cap.add_separator()
        cap.add_command(label="Open captures folder", command=self._open_folder)
        menubar.add_cascade(label="Capture", menu=cap)

        # Overlay (fusion alignment)
        ov = tk.Menu(menubar, tearoff=0)
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
        r.bind("<Escape>", lambda e: self._on_close())
        r.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    def run(self):
        print("Opening cameras (this powers on the IR emitter)...", flush=True)
        self.cams.open()
        print(f"  color: {self.cams.color_size}   ir: {self.cams.ir_size}", flush=True)
        self.root.after(0, self._tick)
        self.root.mainloop()

    def _tick(self):
        try:
            color = self.cams.read_color()
            ir = self.cams.read_ir()
            frame = self._render(color, ir)
            if frame is not None:
                self._update_fps()
                if self.writer is not None:
                    self._write(frame)
                self._show(frame)
                self._update_status(frame)
        except Exception as e:  # keep the loop alive, surface once
            self.status.config(text=f"error: {e}")
        self.root.after(15, self._tick)

    # ------------------------------------------------------------------
    def _render(self, color, ir):
        name = self.mode_var.get()
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
        if color is None or ir is None:
            return color if color is not None else (
                P.ir_view(ir) if ir is not None else None)
        aligned = self.aligner.warp_ir_to_color(ir, color.shape)
        if name == "fuse":
            return P.fuse(color, aligned, alpha=self.alpha)
        if name == "edgefuse":
            return P.edge_fuse(color, aligned)
        return color

    def _show(self, frame):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        disp = P.apply_aspect(frame, cw, ch, self.aspect_var.get())
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.config(image=self._photo)

    # ------------------------------------------------------------------
    def _update_fps(self):
        now = time.time()
        dt = now - self._last
        self._last = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)

    def _update_status(self, frame):
        if not self.statusbar_var.get():
            return
        desc = dict(MODES)[self.mode_var.get()]
        rec = "  ● REC" if self.writer is not None else ""
        h, w = frame.shape[:2]
        self.status.config(
            text=f"{desc}   |   aspect: {self.aspect_var.get()}   |   "
                 f"src {w}x{h}   |   alpha {self.alpha:.2f}   |   "
                 f"{self._fps:4.1f} fps{rec}")

    # -- menu/key callbacks ------------------------------------------------
    def _on_mode_change(self):
        pass  # mode_var already updated by radiobutton

    def _set_mode(self, key):
        self.mode_var.set(key)

    def _nudge(self, dx=0, dy=0, dscale=0.0):
        self.aligner.nudge(dx=dx, dy=dy, dscale=dscale)

    def _set_alpha(self, d):
        self.alpha = float(min(1.0, max(0.0, self.alpha + d)))

    def _reset_align(self):
        self.aligner = P.Aligner()
        self.alpha = 0.5

    def _toggle_statusbar(self):
        if self.statusbar_var.get():
            self.status.pack(fill="x", side="bottom")
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
        color = self.cams.read_color()
        ir = self.cams.read_ir()
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
            "Fusion align: arrow keys move, [ ] zoom the IR overlay\n"
            "Opacity:      + / -\n"
            "Snapshot:     Ctrl+S\n"
            "Record:       Ctrl+R\n"
            "Quit:         Esc")

    def _show_about(self):
        messagebox.showinfo(
            "About SurfaceCam",
            "SurfaceCam - RGB + IR viewer for the Surface Windows Hello camera.\n\n"
            "Reads the COLOR and INFRARED sensors via the WinRT Media Frame\n"
            "Source API. The module has no depth sensor, so 'proximity' is an\n"
            "IR-intensity closeness proxy, not metric depth.")

    def _on_close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
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
