# SurfaceCam — Windows Hello RGB + IR viewer

View and fuse the two cameras inside a Surface Laptop's Windows Hello module:
the normal **RGB webcam** and the **near-infrared (IR)** sensor that Hello uses
for face unlock. The IR sensor is not visible to DirectShow apps (ffmpeg, OBS,
the Camera app) — it lives on the modern **MediaFoundation / Media Frame Source**
API, which this project drives from Python via the `winsdk` WinRT projection.

Nothing here bypasses a security control. Windows exposes these sources to
normal apps by design; it only reserves the IR stream momentarily while Hello is
mid-authentication. This tool just *reads* the sensors — it does **not** feed
frames back to spoof Hello.

## Features

- **IR** — raw near-IR grayscale (the IR emitter powers on while streaming)
- **RGB** — the standard color webcam
- **Side by side** — RGB next to a false-colored IR view
- **RGB + IR fusion** — IR overlaid on the color image, live-adjustable alignment
- **RGB + IR edges** — IR-derived edges drawn over the RGB frame
- **Proximity map** — IR intensity false-colored as a closeness proxy
  (see *Depth*, below)
- **IR edges** — Canny edges of the IR frame
- Snapshot to PNG and record to MP4

## Install

```bash
pip install -r requirements.txt
```

Requires Windows with a Hello IR camera and Python 3.10–3.12.

## Run

```bash
python main.py                 # native GUI viewer (menu bar) - default
python main.py --cv            # legacy OpenCV window (keyboard only)
python main.py --enumerate     # list camera sources
python main.py --selftest      # open both cams, save sample frames, exit (no window)
```

### GUI

The default viewer is a native Windows window with a **menu bar**:

- **View → Mode** — IR / RGB / side-by-side / fusion / edge-fusion / proximity / IR-edges
- **View → Aspect ratio** — **Fit** (letterbox, no crop), **Fill** (crop to fill),
  **Stretch** (ignore aspect). Default is *Fit*, so the video is never distorted.
- **View → Show FPS overlay** — draw the live frame rate on the image
- **View → Anti-flicker** — hold the last illuminated IR frame so the strobing
  emitter doesn't make the IR view flicker (on by default; turn off to see the
  raw alternating bright/dark frames)
- **Capture** — Snapshot, Start/Stop recording, Open captures folder
- **Overlay** — **Auto-align** the IR onto the RGB feed, plus manual
  move / zoom / opacity for the fusion modes
- **Help** — controls & about

### Auto-align

**Overlay → Auto-align now** (`A`) lines the IR image up with the RGB feed
automatically: it matches shared features (ORB) between the two views, pools
the correspondences across ~15 frames, and solves a similarity transform
(zoom + rotate + move) with RANSAC. Because the two sensors are fixed in the
chassis, the transform is constant — so the result is **saved to
`calibration.json`** and reloaded on the next launch. **Overlay → Auto-align
(live)** keeps re-solving as you move. Fill the frame with texture (your face
+ background) for the best lock.

> The IR emitter *strobes*, so raw IR frames alternate bright/dark. The capture
> layer detects and holds the last illuminated frame, which stabilises the live
> view and gives auto-align a usable image.

### Frame rate

The IR sensor exposes a single mode — **480×480 L8 @ 60fps** — and that is its
maximum; there is no higher-rate format to request. To actually capture at that
rate, frames are pulled on a **background thread** (`SurfaceCameras.start_pump`)
and cached, so the sensor runs full-speed independently of the render loop. The
GUI then draws the freshest frame as fast as it can (the "FPS" overlay shows the
draw rate; the "IR fps" line shows the true capture rate). Heavier modes
(fusion) draw slower than light ones (raw IR), but no captured frames are
dropped either way.

Keyboard shortcuts mirror the menus:

| Key | Action |
|-----|--------|
| `1`–`7` | switch mode |
| `F` / `L` / `T` | aspect Fit / Fill / Stretch |
| `A` | auto-align IR to RGB |
| arrows | nudge the IR overlay alignment |
| `[` `]` | scale the IR overlay |
| `-` `+` | fusion overlay opacity |
| `Ctrl+S` | save a snapshot (PNG) |
| `Ctrl+R` | start/stop recording (MP4) |
| `Esc` | quit (releases the camera, IR emitter off) |

Output files go to `captures/`.

## About "depth"

The Surface Hello camera has a **color** and an **infrared** sensor — but **no
depth sensor** (no stereo pair, no time-of-flight). So there is no true metric
depth to read. Two honest alternatives:

1. **Proximity map (implemented).** IR flood illumination falls off with
   distance, so IR brightness is a rough proxy for how close something is.
   The `proximity` mode false-colors this. It is a real signal, but it is
   *not* calibrated depth — a white shirt up close and a face further away can
   read similarly.
2. **ML monocular depth (optional, not installed).** A model such as MiDaS can
   *estimate* depth from the RGB image alone. It needs PyTorch (~2GB), so it is
   left as an opt-in dependency in `requirements.txt`.

## Layout

```
main.py                     entry point / CLI
surfacecam/
  capture.py                MediaFrameSource wrapper (RGB + IR -> numpy)
  processing.py             colormaps, fusion, proximity, alignment, aspect
  gui.py                    native Tkinter viewer with menu bar (default)
  app.py                    legacy OpenCV viewer (--cv)
scripts/enumerate_sources.py  standalone source enumeration probe
```

## How it works

1. Enumerate `MediaFrameSourceGroup`s and pick the one exposing both a
   `COLOR` and an `INFRARED` source (the Hello "sensor group").
2. Open **one** `MediaCapture` on that group in exclusive mode (CPU memory).
3. Create one `MediaFrameReader` per sensor; each `SoftwareBitmap` is copied
   into a numpy array (IR → GRAY8, color → BGRA8 → BGR).
4. Process and display with OpenCV.
