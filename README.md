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
- **View → IR source** — which strobe phase to show (see *IR source* below):
  **Active** (emitter, anti-flicker; default), **Passive** (ambient IR, emitter
  excluded), or **Raw** (unfiltered strobe)
- **View → Freeze / resume** (**Space**) — freeze the frame; see *Power saving*
- **View → Auto-pause when minimized** — pause work while minimized (default on)
- **Capture** — Snapshot, Start/Stop recording, Open captures folder
- **Overlay** — **Auto-align** the IR onto the RGB feed, plus manual
  move / zoom / opacity for the fusion modes
- **Help** — controls & about

### Power saving

The app powers down what it isn't using:

- **Unused camera off** — each mode only needs some sensors (IR modes need the IR
  camera, *RGB* needs the color camera, fusion needs both). After a few seconds
  in a mode, the camera it doesn't use is stopped (and for IR, the emitter with
  it). Switching to a mode that needs it powers it straight back on.
- **Freeze** (**Space**) — holds the current frame and stops processing. If it
  stays frozen for ~10s, both cameras are suspended entirely; pressing Space
  again resumes them.
- **Auto-pause when minimized** (on by default) — minimizing pauses rendering
  immediately and suspends the cameras a few seconds later; restoring the window
  brings them back.

Suspending keeps the capture device initialised, so resume is quick — it doesn't
re-run the full open sequence.

### Aligning the IR overlay

The IR and RGB sensors are fixed in the chassis, so their alignment is a
**constant** — you only need to get it right once. It's saved to
`calibration.json` and reloaded automatically on the next launch.

Recommended workflow:

1. **Align by hand** — in a fusion mode, use the **arrow keys** to move and
   `[` / `]` to zoom the IR overlay until it roughly lines up. This is
   auto-saved as you go.
2. **Refine** — press **`E`** (Overlay → Refine alignment). This runs a
   photometric ECC alignment in the gradient domain that polishes the current
   transform to sub-pixel and saves it. It only keeps the result if it
   measurably improves the overlap, so it never makes things worse.

**`A`** (Overlay → Auto-align) tries to find the alignment from scratch by
matching shared features (ORB) across ~15 frames, then refines with ECC. It's
convenient when it works, but cross-modal (near-IR vs visible) feature matching
is inherently marginal — if it misses, just nudge it close by hand and press
`E`. **Overlay → Auto-align (live)** keeps re-solving as you move.

### IR source (emitter phase)

The IR emitter *strobes*: frames alternate between **illuminated** (emitter on)
and **non-illuminated** (ambient IR only). **View → IR source** picks which the
app shows:

- **Active** *(default)* — holds the illuminated frames. Steady, no flicker,
  and what feeds auto-align.
- **Passive** — holds the non-illuminated frames: what the IR sensor sees from
  *ambient* light alone, with the emitter's floodlight excluded (good for
  spotting IR light sources, sunlight, remote-control LEDs, etc.).
- **Raw** — the unfiltered stream, so you see the raw bright/dark strobe.

Note: this Surface's driver reports no `InfraredTorchControl` support, so the
emitter can't actually be switched off in firmware — it still fires on alternate
frames. *Passive* simply shows only the frames it didn't light, which looks like
the emitter is off. In a dark room the passive view will be dim (no ambient IR
to see by); it's brightest with sunlight or IR-rich lighting.

### Frame rate

The IR sensor exposes a single mode — **480×480 L8 @ 60fps** — and that is its
maximum; there is no higher-rate format to request. Getting the app to actually
run near that rate came down to three things (all CPU — the work is light enough
that GPU offload isn't needed and would only add transfer overhead):

1. **Background capture pump** (`SurfaceCameras.start_pump`) pulls frames on a
   thread so the sensor runs full-speed independently of rendering.
2. **Fast fusion** — the RGB+IR blend uses OpenCV SIMD `addWeighted` +
   `cv2.copyTo` instead of full-frame float math (≈48ms → ≈1.5ms per frame).
3. **Render on new frames only** — the draw loop redraws when the pump reports a
   new frame (tracked by `frame_ver`) rather than on a fixed timer, so it neither
   throttles nor wastes work re-drawing duplicates.

Result: all modes, including fusion, render around 45–60fps. The "FPS" overlay
shows the draw rate; the "IR fps" line shows the true illuminated capture rate.

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

## IR transmit (proof-of-concept)

You can't send *real* IR signals (TV-remote protocols, data) with this emitter:
those need a modulated ~38kHz carrier with microsecond pulse timing, and the
firmware gives us no per-pulse control — plus this device reports no
`InfraredTorchControl`. The one thing we *can* control is the whole IR stream:
starting it lights the emitter, stopping it dark. That's on-off keying at
**~2-3 bits/sec** — useless for anything practical, but enough to blink a
message. Purely for fun.

```bash
python scripts/ir_transmit.py "SOS"          # blink text as IR Morse
python scripts/ir_transmit.py --unit 0.5 "HI"  # slower = easier to read
```

Point a phone camera at the IR emitter (phones see near-IR) to watch it flash.
To decode a recording back to text:

```bash
python scripts/ir_receive.py clip.mp4        # tracks brightness -> Morse -> text
```

The encode/decode round-trip is verified end-to-end in software; the only
analog link is your phone recording the blink.

## Layout

```
main.py                     entry point / CLI
surfacecam/
  capture.py                MediaFrameSource wrapper (RGB + IR -> numpy)
  processing.py             colormaps, fusion, proximity, alignment, aspect
  gui.py                    native Tkinter viewer with menu bar (default)
  app.py                    legacy OpenCV viewer (--cv)
scripts/enumerate_sources.py  standalone source enumeration probe
scripts/ir_transmit.py        blink a Morse message via the IR emitter (PoC)
scripts/ir_receive.py         decode a phone recording of the blink back to text
```

## How it works

1. Enumerate `MediaFrameSourceGroup`s and pick the one exposing both a
   `COLOR` and an `INFRARED` source (the Hello "sensor group").
2. Open **one** `MediaCapture` on that group in exclusive mode (CPU memory).
3. Create one `MediaFrameReader` per sensor; each `SoftwareBitmap` is copied
   into a numpy array (IR → GRAY8, color → BGRA8 → BGR).
4. Process and display with OpenCV.
