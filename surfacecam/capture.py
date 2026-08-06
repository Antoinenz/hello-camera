"""Access the Surface's Windows Hello camera module (RGB + IR) via the
Windows MediaFoundation / Media Frame Source API (WinRT, through `winsdk`).

The Hello module exposes two sensors on one physical USB device:
  - COLOR    (MI_00) - normal RGB webcam
  - INFRARED (MI_02) - near-IR sensor used by Windows Hello, with an IR emitter

Both are bundled in a single "sensor group", so we open ONE MediaCapture on
that group and create one frame reader per sensor. Opening the IR reader in
exclusive mode is what powers on the IR emitter.
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass

import numpy as np

from winsdk.windows.media.capture.frames import (
    MediaFrameSourceGroup,
    MediaFrameSourceKind,
    MediaFrameReaderStartStatus,
)
from winsdk.windows.media.capture import (
    MediaCapture,
    MediaCaptureInitializationSettings,
    MediaCaptureMemoryPreference,
    MediaCaptureSharingMode,
    StreamingCaptureMode,
)
from winsdk.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat
from winsdk.windows.storage.streams import Buffer
from winsdk.windows.security.cryptography import CryptographicBuffer


# ---------------------------------------------------------------------------
# SoftwareBitmap -> numpy helpers
# ---------------------------------------------------------------------------
def _bitmap_bytes(bmp: SoftwareBitmap, bpp: int) -> bytes:
    w, h = bmp.pixel_width, bmp.pixel_height
    buf = Buffer(w * h * bpp)
    bmp.copy_to_buffer(buf)
    return bytes(CryptographicBuffer.copy_to_byte_array(buf))


def bitmap_to_gray(bmp: SoftwareBitmap) -> np.ndarray:
    """IR frame -> 2D uint8 array (16-bit sources are scaled down to 8-bit)."""
    fmt = bmp.bitmap_pixel_format
    w, h = bmp.pixel_width, bmp.pixel_height
    if fmt == BitmapPixelFormat.GRAY16:
        raw = _bitmap_bytes(bmp, 2)
        arr = np.frombuffer(raw, np.uint16).reshape(h, w)
        return (arr >> 8).astype(np.uint8)
    if fmt != BitmapPixelFormat.GRAY8:
        bmp = SoftwareBitmap.convert(bmp, BitmapPixelFormat.GRAY8)
        w, h = bmp.pixel_width, bmp.pixel_height
    raw = _bitmap_bytes(bmp, 1)
    return np.frombuffer(raw, np.uint8).reshape(h, w)


def bitmap_to_bgr(bmp: SoftwareBitmap) -> np.ndarray:
    """Color frame -> HxWx3 uint8 BGR array (ready for OpenCV)."""
    if bmp.bitmap_pixel_format != BitmapPixelFormat.BGRA8:
        bmp = SoftwareBitmap.convert(bmp, BitmapPixelFormat.BGRA8)
    w, h = bmp.pixel_width, bmp.pixel_height
    raw = _bitmap_bytes(bmp, 4)
    arr = np.frombuffer(raw, np.uint8).reshape(h, w, 4)
    # WinRT BGRA8 stores bytes as B,G,R,A -> first three channels are already BGR
    return arr[:, :, :3].copy()


# ---------------------------------------------------------------------------
# Camera wrapper
# ---------------------------------------------------------------------------
@dataclass
class _SourceRef:
    info_id: str
    reader: object = None


class SurfaceCameras:
    """Synchronous facade over the async WinRT camera API.

    Usage:
        cams = SurfaceCameras(color=True, ir=True)
        cams.open()
        bgr = cams.read_color()   # HxWx3 uint8 or None
        ir  = cams.read_ir()      # HxW  uint8 or None
        cams.close()
    """

    def __init__(self, color: bool = True, ir: bool = True, max_color_width: int = 1280,
                 ir_mode: str = "active"):
        self.want_color = color
        self.want_ir = ir
        self.max_color_width = max_color_width
        # Windows Hello IR cameras strobe the emitter: frames alternate between
        # illuminated (bright) and non-illuminated (ambient IR only). ir_mode:
        #   "active"  - hold the emitter-lit frames (anti-flicker; default)
        #   "passive" - hold the non-illuminated frames (ambient IR, no emitter
        #               floodlight). The driver won't let us stop the emitter
        #               firing, but this shows only the frames it didn't light.
        #   "raw"     - no gating; the unfiltered strobing stream
        self.ir_mode = ir_mode
        self._mc: MediaCapture | None = None
        self._color = _SourceRef("")
        self._ir = _SourceRef("")
        self.color_size: tuple[int, int] | None = None
        self.ir_size: tuple[int, int] | None = None
        self._last_ir: np.ndarray | None = None
        self._ir_bright: float = 0.0     # tracked illuminated-frame level
        self._ir_dark: float = 0.0       # tracked non-illuminated level
        # counts distinct *illuminated* IR frames accepted (held-frame stream);
        # lets a consumer measure the true illuminated frame rate
        self.ir_new_count: int = 0
        # bumps whenever the cached color/IR frame changes, so a renderer can
        # draw only on new frames instead of re-drawing duplicates
        self.frame_ver: int = 0
        # background capture pump (decouples sensor rate from the render loop)
        self._pump: threading.Thread | None = None
        self._pump_stop: threading.Event | None = None
        self._lock = threading.Lock()
        self._cache_color: np.ndarray | None = None
        self._cache_ir: np.ndarray | None = None
        # power management: desired per-source enable (mode-driven) and a global
        # suspend (freeze / minimised). The pump reconciles the actual reader
        # streaming state to these, so unused cameras are powered down.
        self._want_color = True
        self._want_ir = True
        self._suspend = False
        self._color_streaming = False
        self._ir_streaming = False

    # -- public sync API ---------------------------------------------------
    def open(self, retries: int = 5, retry_delay: float = 1.5):
        # The Hello camera is exclusive; right after a prior handle closes the
        # MFT can briefly report "lack of hardware resources". Retry a few times.
        last = None
        for attempt in range(retries):
            try:
                asyncio.run(self._open_async())
                self._color_streaming = self._color.reader is not None
                self._ir_streaming = self._ir.reader is not None
                return
            except OSError as e:
                last = e
                try:
                    asyncio.run(self._close_async())    # release partial handle
                except Exception:
                    pass
                time.sleep(retry_delay)
        raise last

    def read_color(self) -> np.ndarray | None:
        return self._read(self._color, bitmap_to_bgr) if self.want_color else None

    def read_ir(self) -> np.ndarray | None:
        if not self.want_ir:
            return None
        arr = self._read(self._ir, bitmap_to_gray)
        if self.ir_mode == "raw":
            if arr is not None:             # raw stream: every new frame counts
                self.ir_new_count += 1
            return arr
        if arr is None:
            return self._last_ir            # keep last good frame across gaps
        m = float(arr.mean())
        if self._last_ir is None:           # first frame: accept and seed levels
            self._ir_bright = self._ir_dark = m
            self._last_ir = arr
            self.ir_new_count += 1
            return arr
        # Track the two strobe levels: bright rises fast / decays slow, dark
        # falls fast / rises slow. Split illuminated vs dark at their midpoint,
        # but only when there's real strobe contrast - otherwise pass frames
        # through (nothing to gate).
        self._ir_bright = max(m, 0.90 * self._ir_bright + 0.10 * m)
        self._ir_dark = min(m, 0.90 * self._ir_dark + 0.10 * m)
        contrast = self._ir_bright - self._ir_dark
        midpoint = 0.5 * (self._ir_bright + self._ir_dark)
        # gate whenever there's a real strobe, even a weak one (a few gray
        # levels) - the emitter's contribution shrinks when ambient IR is high
        if contrast > max(4.0, 0.06 * self._ir_bright):
            is_bright = m >= midpoint
            want_bright = self.ir_mode == "active"
            if is_bright != want_bright:    # wrong strobe phase -> hold previous
                return self._last_ir
        self._last_ir = arr                 # wanted phase -> use, remember, count
        self.ir_new_count += 1
        return arr

    def close(self):
        self.stop_pump()
        try:
            asyncio.run(self._close_async())
        except Exception:
            pass

    # -- power management (mode-driven enable + global suspend) -----------
    def request_sources(self, color: bool, ir: bool):
        """Set which sources the current use actually needs; the pump powers
        down the others (and powers them back up on demand)."""
        self._want_color = bool(color)
        self._want_ir = bool(ir)

    def suspend(self):
        """Stop streaming both cameras (emitter off) while keeping the device
        initialised, so it resumes cheaply. Used for freeze / minimise."""
        self._suspend = True

    def resume(self):
        self._suspend = False

    @property
    def suspended(self) -> bool:
        return self._suspend

    @property
    def color_on(self) -> bool:
        return self._color_streaming

    @property
    def ir_on(self) -> bool:
        return self._ir_streaming

    def _reconcile(self, ev_loop):
        """Bring actual reader streaming in line with desired state. Runs only
        in the pump thread, so readers are touched by exactly one thread."""
        want_c = self.want_color and self._want_color and not self._suspend
        want_i = self.want_ir and self._want_ir and not self._suspend
        if self._color.reader is not None and want_c != self._color_streaming:
            op = (self._color.reader.start_async() if want_c
                  else self._color.reader.stop_async())
            ev_loop.run_until_complete(_await_op(op))
            self._color_streaming = want_c
            if not want_c:
                with self._lock:
                    self._cache_color = None
        if self._ir.reader is not None and want_i != self._ir_streaming:
            op = (self._ir.reader.start_async() if want_i
                  else self._ir.reader.stop_async())
            ev_loop.run_until_complete(_await_op(op))
            self._ir_streaming = want_i
            if not want_i:
                self._last_ir = None            # reset strobe tracking
                with self._lock:
                    self._cache_ir = None

    # -- background capture pump ------------------------------------------
    def start_pump(self, interval: float = 0.002):
        """Continuously pull frames off the sensor on a background thread so
        capture runs at the full sensor rate regardless of how fast the render
        loop consumes them. Consumers then read latest_color()/latest_ir().
        The pump also owns reader start/stop (see _reconcile)."""
        if self._pump is not None:
            return
        self._pump_stop = threading.Event()

        def loop():
            ev_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(ev_loop)
            try:
                while not self._pump_stop.is_set():
                    try:
                        self._reconcile(ev_loop)
                    except Exception:
                        pass
                    if self._suspend:
                        time.sleep(0.1)         # idle: cameras off
                        continue
                    c = self.read_color() if self._color_streaming else None
                    ir = self.read_ir() if self._ir_streaming else None
                    if c is not None or ir is not None:
                        with self._lock:
                            if c is not None:
                                self._cache_color = c
                            if ir is not None:
                                self._cache_ir = ir
                            self.frame_ver += 1
                    time.sleep(interval)
            finally:
                ev_loop.close()

        self._pump = threading.Thread(target=loop, name="ir-pump", daemon=True)
        self._pump.start()

    def stop_pump(self):
        if self._pump is None:
            return
        self._pump_stop.set()
        self._pump.join(timeout=2.0)
        self._pump = None

    def latest_color(self) -> np.ndarray | None:
        with self._lock:
            return self._cache_color

    def latest_ir(self) -> np.ndarray | None:
        with self._lock:
            return self._cache_ir

    # -- internals ---------------------------------------------------------
    def _read(self, ref: _SourceRef, conv):
        if ref.reader is None:
            return None
        frame = ref.reader.try_acquire_latest_frame()
        if frame is None:
            return None
        vmf = frame.video_media_frame
        if vmf is None:
            return None
        bmp = vmf.software_bitmap
        if bmp is None:
            return None
        return conv(bmp)

    async def _open_async(self):
        groups = await MediaFrameSourceGroup.find_all_async()

        # Prefer a single group that contains every source kind we want.
        need = set()
        if self.want_color:
            need.add(MediaFrameSourceKind.COLOR)
        if self.want_ir:
            need.add(MediaFrameSourceKind.INFRARED)

        chosen = None
        for g in groups:
            kinds = {si.source_kind for si in g.source_infos}
            if need.issubset(kinds):
                chosen = g
                break
        if chosen is None:
            raise RuntimeError(
                f"No sensor group exposes all of {[k.name for k in need]}. "
                f"Available groups: {[g.display_name for g in groups]}"
            )

        for si in chosen.source_infos:
            if si.source_kind == MediaFrameSourceKind.COLOR and self.want_color:
                self._color.info_id = si.id
            elif si.source_kind == MediaFrameSourceKind.INFRARED and self.want_ir:
                self._ir.info_id = si.id

        settings = MediaCaptureInitializationSettings()
        settings.source_group = chosen
        settings.memory_preference = MediaCaptureMemoryPreference.CPU
        settings.streaming_capture_mode = StreamingCaptureMode.VIDEO
        settings.sharing_mode = MediaCaptureSharingMode.EXCLUSIVE_CONTROL

        mc = MediaCapture()
        await mc.initialize_async(settings)
        self._mc = mc

        if self.want_color and self._color.info_id:
            src = mc.frame_sources[self._color.info_id]
            await self._maybe_set_color_format(src)
            self.color_size = (src.current_format.video_format.width,
                               src.current_format.video_format.height)
            reader = await mc.create_frame_reader_async(src)
            status = await reader.start_async()
            _check(status, "color")
            self._color.reader = reader

        if self.want_ir and self._ir.info_id:
            src = mc.frame_sources[self._ir.info_id]
            self.ir_size = (src.current_format.video_format.width,
                            src.current_format.video_format.height)
            reader = await mc.create_frame_reader_async(src)
            status = await reader.start_async()
            _check(status, "infrared")
            self._ir.reader = reader

    async def _maybe_set_color_format(self, src):
        """Pick a reasonable uncompressed color format so software_bitmap is
        populated and resolution stays manageable."""
        best = None
        for f in src.supported_formats:
            sub = (f.subtype or "").upper()
            if sub not in ("NV12", "YUY2", "YUYV"):
                continue
            vf = f.video_format
            if vf.width > self.max_color_width:
                continue
            area = vf.width * vf.height
            if best is None or area > best[0]:
                best = (area, f)
        if best is not None:
            await src.set_format_async(best[1])

    async def _close_async(self):
        for ref in (self._color, self._ir):
            if ref.reader is not None:
                try:
                    await ref.reader.stop_async()
                except Exception:
                    pass
                ref.reader = None
        if self._mc is not None:
            self._mc.close()
            self._mc = None


def _check(status, name):
    if status != MediaFrameReaderStartStatus.SUCCESS:
        raise RuntimeError(f"Failed to start {name} reader: {status!r}")


async def _await_op(op):
    """Await a WinRT IAsyncOperation from a thread's own event loop."""
    return await op
