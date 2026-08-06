"""Proof-of-concept IR transmitter using the Surface Hello emitter.

We can't control the emitter's individual pulses (the firmware strobes it and
Windows exposes no timing API), so this does the one thing we *can*: gate the
whole IR stream on/off. Stream running = emitter firing = "IR on"; stopped =
"IR off". That's on-off keying (OOK) at ~2-3 bits/sec - useless for real IR
protocols, fine for a blink-a-message demo.

Encodes text as Morse and blinks it. Point a phone camera at the little IR
window next to the lens: phones see near-IR, so you'll watch it flash and can
decode by eye (or record and analyse the brightness).

    python scripts/ir_transmit.py "HI"
    python scripts/ir_transmit.py --unit 0.5 "SOS"
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

from winsdk.windows.media.capture.frames import (
    MediaFrameSourceGroup, MediaFrameSourceKind)
from winsdk.windows.media.capture import (
    MediaCapture, MediaCaptureInitializationSettings,
    MediaCaptureMemoryPreference, MediaCaptureSharingMode, StreamingCaptureMode)

MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..", "0": "-----", "1": ".----", "2": "..---",
    "3": "...--", "4": "....-", "5": ".....", "6": "-....", "7": "--...",
    "8": "---..", "9": "----.",
}


async def _open_ir_reader():
    groups = await MediaFrameSourceGroup.find_all_async()
    grp = next(g for g in groups
               if any(si.source_kind == MediaFrameSourceKind.INFRARED
                      for si in g.source_infos))
    info = next(si for si in grp.source_infos
                if si.source_kind == MediaFrameSourceKind.INFRARED)
    s = MediaCaptureInitializationSettings()
    s.source_group = grp
    s.memory_preference = MediaCaptureMemoryPreference.CPU
    s.streaming_capture_mode = StreamingCaptureMode.VIDEO
    s.sharing_mode = MediaCaptureSharingMode.EXCLUSIVE_CONTROL
    mc = MediaCapture()
    await mc.initialize_async(s)
    src = mc.frame_sources[info.id]
    reader = await mc.create_frame_reader_async(src)
    return mc, reader


async def transmit(message: str, unit: float = 0.4):
    message = message.upper()
    mc, reader = await _open_ir_reader()
    streaming = False

    async def emit(on: bool):
        nonlocal streaming
        if on and not streaming:
            await reader.start_async(); streaming = True
        elif not on and streaming:
            await reader.stop_async(); streaming = False

    async def flash(on_units: float):
        await emit(True)
        await asyncio.sleep(on_units * unit)
        await emit(False)

    print(f'Transmitting "{message}" as IR Morse '
          f'(unit={unit}s). Point a phone camera at the IR emitter.\n')
    try:
        # long preamble (6 units - longer than any symbol) so a receiver can
        # spot the start and drop it as sync rather than decode it as a dash
        print("[preamble]"); await flash(6)
        await asyncio.sleep(3 * unit)

        for ch in message:
            if ch == " ":
                print("  (word gap)")
                await asyncio.sleep(7 * unit)
                continue
            code = MORSE.get(ch)
            if not code:
                continue
            print(f"  {ch}: {code}")
            for i, sym in enumerate(code):
                await flash(3 if sym == "-" else 1)     # dash=3u, dot=1u
                if i < len(code) - 1:
                    await asyncio.sleep(unit)            # intra-letter gap
            await asyncio.sleep(3 * unit)                # inter-letter gap
        print("\n[done]")
    finally:
        await emit(False)
        mc.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("message", help="text to transmit (letters/digits/spaces)")
    ap.add_argument("--unit", type=float, default=0.4,
                    help="Morse time unit in seconds (default 0.4; larger = "
                         "slower but easier to read)")
    args = ap.parse_args()
    t0 = time.time()
    asyncio.run(transmit(args.message, args.unit))
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
