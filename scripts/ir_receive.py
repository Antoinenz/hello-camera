"""Decode an IR-Morse transmission (from scripts/ir_transmit.py) out of a video.

The transmitter blinks the Windows Hello IR emitter via stream gating. Record that
blinking emitter with a phone camera (phones see near-IR), then feed the clip
here: it tracks the brightest region's intensity over time, thresholds it to
on/off, segments by the Morse timing, and prints the decoded text.

    python scripts/ir_receive.py clip.mp4
    python scripts/ir_receive.py clip.mp4 --unit 0.4     # match transmit --unit

If --unit is omitted it is estimated from the shortest pulse in the clip.
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

MORSE_INV = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F",
    "--.": "G", "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
    "--": "M", "-.": "N", "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R",
    "...": "S", "-": "T", "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
    "-.--": "Y", "--..": "Z", "-----": "0", ".----": "1", "..---": "2",
    "...--": "3", "....-": "4", ".....": "5", "-....": "6", "--...": "7",
    "---..": "8", "----.": "9",
}


def brightness_series(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vals = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # emitter is a small bright blob; track the top 0.1% of pixels so a
        # small IR source dominates over the rest of the scene
        k = max(1, g.size // 1000)
        vals.append(float(np.partition(g.ravel(), -k)[-k:].mean()))
    cap.release()
    return np.array(vals), fps


def to_onoff(vals: np.ndarray):
    lo, hi = np.percentile(vals, 10), np.percentile(vals, 90)
    if hi - lo < 8:
        raise SystemExit("no clear blink detected (is the emitter in frame?)")
    thresh = (lo + hi) / 2
    return vals > thresh


def runs(mask: np.ndarray):
    out = []
    cur = mask[0]
    n = 1
    for v in mask[1:]:
        if v == cur:
            n += 1
        else:
            out.append((cur, n))
            cur, n = v, 1
    out.append((cur, n))
    return out


def decode(path: str, unit: float | None):
    vals, fps = brightness_series(path)
    mask = to_onoff(vals)
    seq = runs(mask)
    # drop leading/trailing OFF
    while seq and not seq[0][0]:
        seq.pop(0)
    while seq and not seq[-1][0]:
        seq.pop()
    if not seq:
        raise SystemExit("no ON pulses found")

    on_lens = sorted(n for on, n in seq if on)
    if unit is not None:
        unit_frames = unit * fps
    else:
        # shortest ON pulse ~ 1 unit (dot); use the 20th percentile for safety
        unit_frames = on_lens[max(0, len(on_lens) // 5)]
    print(f"video {fps:.1f} fps, unit ~ {unit_frames / fps:.2f}s "
          f"({unit_frames:.1f} frames)\n")

    # drop the sync preamble: a leading ON run longer than any real symbol
    # (dash = 3 units), plus the gap after it
    if seq and seq[0][0] and seq[0][1] / unit_frames >= 4.5:
        seq.pop(0)
        if seq and not seq[0][0]:
            seq.pop(0)

    text, letter = [], ""

    def flush_letter():
        nonlocal letter
        if letter:
            text.append(MORSE_INV.get(letter, "?"))
            letter = ""

    for on, n in seq:
        u = n / unit_frames
        if on:
            letter += "-" if u >= 2 else "."
        else:
            if u >= 5:            # word gap
                flush_letter()
                text.append(" ")
            elif u >= 2:          # letter gap
                flush_letter()
            # else intra-letter gap -> keep building
    flush_letter()
    return "".join(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="recording of the blinking IR emitter")
    ap.add_argument("--unit", type=float, default=None,
                    help="Morse unit in seconds (match transmit; else estimated)")
    args = ap.parse_args()
    print(f"decoded: {decode(args.video, args.unit)!r}")


if __name__ == "__main__":
    sys.exit(main())
