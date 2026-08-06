"""Generate a printable checkerboard for scripts/stereo_calibrate.py.

Outputs an A4 PDF (and PNG) at exact physical scale. Defaults match the
calibrator: 9x6 *inner* corners (=10x7 squares), 25 mm squares.

    python scripts/make_checkerboard.py
    python scripts/make_checkerboard.py --cols 9 --rows 6 --square-mm 25

IMPORTANT: print the PDF at 100% / "Actual size" - NOT "fit to page", or the
squares won't be 25 mm and the calibration scale will be wrong. A 50 mm scale
bar is printed on the sheet; measure it with a ruler to confirm.
"""
from __future__ import annotations

import argparse
import sys

from PIL import Image, ImageDraw, ImageFont

DPI = 300
A4_MM = (210.0, 297.0)          # portrait width x height


def mm2px(mm):
    return int(round(mm / 25.4 * DPI))


def _font(px):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except Exception:
            continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--out", default="captures/checkerboard_A4")
    args = ap.parse_args()

    nx, ny = args.cols + 1, args.rows + 1          # squares = inner corners + 1
    board_mm = (nx * args.square_mm, ny * args.square_mm)

    # choose orientation that fits A4 with margin; prefer the tighter fit
    portrait, landscape = A4_MM, (A4_MM[1], A4_MM[0])
    fits = [p for p in (portrait, landscape)
            if board_mm[0] <= p[0] - 10 and board_mm[1] <= p[1] - 20]
    if not fits:
        print(f"Board {board_mm[0]:.0f}x{board_mm[1]:.0f} mm doesn't fit A4. "
              f"Reduce --square-mm or corner counts.")
        return 1
    page_mm = min(fits, key=lambda p: p[0] * p[1])

    W, H = mm2px(page_mm[0]), mm2px(page_mm[1])
    sq = mm2px(args.square_mm)
    bw, bh = nx * sq, ny * sq
    x0, y0 = (W - bw) // 2, (H - bh) // 2

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                x, y = x0 + i * sq, y0 + j * sq
                d.rectangle([x, y, x + sq, y + sq], fill="black")

    # caption + 50 mm scale bar (to verify print scale)
    f = _font(mm2px(4))
    cap = (f"Checkerboard  {args.cols}x{args.rows} inner corners  |  "
           f"{args.square_mm:.0f} mm squares  |  print at 100% (Actual size)")
    d.text((x0, max(mm2px(4), y0 - mm2px(8))), cap, fill="black", font=f)
    bar_mm, by = 50.0, y0 + bh + mm2px(8)
    d.line([x0, by, x0 + mm2px(bar_mm), by], fill="black", width=mm2px(0.6))
    for xx in (x0, x0 + mm2px(bar_mm)):
        d.line([xx, by - mm2px(2), xx, by + mm2px(2)], fill="black", width=mm2px(0.6))
    d.text((x0, by + mm2px(3)), "50 mm - measure this to confirm scale",
           fill="black", font=f)

    png, pdf = args.out + ".png", args.out + ".pdf"
    img.save(png, dpi=(DPI, DPI))
    img.save(pdf, "PDF", resolution=float(DPI))
    print(f"Board {board_mm[0]:.0f}x{board_mm[1]:.0f} mm on "
          f"{'portrait' if page_mm == portrait else 'landscape'} A4")
    print(f"Saved {pdf}  and  {png}")
    print("Print the PDF at 100% / Actual size, then verify the 50 mm bar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
