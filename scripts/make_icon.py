"""Generate the HelloCam app icon (hellocam/assets/icon.ico).

A camera lens on a dark rounded square: an outer teal ring (the visible / RGB
side, matching the app's aquamarine UI accent) wrapping an inner infrared-red
glow (the IR sensor + emitter). Drawn at 4x and downsampled for clean edges,
then written as a multi-resolution .ico (16-256px) for the window + exe.

    python scripts/make_icon.py
"""
import os

from PIL import Image, ImageDraw, ImageFilter

OUT = os.path.join(os.path.dirname(__file__), "..", "hellocam", "assets", "icon.ico")
S = 1024                      # supersampled working size (downsampled at the end)
TEAL = (127, 255, 212)       # aquamarine - the app's UI accent (RGB/visible side)
IR = (255, 60, 70)           # infrared red glow


def _rounded(draw, box, radius, fill):
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def render(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # dark rounded-square backing tile
    m = int(size * 0.02)
    _rounded(d, (m, m, size - m, size - m), int(size * 0.22), (18, 20, 26, 255))
    _rounded(d, (m, m, size - m, size - m - int(size * 0.02)),
             int(size * 0.22), (26, 29, 38, 255))          # subtle top highlight

    cx = cy = size / 2
    r = size * 0.34

    # soft IR glow behind the lens
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((cx - r * 1.15, cy - r * 1.15, cx + r * 1.15, cy + r * 1.15),
               fill=(*IR, 130))
    glow = glow.filter(ImageFilter.GaussianBlur(size * 0.05))
    img.alpha_composite(glow)

    def ring(rr, width, color):
        d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=color,
                  width=max(1, int(width)))

    # outer teal ring (visible/RGB) -> inner IR-red ring
    ring(r, size * 0.055, (*TEAL, 255))
    ring(r * 0.80, size * 0.030, (*IR, 235))

    # lens body: dark disc with an off-centre specular highlight
    lr = r * 0.66
    d.ellipse((cx - lr, cy - lr, cx + lr, cy + lr), fill=(12, 14, 19, 255))
    d.ellipse((cx - lr * 0.5, cy - lr * 0.5, cx + lr * 0.5, cy + lr * 0.5),
              fill=(30, 34, 44, 255))
    hl = lr * 0.28
    d.ellipse((cx - lr * 0.42 - hl, cy - lr * 0.42 - hl,
               cx - lr * 0.42 + hl, cy - lr * 0.42 + hl), fill=(150, 235, 210, 230))

    # two IR emitter dots at the lower corners
    er = size * 0.035
    for ex in (size * 0.30, size * 0.70):
        ey = size * 0.80
        d.ellipse((ex - er, ey - er, ex + er, ey + er), fill=(*IR, 255))

    return img.resize((size, size), Image.LANCZOS)


def main():
    base = render(S)
    sizes = [16, 32, 48, 64, 128, 256]
    frames = [base.resize((s, s), Image.LANCZOS) for s in sizes]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    frames[-1].save(OUT, format="ICO",
                    sizes=[(s, s) for s in sizes], append_images=frames[:-1])
    # also drop a PNG preview next to it (gitignored) for eyeballing
    base.resize((256, 256), Image.LANCZOS).save(
        os.path.join(os.path.dirname(OUT), "icon_preview.png"))
    print(f"wrote {os.path.normpath(OUT)}  ({', '.join(str(s) for s in sizes)}px)")


if __name__ == "__main__":
    main()
