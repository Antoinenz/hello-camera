"""HelloCam - view and fuse the Windows Hello RGB + IR cameras.

Examples:
    python main.py                 # launch the live multi-mode viewer
    python main.py --selftest      # open both cameras, save sample frames, exit
    python main.py --enumerate     # list media frame sources and quit
"""
import argparse
import sys


def cmd_enumerate():
    import asyncio
    from winsdk.windows.media.capture.frames import MediaFrameSourceGroup

    async def _run():
        groups = await MediaFrameSourceGroup.find_all_async()
        print(f"Found {len(groups)} media frame source group(s)\n")
        for g in groups:
            print(f"GROUP: {g.display_name}")
            for si in g.source_infos:
                print(f"   - kind={si.source_kind.name:9s} "
                      f"stream={si.media_stream_type.name}")
            print()
    asyncio.run(_run())


def cmd_check():
    """Import the heavy modules and exit. Used by CI to confirm a packaged
    build's import graph is intact, without needing a camera or Media
    Foundation (which isn't present on headless Windows Server runners)."""
    import cv2                                              # noqa: F401
    import numpy                                            # noqa: F401
    from winsdk.windows.media.capture.frames import (       # noqa: F401
        MediaFrameSourceGroup)
    import hellocam                                         # noqa: F401
    try:
        import onnxruntime                                  # noqa: F401
        ort = "yes"
    except Exception:
        ort = "no"
    print(f"imports OK (onnxruntime available: {ort})", flush=True)
    return 0


def cmd_selftest():
    """Open both cameras, grab a few frames, save PNGs. Headless (no window)."""
    import os
    import time
    import cv2
    from hellocam import HelloCameras
    from hellocam import processing as P

    out = "captures"
    os.makedirs(out, exist_ok=True)
    cams = HelloCameras(color=True, ir=True)
    print("Opening cameras...", flush=True)
    cams.open()
    print(f"  color size: {cams.color_size}", flush=True)
    print(f"  ir size:    {cams.ir_size}", flush=True)

    color = ir = None
    # give the pipeline a moment; grab the first non-empty frame of each
    deadline = time.time() + 8
    while time.time() < deadline and (color is None or ir is None):
        color = cams.read_color() if color is None else color
        ir = cams.read_ir() if ir is None else ir
        time.sleep(0.05)

    ok = True
    if color is not None:
        cv2.imwrite(f"{out}/selftest_color.png", color)
        print(f"  wrote {out}/selftest_color.png  shape={color.shape}", flush=True)
    else:
        print("  !! no color frame", flush=True); ok = False
    if ir is not None:
        cv2.imwrite(f"{out}/selftest_ir.png", P.ir_view(ir))
        cv2.imwrite(f"{out}/selftest_proximity.png", P.proximity_map(ir))
        print(f"  wrote {out}/selftest_ir.png  shape={ir.shape}", flush=True)
    else:
        print("  !! no ir frame", flush=True); ok = False

    if color is not None and ir is not None:
        aligned = P.Aligner().warp_ir_to_color(ir, color.shape)
        cv2.imwrite(f"{out}/selftest_fuse.png", P.fuse(color, aligned, 0.5))
        cv2.imwrite(f"{out}/selftest_side.png", P.side_by_side(color, P.proximity_map(ir)))
        print(f"  wrote {out}/selftest_fuse.png and selftest_side.png", flush=True)

    cams.close()
    print("Self-test OK." if ok else "Self-test finished WITH MISSING FRAMES.",
          flush=True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="open cameras, save sample frames, exit (no GUI)")
    ap.add_argument("--enumerate", action="store_true",
                    help="list media frame sources and exit")
    ap.add_argument("--check", action="store_true",
                    help="import core modules and exit (build sanity check)")
    ap.add_argument("--cv", action="store_true",
                    help="use the legacy OpenCV window instead of the native GUI")
    args = ap.parse_args()

    if args.check:
        return cmd_check()
    if args.enumerate:
        cmd_enumerate()
        return 0
    if args.selftest:
        return cmd_selftest()

    if args.cv:
        from hellocam.app import run
    else:
        from hellocam.gui import run
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
