"""One-time stereo calibration of the Windows Hello RGB + IR pair.

This is the "true quality" depth path: with a proper checkerboard calibration
we can rectify the two views so epipolar lines are horizontal and run real
stereo matching (metric-ish depth), instead of the uncalibrated flow estimate.

You need a printed **checkerboard** (default 9x6 *inner* corners). Print one,
mount it flat (on cardboard), and hold it in front of the camera. The script
grabs pairs automatically whenever the board is detected in BOTH the color and
IR views, from varied angles/distances. Aim for ~20 good pairs filling the frame.

    python scripts/stereo_calibrate.py                 # 9x6, 25mm squares
    python scripts/stereo_calibrate.py --cols 9 --rows 6 --square-mm 25

Output: captures/stereo_calib.npz  (rectification maps + Q). The app's
Depth method > Calibrated then uses it.

Note: matte paper reads well in IR under the emitter; glossy paper may glare.
If the IR view can't find the board, dim the room (less ambient washout) or
tilt to avoid the emitter hot-spot.
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit("scripts", 1)[0])
from hellocam import HelloCameras          # noqa: E402

CALIB_W, CALIB_H = 640, 480                     # common resolution for both cams


def _find(gray, pattern):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if ok:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    return ok, corners


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--need", type=int, default=20, help="pairs to collect")
    ap.add_argument("--out", default="captures/stereo_calib.npz")
    args = ap.parse_args()
    pattern = (args.cols, args.rows)

    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_mm

    objpoints, cpts, ipts = [], [], []
    cams = HelloCameras(color=True, ir=True)
    cams.open(); cams.start_pump()
    print(f"Collecting {args.need} checkerboard pairs ({args.cols}x{args.rows}). "
          f"Move the board around; Ctrl+C to stop early.\n")
    last_t = 0.0
    try:
        while len(objpoints) < args.need:
            c = cams.latest_color(); ir = cams.latest_ir()
            if c is None or ir is None:
                time.sleep(0.03); continue
            cg = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), (CALIB_W, CALIB_H))
            ig = cv2.resize(ir, (CALIB_W, CALIB_H))
            ig = cv2.createCLAHE(2.0, (8, 8)).apply(
                cv2.normalize(ig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
            okc, cc = _find(cg, pattern)
            oki, ci = _find(ig, pattern)
            if okc and oki and time.time() - last_t > 0.8:
                objpoints.append(objp.copy()); cpts.append(cc); ipts.append(ci)
                last_t = time.time()
                print(f"  captured pair {len(objpoints)}/{args.need}", flush=True)
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nstopped early")
    finally:
        cams.close()

    if len(objpoints) < 6:
        print(f"Only {len(objpoints)} pairs - need >=6. Aborting.")
        return 1

    size = (CALIB_W, CALIB_H)
    print("\nCalibrating intrinsics...")
    _, Kc, dc, _, _ = cv2.calibrateCamera(objpoints, cpts, size, None, None)
    _, Ki, di, _, _ = cv2.calibrateCamera(objpoints, ipts, size, None, None)
    print("Stereo calibrating...")
    flags = cv2.CALIB_FIX_INTRINSIC
    rms, Kc, dc, Ki, di, R, T, _, _ = cv2.stereoCalibrate(
        objpoints, cpts, ipts, Kc, dc, Ki, di, size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=flags)
    print(f"  stereo RMS reprojection error: {rms:.3f} px "
          f"(<1 great, 1-2 ok, >3 redo)")

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        Kc, dc, Ki, di, size, R, T, alpha=0)
    mcx, mcy = cv2.initUndistortRectifyMap(Kc, dc, R1, P1, size, cv2.CV_32FC1)
    mix, miy = cv2.initUndistortRectifyMap(Ki, di, R2, P2, size, cv2.CV_32FC1)

    np.savez(args.out,
             calib_w=CALIB_W, calib_h=CALIB_H,
             map_color_x=mcx, map_color_y=mcy,
             map_ir_x=mix, map_ir_y=miy, Q=Q,
             num_disparities=96, block_size=7, rms=rms)
    print(f"\nSaved {args.out}. In the app: Depth method > Calibrated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
