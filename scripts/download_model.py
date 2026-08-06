"""Download the MiDaS-small ONNX model for the ML depth mode (~66 MB).

    python scripts/download_model.py
"""
import sys

sys.path.insert(0, __file__.rsplit("scripts", 1)[0])
from surfacecam import mldepth               # noqa: E402


def main():
    print(f"Downloading {mldepth.MODEL_URL}")
    path = mldepth.download_model()
    print(f"Saved {path}. In the app: Depth method > ML (monocular).")


if __name__ == "__main__":
    main()
