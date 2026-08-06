"""Download ONNX depth model(s) for the ML depth mode.

    python scripts/download_model.py                 # all models
    python scripts/download_model.py midas_small      # just one
"""
import sys

sys.path.insert(0, __file__.rsplit("scripts", 1)[0])
from surfacecam import mldepth               # noqa: E402


def main():
    keys = sys.argv[1:] or list(mldepth.MODELS)
    for key in keys:
        if key not in mldepth.MODELS:
            print(f"unknown model '{key}'. options: {list(mldepth.MODELS)}")
            continue
        print(f"Downloading {key}: {mldepth.MODELS[key]['url']}")
        path = mldepth.download_model(key)
        print(f"  saved {path}")
    print("Done. In the app: Depth method > ML, and View > ML depth model.")


if __name__ == "__main__":
    main()
