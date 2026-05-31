"""
Probe every camera index with every common Windows backend, grab one
frame from each that opens, and save it as a thumbnail so you can tell
which index is which physical camera.

Usage:
    python list_cameras.py             # writes _tmp/cam_<backend>_<idx>.png
"""

from __future__ import annotations

from pathlib import Path

import cv2

BACKENDS = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("ANY", cv2.CAP_ANY),
]


def main() -> None:
    out_dir = Path("_tmp")
    out_dir.mkdir(exist_ok=True)

    print(f"Probing indices 0..7 with each backend (saving thumbnails to {out_dir}/)")
    print("-" * 70)

    for name, backend in BACKENDS:
        for idx in range(8):
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if ok and frame is not None:
                path = out_dir / f"cam_{name}_idx{idx}.png"
                cv2.imwrite(str(path), frame)
                print(f"  [{name:5s} idx={idx}]  OK   {w}x{h}   -> {path}")
            else:
                print(f"  [{name:5s} idx={idx}]  opened but no frame")
            cap.release()

    print("-" * 70)
    print(f"Open the PNGs in {out_dir}/ to identify which index is which camera.")
    print("Then run: python land.py --camera <idx>    (land.py currently uses the DSHOW backend)")


if __name__ == "__main__":
    main()
