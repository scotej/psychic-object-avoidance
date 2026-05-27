"""
HSV calibration tool. Pick a target (led / rotors / pad), drag the sliders
until just that thing is white in the mask preview, then press 's' to save.
The output JSON is loaded by land.py via --hsv.

Set H_min > H_max to wrap around the hue circle (this is how red gets the
low-H 0..10 and high-H 170..180 ranges combined into one target).

Keys:
    n       cycle target  (led -> rotors -> pad -> led ...)
    s       save current HSV settings to --out
    r       reset current target to its built-in default
    p       print current HSV settings to the terminal
    q/ESC   quit

Usage:
    python calibrate_hsv.py                          # camera 0, defaults
    python calibrate_hsv.py --camera 1 --out custom.json
    python calibrate_hsv.py --width-mm 1500 --height-mm 1500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from poa.config import (
    Config,
    ColorTarget,
    HSVRange,
    _led_default,
    _rotors_default,
    _pad_default,
    load_hsv,
    save_hsv,
)
from poa.perception import build_mask, detect_workspace, make_aruco_detector


WINDOW = "calibrate_hsv  (n=next target  s=save  r=reset  p=print  q=quit)"


def _ranges_from_trackbars(h_lo: int, h_hi: int, s_lo: int, s_hi: int,
                           v_lo: int, v_hi: int) -> list[HSVRange]:
    """If h_lo <= h_hi, single range. Otherwise wrap into two ranges
    (0..h_hi) and (h_lo..180) so we cover the H circle."""
    if h_lo <= h_hi:
        return [HSVRange(h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)]
    return [
        HSVRange(0, h_hi, s_lo, s_hi, v_lo, v_hi),
        HSVRange(h_lo, 180, s_lo, s_hi, v_lo, v_hi),
    ]


def _trackbars_from_target(target: ColorTarget) -> dict[str, int]:
    if not target.ranges:
        return {"H_min": 0, "H_max": 180, "S_min": 0, "S_max": 255,
                "V_min": 0, "V_max": 255, "min_area": target.min_area_px}
    if len(target.ranges) == 1:
        r = target.ranges[0]
        h_lo, h_hi = r.h_lo, r.h_hi
    else:
        # Wrap: low-H range first by our convention, but be defensive
        r1, r2 = sorted(target.ranges, key=lambda r: r.h_lo)
        h_lo = r2.h_lo  # upper wrap start
        h_hi = r1.h_hi  # lower wrap end
    r = target.ranges[0]
    return {
        "H_min": h_lo, "H_max": h_hi,
        "S_min": r.s_lo, "S_max": r.s_hi,
        "V_min": r.v_lo, "V_max": r.v_hi,
        "min_area": target.min_area_px,
    }


def _push_trackbars(window: str, vals: dict[str, int]) -> None:
    for k, v in vals.items():
        cv2.setTrackbarPos(k, window, int(v))


def _pull_trackbars(window: str) -> dict[str, int]:
    return {
        "H_min": cv2.getTrackbarPos("H_min", window),
        "H_max": cv2.getTrackbarPos("H_max", window),
        "S_min": cv2.getTrackbarPos("S_min", window),
        "S_max": cv2.getTrackbarPos("S_max", window),
        "V_min": cv2.getTrackbarPos("V_min", window),
        "V_max": cv2.getTrackbarPos("V_max", window),
        "min_area": cv2.getTrackbarPos("min_area", window),
    }


def _apply_trackbars(target: ColorTarget, vals: dict[str, int]) -> None:
    target.ranges = _ranges_from_trackbars(
        vals["H_min"], vals["H_max"], vals["S_min"], vals["S_max"],
        vals["V_min"], vals["V_max"],
    )
    target.min_area_px = max(1, vals["min_area"])


def _scale_for_display(img: np.ndarray, max_h: int = 600) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= max_h:
        return img
    s = max_h / h
    return cv2.resize(img, (int(round(w * s)), max_h))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width-mm", type=float, default=1500.0)
    parser.add_argument("--height-mm", type=float, default=1500.0)
    parser.add_argument("--px-per-mm", type=float, default=1.0)
    parser.add_argument("--in", dest="in_path", type=Path, default=None,
                        help="Load starting HSV from this JSON.")
    parser.add_argument("--out", type=Path, default=Path("hsv_config.json"),
                        help="Save HSV here when 's' is pressed.")
    args = parser.parse_args()

    cfg = Config()
    cfg.workspace.width_mm = args.width_mm
    cfg.workspace.height_mm = args.height_mm
    cfg.workspace.px_per_mm = args.px_per_mm
    if args.in_path is not None and args.in_path.exists():
        cfg = load_hsv(args.in_path, cfg)
        print(f"[calib] loaded starting HSV from {args.in_path}")

    targets: list[tuple[str, ColorTarget, callable]] = [
        ("led",    cfg.led,    _led_default),
        ("rotors", cfg.rotors, _rotors_default),
        ("pad",    cfg.pad,    _pad_default),
    ]
    target_ix = 0

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    for name, hi in (("H_min", 180), ("H_max", 180),
                     ("S_min", 255), ("S_max", 255),
                     ("V_min", 255), ("V_max", 255),
                     ("min_area", 2000)):
        cv2.createTrackbar(name, WINDOW, 0, hi, lambda _v: None)
    _push_trackbars(WINDOW, _trackbars_from_target(targets[target_ix][1]))

    detector = make_aruco_detector()

    def status(extra: str = "") -> str:
        name = targets[target_ix][0]
        return f"target={name}   ranges={len(targets[target_ix][1].ranges)}   {extra}"

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        name, target, defaulter = targets[target_ix]
        _apply_trackbars(target, _pull_trackbars(WINDOW))

        ws, _visible_ids = detect_workspace(frame, detector, cfg)
        if ws is not None:
            base = ws.rectified
            base_label = "workspace (rectified)"
        else:
            base = frame
            base_label = "raw (no workspace - tuning against camera)"

        hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
        mask = build_mask(hsv, target)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Make base + mask_bgr the same height for side-by-side
        h = min(base.shape[0], mask_bgr.shape[0], 600)
        base_d = _scale_for_display(base, max_h=h)
        mask_d = _scale_for_display(mask_bgr, max_h=h)
        # Pad widths if needed (shouldn't be, since heights match)
        if base_d.shape[0] != mask_d.shape[0]:
            mask_d = cv2.resize(mask_d, (mask_d.shape[1], base_d.shape[0]))
        display = cv2.hconcat([base_d, mask_d])

        cv2.putText(display, f"{base_label}     |     mask  ({status()})",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, f"{base_label}     |     mask  ({status()})",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("n"):
            target_ix = (target_ix + 1) % len(targets)
            _push_trackbars(WINDOW, _trackbars_from_target(targets[target_ix][1]))
            print(f"[calib] switched to '{targets[target_ix][0]}'")
        elif key == ord("r"):
            default_target = defaulter()
            target.ranges = default_target.ranges
            target.min_area_px = default_target.min_area_px
            target.expected_blobs = default_target.expected_blobs
            target.name = default_target.name
            _push_trackbars(WINDOW, _trackbars_from_target(target))
            print(f"[calib] reset '{name}' to defaults")
        elif key == ord("s"):
            save_hsv(args.out, cfg)
            print(f"[calib] saved HSV -> {args.out}")
        elif key == ord("p"):
            print(f"[calib] {name}: ranges={target.ranges}  min_area={target.min_area_px}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
