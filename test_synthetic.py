"""
End-to-end smoke test that runs the full perception + controller pipeline
on a synthetic camera frame. No camera or drone required.

Renders a fake "overhead view" with 4 ArUco markers, a green LED, 2 red
rotors, and a blue pad, then asserts that:
  - workspace is detected,
  - drone position (LED), heading (rotors), and pad are all found,
  - the controller produces a sane body-frame command toward the pad.

Run from the repo root:

    python test_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from poa.config import Config
from poa.controller import LandingController
from poa.overlay import build_overlay
from poa.perception import detect_in_workspace, detect_workspace, make_aruco_detector


def render_synthetic_frame() -> tuple[np.ndarray, dict]:
    """Synthetic 1280x720 overhead view with markers, drone, rotors, pad."""
    img = np.full((720, 1280, 3), 240, dtype=np.uint8)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 100

    marker_centers = {
        0: (150, 100),    # TL
        1: (1130, 100),   # TR
        2: (1130, 620),   # BR
        3: (150, 620),    # BL
    }
    for mid, (cx, cy) in marker_centers.items():
        marker = cv2.aruco.generateImageMarker(dictionary, mid, marker_size)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        x = cx - marker_size // 2
        y = cy - marker_size // 2
        img[y:y + marker_size, x:x + marker_size] = marker_bgr

    # Drone at the center, "facing" image-down (toward the BL/BR side)
    drone_xy = (640, 360)
    cv2.circle(img, drone_xy, 20, (0, 255, 0), -1)  # green LED

    # Rotors in front of LED (front = +y direction in image)
    nose_dy = 50
    rotor_dx = 35
    rotor_l = (drone_xy[0] - rotor_dx, drone_xy[1] + nose_dy)
    rotor_r = (drone_xy[0] + rotor_dx, drone_xy[1] + nose_dy)
    cv2.circle(img, rotor_l, 12, (0, 0, 255), -1)
    cv2.circle(img, rotor_r, 12, (0, 0, 255), -1)

    # Light-blue pad in the lower-left quadrant
    pad_xy = (320, 500)
    cv2.rectangle(img, (pad_xy[0] - 50, pad_xy[1] - 50),
                  (pad_xy[0] + 50, pad_xy[1] + 50),
                  (250, 206, 135), -1)  # BGR for sky-blue (~RGB 135,206,250)

    return img, {
        "marker_centers": marker_centers,
        "drone_xy_raw": drone_xy,
        "pad_xy_raw": pad_xy,
        "rotor_l_raw": rotor_l,
        "rotor_r_raw": rotor_r,
    }


def main() -> int:
    raw, gt = render_synthetic_frame()
    out_dir = Path("_tmp")
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / "synthetic_raw.png"), raw)

    cfg = Config()
    detector = make_aruco_detector()

    ws = detect_workspace(raw, detector, cfg)
    assert ws is not None, "workspace not detected (ArUco failure)"
    print(f"[ok] workspace detected, marker centers: "
          f"{ {k: (round(x, 1), round(y, 1)) for k, (x, y) in ws.raw_marker_centers.items()} }")

    det, masks = detect_in_workspace(ws.rectified, cfg)
    assert det.led_xy_mm is not None, "LED not detected"
    assert len(det.rotors_xy_mm) == 2, f"expected 2 rotors, got {len(det.rotors_xy_mm)}"
    assert det.pad_xy_mm is not None, "pad not detected"
    assert det.heading is not None, "heading not computed"
    print(f"[ok] LED   = ({det.led_xy_mm[0]:7.1f}, {det.led_xy_mm[1]:7.1f}) mm")
    print(f"[ok] nose  = ({det.nose_mm[0]:7.1f}, {det.nose_mm[1]:7.1f}) mm")
    print(f"[ok] pad   = ({det.pad_xy_mm[0]:7.1f}, {det.pad_xy_mm[1]:7.1f}) mm")
    print(f"[ok] head  = ({det.heading[0]:+.3f}, {det.heading[1]:+.3f})")

    # Drone is "facing" image-down in the raw frame; after rectification the
    # heading should also be roughly (0, +1) in workspace mm.
    assert det.heading[1] > 0.9, f"expected heading roughly (0, +1), got {det.heading}"
    assert abs(det.heading[0]) < 0.2, f"heading x should be near 0, got {det.heading[0]}"

    controller = LandingController(cfg.controller)
    ctrl = controller.step(det)
    print(f"[ok] distance = {ctrl.distance_mm:6.1f} mm   "
          f"err_body = (fwd {ctrl.error_body_mm[0]:+.1f}, right {ctrl.error_body_mm[1]:+.1f})")
    print(f"[ok] command  : roll={ctrl.command.roll:+d}  pitch={ctrl.command.pitch:+d}  "
          f"yaw={ctrl.command.yaw:+d}  throttle={ctrl.command.throttle:+d}")

    # Pad is lower-left in image; drone faces down, so:
    #   forward (image-down direction) should be POSITIVE -> +pitch
    #   right (rotated 90deg CW = image-left) should be POSITIVE -> +roll
    assert ctrl.error_body_mm[0] > 0, f"expected pad forward of drone, got {ctrl.error_body_mm[0]}"
    assert ctrl.error_body_mm[1] > 0, f"expected pad to drone's right, got {ctrl.error_body_mm[1]}"
    assert ctrl.command.pitch > 0, f"expected +pitch, got {ctrl.command.pitch}"
    assert ctrl.command.roll > 0, f"expected +roll, got {ctrl.command.roll}"

    overlay = build_overlay(ws.rectified, det, ctrl, cfg, fps=30.0, state="TRACKING",
                            extras=["synthetic test"])
    cv2.imwrite(str(out_dir / "synthetic_rectified.png"), overlay)
    print(f"[ok] saved overlay to {out_dir / 'synthetic_rectified.png'}")

    print("\nAll synthetic-frame assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
