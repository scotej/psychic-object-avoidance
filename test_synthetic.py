"""
End-to-end smoke test on a synthetic frame. No camera or drone required.

Renders a fake overhead view with the four corner markers, the drone marker
(rotated so it "faces" image-down), and the pad marker, then checks that:
  - the workspace is detected,
  - the drone and pad are found,
  - the recovered heading matches the way the drone marker is pointing,
  - the controller produces a sane body-frame command toward the pad.

Run from the repo root:

    python test_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from poa.config import CORNER_IDS, DRONE_ID, PAD_ID, Config
from poa.controller import LandingController
from poa.overlay import build_overlay
from poa.perception import detect, make_aruco_detector


def _paste(img: np.ndarray, marker: np.ndarray, center: tuple[int, int]) -> None:
    side = marker.shape[0]
    x = center[0] - side // 2
    y = center[1] - side // 2
    img[y:y + side, x:x + side] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)


def render_synthetic_frame() -> tuple[np.ndarray, dict]:
    """1280x720 overhead view: 4 corners, drone marker (facing down), pad marker."""
    img = np.full((720, 1280, 3), 240, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    corner_centers = {
        CORNER_IDS[0]: (150, 100),    # TL
        CORNER_IDS[1]: (1130, 100),   # TR
        CORNER_IDS[2]: (1130, 620),   # BR
        CORNER_IDS[3]: (150, 620),    # BL
    }
    for mid, center in corner_centers.items():
        _paste(img, cv2.aruco.generateImageMarker(dictionary, mid, 100), center)

    # Drone marker at the centre, rotated 180 deg so its top edge (= forward)
    # points toward image-down, i.e. toward the BL/BR side.
    drone_center = (640, 360)
    drone = cv2.rotate(cv2.aruco.generateImageMarker(dictionary, DRONE_ID, 120), cv2.ROTATE_180)
    _paste(img, drone, drone_center)

    # Pad marker in the lower-left quadrant.
    pad_center = (340, 500)
    _paste(img, cv2.aruco.generateImageMarker(dictionary, PAD_ID, 100), pad_center)

    return img, {"drone_center": drone_center, "pad_center": pad_center}


def main() -> int:
    raw, _ = render_synthetic_frame()
    out_dir = Path("_tmp")
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / "synthetic_raw.png"), raw)

    cfg = Config()
    detector = make_aruco_detector()

    ws, det, visible_ids = detect(raw, detector, cfg)
    assert ws is not None, f"workspace not detected (visible_ids={visible_ids})"
    assert det.have_drone, f"drone marker not detected (visible_ids={visible_ids})"
    assert det.have_pad, f"pad marker not detected (visible_ids={visible_ids})"
    print(f"[ok] workspace detected; markers seen: {visible_ids}")
    print(f"[ok] drone = ({det.drone.center_mm[0]:6.1f}, {det.drone.center_mm[1]:6.1f}) mm")
    print(f"[ok] pad   = ({det.pad.center_mm[0]:6.1f}, {det.pad.center_mm[1]:6.1f}) mm")
    print(f"[ok] head  = ({det.drone.forward[0]:+.3f}, {det.drone.forward[1]:+.3f})")

    # The drone marker faces image-down, so after rectification the heading
    # should be roughly (0, +1).
    assert det.drone.forward[1] > 0.9, f"expected heading ~ (0, +1), got {det.drone.forward}"
    assert abs(det.drone.forward[0]) < 0.2, f"heading x should be near 0, got {det.drone.forward[0]}"

    controller = LandingController(cfg.controller)
    ctrl = controller.step(det)
    print(f"[ok] distance = {ctrl.distance_mm:6.1f} mm   "
          f"err_body = (fwd {ctrl.error_body_mm[0]:+.1f}, right {ctrl.error_body_mm[1]:+.1f})")
    print(f"[ok] command  : roll={ctrl.command.roll:+d}  pitch={ctrl.command.pitch:+d}  "
          f"yaw={ctrl.command.yaw:+d}  throttle={ctrl.command.throttle:+d}")

    # Pad is lower-left in the image while the drone faces down, so:
    #   forward (image-down)         should be positive -> +pitch
    #   right   (down rotated 90 CW = image-left) should be positive -> +roll
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
