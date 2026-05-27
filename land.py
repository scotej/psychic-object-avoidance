"""
Vision-guided autonomous landing for a CoDrone EDU.

Per-frame pipeline:
    raw frame -> 4-corner ArUco -> rectified workspace
                                -> LED blob (drone position)
                                -> 2 red rotor blobs (-> nose midpoint -> heading)
                                -> blue pad blob (landing target)
    -> body-frame error -> proportional roll/pitch
    -> drone.sendControl()
    -> overlay window

Safety:
  - Pre-flight phase: takeoff only after operator presses SPACE *and* perception
    is currently locked on (LED + 2 rotors + pad all visible).
  - Detection loss for N frames -> auto-land.
  - Drone outside workspace bounds for N frames -> auto-land.
  - 'l' during flight -> immediate land.
  - 'q' / ESC anywhere -> land then quit.
  - Any unhandled exception -> emergency_stop via DroneIO context manager.

Usage:
    python land.py                            # camera 0, fly
    python land.py --no-fly                   # perception only, no drone
    python land.py --hsv hsv_config.json
    python land.py --width-mm 1500 --height-mm 1500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from poa.config import Config, load_hsv
from poa.controller import LandingController
from poa.drone_io import DroneIO
from poa.overlay import build_overlay, fit_to_screen
from poa.perception import detect_in_workspace, detect_workspace, make_aruco_detector


WINDOW = "land (SPACE=takeoff  l=land  q=quit)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--camera", type=int, default=0,
                   help="Camera index (default: 0 = laptop webcam)")
    p.add_argument("--no-fly", action="store_true",
                   help="Skip all drone commands; perception loop only.")
    p.add_argument("--hsv", type=Path, default=Path("hsv_config.json"),
                   help="HSV ranges JSON written by calibrate_hsv.py")
    p.add_argument("--width-mm", type=float, default=1500.0,
                   help="Workspace width in mm (center-to-center of TL/TR markers)")
    p.add_argument("--height-mm", type=float, default=1500.0,
                   help="Workspace height in mm (center-to-center of TL/BL markers)")
    p.add_argument("--px-per-mm", type=float, default=1.0,
                   help="Rectified workspace resolution in pixels per mm")
    p.add_argument("--port", type=str, default=None,
                   help="codrone-edu serial port (default: auto-detect)")
    return p.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.workspace.width_mm = args.width_mm
    cfg.workspace.height_mm = args.height_mm
    cfg.workspace.px_per_mm = args.px_per_mm
    if args.hsv.exists():
        load_hsv(args.hsv, cfg)
        print(f"[land] loaded HSV from {args.hsv}")
    else:
        print(f"[land] WARNING: {args.hsv} not found - using built-in HSV defaults. "
              "Run calibrate_hsv.py first for reliable detection.")
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}.")
    print(f"[land] camera {args.camera} opened "
          f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")

    detector = make_aruco_detector()
    controller = LandingController(cfg.controller)

    state = "PREFLIGHT"  # PREFLIGHT -> TRACKING -> LANDING -> DONE
    consecutive_misses = 0
    consecutive_oob = 0
    fps = 0.0
    last_t = time.time()

    with DroneIO(enabled=not args.no_fly, port=args.port) as drone:
        drone.set_led(*cfg.led_rgb, brightness=cfg.led_brightness)

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                now = time.time()
                dt = max(now - last_t, 1e-6)
                last_t = now
                inst = 1.0 / dt
                fps = 0.9 * fps + 0.1 * inst if fps > 0 else inst

                ws, visible_ids = detect_workspace(frame, detector, cfg)
                det = None
                ctrl = None
                base = None
                if ws is not None:
                    base = ws.rectified
                    det, _ = detect_in_workspace(ws.rectified, cfg)
                    ctrl = controller.step(det)

                extras: list[str] = []

                if state == "PREFLIGHT":
                    ready = det is not None and det.have_drone and det.have_pad
                    extras.append("READY for takeoff (SPACE)" if ready
                                  else "waiting for LED + 2 rotors + pad in frame")

                elif state == "TRACKING":
                    if det is None or not det.have_drone:
                        consecutive_misses += 1
                        if drone.airborne:
                            drone.hover()
                        if consecutive_misses >= cfg.safety.max_consecutive_misses:
                            print(f"[land] drone lost for {consecutive_misses} frames -> landing")
                            state = "LANDING"
                    else:
                        consecutive_misses = 0
                        lx, ly = det.led_xy_mm
                        margin = cfg.safety.out_of_bounds_margin_mm
                        oob = (lx < -margin or lx > cfg.workspace.width_mm + margin or
                               ly < -margin or ly > cfg.workspace.height_mm + margin)
                        if oob:
                            consecutive_oob += 1
                            if drone.airborne:
                                drone.hover()
                            if consecutive_oob >= cfg.safety.out_of_bounds_dwell_frames:
                                print(f"[land] drone outside workspace ({lx:.0f},{ly:.0f}) "
                                      f"-> landing")
                                state = "LANDING"
                        else:
                            consecutive_oob = 0
                            if ctrl is not None:
                                if ctrl.over_pad:
                                    print(f"[land] over pad (dist={ctrl.distance_mm:.0f}mm) -> landing")
                                    state = "LANDING"
                                else:
                                    drone.send(ctrl.command.roll, ctrl.command.pitch,
                                               ctrl.command.yaw, ctrl.command.throttle)
                    extras.append(f"misses={consecutive_misses}/{cfg.safety.max_consecutive_misses}  "
                                  f"oob={consecutive_oob}/{cfg.safety.out_of_bounds_dwell_frames}")

                elif state == "LANDING":
                    if drone.airborne:
                        drone.land()
                    state = "DONE"
                    extras.append("LANDED")

                display = build_overlay(
                    base, det, ctrl, cfg, fps, state,
                    extras=extras, raw_bgr=frame, detected_marker_ids=visible_ids,
                )
                display = fit_to_screen(display, max_dim=900)
                cv2.imshow(WINDOW, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[land] quit -> landing if airborne")
                    if drone.airborne:
                        state = "LANDING"
                    else:
                        break
                elif key == 32 and state == "PREFLIGHT":  # SPACE
                    if det is None or not det.have_drone or not det.have_pad:
                        print("[land] cannot takeoff: need LED + 2 rotors + pad visible")
                        continue
                    print("[land] taking off...")
                    try:
                        drone.takeoff()
                    except Exception as e:
                        print(f"[land] takeoff failed: {e}")
                        break
                    controller.reset()
                    consecutive_misses = 0
                    consecutive_oob = 0
                    state = "TRACKING"
                elif key == ord("l") and state == "TRACKING":
                    print("[land] manual land requested")
                    state = "LANDING"

                if state == "DONE":
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
