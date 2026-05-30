"""
Vision-guided autonomous landing for a CoDrone EDU.

Per-frame pipeline:
    raw frame -> 4 corner ArUco markers -> rectified workspace
              -> drone marker  (-> position + heading)
              -> pad marker    (-> landing target)
    -> body-frame error -> proportional roll/pitch
    -> drone.sendControl()
    -> one overlay window

Marker IDs (see generate_markers.py): 0-3 are the workspace corners,
4 is stuck on top of the drone (top edge pointing at the nose), and 5 marks
the landing pad on the floor.

Safety:
  - Pre-flight: takeoff only once the operator presses SPACE *and* both the
    drone and pad markers are currently in view.
  - Drone marker lost for N frames    -> land.
  - Camera stops delivering frames    -> land.
  - Drone outside the workspace bounds -> land.
  - 'l' during flight -> land now.
  - 'q' / ESC anywhere -> land then quit.
  - Any unhandled exception -> emergency_stop via the DroneIO context manager.

Usage:
    python land.py                       # camera 0, fly
    python land.py --no-fly              # perception only, no drone
    python land.py --camera 1
    python land.py --width-mm 300 --height-mm 300
"""

from __future__ import annotations

import argparse
import time

import cv2

from poa.config import Config
from poa.controller import LandingController
from poa.drone_io import DroneIO
from poa.overlay import build_overlay, fit_to_screen
from poa.perception import detect, make_aruco_detector


WINDOW = "land  (SPACE=takeoff  l=land  q=quit)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--camera", type=int, default=0,
                   help="Camera index (default: 0)")
    p.add_argument("--no-fly", action="store_true",
                   help="Skip every drone command; run the perception loop only.")
    p.add_argument("--width-mm", type=float, default=300.0,
                   help="Workspace width in mm (TL->TR marker centres)")
    p.add_argument("--height-mm", type=float, default=300.0,
                   help="Workspace height in mm (TL->BL marker centres)")
    p.add_argument("--px-per-mm", type=float, default=2.0,
                   help="Rectified workspace resolution in pixels per mm")
    p.add_argument("--port", type=str, default=None,
                   help="codrone-edu serial port (default: auto-detect)")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.workspace.width_mm = args.width_mm
    cfg.workspace.height_mm = args.height_mm
    cfg.workspace.px_per_mm = args.px_per_mm
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}.")
    print(f"[land] camera {args.camera} opened "
          f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")

    detector = make_aruco_detector()
    controller = LandingController(cfg.controller)

    state = "PREFLIGHT"  # PREFLIGHT -> TRACKING -> (land and exit)
    misses = 0      # frames with the workspace up but the drone marker missing
    oob = 0         # consecutive frames the drone has been out of bounds
    read_fails = 0  # consecutive failed camera reads
    fps = 0.0
    last_t = time.time()

    with DroneIO(enabled=not args.no_fly, port=args.port) as drone:
        # Landing is done inline rather than via a deferred state so it never
        # depends on the next (possibly failing) camera read happening.
        def land_now(reason: str) -> None:
            print(f"[land] {reason}")
            if drone.airborne:
                drone.land()

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    read_fails += 1
                    if drone.airborne:
                        drone.hover()
                        if read_fails >= cfg.safety.max_consecutive_misses:
                            land_now(f"camera stalled for {read_fails} frames -> landing")
                            break
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        land_now("quit requested -> landing")
                        break
                    continue
                read_fails = 0

                now = time.time()
                dt = max(now - last_t, 1e-6)
                last_t = now
                fps = 0.9 * fps + 0.1 / dt if fps > 0 else 1.0 / dt

                ws, det, visible_ids = detect(frame, detector, cfg)
                ctrl = controller.step(det) if ws is not None else None
                base = ws.rectified if ws is not None else None
                extras: list[str] = []
                landing = False

                if state == "PREFLIGHT":
                    ready = det.have_drone and det.have_pad
                    extras.append("READY for takeoff (SPACE)" if ready
                                  else "waiting for drone + pad markers in view")

                elif state == "TRACKING":
                    if not det.have_drone:
                        misses += 1
                        oob = 0
                        if drone.airborne:
                            drone.hover()
                        if misses >= cfg.safety.max_consecutive_misses:
                            land_now(f"drone lost for {misses} frames -> landing")
                            landing = True
                    else:
                        misses = 0
                        dx, dy = det.drone.center_mm
                        margin = cfg.safety.out_of_bounds_margin_mm
                        outside = (dx < -margin or dx > cfg.workspace.width_mm + margin or
                                   dy < -margin or dy > cfg.workspace.height_mm + margin)
                        if outside:
                            oob += 1
                            if drone.airborne:
                                drone.hover()
                            if oob >= cfg.safety.out_of_bounds_dwell_frames:
                                land_now(f"drone left workspace ({dx:.0f},{dy:.0f}) -> landing")
                                landing = True
                        else:
                            oob = 0
                            if ctrl.over_pad:
                                land_now(f"over pad (dist={ctrl.distance_mm:.0f}mm) -> landing")
                                landing = True
                            else:
                                drone.send(ctrl.command.roll, ctrl.command.pitch,
                                           ctrl.command.yaw, ctrl.command.throttle)
                    extras.append(f"misses={misses}/{cfg.safety.max_consecutive_misses}   "
                                  f"oob={oob}/{cfg.safety.out_of_bounds_dwell_frames}")

                if landing:
                    extras.append("LANDING")

                display = build_overlay(base, det, ctrl, cfg, fps, state,
                                        extras=extras, raw_bgr=frame,
                                        detected_marker_ids=visible_ids)
                cv2.imshow(WINDOW, fit_to_screen(display, max_dim=900))

                if landing:
                    break

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    land_now("quit requested -> landing")
                    break
                elif key == 32 and state == "PREFLIGHT":  # SPACE
                    if not (det.have_drone and det.have_pad):
                        print("[land] cannot take off: need the drone and pad markers in view")
                        continue
                    print("[land] taking off...")
                    try:
                        drone.takeoff()
                    except Exception as e:
                        print(f"[land] takeoff failed: {e}")
                        break
                    controller.reset()
                    misses = oob = 0
                    state = "TRACKING"
                elif key == ord("l") and state == "TRACKING":
                    land_now("manual land requested")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
