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

Workspace sizing:
  By default the workspace size is measured automatically from the corner
  markers, each of which is a printed square of known size (--corner-mm). The
  cards can therefore be laid out at any scale — roughly 1 m up to 3 m or more —
  and the system figures out the play-area dimensions during pre-flight. Pass
  --no-auto-size to instead trust the fixed --width-mm / --height-mm values.

Safety:
  - Pre-flight: takeoff only once the workspace is calibrated, the operator
    presses SPACE, *and* both the drone and pad markers are currently in view.
  - Drone marker lost for N frames    -> land.
  - Camera stops delivering frames    -> land.
  - Drone outside the workspace bounds -> land.
  - 'l' during flight -> land now.
  - 'c' during pre-flight -> re-measure the workspace.
  - 'q' / ESC anywhere -> land then quit.
  - Any unhandled exception -> emergency_stop via the DroneIO context manager.

Usage:
    python land.py                       # camera 0, fly, auto-size the workspace
    python land.py --no-fly              # perception only, no drone
    python land.py --camera 1
    python land.py --corner-mm 40        # corner-marker side length (the ruler)
    python land.py --no-auto-size --width-mm 300 --height-mm 300
"""

from __future__ import annotations

import argparse
import time

import cv2

from poa.config import Config
from poa.controller import LandingController
from poa.drone_io import DroneIO
from poa.overlay import build_overlay, fit_to_screen
from poa.perception import WorkspaceCalibrator, detect, make_aruco_detector


WINDOW = "land  (SPACE=takeoff  l=land  c=recalibrate  q=quit)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--camera", type=int, default=0,
                   help="Camera index (default: 0)")
    p.add_argument("--no-fly", action="store_true",
                   help="Skip every drone command; run the perception loop only.")
    p.add_argument("--no-auto-size", action="store_true",
                   help="Don't measure the workspace from the corner markers; "
                        "use the fixed --width-mm / --height-mm instead.")
    p.add_argument("--corner-mm", type=float, default=40.0,
                   help="Corner-marker side length in mm; the ruler used to "
                        "auto-measure the workspace (default: 40, matches "
                        "generate_markers.py --corner-size).")
    p.add_argument("--rectified-px", type=int, default=900,
                   help="Target size of the longer rectified-image side, in px, "
                        "when auto-sizing (default: 900).")
    p.add_argument("--width-mm", type=float, default=300.0,
                   help="Workspace width in mm (TL->TR centres). Used with "
                        "--no-auto-size, or as a fallback until calibrated.")
    p.add_argument("--height-mm", type=float, default=300.0,
                   help="Workspace height in mm (TL->BL centres). Used with "
                        "--no-auto-size, or as a fallback until calibrated.")
    p.add_argument("--px-per-mm", type=float, default=2.0,
                   help="Rectified resolution in px/mm (only with --no-auto-size)")
    p.add_argument("--port", type=str, default=None,
                   help="codrone-edu serial port (default: auto-detect)")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.workspace.auto_size = not args.no_auto_size
    cfg.workspace.corner_marker_mm = args.corner_mm
    cfg.workspace.target_rectified_px = args.rectified_px
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

    # Auto-sizing: measure the workspace from the corner markers during
    # pre-flight, lock it in, then fly. None when --no-auto-size is set.
    calibrator = (WorkspaceCalibrator(cfg.workspace.corner_marker_mm)
                  if cfg.workspace.auto_size else None)

    def apply_calibration(size_mm: tuple[float, float]) -> None:
        cfg.workspace.width_mm, cfg.workspace.height_mm = size_mm
        cfg.workspace.fit_resolution()
        print(f"[land] workspace measured: {size_mm[0]:.0f} x {size_mm[1]:.0f} mm "
              f"({cfg.workspace.px_per_mm:.3f} px/mm)")

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
                    if drone.airborne and read_fails == 1:
                        drone.hover()
                    if drone.airborne and read_fails >= cfg.safety.max_consecutive_misses:
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

                # Auto-size the workspace from the corner markers (pre-flight
                # only — once flying we keep the size we locked in).
                if calibrator is not None and ws is not None and not calibrator.locked:
                    if calibrator.update(ws.corner_corners) is not None:
                        apply_calibration(calibrator.result)

                calibrated = calibrator is None or calibrator.locked

                if state == "PREFLIGHT":
                    if calibrator is not None and not calibrator.locked:
                        extras.append(f"measuring workspace... "
                                      f"{calibrator.count}/{calibrator.samples}")
                    else:
                        extras.append(f"workspace: {cfg.workspace.width_mm:.0f} x "
                                      f"{cfg.workspace.height_mm:.0f} mm")
                    ready = calibrated and det.have_drone and det.have_pad
                    if ready:
                        extras.append("READY for takeoff (SPACE)")
                    elif not calibrated:
                        extras.append("need all 4 corner markers to measure workspace")
                    else:
                        extras.append("waiting for drone + pad markers in view")

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
                elif key == ord("c") and state == "PREFLIGHT" and calibrator is not None:
                    calibrator.reset()
                    print("[land] re-measuring workspace...")
                elif key == 32 and state == "PREFLIGHT":  # SPACE
                    if not calibrated:
                        print("[land] cannot take off: workspace not measured yet "
                              "(need all 4 corner markers in view)")
                        continue
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
