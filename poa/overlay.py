"""
Debug visualization. Draws the perception and control state on top of the
rectified workspace so the operator can see what the controller is seeing in
real time. Everything goes into a single composed image — there is only ever
one window.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import CORNER_IDS, Config
from .controller import ControlOutput
from .perception import Detection, Marker


def _mm_to_px(xy_mm: tuple[float, float], px_per_mm: float) -> tuple[int, int]:
    return int(round(xy_mm[0] * px_per_mm)), int(round(xy_mm[1] * px_per_mm))


def _format_xy(xy: Optional[tuple[float, float]]) -> str:
    if xy is None:
        return "    -    ,    -    "
    return f"{xy[0]:7.1f}, {xy[1]:7.1f}"


def draw_status_panel(img: np.ndarray, lines: list[str]) -> None:
    """Translucent dark panel + status text at the top-left."""
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    line_h = 20
    pad = 8

    widths = [cv2.getTextSize(s, font, scale, thickness)[0][0] for s in lines]
    box_w = min(img.shape[1], max(widths) + pad * 2)
    box_h = pad * 2 + line_h * len(lines)

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)

    for i, s in enumerate(lines):
        y = pad + line_h * (i + 1) - 6
        cv2.putText(img, s, (pad, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_marker(img: np.ndarray, marker: Marker, px_per_mm: float,
                 color: tuple[int, int, int], label: str) -> None:
    poly = np.array([_mm_to_px((x, y), px_per_mm) for x, y in marker.corners_mm],
                    dtype=np.int32)
    cv2.polylines(img, [poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    cx, cy = _mm_to_px(marker.center_mm, px_per_mm)
    cv2.putText(img, label, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1, cv2.LINE_AA)


def draw_perception(base_bgr: np.ndarray, det: Detection, cfg: Config) -> np.ndarray:
    img = base_bgr.copy()
    px = cfg.workspace.px_per_mm

    # Landing pad: yellow outline, tilted cross, tolerance ring.
    if det.pad is not None:
        _draw_marker(img, det.pad, px, (0, 255, 255), f"pad #{det.pad.id}")
        cx, cy = _mm_to_px(det.pad.center_mm, px)
        cv2.circle(img, (cx, cy), int(round(cfg.controller.land_tolerance_mm * px)),
                   (0, 200, 200), 1, cv2.LINE_AA)
        cv2.drawMarker(img, (cx, cy), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 22, 2, cv2.LINE_AA)

    # Drone: green outline + heading arrow out the nose.
    if det.drone is not None:
        _draw_marker(img, det.drone, px, (0, 255, 0), f"drone #{det.drone.id}")
        cx, cy = _mm_to_px(det.drone.center_mm, px)
        fx, fy = det.drone.forward
        arrow_mm = 45.0
        tip = _mm_to_px((det.drone.center_mm[0] + fx * arrow_mm,
                         det.drone.center_mm[1] + fy * arrow_mm), px)
        cv2.arrowedLine(img, (cx, cy), tip, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.3)

    # Error vector: thin grey line from drone to pad.
    if det.have_drone and det.have_pad:
        cv2.line(img, _mm_to_px(det.drone.center_mm, px),
                 _mm_to_px(det.pad.center_mm, px), (200, 200, 200), 1, cv2.LINE_AA)

    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (80, 80, 80), 1)
    return img


def build_overlay(
    base_bgr: Optional[np.ndarray],
    det: Optional[Detection],
    ctrl: Optional[ControlOutput],
    cfg: Config,
    fps: float,
    state: str,
    extras: Optional[list[str]] = None,
    raw_bgr: Optional[np.ndarray] = None,
    detected_marker_ids: Optional[list[int]] = None,
) -> np.ndarray:
    """Compose the debug frame. Falls back to the raw camera with a banner when
    the workspace isn't detected, so the operator can still aim the camera."""
    if base_bgr is not None:
        img = draw_perception(base_bgr, det, cfg) if det is not None else base_bgr.copy()
    elif raw_bgr is not None:
        img = raw_bgr.copy()
        h, w = img.shape[:2]
        banner_y = h - 60
        cv2.rectangle(img, (0, banner_y - 30), (w, h), (0, 0, 0), -1)
        cv2.putText(img, "NO WORKSPACE - showing raw camera", (16, banner_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        found = sorted(detected_marker_ids) if detected_marker_ids else []
        cv2.putText(img, f"markers seen: {found}   need all of {list(CORNER_IDS)}",
                    (16, banner_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 1, cv2.LINE_AA)
    else:
        img = np.zeros((cfg.workspace.height_px, cfg.workspace.width_px, 3), dtype=np.uint8)
        cv2.putText(img, "NO CAMERA", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (0, 0, 255), 3, cv2.LINE_AA)

    lines = [f"{fps:5.1f} FPS   state: {state}"]
    if det is not None:
        drone_xy = det.drone.center_mm if det.drone else None
        pad_xy = det.pad.center_mm if det.pad else None
        heading = det.drone.forward if det.drone else None
        lines.append(f"drone  : {_format_xy(drone_xy)} mm   "
                     f"head: {('%+.2f, %+.2f' % heading) if heading else '   -  ,   -  '}")
        lines.append(f"pad    : {_format_xy(pad_xy)} mm")
    if ctrl is not None:
        if ctrl.distance_mm is not None:
            ef = ctrl.error_body_mm
            lines.append(f"dist   : {ctrl.distance_mm:6.1f} mm   "
                         f"fwd: {ef[0]:+7.1f}  right: {ef[1]:+7.1f}")
        c = ctrl.command
        lines.append(f"cmd    : roll {c.roll:+4d}  pitch {c.pitch:+4d}  "
                     f"yaw {c.yaw:+4d}  throttle {c.throttle:+4d}")
        if ctrl.over_pad:
            lines.append("OVER PAD - landing")
    if extras:
        lines.extend(extras)

    draw_status_panel(img, lines)
    return img


def fit_to_screen(img: np.ndarray, max_dim: int = 900) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(max_dim / max(h, w), 1.0)
    if s < 1.0:
        return cv2.resize(img, (int(round(w * s)), int(round(h * s))))
    return img
