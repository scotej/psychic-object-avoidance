"""
Per-frame perception: find the markers, rectify the workspace, and report the
drone and the pad in workspace millimetres.

It all happens in a single detection pass on the raw frame. The four corner
markers give us a homography from camera pixels to a flat top-down workspace;
the drone and pad markers are then pushed through that same homography, so
their positions (and the drone's heading) come out directly in workspace mm.

We detect on the raw frame rather than on the warped image on purpose: warping
first throws away resolution and smears the marker borders, which hurts both
detection and the corner accuracy the heading depends on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import CORNER_IDS, DRONE_ID, PAD_ID, Config


def make_aruco_detector() -> cv2.aruco.ArucoDetector:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


@dataclass
class Marker:
    """A detected marker, expressed in workspace millimetres."""
    id: int
    corners_mm: np.ndarray            # (4, 2); ArUco order TL, TR, BR, BL
    center_mm: tuple[float, float]
    forward: tuple[float, float]      # unit vector; the marker's top edge is "forward"


@dataclass
class WorkspaceView:
    rectified: np.ndarray                            # top-down BGR, width_px x height_px
    homography: np.ndarray                           # raw px -> rectified px
    corner_centers: dict[int, tuple[float, float]]   # raw px, for debugging


@dataclass
class Detection:
    drone: Optional[Marker] = None
    pad: Optional[Marker] = None

    @property
    def have_drone(self) -> bool:
        return self.drone is not None

    @property
    def have_pad(self) -> bool:
        return self.pad is not None


def _detect_all(raw_bgr: np.ndarray,
                detector: cv2.aruco.ArucoDetector) -> tuple[dict[int, np.ndarray], list[int]]:
    """Return {id: (4, 2) raw-pixel corners} plus the sorted list of visible IDs."""
    corners_list, ids, _ = detector.detectMarkers(raw_bgr)
    if ids is None:
        return {}, []
    found = {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners_list)}
    return found, sorted(found)


def _center(corners: np.ndarray) -> tuple[float, float]:
    return float(corners[:, 0].mean()), float(corners[:, 1].mean())


def _to_workspace_marker(marker_id: int, raw_corners: np.ndarray,
                         H: np.ndarray, px_per_mm: float) -> Marker:
    """Push a marker's raw-pixel corners through the homography and into mm."""
    pts = raw_corners.reshape(1, 4, 2).astype(np.float32)
    corners_mm = cv2.perspectiveTransform(pts, H).reshape(4, 2) / px_per_mm
    cx, cy = _center(corners_mm)

    # ArUco corners come back as TL, TR, BR, BL, so the top edge spans 0->1 and
    # the bottom edge spans 3->2. Forward is the direction the top edge faces.
    top_mid = (corners_mm[0] + corners_mm[1]) / 2.0
    bottom_mid = (corners_mm[2] + corners_mm[3]) / 2.0
    fx, fy = float(top_mid[0] - bottom_mid[0]), float(top_mid[1] - bottom_mid[1])
    norm = float(np.hypot(fx, fy))
    forward = (fx / norm, fy / norm) if norm > 1e-6 else (0.0, -1.0)

    return Marker(id=marker_id, corners_mm=corners_mm, center_mm=(cx, cy), forward=forward)


def detect(raw_bgr: np.ndarray,
           detector: cv2.aruco.ArucoDetector,
           cfg: Config) -> tuple[Optional[WorkspaceView], Detection, list[int]]:
    """Run the whole per-frame pipeline.

    Returns (workspace, detection, visible_ids). `workspace` is None when any
    corner marker is missing — without all four we can't rectify, so there is
    nothing meaningful to track. `visible_ids` is every marker seen this frame,
    which the overlay uses to tell the operator what's still missing.
    """
    markers, visible_ids = _detect_all(raw_bgr, detector)
    if any(cid not in markers for cid in CORNER_IDS):
        return None, Detection(), visible_ids

    # Map the corner centres onto a (w_px, h_px) rectangle. We use w_px/h_px
    # rather than w_px-1/h_px-1 so that one rectified pixel is exactly
    # 1/px_per_mm of a millimetre — that keeps _to_workspace_marker's division
    # by px_per_mm exact instead of off by a pixel.
    src = np.array([_center(markers[cid]) for cid in CORNER_IDS], dtype=np.float32)
    w_px, h_px = cfg.workspace.width_px, cfg.workspace.height_px
    dst = np.array([[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    rectified = cv2.warpPerspective(raw_bgr, H, (w_px, h_px))

    det = Detection()
    px_per_mm = cfg.workspace.px_per_mm
    if DRONE_ID in markers:
        det.drone = _to_workspace_marker(DRONE_ID, markers[DRONE_ID], H, px_per_mm)
    if PAD_ID in markers:
        det.pad = _to_workspace_marker(PAD_ID, markers[PAD_ID], H, px_per_mm)

    view = WorkspaceView(
        rectified=rectified,
        homography=H,
        corner_centers={cid: _center(markers[cid]) for cid in CORNER_IDS},
    )
    return view, det, visible_ids
