"""
Per-frame perception: workspace rectification + color-blob detection.

The workspace is defined by the centers of the 4 corner ArUco markers; we
build a homography from those centers to a known mm-scale rectangle and warp
the raw frame into a top-down view. Color detection then runs on the
rectified image so positions come out in workspace millimeters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .config import CORNER_IDS, ColorTarget, Config


def make_aruco_detector() -> cv2.aruco.ArucoDetector:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


@dataclass
class WorkspaceFrame:
    homography: np.ndarray
    rectified: np.ndarray
    raw_marker_centers: dict[int, tuple[float, float]]


def _marker_center(corners: np.ndarray) -> tuple[float, float]:
    pts = corners.reshape(-1, 2)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def detect_workspace(
    raw_bgr: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    cfg: Config,
) -> Optional[WorkspaceFrame]:
    """Detect the 4 corner markers and warp the raw frame to a top-down
    workspace image. Returns None if any expected marker ID is missing."""
    corners_list, ids, _ = detector.detectMarkers(raw_bgr)
    if ids is None:
        return None

    centers: dict[int, tuple[float, float]] = {}
    for mid, corners in zip(ids.flatten().tolist(), corners_list):
        if mid in CORNER_IDS:
            centers[mid] = _marker_center(corners)
    if any(mid not in centers for mid in CORNER_IDS):
        return None

    src = np.array([centers[mid] for mid in CORNER_IDS], dtype=np.float32)
    w_px, h_px = cfg.workspace.width_px, cfg.workspace.height_px
    dst = np.array(
        [[0, 0], [w_px - 1, 0], [w_px - 1, h_px - 1], [0, h_px - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src, dst)
    rectified = cv2.warpPerspective(raw_bgr, H, (w_px, h_px))
    return WorkspaceFrame(homography=H, rectified=rectified, raw_marker_centers=centers)


def build_mask(hsv: np.ndarray, target: ColorTarget) -> np.ndarray:
    """OR together every HSVRange in the target, then morphologically clean."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in target.ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, r.lo(), r.hi()))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_blobs(mask: np.ndarray, target: ColorTarget) -> list[tuple[float, float, float]]:
    """Return [(cx_px, cy_px, area_px)] for the N=expected_blobs biggest
    blobs that exceed min_area_px, ordered by area descending."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: list[tuple[float, float, float]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < target.min_area_px:
            continue
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        blobs.append((cx, cy, area))
    blobs.sort(key=lambda b: b[2], reverse=True)
    return blobs[: target.expected_blobs]


@dataclass
class Detection:
    led_xy_mm: Optional[tuple[float, float]] = None          # drone position (LED centroid)
    rotors_xy_mm: list[tuple[float, float]] = field(default_factory=list)  # 0/1/2 red blobs
    pad_xy_mm: Optional[tuple[float, float]] = None          # landing target
    nose_mm: Optional[tuple[float, float]] = None            # midpoint of the two rotors
    heading: Optional[tuple[float, float]] = None            # unit vector LED -> nose (world frame)

    @property
    def have_drone(self) -> bool:
        return self.led_xy_mm is not None and self.heading is not None

    @property
    def have_pad(self) -> bool:
        return self.pad_xy_mm is not None


def detect_in_workspace(
    rectified_bgr: np.ndarray,
    cfg: Config,
) -> tuple[Detection, dict[str, np.ndarray]]:
    """Run all 3 color detections on a rectified workspace image."""
    hsv = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2HSV)
    px_per_mm = cfg.workspace.px_per_mm

    def to_mm(xy_px: tuple[float, float]) -> tuple[float, float]:
        return xy_px[0] / px_per_mm, xy_px[1] / px_per_mm

    masks: dict[str, np.ndarray] = {}
    det = Detection()

    masks["led"] = build_mask(hsv, cfg.led)
    led_blobs = find_blobs(masks["led"], cfg.led)
    if led_blobs:
        det.led_xy_mm = to_mm((led_blobs[0][0], led_blobs[0][1]))

    masks["rotors"] = build_mask(hsv, cfg.rotors)
    rot_blobs = find_blobs(masks["rotors"], cfg.rotors)
    det.rotors_xy_mm = [to_mm((b[0], b[1])) for b in rot_blobs]

    if len(det.rotors_xy_mm) == 2 and det.led_xy_mm is not None:
        nx = (det.rotors_xy_mm[0][0] + det.rotors_xy_mm[1][0]) / 2
        ny = (det.rotors_xy_mm[0][1] + det.rotors_xy_mm[1][1]) / 2
        det.nose_mm = (nx, ny)
        dx, dy = nx - det.led_xy_mm[0], ny - det.led_xy_mm[1]
        norm = (dx * dx + dy * dy) ** 0.5
        if norm > 1e-3:
            det.heading = (dx / norm, dy / norm)

    masks["pad"] = build_mask(hsv, cfg.pad)
    pad_blobs = find_blobs(masks["pad"], cfg.pad)
    if pad_blobs:
        det.pad_xy_mm = to_mm((pad_blobs[0][0], pad_blobs[0][1]))

    return det, masks
