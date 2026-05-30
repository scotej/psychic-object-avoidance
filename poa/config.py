"""
Shared configuration for the marker-based landing pipeline.

Everything the perception, control, and overlay code has to agree on lives
here: which ArUco IDs mean what, how big the workspace is, the controller
gains, and the safety limits.

There is no colour calibration anymore. The drone and the landing pad are
each found by their own ArUco marker rather than by HSV blobs, so the only
things that have to match the physical setup are the marker IDs below and the
workspace size.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Every marker (corners, drone, pad) is drawn from this one dictionary.
ARUCO_DICT_NAME = "DICT_4X4_50"

# The four corner markers, listed in the order the rectification homography
# expects them so the warped workspace always comes out the same way up.
CORNER_IDS: tuple[int, int, int, int] = (0, 1, 2, 3)  # TL, TR, BR, BL
CORNER_NAMES: tuple[str, str, str, str] = ("TL", "TR", "BR", "BL")

# The two moving targets.
DRONE_ID = 4  # marker on top of the drone; its top edge points to the nose
PAD_ID = 5    # marker on the floor marking where to land


@dataclass
class Workspace:
    """The rectangle bounded by the centres of the four corner markers."""
    width_mm: float = 300.0    # centre-to-centre distance, TL -> TR
    height_mm: float = 300.0   # centre-to-centre distance, TL -> BL
    px_per_mm: float = 2.0     # resolution of the rectified top-down image

    @property
    def width_px(self) -> int:
        """
        Compute the workspace width in pixels from `width_mm` and `px_per_mm`.
        
        Returns:
            int: Width in pixels, rounded to the nearest integer.
        """
        return int(round(self.width_mm * self.px_per_mm))

    @property
    def height_px(self) -> int:
        return int(round(self.height_mm * self.px_per_mm))


@dataclass
class Controller:
    kp_xy: float = 0.15            # roll/pitch power per mm of error
    max_command: int = 20         # cap on roll/pitch (valid range is -100..100)
    deadband_mm: float = 15.0     # don't chase noise inside this radius
    land_tolerance_mm: float = 40.0   # counts as "above the pad" within this radius
    land_dwell_frames: int = 8    # frames within tolerance before we commit to landing
    yaw_command: int = 0          # heading is handled by the marker, so we never yaw
    throttle_command: int = 0     # the drone holds its own altitude


@dataclass
class Safety:
    max_consecutive_misses: int = 20      # frames without the drone before auto-land
    out_of_bounds_margin_mm: float = 40.0  # how far past the edge we tolerate
    out_of_bounds_dwell_frames: int = 5    # frames out of bounds before auto-land


@dataclass
class Config:
    workspace: Workspace = field(default_factory=Workspace)
    controller: Controller = field(default_factory=Controller)
    safety: Safety = field(default_factory=Safety)
