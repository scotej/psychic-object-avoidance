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
    """The rectangle bounded by the centres of the four corner markers.

    With ``auto_size`` on (the default) the centre-to-centre ``width_mm`` and
    ``height_mm`` are measured at runtime from the corner markers themselves —
    each marker is a printed square of known size (``corner_marker_mm``), which
    acts as a ruler in the image. That lets the same printed cards define a
    workspace of any size, from ~1 m to 3 m or more, without measuring by hand.
    Until calibration finishes the values below are used as a fallback, and
    they are also what gets used when ``auto_size`` is off.
    """
    width_mm: float = 300.0    # centre-to-centre distance, TL -> TR
    height_mm: float = 300.0   # centre-to-centre distance, TL -> BL
    px_per_mm: float = 2.0     # resolution of the rectified top-down image

    # Auto-sizing: measure the workspace from the corner markers' known size.
    auto_size: bool = True
    corner_marker_mm: float = 40.0   # printed side length of corner markers 0-3
    # When auto-sizing, px_per_mm is derived so the larger rectified dimension
    # lands near this many pixels. This keeps warpPerspective cheap whether the
    # workspace is 1 m or 3 m across, instead of ballooning to a huge image.
    target_rectified_px: int = 900

    @property
    def width_px(self) -> int:
        return int(round(self.width_mm * self.px_per_mm))

    @property
    def height_px(self) -> int:
        return int(round(self.height_mm * self.px_per_mm))

    def fit_resolution(self) -> None:
        """Pick px_per_mm so the larger side is ~target_rectified_px pixels."""
        longest_mm = max(self.width_mm, self.height_mm)
        if longest_mm > 0:
            self.px_per_mm = self.target_rectified_px / longest_mm


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
