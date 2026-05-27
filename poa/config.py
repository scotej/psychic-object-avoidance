"""
Shared configuration for the landing pipeline.

Holds workspace dimensions, HSV color targets, controller gains, and safety
thresholds. The HSV targets can be loaded/saved as JSON so that
calibrate_hsv.py and land.py share the same calibration.

Three color targets matter:
  - led:    the top LED on the drone (single blob -> drone position)
  - rotors: the two red front rotors (two blobs -> nose midpoint -> heading)
  - pad:    the light-blue landing pad (single blob -> goal)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# Marker IDs printed by generate_markers.py, in workspace-corner order.
# Same ordering matters here because the homography is built from this list.
CORNER_IDS: tuple[int, int, int, int] = (0, 1, 2, 3)  # TL, TR, BR, BL
CORNER_NAMES: tuple[str, str, str, str] = ("TL", "TR", "BR", "BL")
ARUCO_DICT_NAME = "DICT_4X4_50"


@dataclass
class HSVRange:
    """A single HSV box. OpenCV H is 0..180, S and V are 0..255."""
    h_lo: int = 0
    h_hi: int = 180
    s_lo: int = 0
    s_hi: int = 255
    v_lo: int = 0
    v_hi: int = 255

    def lo(self) -> np.ndarray:
        return np.array([self.h_lo, self.s_lo, self.v_lo], dtype=np.uint8)

    def hi(self) -> np.ndarray:
        return np.array([self.h_hi, self.s_hi, self.v_hi], dtype=np.uint8)


@dataclass
class ColorTarget:
    """One physical thing to detect. Multiple HSV ranges are OR'd together,
    which is how we handle red (it wraps the hue circle)."""
    name: str
    ranges: list[HSVRange] = field(default_factory=list)
    expected_blobs: int = 1
    min_area_px: int = 30


def _led_default() -> ColorTarget:
    # Green LED on the drone. Default chosen to be far from red and blue in
    # HSV space so the three targets don't bleed into each other.
    return ColorTarget(
        name="led_green",
        ranges=[HSVRange(40, 85, 80, 255, 80, 255)],
        expected_blobs=1,
        min_area_px=40,
    )


def _rotors_default() -> ColorTarget:
    # Red wraps around H=0/180, so two ranges OR'd together.
    return ColorTarget(
        name="rotors_red",
        ranges=[
            HSVRange(0, 10, 120, 255, 80, 255),
            HSVRange(170, 180, 120, 255, 80, 255),
        ],
        expected_blobs=2,
        min_area_px=20,
    )


def _pad_default() -> ColorTarget:
    # Light-blue CoDrone EDU landing pad. Will almost certainly need
    # calibration in the actual lighting.
    return ColorTarget(
        name="pad_blue",
        ranges=[HSVRange(90, 115, 100, 255, 100, 255)],
        expected_blobs=1,
        min_area_px=200,
    )


@dataclass
class Workspace:
    width_mm: float = 1500.0
    height_mm: float = 1500.0
    px_per_mm: float = 1.0  # rectified output resolution

    @property
    def width_px(self) -> int:
        return int(round(self.width_mm * self.px_per_mm))

    @property
    def height_px(self) -> int:
        return int(round(self.height_mm * self.px_per_mm))


@dataclass
class Controller:
    kp_xy: float = 0.08              # roll/pitch power per mm of error
    max_command: int = 25            # saturation on roll/pitch (-100..100 valid)
    deadband_mm: float = 30.0        # zero command within this radius
    land_tolerance_mm: float = 60.0  # over-pad radius that counts as "above"
    land_dwell_frames: int = 8       # frames over the pad before triggering land
    yaw_command: int = 0             # we hold yaw constant (visual heading does the rotation)
    throttle_command: int = 0        # the drone holds its own altitude


@dataclass
class Safety:
    max_consecutive_misses: int = 20     # detection-loss frames before emergency land
    out_of_bounds_margin_mm: float = 50  # tolerance past the workspace edge
    out_of_bounds_dwell_frames: int = 5  # frames out-of-bounds before emergency land


@dataclass
class Config:
    workspace: Workspace = field(default_factory=Workspace)
    controller: Controller = field(default_factory=Controller)
    safety: Safety = field(default_factory=Safety)
    led: ColorTarget = field(default_factory=_led_default)
    rotors: ColorTarget = field(default_factory=_rotors_default)
    pad: ColorTarget = field(default_factory=_pad_default)
    led_rgb: tuple[int, int, int] = (0, 255, 0)  # what we set the drone LED to
    led_brightness: int = 255


def _serialize_target(ct: ColorTarget) -> dict:
    return {
        "name": ct.name,
        "expected_blobs": ct.expected_blobs,
        "min_area_px": ct.min_area_px,
        "ranges": [
            {"h_lo": r.h_lo, "h_hi": r.h_hi, "s_lo": r.s_lo, "s_hi": r.s_hi, "v_lo": r.v_lo, "v_hi": r.v_hi}
            for r in ct.ranges
        ],
    }


def _deserialize_target(name: str, data: dict, fallback: ColorTarget) -> ColorTarget:
    ranges = [HSVRange(**r) for r in data.get("ranges", [])]
    return ColorTarget(
        name=data.get("name", fallback.name),
        ranges=ranges or fallback.ranges,
        expected_blobs=data.get("expected_blobs", fallback.expected_blobs),
        min_area_px=data.get("min_area_px", fallback.min_area_px),
    )


def save_hsv(path: Path, config: Config) -> None:
    """Write just the HSV color targets to a JSON file."""
    data = {
        "led": _serialize_target(config.led),
        "rotors": _serialize_target(config.rotors),
        "pad": _serialize_target(config.pad),
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_hsv(path: Path, config: Config | None = None) -> Config:
    """Load HSV color targets from a JSON file into a new or existing Config."""
    if config is None:
        config = Config()
    data = json.loads(Path(path).read_text())
    if "led" in data:
        config.led = _deserialize_target("led", data["led"], config.led)
    if "rotors" in data:
        config.rotors = _deserialize_target("rotors", data["rotors"], config.rotors)
    if "pad" in data:
        config.pad = _deserialize_target("pad", data["pad"], config.pad)
    return config
