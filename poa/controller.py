"""
Closed-loop landing controller.

Converts (drone position, drone heading, pad position) in world mm into
CoDrone EDU roll/pitch commands in the drone body frame, with:
  - deadband (no command when within deadband_mm of the pad center)
  - saturation (commands capped at +/- max_command)
  - "over pad" dwell detection (used by land.py to decide when to land())

World frame: image-style axes (+x right, +y down) in millimeters.
Body frame: forward = heading vector; right = heading rotated 90 deg CW on
screen, i.e. (-fy, fx) — the drone's right wing in world axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import Controller as ControllerCfg
from .perception import Detection


@dataclass
class Command:
    roll: int = 0      # CoDrone: + right, - left
    pitch: int = 0     # CoDrone: + forward, - backward
    yaw: int = 0       # + clockwise, - counterclockwise (we keep 0)
    throttle: int = 0  # + climb, - descend (we keep 0; drone holds altitude)

    @property
    def is_zero(self) -> bool:
        return self.roll == 0 and self.pitch == 0 and self.yaw == 0 and self.throttle == 0


@dataclass
class ControlOutput:
    command: Command
    error_world_mm: Optional[tuple[float, float]]   # pad - drone, in world frame
    error_body_mm: Optional[tuple[float, float]]    # (forward, right), in body frame
    distance_mm: Optional[float]
    over_pad: bool                                  # within tolerance for required dwell


class LandingController:
    def __init__(self, cfg: ControllerCfg) -> None:
        self.cfg = cfg
        self._over_pad_streak = 0

    def reset(self) -> None:
        self._over_pad_streak = 0

    def step(self, det: Detection) -> ControlOutput:
        cmd = Command(yaw=self.cfg.yaw_command, throttle=self.cfg.throttle_command)
        if not (det.have_drone and det.have_pad):
            self._over_pad_streak = 0
            return ControlOutput(cmd, None, None, None, False)

        # det.have_drone => led_xy_mm and heading are not None
        ex = det.pad_xy_mm[0] - det.led_xy_mm[0]
        ey = det.pad_xy_mm[1] - det.led_xy_mm[1]
        fx, fy = det.heading

        forward = ex * fx + ey * fy        # body +x (drone's nose direction)
        right = -ex * fy + ey * fx         # body +y (drone's right wing)
        distance = (forward * forward + right * right) ** 0.5

        if distance <= self.cfg.land_tolerance_mm:
            self._over_pad_streak += 1
        else:
            self._over_pad_streak = 0
        over_pad = self._over_pad_streak >= self.cfg.land_dwell_frames

        if distance > self.cfg.deadband_mm:
            mx = self.cfg.max_command
            pitch_f = max(-mx, min(mx, self.cfg.kp_xy * forward))
            roll_f = max(-mx, min(mx, self.cfg.kp_xy * right))
            cmd.pitch = int(round(pitch_f))
            cmd.roll = int(round(roll_f))

        return ControlOutput(
            command=cmd,
            error_world_mm=(ex, ey),
            error_body_mm=(forward, right),
            distance_mm=distance,
            over_pad=over_pad,
        )
