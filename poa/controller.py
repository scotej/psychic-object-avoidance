"""
Closed-loop landing controller.

Turns the drone marker and the pad marker into CoDrone EDU roll/pitch commands
in the drone's own body frame, with a deadband near the pad, saturation on the
output, and an "over the pad" dwell counter that land.py uses to decide when to
actually descend.

World frame: image-style axes, +x right and +y down, in millimetres.
Body frame: forward = the drone's heading (its marker's top edge); right = that
heading turned 90 deg clockwise on screen, i.e. (-fy, fx) — the right wing.
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
    yaw: int = 0       # + clockwise, - counterclockwise (we hold 0)
    throttle: int = 0  # + climb, - descend (we hold 0; the drone keeps its altitude)

    @property
    def is_zero(self) -> bool:
        return self.roll == 0 and self.pitch == 0 and self.yaw == 0 and self.throttle == 0


@dataclass
class ControlOutput:
    command: Command
    error_world_mm: Optional[tuple[float, float]]   # pad - drone, world frame
    error_body_mm: Optional[tuple[float, float]]    # (forward, right), body frame
    distance_mm: Optional[float]
    over_pad: bool                                  # within tolerance for the required dwell


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

        dx, dy = det.drone.center_mm
        px, py = det.pad.center_mm
        fx, fy = det.drone.forward
        ex, ey = px - dx, py - dy

        forward = ex * fx + ey * fy        # body +x (toward the nose)
        right = -ex * fy + ey * fx         # body +y (toward the right wing)
        distance = (forward * forward + right * right) ** 0.5

        if distance <= self.cfg.land_tolerance_mm:
            self._over_pad_streak += 1
        else:
            self._over_pad_streak = 0
        over_pad = self._over_pad_streak >= self.cfg.land_dwell_frames

        if distance > self.cfg.deadband_mm:
            mx = self.cfg.max_command
            cmd.pitch = int(round(max(-mx, min(mx, self.cfg.kp_xy * forward))))
            cmd.roll = int(round(max(-mx, min(mx, self.cfg.kp_xy * right))))

        return ControlOutput(
            command=cmd,
            error_world_mm=(ex, ey),
            error_body_mm=(forward, right),
            distance_mm=distance,
            over_pad=over_pad,
        )
