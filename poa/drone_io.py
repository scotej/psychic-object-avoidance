"""
Thin safety-conscious wrapper around codrone-edu's Drone.

Use as a context manager so that __exit__ always lands and disconnects, even
on exceptions:

    with DroneIO(enabled=True) as drone:
        drone.set_led(0, 255, 0, 255)
        drone.takeoff()
        drone.send(roll=0, pitch=0, yaw=0, throttle=0)
        drone.land()

`enabled=False` makes every operation a no-op print, so the perception
pipeline can be exercised end-to-end without hardware (or without the
codrone-edu package installed).
"""

from __future__ import annotations

from typing import Optional


class DroneIO:
    def __init__(self, enabled: bool = True, port: Optional[str] = None) -> None:
        self.enabled = enabled
        self.port = port
        self._drone = None
        self._airborne = False

    # --- lifecycle -------------------------------------------------------

    def connect(self) -> None:
        if not self.enabled:
            print("[no-fly] connect (skipped)")
            return
        # Import here so --no-fly works even if codrone-edu isn't installed.
        from codrone_edu.drone import Drone

        self._drone = Drone()
        self._drone.pair(self.port) if self.port else self._drone.pair()
        battery = self.battery()
        print(f"[drone] paired (battery={battery if battery is not None else '?'}%)")

    def disconnect(self) -> None:
        if self._drone is None:
            return
        try:
            self._drone.close()
        except Exception as e:
            print(f"[drone] close error: {e}")
        self._drone = None

    def __enter__(self) -> "DroneIO":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Always attempt to land before disconnecting. If the drone is in an
        # unknown state, emergency_stop is the safer fallback.
        try:
            if self._airborne:
                if exc is not None:
                    self.emergency_stop()
                else:
                    self.land()
        finally:
            self.disconnect()

    # --- commands --------------------------------------------------------

    def set_led(self, r: int, g: int, b: int, brightness: int = 255) -> None:
        if not self.enabled:
            print(f"[no-fly] set_led({r},{g},{b},{brightness})")
            return
        if self._drone is not None:
            self._drone.set_drone_LED(int(r), int(g), int(b), int(brightness))

    def led_off(self) -> None:
        if not self.enabled:
            print("[no-fly] led_off")
            return
        if self._drone is not None:
            try:
                self._drone.drone_LED_off()
            except Exception:
                pass

    def takeoff(self) -> None:
        if not self.enabled:
            print("[no-fly] takeoff (skipped)")
            self._airborne = True
            return
        if self._drone is None:
            raise RuntimeError("takeoff: drone not connected")
        self._drone.takeoff()
        self._airborne = True

    def land(self) -> None:
        if not self.enabled:
            print("[no-fly] land (skipped)")
            self._airborne = False
            return
        if self._drone is None or not self._airborne:
            return
        try:
            self._drone.land()
        finally:
            self._airborne = False

    def send(self, roll: int, pitch: int, yaw: int, throttle: int) -> None:
        """One control packet. Safe to call every loop iteration."""
        if not self.enabled:
            return
        if self._drone is None:
            return
        self._drone.sendControl(int(roll), int(pitch), int(yaw), int(throttle))

    def hover(self) -> None:
        """Zero command. Use when perception is lost but we're still aloft."""
        self.send(0, 0, 0, 0)

    def emergency_stop(self) -> None:
        if not self.enabled:
            print("[no-fly] emergency_stop")
            self._airborne = False
            return
        if self._drone is None:
            return
        try:
            self._drone.emergency_stop()
        except Exception as e:
            print(f"[drone] emergency_stop failed ({e}); falling back to land()")
            try:
                self._drone.land()
            except Exception:
                pass
        finally:
            self._airborne = False

    # --- telemetry -------------------------------------------------------

    def battery(self) -> Optional[int]:
        if not self.enabled or self._drone is None:
            return None
        try:
            return int(self._drone.get_battery())
        except Exception:
            return None

    @property
    def airborne(self) -> bool:
        return self._airborne
