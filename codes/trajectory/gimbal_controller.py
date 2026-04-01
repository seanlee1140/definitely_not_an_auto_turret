"""
Gimbal Controller
==================

Controls the 2-axis turret over serial to the ESP32.

    Pan  axis : Unipolar stepper + ULN2003 — absolute angle from home
    Tilt axis : Servo 0–180° (90° = level, <90° = up, >90° = down)

Serial protocol (matches turret_firmware.ino):
    PAN_TO <deg>    → PAN_DONE <steps>
    TILT <deg>      → TILT_DONE
    SPEED <us>      → OK
    FIRE            → FIRED
    HOME            → HOME_DONE
    STATUS          → STATUS pan_steps=... pan_deg=... tilt_deg=...

Dependencies:
    pyserial
"""

import time
import numpy as np
import serial


class GimbalController:
    """
    Drives the gimbal over serial to the ESP32 firmware.

    Args:
        port: Serial port string, e.g. '/dev/ttyUSB0' or 'COM3'.
        baud: Baud rate — must match firmware (default 115200).
        steps_per_rev: Stepper half-steps per revolution (default 2048
            for 28BYJ-48 in half-step mode).
        pan_limit_deg: Maximum pan angle in either direction from home.
        tilt_min_deg: Minimum servo angle (hard lower bound).
        tilt_max_deg: Maximum servo angle (hard upper bound).
        tilt_center_deg: Servo angle that means "level" (default 90°).
        timeout: Serial readline timeout in seconds.
    """

    STEPS_PER_REV = 2048   # 28BYJ-48 half-step

    def __init__(
        self,
        port: str = '/dev/ttyUSB0',
        baud: int = 115200,
        steps_per_rev: int = 2048,
        pan_limit_deg: float = 180.0,
        tilt_min_deg: float = 30.0,
        tilt_max_deg: float = 150.0,
        tilt_center_deg: float = 90.0,
        timeout: float = 3.0,
    ):
        self.steps_per_rev  = steps_per_rev
        self.steps_per_deg  = steps_per_rev / 360.0
        self.pan_limit_deg  = pan_limit_deg
        self.tilt_min_deg   = tilt_min_deg
        self.tilt_max_deg   = tilt_max_deg
        self.tilt_center_deg = tilt_center_deg

        self.current_pan_deg  = 0.0
        self.current_tilt_deg = tilt_center_deg

        self._ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2.0)   # ESP32 resets on serial connect; wait for boot
        self._drain()

    # ── Private helpers ───────────────────────────────────────────────────

    def _drain(self) -> None:
        """Discard any buffered bytes (boot messages, etc.)."""
        while self._ser.in_waiting:
            self._ser.readline()

    def _send(self, cmd: str) -> str:
        """Write a command line and return the response line (stripped)."""
        self._ser.write((cmd + '\n').encode())
        resp = self._ser.readline().decode(errors='replace').strip()
        return resp

    # ── Pan (stepper) ─────────────────────────────────────────────────────

    def pan_to(self, angle_deg: float) -> None:
        """
        Rotate the base stepper to an absolute angle.

        Args:
            angle_deg: Target angle in degrees from the home position.
                Positive = clockwise viewed from above, negative = CCW.
                Clamped to ±pan_limit_deg.
        """
        angle_deg = float(np.clip(angle_deg, -self.pan_limit_deg, self.pan_limit_deg))
        resp = self._send(f"PAN_TO {angle_deg:.2f}")
        if resp.startswith("PAN_DONE"):
            self.current_pan_deg = angle_deg

    # ── Tilt (servo) ──────────────────────────────────────────────────────

    def tilt_to(self, angle_deg: float) -> None:
        """
        Set the tilt servo angle.

        Args:
            angle_deg: Servo angle 0–180°.
                90° = level, values <90° aim up, values >90° aim down.
                Clamped to [tilt_min_deg, tilt_max_deg].
        """
        angle_deg = float(np.clip(angle_deg, self.tilt_min_deg, self.tilt_max_deg))
        resp = self._send(f"TILT {angle_deg:.1f}")
        if resp.startswith("TILT_DONE"):
            self.current_tilt_deg = angle_deg

    # ── Combined ──────────────────────────────────────────────────────────

    def aim(self, pan_deg: float, tilt_deg: float) -> None:
        """Move pan then tilt in sequence."""
        self.pan_to(pan_deg)
        self.tilt_to(tilt_deg)

    # ── Firing ────────────────────────────────────────────────────────────

    def fire(self) -> None:
        """Trigger the firing mechanism via ESP32 FIRE_PIN relay/solenoid."""
        self._send("FIRE")

    # ── Utility ───────────────────────────────────────────────────────────

    def home(self) -> None:
        """Return both axes to home (0° pan, 90° tilt)."""
        resp = self._send("HOME")
        if resp.startswith("HOME_DONE"):
            self.current_pan_deg  = 0.0
            self.current_tilt_deg = self.tilt_center_deg

    def status(self) -> dict:
        """
        Query current position from the firmware.

        Returns:
            dict with keys: pan_steps, pan_deg, tilt_deg
        """
        resp = self._send("STATUS")
        result: dict = {}
        for token in resp.replace("STATUS ", "").split():
            if "=" in token:
                k, v = token.split("=", 1)
                try:
                    result[k] = float(v)
                except ValueError:
                    pass
        return result

    def set_speed(self, us_per_step: int) -> None:
        """
        Set stepper speed.

        Args:
            us_per_step: Microseconds between half-steps.
                ~500 = fast (may stall), ~1500 = default, ~3000 = slow/quiet.
        """
        self._send(f"SPEED {int(us_per_step)}")

    def close(self) -> None:
        """Close the serial connection."""
        if self._ser.is_open:
            self._ser.close()

    # ── Angle conversion helpers (static) ─────────────────────────────────

    @staticmethod
    def pixel_to_pan_offset(cx: float, frame_width: int,
                            hfov_deg: float = 60.0) -> float:
        """
        Convert a horizontal pixel position to a pan angle offset.

        The offset is relative to the current pan position — add it to
        current_pan_deg to get the new absolute target.

        Args:
            cx: Ball centroid x-pixel.
            frame_width: Camera frame width in pixels.
            hfov_deg: Camera horizontal field of view in degrees.

        Returns:
            Angle offset in degrees (+ve = right of centre, -ve = left).
        """
        return ((cx - frame_width / 2.0) / frame_width) * hfov_deg

    @staticmethod
    def pixel_to_tilt(cy: float, frame_height: int, vfov_deg: float = 45.0,
                      tilt_center_deg: float = 90.0,
                      ballistic_correction_deg: float = 0.0) -> float:
        """
        Convert a vertical pixel position to an absolute servo tilt angle,
        with optional upward ballistic correction for projectile drop.

        Args:
            cy: Ball centroid y-pixel (0 = top of frame).
            frame_height: Camera frame height in pixels.
            vfov_deg: Camera vertical field of view in degrees.
            tilt_center_deg: Servo angle for horizontal aim (default 90°).
            ballistic_correction_deg: Extra upward angle to compensate for
                projectile drop. Compute with ballistic_correction_deg().
                Positive = aim higher (servo angle decreases).

        Returns:
            Servo angle in degrees, including ballistic correction.
            Ball above frame centre → servo < 90° (aim up).
            Ball below frame centre → servo > 90° (aim down).
        """
        # cy increases downward in image; invert so "up in frame = servo up"
        offset = -((cy - frame_height / 2.0) / frame_height) * vfov_deg
        # Subtract correction: positive correction means aim higher = smaller servo angle
        return tilt_center_deg - offset - ballistic_correction_deg

    @staticmethod
    def estimate_distance(pixel_radius: float, frame_width: int,
                          hfov_deg: float = 60.0,
                          ball_diameter_mm: float = 67.0) -> float:
        """
        Estimate distance to the tennis ball using its known physical size.

        A standard tennis ball is 67mm in diameter. By comparing its apparent
        pixel radius to the camera focal length we get metric distance.

        focal_length_px = (frame_width / 2) / tan(hfov / 2)

        Args:
            pixel_radius: Detected ball radius in pixels (from detect_ball_hsv).
            frame_width: Camera frame width in pixels.
            hfov_deg: Camera horizontal field of view in degrees.
            ball_diameter_mm: Real ball diameter in mm (default 67 for tennis ball).

        Returns:
            Estimated distance in mm, or None if pixel_radius is too small
            to be reliable (< 3px).
        """
        if pixel_radius < 3:
            return None
        focal_length_px = (frame_width / 2.0) / np.tan(np.radians(hfov_deg / 2.0))
        real_radius_mm  = ball_diameter_mm / 2.0
        return (real_radius_mm * focal_length_px) / pixel_radius

    @staticmethod
    def ballistic_correction_deg(distance_mm: float,
                                 projectile_speed_mms: float,
                                 gravity_mms2: float = 9810.0) -> float:
        """
        Compute the extra upward tilt needed to compensate for projectile drop.

        Uses the flat-fire approximation (valid when launch angle is small):
            Δθ = arctan( g * d / (2 * v²) )
        where d is horizontal distance and v is muzzle speed.

        Args:
            distance_mm: Horizontal distance to target in mm.
            projectile_speed_mms: Muzzle speed in mm/s.
                Example: 30 m/s → 30_000 mm/s.
            gravity_mms2: Gravitational acceleration in mm/s² (default 9810).

        Returns:
            Correction angle in degrees (always positive — aim this much higher).
        """
        correction_rad = np.arctan(
            (gravity_mms2 * distance_mm) / (2.0 * projectile_speed_mms ** 2)
        )
        return float(np.degrees(correction_rad))
