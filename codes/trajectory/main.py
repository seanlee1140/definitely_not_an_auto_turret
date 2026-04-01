"""
Turret Main Loop — Stationary Tennis Ball
==========================================

Sweep-and-shoot state machine:

    SWEEP  →  Rotate the stepper slowly across the FOV, checking every
              frame for a tennis ball.
    AIM    →  Ball found; compute pan/tilt from pixel position, send
              commands to ESP32, track it while stationary.
    FIRE   →  Aim confirmed for enough consecutive frames; shoot.
    DONE   →  Fired. Press R to reset and sweep again, Q to quit.

Hardware:
    Pan  : Unipolar stepper + ULN2003 → ESP32  (Z-axis rotation)
    Tilt : Servo 0–180°               → ESP32
    Cam  : Single USB camera

Edit the CONFIG section below to match your setup.

Run:
    python main.py
    (ESP32 must be flashed with esp32_firmware/turret_firmware.ino)
"""

import cv2
import numpy as np

from camera_utils import open_camera, detect_ball_hsv, draw_detection, draw_crosshair
from gimbal_controller import GimbalController
from firing_system import FiringSystem

# ── CONFIG ────────────────────────────────────────────────────────────────────

SERIAL_PORT      = '/dev/ttyUSB0'  # Windows: 'COM3' | Linux: '/dev/ttyUSB0'
BAUD_RATE        = 115200

CAMERA_INDEX     = 0
FRAME_WIDTH      = 640
FRAME_HEIGHT     = 480

# Camera field of view — measure or look up your lens spec
HFOV_DEG         = 60.0   # horizontal FOV in degrees
VFOV_DEG         = 45.0   # vertical  FOV in degrees

# Sweep behaviour
SWEEP_STEP_DEG   = 3.0    # degrees per frame during sweep
SWEEP_MAX_DEG    = 170.0  # total sweep range (will go ±SWEEP_MAX_DEG/2)
SWEEP_SPEED_US   = 2000   # stepper µs/step during sweep  (slower, quieter)

# Aim behaviour
AIM_SPEED_US     = 1000   # stepper µs/step when centering on target (faster)
AIM_TOLERANCE_PX = 12     # pixel radius around centre counted as "aimed"
CONFIRM_FRAMES   = 8      # consecutive aimed frames before firing

# Ballistic correction
# Measure your launcher muzzle speed and enter it here.
# Example: if it shoots ~20 m/s, enter 20_000.
# At 2m range with 20 m/s: correction ≈ 0.14°  (small but included for accuracy)
# At 2m range with  5 m/s: correction ≈ 2.2°   (significant at slow speeds)
PROJECTILE_SPEED_MMS = 20_000   # mm/s — MEASURE AND SET THIS

# Tennis ball physical size for distance estimation
BALL_DIAMETER_MM = 67.0   # standard tennis ball diameter in mm

# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print("Connecting to gimbal...")
    gimbal = GimbalController(
        port=SERIAL_PORT,
        baud=BAUD_RATE,
        pan_limit_deg=SWEEP_MAX_DEG / 2,
    )
    gimbal.home()
    gimbal.set_speed(SWEEP_SPEED_US)

    firing = FiringSystem(
        aim_tolerance_px=AIM_TOLERANCE_PX,
        confirm_frames=CONFIRM_FRAMES,
    )

    print("Opening camera...")
    cap = open_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)

    # Sweep state
    sweep_angle = -(SWEEP_MAX_DEG / 2)
    sweep_dir   = 1   # +1 = sweeping right (CW), -1 = sweeping left

    state = "SWEEP"
    print(f"Ready. State: {state}  |  Press Q to quit, R to reset.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read error — check connection.")
            break

        h, w = frame.shape[:2]
        detection = detect_ball_hsv(frame)
        draw_detection(frame, detection)
        draw_crosshair(frame)

        # ── State machine ──────────────────────────────────────────────────

        if state == "SWEEP":
            cv2.putText(frame, f"SWEEP  pan={sweep_angle:.0f}deg",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 165, 255), 2)

            if detection is not None:
                cx, cy, _ = detection
                print(f"Ball found at pixel ({cx:.0f}, {cy:.0f})")
                gimbal.set_speed(AIM_SPEED_US)
                firing.reset()
                state = "AIM"
            else:
                # Advance sweep angle
                sweep_angle += sweep_dir * SWEEP_STEP_DEG
                half = SWEEP_MAX_DEG / 2
                if sweep_angle > half:
                    sweep_angle = half
                    sweep_dir = -1
                elif sweep_angle < -half:
                    sweep_angle = -half
                    sweep_dir = 1
                gimbal.pan_to(sweep_angle)

        elif state == "AIM":
            if detection is None:
                print("Ball lost — returning to SWEEP")
                gimbal.set_speed(SWEEP_SPEED_US)
                firing.reset()
                state = "SWEEP"
            else:
                cx, cy, radius = detection

                # Estimate distance from known ball size, then compute
                # ballistic correction for projectile drop over that range
                distance_mm = GimbalController.estimate_distance(
                    radius, w, HFOV_DEG, BALL_DIAMETER_MM
                )
                if distance_mm is not None:
                    correction = GimbalController.ballistic_correction_deg(
                        distance_mm, PROJECTILE_SPEED_MMS
                    )
                else:
                    correction = 0.0   # ball too small to range — no correction

                # Compute where the gimbal should point based on pixel error
                pan_offset  = GimbalController.pixel_to_pan_offset(cx, w, HFOV_DEG)
                target_pan  = gimbal.current_pan_deg + pan_offset
                target_tilt = GimbalController.pixel_to_tilt(
                    cy, h, VFOV_DEG,
                    ballistic_correction_deg=correction
                )
                gimbal.aim(target_pan, target_tilt)

                result = firing.update(cx, cy, w, h)
                dist_str = f"{distance_mm/1000:.2f}m" if distance_mm else "?"
                cv2.putText(
                    frame,
                    f"AIM  err=({result['pan_error_px']:+.0f},{result['tilt_error_px']:+.0f})px"
                    f"  d={dist_str}  +{correction:.2f}deg  [{result['action']}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                )

                if result["action"] == "FIRE":
                    state = "FIRE"

        elif state == "FIRE":
            print(">>> FIRE <<<")
            gimbal.fire()
            state = "DONE"

        elif state == "DONE":
            cv2.putText(frame, "DONE  |  R = reset   Q = quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 0), 2)

        # State label at bottom
        cv2.putText(frame, f"State: {state}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1)

        cv2.imshow("Turret  —  Q quit  R reset", frame)

        # ── Key handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Reset — back to SWEEP")
            gimbal.home()
            gimbal.set_speed(SWEEP_SPEED_US)
            sweep_angle = -(SWEEP_MAX_DEG / 2)
            sweep_dir   = 1
            firing.reset()
            state = "SWEEP"

    gimbal.home()
    gimbal.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
