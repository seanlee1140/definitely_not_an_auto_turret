"""
Firing Decision System — Stationary Ball
==========================================

Since the ball is stationary, there is no intercept calculation.
The firing logic is purely:

    1. Ball is detected in frame
    2. Compute pixel error from frame centre
    3. If error < tolerance for N consecutive frames → FIRE

No Kalman filter, no trajectory prediction, no stereo depth needed.
"""


class FiringSystem:
    """
    State machine that decides when to fire at a stationary target.

    Args:
        aim_tolerance_px: Ball centroid must be within this many pixels
            of the frame centre (in both axes) to count as "aimed".
        confirm_frames: Number of consecutive aimed frames required
            before a FIRE command is issued.  Prevents firing on a
            momentary fluke detection.
    """

    def __init__(self, aim_tolerance_px: int = 15, confirm_frames: int = 8):
        self.aim_tolerance_px = aim_tolerance_px
        self.confirm_frames   = confirm_frames
        self._aimed_count     = 0

    def update(self, cx: float, cy: float,
               frame_width: int, frame_height: int) -> dict:
        """
        Evaluate current aim and update state.  Call once per frame
        when the ball is visible.

        Args:
            cx: Ball centroid x-pixel.
            cy: Ball centroid y-pixel.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            dict:
                action        : "TRACK" | "AIM" | "FIRE"
                aimed         : bool — within pixel tolerance this frame
                confirmed     : bool — aimed for enough frames to fire
                pan_error_px  : horizontal pixel error (+ve = ball right of centre)
                tilt_error_px : vertical pixel error (+ve = ball below centre)
        """
        pan_err  = cx - frame_width  / 2.0
        tilt_err = cy - frame_height / 2.0

        aimed = (abs(pan_err)  < self.aim_tolerance_px and
                 abs(tilt_err) < self.aim_tolerance_px)

        if aimed:
            self._aimed_count += 1
        else:
            self._aimed_count = 0

        confirmed = self._aimed_count >= self.confirm_frames

        if confirmed:
            action = "FIRE"
        elif aimed:
            action = "AIM"
        else:
            action = "TRACK"

        return {
            "action":        action,
            "aimed":         aimed,
            "confirmed":     confirmed,
            "pan_error_px":  pan_err,
            "tilt_error_px": tilt_err,
        }

    def reset(self) -> None:
        """Reset the confirmation counter (call when ball is lost or after firing)."""
        self._aimed_count = 0
