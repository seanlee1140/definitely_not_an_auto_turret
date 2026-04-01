"""
Camera Utilities — Single Camera
==================================

Camera opening and tennis ball detection for the single-camera turret.
The stereo rectification / triangulation code has been removed; the
turret now aims using pixel position directly.

Dependencies:
    opencv-python (cv2)
    numpy
"""

import cv2
import numpy as np


def open_camera(index: int = 0, width: int = 640, height: int = 480,
                fps: int = 30) -> cv2.VideoCapture:
    """
    Open a USB (UVC) or built-in camera.

    Args:
        index: Device index passed to cv2.VideoCapture.
        width: Requested capture width in pixels.
        height: Requested capture height in pixels.
        fps: Requested frame rate.

    Returns:
        Opened cv2.VideoCapture.

    Raises:
        RuntimeError: If the camera cannot be opened.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    return cap


def detect_ball_hsv(
    frame,
    lower: tuple = (25, 100, 100),
    upper: tuple = (65, 255, 255),
    min_area: float = 200.0,
):
    """
    Detect a tennis ball via HSV colour thresholding.

    The default HSV range targets the yellow-green of a tennis ball.
    Tune lower/upper for your lighting and ball colour:
        Yellow-green tennis ball : lower=(25,100,100)  upper=(65,255,255)
        Orange ball              : lower=(5,150,150)   upper=(25,255,255)

    Args:
        frame: BGR image from cv2.VideoCapture.read().
        lower: HSV lower bound (H 0-179, S 0-255, V 0-255).
        upper: HSV upper bound.
        min_area: Minimum contour area in px² — filters noise.

    Returns:
        (cx, cy, radius) — centroid and approximate radius in pixels,
        or None if no ball is found.
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx     = M["m10"] / M["m00"]
    cy     = M["m01"] / M["m00"]
    radius = np.sqrt(cv2.contourArea(largest) / np.pi)
    return (cx, cy, radius)


def draw_detection(frame, detection, color: tuple = (0, 255, 0)) -> None:
    """
    Draw detection overlay on a frame in-place.

    Args:
        frame: BGR image (modified in-place).
        detection: (cx, cy, radius) from detect_ball_hsv, or None.
        color: BGR color for the circle and label.
    """
    if detection is None:
        return
    cx, cy, radius = detection
    cv2.circle(frame, (int(cx), int(cy)), int(radius), color, 2)
    cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)
    cv2.putText(frame, f"({int(cx)},{int(cy)})",
                (int(cx) + 8, int(cy) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_crosshair(frame, color: tuple = (255, 80, 0), size: int = 20) -> None:
    """Draw a crosshair at the frame centre."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 1)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 1)
