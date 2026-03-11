"""
Live Stereo Triangulation Test
================================

Runs both cameras, rectifies frames, detects a ping pong ball using 
colour thresholding (simple and fast — no YOLO needed for testing), 
and triangulates its 3D position.

This is for testing and verifying calibration before integrating 
with YOLOv8 and the full ROS2 pipeline. The output is a live 3D 
coordinate (X, Y, Z) in mm printed to the console.

Prerequisites:
    - stereo_calibration.npz from calibrate_stereo.py
    - stereo_rectification.npz from compute_rectification.py
    - Both cameras connected and working

Usage:
    - Hold a ping pong ball in front of both cameras
    - The 3D position is printed each frame it's detected in both views
    - Press Q to quit

Replacing the detector:
    - The detect_ball() function uses simple HSV colour thresholding
    - Swap it out for your YOLOv8 detector when ready, the 
      triangulation code stays the same
"""

import cv2
import numpy as np
from pathlib import Path

CALIBRATION_FILE = Path("stereo_calibration.npz")
RECTIFICATION_FILE = Path("stereo_rectification.npz")


def gstreamer_pipeline(sensor_id=0, width=1920, height=1080, fps=30,
                       display_width=960, display_height=540):
    """Build GStreamer pipeline string for Jetson CSI camera."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},"
        f"framerate={fps}/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw,width={display_width},height={display_height},"
        f"format=BGRx ! videoconvert ! "
        f"video/x-raw,format=BGR ! appsink"
    )


def detect_ball(frame):
    """
    Detect a ping pong ball using HSV colour thresholding.

    This is a simple detector for testing triangulation. Replace with 
    YOLOv8 inference for the real system.

    Args:
        frame: BGR image from camera.

    Returns:
        (cx, cy) centroid of the detected ball, or None if not found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Orange/yellow ping pong ball range — adjust for your ball colour
    # For white balls, try: lower=(0, 0, 200), upper=(180, 40, 255)
    lower = np.array([15, 100, 100])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Pick the largest contour (most likely the ball)
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # Filter out noise (too small) and non-circular shapes
    if area < 100:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (cx, cy)


def triangulate(pt_left, pt_right, P1, P2):
    """
    Compute 3D position from corresponding 2D points in rectified images.

    Args:
        pt_left: (x, y) pixel coordinates in rectified left image.
        pt_right: (x, y) pixel coordinates in rectified right image.
        P1: 3x4 projection matrix for left camera.
        P2: 3x4 projection matrix for right camera.

    Returns:
        Numpy array [X, Y, Z] in mm (world coordinates relative to 
        left camera).
    """
    pts_l = np.array([[pt_left[0], pt_left[1]]], dtype=np.float64).T
    pts_r = np.array([[pt_right[0], pt_right[1]]], dtype=np.float64).T

    points_4d = cv2.triangulatePoints(P1, P2, pts_l, pts_r)
    point_3d = (points_4d[:3] / points_4d[3]).flatten()
    return point_3d


def main():
    """Run live stereo triangulation test."""
    # Load calibration and rectification data
    if not CALIBRATION_FILE.exists() or not RECTIFICATION_FILE.exists():
        print("Error: Run calibrate_stereo.py and compute_rectification.py first.")
        return

    rect_data = np.load(str(RECTIFICATION_FILE))
    P1 = rect_data['P1']
    P2 = rect_data['P2']
    map1_left = rect_data['map1_left']
    map2_left = rect_data['map2_left']
    map1_right = rect_data['map1_right']
    map2_right = rect_data['map2_right']

    # Open cameras
    cap_left = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)
    cap_right = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1), cv2.CAP_GSTREAMER)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Failed to open cameras.")
        return

    print("Hold a ping pong ball in view of both cameras.")
    print("3D position will print when detected in both views.")
    print("Press Q to quit.\n")

    while True:
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()

        if not ret_l or not ret_r:
            print("Failed to grab frame.")
            break

        # Rectify both frames
        rect_left = cv2.remap(frame_l, map1_left, map2_left, cv2.INTER_LINEAR)
        rect_right = cv2.remap(frame_r, map1_right, map2_right, cv2.INTER_LINEAR)

        # Detect ball in both rectified frames
        det_left = detect_ball(rect_left)
        det_right = detect_ball(rect_right)

        # Draw detections for visual feedback
        display_left = rect_left.copy()
        display_right = rect_right.copy()

        if det_left is not None:
            cv2.circle(display_left, (int(det_left[0]), int(det_left[1])), 10, (0, 255, 0), 2)
        if det_right is not None:
            cv2.circle(display_right, (int(det_right[0]), int(det_right[1])), 10, (0, 255, 0), 2)

        # Triangulate if detected in both
        if det_left is not None and det_right is not None:
            point_3d = triangulate(det_left, det_right, P1, P2)
            x, y, z = point_3d
            print(f"3D Position — X: {x:7.1f} mm  Y: {y:7.1f} mm  Z: {z:7.1f} mm")

            # Show coordinates on display
            text = f"X:{x:.0f} Y:{y:.0f} Z:{z:.0f} mm"
            cv2.putText(display_left, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        combined = cv2.hconcat([display_left, display_right])
        cv2.imshow("Stereo Triangulation Test - Q to quit", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()