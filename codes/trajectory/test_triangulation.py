"""
Live Stereo Triangulation Test
================================

Runs both cameras, rectifies frames, detects a ball using HSV colour 
thresholding, and triangulates its 3D position. For testing calibration 
before integrating with YOLOv8.

Prerequisites:
    - stereo_calibration.npz from calibrate_stereo.py
    - stereo_rectification.npz from compute_rectification.py
    - Both cameras connected

Usage:
    - Hold a coloured ball in view of both cameras
    - 3D position prints each frame it's detected in both views
    - Press Q to quit
"""

import cv2
import numpy as np
from camera_utils import (
    gstreamer_pipeline, load_rectification, rectify_frames,
    triangulate, detect_ball_hsv
)


def main():
    """Run live stereo triangulation test."""
    rect_data = load_rectification()
    P1, P2 = rect_data['P1'], rect_data['P2']

    cap_left = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)
    cap_right = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1), cv2.CAP_GSTREAMER)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Failed to open cameras.")
        return

    print("Hold a ball in view of both cameras.")
    print("3D position prints when detected in both views.")
    print("Press Q to quit.\n")

    while True:
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()

        if not ret_l or not ret_r:
            print("Failed to grab frame.")
            break

        rect_left, rect_right = rectify_frames(frame_l, frame_r, rect_data)

        det_left = detect_ball_hsv(rect_left)
        det_right = detect_ball_hsv(rect_right)

        display_left = rect_left.copy()
        display_right = rect_right.copy()

        if det_left is not None:
            cv2.circle(display_left, (int(det_left[0]), int(det_left[1])),
                       10, (0, 255, 0), 2)
        if det_right is not None:
            cv2.circle(display_right, (int(det_right[0]), int(det_right[1])),
                       10, (0, 255, 0), 2)

        if det_left is not None and det_right is not None:
            point_3d = triangulate(det_left, det_right, P1, P2)
            x, y, z = point_3d
            print(f"3D: X={x:7.1f}  Y={y:7.1f}  Z={z:7.1f} mm")

            text = f"X:{x:.0f} Y:{y:.0f} Z:{z:.0f} mm"
            cv2.putText(display_left, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        combined = cv2.hconcat([display_left, display_right])
        cv2.imshow("Triangulation Test - Q to quit", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()