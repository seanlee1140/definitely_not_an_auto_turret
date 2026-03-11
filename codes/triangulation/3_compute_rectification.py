"""
Takes the calibration data from calibrate_stereo.py and precomputes
rectification maps. These maps are used every frame during live 
triangulation to undistort and align both camera images so that 
corresponding points lie on the same horizontal line.

Prerequisites:
    - stereo_calibration.npz from calibrate_stereo.py

Output:
    - stereo_rectification.npz containing:
        P1, P2    — projection matrices for triangulation
        map1_left, map2_left   — rectification maps for left camera
        map1_right, map2_right — rectification maps for right camera

This only needs to be run once every calirbation.
"""

import cv2
import numpy as np
from pathlib import Path

CALIBRATION_FILE = Path("stereo_calibration.npz")
OUTPUT_FILE = Path("stereo_rectification.npz")

def main():
    if not CALIBRATION_FILE.exists():
        print(f"Error: {CALIBRATION_FILE} not found. Run calibrate_stereo.py")
        return
    
    data = np.load(str(CALIBRATION_FILE))
    K1, dist1 = data['K1'], data['dist1']
    K2, dist2 = data['K2'], data['dist2']
    R, T = data['R'], data['T']
    image_size = tuple(data['image_size'])

    print(f"Loaded calibration for image size {image_size}")
    print(f"baseline: {np.linalg.norm(T):.1f} mm")

    # Compute rectification transforms
    # alpha=0 crops to valid pixels only (no black borders)
    # alpha=1 keeps all pixels (may have black borders)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, dist1, K2, dist2, image_size, R, T, alpha = 0
    )

    map1_left, map2_left = cv2.initUndistortRectifyMap(
        K1, dist1, R1, P1, image_size, cv2.CV_32FC1
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        K2, dist2, R2, P2, image_size, cv2.CV_32FC1
    )

    np.savez(str(OUTPUT_FILE),
        P1=P1, P2=P2, Q=Q,
        map1_left=map1_left, map2_left=map2_left,
        map1_right=map1_right, map2_right=map2_right,
        roi1=roi1, roi2=roi2
    )

    print(f"\nProjection matrix P1:\n{P1}")
    print(f"\nProjection matrix P2:\n{P2}")
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()