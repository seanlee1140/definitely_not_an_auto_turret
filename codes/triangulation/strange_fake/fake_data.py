"""
Generate fake stereo calibration data for pipeline testing.
This is NOT a real calibration — replace with actual calibration before 
any real use.
"""

import cv2
import numpy as np
from pathlib import Path

# Image size (must match your camera capture resolution)
IMAGE_SIZE = (960, 540)

# Fake but reasonable intrinsics for IMX219 at 960x540
# Focal length ~500px, optical center at image center
K1 = np.array([
    [500.0,   0.0, 480.0],
    [  0.0, 500.0, 270.0],
    [  0.0,   0.0,   1.0]
])
K2 = K1.copy()

# Minimal distortion
dist1 = np.array([0.05, -0.1, 0.0, 0.0, 0.0])
dist2 = np.array([0.05, -0.1, 0.0, 0.0, 0.0])

# Cameras ~200mm apart horizontally (adjust to your actual spacing)
R = np.eye(3)  # no rotation between cameras
T = np.array([[200.0], [0.0], [0.0]])  # 200mm baseline

# Essential and fundamental matrices (derived from R, T, K)
T_cross = np.array([
    [0, -T[2, 0], T[1, 0]],
    [T[2, 0], 0, -T[0, 0]],
    [-T[1, 0], T[0, 0], 0]
])
E = T_cross @ R
F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

# Save calibration
np.savez("stereo_calibration.npz",
    K1=K1, dist1=dist1, K2=K2, dist2=dist2,
    R=R, T=T, E=E, F=F, image_size=np.array(IMAGE_SIZE)
)
print("Saved stereo_calibration.npz")

# Also generate rectification maps
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, dist1, K2, dist2, IMAGE_SIZE, R, T, alpha=0
)
map1_left, map2_left = cv2.initUndistortRectifyMap(
    K1, dist1, R1, P1, IMAGE_SIZE, cv2.CV_32FC1
)
map1_right, map2_right = cv2.initUndistortRectifyMap(
    K2, dist2, R2, P2, IMAGE_SIZE, cv2.CV_32FC1
)

np.savez("stereo_rectification.npz",
    P1=P1, P2=P2, Q=Q,
    map1_left=map1_left, map2_left=map2_left,
    map1_right=map1_right, map2_right=map2_right,
    roi1=roi1, roi2=roi2
)
print("Saved stereo_rectification.npz")
print("\nThese are FAKE values for testing only.")
print("Recalibrate with a printed checkerboard before real use.")