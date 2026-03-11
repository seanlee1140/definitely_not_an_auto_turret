'''
Processes the image pairs captured by capture_calibration.py and computes:
    — Intrinsic calibration (each camera's focal length, optical 
        center, and lens distortion)
    — Extrinsic stereo calibration (rotation and translation 
        between the two cameras)

Prerequisites:
    - Image pairs in calib_left/ and calib_right/ from capture_calibration.py
    - Update CHECKERBOARD and SQUARE_SIZE below:
        CHECKERBOARD = number of INNER corners (intersections, not squares)
        SQUARE_SIZE = physical size of one square in mm (measure with a ruler)

Output:
    - stereo_calibration.npz containing:
      K1, dist1, K2, dist2, R, T, E, F, image_size

Quality checks:
    - Reprojection errors should be < 0.5 px (definitely < 1.0 px)
    - Baseline distance (norm of T) should match physical camera spacing
'''

import cv2
import numpy as np
from pathlib import Path

# Edit these parameters
CHECKERBOARD = (9, 6)   # inner corners (count intersections, not squares)
SQUARE_SIZE = 25.0       # mm — measure your printed board with a ruler

CALIB_LEFT = Path("calib_left")
CALIB_RIGHT = Path("calib_right")
OUTPUT_FILE = Path("stereo_calibration.npz")

def find_checkerboard_points(left_images, right_images, checkerboard, square_size):
    """
    Detect checkerboard corners in all image pairs.

    Iterates through paired left/right images, finds the checkerboard in
    both, and collects 2D corner positions with corresponding 3D object points.

    Args:
        left_images: Sorted list of Paths to left camera images.
        right_images: Sorted list of Paths to right camera images.
        checkerboard: Tuple (cols, rows) of inner corner counts.
        square_size: Physical size of one square in mm.

    Returns:
        obj_points: List of 3D object point arrays (one per usable pair).
        img_points_left: List of 2D corner arrays from the left camera.
        img_points_right: List of 2D corner arrays from the right camera.
        image_size: (width, height) tuple.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []
    img_points_left = []
    img_points_right = []
    image_size = None

    for l_path, r_path in zip(left_images, right_images):
        img_l = cv2.imread(str(l_path))
        img_r = cv2.imread(str(r_path))
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray_l.shape[::-1]

        ret_l, corners_l = cv2.findChessboardCorners(gray_l, checkerboard, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, checkerboard, None)

        if ret_l and ret_r:
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points_left.append(corners_l)
            img_points_right.append(corners_r)
            print(f"  OK: {l_path.name}")
        else:
            print(f"  SKIPPED (board not found in both): {l_path.name}")

    return obj_points, img_points_left, img_points_right, image_size


def calibrate_individual(obj_points, img_points, image_size, camera_name):
    """
    Intrinsic calibration for a single camera.

    Args:
        obj_points: List of 3D checkerboard point arrays.
        img_points: List of 2D detected corner arrays.
        image_size: (width, height) of the images.
        camera_name: Label for output (e.g. "left" or "right").

    Returns:
        K: 3x3 camera matrix.
        dist: Distortion coefficients.
    """
    print(f"\n--- Calibrating {camera_name} camera ---")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    print(f"{camera_name.capitalize()} reprojection error: {ret:.4f} px")
    if ret > 1.0:
        print(f"  WARNING: Error is high. Consider removing bad image pairs.")
    return K, dist


def calibrate_stereo(obj_points, img_points_left, img_points_right,
                     K1, dist1, K2, dist2, image_size):
    """
    Stereo calibration to find rotation and translation between cameras.

    Args:
        obj_points: List of 3D checkerboard point arrays.
        img_points_left: 2D corners from left camera.
        img_points_right: 2D corners from right camera.
        K1, dist1: Left camera intrinsics.
        K2, dist2: Right camera intrinsics.
        image_size: (width, height) of the images.

    Returns:
        R: 3x3 rotation matrix (right camera relative to left).
        T: 3x1 translation vector in mm.
        E: Essential matrix.
        F: Fundamental matrix.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("\n--- Stereo calibration ---")
    ret, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_left, img_points_right,
        K1, dist1, K2, dist2, image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    print(f"Stereo reprojection error: {ret:.4f} px")
    if ret > 1.0:
        print("  WARNING: Stereo error is high. Check image pair quality.")
    return R, T, E, F


def main():
    """Run full calibration pipeline and save results."""
    left_images = sorted(CALIB_LEFT.glob("*.png"))
    right_images = sorted(CALIB_RIGHT.glob("*.png"))

    print(f"Found {len(left_images)} left and {len(right_images)} right images\n")

    if len(left_images) == 0 or len(left_images) != len(right_images):
        print("Error: Need equal number of left and right images.")
        print("Run capture_calibration.py first.")
        return

    obj_points, img_points_left, img_points_right, image_size = \
        find_checkerboard_points(left_images, right_images, CHECKERBOARD, SQUARE_SIZE)

    print(f"\nUsable pairs: {len(obj_points)}")
    if len(obj_points) < 10:
        print("WARNING: Fewer than 10 usable pairs. Results may be poor.\n")

    # Step 1: Intrinsic calibration
    K1, dist1 = calibrate_individual(obj_points, img_points_left, image_size, "left")
    K2, dist2 = calibrate_individual(obj_points, img_points_right, image_size, "right")

    # Step 2: Stereo calibration
    R, T, E, F = calibrate_stereo(
        obj_points, img_points_left, img_points_right,
        K1, dist1, K2, dist2, image_size
    )

    # Save
    np.savez(str(OUTPUT_FILE),
        K1=K1, dist1=dist1, K2=K2, dist2=dist2,
        R=R, T=T, E=E, F=F, image_size=image_size
    )

    # Summary
    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Usable image pairs: {len(obj_points)}")
    print(f"Image size: {image_size}")
    print(f"\nLeft camera matrix:\n{K1}")
    print(f"\nRight camera matrix:\n{K2}")
    print(f"\nRotation between cameras:\n{R}")
    print(f"\nTranslation between cameras (mm): {T.flatten()}")
    print(f"Baseline distance: {np.linalg.norm(T):.1f} mm")
    print(f"\nSanity check: does the baseline distance roughly")
    print(f"match the physical distance between your cameras?")
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()