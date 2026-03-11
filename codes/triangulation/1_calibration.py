'''
Calibrate the cameras

Prereq:
    - Checkerboard
    - Mounted camera (recalibrate if camera changes)

Usage:
    1. RUn scri[t
    2. Hold checkboard in view for both cam
    3. SPACE to capture
    4. move board (new position/angle)
    5. Press Q when done (the more the better but try 20+)

Tips from AI for good calibration images:
    - Cover the entire frame: center, corners, edges
    - Vary the distance: close up and far away
    - Tilt the board in different directions
    - Make sure the FULL checkerboard is visible in BOTH cameras for 
      every capture
    - Avoid motion blur — hold the board still when you press SPACE

Output:
    - calib_left/img_000.png, img_001.png, ...
    - calib_right/img_000.png, img_001.png, ...
    These paired images are used by calibrate_stereo.py.
'''

import cv2
from pathlib import Path

# Settings
CALIB_LEFT = Path("calib_left")
CALIB_RIGHT = Path("calib_right")

def gstreamer_pipeline(sensor_id=0, width = 1920, height = 1080, fps = 30,
                       display_width = 960, display_height = 540):
    ''' Build Gstreamer for capturing in Jetson'''

    return(
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},"
        f"framerate={fps}/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw,width={display_width},height={display_height},"
        f"format=BGRx ! videoconvert ! "
        f"video/x-raw,format=BGR ! appsink"
    )

def main():
    """Capture synchronized stereo image pairs for calibration"""
    CALIB_LEFT.mkdir(parents=True, exist_ok=True)
    CALIB_RIGHT.mkdir(parents=True, exist_ok=True)

    cap_left = cv2.VideoCapture(gstreamer_pipeline(sensor_id = 0), cv2.CAP_GSTREAMER)
    cap_right = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1), cv2.CAP_GSTREAMER)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Failed to open both cameras")
        return
    
    count = 0
    print("Hold checkerboard in view of both cameras")
    print("Press SPACE to capture, Q to quit (after 20 pics)")

    while True:
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()

        if not ret_l or not ret_r:
            print("Failed to grab frame")
            break

        combined = cv2.hconcat([frame_l, frame_r])
        cv2.imshow("Left | Right", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            cv2.imwrite(str(CALIB_LEFT / f"img_{count:03d}.png"), frame_l)
            cv2.imwrite(str(CALIB_RIGHT / f"img_{count:03d}.png"), frame_r)
            count += 1
            print(f"Captured pair {count}")

        elif key == ord('q'):
            break 

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
    print(f"\nDone, captured {count} images")


if __name__ == "__main__":
    main()