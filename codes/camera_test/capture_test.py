import cv2

def gstreamer_pipeline(sensor_id = 0, width = 1920, height = 1080, fps = 30, 
                       display_width = 960, display_height = 540):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},"
        f"framerate={fps}/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw,width={display_width},height={display_height},"
        f"format=BGRx ! videoconvert ! "
        f"video/x-raw,format=BGR ! appsink"
    )

cap_left = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)
cap_right = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1), cv2.CAP_GSTREAMER)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("Failed to open some cameras")
    exit()

while True:
    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()

    if not ret_l or not ret_r:
        print("Failed to get thee frame")
        break

    # Show side by side
    combined = cv2.hconcat([frame_l, frame_r])
    cv2.imshow("Left | Right", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()