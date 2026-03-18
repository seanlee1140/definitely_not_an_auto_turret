from __future__ import annotations

try:
    import cv2
except ImportError:  # pragma: no cover - exercised on device
    cv2 = None


class CameraOpenError(RuntimeError):
    """Raised when the Jetson CSI camera cannot be opened."""


def build_gstreamer_pipeline(
    sensor_id: int = 0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    display_width: int = 960,
    display_height: int = 540,
    flip_method: int = 0,
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},"
        f"framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw,width={display_width},height={display_height},"
        f"format=BGRx ! videoconvert ! "
        f"video/x-raw,format=BGR ! appsink"
    )


def open_jetson_camera(
    *,
    sensor_id: int,
    width: int,
    height: int,
    fps: int,
    display_width: int,
    display_height: int,
    video_capture_factory=None,
    api_preference=None,
):
    pipeline = build_gstreamer_pipeline(
        sensor_id=sensor_id,
        width=width,
        height=height,
        fps=fps,
        display_width=display_width,
        display_height=display_height,
    )

    if video_capture_factory is None:
        if cv2 is None:
            raise RuntimeError("OpenCV is required to open the Jetson CSI camera.")
        video_capture_factory = cv2.VideoCapture
        if api_preference is None:
            api_preference = cv2.CAP_GSTREAMER

    if api_preference is None:
        capture = video_capture_factory(pipeline)
    else:
        capture = video_capture_factory(pipeline, api_preference)

    if not capture.isOpened():
        if hasattr(capture, "release"):
            capture.release()
        raise CameraOpenError(
            "Failed to open Jetson CSI camera "
            f"on sensor-id={sensor_id}. Pipeline: {pipeline}"
        )

    return capture
