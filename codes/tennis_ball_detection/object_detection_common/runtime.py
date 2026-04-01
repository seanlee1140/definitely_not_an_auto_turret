from __future__ import annotations

try:
    import cv2
except ImportError:  # pragma: no cover - exercised on device
    cv2 = None

from .camera import open_jetson_camera
from .overlay import draw_detection_overlay
from .types import DetectionResult, best_detection


def format_detection_summary(detection: DetectionResult) -> str:
    x1, y1, x2, y2 = detection.bbox
    cx, cy = detection.center
    return (
        f"detected label={detection.label} "
        f"confidence={detection.confidence:.3f} "
        f"center=({cx}, {cy}) "
        f"bbox=({x1}, {y1}, {x2}, {y2})"
    )


def run_live_detection(args, detector, *, window_title: str):
    detector.load()

    if cv2 is None:
        raise RuntimeError("OpenCV is required to run the live camera application.")

    capture = open_jetson_camera(
        sensor_id=args.sensor_id,
        width=args.width,
        height=args.height,
        fps=args.fps,
        display_width=args.display_width,
        display_height=args.display_height,
        flip_method=args.flip_method,
    )

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the Jetson CSI camera.")

            detections = detector.predict(frame)
            detection = best_detection(detections)
            display_frame = frame.copy() if hasattr(frame, "copy") else frame

            if detection is not None:
                print(format_detection_summary(detection), flush=True)
                draw_detection_overlay(display_frame, detection)

            cv2.imshow(window_title, display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if hasattr(capture, "release"):
            capture.release()
        cv2.destroyAllWindows()
