from __future__ import annotations

from .types import DetectionResult

try:
    import cv2
except ImportError:  # pragma: no cover - exercised on device
    cv2 = None


def draw_detection_overlay(frame, detection: DetectionResult | None, *, cv2_module=None):
    if detection is None:
        return frame

    renderer = cv2_module or cv2
    if renderer is None:
        raise RuntimeError("OpenCV is required to draw detection overlays.")

    x1, y1, x2, y2 = detection.bbox
    cx, cy = detection.center
    label_text = f"{detection.label} {detection.confidence:.2f}"

    renderer.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    renderer.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    renderer.putText(
        frame,
        label_text,
        (x1, max(y1 - 10, 0)),
        getattr(renderer, "FONT_HERSHEY_SIMPLEX", 0),
        0.7,
        (0, 255, 0),
        2,
    )
    return frame
