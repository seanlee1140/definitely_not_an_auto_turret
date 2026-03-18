"""Shared utilities for live tennis-ball detection apps."""

from .camera import CameraOpenError, build_gstreamer_pipeline, open_jetson_camera
from .overlay import draw_detection_overlay
from .runtime import format_detection_summary, run_live_detection
from .types import DetectionResult, best_detection

__all__ = [
    "CameraOpenError",
    "DetectionResult",
    "best_detection",
    "build_gstreamer_pipeline",
    "draw_detection_overlay",
    "format_detection_summary",
    "open_jetson_camera",
    "run_live_detection",
]
