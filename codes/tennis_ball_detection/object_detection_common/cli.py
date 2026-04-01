from __future__ import annotations

import argparse
from pathlib import Path


def build_live_detection_parser(
    *,
    description: str,
    default_model_path: Path,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--sensor-id", type=int, default=0, help="Jetson CSI sensor-id.")
    parser.add_argument("--width", type=int, default=1920, help="Camera capture width.")
    parser.add_argument("--height", type=int, default=1080, help="Camera capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Camera capture frame rate.")
    parser.add_argument(
        "--display-width",
        type=int,
        default=960,
        help="Width of the frame after Jetson nvvidconv conversion.",
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=540,
        help="Height of the frame after Jetson nvvidconv conversion.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_model_path,
        help="Path to the model file or model directory.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="Minimum confidence required to keep a detection.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device string for Jetson TensorRT inference, usually '0'.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square inference image size used by Ultralytics.",
    )
    parser.add_argument(
        "--flip-method",
        type=int,
        default=0,
        help="Jetson nvvidconv flip-method value for the CSI camera pipeline.",
    )
    return parser
