from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tennis_ball_detection.object_detection_common.cli import build_live_detection_parser
from tennis_ball_detection.object_detection_common.runtime import run_live_detection
from tennis_ball_detection.object_detection_YOLO.backend import YoloOnnxDetector


def main(argv=None) -> int:
    default_model_path = (
        Path(__file__).resolve().parents[1] / "model" / "tennis-ball-detection" / "best.onnx"
    )
    parser = build_live_detection_parser(
        description="Live Jetson CSI tennis-ball detection using the local YOLO ONNX model.",
        default_model_path=default_model_path,
    )
    args = parser.parse_args(argv)

    detector = YoloOnnxDetector(
        model_path=args.model_path,
        confidence_threshold=args.confidence_threshold,
    )

    try:
        run_live_detection(args, detector, window_title="Tennis Ball Detection YOLO")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
