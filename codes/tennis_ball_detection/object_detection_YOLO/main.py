from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tennis_ball_detection.object_detection_common.cli import build_live_detection_parser
from tennis_ball_detection.object_detection_common.runtime import run_live_detection
from tennis_ball_detection.object_detection_YOLO.backend import YoloDetector


def resolve_setup_script_path() -> Path:
    return Path(__file__).resolve().parent / "setup_jetson_yolo.sh"


def ensure_gpu_device(device: str) -> str:
    if str(device).strip().lower() == "cpu":
        raise RuntimeError("This Jetson YOLO entrypoint is GPU-only. Use --device 0.")
    return device


def ensure_engine_model_path(model_path: Path) -> Path:
    if model_path.suffix != ".engine":
        raise RuntimeError(
            "This Jetson YOLO entrypoint is GPU-only and expects a TensorRT '.engine' model file. "
            "Run object_detection_YOLO/setup_jetson_yolo.sh or pass a custom .engine path."
        )
    return model_path


def ensure_model_path_ready(
    model_path: Path,
    *,
    default_model_path: Path,
    setup_script_path: Path | None = None,
) -> Path:
    if model_path.exists():
        return model_path

    setup_script = setup_script_path or resolve_setup_script_path()
    if model_path.resolve() == default_model_path.resolve():
        raise RuntimeError(
            "TensorRT engine not found at "
            f"{model_path}. Run {setup_script} to install the editable package and "
            "export best.engine from best.pt before starting the live camera."
        )

    raise RuntimeError(f"Model file not found: {model_path}")


def main(argv=None) -> int:
    default_model_path = (
        Path(__file__).resolve().parents[1] / "model" / "tennis-ball-detection" / "best.engine"
    )
    parser = build_live_detection_parser(
        description="Live Jetson CSI tennis-ball detection using the local YOLO TensorRT engine.",
        default_model_path=default_model_path,
    )
    args = parser.parse_args(argv)

    try:
        ensure_gpu_device(args.device)
        ensure_engine_model_path(args.model_path)
        ensure_model_path_ready(args.model_path, default_model_path=default_model_path)
        detector = YoloDetector(
            model_path=args.model_path,
            confidence_threshold=args.confidence_threshold,
            device=args.device,
            imgsz=args.imgsz,
            task="detect",
        )
        run_live_detection(args, detector, window_title="Tennis Ball Detection YOLO")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
