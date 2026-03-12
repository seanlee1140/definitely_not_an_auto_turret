# YOLO Tennis Ball Detection

This project runs live tennis-ball detection on a Jetson CSI camera using the local ONNX model at `../model/tennis-ball-detection/best.onnx`.

## Features

- Uses the Jetson `nvarguscamerasrc` GStreamer pipeline.
- Loads the local YOLO ONNX checkpoint through Ultralytics.
- Draws the best tennis-ball detection with confidence and center point.
- Prints the best detection center coordinates and confidence to stdout.

## Install

```bash
pip install -r tennis_ball_detection/object_detection_YOLO/requirements.txt
```

## Run

From the repository root:

```bash
python -m tennis_ball_detection.object_detection_YOLO --sensor-id 0
```

Useful flags:

- `--model-path`
- `--sensor-id`
- `--width`
- `--height`
- `--fps`
- `--display-width`
- `--display-height`
- `--confidence-threshold`

Press `q` to exit the live window.
