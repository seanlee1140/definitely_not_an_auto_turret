# YOLO Tennis Ball Detection

This project runs live tennis-ball detection on a Jetson CSI camera using a GPU-only TensorRT engine at `../model/tennis-ball-detection/best.engine`.

## Features

- Uses the Jetson `nvarguscamerasrc` GStreamer pipeline.
- Uses a TensorRT-only YOLO runtime on Jetson.
- Builds an end-to-end TensorRT engine with NMS inside the engine, so live detection does not fall back to CPU post-processing.
- Draws the best tennis-ball detection with confidence and center point.
- Prints the best detection center coordinates and confidence to stdout.

## Jetson Setup

```bash
source /home/jetson/repository/definitely_not_an_auto_turret/codes/tennis_ball_detection/tennis_ball_detection/.venv-yolo/bin/activate
./codes/tennis_ball_detection/object_detection_YOLO/setup_jetson_yolo.sh
```

The setup script does a one-time ONNX export on the host, then builds a metadata-aware `best.engine`. Live camera detection uses the GPU-only TensorRT engine.

## Run

From the repository root:

```bash
source /home/jetson/repository/definitely_not_an_auto_turret/codes/tennis_ball_detection/tennis_ball_detection/.venv-yolo/bin/activate
tennis-ball-yolo --sensor-id 0
```

Useful flags:

- `--model-path`
- `--sensor-id`
- `--width`
- `--height`
- `--fps`
- `--display-width`
- `--display-height`
- `--flip-method`
- `--device`
- `--imgsz`
- `--confidence-threshold`

The live entrypoint only accepts TensorRT `.engine` models. Keep `--device 0` for the Jetson GPU.

The default `--imgsz` is `640`, which is a more practical TensorRT build and runtime size for this Jetson-class device than the source model's original `1280` export size.

After `setup_jetson_yolo.sh` installs the package editable, the module entrypoint also works:

```bash
python -m tennis_ball_detection.object_detection_YOLO --sensor-id 0
```

Press `q` to exit the live window.
