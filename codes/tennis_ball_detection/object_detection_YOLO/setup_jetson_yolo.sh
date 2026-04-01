#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PACKAGE_ROOT}/tennis_ball_detection/.venv-yolo"
PYTHON_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"
MODEL_DIR="${PACKAGE_ROOT}/model/tennis-ball-detection"
PT_MODEL="${MODEL_DIR}/best.pt"
ONNX_MODEL="${MODEL_DIR}/best.onnx"
ENGINE_MODEL="${MODEL_DIR}/best.engine"
ENGINE_IMGSZ="${YOLO_ENGINE_IMGSZ:-640}"
TRT_WORKSPACE_MIB="${YOLO_TRT_WORKSPACE_MIB:-512}"
TRT_AVG_TIMING="${YOLO_TRT_AVG_TIMING:-1}"
TRT_OPT_LEVEL="${YOLO_TRT_OPT_LEVEL:-0}"
RAW_ENGINE_MODEL="$(mktemp /tmp/tennis-ball-best.XXXXXX.engine)"

cleanup() {
    rm -f "${RAW_ENGINE_MODEL}"
}

trap cleanup EXIT

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating uv environment at ${VENV_DIR}"
    uv venv --python 3.10 --system-site-packages "${VENV_DIR}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found in ${VENV_DIR}" >&2
    exit 1
fi

if [[ ! -f "${PT_MODEL}" ]]; then
    echo "Expected PyTorch checkpoint not found at ${PT_MODEL}" >&2
    exit 1
fi

export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PATH="/usr/src/tensorrt/bin:${PATH}"

"${PIP_BIN}" install --no-build-isolation --no-deps -e "${PACKAGE_ROOT}"

"${PYTHON_BIN}" - <<'PY'
import cv2
import tensorrt as trt
import torch

print(f"Using OpenCV {cv2.__version__} from {cv2.__file__}")
print(f"Using TensorRT {trt.__version__}")
print(f"Using PyTorch {torch.__version__} with CUDA {torch.version.cuda}")

if "GStreamer:                   YES" not in cv2.getBuildInformation():
    raise SystemExit("OpenCV import succeeded, but it was not built with GStreamer support.")

if not torch.cuda.is_available():
    raise SystemExit(
        "torch.cuda.is_available() is False. Run this script directly on the Jetson host with GPU access."
    )
PY

if [[ ! -x "/usr/src/tensorrt/bin/trtexec" ]]; then
    echo "TensorRT trtexec was not found at /usr/src/tensorrt/bin/trtexec" >&2
    exit 1
fi

"${PYTHON_BIN}" - <<PY
from pathlib import Path

from ultralytics import YOLO

pt_model = Path("${PT_MODEL}")
onnx_model = Path("${ONNX_MODEL}")
imgsz = int("${ENGINE_IMGSZ}")

model = YOLO(str(pt_model), task="detect")
exported_path = model.export(
    format="onnx",
    device="cpu",
    imgsz=imgsz,
    opset=20,
    simplify=False,
    nms=True,
    verbose=True,
)
print(f"ONNX export reported: {exported_path}")

if not onnx_model.exists():
    raise SystemExit(f"Expected ONNX model was not written to {onnx_model}")
PY

/usr/src/tensorrt/bin/trtexec \
    --onnx="${ONNX_MODEL}" \
    --saveEngine="${RAW_ENGINE_MODEL}" \
    --fp16 \
    --skipInference \
    --builderOptimizationLevel="${TRT_OPT_LEVEL}" \
    --avgTiming="${TRT_AVG_TIMING}" \
    --memPoolSize="workspace:${TRT_WORKSPACE_MIB}" \
    --profilingVerbosity=none

"${PYTHON_BIN}" - <<PY
import ast
import json
from pathlib import Path

import onnx

onnx_model = Path("${ONNX_MODEL}")
raw_engine_model = Path("${RAW_ENGINE_MODEL}")
engine_model = Path("${ENGINE_MODEL}")

model = onnx.load(str(onnx_model))
metadata = {}
for item in model.metadata_props:
    value = item.value
    if item.key in {"args", "end2end", "imgsz", "kpt_names", "kpt_shape", "names"}:
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            pass
    elif item.key in {"batch", "channels", "stride"}:
        try:
            value = int(value)
        except ValueError:
            pass
    metadata[item.key] = value

meta = json.dumps(metadata).encode("utf-8")
with raw_engine_model.open("rb") as source, engine_model.open("wb") as target:
    target.write(len(meta).to_bytes(4, byteorder="little", signed=True))
    target.write(meta)
    target.write(source.read())

print(f"TensorRT engine build completed at {engine_model}")
PY

if [[ ! -f "${ENGINE_MODEL}" ]]; then
    echo "Expected TensorRT engine was not written to ${ENGINE_MODEL}" >&2
    exit 1
fi

"${PYTHON_BIN}" - <<PY
from pathlib import Path

import numpy as np

from tennis_ball_detection.object_detection_YOLO.backend import YoloDetector

engine_model = Path("${ENGINE_MODEL}")
imgsz = int("${ENGINE_IMGSZ}")

detector = YoloDetector(
    model_path=engine_model,
    confidence_threshold=0.25,
    device="0",
    imgsz=imgsz,
)
detections = detector.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))
print(f"TensorRT runtime validation detections: {len(detections)}")
PY

echo "TensorRT engine ready at ${ENGINE_MODEL}"
