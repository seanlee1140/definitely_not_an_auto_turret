from __future__ import annotations

import ast
from pathlib import Path

from tennis_ball_detection.object_detection_common.types import DetectionResult

try:
    import torch
except ImportError:  # pragma: no cover - exercised on device
    torch = None

try:
    from ultralytics import YOLO as UltralyticsYOLO
    from ultralytics.nn.autobackend import AutoBackend
except ImportError:  # pragma: no cover - exercised on device
    UltralyticsYOLO = None
    AutoBackend = None

try:
    import onnx
except ImportError:  # pragma: no cover - exercised on device
    onnx = None


_METADATA_LITERAL_KEYS = {"args", "end2end", "imgsz", "kpt_names", "kpt_shape", "names"}
_METADATA_INT_KEYS = {"batch", "channels", "stride"}
_ULTRALYTICS_WARMUP_PATCHED = False


def _normalize_list(value):
    if value is None:
        return []
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _lookup_label(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), f"class_{class_id}")))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


def _needs_name_override(names) -> bool:
    if not isinstance(names, dict):
        return False
    return len(names) >= 999 and names.get(0) == "class0"


def _load_names_from_onnx_metadata(model_path: Path):
    metadata = _load_onnx_metadata(model_path) or {}
    names = metadata.get("names")
    if names is None:
        return None
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return None


def _parse_onnx_metadata_value(key: str, value: str):
    if key in _METADATA_LITERAL_KEYS:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    if key in _METADATA_INT_KEYS:
        try:
            return int(value)
        except ValueError:
            return value
    return value


def _load_onnx_metadata(model_path: Path):
    if onnx is None or not model_path.exists():
        return None

    model = onnx.load(str(model_path))
    metadata = {item.key: _parse_onnx_metadata_value(item.key, item.value) for item in model.metadata_props}
    return metadata or None


def _engine_has_end_to_end_nms(onnx_metadata) -> bool:
    if not isinstance(onnx_metadata, dict):
        return False
    args = onnx_metadata.get("args")
    return isinstance(args, dict) and bool(args.get("nms"))


def _patch_autobackend_warmup_for_tensorrt():
    global _ULTRALYTICS_WARMUP_PATCHED

    if _ULTRALYTICS_WARMUP_PATCHED or AutoBackend is None or torch is None:
        return

    from ultralytics.utils.nms import non_max_suppression

    def _jetson_tensorrt_warmup(self, imgsz: tuple[int, int, int, int] = (1, 3, 640, 640)) -> None:
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton, self.nn_module
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)
            for _ in range(2 if self.jit else 1):
                self.forward(im)
                if not self.engine:
                    warmup_boxes = torch.rand(1, 84, 16, device=self.device)
                    warmup_boxes[:, :4] *= imgsz[-1]
                    non_max_suppression(warmup_boxes)

    AutoBackend.warmup = _jetson_tensorrt_warmup
    _ULTRALYTICS_WARMUP_PATCHED = True


class YoloDetector:
    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.25,
        *,
        device: str | None = None,
        imgsz: int | None = None,
        task: str = "detect",
        model_factory=None,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.imgsz = imgsz
        self.task = task
        self._model_factory = model_factory
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model

        if not self.model_path.exists():
            raise RuntimeError(f"Model file not found: {self.model_path}")

        onnx_metadata = None
        if self.model_path.suffix == ".engine":
            _patch_autobackend_warmup_for_tensorrt()
            onnx_metadata = _load_onnx_metadata(self.model_path.with_suffix(".onnx"))
            if onnx_metadata and not _engine_has_end_to_end_nms(onnx_metadata):
                raise RuntimeError(
                    "The TensorRT engine at "
                    f"{self.model_path} was not exported with end-to-end NMS. "
                    "Run object_detection_YOLO/setup_jetson_yolo.sh to rebuild best.engine for GPU-only Jetson inference."
                )

        model_factory = self._model_factory or UltralyticsYOLO
        if model_factory is None:
            raise RuntimeError(
                "Ultralytics is required for YOLO inference. "
                "Install dependencies from object_detection_YOLO/requirements.txt."
            )

        try:
            self._model = model_factory(str(self.model_path), task=self.task)
        except TypeError:
            self._model = model_factory(str(self.model_path))

        if self.model_path.suffix == ".engine" and _needs_name_override(getattr(self._model, "names", None)):
            fallback_names = None
            if onnx_metadata:
                fallback_names = onnx_metadata.get("names")
                if isinstance(fallback_names, dict):
                    fallback_names = {int(key): str(value) for key, value in fallback_names.items()}
            if fallback_names is None:
                fallback_names = _load_names_from_onnx_metadata(self.model_path.with_suffix(".onnx"))
            if fallback_names:
                self._model.names = fallback_names
        return self._model

    def predict(self, frame):
        model = self.load()
        predict_kwargs = {
            "source": frame,
            "conf": self.confidence_threshold,
            "verbose": False,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device
        if self.imgsz is not None:
            predict_kwargs["imgsz"] = self.imgsz
        results = model.predict(**predict_kwargs)
        return self.parse_detections(results)

    @staticmethod
    def parse_detections(results) -> list[DetectionResult]:
        detections: list[DetectionResult] = []

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            coords = _normalize_list(getattr(boxes, "xyxy", []))
            confidences = _normalize_list(getattr(boxes, "conf", []))
            class_ids = _normalize_list(getattr(boxes, "cls", []))
            names = getattr(result, "names", {})

            for xyxy, confidence, class_id in zip(coords, confidences, class_ids):
                x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
                class_index = int(class_id)
                detections.append(
                    DetectionResult(
                        label=_lookup_label(names, class_index),
                        confidence=float(confidence),
                        bbox=(x1, y1, x2, y2),
                        center=(int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))),
                    )
                )

        return detections


YoloOnnxDetector = YoloDetector
