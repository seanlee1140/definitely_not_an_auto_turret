from __future__ import annotations

from pathlib import Path

from tennis_ball_detection.object_detection_common.types import DetectionResult

try:
    from ultralytics import YOLO as UltralyticsYOLO
except ImportError:  # pragma: no cover - exercised on device
    UltralyticsYOLO = None


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


class YoloOnnxDetector:
    def __init__(self, model_path: Path, confidence_threshold: float = 0.25, model_factory=None):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self._model_factory = model_factory
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model

        model_factory = self._model_factory or UltralyticsYOLO
        if model_factory is None:
            raise RuntimeError(
                "Ultralytics is required for YOLO inference. "
                "Install dependencies from object_detection_YOLO/requirements.txt."
            )

        self._model = model_factory(str(self.model_path))
        return self._model

    def predict(self, frame):
        model = self.load()
        results = model.predict(source=frame, conf=self.confidence_threshold, verbose=False)
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
