from __future__ import annotations

import json
import struct
from pathlib import Path


DETECTION_HEAD_PATTERNS = (
    "class_labels_classifier",
    "bbox_predictor",
    "class_embed",
    "bbox_embed",
)
REQUIRED_CLASS_METADATA = ("id2label", "label2id", "num_labels")


class IncompleteDetrCheckpointError(RuntimeError):
    """Raised when the local DETR folder is not a complete detection checkpoint."""


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as file:
        header_len = struct.unpack("<Q", file.read(8))[0]
        return json.loads(file.read(header_len))


class DetrCheckpointValidator:
    def __init__(self, model_path: Path, confidence_threshold: float = 0.25):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold

    def load(self):
        self.validate()
        return self

    def predict(self, frame):  # pragma: no cover - current scaffold intentionally fails before predict
        raise RuntimeError(
            "DETR inference is unavailable because the current checkpoint is incomplete "
            "for object detection."
        )

    def validate(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"DETR model path does not exist: {self.model_path}")

        config_path = self.model_path / "config.json"
        safetensors_path = self.model_path / "model.safetensors"

        if not config_path.exists():
            raise FileNotFoundError(f"Missing DETR config file: {config_path}")
        if not safetensors_path.exists():
            raise FileNotFoundError(f"Missing DETR checkpoint file: {safetensors_path}")

        config = _read_json(config_path)
        header = _read_safetensors_header(safetensors_path)

        missing_metadata = [
            field for field in REQUIRED_CLASS_METADATA if not config.get(field)
        ]
        tensor_keys = [key for key in header.keys() if key != "__metadata__"]
        head_keys = [
            key
            for key in tensor_keys
            if any(pattern in key for pattern in DETECTION_HEAD_PATTERNS)
        ]

        if missing_metadata or not head_keys:
            raise IncompleteDetrCheckpointError(
                self._format_incomplete_checkpoint_error(
                    missing_metadata=missing_metadata,
                    head_keys=head_keys,
                )
            )

    def _format_incomplete_checkpoint_error(
        self,
        *,
        missing_metadata: list[str],
        head_keys: list[str],
    ) -> str:
        parts = [
            "Checkpoint at "
            f"'{self.model_path}' is incomplete for object detection.",
            "The current local 'detr-finetuned-tennis-ball-v2' folder contains "
            "backbone, encoder, and decoder weights but no usable detection head.",
        ]

        if missing_metadata:
            parts.append(
                "Missing class metadata in config.json: " + ", ".join(missing_metadata) + "."
            )
        if not head_keys:
            parts.append(
                "Missing detection-head tensors matching: "
                + ", ".join(DETECTION_HEAD_PATTERNS)
                + "."
            )

        parts.append(
            "Provide a complete DETR object-detection checkpoint with class mappings "
            "before enabling live inference."
        )
        return " ".join(parts)
