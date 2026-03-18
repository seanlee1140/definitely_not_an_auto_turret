from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class DetectionResult:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]


def best_detection(detections: Iterable[DetectionResult]) -> Optional[DetectionResult]:
    return max(detections, key=lambda detection: detection.confidence, default=None)
