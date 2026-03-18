import unittest

from tennis_ball_detection.object_detection_common.types import DetectionResult, best_detection


class TypeTests(unittest.TestCase):
    def test_best_detection_returns_highest_confidence_detection(self):
        detections = [
            DetectionResult(
                label="tennis-ball",
                confidence=0.51,
                bbox=(0, 0, 10, 10),
                center=(5, 5),
            ),
            DetectionResult(
                label="tennis-ball",
                confidence=0.93,
                bbox=(10, 10, 30, 30),
                center=(20, 20),
            ),
        ]

        best = best_detection(detections)

        self.assertIsNotNone(best)
        self.assertEqual(best.confidence, 0.93)
        self.assertEqual(best.center, (20, 20))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
