import unittest

from tennis_ball_detection.object_detection_common.overlay import draw_detection_overlay
from tennis_ball_detection.object_detection_common.runtime import format_detection_summary
from tennis_ball_detection.object_detection_common.types import DetectionResult


class FakeCV2:
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self):
        self.calls = []

    def rectangle(self, frame, pt1, pt2, color, thickness):
        self.calls.append(("rectangle", pt1, pt2, color, thickness))

    def circle(self, frame, center, radius, color, thickness):
        self.calls.append(("circle", center, radius, color, thickness))

    def putText(self, frame, text, origin, font, scale, color, thickness):
        self.calls.append(("putText", text, origin, font, scale, color, thickness))


class OverlayTests(unittest.TestCase):
    def test_draw_detection_overlay_emits_expected_draw_calls(self):
        renderer = FakeCV2()
        frame = object()
        detection = DetectionResult(
            label="tennis-ball",
            confidence=0.91,
            bbox=(10, 20, 30, 40),
            center=(20, 30),
        )

        returned_frame = draw_detection_overlay(frame, detection, cv2_module=renderer)

        self.assertIs(returned_frame, frame)
        self.assertEqual(renderer.calls[0], ("rectangle", (10, 20), (30, 40), (0, 255, 0), 2))
        self.assertEqual(renderer.calls[1], ("circle", (20, 30), 5, (0, 0, 255), -1))
        self.assertEqual(renderer.calls[2][0], "putText")
        self.assertIn("tennis-ball 0.91", renderer.calls[2][1])

    def test_format_detection_summary_is_stdout_ready(self):
        detection = DetectionResult(
            label="tennis-ball",
            confidence=0.8765,
            bbox=(100, 120, 180, 210),
            center=(140, 165),
        )

        summary = format_detection_summary(detection)

        self.assertEqual(
            summary,
            "detected label=tennis-ball confidence=0.876 center=(140, 165) bbox=(100, 120, 180, 210)",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
