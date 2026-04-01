import unittest
from pathlib import Path

from tennis_ball_detection.object_detection_common.cli import build_live_detection_parser


class CliTests(unittest.TestCase):
    def test_parser_includes_jetson_runtime_options(self):
        default_model_path = Path("/tmp/best.engine")
        parser = build_live_detection_parser(
            description="Live detection test parser.",
            default_model_path=default_model_path,
        )

        args = parser.parse_args([])

        self.assertEqual(args.model_path, default_model_path)
        self.assertEqual(args.device, "0")
        self.assertEqual(args.imgsz, 640)
        self.assertEqual(args.flip_method, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
