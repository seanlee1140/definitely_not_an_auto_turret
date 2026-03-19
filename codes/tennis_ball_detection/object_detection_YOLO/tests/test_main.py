import unittest
from pathlib import Path

from tennis_ball_detection.object_detection_YOLO.main import (
    ensure_engine_model_path,
    ensure_gpu_device,
    ensure_model_path_ready,
)


class YoloMainTests(unittest.TestCase):
    def test_missing_default_engine_has_actionable_error(self):
        default_model_path = Path("/tmp/missing-best.engine")
        setup_script_path = Path("/tmp/setup_jetson_yolo.sh")

        with self.assertRaises(RuntimeError) as context:
            ensure_model_path_ready(
                default_model_path,
                default_model_path=default_model_path,
                setup_script_path=setup_script_path,
            )

        self.assertIn("TensorRT engine not found", str(context.exception))
        self.assertIn(str(setup_script_path), str(context.exception))

    def test_missing_custom_model_has_generic_error(self):
        default_model_path = Path("/tmp/default-best.engine")
        custom_model_path = Path("/tmp/custom.engine")

        with self.assertRaises(RuntimeError) as context:
            ensure_model_path_ready(
                custom_model_path,
                default_model_path=default_model_path,
                setup_script_path=Path("/tmp/setup_jetson_yolo.sh"),
            )

        self.assertEqual(str(context.exception), f"Model file not found: {custom_model_path}")

    def test_cpu_device_is_rejected(self):
        with self.assertRaises(RuntimeError) as context:
            ensure_gpu_device("cpu")

        self.assertIn("GPU-only", str(context.exception))

    def test_non_engine_model_is_rejected(self):
        with self.assertRaises(RuntimeError) as context:
            ensure_engine_model_path(Path("/tmp/best.pt"))

        self.assertIn("TensorRT '.engine' model file", str(context.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
