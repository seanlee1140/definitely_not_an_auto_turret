import unittest
from pathlib import Path

from tennis_ball_detection.object_detection_DETR.backend import (
    DetrCheckpointValidator,
    IncompleteDetrCheckpointError,
)


class DetrBackendTests(unittest.TestCase):
    def test_local_checkpoint_fails_with_incomplete_error(self):
        model_path = (
            Path(__file__).resolve().parents[2] / "model" / "detr-finetuned-tennis-ball-v2"
        )
        validator = DetrCheckpointValidator(model_path=model_path)

        with self.assertRaises(IncompleteDetrCheckpointError) as context:
            validator.validate()

        message = str(context.exception)
        self.assertIn("incomplete for object detection", message)
        self.assertIn("Missing class metadata", message)
        self.assertIn("Missing detection-head tensors", message)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
