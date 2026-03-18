import unittest
from pathlib import Path

from tennis_ball_detection.object_detection_YOLO.backend import YoloOnnxDetector


class ListLike:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = ListLike(xyxy)
        self.conf = ListLike(conf)
        self.cls = ListLike(cls)


class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class FakeModel:
    def __init__(self, results):
        self.results = results
        self.last_predict_kwargs = None

    def predict(self, **kwargs):
        self.last_predict_kwargs = kwargs
        return self.results


class YoloBackendTests(unittest.TestCase):
    def test_parse_detections_converts_yolo_outputs(self):
        result = FakeResult(
            boxes=FakeBoxes(
                xyxy=[[10.2, 20.1, 40.8, 60.6], [5.0, 5.0, 15.0, 15.0]],
                conf=[0.92, 0.35],
                cls=[0, 1],
            ),
            names={0: "tennis-ball", 1: "background"},
        )

        detections = YoloOnnxDetector.parse_detections([result])

        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0].label, "tennis-ball")
        self.assertEqual(detections[0].bbox, (10, 20, 41, 61))
        self.assertEqual(detections[0].center, (26, 40))
        self.assertAlmostEqual(detections[0].confidence, 0.92)

    def test_predict_uses_model_factory_and_threshold(self):
        fake_result = FakeResult(
            boxes=FakeBoxes(xyxy=[[100, 120, 180, 220]], conf=[0.88], cls=[0]),
            names={0: "tennis-ball"},
        )
        fake_model = FakeModel([fake_result])

        detector = YoloOnnxDetector(
            model_path=Path("dummy.onnx"),
            confidence_threshold=0.4,
            model_factory=lambda _: fake_model,
        )

        detections = detector.predict(frame="frame-object")

        self.assertEqual(len(detections), 1)
        self.assertEqual(fake_model.last_predict_kwargs["source"], "frame-object")
        self.assertEqual(fake_model.last_predict_kwargs["conf"], 0.4)
        self.assertFalse(fake_model.last_predict_kwargs["verbose"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
