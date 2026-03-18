import unittest

from tennis_ball_detection.object_detection_common.camera import (
    CameraOpenError,
    build_gstreamer_pipeline,
    open_jetson_camera,
)


class FakeCapture:
    def __init__(self, pipeline, api_preference=None, opened=True):
        self.pipeline = pipeline
        self.api_preference = api_preference
        self._opened = opened
        self.released = False

    def isOpened(self):
        return self._opened

    def release(self):
        self.released = True


class CameraTests(unittest.TestCase):
    def test_build_gstreamer_pipeline_matches_expected_shape(self):
        pipeline = build_gstreamer_pipeline(
            sensor_id=1,
            width=1280,
            height=720,
            fps=60,
            display_width=640,
            display_height=360,
        )

        self.assertIn("nvarguscamerasrc sensor-id=1", pipeline)
        self.assertIn("width=1280,height=720", pipeline)
        self.assertIn("framerate=60/1", pipeline)
        self.assertIn("width=640,height=360", pipeline)
        self.assertTrue(pipeline.endswith("video/x-raw,format=BGR ! appsink"))

    def test_open_jetson_camera_raises_on_failure(self):
        captures = []

        def fake_factory(pipeline, api_preference):
            capture = FakeCapture(pipeline, api_preference, opened=False)
            captures.append(capture)
            return capture

        with self.assertRaises(CameraOpenError):
            open_jetson_camera(
                sensor_id=0,
                width=1920,
                height=1080,
                fps=30,
                display_width=960,
                display_height=540,
                video_capture_factory=fake_factory,
                api_preference=123,
            )

        self.assertEqual(len(captures), 1)
        self.assertTrue(captures[0].released)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
