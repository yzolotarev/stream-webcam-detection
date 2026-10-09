"""Live-webcam rules: static art is never a webcam; live faces are; stream context decides content.

    python -m unittest live/test_live_cams.py
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import live_cams  # noqa: E402


def frames(seed_changes: bool, count: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    base = np.full((1080, 1920, 3), 40, np.uint8)
    out = []
    for index in range(count):
        image = base.copy()
        # webcam window bottom-left: live pixels change between samples
        image[600:1080, 0:640] = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) if seed_changes \
            else np.full((480, 640, 3), 120, np.uint8)
        out.append(image)
    return out


FACE = [0.1, 0.7, 0.07, 0.15]


class RuleTests(unittest.TestCase):
    def setUp(self):
        self._detect = live_cams.detect_faces

    def tearDown(self):
        live_cams.detect_faces = self._detect

    def run_with_faces(self, images, faces, stream=None, stream_faces=None):
        calls = []

        def fake(batch):
            calls.append(len(batch))
            if len(calls) == 1:
                return [faces for _ in batch]
            return [stream_faces or [] for _ in batch]

        live_cams.detect_faces = fake
        return live_cams.detect(images, stream)

    def test_static_picture_is_never_a_webcam(self):
        result = self.run_with_faces(frames(False), [FACE + [0.9]])
        self.assertEqual(result["mode"], "full_frame")
        self.assertEqual(result["candidates"][0]["verdict"], "static")

    def test_live_face_in_own_window_is_a_webcam_without_stream_context(self):
        result = self.run_with_faces(frames(True), [FACE + [0.9]])
        self.assertEqual(result["mode"], "webcam")
        panel = result["panels"][0]
        self.assertLessEqual(panel[0], FACE[0])
        self.assertGreaterEqual(panel[0] + panel[2], FACE[0] + FACE[2])

    def test_recurring_face_across_stream_is_primary(self):
        stream = frames(True, 12)
        result = self.run_with_faces(frames(True), [FACE + [0.9]], stream, [FACE + [0.9]])
        self.assertEqual(result["mode"], "webcam")
        self.assertEqual(result["candidates"][0]["recurrence"], 1.0)

    def test_live_face_absent_from_stream_is_content(self):
        stream = frames(True, 12)
        result = self.run_with_faces(frames(True), [FACE + [0.9]], stream, [])
        self.assertEqual(result["mode"], "full_frame")

    def test_face_framed_box_is_16_9_and_inside_frame(self):
        box = live_cams.face_framed_box([0.95, 0.9, 0.05, 0.08])
        self.assertAlmostEqual(box[2], box[3])
        self.assertLessEqual(box[0] + box[2], 1.0 + 1e-9)
        self.assertLessEqual(box[1] + box[3], 1.0 + 1e-9)


if __name__ == "__main__":
    unittest.main()
