"""Simple tests: python -m unittest discover tests"""
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

import app as app_module
from model_service import preprocess_image
from recommendations import DEFAULT, get_recommendation


def image_bytes(fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", (60, 40), "green").save(buf, fmt)
    buf.seek(0)
    return buf


class TestHelpers(unittest.TestCase):
    def test_preprocess_shape(self):
        arr = preprocess_image(Image.new("RGBA", (300, 100)))
        self.assertEqual(arr.shape, (1, 224, 224, 3))

    def test_recommendations(self):
        self.assertIn("glass", get_recommendation("Glass").lower())
        self.assertEqual(get_recommendation("unknown"), DEFAULT)


class TestRoutes(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()

    def test_index_and_health(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertIn("model_ready", self.client.get("/health").get_json())

    def test_missing_file(self):
        self.assertEqual(self.client.post("/predict").status_code, 400)

    def test_invalid_file(self):
        data = {"image": (io.BytesIO(b"not an image"), "x.png")}
        r = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)

    def test_oversized_file(self):
        data = {"image": (io.BytesIO(b"0" * (6 * 1024 * 1024)), "big.png")}
        r = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 413)

    def test_valid_image_with_fake_model(self):
        class Fake:
            ready = True
            def predict(self, img):
                return {"category": "glass", "confidence": 90.0,
                        "top_predictions": [{"label": "glass", "confidence": 90.0}]}
        original = app_module.service
        app_module.service = Fake()
        try:
            data = {"image": (image_bytes(), "a.png")}
            r = self.client.post("/predict", data=data, content_type="multipart/form-data")
            self.assertEqual(r.status_code, 200)
            self.assertIn("recommendation", r.get_json())
        finally:
            app_module.service = original


if __name__ == "__main__":
    unittest.main()
