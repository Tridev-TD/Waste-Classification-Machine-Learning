"""Model loading and prediction (no Flask code here)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = 224
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.keras"
CLASSES_PATH = BASE_DIR / "models" / "class_names.json"


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize to 224x224 RGB, shape (1, 224, 224, 3), float32 in 0-255.

    MobileNetV2 scaling (to [-1, 1]) is a layer inside the saved model,
    so training and inference share exactly the same preprocessing.
    """
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.expand_dims(np.asarray(img, dtype=np.float32), axis=0)


class ModelService:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.error = None

    @property
    def ready(self) -> bool:
        return self.model is not None and bool(self.class_names)

    def load(self):
        """Load the model once. Never raises; stores the error instead."""
        try:
            if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
                raise FileNotFoundError(
                    "Trained model not found. Run `python train.py` first."
                )
            import tensorflow as tf  # imported lazily so tests stay light

            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.class_names = json.loads(CLASSES_PATH.read_text())
            self.error = None
        except Exception as exc:
            self.model = None
            self.error = str(exc)

    def predict(self, img: Image.Image, top_k: int = 3) -> dict:
        if not self.ready:
            raise RuntimeError(self.error or "Model is not loaded.")
        probs = self.model.predict(preprocess_image(img), verbose=0)[0]
        order = np.argsort(probs)[::-1][:top_k]
        top = [
            {"label": self.class_names[i], "confidence": round(float(probs[i]) * 100, 2)}
            for i in order
        ]
        return {"category": top[0]["label"], "confidence": top[0]["confidence"], "top_predictions": top}
