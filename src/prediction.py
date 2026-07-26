"""Loads the trained model once and exposes predict() for a raw uploaded
image, routed through the same inference preprocessing path used everywhere
else (see preprocessing.py).
"""
import io
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import preprocess_upload, CLASS_NAMES

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("models", "mobilenetv2_dr.keras"))

_model = None


def get_model() -> tf.keras.Model:
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def reload_model() -> None:
    """Drops the cached model so the next get_model() call reloads MODEL_PATH
    from disk. Used after retraining promotes a new model file (Stage 5).
    """
    global _model
    _model = None


def load_image_from_bytes(data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(image)


def predict(image: np.ndarray) -> dict:
    model = get_model()
    processed = preprocess_upload(image)
    batch = tf.expand_dims(processed, axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    predicted_idx = int(np.argmax(probabilities))
    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": float(probabilities[predicted_idx]),
        "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probabilities)},
    }
