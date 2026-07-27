"""Shared preprocessing path for RetinaScope, used identically by training
and inference so the model never sees a distribution shift between the two.

The stored training images (data/train, data/test, data/retrain_pool) are
already Ben-Graham gaussian-filtered by the dataset provider - that's what
"gaussian_filtered_images" meant in the raw source. A raw user upload is not
filtered yet, so the inference path applies the same filter before handing
the image to the shared resize/normalize step used by training.
"""
import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

sys.path.insert(0, os.path.dirname(__file__))
from constants import IMG_SIZE, CLASS_NAMES  # re-exported for existing importers


def ben_graham_filter(image: np.ndarray, sigma_x: float = 10) -> np.ndarray:
    """Reproduces the contrast-enhancement already baked into the training
    images, for use on raw uploads only. Channel-order agnostic (RGB or BGR
    both work identically since the same weighting is applied per-channel).
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma_x)
    return cv2.addWeighted(image, 4, blurred, -4, 128)


def resize_and_normalize(image):
    """The shared path: resize to the model's input size and apply
    MobileNetV2's expected normalization. Called identically by the training
    tf.data pipeline and by the inference route.
    """
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return preprocess_input(image)


def preprocess_upload(image: np.ndarray) -> tf.Tensor:
    """Full inference-time path for a raw user-uploaded image."""
    filtered = ben_graham_filter(image)
    return resize_and_normalize(filtered)


def make_dataset(directory: str, batch_size: int = 32, shuffle: bool = True, seed: int = 42):
    """Loads a class-per-folder image directory (already gaussian-filtered on
    disk) into a tf.data.Dataset, using the shared resize/normalize path.
    class_names is passed explicitly so label indices always match
    CLASS_NAMES regardless of alphabetical folder order.
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    ds = ds.map(
        lambda x, y: (resize_and_normalize(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def make_train_val_datasets(directory: str, val_split: float = 0.15,
                            batch_size: int = 32, seed: int = 42):
    """Splits a training directory into (train_ds, val_ds) using a deterministic
    seed, so the validation set used for early stopping / checkpointing is carved
    out of the training data and the separate test set stays completely frozen
    for final evaluation. Both subsets go through the shared resize/normalize path.
    """
    def _load(subset):
        ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="int",
            class_names=CLASS_NAMES,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            validation_split=val_split,
            subset=subset,
        )
        return ds.map(
            lambda x, y: (resize_and_normalize(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)

    return _load("training"), _load("validation")
