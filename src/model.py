"""MobileNetV2 and scratch-CNN architectures for RetinaScope's 3-class DR
grading. Both expect input already run through preprocessing.resize_and_normalize
(or preprocess_upload) - no normalization is duplicated inside either model.
"""
import os
import sys

import tensorflow as tf
from tensorflow.keras import layers, models

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import make_dataset, IMG_SIZE, CLASS_NAMES

MODEL_PATH = os.path.join("models", "mobilenetv2_dr.keras")


def build_mobilenetv2(num_classes: int = len(CLASS_NAMES), fine_tune_layers: int = 0) -> tf.keras.Model:
    """fine_tune_layers=0 freezes the entire base (fast, throwaway-style).
    A positive value unfreezes that many of the base's top layers. BatchNorm
    layers stay in inference mode regardless (base called with training=False),
    which is the standard recipe for fine-tuning without unstable BN stats.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = fine_tune_layers > 0
    if fine_tune_layers > 0:
        for layer in base.layers[:-fine_tune_layers]:
            layer.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3 if fine_tune_layers == 0 else 1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_scratch_cnn(num_classes: int = len(CLASS_NAMES)) -> tf.keras.Model:
    """A small CNN trained from random initialization - no pretrained weights -
    as the comparison baseline against MobileNetV2 transfer learning.
    """
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    """Quick CLI entrypoint kept for the Stage-2-style throwaway rebuild;
    the notebook is where the real Stage 4 training + evaluation happens.
    """
    train_ds = make_dataset("data/train", batch_size=32, shuffle=True)
    test_ds = make_dataset("data/test", batch_size=32, shuffle=False)

    model = build_mobilenetv2()
    model.fit(train_ds, validation_data=test_ds, epochs=3)

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved throwaway model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
