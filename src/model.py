"""MobileNetV2 transfer-learning architecture for RetinaScope's 3-class DR
grading. Expects input already run through preprocessing.resize_and_normalize
(or preprocess_upload) - no normalization is duplicated inside the model.
"""
import os
import sys

import tensorflow as tf
from tensorflow.keras import layers, models

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import make_dataset, IMG_SIZE, CLASS_NAMES

MODEL_PATH = os.path.join("models", "mobilenetv2_dr.keras")


def build_model(num_classes: int = len(CLASS_NAMES)) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    train_ds = make_dataset("data/train", batch_size=32, shuffle=True)
    test_ds = make_dataset("data/test", batch_size=32, shuffle=False)

    model = build_model()
    model.fit(train_ds, validation_data=test_ds, epochs=3)

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved throwaway model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
