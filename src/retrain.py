"""Retraining pipeline for RetinaScope.

Core guarantees (see project plan):
- Retraining ALWAYS merges the base training set + retrain_pool + any uploaded
  images. It never trains on uploaded data alone, so the model cannot collapse
  onto a narrow new distribution.
- A retrained model is promoted ONLY if it beats the current model's macro-F1
  on the frozen test set. The old model is archived first.
- Metrics and class weights are computed with numpy so the serving container
  needs no scikit-learn dependency.
"""
import glob
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import resize_and_normalize, make_dataset, ben_graham_filter, CLASS_NAMES, IMG_SIZE
from prediction import MODEL_PATH

BASE_TRAIN_DIRS = ["data/train", "data/retrain_pool"]
# Paths are env-overridable so the retraining flow can be exercised in a fully
# isolated sandbox (tests) without touching the real data/model artifacts.
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "data/uploads")
TEST_DIR = os.environ.get("TEST_DIR", "data/test")
ARCHIVE_DIR = os.environ.get("ARCHIVE_DIR", "models/archive")
META_PATH = os.environ.get("META_PATH", "models/model_meta.json")

RETRAIN_EPOCHS = int(os.environ.get("RETRAIN_EPOCHS", "5"))
# Count-based auto-trigger threshold: retraining auto-triggers once the total
# number of pending uploaded images (accumulated in UPLOAD_DIR this session)
# reaches this count and no retrain is already running.
AUTO_RETRAIN_THRESHOLD = int(os.environ.get("AUTO_RETRAIN_THRESHOLD", "20"))


# ---------------------------------------------------------------- uploads ----
def _safe_name(name: str) -> str:
    return os.path.basename(name).replace("..", "_")


def save_uploaded_images(files, label: str) -> int:
    """files: iterable of (filename, bytes). Saves them under the given class
    label's upload folder. Returns the number saved.
    """
    if label not in CLASS_NAMES:
        raise ValueError(f"label must be one of {CLASS_NAMES}, got {label!r}")
    dest = os.path.join(UPLOAD_DIR, label)
    os.makedirs(dest, exist_ok=True)
    saved = 0
    for filename, data in files:
        with open(os.path.join(dest, _safe_name(filename)), "wb") as fh:
            fh.write(data)
        saved += 1
    return saved


def count_uploads() -> int:
    if not os.path.isdir(UPLOAD_DIR):
        return 0
    return sum(
        len(glob.glob(os.path.join(UPLOAD_DIR, cls, "*")))
        for cls in os.listdir(UPLOAD_DIR)
        if os.path.isdir(os.path.join(UPLOAD_DIR, cls))
    )


# --------------------------------------------------------------- metadata ----
def load_meta() -> dict:
    if os.path.exists(META_PATH):
        with open(META_PATH) as fh:
            return json.load(fh)
    return {"version": 0, "trained_at": None, "last_retrain": None,
            "architecture": "unknown", "test_accuracy": None, "test_macro_f1": None}


def save_meta(meta: dict) -> None:
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, "w") as fh:
        json.dump(meta, fh, indent=2)


# --------------------------------------------------------- data + metrics ----
def _list_labeled_files(dirs):
    paths, labels = [], []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for idx, cls in enumerate(CLASS_NAMES):
            cls_dir = os.path.join(d, cls)
            if not os.path.isdir(cls_dir):
                continue
            for p in glob.glob(os.path.join(cls_dir, "*")):
                paths.append(p)
                labels.append(idx)
    return paths, labels


def _load_path_maybe_filter(path, label, needs_filter):
    """Decode an image and, only when needs_filter is true, apply the same
    Ben-Graham gaussian filter the prediction path uses. Base images on disk are
    already filtered (needs_filter=False); raw uploaded images are not, so they
    get filtered here to match the base distribution and the inference path."""
    data = tf.io.read_file(path)
    img = tf.io.decode_image(data, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.cond(
        needs_filter,
        lambda: tf.ensure_shape(
            tf.numpy_function(ben_graham_filter, [img], tf.uint8), [None, None, 3]),
        lambda: img,
    )
    return resize_and_normalize(img), label


def build_training_dataset(batch_size: int = 32, seed: int = 42):
    # Base dirs are already Ben-Graham filtered on disk -> no filter. The upload
    # dir holds raw user images -> flag them so _load_path_maybe_filter applies
    # the same filter the prediction path does. Shuffling happens at the (light)
    # path level before decoding, so the shuffle buffer stays cheap.
    base_paths, base_labels = _list_labeled_files(BASE_TRAIN_DIRS)
    upload_paths, upload_labels = _list_labeled_files([UPLOAD_DIR])
    paths = base_paths + upload_paths
    labels = base_labels + upload_labels
    flags = [False] * len(base_paths) + [True] * len(upload_paths)
    if not paths:
        raise RuntimeError("No training images found to retrain on.")
    labels_arr = np.array(labels)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels, flags))
    ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_load_path_maybe_filter, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, labels_arr


def balanced_class_weights(labels: np.ndarray) -> dict:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    n = len(labels)
    # sklearn "balanced" formula: n_samples / (n_classes * count_c)
    weights = {i: (n / (len(CLASS_NAMES) * c) if c > 0 else 0.0) for i, c in enumerate(counts)}
    return weights


def macro_f1_and_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    n_classes = len(CLASS_NAMES)
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    accuracy = float(np.mean(y_true == y_pred))
    return float(np.mean(f1s)), accuracy


def evaluate_on_test(model) -> dict:
    test_ds = make_dataset(TEST_DIR, batch_size=32, shuffle=False)
    y_true, y_pred = [], []
    for xb, yb in test_ds:
        y_pred.append(model.predict(xb, verbose=0).argmax(1))
        y_true.append(yb.numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    macro_f1, accuracy = macro_f1_and_accuracy(y_true, y_pred)
    return {"macro_f1": round(macro_f1, 4), "accuracy": round(accuracy, 4)}


# --------------------------------------------------------------- retrain ----
def retrain_model(epochs: int = RETRAIN_EPOCHS, on_epoch_end=None) -> dict:
    """Retrain on merged data, evaluate on the frozen test set, and promote the
    new model only if its macro-F1 beats the current model's. Returns a summary
    dict with before/after metrics and whether promotion happened.
    """
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"No current model at {MODEL_PATH} to retrain from.")

    # Evaluate the current model (the promotion baseline).
    current = tf.keras.models.load_model(MODEL_PATH)
    old_metrics = evaluate_on_test(current)
    del current

    # Warm-start a separate copy from disk so the served model is untouched
    # until (and unless) we promote and hot-reload.
    model = tf.keras.models.load_model(MODEL_PATH)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_ds, labels_arr = build_training_dataset(batch_size=32)
    class_weight = balanced_class_weights(labels_arr)

    callbacks = []
    if on_epoch_end is not None:
        callbacks.append(tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: on_epoch_end(epoch, logs or {})
        ))

    model.fit(train_ds, epochs=epochs, class_weight=class_weight, callbacks=callbacks, verbose=0)

    new_metrics = evaluate_on_test(model)
    promoted = new_metrics["macro_f1"] > old_metrics["macro_f1"]

    result = {
        "promoted": promoted,
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "training_images": int(len(labels_arr)),
        "epochs": epochs,
    }

    if promoted:
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(MODEL_PATH, os.path.join(ARCHIVE_DIR, f"mobilenetv2_dr_{stamp}.keras"))
        model.save(MODEL_PATH)
        meta = load_meta()
        meta["version"] = int(meta.get("version", 0)) + 1
        meta["last_retrain"] = datetime.now(timezone.utc).isoformat()
        meta["test_accuracy"] = new_metrics["accuracy"]
        meta["test_macro_f1"] = new_metrics["macro_f1"]
        meta["training_images"] = int(len(labels_arr))
        save_meta(meta)
        result["new_version"] = meta["version"]

    return result
