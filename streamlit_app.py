"""RetinaScope UI - a Streamlit front end over the FastAPI service.

Covers the four required surfaces: model uptime/status, dataset visualizations
(three feature interpretations), single-image prediction, and bulk upload +
retraining with live status. Talks to the API over HTTP; point it at a local
container for the retrain demo or at the Render URL for prediction via the
API_URL env var (overridable in the sidebar).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import requests
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from constants import CLASS_NAMES  # lightweight, avoids importing TensorFlow into the UI

DEFAULT_API = os.environ.get("API_URL", "http://localhost:8000")
DATA_DIR = "data"
sns.set_theme(style="whitegrid")

st.set_page_config(page_title="RetinaScope", page_icon="RS", layout="wide")


# ------------------------------------------------------------- helpers ----
def api_url() -> str:
    return st.session_state.get("api_url", DEFAULT_API).rstrip("/")


def api_get(path, **kw):
    return requests.get(f"{api_url()}{path}", timeout=kw.pop("timeout", 15), **kw)


def api_post(path, **kw):
    return requests.post(f"{api_url()}{path}", timeout=kw.pop("timeout", 60), **kw)


@st.cache_data(show_spinner=False)
def load_manifest():
    return pd.read_csv(os.path.join(DATA_DIR, "manifest.csv"))


@st.cache_data(show_spinner="Sampling images for feature analysis...")
def sample_channel_and_texture(sample_n=120):
    rng = np.random.default_rng(42)
    rows = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(DATA_DIR, "train", cls)
        if not os.path.isdir(cls_dir):
            continue
        files = sorted(os.listdir(cls_dir))
        for fname in rng.choice(files, size=min(sample_n, len(files)), replace=False):
            img = cv2.cvtColor(cv2.imread(os.path.join(cls_dir, fname)), cv2.COLOR_BGR2RGB).astype(np.float32)
            r, g, b = img[..., 0].mean(), img[..., 1].mean(), img[..., 2].mean()
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            rows.append({"class_name": cls, "red_share": r / (r + g + b), "pixel_std": float(gray.std())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- pages ----
def page_status():
    st.header("Model Status & Uptime")
    if st.button("Refresh"):
        st.rerun()
    try:
        health = api_get("/health").json()
        metrics = api_get("/metrics").json()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not reach the API at {api_url()} - is it running?\n\n{exc}")
        return

    st.success(f"API is UP - {api_url()}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Uptime (s)", health.get("uptime_seconds"))
    c2.metric("Model version", health.get("model_version"))
    c3.metric("Test macro-F1", health.get("test_macro_f1"))
    c1.metric("Last retrain", str(health.get("last_retrain") or "never"))
    c2.metric("Total requests", metrics.get("total_requests"))
    c3.metric("Total predictions", metrics.get("total_predictions"))
    with st.expander("Raw /health and /metrics"):
        st.json({"health": health, "metrics": metrics})


def page_visualizations():
    st.header("Dataset Visualizations")
    manifest = load_manifest()

    st.subheader("Feature 1 - Class imbalance reflects a real screening population")
    counts = manifest["class_name"].value_counts().reindex(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="flare", legend=False, ax=ax)
    ax.set_ylabel("Image count")
    ax.tick_params(axis="x", rotation=10)
    st.pyplot(fig)
    st.caption("Most screened eyes are healthy; vision-threatening cases are rare. "
               "This is why the 3-class regrouping was needed for a trainable split.")

    df = sample_channel_and_texture()

    st.subheader("Feature 2 - Whole-image red balance does NOT separate severity")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.boxplot(data=df, x="class_name", y="red_share", hue="class_name",
                order=CLASS_NAMES, palette="flare", legend=False, ax=ax)
    ax.set_ylabel("R / (R+G+B)")
    ax.tick_params(axis="x", rotation=10)
    st.pyplot(fig)
    st.caption("Red-share is flat across classes: lesions are too small to move a whole-image average. "
               "The signal is spatially localized - an argument for a CNN over color features.")

    st.subheader("Feature 3 - Pixel variance runs opposite to the naive hypothesis")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.boxplot(data=df, x="class_name", y="pixel_std", hue="class_name",
                order=CLASS_NAMES, palette="flare", legend=False, ax=ax)
    ax.set_ylabel("Grayscale pixel std. dev.")
    ax.tick_params(axis="x", rotation=10)
    st.pyplot(fig)
    st.caption("Variance is highest in healthy retinas (dominated by broad anatomical contrast), "
               "reinforcing that global statistics miss the localized lesion signal.")


def page_predict():
    st.header("Single Image Prediction")
    uploaded = st.file_uploader("Upload one retina image (PNG or JPEG)", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload an image to get a diabetic-retinopathy grade.")
        return

    st.image(uploaded, caption=uploaded.name, width=300)
    if st.button("Predict"):
        try:
            resp = api_post("/predict",
                            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)})
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction request failed: {exc}")
            return
        if resp.status_code != 200:
            st.error(f"API returned {resp.status_code}: {resp.text}")
            return
        result = resp.json()
        st.success(f"Predicted: **{result['predicted_class']}**  "
                   f"(confidence {result['confidence']:.1%})")
        probs = pd.DataFrame({"probability": result["probabilities"]})
        st.bar_chart(probs)


def page_retrain():
    st.header("Upload Data & Retrain")

    st.subheader("1. Bulk-upload labelled images")
    label = st.selectbox("Class label for this batch", CLASS_NAMES)
    files = st.file_uploader("Upload multiple images of the selected class",
                             type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if st.button("Upload batch") and files:
        multipart = [("files", (f.name, f.getvalue(), f.type)) for f in files]
        try:
            resp = api_post("/upload", data={"label": label}, files=multipart)
            body = resp.json()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Upload failed: {exc}")
            return
        if resp.status_code != 200:
            st.error(f"API returned {resp.status_code}: {resp.text}")
        else:
            st.success(f"Saved {body['saved']} image(s). Pending uploads: {body['total_pending_uploads']} "
                       f"(auto-retrain threshold {body['auto_retrain_threshold']}).")
            if body["auto_retrain_triggered"]:
                st.session_state["job_id"] = body["job_id"]
                st.warning(f"Auto-retrain triggered (job {body['job_id']}).")

    st.subheader("2. Trigger retraining manually")
    if st.button("Trigger Retraining"):
        try:
            resp = api_post("/retrain")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Retrain request failed: {exc}")
            return
        if resp.status_code == 202:
            st.session_state["job_id"] = resp.json()["job_id"]
            st.success(f"Retraining started (job {resp.json()['job_id']}).")
        elif resp.status_code == 409:
            st.warning(resp.json().get("detail", "A retrain is already running."))
        else:
            st.error(f"API returned {resp.status_code}: {resp.text}")

    st.subheader("3. Retraining status")
    if st.button("Refresh status"):
        st.rerun()
    job_id = st.session_state.get("job_id")
    try:
        params = {"job_id": job_id} if job_id else None
        status = api_get("/status", params=params).json()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not fetch status: {exc}")
        return

    st.write(f"**Status:** {status.get('status')}  |  **Job:** {status.get('job_id')}")
    if status.get("progress"):
        st.write("Progress:", status["progress"])
    if status.get("status") == "done" and status.get("result"):
        res = status["result"]
        st.write(f"Promoted: **{res['promoted']}** "
                 f"({'new model deployed' if res['promoted'] else 'kept previous model'})")
        col1, col2 = st.columns(2)
        col1.metric("Old macro-F1", res["old_metrics"]["macro_f1"])
        col2.metric("New macro-F1", res["new_metrics"]["macro_f1"],
                    delta=round(res["new_metrics"]["macro_f1"] - res["old_metrics"]["macro_f1"], 4))
    elif status.get("status") == "error":
        st.error(f"Retrain failed: {status.get('message')}")


# --------------------------------------------------------------- main ----
def main():
    st.sidebar.title("RetinaScope")
    st.session_state["api_url"] = st.sidebar.text_input("API URL", value=api_url())
    page = st.sidebar.radio("Page", ["Status", "Visualizations", "Predict", "Retrain"])
    {"Status": page_status, "Visualizations": page_visualizations,
     "Predict": page_predict, "Retrain": page_retrain}[page]()


if __name__ == "__main__":
    main()
