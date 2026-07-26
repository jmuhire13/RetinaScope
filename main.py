"""RetinaScope API: prediction endpoint plus health/metrics for uptime
monitoring. Kept as the single-worker entrypoint - see Dockerfile CMD.
"""
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from prediction import MODEL_PATH, get_model, load_image_from_bytes, predict

START_TIME = time.time()
REQUEST_COUNT = 0
PREDICTION_COUNT = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()  # load once at startup instead of on the first request
    yield


app = FastAPI(title="RetinaScope API", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "model_path": MODEL_PATH,
    }


@app.get("/metrics")
def metrics():
    return {
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "total_requests": REQUEST_COUNT,
        "total_predictions": PREDICTION_COUNT,
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    global REQUEST_COUNT, PREDICTION_COUNT
    REQUEST_COUNT += 1

    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(status_code=400, detail="Upload must be a PNG or JPEG image")

    data = await file.read()
    try:
        image = load_image_from_bytes(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded file as an image")

    result = predict(image)
    PREDICTION_COUNT += 1
    return result
