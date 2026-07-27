"""RetinaScope API: prediction, health/metrics for uptime monitoring, and the
retraining control surface (upload / retrain / status).

Runs as a SINGLE uvicorn worker (see Dockerfile): the in-memory job state and
lock below only coordinate correctly within one process. Retraining runs as a
FastAPI background task so /retrain returns immediately (202 + job_id) and the
long training never blocks the event loop.
"""
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from prediction import MODEL_PATH, get_model, load_image_from_bytes, predict, reload_model
from preprocessing import CLASS_NAMES
import retrain

START_TIME = time.time()
REQUEST_COUNT = 0
PREDICTION_COUNT = 0

ALLOWED_IMAGE_TYPES = ("image/png", "image/jpeg", "image/jpg")

# ------------------------------------------------------------- job state ----
# Guarded by _job_lock. A fresh process starts "idle" with job_id None, so a
# job that vanished in a cold restart is reported "unknown" (never a stale
# "running"), because a client's job_id won't match the reset state.
_job_lock = threading.Lock()
_job = {
    "status": "idle",       # idle | running | done | error
    "job_id": None,
    "message": None,
    "progress": None,        # {"epoch": n, "loss": .., "accuracy": ..}
    "result": None,          # retrain summary dict once done
    "started_at": None,
    "finished_at": None,
}


def _run_retrain_job(job_id: str) -> None:
    """Executes the retrain in a threadpool thread (Starlette runs sync
    background tasks off the event loop). Updates shared state as it goes and
    hot-reloads the served model if the new one is promoted."""
    def on_epoch(epoch, logs):
        with _job_lock:
            _job["progress"] = {"epoch": epoch + 1,
                                **{k: round(float(v), 4) for k, v in logs.items()}}
    try:
        result = retrain.retrain_model(on_epoch_end=on_epoch)
        with _job_lock:
            _job.update(status="done", result=result, finished_at=time.time(),
                        message="promoted" if result["promoted"] else "not promoted (no improvement)")
        if result["promoted"]:
            reload_model()  # swap the served model without a restart
    except Exception as exc:  # noqa: BLE001 - surface any failure via /status
        with _job_lock:
            _job.update(status="error", message=str(exc), finished_at=time.time())


def _start_retrain_job(background: BackgroundTasks) -> str:
    """Caller must already hold _job_lock and have confirmed no job is running."""
    job_id = uuid.uuid4().hex[:12]
    _job.update(status="running", job_id=job_id, message="starting",
                progress=None, result=None, started_at=time.time(), finished_at=None)
    background.add_task(_run_retrain_job, job_id)
    return job_id


# ------------------------------------------------------------------ app ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()  # load once at startup instead of on the first request
    yield


app = FastAPI(title="RetinaScope API", lifespan=lifespan)


@app.get("/health")
def health():
    meta = retrain.load_meta()
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "model_path": MODEL_PATH,
        "model_version": meta.get("version"),
        "last_retrain": meta.get("last_retrain"),
        "test_macro_f1": meta.get("test_macro_f1"),
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

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Upload must be a PNG or JPEG image")

    data = await file.read()
    try:
        image = load_image_from_bytes(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded file as an image")

    result = predict(image)
    PREDICTION_COUNT += 1
    return result


@app.post("/upload")
async def upload_endpoint(background: BackgroundTasks,
                          label: str = Form(...),
                          files: list[UploadFile] = File(...)):
    if label not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail=f"label must be one of {CLASS_NAMES}")
    payload = []
    for f in files:
        if f.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=400, detail=f"{f.filename}: must be a PNG or JPEG image")
        payload.append((f.filename, await f.read()))

    saved = retrain.save_uploaded_images(payload, label)
    pending = retrain.count_uploads()

    # Count-based automatic trigger: once total pending uploads reach the
    # threshold, kick off retraining automatically (if none is already running).
    # Note: pending uploads are cumulative for the session (not reset per
    # retrain), so once past the threshold, later uploads may re-trigger.
    auto_triggered, job_id = False, None
    if pending >= retrain.AUTO_RETRAIN_THRESHOLD:
        with _job_lock:
            if _job["status"] != "running":
                job_id = _start_retrain_job(background)
                auto_triggered = True

    return {
        "saved": saved,
        "label": label,
        "total_pending_uploads": pending,
        "auto_retrain_threshold": retrain.AUTO_RETRAIN_THRESHOLD,
        "auto_retrain_triggered": auto_triggered,
        "job_id": job_id,
    }


@app.post("/retrain")
async def retrain_endpoint(background: BackgroundTasks):
    with _job_lock:
        if _job["status"] == "running":
            raise HTTPException(status_code=409,
                                detail=f"A retrain job is already running (job {_job['job_id']})")
        job_id = _start_retrain_job(background)
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "running"})


@app.get("/status")
def status(job_id: str = None):
    with _job_lock:
        current = dict(_job)
    if job_id is not None and current["job_id"] != job_id:
        return {"status": "unknown", "job_id": job_id,
                "detail": "No record of this job; the server may have restarted mid-job."}
    return current
