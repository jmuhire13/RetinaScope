"""Locust load profile for RetinaScope (Stage 7).

Floods the API (through nginx at http://localhost:8080) with prediction
requests using real test images, plus a light /health check. Run identically
against 1, 2, and 4 API replicas to compare throughput and latency:

  locust -f locustfile.py --host http://localhost:8080 \
         --users 50 --spawn-rate 10 --run-time 60s --headless --csv results_1
"""
import os
import random

from locust import HttpUser, between, task

SAMPLE_DIR = os.environ.get("LOCUST_SAMPLE_DIR", "data/test")
MAX_PER_CLASS = 8

# Load a handful of real images per class once at import, so every request
# sends genuine payloads without touching disk in the hot path.
_images = []
for _cls in os.listdir(SAMPLE_DIR):
    _cls_dir = os.path.join(SAMPLE_DIR, _cls)
    if not os.path.isdir(_cls_dir):
        continue
    for _fname in os.listdir(_cls_dir)[:MAX_PER_CLASS]:
        with open(os.path.join(_cls_dir, _fname), "rb") as _fh:
            _images.append((_fname, _fh.read()))

if not _images:
    raise RuntimeError(f"No sample images found under {SAMPLE_DIR}")


class RetinaUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict(self):
        name, data = random.choice(_images)
        self.client.post("/predict", files={"file": (name, data, "image/png")})

    @task(1)
    def health(self):
        self.client.get("/health")
