FROM python:3.13-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY main.py .
COPY src/ src/
COPY models/ models/
# Base training set + frozen test set baked in, so retraining can merge base
# data with uploads and evaluate promotions without any external fetch.
COPY data/ data/

EXPOSE 8000

# Single worker: the in-memory job lock planned for retraining (Stage 5)
# only holds correctly within one process. exec replaces the shell so
# uvicorn receives SIGTERM directly instead of a shell forwarding it late.
CMD ["/bin/sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
