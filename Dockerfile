# Human Activity Recognition (HIMK) — implementation + frontend
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# Use headless OpenCV in Docker (no libgl1-mesa-glx needed)
RUN sed 's/opencv-python>/opencv-python-headless>/' requirements.txt > requirements.docker.txt && \
    pip install --no-cache-dir -r requirements.docker.txt

COPY config.yaml .
COPY src/ src/
COPY pipelines/ pipelines/
COPY scripts/ scripts/
COPY frontend/ frontend/

# Data and models are mounted at runtime
ENV PORT=5050
EXPOSE 5050

# Default: run the frontend (results + thesis preview)
CMD ["python", "frontend/app.py"]
