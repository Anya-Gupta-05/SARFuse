FROM python:3.11-slim

# GDAL/rasterio system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download YOLO weights at build time — no cold pull at runtime
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m-obb.pt')"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]