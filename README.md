# SARFuse

![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=flat&logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat)

**Dual-stream SAR + Optical fusion microservice for satellite object detection.**

Ingests paired Sentinel-1 (VV/VH) and Sentinel-2 (Red/NIR/SWIR) GeoTIFFs,
runs independent YOLOv8-OBB inference streams per sensor, and merges
detections via Weighted Box Fusion into a production-grade GeoJSON
FeatureCollection — complete with geodetic coordinates, oriented bounding
box polygons, GSD-calibrated footprint area, and heading angle.

---

## Architectural Highlights

- **Streaming tile ingestion** — `rasterio` windowed reads prevent full-scene
  RAM loading; arbitrarily large GeoTIFFs handled without OOM risk.
- **Dual-stream preprocessing** — SAR bands converted to dB scale, clipped,
  and normalized into pseudo-RGB; Optical bands composited as NIR/Red/SWIR
  false-color. Both streams independently ready for direct YOLOv8 ingestion.
- **Singleton model loader** — thread-safe double-checked locking pattern;
  `yolov8m-obb.pt` loaded once at Uvicorn startup via lifespan hook,
  reused across all requests. Zero per-request cold-start overhead.
- **OBB-aware extraction** — correctly consumes `result.obb.xyxy` and
  `result.obb.xywhr`; heading angle extracted from rotation tensor, not
  approximated. Axis-aligned fallback if `xywhr` unavailable.
- **Geodetic intelligence** — each detection backprojected through the
  affine transform + CRS reprojection chain to yield true WGS84 centroid,
  rotated polygon corners, and GSD-calibrated footprint in m².
- **WBF fusion** — hand-rolled Weighted Box Fusion (no external dependency);
  SAR stream upweighted (`1.2×`) to compensate for cloud-occluded optical.
  `sar_bonus` metric quantifies detections optical missed entirely.
- **Containerized with GDAL** — Dockerfile resolves the GDAL/rasterio
  native lib dependency chain and OpenCV headless conflict; reproducible
  build on any machine.

---

## Tech Stack

| Layer | Library |
|---|---|
| Inference | `ultralytics` YOLOv8m-OBB (DOTA weights) |
| Geospatial I/O | `rasterio`, `pyproj` |
| API | `FastAPI`, `Uvicorn` |
| Preprocessing | `numpy`, `opencv-python-headless` |
| Containerization | Docker, Docker Compose |
| Output | RFC 7946 GeoJSON, GeoTIFF heatmap, PNG triptych |

---

## Process Flow
POST /api/v1/detect/tile
→ Upload SAR GeoTIFF (VV, VH) + Optical GeoTIFF (Red, NIR, SWIR)
→ Windowed rasterio read → dB normalization + false-color composite
→ YOLOv8m-OBB inference on both streams independently
→ OBB extraction (xyxy + xywhr) → affine backprojection → WGS84
→ Weighted Box Fusion → EnrichedDetection(lat, lon, m², heading°, polygon)
→ GeoJSON FeatureCollection response
