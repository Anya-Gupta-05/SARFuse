# SARFuse

![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=flat&logo=python&logoColor=white)

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

---

## Quick Start

```bash
git clone https://github.com/<your-handle>/sarfuse.git
cd sarfuse

docker compose up --build
# Uvicorn binds on http://localhost:8000
# Model warmup runs automatically on startup
```

---

## API

### `POST /api/v1/detect/tile`

Synchronous single-tile inference. Returns GeoJSON immediately.

```bash
curl -X POST http://localhost:8000/api/v1/detect/tile \
  -F "sar_file=@processed_tiles/inference_ready_tile.tif" \
  -F "opt_file=@processed_tiles/optical_tile.tif"
```

**Response — GeoJSON FeatureCollection:**

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": 0,
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [-74.04451, 40.68921],
          [-74.04398, 40.68934],
          [-74.04381, 40.68908],
          [-74.04434, 40.68895],
          [-74.04451, 40.68921]
        ]]
      },
      "properties": {
        "class":        "ship",
        "confidence":   0.812,
        "source":       "fused",
        "centroid_lat": 40.68921,
        "centroid_lon": -74.04451,
        "footprint_m2": 2847.3,
        "heading_deg":  47.3,
        "bbox_norm":    [0.31, 0.20, 0.39, 0.26]
      }
    }
  ],
  "metadata": {
    "tile_id":   "tile_143022123456",
    "crs":       "EPSG:32618",
    "n_sar_raw": 7,
    "n_opt_raw": 3,
    "n_fused":   7,
    "sar_bonus": 4
  }
}
```

### `POST /api/v1/ingest`

Asynchronous full-scene ingestion. Tiles the scene internally, returns a
`job_id` immediately. Poll `/api/v1/jobs/{job_id}` for the merged
FeatureCollection once `status == "done"`.

### `GET /api/v1/health`

```json
{ "status": "ok", "models": ["yolov8m-obb-sar", "yolov8m-obb-opt"] }
```

Full interactive docs at **`http://localhost:8000/docs`** (Swagger UI).

---

## Output Artifacts

| File | Description |
|---|---|
| `*.geojson` | RFC 7946 FeatureCollection — drop into QGIS / GEE / Mapbox |
| `*_triptych.png` | 3-panel SAR / Optical / Fused comparison with detection overlay |
| `*_heatmap.tif` | Single-band float32 GeoTIFF, confidence values, CRS preserved |

---

## Notes

Model weights (`*.pt`) and raw scene data (`*.tif`) are excluded from
version control — download `yolov8m-obb.pt` via `ultralytics` on first
run; source your own Sentinel-1/2 tiles from ESA Copernicus Open Access Hub.
