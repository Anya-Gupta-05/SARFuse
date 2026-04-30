from __future__ import annotations          
import uuid
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import rasterio
import rasterio.windows
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from inference.preprocess  import preprocess_sar, preprocess_optical
from inference.model       import get_sar_model, get_opt_model
from inference.geo_metrics import extract_enriched, EnrichedDetection
from inference.fusion      import Detection as FusionDet, weighted_box_fusion
from output.geojson_writer import build_feature_collection
from .schemas import JobStatus, HealthResponse

router = APIRouter()

_jobs: dict[str, JobStatus] = {}

TILE_SIZE = 512
CONF      = 0.25
IOU_THR   = 0.45
SAR_W     = 1.2
OPT_W     = 1.0


# ── Health ──────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok", "models": ["yolov8m-obb-sar", "yolov8m-obb-opt"]}


# ── Single-tile ─────────────────────────────────────────────────────────────

@router.post("/detect/tile")
async def detect_tile(
    sar_file: UploadFile = File(..., description="2-band GeoTIFF: Band1=VV, Band2=VH"),
    opt_file: UploadFile = File(..., description="3-band GeoTIFF: Band1=Red, Band2=NIR, Band3=SWIR"),
):
    sar_path = await _save_upload(sar_file)
    opt_path = await _save_upload(opt_file)
    try:
        result = _run_enriched_inference(sar_path, opt_path)
    finally:
        sar_path.unlink(missing_ok=True)
        opt_path.unlink(missing_ok=True)
    return JSONResponse(content=result)


# ── Async full-scene ────────────────────────────────────────────────────────

@router.post("/ingest", response_model=JobStatus, status_code=202)
async def ingest_scene(
    background_tasks: BackgroundTasks, 
    sar_file:         UploadFile = File(...),
    opt_file:         UploadFile = File(...),        
):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = JobStatus(job_id=job_id, status="queued")
    sar_bytes = await sar_file.read()
    opt_bytes = await opt_file.read()
    background_tasks.add_task(_process_scene, job_id, sar_bytes, opt_bytes)
    return _jobs[job_id]


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JSONResponse(content=job.model_dump())


# ── Core inference ──────────────────────────────────────────────────────────

def _run_enriched_inference(
    sar_path: Path,
    opt_path: Path,
    tile_id:  str = "",
) -> dict[str, Any]:
    tile_id = tile_id or f"tile_{datetime.utcnow().strftime('%H%M%S%f')}"

    with rasterio.open(sar_path) as sar_src, \
         rasterio.open(opt_path) as opt_src:

        vv   = sar_src.read(1).astype(np.float32)
        vh   = sar_src.read(2).astype(np.float32)
        red  = opt_src.read(1).astype(np.float32)
        nir  = opt_src.read(2).astype(np.float32)
        swir = opt_src.read(3).astype(np.float32)

        sar_rgb = preprocess_sar(vv, vh)
        opt_rgb = preprocess_optical(red, nir, swir)

        sar_result = get_sar_model().predict(sar_rgb, conf=CONF, verbose=False)[0]
        opt_result = get_opt_model().predict(opt_rgb, conf=CONF, verbose=False)[0]

        names = get_sar_model().names

        # Pass open rasterio source — extract_enriched reads transform + CRS
        sar_dets: list[EnrichedDetection] = extract_enriched(
            sar_result, sar_src, "sar", names
        )
        opt_dets: list[EnrichedDetection] = extract_enriched(
            opt_result, sar_src, "opt", names
        )

        src_crs = str(sar_src.crs)

    # WBF fusion
    fused_raw = weighted_box_fusion(
        [_to_fusion(d) for d in sar_dets],
        [_to_fusion(d) for d in opt_dets],
        iou_thr    = IOU_THR,
        sar_weight = SAR_W,
        opt_weight = OPT_W,
    )

    # Re-attach nearest enriched detection to each fused box
    all_enriched = sar_dets + opt_dets
    fused: list[EnrichedDetection] = []
    for f in fused_raw:
        nearest = min(
            all_enriched,
            key=lambda d: abs(d.x1-f.x1) + abs(d.y1-f.y1)
                        + abs(d.x2-f.x2) + abs(d.y2-f.y2)
        )
        nearest.source = f.source
        fused.append(nearest)

    return build_feature_collection(
        detections = fused,
        metadata   = {
            "tile_id":   tile_id,
            "crs":       src_crs,
            "n_sar_raw": len(sar_dets),
            "n_opt_raw": len(opt_dets),
            "n_fused":   len(fused),
            "sar_bonus": sum(1 for d in fused if d.source == "sar"),
        },
    )


def _to_fusion(d: EnrichedDetection) -> FusionDet:
    return FusionDet(d.x1, d.y1, d.x2, d.y2, d.confidence, 0, d.source)


# ── Background scene processor ──────────────────────────────────────────────

async def _process_scene(
    job_id:    str,
    sar_bytes: bytes,
    opt_bytes: bytes,
) -> None:
    _jobs[job_id].status = "processing"
    all_features: list[dict[str, Any]] = []

    try:
        sar_path = _bytes_to_tempfile(sar_bytes)
        opt_path = _bytes_to_tempfile(opt_bytes)

        with rasterio.open(sar_path) as sar_src, \
             rasterio.open(opt_path) as opt_src:
            H, W   = sar_src.height, sar_src.width
            tile_n = 0

            for row in range(0, H, TILE_SIZE):
                for col in range(0, W, TILE_SIZE):
                    win = rasterio.windows.Window(
                        col, row,
                        min(TILE_SIZE, W - col),
                        min(TILE_SIZE, H - row),
                    )
                    tile_sar = _write_window_to_temp(sar_src, win)
                    tile_opt = _write_window_to_temp(opt_src, win)

                    fc = _run_enriched_inference(
                        tile_sar, tile_opt,
                        tile_id=f"tile_{tile_n:04d}_r{row}_c{col}"
                    )
                    all_features.extend(fc["features"])

                    tile_sar.unlink(missing_ok=True)
                    tile_opt.unlink(missing_ok=True)
                    tile_n += 1
                    await asyncio.sleep(0)

        sar_path.unlink(missing_ok=True)
        opt_path.unlink(missing_ok=True)

        _jobs[job_id].status = "done"
        _jobs[job_id].result = {
            "type":     "FeatureCollection",
            "features": all_features,
            "metadata": {
                "job_id":            job_id,
                "total_tiles":       tile_n,
                "total_detections":  len(all_features),
            },
        }

    except Exception as e:
        _jobs[job_id].status  = "error"
        _jobs[job_id].message = str(e)
        raise


# ── File helpers ────────────────────────────────────────────────────────────

async def _save_upload(f: UploadFile) -> Path:
    return _bytes_to_tempfile(await f.read())


def _bytes_to_tempfile(data: bytes, suffix: str = ".tif") -> Path:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        return Path(tmp.name)


def _write_window_to_temp(
    src: rasterio.io.DatasetReader,   # ← FIX 1: correct fully-qualified type
    win: rasterio.windows.Window,     # ← FIX 1: correct fully-qualified type
) -> Path:
    data      = src.read(window=win)
    transform = src.window_transform(win)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        with rasterio.open(
            tmp.name, "w",
            driver    = "GTiff",
            height    = win.height,
            width     = win.width,
            count     = src.count,
            dtype     = data.dtype,
            crs       = src.crs,
            transform = transform,
        ) as dst:
            dst.write(data)
        return Path(tmp.name)