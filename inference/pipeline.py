"""
Full enriched pipeline. Reads a 2-band SAR GeoTIFF.
Outputs: enriched detections + GeoJSON + triptych + heatmap.
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import rasterio

from .preprocess  import preprocess_sar, preprocess_optical
from .model       import get_sar_model, get_opt_model
from .geo_metrics import extract_enriched, EnrichedDetection
from .fusion      import weighted_box_fusion

# Lazy imports — output layer only needed when saving
from output.geojson_writer import write_geojson
from output.visualizer     import triptych, confidence_heatmap

SAR_TILE_PATH  = Path("processed_tiles/inference_ready_tile.tif")
OPT_TILE_PATH  = Path("processed_tiles/optical_tile.tif")
OUT_DIR        = Path("outputs")
CONF_THRESHOLD = 0.25


def load_bands(path: Path, n_bands: int) -> tuple[np.ndarray, ...]:
    with rasterio.open(path) as src:
        bands = tuple(src.read(i+1).astype(np.float32) for i in range(n_bands))
        meta = {
            "transform": src.transform,
            "crs":       src.crs,
            "height":    src.height,
            "width":     src.width,
        }
        print(f"[INFO] {path.name}  CRS={src.crs}  "
              f"shape=({src.height}×{src.width})  bands={src.count}")
    return bands, meta


def run(
    sar_path: Path = SAR_TILE_PATH,
    opt_path: Path = OPT_TILE_PATH,
) -> dict:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tile_id = f"{sar_path.stem}_{ts}"

    # ── 1. Load ──────────────────────────────────────────────────────────
    (vv, vh), sar_meta         = load_bands(sar_path, 2)
    (red, nir, swir), opt_meta = load_bands(opt_path, 3)

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    sar_rgb = preprocess_sar(vv, vh)
    opt_rgb = preprocess_optical(red, nir, swir)
    H, W    = sar_rgb.shape[:2]

    # ── 3. Inference ──────────────────────────────────────────────────────
    print("\n[INFER] Running yolov8m-obb on SAR + Optical streams ...")
    sar_result = get_sar_model().predict(sar_rgb, conf=CONF_THRESHOLD, verbose=False)[0]
    opt_result = get_opt_model().predict(opt_rgb, conf=CONF_THRESHOLD, verbose=False)[0]

    # ── 4. Geo-enriched extraction ─────────────────────────────────────────
    names = get_sar_model().names
    with rasterio.open(sar_path) as src:
        sar_dets = extract_enriched(sar_result, src, "sar", names)
        opt_dets = extract_enriched(opt_result, src, "opt", names)

    # ── 5. WBF Fusion ──────────────────────────────────────────────────────
    # Convert EnrichedDetection → simple Detection for WBF, then re-enrich
    from .fusion import Detection as FusionDet
    def to_fusion(d: EnrichedDetection) -> FusionDet:
        return FusionDet(d.x1, d.y1, d.x2, d.y2, d.confidence, 0, d.source)

    fused_raw = weighted_box_fusion(
        [to_fusion(d) for d in sar_dets],
        [to_fusion(d) for d in opt_dets],
    )

    # Map fused boxes back to nearest enriched detection for full metadata
    fused: list[EnrichedDetection] = []
    all_enriched = sar_dets + opt_dets
    for f in fused_raw:
        nearest = min(
            all_enriched,
            key=lambda d: abs(d.x1-f.x1)+abs(d.y1-f.y1)+abs(d.x2-f.x2)+abs(d.y2-f.y2)
        )
        # Update source to "fused" if box came from both streams
        nearest.source = f.source
        fused.append(nearest)

    # ── 6. Print enriched results ──────────────────────────────────────────
    _print_results(fused)

    # ── 7. Write outputs ───────────────────────────────────────────────────
    OUT_DIR.mkdir(exist_ok=True)

    geojson_path = write_geojson(
        fused,
        OUT_DIR / f"{tile_id}.geojson",
        metadata={
            "tile_id":    tile_id,
            "crs":        str(sar_meta["crs"]),
            "n_sar_raw":  len(sar_dets),
            "n_opt_raw":  len(opt_dets),
            "n_fused":    len(fused),
            "sar_bonus":  sum(1 for d in fused if d.source == "sar"),
        }
    )

    triptych_path = triptych(
        vv, vh, red, nir, swir,
        fused,
        OUT_DIR / f"{tile_id}_triptych.png",
        tile_id=tile_id,
    )

    heatmap_path = confidence_heatmap(
        fused,
        src_transform = sar_meta["transform"],
        src_crs       = sar_meta["crs"],
        tile_h        = H,
        tile_w        = W,
        out_path      = OUT_DIR / f"{tile_id}_heatmap.tif",
    )

    print(f"\n[DONE] Outputs in {OUT_DIR}/")
    print(f"  GeoJSON  → {geojson_path.name}")
    print(f"  Triptych → {triptych_path.name}")
    print(f"  Heatmap  → {heatmap_path.name}")

    return {
        "tile_id":     tile_id,
        "n_detections": len(fused),
        "sar_bonus":   sum(1 for d in fused if d.source == "sar"),
        "geojson":     str(geojson_path),
        "triptych":    str(triptych_path),
        "heatmap":     str(heatmap_path),
    }


def _print_results(dets: list[EnrichedDetection]) -> None:
    print("\n" + "═" * 80)
    print("  SARFUSE — ENRICHED DETECTION RESULTS")
    print("═" * 80)
    if not dets:
        print("  No detections above confidence threshold.")
    else:
        hdr = f"  {'#':<3} {'class':<16} {'conf':>5}  {'lat':>10} {'lon':>11}  {'area_m2':>9}  {'hdg°':>6}  src"
        print(hdr)
        print("  " + "─" * 75)
        for i, d in enumerate(dets):
            print(
                f"  {i:<3} {d.class_name:<16} {d.confidence:>5.3f}  "
                f"{d.centroid_lat:>10.5f} {d.centroid_lon:>11.5f}  "
                f"{d.footprint_m2:>9.1f}  {d.heading_deg:>6.1f}  "
                f"{d.source}"
            )
        print("  " + "─" * 75)
        print(f"  Total: {len(dets)}  |  "
              f"SAR bonus: {sum(1 for d in dets if d.source=='sar')}  |  "
              f"Avg conf: {sum(d.confidence for d in dets)/len(dets):.3f}")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    sar = Path(sys.argv[1]) if len(sys.argv) > 1 else SAR_TILE_PATH
    opt = Path(sys.argv[2]) if len(sys.argv) > 2 else OPT_TILE_PATH
    run(sar, opt)