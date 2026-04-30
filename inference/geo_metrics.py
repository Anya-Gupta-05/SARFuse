"""
Converts raw OBB pixel detections into geospatially enriched records.

Requires:
    - rasterio.DatasetReader (open source file, for transform + CRS)
    - obb.xywhr  → (N, 5) [x_c, y_c, w, h, angle_rad] in pixel space
    - obb.xyxy   → (N, 4) absolute pixel coords (axis-aligned enclosing rect)
    - obb.conf   → (N,) confidence scores
    - obb.cls    → (N,) class indices
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
import rasterio
import rasterio.warp
from rasterio.crs import CRS


WGS84 = CRS.from_epsg(4326)


@dataclass
class EnrichedDetection:
    # Core
    class_name:     str
    confidence:     float
    source:         str           # "sar" | "opt" | "fused"

    # Pixel-space (normalized [0,1])
    x1: float; y1: float; x2: float; y2: float

    # Geospatial
    centroid_lat:   float
    centroid_lon:   float
    footprint_m2:   float         # GSD-aware bounding box area
    heading_deg:    float         # OBB rotation angle, 0=East, CW positive

    # OBB corners in WGS84 (for GeoJSON Polygon geometry)
    corners_lonlat: list[list[float]] = field(default_factory=list)


def extract_enriched(
    result,
    src: rasterio.DatasetReader,
    source: str,
    model_names: dict[int, str],
) -> list[EnrichedDetection]:
    """
    Full extraction: OBB pixel coords → geodetic enrichment.

    Uses obb.xywhr for heading + oriented corners.
    Falls back to obb.xyxy if xywhr unavailable.
    """
    obb = result.obb if (hasattr(result, "obb") and result.obb is not None) else None
    if obb is None or len(obb) == 0:
        return []

    H, W = src.height, src.width
    transform = src.transform
    src_crs   = src.crs

    # GSD: metres per pixel (from affine transform, assumes metric CRS)
    # If CRS is geographic (degrees), gsd_m will be ~111000 * degrees/pixel
    gsd_x = abs(transform.a)   # metres per pixel in X
    gsd_y = abs(transform.e)   # metres per pixel in Y

    # --- Extract tensors ---
    xyxy_abs  = obb.xyxy.cpu().numpy().astype(np.float64)     # (N, 4) pixels
    confs     = obb.conf.cpu().numpy().astype(np.float64)     # (N,)
    labels    = obb.cls.cpu().numpy().astype(int)             # (N,)

    # xywhr: [x_center, y_center, width, height, angle_rad] — pixel space
    has_xywhr = hasattr(obb, "xywhr") and obb.xywhr is not None
    xywhr_abs = obb.xywhr.cpu().numpy().astype(np.float64) if has_xywhr else None

    detections: list[EnrichedDetection] = []

    for i in range(len(xyxy_abs)):
        x1_px, y1_px, x2_px, y2_px = xyxy_abs[i]
        conf  = float(confs[i])
        label = int(labels[i])
        cls   = model_names.get(label, str(label))

        # --- Centroid (pixel → world → WGS84) ---
        cx_px = (x1_px + x2_px) / 2.0
        cy_px = (y1_px + y2_px) / 2.0

        centroid_lon, centroid_lat = _pixel_to_lonlat(
            cx_px, cy_px, transform, src_crs
        )

        # --- Footprint area (m²) ---
        box_w_px = abs(x2_px - x1_px)
        box_h_px = abs(y2_px - y1_px)
        footprint_m2 = round(box_w_px * gsd_x * box_h_px * gsd_y, 1)

        # --- Heading from OBB rotation angle ---
        if xywhr_abs is not None:
            cx_ob, cy_ob, w_ob, h_ob, angle_rad = xywhr_abs[i]
            # YOLOv8-OBB: angle is rotation of the long axis from X-axis
            # Convert to compass bearing: 0=North, clockwise
            heading_deg = round(math.degrees(angle_rad) % 360, 1)
        else:
            heading_deg = 0.0

        # --- OBB corners in WGS84 (for GeoJSON Polygon) ---
        if xywhr_abs is not None:
            corners_px  = _obb_corners(cx_ob, cy_ob, w_ob, h_ob, angle_rad)
            corners_lonlat = [
                list(_pixel_to_lonlat(px, py, transform, src_crs))
                for px, py in corners_px
            ]
            corners_lonlat.append(corners_lonlat[0])   # close ring
        else:
            # Fall back to axis-aligned rect corners
            corners_lonlat = [
                list(_pixel_to_lonlat(x1_px, y1_px, transform, src_crs)),
                list(_pixel_to_lonlat(x2_px, y1_px, transform, src_crs)),
                list(_pixel_to_lonlat(x2_px, y2_px, transform, src_crs)),
                list(_pixel_to_lonlat(x1_px, y2_px, transform, src_crs)),
                list(_pixel_to_lonlat(x1_px, y1_px, transform, src_crs)),
            ]

        detections.append(EnrichedDetection(
            class_name    = cls,
            confidence    = round(conf, 4),
            source        = source,
            x1 = round(x1_px / W, 4),
            y1 = round(y1_px / H, 4),
            x2 = round(x2_px / W, 4),
            y2 = round(y2_px / H, 4),
            centroid_lat  = round(centroid_lat,  6),
            centroid_lon  = round(centroid_lon,  6),
            footprint_m2  = footprint_m2,
            heading_deg   = heading_deg,
            corners_lonlat= corners_lonlat,
        ))

    return detections


# ── Helpers ─────────────────────────────────────────────────────────────────

def _pixel_to_lonlat(
    px: float, py: float,
    transform,
    src_crs: CRS,
) -> tuple[float, float]:
    """Pixel (col, row) → (lon, lat) WGS84."""
    x_world, y_world = rasterio.transform.xy(transform, py, px, offset="center")
    (lon,), (lat,) = rasterio.warp.transform(src_crs, WGS84, [x_world], [y_world])
    return float(lon), float(lat)


def _obb_corners(
    cx: float, cy: float,
    w: float, h: float,
    angle_rad: float,
) -> list[tuple[float, float]]:
    """Compute 4 corner pixel coordinates of an oriented bounding box."""
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    hw, hh = w / 2.0, h / 2.0

    offsets = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [
        (cx + dx * cos_a - dy * sin_a,
         cy + dx * sin_a + dy * cos_a)
        for dx, dy in offsets
    ]