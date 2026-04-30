from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel


# ── Job lifecycle (unchanged) ──────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id:  str
    status:  Literal["queued", "processing", "done", "error"]
    message: str | None = None
    # Full GeoJSON FeatureCollection stored here when status == "done"
    result:  dict[str, Any] | None = None


# ── GeoJSON types (RFC 7946) ───────────────────────────────────────────────
# We do NOT use Pydantic to validate the geometry internals.
# Reason: Pydantic response_model performs field filtering — any key not
# declared in the model gets silently stripped. GeoJSON has deeply nested
# arbitrary geometry (Polygon coordinates, properties dict) that we must
# preserve exactly as the pipeline emits it.
#
# Solution: declare the top-level shape for documentation clarity,
# but type `features` and `metadata` as Any so nothing gets stripped.

class GeoJSONFeature(BaseModel):
    type:       Literal["Feature"]
    id:         int
    geometry:   dict[str, Any]    # {type: Polygon, coordinates: [...]}
    properties: dict[str, Any]    # class, conf, lat, lon, area, heading...

    model_config = {"extra": "allow"}   # forward-compatible with new fields


class GeoJSONFeatureCollection(BaseModel):
    type:     Literal["FeatureCollection"]
    features: list[GeoJSONFeature]
    metadata: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


# ── Health ─────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    models: list[str]