"""
Two output artifacts:
    1. triptych()        → 3-panel comparison PNG
    2. confidence_heatmap() → single-band GeoTIFF, same CRS as source
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
import rasterio.transform

from inference.geo_metrics import EnrichedDetection
from inference.preprocess  import preprocess_sar, preprocess_optical


# ── Triptych ─────────────────────────────────────────────────────────────────

SAR_COLOR    = "#4af0a0"
OPT_COLOR    = "#5ab0ff"
FUSED_COLOR  = "#ffb340"
BONUS_COLOR  = "#ff4444"


def triptych(
    vv:   np.ndarray,
    vh:   np.ndarray,
    red:  np.ndarray,
    nir:  np.ndarray,
    swir: np.ndarray,
    detections: list[EnrichedDetection],
    out_path: Path,
    tile_id: str = "",
) -> Path:
    """
    Three-panel comparison PNG:
        Panel 1  — SAR pseudo-RGB  + SAR-only detections (green)
        Panel 2  — Optical false-color + optical-only detections (blue)
        Panel 3  — SAR pseudo-RGB  + ALL fused detections
                   SAR-bonus targets circled in red
    """
    sar_rgb = preprocess_sar(vv, vh)
    opt_rgb = preprocess_optical(red, nir, swir)
    H, W    = sar_rgb.shape[:2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0a0b0d")

    titles   = ["SAR STREAM\nVV / VH / ratio", "OPTICAL STREAM\nNIR / Red / SWIR", "FUSED INFERENCE"]
    images   = [sar_rgb, opt_rgb, sar_rgb]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, color="#aaa", fontsize=9, fontfamily="monospace",
                     pad=6, loc="left")
        ax.axis("off")
        ax.set_facecolor("#0a0b0d")

    # ── Panel 1: SAR-only detections ──────────────────────────────────────
    sar_dets = [d for d in detections if d.source == "sar"]
    _draw_boxes(axes[0], sar_dets, W, H, SAR_COLOR, label=True)

    # ── Panel 2: Optical-only detections ──────────────────────────────────
    opt_dets = [d for d in detections if d.source == "opt"]
    _draw_boxes(axes[1], opt_dets, W, H, OPT_COLOR, label=True)

    # ── Panel 3: Fused — colour by source, bonus circled ─────────────────
    for d in detections:
        color = SAR_COLOR if d.source == "sar" else OPT_COLOR \
                if d.source == "opt" else FUSED_COLOR
        _draw_single_box(axes[2], d, W, H, color, label=True)

        # Bonus: SAR targets with no optical overlap get a red circle
        if d.source == "sar":
            cx = ((d.x1 + d.x2) / 2) * W
            cy = ((d.y1 + d.y2) / 2) * H
            r  = max((d.x2 - d.x1) * W, (d.y2 - d.y1) * H) * 0.75
            circle = plt.Circle((cx, cy), r, color=BONUS_COLOR,
                                 fill=False, lw=1.2, linestyle="--")
            axes[2].add_patch(circle)

    # ── Stats overlay on panel 3 ──────────────────────────────────────────
    fused_n = len(detections)
    bonus_n = sum(1 for d in detections if d.source == "sar")
    stats_txt = (
        f"TOTAL FUSED: {fused_n}   "
        f"SAR BONUS: +{bonus_n}   "
        f"TILE: {tile_id}"
    )
    fig.text(0.5, 0.02, stats_txt, ha="center", va="bottom",
             color="#888", fontsize=8, fontfamily="monospace")

    # ── Legend ──────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(color=SAR_COLOR,   label="SAR only"),
        mpatches.Patch(color=OPT_COLOR,   label="Optical only"),
        mpatches.Patch(color=FUSED_COLOR, label="Fused"),
        mpatches.Patch(color=BONUS_COLOR, label="SAR bonus (cloud-missed)"),
    ]
    axes[2].legend(handles=legend_elements, loc="lower right",
                   fontsize=7, facecolor="#111", labelcolor="white",
                   framealpha=.85, edgecolor="#333")

    plt.tight_layout(pad=1.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    print(f"[Triptych] Written → {out_path}")
    return out_path


def _draw_boxes(ax, dets, W, H, color, label=False):
    for d in dets:
        _draw_single_box(ax, d, W, H, color, label)

def _draw_single_box(ax, d, W, H, color, label=False):
    x1, y1 = d.x1 * W, d.y1 * H
    bw, bh  = (d.x2 - d.x1) * W, (d.y2 - d.y1) * H
    rect = mpatches.FancyBboxPatch(
        (x1, y1), bw, bh,
        linewidth=1.2, edgecolor=color,
        facecolor="none",
        boxstyle="square,pad=0"
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            x1 + 2, y1 - 3,
            f"{d.class_name} {d.confidence:.2f}",
            color=color, fontsize=6.5, fontfamily="monospace",
            va="bottom",
            bbox=dict(facecolor="#000", alpha=.55, pad=1, edgecolor="none")
        )


# ── Confidence heatmap GeoTIFF ────────────────────────────────────────────

def confidence_heatmap(
    detections:     list[EnrichedDetection],
    src_transform,
    src_crs,
    tile_h: int,
    tile_w: int,
    out_path: Path,
) -> Path:
    """
    Rasterise per-detection confidence into a (tile_h, tile_w) float32 array.
    Saved as a single-band GeoTIFF preserving the source CRS + affine transform.
    Directly re-ingestable by QGIS / GEE / rasterio.

    Algorithm: for each detection, paint its bounding box pixels with
    max(existing_value, confidence). Overlapping detections take the max.
    """
    heatmap = np.zeros((tile_h, tile_w), dtype=np.float32)

    for d in detections:
        x1_px = int(d.x1 * tile_w)
        y1_px = int(d.y1 * tile_h)
        x2_px = int(d.x2 * tile_w)
        y2_px = int(d.y2 * tile_h)

        # Clamp to tile bounds
        x1_px = max(0, min(x1_px, tile_w - 1))
        y1_px = max(0, min(y1_px, tile_h - 1))
        x2_px = max(0, min(x2_px, tile_w))
        y2_px = max(0, min(y2_px, tile_h))

        region = heatmap[y1_px:y2_px, x1_px:x2_px]
        heatmap[y1_px:y2_px, x1_px:x2_px] = np.maximum(
            region, d.confidence
        )

    # Write GeoTIFF — CRS + transform preserved exactly
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w",
        driver   = "GTiff",
        height   = tile_h,
        width    = tile_w,
        count    = 1,
        dtype    = "float32",
        crs      = src_crs,
        transform= src_transform,
        nodata   = 0.0,
    ) as dst:
        dst.write(heatmap, 1)

    print(f"[Heatmap] Written → {out_path}  "
          f"(max conf: {heatmap.max():.3f}, "
          f"coverage: {(heatmap > 0).mean()*100:.1f}%)")
    return out_path