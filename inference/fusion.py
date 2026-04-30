from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ultralytics.engine.results import Results


@dataclass
class Detection:
    x1: float; y1: float; x2: float; y2: float   # normalized [0,1]
    score: float
    label: int
    source: str   # "sar" | "opt" | "fused"


def _extract(result: Results, source: str) -> list[Detection]:
    """Pull normalized xyxy boxes from a YOLOv8 result object."""
    detections = []
    boxes = result.obb if hasattr(result, "obb") and result.obb else result.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    xyxyn  = boxes.xyxyn.cpu().numpy()   # (N,4) normalized
    confs  = boxes.conf.cpu().numpy()    # (N,)
    labels = boxes.cls.cpu().numpy().astype(int)  # (N,)

    for (x1, y1, x2, y2), sc, lb in zip(xyxyn, confs, labels):
        detections.append(Detection(x1, y1, x2, y2, float(sc), int(lb), source))
    return detections


def weighted_box_fusion(
    sar_dets: list[Detection],
    opt_dets: list[Detection],
    iou_thr: float = 0.45,
    skip_thr: float = 0.25,
    sar_weight: float = 1.0,
    opt_weight: float = 1.0,
) -> list[Detection]:
    """
    Manual WBF implementation — no external dependency.
    Fuses detections from two streams by weighted centroid averaging
    of IoU-overlapping clusters.
    """
    all_dets = (
        [(d, sar_weight) for d in sar_dets] +
        [(d, opt_weight) for d in opt_dets]
    )
    all_dets = [(d, w) for d, w in all_dets if d.score >= skip_thr]

    if not all_dets:
        return []

    # Sort descending by weighted score
    all_dets.sort(key=lambda x: x[0].score * x[1], reverse=True)

    clusters: list[list[tuple[Detection, float]]] = []
    used = [False] * len(all_dets)

    for i, (di, wi) in enumerate(all_dets):
        if used[i]:
            continue
        cluster = [(di, wi)]
        used[i] = True
        for j, (dj, wj) in enumerate(all_dets):
            if used[j] or i == j:
                continue
            if _iou(di, dj) >= iou_thr and di.label == dj.label:
                cluster.append((dj, wj))
                used[j] = True
        clusters.append(cluster)

    fused = []
    for cluster in clusters:
        total_w = sum(sc * w for d, w in cluster for sc in [d.score])
        x1 = sum(d.x1 * d.score * w for d, w in cluster) / total_w
        y1 = sum(d.y1 * d.score * w for d, w in cluster) / total_w
        x2 = sum(d.x2 * d.score * w for d, w in cluster) / total_w
        y2 = sum(d.y2 * d.score * w for d, w in cluster) / total_w
        score  = total_w / sum(w for _, w in cluster)
        label  = max(set(d.label for d, _ in cluster),
                     key=lambda lb: sum(d.score for d, _ in cluster if d.label == lb))
        fused.append(Detection(x1, y1, x2, y2, score, label, "fused"))

    return fused


def _iou(a: Detection, b: Detection) -> float:
    """Axis-aligned IoU on normalized coords."""
    ix1 = max(a.x1, b.x1); iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2); iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / (area_a + area_b - inter)