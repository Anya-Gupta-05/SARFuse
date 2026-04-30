import numpy as np


def preprocess_sar(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """
    Input:  vv, vh — (512,512) float32, raw linear power
    Output: (512,512,3) uint8 — pseudo-RGB ready for YOLOv8
    Channels: [VV_dB, VH_dB, VV-VH_ratio]
    """
    eps = 1e-10

    vv_db = 10.0 * np.log10(np.clip(vv, eps, None))
    vh_db = 10.0 * np.log10(np.clip(vh, eps, None))
    ratio = vv_db - vh_db

    def norm_db(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    stack = np.stack([
        norm_db(vv_db, -25.0,  0.0),
        norm_db(vh_db, -30.0, -5.0),
        norm_db(ratio,  -5.0, 15.0),
    ], axis=-1)

    return (stack * 255).astype(np.uint8)   # ← FIX: uint8, not float


def preprocess_optical(
    red:  np.ndarray,
    nir:  np.ndarray,
    swir: np.ndarray,
) -> np.ndarray:
    """
    Input:  red, nir, swir — (512,512) float32, raw S2 DN [0, 10000]
    Output: (512,512,3) uint8 — false-color composite for YOLOv8
    """
    def norm_percentile(band: np.ndarray) -> np.ndarray:
        p2  = np.percentile(band, 2)
        p98 = np.percentile(band, 98)
        return np.clip((band - p2) / (p98 - p2 + 1e-10), 0.0, 1.0)

    stack = np.stack([
        norm_percentile(nir),
        norm_percentile(red),
        norm_percentile(swir),
    ], axis=-1)

    return (stack * 255).astype(np.uint8)   # consistent — same fix applied here