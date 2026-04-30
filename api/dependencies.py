"""
Model warmup on startup. Called once via lifespan.
Prevents cold-start latency on first real request.
"""
from inference.model import get_sar_model, get_opt_model


def warmup_models() -> None:
    import numpy as np
    print("[STARTUP] Warming up SAR model ...")
    m = get_sar_model()
    dummy = np.zeros((512, 512, 3), dtype=np.uint8)
    m.predict(dummy, conf=0.99, verbose=False)   # dummy → no real detections

    print("[STARTUP] Warming up OPT model ...")
    m2 = get_opt_model()
    m2.predict(dummy, conf=0.99, verbose=False)

    print("[STARTUP] Models warm. Ready.")