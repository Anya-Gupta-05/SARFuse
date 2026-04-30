from ultralytics import YOLO
import threading

_lock   = threading.Lock()
_models: dict[str, YOLO] = {}

# Bumped: nano → medium. Same API, 26M params vs 3.1M.
# FAIR1M alternative: "keremberke/yolov8m-satellite-image-detection"
SAR_WEIGHTS = "yolov8m-obb.pt"
OPT_WEIGHTS = "yolov8m-obb.pt"

def get_model(key: str, weights: str) -> YOLO:
    if key not in _models:
        with _lock:
            if key not in _models:
                print(f"[MODEL] Loading {weights} ...")
                _models[key] = YOLO(weights)
    return _models[key]

def get_sar_model() -> YOLO: return get_model("sar", SAR_WEIGHTS)
def get_opt_model() -> YOLO: return get_model("opt", OPT_WEIGHTS)