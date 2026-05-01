import os
from fastapi import FastAPI
import rasterio
from rasterio.windows import Window
from pathlib import Path
from inference.pipeline import run 

app = FastAPI(title="Sentinel Geo-Tiler API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_tiles")
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/")
def health_check():
    return {"status": "Active", "message": "Terra-Edge Pipeline is ready for data ingestion."}

@app.get("/slice")
def slice_satellite_data(url: str = "https://raw.githubusercontent.com/rasterio/rasterio/main/tests/data/RGB.byte.tif"):
    try:
        with rasterio.open(url) as dataset:
            tile_size = 512
            window = Window(col_off=0, row_off=0, width=tile_size, height=tile_size)
            new_transform = dataset.window_transform(window)
            
            data = dataset.read(window=window)
            
            profile = dataset.profile
            profile.update({
                'height': tile_size,
                'width': tile_size,
                'transform': new_transform
            })
            
            output_file = os.path.join(OUTPUT_DIR, "inference_ready_tile.tif")
            
            with rasterio.open(output_file, 'w', **profile) as dest:
                dest.write(data)
            
            return {
                "pipeline_status": "Success",
                "extracted_tile": output_file
            }
    except Exception as e:
        return {"pipeline_status": "Failed", "error": str(e)}

@app.get("/detect")
def run_target_detection():
    """
    This endpoint grabs the file we just sliced and passes it to the YOLOv8 ML pipeline.
    """
    try:
        file_path = os.path.join(OUTPUT_DIR, "inference_ready_tile.tif")
        
        if not os.path.exists(file_path):
            return {"error": "Tile not found. Please run the /slice endpoint first."}
        
      
        run(Path(file_path))
        
        return {"status": "Success", "message": "Inference complete. Check terminal or results folder."}
        
    except Exception as e:
        return {"status": "Failed", "error": str(e)}
