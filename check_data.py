import rasterio

# Updated URL pointing to the 'main' branch instead of 'master'
url = "https://raw.githubusercontent.com/rasterio/rasterio/main/tests/data/RGB.byte.tif"

print("Connecting to satellite data stream...")
try:
    with rasterio.open(url) as dataset:
        print("\n✅ SUCCESS: Data Stream Opened.")
        print(f"Width: {dataset.width} pixels")
        print(f"Height: {dataset.height} pixels")
        print(f"Band Count: {dataset.count} (RGB)")
        print(f"Coordinate Reference System (CRS): {dataset.crs}")
        print(f"Bounding Box: {dataset.bounds}")
except Exception as e:
    print(f"\n❌ ERROR: {e}")