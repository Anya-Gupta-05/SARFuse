def build_feature_collection(
    detections: list,
    metadata:   dict | None = None,
) -> dict:
    """
    Returns plain Python dict — RFC 7946 FeatureCollection.
    No file I/O. Called directly by API routes.
    """
    features = []
    for i, d in enumerate(detections):
        features.append({
            "type": "Feature",
            "id":   i,
            "geometry": {
                "type":        "Polygon",
                "coordinates": [d.corners_lonlat],
            },
            "properties": {
                "id":           i,
                "class":        d.class_name,
                "confidence":   d.confidence,
                "source":       d.source,
                "centroid_lat": d.centroid_lat,
                "centroid_lon": d.centroid_lon,
                "footprint_m2": d.footprint_m2,
                "heading_deg":  d.heading_deg,
                "bbox_norm":    [d.x1, d.y1, d.x2, d.y2],
            },
        })
    return {
        "type":     "FeatureCollection",
        "features": features,
        "metadata": metadata or {},
    }