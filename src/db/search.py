## Define look-up function that can find K-NN (search logic)
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.database import SessionLocal
from src.db.models import Crop, Track, Video


def search_similar_crops(
    query_vector: list[float], 
    top_k: int = 5
) -> list[dict]:
    """
    Perform a K-Nearest Neighbors (K-NN) search using Cosine Similarity.
    Groups results by track_id to ensure only the best matching crop per object is returned.
    
    Args:
        query_vector: The 512-D L2-normalized vector generated from the text query.
        top_k: The number of distinct object tracks to return.
        
    Returns:
        A list of dictionaries containing crop metadata, spatial information, and similarity distance.
    """
    with SessionLocal() as session:
        # Calculate cosine distance using pgvector's <=> operator
        # Distance = 1 - cosine_similarity. Smaller distance means higher similarity.
        distance_expr = Crop.embedding.cosine_distance(query_vector).label("distance")

        # Step 1: Subquery to find the best (minimum distance) crop for EACH track
        # DISTINCT ON (track_id) requires ORDER BY track_id, then distance
        subq = (
            session.query(
                Crop.id.label("crop_id"), 
                distance_expr
            )
            .distinct(Crop.track_id)
            .order_by(Crop.track_id, distance_expr.asc())
            .subquery()
        )

        # Step 2: Outer query to rank the unique tracks globally by their best distance
        results = (
            session.query(
                Crop, 
                subq.c.distance,
                Track.class_name,
                Video.video_id,
                Video.file_path
            )
            .join(subq, Crop.id == subq.c.crop_id)
            .join(Track, Crop.track_id == Track.id)
            .join(Video, Crop.video_id == Video.id)
            .order_by(subq.c.distance.asc())
            .limit(top_k)
            .all()
        )

        # Step 3: Format the results for the API layer
        formatted_results = []
        for crop, distance, class_name, video_id, file_path in results:
            formatted_results.append({
                "video_id": video_id,
                "video_path": file_path,
                "track_db_id": crop.track_id,
                "class_name": class_name,
                "frame_idx": crop.frame_idx,
                "timestamp_sec": crop.timestamp_sec,
                "bbox_xyxy": crop.bbox_xyxy,
                "crop_path": crop.crop_path,
                "cosine_distance": float(distance),
                "similarity_score": 1.0 - float(distance)  # Convert back to similarity [0, 1]
            })

        return formatted_results