# Ingestion Script: Bring vector (from embed_crop.py) into Database.
import json
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from src.db.database import SessionLocal, init_db
from src.db.models import Video, Track, Crop


def ingest_video_data(
    video_info: dict, 
    crops_data: list[dict]
) -> None:
    """
    Ingest a batch of tracked crops and their embeddings into the PostgreSQL database.
    
    Args:
        video_info: Dictionary containing video metadata 
                    (e.g., video_id, name, file_path, duration_sec, fps).
        crops_data: List of dictionaries, where each dict represents a crop 
                    with its corresponding 512-D embedding and track_id.
    """
    # Ensure tables and extensions exist
    init_db()
    
    with SessionLocal() as session:
        try:
            # 1. UPSERT Video (Insert or Ignore if exists based on video_id)
            stmt_video = insert(Video).values(**video_info)
            stmt_video = stmt_video.on_conflict_do_nothing(index_elements=['video_id'])
            session.execute(stmt_video)
            session.commit()
            
            # Fetch the actual database ID of the video
            db_video = session.query(Video).filter_by(video_id=video_info["video_id"]).first()
            if not db_video:
                raise ValueError(f"Failed to retrieve or create video {video_info['video_id']}")

            # 2. Prepare and UPSERT Tracks
            unique_tracks = {}
            for crop in crops_data:
                t_id = crop["track_id"]
                if t_id not in unique_tracks:
                    unique_tracks[t_id] = {
                        "video_id": db_video.id,
                        "track_id": t_id,
                        "class_name": crop["class_name"]
                    }
            
            for t_data in unique_tracks.values():
                stmt_track = insert(Track).values(**t_data)
                stmt_track = stmt_track.on_conflict_do_nothing(
                    index_elements=['video_id', 'track_id']
                )
                session.execute(stmt_track)
            session.commit()

            # Fetch the track IDs from the database to map them
            db_tracks = session.query(Track).filter_by(video_id=db_video.id).all()
            track_map = {t.track_id: t.id for t in db_tracks}

            # 3. Bulk Insert Crops
            crop_objects = []
            for crop in crops_data:
                crop_objects.append(
                    Crop(
                        video_id=db_video.id,
                        track_id=track_map[crop["track_id"]],
                        frame_idx=crop["frame_idx"],
                        timestamp_sec=crop["timestamp_sec"],
                        bbox_xyxy=crop["bbox_xyxy"],
                        crop_path=crop["crop_path"],
                        embedding=crop["embedding"]
                    )
                )
            
            # Use bulk_save_objects for optimal performance
            session.bulk_save_objects(crop_objects)
            session.commit()
            print(f"Successfully ingested {len(crop_objects)} crops for video {video_info['video_id']}.")

        except Exception as e:
            session.rollback()
            print(f"Ingestion failed: {e}")
            raise