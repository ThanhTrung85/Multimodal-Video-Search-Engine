import random
from src.db.ingest import ingest_video_data

def generate_dummy_embedding(dim: int = 512) -> list[float]:
    """Generate a dummy 512-D L2-normalized vector simulating CLIP output."""
    vec = [random.uniform(-1, 1) for _ in range(dim)]
    norm = sum(x**2 for x in vec) ** 0.5
    return [x / norm for x in vec]

def test_database_ingestion():
    print("1. Preparing Video Info Dictionary...")
    # Structure 1: Original video metadata
    video_info = {
        "video_id": "test_vid_001",
        "name": "test_video.mp4",
        "file_path": "data/raw_videos/test_video.mp4",
        "duration_sec": 60.5,
        "fps": 30.0
    }

    print("2. Preparing Crops Data (List of Dictionaries)...")
    # Structure 2: List of tracked objects and their embeddings
    # Simulating 2 crops of a car (track_id = 3) and 1 crop of a person (track_id = 5)
    crops_data = [
        {
            "track_id": 3,
            "class_name": "car",
            "frame_idx": 15,
            "timestamp_sec": 0.5,
            "bbox_xyxy": [100.5, 120.0, 300.5, 400.0],
            "crop_path": "data/crops/id_3_frame_15.jpg",
            "embedding": generate_dummy_embedding()
        },
        {
            "track_id": 3,
            "class_name": "car",
            "frame_idx": 30,
            "timestamp_sec": 1.0,
            "bbox_xyxy": [150.0, 130.0, 320.0, 410.0],
            "crop_path": "data/crops/id_3_frame_30.jpg",
            "embedding": generate_dummy_embedding()
        },
        {
            "track_id": 5,
            "class_name": "person",
            "frame_idx": 30,
            "timestamp_sec": 1.0,
            "bbox_xyxy": [50.0, 60.0, 90.0, 150.0],
            "crop_path": "data/crops/id_5_frame_30.jpg",
            "embedding": generate_dummy_embedding()
        }
    ]

    print("3. Starting ingestion into PostgreSQL...")
    try:
        # Call the ingest function you just wrote
        ingest_video_data(video_info=video_info, crops_data=crops_data)
        print("=> TEST SUCCESSFUL! Data has been ingested into the DB.")
    except Exception as e:
        print(f"=> ERROR: {e}")

if __name__ == "__main__":
    test_database_ingestion()