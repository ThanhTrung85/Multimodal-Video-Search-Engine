from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: str = Field(..., description="Text query describing the object to search for")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")

class SearchResultItem(BaseModel):
    video_id: str
    video_path: str
    track_db_id: int
    class_name: str
    frame_idx: int
    timestamp_sec: float
    bbox_xyxy: list[float] | dict  # Supports list or dict depending on DB serialization
    crop_path: str
    cosine_distance: float
    similarity_score: float

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]