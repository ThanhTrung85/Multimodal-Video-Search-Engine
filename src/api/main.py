import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from src.api.schemas import SearchRequest, SearchResponse
from src.db.database import SessionLocal, init_db
from src.db.models import Video
from src.db.search import search_similar_crops
from src.vision.embedder import CLIPEmbedder

# Global dictionary to hold loaded AI models
ai_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle: load AI models and init DB connection once on startup."""
    print("🚀 Starting up the system...")
    init_db()  # Ensure DB and pgvector tables are ready
    
    print("🧠 Loading CLIP model into memory... (This may take a few seconds)")
    ai_models["embedder"] = CLIPEmbedder()
    print("✅ System is ready to receive queries!")
    
    yield  # Server starts processing requests here
    
    print("🛑 Shutting down system and releasing resources...")
    ai_models.clear()

app = FastAPI(title="Multimodal Video Search Engine", lifespan=lifespan)

# Allow API calls from frontends hosted on different ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/search", response_model=SearchResponse)
async def search_video_objects(request: SearchRequest):
    """
    1. Receive text query -> 2. Generate 512-D Vector -> 3. Run K-NN search with PostgreSQL
    """
    try:
        embedder: CLIPEmbedder = ai_models["embedder"]
        query_vector = embedder.embed_text(request.query)
        
        # Execute the search pipeline from Stage 4
        results = search_similar_crops(
            query_vector=query_vector, 
            top_k=request.top_k
        )
        
        return SearchResponse(query=request.query, results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/videos/{video_id}/stream")
async def stream_video(video_id: str, request: Request):
    """
    Serve H264 video stream supporting fast-forward/rewind via HTTP 206 Partial Content.
    """
    with SessionLocal() as db:
        video_record = db.query(Video).filter(Video.video_id == video_id).first()
        if not video_record or not os.path.exists(video_record.file_path):
            raise HTTPException(status_code=404, detail="Video not found")
            
    file_path = video_record.file_path
    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("Range")

    if not range_header:
        # If the client doesn't specify a byte range, return the entire file
        with open(file_path, "rb") as f:
            return Response(content=f.read(), media_type="video/mp4")

    # Process byte slicing based on the Range header
    byte1, byte2 = 0, None
    try:
        match = range_header.replace("bytes=", "").split("-")
        byte1 = int(match[0])
        if len(match) > 1 and match[1]:
            byte2 = int(match[1])
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Range header")

    byte2 = byte2 if byte2 is not None else file_size - 1
    length = byte2 - byte1 + 1

    def video_generator():
        with open(file_path, "rb") as f:
            f.seek(byte1)
            chunk_size = 1024 * 1024  # Stream in 1MB chunks
            remaining = length
            while remaining > 0:
                data = f.read(min(chunk_size, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {byte1}-{byte2}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
    }
    
    return StreamingResponse(
        video_generator(), 
        status_code=206, 
        headers=headers, 
        media_type="video/mp4"
    )