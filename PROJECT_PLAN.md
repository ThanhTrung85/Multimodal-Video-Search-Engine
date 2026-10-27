# PROJECT CONTEXT & MASTER EXECUTION PLAN: Multimodal Video Search Engine

> **Instructions for AI Coding Assistant:**
> You are acting as a Senior AI & MLOps Engineer Pair-Programmer. Read this entire document carefully before proposing or writing any code. It contains the exact system environment, current repository state, completed modules, and the step-by-step roadmap we are following. Always build incrementally, write modular production-grade code with type hints, and provide verification commands after each step.

---

## 1. Project Overview & Core Objective

We are building an **End-to-End Multimodal Video Search Engine (Video RAG)** for a Computer Vision / AI Engineer portfolio.
- **Goal:** Allow users to search for specific objects, people, vehicles, or moments inside surveillance/traffic videos using natural language queries (e.g., *"person wearing a red jacket carrying a backpack"* or *"white SUV turning left"*).
- **Core Architecture:**
  1. **Ingestion Pipeline (Offline / Background):** Decode raw video (OpenCV) -> Detect & Track objects (YOLOv10s + ByteTrack) -> Smart Crop & Quality Filter -> Extract normalized 512-D visual embeddings (CLIP) -> Store metadata & vectors in PostgreSQL (pgvector with HNSW index).
  2. **Retrieval Pipeline (Online / Real-time):** User inputs text query -> Encode text into 512-D vector (CLIP Text Encoder) -> Cosine similarity search (pgvector k-NN grouped by track_id) + Metadata filtering -> Serve JSON results, cropped thumbnails, and video streams (with HTTP 206 Range Requests) via FastAPI -> Render interactive Web UI with timestamp seeking and bounding box overlay.

---

## 2. System Environment & Infrastructure (ALREADY CONFIGURED)

Do NOT ask the user to reinstall these unless adding a strictly necessary new dependency:
- **OS:** Ubuntu Linux
- **Project Root:** `/media/thanh-trung/DATA/Project/Multimodal-Video-Search-Engine`
- **Conda Environment:** `video_rag` (`Python 3.10.12`)
  - Installed packages: `torch`, `torchvision` (CUDA 12.1), `mkl=2024.0.0` (fixed libtorch_cpu.so symbol issue), `ultralytics`, `lap==0.5.13`, `transformers`, `opencv-python`, `fastapi`, `uvicorn[standard]`, `psycopg2-binary`, `sqlalchemy`, `pgvector`.
- **Database Container (Running via Docker Compose):**
  - Image: `ankane/pgvector:latest`
  - Container Name: `video_rag_db`
  - Host/Port: `localhost:5432`
  - Credentials: `POSTGRES_USER=root`, `POSTGRES_PASSWORD=password123`, `POSTGRES_DB=videorag`
  - Connection URI: `postgresql+psycopg2://root:password123@localhost:5432/videorag`

---

## 3. Current Repository Structure & Completed Work (Phase 1 - DONE)

[Folder Structure]
Multimodal-Video-Search-Engine/
├── docker/
│   └── docker-compose.yml       # [DONE] PostgreSQL + pgvector container config
├── data/                        # [GIT IGNORED]
│   ├── yolov10s.pt              # [DONE] Pre-downloaded YOLOv10s weights
│   ├── raw_videos/
│   │   └── test_video.mp4       # [DONE] Sample test video
│   └── crops/                   # [DONE] Generated object crops (id_X_frame_Y.jpg)
├── docs/
│   └── phase_1.md  # [DONE] Phase 1 documentation
├── notebooks/                   # For Stage 3 FAISS baseline & evaluation metrics
├── configs/                     # For configuration files
├── src/
│   ├── __init__.py
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── tracker.py           # [DONE - Basic v1] YOLOv10s + ByteTrack + Smart Cropping
│   │   └── embedder.py          # [DONE - Basic v1] CLIPEmbedder (image & text 512-D L2-norm)
│   ├── db/
│   │   └── __init__.py          # [TODO - Stage 4] Database models, connection, ingestion, search
│   ├── api/
│   │   └── __init__.py          # [TODO - Stage 5] FastAPI endpoints & video streaming
│   └── utils/                   # [TODO] Helper utilities (config, logging, image quality)
├── .gitignore                   # [DONE] Ignores data/, *.mp4, *.pt, __pycache__, etc.
├── Makefile                     # [TODO] Automation commands
└── README.md                    # [TODO] Final documentation
└── requirements.txt
└── run_test_ingest.txt

### Summary of Existing Code (Phase 1):
1. **`src/vision/tracker.py`**:
   - Loads `data/yolov10s.pt` via `ultralytics.YOLO`.
   - Reads `data/raw_videos/test_video.mp4` using `cv2.VideoCapture`.
   - Runs `model.track(frame, persist=True, classes=[0, 2], tracker="bytetrack.yaml", verbose=False)`.
   - Uses `sample_rate=15` per `track_id` to crop bounding boxes and save `.jpg` files to `data/crops/id_{track_id}_frame_{frame_idx}.jpg`.
   - *Verified working:* Successfully tracked objects and saved cropped images to `data/crops/`.
2. **`src/vision/embedder.py`**:
   - Defines class `CLIPEmbedder(model_name="openai/clip-vit-base-patch32")`.
   - Automatically selects `"cuda"` or `"cpu"`.
   - Implements `embed_image(image_path) -> list[float]` and `embed_text(text) -> list[float]` with L2 normalization, returning a 512-dimensional float list.

---

## 4. Master Roadmap: What Needs to Be Built Next

We will execute the remaining stages sequentially. **Wait for the user's confirmation after completing each stage before moving to the next.**

### Stage 2 (Refinement): Enhanced Tracking, Quality Filtering & Metadata Export
Upgrade `src/vision/tracker.py` from a simple image saver to a structured metadata & quality-aware pipeline:
- **BBox Padding:** Add configurable padding (e.g., 10-15% context around the bounding box) before cropping so CLIP captures surrounding context.
- **Quality Filtering:**
  - Filter out tiny boxes (e.g., width < 32 or height < 32) and low confidence detections.
  - Compute blur score using Laplacian variance to discard severely motion-blurred crops.
  - Select top-N sharpest crops per track_id.
- **Structured Metadata Output:** Return/export structured metadata (`video_id`, `track_id`, `class_name`, `frame_idx`, `timestamp_sec`, `bbox_xyxy`, `crop_path`) so downstream stages can ingest it cleanly.

### Stage 3: Embedding Baseline & Quantitative Evaluation (Recall@K)
Before touching PostgreSQL, establish an evaluation benchmark:
- **Batch Embedding:** Add batch inference (`embed_images_batch`) to `CLIPEmbedder`.
- **Baseline Search Script:** Encode crops, perform exact cosine search via numpy/faiss, and group results by `track_id`. Compute Recall@K metrics.

### Stage 4: PostgreSQL + pgvector Storage & Ingestion Pipeline
Build the production storage layer inside `src/db/`:
- **`src/db/database.py`:** SQLAlchemy engine, session, and `pgvector` initialization (`CREATE EXTENSION IF NOT EXISTS vector`).
- **`src/db/models.py`:** Define 3 normalized ORM tables: `videos`, `tracks`, and `crops` (with `VECTOR(512)` column and HNSW index).
- **`src/db/ingest.py`:** End-to-end ingestion script: `tracker.py` -> `embedder.py` -> bulk insert into PostgreSQL.
- **`src/db/search.py`:** Query function performing Cosine Similarity search (`1 - (embedding <=> query_vec)`).

### Stage 5: FastAPI Backend & Video Streaming Server
Build the REST API inside `src/api/`:
- **`src/api/main.py`:**
  - `POST /api/search`: Accepts text query, runs CLIP text encoder, searches DB, returns JSON.
  - `GET /api/videos/{video_id}/stream`: Implements HTTP 206 Partial Content (Range Requests) for smooth video seeking in UI.

### Stage 6: Interactive Web Frontend
- Search bar for queries.
- Results grid displaying thumbnails.
- Interactive Video Player modal: Click a result -> play video sought to target timestamp with bounding box overlay.

### Stage 8: Packaging, Dockerization & Portfolio Polish
- Write a clean `Makefile`.
- Write a slim production `Dockerfile` (using pip).
- Write an executive-grade `README.md` with Demo GIF and Benchmark Tables.

---

## 5. Coding Rules & Collaboration Protocol for AI Assistant

1. **Step-by-Step Execution:** Never dump code for multiple stages at once. Focus only on the specific stage or task the user asks for.
2. **Code Quality:**
   - Write clean, modular Python 3.10+ code with type annotations (`typing`).
   - Use a central config module (`src/utils/config.py`) instead of hardcoding paths and DB credentials.
   - Ensure proper resource cleanup (DB sessions, file handles).
3. **Hardware Awareness:** Keep batch sizes reasonable for local GPU setup and wrap PyTorch inference in `with torch.no_grad():`.
4. **Actionable Output:** Every time you write or modify a file, tell the user:
   - Which file(s) were created/updated.
   - The exact terminal command to test/verify the change inside the `video_rag` conda environment.
