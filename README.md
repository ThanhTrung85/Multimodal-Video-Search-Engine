# Multimodal Video Search Engine

A video understanding and retrieval project that combines object tracking, crop extraction, CLIP embedding, and later-stage vector search over video metadata and visual embeddings.

## Overview

The project is organized around two main processing flows:

1. Offline ingestion pipeline
   - read raw videos
   - detect and track objects
   - crop objects with quality filtering
   - save metadata and crop artifacts
   - embed crops with CLIP
   - prepare data for database-backed retrieval

2. Search pipeline
   - encode a text query using CLIP text embeddings
   - compare to stored crop embeddings
   - rank results by similarity
   - return matched crops and metadata

## Pipeline diagram

```text
+---------------------+      +----------------------+      +-------------------------+
| Raw video files     | ---> | YOLO tracking +     | ---> | Crop save + metadata   |
| data/raw_videos/    |      | filtering + sampling |      | data/crops/...         |
+---------------------+      +----------------------+      +-------------------------+
                                                       |
                                                       v
                                           +----------------------+
                                           | CLIP image embedding |
                                           | src/vision/embedder.py |
                                           +----------------------+
                                                       |
                                                       v
                                           +----------------------+
                                           | .npy + _meta.jsonl   |
                                           | data/embeddings/     |
                                           +----------------------+
                                                       |
                                                       v
                                           +----------------------+
                                           | Vector search / DB   |
                                           | future retrieval API |
                                           +----------------------+
```

```text
User query
    |
    v
CLIP text embedding
    |
    v
Similarity search against stored crop embeddings
    |
    v
Top results + metadata + crop references
```

## System requirements

- Linux environment
- Python 3.10 in a conda environment named `video_rag`
- CUDA-capable GPU is optional for local development, but the code gracefully falls back to CPU when available
- Docker + Docker Compose for the local PostgreSQL + pgvector service
- OpenCV, Ultralytics, PyTorch, Transformers, and NumPy installed in the project environment

## Setup

### 1. Create and activate the environment

```bash
conda create -n video_rag python=3.10 -y
conda activate video_rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Important note:

```bash
conda install mkl=2024.0.0 -c conda-forge
```

This pin is required to avoid the known `libtorch_cpu.so` symbol issue in the project environment.

### 3. Start the database

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env up -d
```

The compose file reads credentials from `docker/.env` and uses the `pgvector/pgvector:0.8.0-pg17-bookworm` image.

## Run the tracker

```bash
python -m src.vision.tracker \
  --video data/raw_videos/<video>.mp4 \
  --video-id <video_id> \
  --config configs/vision.yaml
```

This creates crop outputs under `data/crops/<video_id>/` and writes:

- `metadata.jsonl`
- `stats.json`

## Run batch embedding

```bash
python -m src.vision.embed_crops --video-id <video_id> --config configs/vision.yaml --no-db
```

Or embed all discovered videos:

```bash
python -m src.vision.embed_crops --all --config configs/vision.yaml --no-db
```

This writes output to `data/embeddings/` as:

- `<video_id>.npy`
- `<video_id>_meta.jsonl`

## Run tests

Fast test suite:

```bash
pytest -q
```

Slow test suite:

```bash
pytest -q -m slow
```

## Repository structure

```text
.
├── configs/
│   └── vision.yaml
├── data/
│   ├── crops/
│   ├── embeddings/
│   └── raw_videos/
├── docker/
│   ├── .env
│   ├── .env.example
│   └── docker-compose.yml
├── docs/
│   └── archive/
├── notebooks/
├── pipeline/
│   └── phase_1.md
├── src/
│   ├── api/
│   ├── db/
│   ├── utils/
│   └── vision/
├── tests/
├── Makefile
├── PROJECT_PLAN.md
├── README.md
├── pytest.ini
├── requirements.txt
├── run_test_ingest.py
└── .gitignore
```

## Stage status

- Stage 1: historical prototype archived in `pipeline/phase_1.md`
- Stage 2: DONE
  - tracking refinement, metadata export, crop filtering, blur scoring
- Stage 3: batch embedding DONE
  - embedding pipeline is working
  - FAISS baseline, query set, and Recall@K remain TODO
- Stage 4: planned
- Stage 5: planned
- Stage 6: planned

## Kết quả

Sẽ cập nhật sau Stage 3.
