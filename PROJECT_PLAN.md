# Project plan: Multimodal Video Search Engine

## 1. Objective

This repository implements a multimodal video search pipeline that:

1. tracks objects in raw videos,
2. saves quality-filtered crops with per-frame metadata,
3. embeds crops and text via CLIP,
4. stores embeddings and metadata for retrieval,
5. exposes a lightweight search workflow for later API/UI integration.

The current codebase is the active implementation and is more advanced than the original Phase 1 notes.

---

## 2. Runtime environment

- OS: Linux
- Python: 3.10 in the `video_rag` conda environment
- Database image: `pgvector/pgvector:0.8.0-pg17-bookworm`
- Database credentials are read from `docker/.env`; values are not committed to source control
- Example connection URI format:
  `postgresql+psycopg2://<user>:<password>@localhost:5432/videorag`

The current project uses a local Docker Postgres + pgvector instance defined in `docker/docker-compose.yml`.

---

## 3. Repository state

### Current implementation

- `src/vision/tracker.py`
  - object tracking via Ultralytics YOLO
  - configurable sampling interval and stride
  - bounding-box padding
  - minimum size, edge, and blur filtering
  - metadata export as `metadata.jsonl`
  - per-track crop saving to `data/crops/<video_id>/<track_id>/fNNNNNN.jpg`

- `src/vision/embedder.py`
  - `CLIPEmbedder` class for image and text embedding
  - methods:
    - `embed_images(paths, batch_size=64)`
    - `embed_images_with_indices(paths, batch_size=64)`
    - `embed_text(text)`
  - L2-normalized embeddings
  - square-padding preprocessing to protect the crop from center crop loss

- `src/utils/config.py`
  - shared config loader
  - crop path resolution relative to the crop root

- `configs/vision.yaml`
  - tracker + embedder configuration with defaults for `classes`, `sample_interval_sec`, `blur_var_min`, `output_dir`, and model settings

- `requirements.txt`
  - active Python dependencies for vision, CLIP, DB, and API layers

- `tests/`
  - fast tests for tracker, filters, pad-to-square, and embedder behavior
  - slow tests are controlled via the pytest marker

---

## 4. Tracking pipeline details

### Supported classes

The tracker is configured for the following COCO IDs:

- `0` person
- `1` bicycle
- `2` car
- `3` motorcycle
- `5` bus
- `7` truck

### Sampling behavior

Current tracker defaults:

- `vid_stride: 1`
- `sample_interval_sec: 0.6`
- `conf_min: 0.25`
- `pad_ratio: 0.1`
- `min_box_w: 32`
- `min_box_h: 32`
- `edge_margin_px: 5`
- `blur_var_min: 30.0`
- `blur_eval_size: 128`

### Crop structure

Crops are stored as:

- `data/crops/<video_id>/<track_id>/f000123.jpg`

This is the live structure used by the pipeline; file names are not the legacy prototype format.

### Metadata schema

Each saved crop record includes at least:

- `video_id`
- `track_id`
- `frame_idx`
- `timestamp_sec`
- `class_id`
- `class_name`
- `conf`
- `bbox_xyxy`
- `frame_w`
- `frame_h`
- `fps`
- `blur_score`
- `crop_path` (relative to the crop root)

Example:

- `"crop_path": "vid1/12/f000123.jpg"`

---

## 5. Embedding API and config

The current embedder layer exposes the following API:

- `embed_images(paths, batch_size=64)`
- `embed_images_with_indices(paths, batch_size=64)`
- `embed_text(text)`

The config file for the current pipeline is:

- `configs/vision.yaml`

This file contains both tracker and embedder sections, including:

- `tracker.weights`
- `tracker.classes`
- `tracker.sample_interval_sec`
- `tracker.blur_var_min`
- `tracker.output_dir`
- `embedder.model_name`
- `embedder.batch_size`
- `embedder.output_dir`

---

## 6. Test and tooling status

### Fast tests

Fast validation is run by default:

- `pytest -q`
- `pytest.ini` contains the default filter: `-m "not slow"`

### Slow tests

Heavy or model-backed checks are marked with:

- `@pytest.mark.slow`

These tests are intentionally excluded from the default suite and should be run separately when needed.

### Requirements

The active dependency set is defined in `requirements.txt` and includes:

- `ultralytics`
- `lap`
- `transformers`
- `opencv-python`
- `numpy`
- `pillow`
- `pyyaml`
- `psycopg2-binary`
- `sqlalchemy`
- `pgvector`
- `fastapi`
- `uvicorn`

The runtime also expects the conda pin:

- `mkl=2024.0.0`

---

## 7. Stage roadmap

### Stage 2 — DONE

Completed items:

- padding-aware crop extraction
- per-track sampling logic
- edge filtering
- size filtering
- blur filtering with normalized scoring
- metadata export with `blur_score` and relative `crop_path`
- per-video stats output in `stats.json`

### Stage 3 — DONE for batch embedding; TODO for FAISS and evaluation

Completed:

- CLIP batch embedding via `embed_images` and `embed_images_with_indices`
- output to `.npy` + `_meta.jsonl`
- support for text embedding and retrieval preparation

Remaining TODO:

- FAISS baseline setup
- query-set generation
- Recall@K evaluation
- benchmark/reporting pipeline

Low-priority TODO:

- choose the top-N sharpest crops for each track when a track has multiple valid detections

### Stage 4+

Planned next areas:

- PostgreSQL + pgvector ingestion
- search API and metadata retrieval
- FastAPI service
- UI integration and polished project docs

---

## 8. Documentation cleanup notes

The historical references that are no longer correct should not be treated as the source of truth. In particular:

- the early prototype wording should not be used for the current system
- legacy sampling settings from the original prototype are historical only
- legacy class filters from the first prototype are not the current defaults
- legacy crop naming from the first prototype is no longer used
- old pgvector image references are superseded by the active `pgvector/pgvector:0.8.0-pg17-bookworm` image
- the current archive note is kept in `pipeline/phase_1.md`, not in the main project plan

---

## 9. Active reference files

- `pipeline/phase_1.md` — historical archive note for original Phase 1 work
- `configs/vision.yaml` — current runtime configuration
- `src/vision/tracker.py` — active tracking pipeline
- `src/vision/embedder.py` — current CLIP embedder API
- `src/utils/config.py` — config and path helpers
- `requirements.txt` — current dependency set
- `pytest.ini` — default test selection rules

This document reflects the repository as it exists now, not the original prototype notes.
