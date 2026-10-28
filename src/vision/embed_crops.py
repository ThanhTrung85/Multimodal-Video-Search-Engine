"""Batch-embed all crops for one or more videos and ingest into PostgreSQL.

Reads ``metadata.jsonl`` produced by the tracker, runs CLIP image
encoding in batches, persists the result locally as:

- ``data/embeddings/{video_id}.npy``       – float32 (N, D)
- ``data/embeddings/{video_id}_meta.jsonl`` – matching metadata rows

And automatically ingests the video, tracks, and 512-D crop embeddings
into the PostgreSQL + pgvector database.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from src.db.ingest import ingest_video_data
from src.vision.embedder import CLIPEmbedder, load_embedder_config

logger = logging.getLogger(__name__)


def _build_video_info(
    video_id: str,
    raw_videos_dir: str = "data/raw_videos",
    records: list[dict] | None = None,
) -> dict:
    """Build video metadata dictionary for database ingestion."""
    video_path = None

    # 1. Check if video_path is already provided in metadata records
    if records and "video_path" in records[0]:
        candidate = Path(records[0]["video_path"])
        if candidate.exists():
            video_path = candidate

    # 2. Otherwise search in raw_videos_dir by video_id
    if video_path is None:
        base_dir = Path(raw_videos_dir)
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            candidate = base_dir / f"{video_id}{ext}"
            if candidate.exists():
                video_path = candidate
                break

    if video_path is None:
        video_path = Path(raw_videos_dir) / f"{video_id}.mp4"

    fps: float | None = None
    width: int | None = None
    height: int | None = None
    duration_sec: float | None = None

    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or None
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
            if fps and fps > 0 and frame_count > 0:
                duration_sec = round(frame_count / fps, 2)
        cap.release()

    return {
        "video_id": video_id,
        "name": video_path.name,
        "file_path": str(video_path),
        "fps": fps,
        "duration_sec": duration_sec,
        # "width": width,
        # "height": height,
    }


def embed_video(
    video_id: str,
    embedder: CLIPEmbedder,
    crops_dir: str = "data/crops",
    output_dir: str = "data/embeddings",
    raw_videos_dir: str = "data/raw_videos",
    batch_size: int = 64,
    ingest_db: bool = True,
) -> None:
    """Embed all crops for a single video, write .npy + _meta.jsonl, and ingest into DB."""
    meta_path = Path(crops_dir) / video_id / "metadata.jsonl"
    if not meta_path.exists():
        logger.error("metadata.jsonl not found: %s", meta_path)
        return

    # ── Read metadata ────────────────────────────────────────────────────
    records: list[dict] = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        logger.warning("No records in %s – nothing to embed", meta_path)
        return

    crop_paths = [r["crop_path"] for r in records]
    logger.info("Embedding %d crops for video '%s' …", len(crop_paths), video_id)

    # ── Embed ────────────────────────────────────────────────────────────
    embeddings = embedder.embed_images(crop_paths, batch_size=batch_size)
    valid_idx = embedder._last_valid_indices

    # Keep only records whose images loaded successfully
    valid_records = [records[i] for i in valid_idx]

    if embeddings.shape[0] != len(valid_records):
        logger.error(
            "Mismatch: %d embeddings vs %d valid records",
            embeddings.shape[0], len(valid_records),
        )
        return

    skipped = len(records) - len(valid_records)
    if skipped:
        logger.warning("Skipped %d crops that failed to load", skipped)

    # ── Persist locally (.npy + _meta.jsonl) ─────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    npy_path = out / f"{video_id}.npy"
    np.save(npy_path, embeddings)

    meta_out_path = out / f"{video_id}_meta.jsonl"
    with open(meta_out_path, "w") as f:
        for rec in valid_records:
            rec_out = dict(rec)
            rec_out["model_name"] = embedder.model_name
            f.write(json.dumps(rec_out) + "\n")

    logger.info(
        "Saved %d embeddings → %s  (%s)",
        embeddings.shape[0], npy_path, meta_out_path,
    )

    # ── Ingest into PostgreSQL (pgvector) ────────────────────────────────
    if ingest_db:
        logger.info("Ingesting %d embedded crops for video '%s' into PostgreSQL …", len(valid_records), video_id)
        video_info = _build_video_info(
            video_id=video_id,
            raw_videos_dir=raw_videos_dir,
            records=valid_records,
        )

        crops_data: list[dict] = []
        for rec, emb in zip(valid_records, embeddings):
            crop_item = dict(rec)
            # Ensure bbox_xyxy key exists regardless of whether tracker saved 'bbox' or 'bbox_xyxy'
            if "bbox_xyxy" not in crop_item and "bbox" in crop_item:
                crop_item["bbox_xyxy"] = crop_item["bbox"]
            crop_item["embedding"] = emb.tolist()
            crops_data.append(crop_item)

        try:
            ingest_video_data(video_info=video_info, crops_data=crops_data)
            logger.info("Database ingestion completed for video '%s'.", video_id)
        except Exception as exc:
            logger.error("Failed to ingest video '%s' into PostgreSQL: %s", video_id, exc)
            raise


def _discover_video_ids(crops_dir: str = "data/crops") -> list[str]:
    """Return video IDs by listing subdirectories that contain metadata.jsonl."""
    base = Path(crops_dir)
    if not base.exists():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and (d / "metadata.jsonl").exists()
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-embed crops, save .npy + metadata, and ingest into DB.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--video-id", help="Embed crops for a single video ID.")
    group.add_argument("--all", action="store_true", help="Embed crops for all discovered videos.")
    p.add_argument("--config", default="configs/vision.yaml", help="Path to vision.yaml.")
    p.add_argument("--no-db", action="store_true", help="Skip PostgreSQL ingestion and only write local files.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    cfg = load_embedder_config(args.config)
    embedder = CLIPEmbedder(config_path=args.config)

    if args.all:
        video_ids = _discover_video_ids(cfg.get("crops_dir", "data/crops"))
        if not video_ids:
            logger.warning("No videos with metadata.jsonl found in data/crops/")
            return
        logger.info("Discovered %d video(s): %s", len(video_ids), video_ids)
    else:
        video_ids = [args.video_id]

    for vid in video_ids:
        embed_video(
            video_id=vid,
            embedder=embedder,
            crops_dir=cfg.get("crops_dir", "data/crops"),
            output_dir=cfg["output_dir"],
            raw_videos_dir=cfg.get("raw_videos_dir", "data/raw_videos"),
            batch_size=cfg["batch_size"],
            ingest_db=not args.no_db,
        )


if __name__ == "__main__":
    main()