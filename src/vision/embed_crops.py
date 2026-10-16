"""Batch-embed all crops for one or more videos.

Reads ``metadata.jsonl`` produced by the tracker, runs CLIP image
encoding in batches, and persists the result as:

- ``data/embeddings/{video_id}.npy``       – float32 (N, D)
- ``data/embeddings/{video_id}_meta.jsonl`` – matching metadata rows
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.vision.embedder import CLIPEmbedder, load_embedder_config

logger = logging.getLogger(__name__)


def embed_video(
    video_id: str,
    embedder: CLIPEmbedder,
    crops_dir: str = "data/crops",
    output_dir: str = "data/embeddings",
    batch_size: int = 64,
) -> None:
    """Embed all crops for a single video and write .npy + _meta.jsonl."""
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

    # ── Persist ──────────────────────────────────────────────────────────
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
    p = argparse.ArgumentParser(description="Batch-embed crops and save .npy + metadata.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--video-id", help="Embed crops for a single video ID.")
    group.add_argument("--all", action="store_true", help="Embed crops for all discovered videos.")
    p.add_argument("--config", default="configs/vision.yaml", help="Path to vision.yaml.")
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
            batch_size=cfg["batch_size"],
        )


if __name__ == "__main__":
    main()
