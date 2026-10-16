"""Object detection, tracking, and smart cropping pipeline.

Uses YOLOv10 + ByteTrack to detect and track objects in video,
then crops bounding boxes with padding, quality filtering, and
structured metadata export.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ── Default configuration ────────────────────────────────────────────────────

_TRACKER_DEFAULTS: dict[str, Any] = {
    "weights": "data/yolov10s.pt",
    "classes": [0, 1, 2, 3, 5, 7],
    "conf_min": 0.25,
    "vid_stride": 1,
    "sample_interval_sec": 0.6,
    "pad_ratio": 0.1,
    "min_box_w": 32,
    "min_box_h": 32,
    "edge_margin_px": 5,
    "blur_var_min": 30.0,
    "output_dir": "data/crops",
}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load tracker config from YAML, falling back to built-in defaults."""
    cfg = dict(_TRACKER_DEFAULTS)
    if config_path is not None:
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f) or {}
        tracker_cfg = file_cfg.get("tracker", {})
        cfg.update(tracker_cfg)
    return cfg


# ── Quality-filter helpers ───────────────────────────────────────────────────

def compute_blur_score(crop_bgr: np.ndarray) -> float:
    """Return the Laplacian variance (higher = sharper)."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_near_edge(
    x1: float, y1: float, x2: float, y2: float,
    frame_w: int, frame_h: int, margin: int,
) -> bool:
    """True if any side of the bbox is within *margin* px of the frame edge."""
    return x1 < margin or y1 < margin or x2 > frame_w - margin or y2 > frame_h - margin


def pad_bbox(
    x1: float, y1: float, x2: float, y2: float,
    pad_ratio: float, frame_w: int, frame_h: int,
) -> tuple[int, int, int, int]:
    """Expand bbox by *pad_ratio* on each side, clamped to frame bounds."""
    w = x2 - x1
    h = y2 - y1
    pad_x = w * pad_ratio
    pad_y = h * pad_ratio
    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(frame_w, int(x2 + pad_x)),
        min(frame_h, int(y2 + pad_y)),
    )


# ── Main tracker class ──────────────────────────────────────────────────────

class VideoTracker:
    """Detect, track, crop, and export metadata for a single video."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.cfg = config if config is not None else dict(_TRACKER_DEFAULTS)

    # ------------------------------------------------------------------ #
    def process_video(self, video_path: str | Path, video_id: str) -> list[dict[str, Any]]:
        """Run the full tracking + cropping pipeline on one video.

        A fresh YOLO model is instantiated each call so that ByteTrack
        state from a previous video is never carried over.

        Returns the list of metadata records (one per saved crop).
        """
        # ── Fresh model per video (A6) ──────────────────────────────────
        model = YOLO(self.cfg["weights"])

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return []

        fps: float = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 25.0
            logger.warning("FPS reported as 0 – falling back to %.1f", fps)

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sample_interval = max(1, round(fps * self.cfg["sample_interval_sec"]))

        # Per-track bookkeeping
        last_crop_frame: dict[int, int] = {}

        # Reject counters
        reject = {"low_conf": 0, "too_small": 0, "near_edge": 0, "too_blurry": 0}

        # Output paths
        output_base = Path(self.cfg["output_dir"]) / video_id
        output_base.mkdir(parents=True, exist_ok=True)

        metadata_records: list[dict[str, Any]] = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame,
                persist=True,
                classes=self.cfg["classes"],
                tracker="bytetrack.yaml",
                conf=self.cfg["conf_min"],
                vid_stride=self.cfg["vid_stride"],
                verbose=False,
            )

            boxes_obj = results[0].boxes
            if boxes_obj.id is None:
                frame_idx += 1
                continue

            xyxy = boxes_obj.xyxy.cpu().numpy()
            track_ids = boxes_obj.id.cpu().numpy().astype(int)
            confs = boxes_obj.conf.cpu().numpy()
            cls_ids = boxes_obj.cls.cpu().numpy().astype(int)

            for box, track_id, conf, cls_id in zip(xyxy, track_ids, confs, cls_ids):
                x1, y1, x2, y2 = box
                bw = x2 - x1
                bh = y2 - y1

                # Filter: confidence
                if conf < self.cfg["conf_min"]:
                    reject["low_conf"] += 1
                    continue

                # Filter: minimum size
                if bw < self.cfg["min_box_w"] or bh < self.cfg["min_box_h"]:
                    reject["too_small"] += 1
                    continue

                # Filter: edge proximity
                if is_near_edge(x1, y1, x2, y2, frame_w, frame_h, self.cfg["edge_margin_px"]):
                    reject["near_edge"] += 1
                    continue

                # Sampling interval: first appearance or enough time elapsed
                if track_id in last_crop_frame and (frame_idx - last_crop_frame[track_id]) < sample_interval:
                    continue

                # Padded crop
                px1, py1, px2, py2 = pad_bbox(x1, y1, x2, y2, self.cfg["pad_ratio"], frame_w, frame_h)
                crop_img = frame[py1:py2, px1:px2]
                if crop_img.size == 0:
                    continue

                # Filter: blur
                blur_var = compute_blur_score(crop_img)
                if blur_var < self.cfg["blur_var_min"]:
                    reject["too_blurry"] += 1
                    continue

                # ── Save crop ───────────────────────────────────────────
                track_dir = output_base / str(track_id)
                track_dir.mkdir(parents=True, exist_ok=True)
                crop_fname = f"f{frame_idx:06d}.jpg"
                crop_abs = track_dir / crop_fname
                cv2.imwrite(str(crop_abs), crop_img)

                crop_rel = str(Path(self.cfg["output_dir"]) / video_id / str(track_id) / crop_fname)

                record: dict[str, Any] = {
                    "video_id": video_id,
                    "track_id": int(track_id),
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 4),
                    "class_id": int(cls_id),
                    "class_name": model.names[int(cls_id)],
                    "conf": round(float(conf), 4),
                    "bbox_xyxy": [
                        round(float(x1), 1),
                        round(float(y1), 1),
                        round(float(x2), 1),
                        round(float(y2), 1),
                    ],
                    "frame_w": frame_w,
                    "frame_h": frame_h,
                    "fps": fps,
                    "crop_path": crop_rel,
                }
                metadata_records.append(record)
                last_crop_frame[track_id] = frame_idx

            frame_idx += 1

            if frame_idx % 500 == 0:
                logger.info("  … %d frames processed, %d crops saved so far", frame_idx, len(metadata_records))

        cap.release()

        # ── Write metadata.jsonl ────────────────────────────────────────
        meta_path = output_base / "metadata.jsonl"
        with open(meta_path, "w") as f:
            for rec in metadata_records:
                f.write(json.dumps(rec) + "\n")

        # ── Summary log ─────────────────────────────────────────────────
        logger.info("Video '%s' complete – %d frames, %d crops saved", video_id, frame_idx, len(metadata_records))
        for reason, count in reject.items():
            if count:
                logger.info("  rejected %-12s %d", reason, count)
        logger.info("  metadata written to %s", meta_path)

        return metadata_records


# ── CLI entry-point ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track objects in a video and export crops + metadata.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--video-id", required=True, help="Unique identifier for this video.")
    parser.add_argument("--config", default="configs/vision.yaml", help="Path to vision.yaml config file.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
    cfg = load_config(args.config)
    tracker = VideoTracker(cfg)
    tracker.process_video(args.video, args.video_id)


if __name__ == "__main__":
    main()