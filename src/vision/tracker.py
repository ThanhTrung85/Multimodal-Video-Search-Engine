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
from ultralytics import YOLO

from src.utils.config import load_section

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
    "blur_eval_size": 128,
    "output_dir": "data/crops",
}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load tracker config from YAML, falling back to built-in defaults."""
    return load_section(config_path, "tracker", _TRACKER_DEFAULTS)


# ── Quality-filter helpers ───────────────────────────────────────────────────

def compute_blur_score(crop_bgr: np.ndarray) -> float:
    """Return the Laplacian variance (higher = sharper)."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_blur_score_normalized(crop_bgr: np.ndarray, size: int = 128) -> float:
    """Compute blur on a normalized square crop without altering the saved crop.

    The input crop is resized to a fixed square size before measuring blur.
    This keeps the blur threshold comparable across variable crop dimensions.
    """
    if crop_bgr.size == 0:
        return 0.0

    h, w = crop_bgr.shape[:2]
    target = max(1, int(size))
    if h == 0 or w == 0:
        return 0.0

    interp = cv2.INTER_AREA if max(h, w) > target else cv2.INTER_LINEAR
    resized = cv2.resize(crop_bgr, (target, target), interpolation=interp)
    return compute_blur_score(resized)


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
    def process_video(
        self,
        video_path: str | Path,
        video_id: str,
        blur_report: bool = False,
    ) -> list[dict[str, Any]]:
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
        vid_stride = max(1, int(self.cfg.get("vid_stride", 1)))
        blur_eval_size = max(1, int(self.cfg.get("blur_eval_size", 128)))

        # Per-track bookkeeping
        last_crop_frame: dict[int, int] = {}

        # Attempt / rejection counters
        stats = {
            "attempts": 0,
            "accepted": 0,
            "rejected_too_small": 0,
            "rejected_near_edge": 0,
            "rejected_too_blurry": 0,
        }
        blur_scores_by_class: dict[str, list[float]] = {}

        # Output paths
        output_base = Path(self.cfg["output_dir"]) / video_id
        output_base.mkdir(parents=True, exist_ok=True)

        metadata_records: list[dict[str, Any]] = []
        frame_idx = 0

        while cap.isOpened():
            if frame_idx % vid_stride != 0:
                if not cap.grab():
                    break
                frame_idx += 1
                continue

            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame,
                persist=True,
                classes=self.cfg["classes"],
                tracker="bytetrack.yaml",
                conf=self.cfg["conf_min"],
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

                # Sampling interval: first appearance or enough time elapsed
                if track_id in last_crop_frame and (frame_idx - last_crop_frame[track_id]) < sample_interval:
                    continue

                stats["attempts"] += 1

                # Filter: minimum size
                if bw < self.cfg["min_box_w"] or bh < self.cfg["min_box_h"]:
                    stats["rejected_too_small"] += 1
                    continue

                # Filter: edge proximity
                if is_near_edge(x1, y1, x2, y2, frame_w, frame_h, self.cfg["edge_margin_px"]):
                    stats["rejected_near_edge"] += 1
                    continue

                # Padded crop
                px1, py1, px2, py2 = pad_bbox(x1, y1, x2, y2, self.cfg["pad_ratio"], frame_w, frame_h)
                crop_img = frame[py1:py2, px1:px2]
                if crop_img.size == 0:
                    continue

                # Filter: blur (normalized across crop size)
                blur_score = compute_blur_score_normalized(crop_img, blur_eval_size)
                class_name = model.names[int(cls_id)]
                blur_scores_by_class.setdefault(class_name, []).append(float(blur_score))
                if blur_score < self.cfg["blur_var_min"]:
                    stats["rejected_too_blurry"] += 1
                    continue

                # ── Save crop ───────────────────────────────────────────
                stats["accepted"] += 1
                track_dir = output_base / str(track_id)
                track_dir.mkdir(parents=True, exist_ok=True)
                crop_fname = f"f{frame_idx:06d}.jpg"
                crop_abs = track_dir / crop_fname
                cv2.imwrite(str(crop_abs), crop_img)

                crop_rel = str(Path(video_id) / str(track_id) / crop_fname)

                record: dict[str, Any] = {
                    "video_id": video_id,
                    "track_id": int(track_id),
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 4),
                    "class_id": int(cls_id),
                    "class_name": class_name,
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
                    "blur_score": round(float(blur_score), 2),
                    "crop_path": crop_rel,
                }
                metadata_records.append(record)
                last_crop_frame[track_id] = frame_idx

            frame_idx += 1

            if frame_idx % 500 == 0:
                logger.info("  … %d frames processed, %d crops saved so far", frame_idx, len(metadata_records))

        cap.release()

        stats_path = output_base / "stats.json"
        attempts = max(1, stats["attempts"])
        stats_report = {
            "attempts": stats["attempts"],
            "accepted": stats["accepted"],
            "rejected_too_small": stats["rejected_too_small"],
            "rejected_near_edge": stats["rejected_near_edge"],
            "rejected_too_blurry": stats["rejected_too_blurry"],
            "percentages": {
                "accepted": round((stats["accepted"] / attempts) * 100.0, 2),
                "rejected_too_small": round((stats["rejected_too_small"] / attempts) * 100.0, 2),
                "rejected_near_edge": round((stats["rejected_near_edge"] / attempts) * 100.0, 2),
                "rejected_too_blurry": round((stats["rejected_too_blurry"] / attempts) * 100.0, 2),
            },
        }
        with open(stats_path, "w") as f:
            json.dump(stats_report, f, indent=2)

        # ── Write metadata.jsonl ────────────────────────────────────────
        meta_path = output_base / "metadata.jsonl"
        with open(meta_path, "w") as f:
            for rec in metadata_records:
                f.write(json.dumps(rec) + "\n")

        # ── Summary log ─────────────────────────────────────────────────
        logger.info("Video '%s' complete – %d frames, %d crops saved", video_id, frame_idx, len(metadata_records))
        logger.info("  attempts=%d accepted=%d", stats["attempts"], stats["accepted"])
        for key, label in (
            ("rejected_too_small", "too_small"),
            ("rejected_near_edge", "near_edge"),
            ("rejected_too_blurry", "too_blurry"),
        ):
            count = stats[key]
            if count:
                pct = (count / attempts) * 100.0
                logger.info("  rejected %-12s %d (%0.2f%%)", label, count, pct)
        logger.info("  metadata written to %s", meta_path)
        logger.info("  stats written to %s", stats_path)

        if blur_report:
            for class_name, values in sorted(blur_scores_by_class.items()):
                arr = np.array(values, dtype=np.float64)
                percentiles = {
                    "p5": float(np.percentile(arr, 5)),
                    "p25": float(np.percentile(arr, 25)),
                    "p50": float(np.percentile(arr, 50)),
                    "p75": float(np.percentile(arr, 75)),
                    "p95": float(np.percentile(arr, 95)),
                }
                logger.info(
                    "  blur_report class=%s p5=%.2f p25=%.2f p50=%.2f p75=%.2f p95=%.2f",
                    class_name,
                    percentiles["p5"],
                    percentiles["p25"],
                    percentiles["p50"],
                    percentiles["p75"],
                    percentiles["p95"],
                )

        return metadata_records


# ── CLI entry-point ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track objects in a video and export crops + metadata.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--video-id", required=True, help="Unique identifier for this video.")
    parser.add_argument("--config", default="configs/vision.yaml", help="Path to vision.yaml config file.")
    parser.add_argument("--blur-report", action="store_true", help="Print blur-score percentiles by class_name after processing.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
    cfg = load_config(args.config)
    tracker = VideoTracker(cfg)
    tracker.process_video(args.video, args.video_id, blur_report=args.blur_report)


if __name__ == "__main__":
    main()