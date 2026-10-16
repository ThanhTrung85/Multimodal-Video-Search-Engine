"""Tests for tracker quality filters: small, edge, blurry."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.vision.tracker import (
    VideoTracker,
    _TRACKER_DEFAULTS,
    compute_blur_score,
    is_near_edge,
    pad_bbox,
)


# ── Helper ───────────────────────────────────────────────────────────────────

def _make_boxes(xyxy, track_ids, confs, cls_ids):
    return SimpleNamespace(
        xyxy=torch.tensor(xyxy, dtype=torch.float32),
        id=torch.tensor(track_ids, dtype=torch.float32) if track_ids else None,
        conf=torch.tensor(confs, dtype=torch.float32),
        cls=torch.tensor(cls_ids, dtype=torch.float32),
    )


def _run_one_frame_tracker(
    cfg: dict,
    tmp_path: Path,
    boxes_xyxy: list[list[float]],
    track_ids: list[int],
    confs: list[float],
    cls_ids: list[int],
    frame: np.ndarray | None = None,
) -> list[dict]:
    """Run a single-frame tracking scenario and return the metadata records."""
    if frame is None:
        frame = np.full((480, 640, 3), 180, dtype=np.uint8)

    det = _make_boxes(boxes_xyxy, track_ids, confs, cls_ids)
    counter = {"n": 0}

    def fake_read():
        if counter["n"] < 1:
            counter["n"] += 1
            return True, frame.copy()
        return False, None

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0, 5: 25.0}.get(prop, 0.0)
    mock_cap.read = fake_read

    mock_model = MagicMock()
    mock_model.names = {0: "person", 2: "car"}
    mock_model.track.return_value = [SimpleNamespace(boxes=det)]

    with (
        patch("src.vision.tracker.YOLO", return_value=mock_model),
        patch("src.vision.tracker.cv2.VideoCapture", return_value=mock_cap),
    ):
        tracker = VideoTracker(cfg)
        return tracker.process_video("v.mp4", "filter_test")


# ── Unit tests for helper functions ──────────────────────────────────────────

class TestIsNearEdge:
    def test_inside_frame(self) -> None:
        assert not is_near_edge(50, 50, 100, 100, 640, 480, margin=5)

    def test_touching_left(self) -> None:
        assert is_near_edge(3, 50, 100, 100, 640, 480, margin=5)

    def test_touching_top(self) -> None:
        assert is_near_edge(50, 2, 100, 100, 640, 480, margin=5)

    def test_touching_right(self) -> None:
        assert is_near_edge(50, 50, 637, 100, 640, 480, margin=5)

    def test_touching_bottom(self) -> None:
        assert is_near_edge(50, 50, 100, 478, 640, 480, margin=5)


class TestPadBbox:
    def test_basic_padding(self) -> None:
        px1, py1, px2, py2 = pad_bbox(100, 100, 200, 200, 0.1, 640, 480)
        assert px1 == 90
        assert py1 == 90
        assert px2 == 210
        assert py2 == 210

    def test_clamped_to_frame(self) -> None:
        px1, py1, px2, py2 = pad_bbox(0, 0, 50, 50, 0.2, 640, 480)
        assert px1 == 0
        assert py1 == 0


class TestBlurScore:
    def test_sharp_image_high_score(self) -> None:
        # Checkerboard pattern → high variance
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[::2, ::2] = 255
        score = compute_blur_score(img)
        assert score > 100

    def test_uniform_image_low_score(self) -> None:
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        score = compute_blur_score(img)
        assert score < 1


# ── Integration tests via mocked tracker ─────────────────────────────────────

class TestFilterTooSmall:
    def test_small_box_rejected(self, tmp_path: Path) -> None:
        cfg = dict(_TRACKER_DEFAULTS)
        cfg["output_dir"] = str(tmp_path / "crops")
        cfg["min_box_w"] = 32
        cfg["min_box_h"] = 32
        cfg["edge_margin_px"] = 0
        cfg["blur_var_min"] = 0.0
        cfg["conf_min"] = 0.1

        # Box is 20×20 → below minimum
        records = _run_one_frame_tracker(
            cfg, tmp_path,
            [[100, 100, 120, 120]], [1], [0.9], [0],
        )
        assert len(records) == 0

    def test_large_box_accepted(self, tmp_path: Path) -> None:
        cfg = dict(_TRACKER_DEFAULTS)
        cfg["output_dir"] = str(tmp_path / "crops")
        cfg["edge_margin_px"] = 0
        cfg["blur_var_min"] = 0.0
        cfg["conf_min"] = 0.1

        records = _run_one_frame_tracker(
            cfg, tmp_path,
            [[100, 100, 200, 200]], [1], [0.9], [0],
        )
        assert len(records) == 1


class TestFilterNearEdge:
    def test_edge_box_rejected(self, tmp_path: Path) -> None:
        cfg = dict(_TRACKER_DEFAULTS)
        cfg["output_dir"] = str(tmp_path / "crops")
        cfg["edge_margin_px"] = 10
        cfg["blur_var_min"] = 0.0
        cfg["conf_min"] = 0.1

        # Box starting at x=2, within the 10px margin
        records = _run_one_frame_tracker(
            cfg, tmp_path,
            [[2, 50, 102, 150]], [1], [0.9], [0],
        )
        assert len(records) == 0


class TestFilterBlurry:
    def test_blurry_crop_rejected(self, tmp_path: Path) -> None:
        cfg = dict(_TRACKER_DEFAULTS)
        cfg["output_dir"] = str(tmp_path / "crops")
        cfg["edge_margin_px"] = 0
        cfg["blur_var_min"] = 500.0  # very high threshold
        cfg["conf_min"] = 0.1

        # Uniform grey frame → Laplacian variance ≈ 0
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        records = _run_one_frame_tracker(
            cfg, tmp_path,
            [[100, 100, 200, 200]], [1], [0.9], [0],
            frame=frame,
        )
        assert len(records) == 0

    def test_sharp_crop_accepted(self, tmp_path: Path) -> None:
        cfg = dict(_TRACKER_DEFAULTS)
        cfg["output_dir"] = str(tmp_path / "crops")
        cfg["edge_margin_px"] = 0
        cfg["blur_var_min"] = 0.0
        cfg["conf_min"] = 0.1

        # Textured frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[::2, ::2] = 255
        records = _run_one_frame_tracker(
            cfg, tmp_path,
            [[100, 100, 200, 200]], [1], [0.9], [0],
            frame=frame,
        )
        assert len(records) == 1


class TestBboxInFrame:
    def test_bbox_within_frame_bounds(self, tmp_path: Path) -> None:
        """All saved bbox_xyxy values must be inside [0, frame_w] × [0, frame_h]."""
        cfg = dict(_TRACKER_DEFAULTS)
        cfg["output_dir"] = str(tmp_path / "crops")
        cfg["edge_margin_px"] = 0
        cfg["blur_var_min"] = 0.0
        cfg["conf_min"] = 0.1

        frame = np.full((480, 640, 3), 180, dtype=np.uint8)
        records = _run_one_frame_tracker(
            cfg, tmp_path,
            [[50, 50, 300, 300]], [1], [0.9], [0],
            frame=frame,
        )
        assert len(records) >= 1
        for r in records:
            x1, y1, x2, y2 = r["bbox_xyxy"]
            assert 0 <= x1 < x2 <= r["frame_w"]
            assert 0 <= y1 < y2 <= r["frame_h"]
