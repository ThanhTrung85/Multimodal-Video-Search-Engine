"""Tests for tracker sampling-interval logic.

We mock YOLO to avoid needing real weights and instead feed synthetic
detections to verify that the per-track temporal sampling works correctly.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.vision.tracker import VideoTracker, _TRACKER_DEFAULTS


def _make_boxes(
    xyxy: list[list[float]],
    track_ids: list[int] | None,
    confs: list[float],
    cls_ids: list[int],
) -> SimpleNamespace:
    """Create a fake ``boxes`` object mimicking ultralytics Results."""
    return SimpleNamespace(
        xyxy=torch.tensor(xyxy, dtype=torch.float32),
        id=torch.tensor(track_ids, dtype=torch.float32) if track_ids else None,
        conf=torch.tensor(confs, dtype=torch.float32),
        cls=torch.tensor(cls_ids, dtype=torch.float32),
    )


def _make_result(boxes: SimpleNamespace) -> list[SimpleNamespace]:
    return [SimpleNamespace(boxes=boxes)]


class TestSamplingInterval:
    """Verify that crops are taken at the correct frame intervals."""

    @pytest.fixture()
    def cfg(self) -> dict:
        cfg = dict(_TRACKER_DEFAULTS)
        cfg["sample_interval_sec"] = 0.5  # 0.5 s
        cfg["conf_min"] = 0.1
        cfg["min_box_w"] = 1
        cfg["min_box_h"] = 1
        cfg["edge_margin_px"] = 0
        cfg["blur_var_min"] = 0.0  # disable blur filter
        cfg["pad_ratio"] = 0.0
        return cfg

    def test_crop_count_matches_interval(self, cfg: dict, tmp_path: Path) -> None:
        """With fps=10 and interval=0.5s → sample every 5 frames.

        A track present in all 20 frames should produce crops at
        frames 0, 5, 10, 15 → 4 crops.
        """
        fps = 10.0
        n_frames = 20
        cfg["output_dir"] = str(tmp_path / "crops")

        # One detection per frame: track_id=1, class=person
        box = [[100.0, 100.0, 200.0, 200.0]]
        det_boxes = _make_boxes(box, [1], [0.9], [0])
        no_det_boxes = _make_boxes([], None, [], [])

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:200, 100:200] = 180  # non-black region to pass blur

        frame_counter = {"n": 0}

        def fake_read():
            if frame_counter["n"] < n_frames:
                frame_counter["n"] += 1
                return True, frame.copy()
            return False, None

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: fps,  # CAP_PROP_POS_MSEC – unused
            3: 640.0,  # CAP_PROP_FRAME_WIDTH
            4: 480.0,  # CAP_PROP_FRAME_HEIGHT
            5: fps,  # CAP_PROP_FPS
        }.get(prop, 0.0)
        mock_cap.read = fake_read

        mock_model = MagicMock()
        mock_model.names = {0: "person"}
        mock_model.track.return_value = _make_result(det_boxes)

        with (
            patch("src.vision.tracker.YOLO", return_value=mock_model),
            patch("src.vision.tracker.cv2.VideoCapture", return_value=mock_cap),
        ):
            tracker = VideoTracker(cfg)
            records = tracker.process_video("fake_video.mp4", "test_vid")

        expected_frames = [0, 5, 10, 15]
        assert len(records) == len(expected_frames)
        actual_frames = [r["frame_idx"] for r in records]
        assert actual_frames == expected_frames

    def test_multiple_tracks_independent(self, cfg: dict, tmp_path: Path) -> None:
        """Each track_id has its own sampling clock."""
        fps = 10.0
        n_frames = 12
        cfg["output_dir"] = str(tmp_path / "crops")

        # Two tracks present every frame
        two_det = _make_boxes(
            [[50, 50, 150, 150], [200, 200, 300, 300]],
            [1, 2], [0.9, 0.8], [0, 2],
        )

        frame = np.full((480, 640, 3), 180, dtype=np.uint8)
        frame_counter = {"n": 0}

        def fake_read():
            if frame_counter["n"] < n_frames:
                frame_counter["n"] += 1
                return True, frame.copy()
            return False, None

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0, 5: fps}.get(prop, 0.0)
        mock_cap.read = fake_read

        mock_model = MagicMock()
        mock_model.names = {0: "person", 2: "car"}
        mock_model.track.return_value = _make_result(two_det)

        with (
            patch("src.vision.tracker.YOLO", return_value=mock_model),
            patch("src.vision.tracker.cv2.VideoCapture", return_value=mock_cap),
        ):
            tracker = VideoTracker(cfg)
            records = tracker.process_video("fake.mp4", "multi")

        # Each track: frames 0, 5, 10 → 3 crops, total 6
        track1 = [r for r in records if r["track_id"] == 1]
        track2 = [r for r in records if r["track_id"] == 2]
        assert len(track1) == 3
        assert len(track2) == 3

    def test_metadata_jsonl_matches_crops(self, cfg: dict, tmp_path: Path) -> None:
        """metadata.jsonl line count must equal saved crop file count."""
        fps = 10.0
        n_frames = 10
        cfg["output_dir"] = str(tmp_path / "crops")

        det = _make_boxes([[80, 80, 180, 180]], [1], [0.9], [0])
        frame = np.full((480, 640, 3), 180, dtype=np.uint8)
        counter = {"n": 0}

        def fake_read():
            if counter["n"] < n_frames:
                counter["n"] += 1
                return True, frame.copy()
            return False, None

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0, 5: fps}.get(prop, 0.0)
        mock_cap.read = fake_read

        mock_model = MagicMock()
        mock_model.names = {0: "person"}
        mock_model.track.return_value = _make_result(det)

        with (
            patch("src.vision.tracker.YOLO", return_value=mock_model),
            patch("src.vision.tracker.cv2.VideoCapture", return_value=mock_cap),
        ):
            tracker = VideoTracker(cfg)
            tracker.process_video("v.mp4", "meta_test")

        meta_path = tmp_path / "crops" / "meta_test" / "metadata.jsonl"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta_lines = [json.loads(line) for line in f if line.strip()]

        # Count crop files on disk
        crop_dir = tmp_path / "crops" / "meta_test"
        crop_files = list(crop_dir.rglob("*.jpg"))
        assert len(meta_lines) == len(crop_files)

    def test_timestamp_increases_within_track(self, cfg: dict, tmp_path: Path) -> None:
        """timestamp_sec must be monotonically increasing per track."""
        fps = 10.0
        n_frames = 30
        cfg["output_dir"] = str(tmp_path / "crops")

        det = _make_boxes([[80, 80, 180, 180]], [1], [0.9], [0])
        frame = np.full((480, 640, 3), 180, dtype=np.uint8)
        counter = {"n": 0}

        def fake_read():
            if counter["n"] < n_frames:
                counter["n"] += 1
                return True, frame.copy()
            return False, None

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0, 5: fps}.get(prop, 0.0)
        mock_cap.read = fake_read

        mock_model = MagicMock()
        mock_model.names = {0: "person"}
        mock_model.track.return_value = _make_result(det)

        with (
            patch("src.vision.tracker.YOLO", return_value=mock_model),
            patch("src.vision.tracker.cv2.VideoCapture", return_value=mock_cap),
        ):
            tracker = VideoTracker(cfg)
            records = tracker.process_video("v.mp4", "ts_test")

        timestamps = [r["timestamp_sec"] for r in records]
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1], f"timestamp not increasing at index {i}"
