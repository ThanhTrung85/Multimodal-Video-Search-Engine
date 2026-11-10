"""Tests for tracker sampling-interval logic.

We mock YOLO to avoid needing real weights and instead feed synthetic
detections to verify that the per-track temporal sampling works correctly.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from src.utils.config import load_section, resolve_crop_path
from src.vision.tracker import (
    VideoTracker,
    _TRACKER_DEFAULTS,
    compute_blur_score_normalized,
)


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


class TestConfigHelpers:
    def test_load_section_uses_defaults_when_missing_file(self) -> None:
        cfg = load_section("missing.yaml", "tracker", _TRACKER_DEFAULTS)
        assert cfg["output_dir"] == _TRACKER_DEFAULTS["output_dir"]

    def test_resolve_crop_path_joins_relative_to_root(self) -> None:
        root = Path("data/crops")
        relative = "vid1/3/f000123.jpg"
        assert resolve_crop_path(root, relative) == root / relative

    def test_resolve_crop_path_preserves_absolute_path(self) -> None:
        root = Path("data/crops")
        absolute = Path("/tmp/f000123.jpg")
        assert resolve_crop_path(root, absolute) == absolute


class TestEmbedVideoFast:
    def test_embed_video_skips_bad_middle_crop_and_keeps_order(self, tmp_path: Path) -> None:
        class FakeEmbedder:
            model_name = "fake"
            dim = 2

            @staticmethod
            def embed_images_with_indices(paths: list[str], batch_size: int = 64):
                embeddings = []
                valid_idx = []
                for idx, path in enumerate(paths):
                    if not Path(path).exists() or "bad" in Path(path).parts:
                        continue
                    embeddings.append(np.array([idx, idx + 1], dtype=np.float32))
                    valid_idx.append(idx)
                if not embeddings:
                    return np.empty((0, 2), dtype=np.float32), []
                return np.vstack(embeddings), valid_idx

        crops_root = tmp_path / "crops"
        video_dir = crops_root / "vid_one"
        video_dir.mkdir(parents=True)
        for rel in ["vid_one/1/f000000.jpg", "vid_one/1/f000001.jpg", "vid_one/1/f000002.jpg"]:
            path = crops_root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"img")

        records = [
            {"video_id": "vid_one", "track_id": 1, "frame_idx": 0, "crop_path": "vid_one/1/f000000.jpg"},
            {"video_id": "vid_one", "track_id": 1, "frame_idx": 1, "crop_path": "vid_one/bad/f000001.jpg"},
            {"video_id": "vid_one", "track_id": 1, "frame_idx": 2, "crop_path": "vid_one/1/f000002.jpg"},
        ]
        meta_path = video_dir / "metadata.jsonl"
        with open(meta_path, "w") as f:
            for row in records:
                f.write(json.dumps(row) + "\n")

        out_dir = tmp_path / "embeddings"
        fake = FakeEmbedder()
        from src.vision.embed_crops import embed_video

        embed_video("vid_one", fake, crops_dir=str(crops_root), output_dir=str(out_dir), ingest_db=False)

        npy_path = out_dir / "vid_one.npy"
        meta_out = out_dir / "vid_one_meta.jsonl"
        assert npy_path.exists()
        assert meta_out.exists()
        assert np.load(npy_path).shape == (2, 2)
        with open(meta_out) as f:
            saved = [json.loads(line) for line in f]
        assert [r["frame_idx"] for r in saved] == [0, 2]

    def test_embed_video_all_failures_skip_output(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        class FakeEmbedder:
            model_name = "fake"
            dim = 2

            @staticmethod
            def embed_images_with_indices(paths: list[str], batch_size: int = 64):
                return np.empty((0, 2), dtype=np.float32), []

        crops_root = tmp_path / "crops"
        video_dir = crops_root / "vid_bad"
        video_dir.mkdir(parents=True)
        records = [{"video_id": "vid_bad", "track_id": 1, "frame_idx": 0, "crop_path": "vid_bad/1/f000000.jpg"}]
        with open(video_dir / "metadata.jsonl", "w") as f:
            f.write(json.dumps(records[0]) + "\n")

        out_dir = tmp_path / "embeddings"
        from src.vision.embed_crops import embed_video

        with caplog.at_level("ERROR"):
            embed_video("vid_bad", FakeEmbedder(), crops_dir=str(crops_root), output_dir=str(out_dir), ingest_db=False)

        assert not (out_dir / "vid_bad.npy").exists()
        assert not (out_dir / "vid_bad_meta.jsonl").exists()
        assert "No valid crop embeddings" in caplog.text

    def test_two_videos_do_not_share_state(self, tmp_path: Path) -> None:
        class FakeEmbedder:
            model_name = "fake"
            dim = 2
            def __init__(self, fail_on: set[int] | None = None):
                self.fail_on = fail_on or set()

            def embed_images_with_indices(self, paths: list[str], batch_size: int = 64):
                embeddings = []
                valid_idx = []
                for idx, path in enumerate(paths):
                    if idx in self.fail_on:
                        continue
                    embeddings.append(np.array([idx, idx + 1], dtype=np.float32))
                    valid_idx.append(idx)
                return np.vstack(embeddings), valid_idx

        crops_root = tmp_path / "crops"
        from src.vision.embed_crops import embed_video
        for video_id, fail_on in [("vid_a", set()), ("vid_b", {0})]:
            video_dir = crops_root / video_id
            video_dir.mkdir(parents=True)
            records = [
                {"video_id": video_id, "track_id": 1, "frame_idx": 0, "crop_path": f"{video_id}/1/f000000.jpg"},
                {"video_id": video_id, "track_id": 1, "frame_idx": 1, "crop_path": f"{video_id}/1/f000001.jpg"},
            ]
            for row in records:
                path = crops_root / row["crop_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"img")
            with open(video_dir / "metadata.jsonl", "w") as f:
                for row in records:
                    f.write(json.dumps(row) + "\n")

        embed_video("vid_a", FakeEmbedder(), crops_dir=str(crops_root), output_dir=str(tmp_path / "embeddings_a"), ingest_db=False)
        embed_video("vid_b", FakeEmbedder(fail_on={0}), crops_dir=str(crops_root), output_dir=str(tmp_path / "embeddings_b"), ingest_db=False)

        assert np.load(tmp_path / "embeddings_a" / "vid_a.npy").shape == (2, 2)
        assert np.load(tmp_path / "embeddings_b" / "vid_b.npy").shape == (1, 2)
        assert np.load(tmp_path / "embeddings_b" / "vid_b.npy")[0].tolist() == [1, 2]


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

    def test_stride_uses_true_frame_ids_in_metadata(self, cfg: dict, tmp_path: Path) -> None:
        """When vid_stride > 1, metadata frame_idx should reflect the original frame number."""
        fps = 10.0
        n_frames = 10
        cfg["output_dir"] = str(tmp_path / "crops")
        cfg["vid_stride"] = 3
        cfg["sample_interval_sec"] = 0.1

        det = _make_boxes([[80, 80, 180, 180]], [1], [0.9], [0])
        frame = np.full((480, 640, 3), 180, dtype=np.uint8)
        counter = {"n": 0}

        def fake_read():
            if counter["n"] < n_frames:
                counter["n"] += 1
                return True, frame.copy()
            return False, None

        read_calls = {"n": 0}

        def fake_read_stride():
            if read_calls["n"] < 4:
                read_calls["n"] += 1
                return True, frame.copy()
            return False, None

        grab_counter = {"n": 0}

        def fake_grab():
            if grab_counter["n"] < 6:
                grab_counter["n"] += 1
                return True
            return False

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0, 5: fps}.get(prop, 0.0)
        mock_cap.read = fake_read_stride
        mock_cap.grab.side_effect = fake_grab

        mock_model = MagicMock()
        mock_model.names = {0: "person"}
        mock_model.track.return_value = _make_result(det)

        with (
            patch("src.vision.tracker.YOLO", return_value=mock_model),
            patch("src.vision.tracker.cv2.VideoCapture", return_value=mock_cap),
        ):
            tracker = VideoTracker(cfg)
            records = tracker.process_video("v.mp4", "stride_test")

        assert [r["frame_idx"] for r in records] == [0, 3, 6, 9]

    def test_stats_are_consistent(self, cfg: dict, tmp_path: Path) -> None:
        """Attempt counts should equal accepted plus all rejection buckets."""
        cfg["output_dir"] = str(tmp_path / "crops")
        cfg["edge_margin_px"] = 5
        cfg["conf_min"] = 0.1
        cfg["blur_var_min"] = 0.0
        cfg["min_box_w"] = 30
        cfg["min_box_h"] = 30
        frame = np.full((480, 640, 3), 180, dtype=np.uint8)

        det = _make_boxes(
            [[100, 100, 200, 200], [2, 100, 25, 130], [10, 10, 60, 60]],
            [1, 2, 3],
            [0.9, 0.9, 0.9],
            [0, 0, 0],
        )
        counter = {"n": 0}

        def fake_read():
            if counter["n"] < 1:
                counter["n"] += 1
                return True, frame.copy()
            return False, None

        mock_cap = MagicMock(); mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0, 5: 10.0}.get(prop, 0.0)
        mock_cap.read = fake_read

        mock_model = MagicMock(); mock_model.names = {0: "person"}; mock_model.track.return_value = _make_result(det)

        with (
            patch("src.vision.tracker.YOLO", return_value=mock_model),
            patch("src.vision.tracker.cv2.VideoCapture", return_value=mock_cap),
        ):
            tracker = VideoTracker(cfg)
            tracker.process_video("v.mp4", "stats_test")

        stats_path = tmp_path / "crops" / "stats_test" / "stats.json"
        assert stats_path.exists()
        stats = json.loads(stats_path.read_text())
        totals = stats["accepted"] + stats["rejected_too_small"] + stats["rejected_near_edge"] + stats["rejected_too_blurry"]
        assert stats["attempts"] == totals
