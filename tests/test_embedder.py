"""Tests for CLIPEmbedder: shape, normalisation, batch consistency, long text.

These tests load the real CLIP model to verify end-to-end correctness.
Mark them with ``pytest -m slow`` if you want to skip on CI without GPU.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.vision.embedder import CLIPEmbedder, pad_to_square

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def embedder() -> CLIPEmbedder:
    """Load CLIPEmbedder once for the whole module (expensive)."""
    return CLIPEmbedder(model_name="openai/clip-vit-base-patch32")


@pytest.fixture()
def sample_images(tmp_path: Path) -> list[Path]:
    """Create a few synthetic test images."""
    paths: list[Path] = []
    for i in range(5):
        img = np.random.randint(0, 255, (64 + i * 20, 48 + i * 10, 3), dtype=np.uint8)
        p = tmp_path / f"img_{i}.jpg"
        Image.fromarray(img).save(p)
        paths.append(p)
    return paths


# ── Tests ────────────────────────────────────────────────────────────────────

class TestEmbedImages:
    def test_output_shape(self, embedder: CLIPEmbedder, sample_images: list[Path]) -> None:
        result = embedder.embed_images(sample_images)
        assert result.shape == (len(sample_images), embedder.dim)
        assert result.dtype == np.float32

    def test_l2_norm_approx_one(self, embedder: CLIPEmbedder, sample_images: list[Path]) -> None:
        result = embedder.embed_images(sample_images)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_batch_vs_single_close(self, embedder: CLIPEmbedder, sample_images: list[Path]) -> None:
        """Batch embedding and individual embedding should be very close."""
        batch_result = embedder.embed_images(sample_images, batch_size=2)

        for i, p in enumerate(sample_images):
            single_result = np.array(embedder.embed_image(p))
            diff = np.abs(batch_result[i] - single_result).max()
            assert diff < 1e-2, f"Image {i}: max diff = {diff}"

    def test_bad_image_skipped(self, embedder: CLIPEmbedder, tmp_path: Path) -> None:
        """Non-image file should be skipped, not crash."""
        bad = tmp_path / "bad.jpg"
        bad.write_text("not an image")
        good = tmp_path / "good.jpg"
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(good)

        result = embedder.embed_images([bad, good])
        assert result.shape == (1, embedder.dim)

    def test_empty_input(self, embedder: CLIPEmbedder) -> None:
        result = embedder.embed_images([])
        assert result.shape == (0, embedder.dim)


class TestEmbedText:
    def test_single_string_shape(self, embedder: CLIPEmbedder) -> None:
        result = embedder.embed_text("a photo of a person")
        assert result.shape == (embedder.dim,)
        assert result.dtype == np.float32

    def test_single_string_l2_norm(self, embedder: CLIPEmbedder) -> None:
        result = embedder.embed_text("a red car")
        norm = float(np.linalg.norm(result))
        assert abs(norm - 1.0) < 1e-5

    def test_list_of_strings_shape(self, embedder: CLIPEmbedder) -> None:
        queries = ["a dog", "a cat", "a bird"]
        result = embedder.embed_text(queries)
        assert result.shape == (3, embedder.dim)

    def test_long_text_no_error(self, embedder: CLIPEmbedder) -> None:
        """Text longer than 77 tokens should be truncated, not raise."""
        long_text = " ".join(["word"] * 200)
        result = embedder.embed_text(long_text)
        assert result.shape == (embedder.dim,)
        norm = float(np.linalg.norm(result))
        assert abs(norm - 1.0) < 1e-5


class TestDimAttribute:
    def test_dim_is_512(self, embedder: CLIPEmbedder) -> None:
        assert embedder.dim == 512

    def test_model_name_attribute(self, embedder: CLIPEmbedder) -> None:
        assert embedder.model_name == "openai/clip-vit-base-patch32"
