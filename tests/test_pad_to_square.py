"""Tests for pad_to_square image preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from src.vision.embedder import pad_to_square


class TestPadToSquare:
    """Verify square-padding preserves content and aspect ratio."""

    def test_tall_image_becomes_square(self) -> None:
        """A 60×180 (w×h) image should become 180×180."""
        img = np.random.randint(0, 255, (180, 60, 3), dtype=np.uint8)  # HWC
        result = pad_to_square(img)
        assert result.shape == (180, 180, 3)

    def test_wide_image_becomes_square(self) -> None:
        """A 200×80 (w×h) image should become 200×200."""
        img = np.random.randint(0, 255, (80, 200, 3), dtype=np.uint8)
        result = pad_to_square(img)
        assert result.shape == (200, 200, 3)

    def test_already_square_unchanged(self) -> None:
        """A 100×100 image should pass through untouched."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = pad_to_square(img)
        assert result.shape == (100, 100, 3)
        np.testing.assert_array_equal(result, img)

    def test_original_pixels_preserved(self) -> None:
        """Original pixels must be present in the centre of the output."""
        h, w = 180, 60
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        result = pad_to_square(img)

        size = max(h, w)  # 180
        y_off = (size - h) // 2  # 0
        x_off = (size - w) // 2  # 60
        np.testing.assert_array_equal(result[y_off : y_off + h, x_off : x_off + w], img)

    def test_padding_is_gray_128(self) -> None:
        """Padding region should be filled with value 128."""
        img = np.zeros((180, 60, 3), dtype=np.uint8)  # all-black content
        result = pad_to_square(img)
        # Left padding column
        assert result[0, 0, 0] == 128
        # Right padding column
        assert result[0, -1, 0] == 128

    def test_grayscale_image(self) -> None:
        """Works with single-channel images too."""
        img = np.random.randint(0, 255, (50, 100), dtype=np.uint8)
        result = pad_to_square(img)
        assert result.shape == (100, 100)
        assert result.ndim == 2

    def test_aspect_ratio_preserved(self) -> None:
        """The object's width/height ratio must stay the same
        (i.e., padding does not rescale)."""
        h, w = 180, 60
        img = np.ones((h, w, 3), dtype=np.uint8) * 200  # known content
        result = pad_to_square(img)

        size = max(h, w)
        y_off = (size - h) // 2
        x_off = (size - w) // 2
        extracted = result[y_off : y_off + h, x_off : x_off + w]
        assert extracted.shape == (h, w, 3)
        np.testing.assert_array_equal(extracted, img)
