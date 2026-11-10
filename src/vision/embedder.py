"""CLIP-based visual and text embedding with square-padding preprocessing.

Provides batch image embedding with fp16 inference on GPU,
L2-normalized output, and text embedding with truncation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.utils.config import load_section

logger = logging.getLogger(__name__)

_EMBEDDER_DEFAULTS: dict[str, Any] = {
    "model_name": "openai/clip-vit-base-patch32",
    "batch_size": 64,
    "output_dir": "data/embeddings",
}


def load_embedder_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load embedder section from vision.yaml, with built-in defaults."""
    return load_section(config_path, "embedder", _EMBEDDER_DEFAULTS)


# ── Preprocessing ────────────────────────────────────────────────────────────

def pad_to_square(img: np.ndarray) -> np.ndarray:
    """Pad an image to a square canvas with neutral gray (128), centered.

    This avoids the CLIP processor's default center-crop, which can
    cut off the head/feet of a person crop that is tall and narrow.

    Parameters
    ----------
    img : np.ndarray
        HWC uint8 image (BGR or RGB).

    Returns
    -------
    np.ndarray
        Square image with the same dtype and number of channels.
    """
    h, w = img.shape[:2]
    size = max(h, w)
    if h == w:
        return img

    if img.ndim == 3:
        canvas = np.full((size, size, img.shape[2]), 128, dtype=img.dtype)
    else:
        canvas = np.full((size, size), 128, dtype=img.dtype)

    y_off = (size - h) // 2
    x_off = (size - w) // 2
    canvas[y_off : y_off + h, x_off : x_off + w] = img
    return canvas


# ── Embedder class ───────────────────────────────────────────────────────────

class CLIPEmbedder:
    """Encode images and text into L2-normalised CLIP vectors.

    Parameters
    ----------
    model_name : str or None
        HuggingFace model identifier.  If *None*, reads from *config_path*.
    config_path : str, Path, or None
        Path to ``vision.yaml`` (used when *model_name* is None).
    """

    def __init__(
        self,
        model_name: str | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        cfg = load_embedder_config(config_path)
        self.model_name: str = model_name or cfg["model_name"]
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        self.dim: int = self.model.config.projection_dim

        logger.info(
            "CLIPEmbedder ready – model=%s  dim=%d  device=%s",
            self.model_name, self.dim, self.device,
        )

    # ── Image embedding ──────────────────────────────────────────────────

    def embed_images_with_indices(
        self,
        paths: list[str | Path],
        batch_size: int = 64,
    ) -> tuple[np.ndarray, list[int]]:
        """Embed image paths batch-by-batch and return ``(embeddings, valid_indices)``.

        Invalid images are skipped and logged. ``valid_indices`` are the original
        positions in ``paths`` whose images were successfully embedded.
        """
        valid_indices: list[int] = []
        batch_embeds: list[np.ndarray] = []

        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            batch_images: list[Image.Image] = []
            batch_valid_indices: list[int] = []

            for offset, p in enumerate(batch_paths):
                original_idx = start + offset
                try:
                    img_bgr = cv2.imread(str(p))
                    if img_bgr is None:
                        raise ValueError("cv2.imread returned None")
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    img_sq = pad_to_square(img_rgb)
                    batch_images.append(Image.fromarray(img_sq))
                    batch_valid_indices.append(original_idx)
                except Exception:
                    logger.warning("Failed to load image, skipping: %s", p)

            if not batch_images:
                continue

            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)

            with torch.inference_mode():
                if self.device == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        out = self.model.get_image_features(**inputs)
                else:
                    out = self.model.get_image_features(**inputs)

            feats = out.pooler_output if hasattr(out, "pooler_output") else out
            feats = feats.float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            batch_embeds.append(feats.cpu().numpy().astype(np.float32))
            valid_indices.extend(batch_valid_indices)

        if not batch_embeds:
            return np.empty((0, self.dim), dtype=np.float32), []

        return np.concatenate(batch_embeds, axis=0).astype(np.float32), valid_indices

    def embed_images(
        self,
        paths: list[str | Path],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Embed image paths and return only the embedding matrix.

        Invalid images are skipped. Use ``embed_images_with_indices`` when the caller
        needs to map embeddings back to their original positions in ``paths``.
        """
        embeddings, _ = self.embed_images_with_indices(paths, batch_size=batch_size)
        return embeddings

    def embed_image(self, path: str | Path) -> list[float]:
        """Embed a single image.  Convenience wrapper around :meth:`embed_images`."""
        arr = self.embed_images([path])
        if arr.shape[0] == 0:
            raise ValueError(f"Failed to embed image: {path}")
        return arr[0].tolist()

    # ── Text embedding ───────────────────────────────────────────────────

    def embed_text(self, text: str | list[str]) -> np.ndarray:
        """Embed one or more text queries.

        Parameters
        ----------
        text : str or list[str]
            A single query string or a list of queries.

        Returns
        -------
        np.ndarray
            ``(dim,)`` for a single string, ``(N, dim)`` for a list.
            Float32, L2-normalised.
        """
        single = isinstance(text, str)
        if single:
            text = [text]

        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)

        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    out = self.model.get_text_features(**inputs)
            else:
                out = self.model.get_text_features(**inputs)

        # transformers ≥5.x returns BaseModelOutputWithPooling
        feats = out.pooler_output if hasattr(out, "pooler_output") else out

        feats = feats.float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        result = feats.cpu().numpy().astype(np.float32)

        return result[0] if single else result
