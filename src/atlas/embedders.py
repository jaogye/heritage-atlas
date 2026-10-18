"""Embedding back-ends behind one small interface.

The default is CLIP (OpenCLIP), which puts images and English text in one
shared space. Heritage metadata is mostly Dutch and French, so text can
optionally go through a multilingual student model distilled into the same CLIP
space (`clip-ViT-B-32-multilingual-v1`), letting "De val van Icarus" land next
to the English caption and the image itself.

`HashEmbedder` is a deterministic, dependency-free stand-in used in tests and
for smoke runs; it is not semantically meaningful.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np
from PIL import Image


class Embedder(Protocol):
    name: str
    dim: int

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...

    def embed_images(self, images: list[Image.Image]) -> np.ndarray: ...


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, eps)


class ClipEmbedder:
    """OpenCLIP image + text encoder, optional multilingual text student."""

    def __init__(
        self,
        model: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        multilingual_text: str | None = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
        device: str | None = None,
        batch_size: int = 32,
    ):
        import open_clip  # heavy imports stay optional
        import torch

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model)
        self.batch_size = batch_size
        self.text_student = None
        if multilingual_text:
            # The multilingual student is distilled against ViT-B-32's text tower,
            # so it is only valid together with that image model.
            if model != "ViT-B-32":
                raise ValueError("the multilingual text student is aligned with ViT-B-32 only")
            from sentence_transformers import SentenceTransformer

            self.text_student = SentenceTransformer(multilingual_text, device=self.device)
        self.name = f"openclip:{model}:{pretrained}" + (
            ":multilingual" if multilingual_text else ""
        )
        self.dim = self.model.visual.output_dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if self.text_student is not None:
            vecs = self.text_student.encode(
                texts, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False
            )
            return l2_normalize(vecs)
        out = []
        with self._torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                tokens = self.tokenizer(texts[i : i + self.batch_size]).to(self.device)
                out.append(self.model.encode_text(tokens).float().cpu().numpy())
        return l2_normalize(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        out = []
        with self._torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = [
                    self.preprocess(im.convert("RGB")) for im in images[i : i + self.batch_size]
                ]
                pixels = self._torch.stack(batch).to(self.device)
                out.append(self.model.encode_image(pixels).float().cpu().numpy())
        return l2_normalize(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)


class HashEmbedder:
    """Deterministic feature hashing for text and a fixed random projection for images.

    Similar texts (shared words) and similar images (similar colour layout) get
    similar vectors, which is enough to test the pipeline end to end without
    downloading a model. Image and text vectors are NOT aligned with each other.
    """

    def __init__(self, dim: int = 64, seed: int = 7):
        self.dim = dim
        self.name = f"hash:{dim}"
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal((8 * 8 * 3, dim)).astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
                out[i, h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        return l2_normalize(out)

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), np.float32)
        feats = np.stack(
            [
                np.asarray(im.convert("RGB").resize((8, 8)), np.float32).ravel() / 255.0 - 0.5
                for im in images
            ]
        )
        return l2_normalize(feats @ self._proj)


def get_embedder(name: str) -> Embedder:
    if name == "hash":
        return HashEmbedder()
    if name == "clip":
        return ClipEmbedder(multilingual_text=None)
    if name == "clip-multilingual":
        return ClipEmbedder()
    raise ValueError(f"unknown embedder {name!r} (use clip, clip-multilingual or hash)")
