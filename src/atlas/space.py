"""The multimodal space: per-modality vectors, fusion, and nearest-neighbour search.

Each object can have an image vector, a text vector, or both. CLIP aligns the
two modalities, so a fused vector is a weighted mean in the shared space, and
an object with only one modality still gets a usable position. We keep the
per-modality vectors as well: comparing them is how we find objects whose
image and metadata disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embedders import l2_normalize


@dataclass
class Space:
    ids: list[str]
    text: np.ndarray  # (n, d), rows normalised
    image: np.ndarray  # (n, d), zero rows where no image
    has_image: np.ndarray  # (n,) bool
    fused: np.ndarray  # (n, d)
    model: str
    image_weight: float

    @classmethod
    def build(
        cls,
        ids: list[str],
        text: np.ndarray,
        image: np.ndarray,
        has_image: np.ndarray,
        model: str,
        image_weight: float = 0.5,
    ) -> Space:
        if not 0.0 <= image_weight <= 1.0:
            raise ValueError("image_weight must be in [0, 1]")
        text, image = l2_normalize(text), l2_normalize(image)
        image = image * has_image[:, None]
        w = np.where(has_image, image_weight, 0.0)[:, None]
        fused = l2_normalize((1 - w) * text + w * image)
        return cls(ids, text, image, has_image.astype(bool), fused, model, image_weight)

    # ------------------------------------------------------------- queries
    def _top(
        self, scores: np.ndarray, k: int, exclude: int | None = None
    ) -> list[tuple[str, float]]:
        if exclude is not None:
            scores = scores.copy()
            scores[exclude] = -np.inf
        k = min(k, len(scores) - (exclude is not None))
        idx = np.argpartition(-scores, k - 1)[:k] if k > 0 else np.array([], int)
        idx = idx[np.argsort(-scores[idx])]
        return [(self.ids[i], float(scores[i])) for i in idx]

    def similar(self, entity_id: str, k: int = 10, view: str = "fused") -> list[tuple[str, float]]:
        i = self.ids.index(entity_id)
        m = self.matrix(view)
        return self._top(m @ m[i], k, exclude=i)

    def search(
        self, query_vec: np.ndarray, k: int = 10, view: str = "fused"
    ) -> list[tuple[str, float]]:
        """Cross-modal search: a text query vector against image, text or fused vectors."""
        m = self.matrix(view)
        scores = m @ l2_normalize(query_vec.reshape(1, -1))[0]
        if view == "image":
            scores = np.where(self.has_image, scores, -np.inf)
        return self._top(scores, k)

    def matrix(self, view: str) -> np.ndarray:
        return {"fused": self.fused, "text": self.text, "image": self.image}[view]

    def knn(self, k: int = 10, view: str = "fused") -> tuple[np.ndarray, np.ndarray]:
        """Exact kNN in blocks (fine up to a few hundred thousand objects; swap in FAISS above)."""
        m = self.matrix(view)
        n = len(m)
        k = min(k, n - 1)
        nbrs = np.zeros((n, k), int)
        sims = np.zeros((n, k), np.float32)
        for s in range(0, n, 2048):
            block = m[s : s + 2048] @ m.T
            for r in range(block.shape[0]):
                block[r, s + r] = -np.inf
            idx = np.argpartition(-block, k - 1, axis=1)[:, :k]
            part = np.take_along_axis(block, idx, axis=1)
            order = np.argsort(-part, axis=1)
            nbrs[s : s + len(block)] = np.take_along_axis(idx, order, axis=1)
            sims[s : s + len(block)] = np.take_along_axis(part, order, axis=1)
        return nbrs, sims

    # ------------------------------------------------------------- persist
    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            ids=np.array(self.ids),
            text=self.text,
            image=self.image,
            has_image=self.has_image,
            fused=self.fused,
            model=self.model,
            image_weight=self.image_weight,
        )

    @classmethod
    def load(cls, path: Path) -> Space:
        z = np.load(path, allow_pickle=False)
        return cls(
            list(z["ids"]),
            z["text"],
            z["image"],
            z["has_image"],
            z["fused"],
            str(z["model"]),
            float(z["image_weight"]),
        )
