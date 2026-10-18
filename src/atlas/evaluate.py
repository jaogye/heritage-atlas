"""How good is the space? Three cheap, label-free or weakly-labelled checks.

* cross-modal recall@k: query with an object's caption, is its own image in the
  top k? Measures whether text and images are really aligned for this domain.
* facet neighbourhood purity: of an object's k nearest neighbours, how many
  share its creator / object type / century? Shows what the space has learned
  to care about, and lets image/text weights be tuned against it.
* silhouette of the clusters (noise excluded).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import silhouette_score

from .space import Space


def cross_modal_recall(space: Space, ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    idx = np.flatnonzero(space.has_image)
    if len(idx) < 2:
        return {}
    sims = space.text[idx] @ space.image[idx].T  # caption i vs image j
    ranks = (sims > np.diag(sims)[:, None]).sum(axis=1)  # 0 = own image ranked first
    return {f"recall@{k}": float((ranks < k).mean()) for k in ks if k <= len(idx)} | {
        "n": int(len(idx)),
        "median_rank": float(np.median(ranks) + 1),
    }


def facet_purity(
    space: Space, meta: list[dict[str, Any]], facet: str, k: int = 5, view: str = "fused"
) -> float | None:
    def value(m):
        v = m.get(facet)
        if facet == "century":
            return None if m.get("year_start") is None else m["year_start"] // 100
        if isinstance(v, list):
            return tuple(sorted(x for x in v if x and x != "anonymous")) or None
        return v

    vals = [value(m) for m in meta]
    usable = space.has_image if view == "image" else np.ones(len(vals), bool)
    labelled = [i for i, v in enumerate(vals) if v is not None and usable[i]]
    if len(labelled) < k + 1:
        return None
    nbrs, _ = space.knn(k, view)
    scores = [np.mean([vals[j] == vals[i] for j in nbrs[i]]) for i in labelled]
    return float(np.mean(scores))


def evaluate(
    space: Space, meta: list[dict[str, Any]], labels: np.ndarray, aligned: bool
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": space.model,
        "n": len(space.ids),
        "with_image": int(space.has_image.sum()),
    }
    if aligned:
        out["cross_modal"] = cross_modal_recall(space)
    out["purity@5"] = {
        view: {
            f: facet_purity(space, meta, f, 5, view) for f in ("creators", "object_type", "century")
        }
        for view in ("fused", "text") + (("image",) if space.has_image.sum() > 5 else ())
    }
    mask = labels >= 0
    if len(set(labels[mask])) >= 2:
        out["silhouette"] = float(
            silhouette_score(space.fused[mask], labels[mask], metric="cosine")
        )
    out["clusters"] = int(len(set(labels) - {-1}))
    out["noise_fraction"] = float((labels == -1).mean()) if len(labels) else 0.0
    return out
