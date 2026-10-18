"""Find structure in the space: clusters, their meaning, and typed relationships.

Relationship types (each row says which evidence produced it):

* ``possible-duplicate`` – two entities that are almost certainly the same object.
  These go back to heritage-lake's review queue: the embedding space catches
  matches that metadata alone could not, e.g. titles in different languages
  with no shared inventory number.
* ``image-text-mismatch`` – an object whose image is far from its own caption
  compared with the rest of the collection. Usually a wrong image attached to a
  record, sometimes a misleading title. Needs an aligned (CLIP-like) model.
* ``visual-echo`` – strong visual similarity between works by different artists,
  decades apart or in different collections: candidate copies, workshop
  versions, shared models or influence. A lead for a researcher, not a claim.
* ``thematic-bridge`` – close neighbours held by different institutions,
  suggesting connections across collections that no catalogue links.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

from .space import Space

STOP = [
    "by",
    "the",
    "of",
    "and",
    "a",
    "an",
    "in",
    "with",
    "on",
    "to",
    "depicts",
    "around",
    "oil",
    "paint",
    "panel",
    "canvas",
    "oak",
    "de",
    "het",
    "van",
    "der",
    "een",
    "met",
    "la",
    "le",
    "les",
    "des",
    "du",
    "et",
    "anonymous",
    "artist",
    "painting",
]


def cluster(space: Space, view: str = "fused", min_cluster_size: int = 3) -> np.ndarray:
    m = space.matrix(view)
    if len(m) < min_cluster_size * 2:
        return np.full(len(m), -1)
    # on unit vectors, euclidean distance is a monotone function of cosine distance
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1, copy=True).fit_predict(m)


def project_2d(space: Space, view: str = "fused", seed: int = 0) -> np.ndarray:
    m = space.matrix(view)
    try:
        import umap  # optional extra

        n_neighbors = max(2, min(15, len(m) - 1))
        return umap.UMAP(n_neighbors=n_neighbors, metric="cosine", random_state=seed).fit_transform(
            m
        )
    except ImportError:
        return PCA(n_components=2, random_state=seed).fit_transform(m)


def label_clusters(labels: np.ndarray, captions: list[str], top_n: int = 4) -> dict[int, list[str]]:
    """Class-based TF-IDF: words frequent in a cluster and rare elsewhere."""
    clusters = sorted(set(labels) - {-1})
    if not clusters:
        return {}
    docs = [
        " ".join(c for c, lab in zip(captions, labels, strict=True) if lab == k) for k in clusters
    ]
    vec = CountVectorizer(stop_words=STOP, token_pattern=r"(?u)\b[^\W\d_]{3,}\b")
    try:
        tf = vec.fit_transform(docs).toarray().astype(float)
    except ValueError:  # empty vocabulary
        return {k: [] for k in clusters}
    tf = tf / np.maximum(tf.sum(axis=1, keepdims=True), 1)
    idf = np.log(1 + tf.shape[0] / np.maximum((tf > 0).sum(axis=0), 1))
    scores = tf * idf
    vocab = np.array(vec.get_feature_names_out())
    return {k: vocab[np.argsort(-scores[i])[:top_n]].tolist() for i, k in enumerate(clusters)}


def describe_clusters(labels: np.ndarray, meta: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out = {}
    for k in sorted(set(labels) - {-1}):
        members = [m for m, lab in zip(meta, labels, strict=True) if lab == k]
        creators = Counter(
            c for m in members for c in (m.get("creators") or []) if c != "anonymous"
        )
        insts = Counter(m.get("institution") for m in members if m.get("institution"))
        centuries = Counter(
            f"{m['year_start'] // 100 + 1}th c." for m in members if m.get("year_start") is not None
        )
        out[int(k)] = {
            "size": len(members),
            "top_creators": creators.most_common(3),
            "institutions": insts.most_common(3),
            "centuries": sorted(centuries.items()),
        }
    return out


def _creators(meta: dict[str, Any]) -> set[str]:
    return {c for c in (meta.get("creators") or []) if c and c != "anonymous"}


def relationships(
    space: Space,
    meta: list[dict[str, Any]],
    aligned: bool,
    k: int = 8,
    dup_threshold: float = 0.95,
    echo_threshold: float = 0.85,
    bridge_threshold: float = 0.80,
    mismatch_z: float = -2.0,
) -> list[dict[str, Any]]:
    rels: list[dict[str, Any]] = []
    ids = space.ids
    seen: set[tuple[str, str, str]] = set()

    def add(kind: str, i: int, j: int | None, score: float, evidence: str) -> None:
        a, b = ids[i], (ids[j] if j is not None else "")
        key = (kind, *sorted((a, b)))
        if key in seen:
            return
        seen.add(key)
        rels.append(
            {
                "type": kind,
                "a": a,
                "b": b or None,
                "score": round(float(score), 4),
                "evidence": evidence,
            }
        )

    nbrs, sims = space.knn(k, "fused")
    img_sim = space.image @ space.image.T
    for i in range(len(ids)):
        for j, s in zip(nbrs[i], sims[i], strict=True):
            both_img = space.has_image[i] and space.has_image[j]
            if s >= dup_threshold and (not both_img or img_sim[i, j] >= dup_threshold):
                add(
                    "possible-duplicate",
                    i,
                    j,
                    s,
                    f"fused cosine {s:.3f}"
                    + (f", image cosine {img_sim[i, j]:.3f}" if both_img else ""),
                )
                continue
            mi, mj = meta[i], meta[j]
            if (
                mi.get("institution")
                and mj.get("institution")
                and mi["institution"] != mj["institution"]
                and s >= bridge_threshold
            ):
                add(
                    "thematic-bridge",
                    i,
                    j,
                    s,
                    f"{mi['institution']} ↔ {mj['institution']}, fused cosine {s:.3f}",
                )

    if space.has_image.sum() >= 2:
        inbrs, isims = space.knn(k, "image")
        for i in np.flatnonzero(space.has_image):
            for j, s in zip(inbrs[i], isims[i], strict=True):
                if not space.has_image[j] or s < echo_threshold or s >= dup_threshold:
                    continue
                ci, cj = _creators(meta[i]), _creators(meta[j])
                yi, yj = meta[i].get("year_start"), meta[j].get("year_start")
                gap = abs(yi - yj) if yi is not None and yj is not None else None
                different_hands = bool(ci and cj and not (ci & cj))
                if different_hands or (gap is not None and gap >= 30):
                    why = []
                    if different_hands:
                        why.append(f"{', '.join(sorted(ci))} vs {', '.join(sorted(cj))}")
                    if gap is not None:
                        why.append(f"{gap} years apart")
                    add("visual-echo", i, j, s, f"image cosine {s:.3f}; " + "; ".join(why))

    if aligned and space.has_image.sum() >= 5:
        idx = np.flatnonzero(space.has_image)
        agreement = np.einsum("ij,ij->i", space.image[idx], space.text[idx])
        z = (agreement - agreement.mean()) / (agreement.std() + 1e-9)
        for i, a, zi in zip(idx, agreement, z, strict=True):
            if zi <= mismatch_z:
                add(
                    "image-text-mismatch",
                    int(i),
                    None,
                    a,
                    f"image–caption cosine {a:.3f} (z = {zi:.1f} vs collection)",
                )

    order = {
        "possible-duplicate": 0,
        "image-text-mismatch": 1,
        "visual-echo": 2,
        "thematic-bridge": 3,
    }
    rels.sort(key=lambda r: (order[r["type"]], -r["score"]))
    return rels
