"""gold.parquet -> embeddings -> space -> clusters, relationships, evaluation, explorer.

Outputs in `out_dir`:
  space.npz               text, image and fused vectors (+ model name)
  items.parquet           one row per object: caption, 2-D position, cluster, facets
  clusters.json           cluster keywords and composition
  relationships.csv       typed relationships with evidence
  duplicate_candidates.csv  possible duplicates, formatted for heritage-lake review
  evaluation.json         quality of the space
  explorer.html           self-contained interactive map of the collection
"""

from __future__ import annotations

import csv
import json
import logging
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .analysis import cluster, describe_clusters, label_clusters, project_2d, relationships
from .documents import Item, load_items
from .embedders import Embedder, HashEmbedder
from .evaluate import evaluate
from .images import ImageStore
from .space import Space

log = logging.getLogger(__name__)


def embed_items(
    items: list[Item], embedder: Embedder, images: ImageStore | None, image_weight: float
) -> Space:
    text = embedder.embed_texts([it.caption for it in items])
    image = np.zeros_like(text)
    has_image = np.zeros(len(items), bool)
    if images is not None:
        loaded = [
            (i, images.get(it.entity_id, it.image_url, it.image_allowed))
            for i, it in enumerate(items)
        ]
        loaded = [(i, im) for i, im in loaded if im is not None]
        if loaded:
            vecs = embedder.embed_images([im for _, im in loaded])
            for (i, _), v in zip(loaded, vecs, strict=True):
                image[i], has_image[i] = v, True
    return Space.build(
        [it.entity_id for it in items], text, image, has_image, embedder.name, image_weight
    )


def build(
    gold_path: Path,
    out_dir: Path,
    embedder: Embedder,
    fetch_images: bool = True,
    image_weight: float = 0.5,
    min_cluster_size: int = 3,
    allow_all_images: bool = False,
    image_store: ImageStore | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_items(str(gold_path), allow_all_images)
    store = image_store or (ImageStore(out_dir / "images") if fetch_images else None)
    space = embed_items(items, embedder, store, image_weight)
    space.save(out_dir / "space.npz")

    aligned = not isinstance(embedder, HashEmbedder) and getattr(embedder, "aligned", True)
    meta = [it.meta for it in items]
    labels = cluster(space, min_cluster_size=min_cluster_size)
    xy = project_2d(space)
    keywords = label_clusters(labels, [it.caption for it in items])
    composition = describe_clusters(labels, meta)
    rels = relationships(space, meta, aligned=aligned)
    report = evaluate(space, meta, labels, aligned)
    if store is not None:
        r = store.report
        report["images"] = {
            "loaded": r.loaded,
            "cached": r.cached,
            "missing": r.missing,
            "skipped_rights": r.skipped_rights,
            "failed": len(r.failed),
        }

    rows = []
    for i, it in enumerate(items):
        k = int(labels[i])
        rows.append(
            {
                "entity_id": it.entity_id,
                "title": it.title,
                "caption": it.caption,
                "x": float(xy[i, 0]),
                "y": float(xy[i, 1]),
                "cluster": k,
                "cluster_label": ", ".join(keywords.get(k, [])) if k >= 0 else "unclustered",
                "has_image": bool(space.has_image[i]),
                "image_url": it.image_url,
                "creators": it.meta.get("creators") or [],
                "year_start": it.meta.get("year_start"),
                "institution": it.meta.get("institution"),
                "themes": it.meta.get("themes") or [],
            }
        )
    pl.DataFrame(rows).write_parquet(out_dir / "items.parquet")
    (out_dir / "clusters.json").write_text(
        json.dumps(
            {str(k): {"keywords": keywords.get(k, []), **v} for k, v in composition.items()},
            indent=2,
            ensure_ascii=False,
        ),
        "utf-8",
    )
    _write_csv(out_dir / "relationships.csv", rels, ["type", "a", "b", "score", "evidence"])
    by_id = {it.entity_id: it for it in items}
    dups = [
        {
            "left_entity": r["a"],
            "right_entity": r["b"],
            "score": r["score"],
            "left_records": " ".join(by_id[r["a"]].meta.get("source_records") or []),
            "right_records": " ".join(by_id[r["b"]].meta.get("source_records") or []),
            "reasons": f"embedding: {r['evidence']}",
            "verdict": "",
        }
        for r in rels
        if r["type"] == "possible-duplicate"
    ]
    _write_csv(
        out_dir / "duplicate_candidates.csv",
        dups,
        [
            "left_entity",
            "right_entity",
            "score",
            "left_records",
            "right_records",
            "reasons",
            "verdict",
        ],
    )
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2), "utf-8")
    write_explorer(out_dir / "explorer.html", rows, rels, keywords)

    summary = {
        "items": len(items),
        "with_image": int(space.has_image.sum()),
        "clusters": report["clusters"],
        "relationships": dict(
            sorted(
                {r["type"]: sum(1 for x in rels if x["type"] == r["type"]) for r in rels}.items()
            )
        ),
        "model": embedder.name,
    }
    log.info("atlas summary: %s", summary)
    return summary


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_explorer(path: Path, rows: list[dict], rels: list[dict], keywords: dict) -> None:
    data = json.dumps({"items": rows, "rels": rels}, ensure_ascii=False, default=str)
    template = files("atlas").joinpath("explorer_template.html").read_text("utf-8")
    path.write_text(template.replace("/*DATA*/null", data.replace("</", "<\\/")), "utf-8")
