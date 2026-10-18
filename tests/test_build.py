"""End-to-end build on a synthetic collection where the right answers are known."""

import csv
import json
from pathlib import Path

import polars as pl
import pytest

from atlas.build import build
from atlas.embedders import HashEmbedder
from atlas.space import Space

from .conftest import ToyAlignedEmbedder

SAMPLE_GOLD = Path(__file__).resolve().parents[1] / "data" / "sample" / "gold.parquet"


@pytest.fixture
def built(collection, tmp_path):
    gold, store = collection
    out = tmp_path / "atlas"
    summary = build(gold, out, ToyAlignedEmbedder(), image_store=store, min_cluster_size=3)
    rels = list(csv.DictReader((out / "relationships.csv").open(encoding="utf-8")))
    return out, summary, rels


def _pairs(rels, kind):
    return [{r["a"], r["b"]} for r in rels if r["type"] == kind]


def test_summary_and_outputs(built):
    out, summary, _ = built
    assert summary["items"] == 22 and summary["with_image"] == 21
    for f in [
        "space.npz",
        "items.parquet",
        "clusters.json",
        "relationships.csv",
        "duplicate_candidates.csv",
        "evaluation.json",
        "explorer.html",
    ]:
        assert (out / f).exists(), f


def test_colour_groups_become_clusters(built):
    out, _, _ = built
    items = pl.read_parquet(out / "items.parquet")
    by_title = dict(zip(items["title"], items["cluster"], strict=True))
    reds = {by_title[f"Red study {j}"] for j in range(6)}
    blues = {by_title[f"Blue study {j}"] for j in range(6)}
    assert len(reds) == 1 and len(blues) == 1 and reds != blues and -1 not in reds
    labels = json.loads((out / "clusters.json").read_text())
    assert any("red" in v["keywords"] for v in labels.values())


def test_finds_the_duplicate_the_metadata_missed(built):
    out, _, rels = built
    assert {"dup", "e00"} in _pairs(rels, "possible-duplicate")
    dups = list(csv.DictReader((out / "duplicate_candidates.csv").open(encoding="utf-8")))
    row = next(d for d in dups if {d["left_entity"], d["right_entity"]} == {"dup", "e00"})
    assert "wikidata:" in row["left_records"] and row["verdict"] == ""


def test_flags_the_wrong_image(built):
    _, _, rels = built
    flagged = {r["a"] for r in rels if r["type"] == "image-text-mismatch"}
    assert flagged == {"bad"}


def test_finds_visual_echo_across_artists_and_centuries(built):
    _, _, rels = built
    echoes = [r for r in rels if r["type"] == "visual-echo" and "echo" in (r["a"], r["b"])]
    assert any({"echo", "e01"} == {r["a"], r["b"]} for r in echoes)
    assert all("Later Copyist" in r["evidence"] for r in echoes)


def test_cross_modal_retrieval_and_evaluation(built):
    out, _, _ = built
    ev = json.loads((out / "evaluation.json").read_text())
    assert ev["cross_modal"]["recall@5"] >= 0.5
    assert ev["purity@5"]["fused"]["creators"] > 0.6
    space = Space.load(out / "space.npz")
    hits = space.search(ToyAlignedEmbedder().embed_texts(["green"])[0], k=6, view="image")
    titles = dict(
        zip(
            *pl.read_parquet(out / "items.parquet")
            .select("entity_id", "title")
            .to_dict(as_series=False)
            .values(),
            strict=True,
        )
    )
    assert all(titles[e].startswith("Green study ") and e != "bad" for e, _ in hits[:5])


def test_explorer_embeds_data_safely(built):
    out, _, _ = built
    html = (out / "explorer.html").read_text(encoding="utf-8")
    assert "/*DATA*/null" not in html and "Homage in red" in html


def test_builds_on_committed_lake_sample_without_models(tmp_path):
    summary = build(
        SAMPLE_GOLD, tmp_path / "a", HashEmbedder(), fetch_images=False, min_cluster_size=2
    )
    assert summary["items"] == pl.read_parquet(SAMPLE_GOLD).height
    assert "image-text-mismatch" not in summary["relationships"]  # needs an aligned model
