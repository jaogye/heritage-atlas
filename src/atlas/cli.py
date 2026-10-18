"""Command-line entry point.

atlas build --gold ../heritage-lake/data/lake/gold.parquet --embedder clip-multilingual
atlas search "a winter village scene with many small figures"
atlas similar hl_3fd3f33167 --view image
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl
import typer

from .build import build
from .embedders import get_embedder
from .space import Space

app = typer.Typer(add_completion=False, help="Build and query the heritage embedding space.")
OUT = typer.Option(Path("data/atlas"), help="Output directory")


@app.command("build")
def build_cmd(
    gold: Path = typer.Option(Path("data/sample/gold.parquet"), help="heritage-lake gold table"),
    out: Path = OUT,
    embedder: str = typer.Option("clip-multilingual", help="clip | clip-multilingual | hash"),
    image_weight: float = typer.Option(0.5, help="0 = text only, 1 = image only"),
    no_images: bool = typer.Option(False, help="Skip image download (text-only space)"),
    allow_all_images: bool = typer.Option(False, help="Ignore licence check (private use only)"),
    min_cluster_size: int = typer.Option(3),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Embed the gold dataset and run clustering, relationship mining and evaluation."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    summary = build(
        gold,
        out,
        get_embedder(embedder),
        fetch_images=not no_images,
        image_weight=image_weight,
        min_cluster_size=min_cluster_size,
        allow_all_images=allow_all_images,
    )
    typer.echo(json.dumps(summary, indent=2))
    typer.echo(f"open {out / 'explorer.html'} to explore")


def _titles(out: Path) -> dict[str, str]:
    df = pl.read_parquet(out / "items.parquet", columns=["entity_id", "title"])
    return dict(zip(df["entity_id"], df["title"], strict=True))


@app.command()
def search(query: str, k: int = 10, view: str = typer.Option("fused"), out: Path = OUT) -> None:
    """Free-text search across the collection (text against images and captions)."""
    space = Space.load(out / "space.npz")
    name = space.model.split(":")
    kind = (
        "hash" if name[0] == "hash" else "clip-multilingual" if "multilingual" in name else "clip"
    )
    vec = get_embedder(kind).embed_texts([query])[0]
    titles = _titles(out)
    for eid, score in space.search(vec, k, view):
        typer.echo(f"{score:6.3f}  {eid}  {titles.get(eid, '')}")


@app.command()
def similar(
    entity_id: str, k: int = 10, view: str = typer.Option("fused"), out: Path = OUT
) -> None:
    """Nearest neighbours of one object (view: fused | image | text)."""
    space = Space.load(out / "space.npz")
    titles = _titles(out)
    typer.echo(f"neighbours of {titles.get(entity_id, entity_id)} ({view}):")
    for eid, score in space.similar(entity_id, k, view):
        typer.echo(f"{score:6.3f}  {eid}  {titles.get(eid, '')}")


if __name__ == "__main__":
    app()
