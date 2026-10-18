"""Build the text view of each object from the integrated (gold) dataset.

CLIP text towers see at most 77 tokens, so the caption leads with what
describes the image (title, subjects, medium) and leaves bookkeeping out.
Institution and city are deliberately excluded from the embedded text: they
would pull objects together by where they hang rather than by what they show,
and they stay available as filters and as facets in the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

OPEN_RIGHTS = ("publicdomain", "creativecommons.org/licenses/by/", "zero/1.0", "/cc0")


@dataclass
class Item:
    entity_id: str
    title: str
    caption: str
    image_url: str | None
    image_allowed: bool
    meta: dict[str, Any]


def _year_phrase(start: int | None, end: int | None, approx: bool | None) -> str:
    if start is None:
        return ""
    if start == end:
        return f"{'around ' if approx else ''}{start}"
    return f"{start}–{end}"


def caption(row: dict[str, Any]) -> str:
    parts = [row.get("title") or "Untitled"]
    alt = [t for t in (row.get("titles") or []) if t != row.get("title")]
    if alt:
        parts.append(f"({'; '.join(alt[:2])})")
    if row.get("object_type"):
        parts.append(f"— {row['object_type']}")
    creators = [c for c in (row.get("creators") or []) if c and c != "anonymous"]
    parts.append(f"by {', '.join(creators)}" if creators else "by an anonymous artist")
    when = _year_phrase(row.get("year_start"), row.get("year_end"), row.get("date_approx"))
    if when:
        parts.append(when)
    text = " ".join(parts) + "."
    if row.get("subjects"):
        text += " Depicts " + ", ".join(row["subjects"][:6]) + "."
    if row.get("materials"):
        text += " " + ", ".join(row["materials"][:3]).capitalize() + "."
    return text


def image_allowed(row: dict[str, Any], allow_all: bool = False) -> bool:
    """Only fetch images we may reuse: open licence or cleared by the research team."""
    if allow_all:
        return True
    if row.get("image_rights_cleared"):
        return True
    rights = (row.get("rights") or "").lower()
    return any(t in rights for t in OPEN_RIGHTS) or "commons.wikimedia.org" in rights


def load_items(gold_path: str, allow_all_images: bool = False) -> list[Item]:
    rows = pl.read_parquet(gold_path).to_dicts()
    items = []
    for r in rows:
        meta = {
            k: r.get(k)
            for k in (
                "creators",
                "year_start",
                "year_end",
                "institution",
                "city",
                "object_type",
                "sources",
                "themes",
                "subjects",
                "wikidata_qid",
                "inhouse_id",
                "source_records",
            )
        }
        items.append(
            Item(
                entity_id=r["entity_id"],
                title=r.get("title") or "Untitled",
                caption=caption(r),
                image_url=r.get("image_url"),
                image_allowed=image_allowed(r, allow_all_images),
                meta=meta,
            )
        )
    return items
