"""A synthetic collection with known structure, and an aligned toy embedder.

Every object has a dominant colour (red / green / blue) and a stripe pattern.
Captions name both ("red", "stripes3"); images show both. ToyAlignedEmbedder
maps the colour and the pattern to the same axes for text and for images, so
it behaves like a tiny CLIP: we know what the right answers are.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest
from PIL import Image

from atlas.embedders import l2_normalize
from atlas.images import ImageStore

COLOURS = {"red": (200, 40, 40), "green": (40, 160, 60), "blue": (40, 60, 200)}
N_PATTERNS = 8
DIM = 3 + N_PATTERNS


def make_image(colour: str, *patterns: int) -> Image.Image:
    img = Image.new("RGB", (64, 64), COLOURS[colour])
    px = img.load()
    for x in range(64):
        if (x // 4) % N_PATTERNS in patterns:  # stripe positions encode the pattern(s)
            for y in range(64):
                px[x, y] = (255, 255, 255)
    return img


class ToyAlignedEmbedder:
    name = "toy-aligned"
    dim = DIM
    aligned = True

    def embed_texts(self, texts):
        out = np.zeros((len(texts), DIM), np.float32)
        for i, t in enumerate(texts):
            for c, colour in enumerate(COLOURS):
                if colour in t.lower():
                    out[i, c] += 1.0
            for p in re.findall(r"stripes(\d)", t):
                out[i, 3 + int(p)] += 0.6
        return l2_normalize(out)

    def embed_images(self, images):
        out = np.zeros((len(images), DIM), np.float32)
        for i, im in enumerate(images):
            a = np.asarray(im.convert("RGB"), np.float32)
            body = a[(a < 250).any(axis=2)]
            out[i, int(np.argmax(body.mean(axis=0)))] = 1.0
            white_cols = np.flatnonzero((a == 255).all(axis=2).all(axis=0))
            for p in {(int(x) // 4) % N_PATTERNS for x in white_cols}:
                out[i, 3 + p] = 0.6
        return l2_normalize(out)


class DictImageStore(ImageStore):
    """ImageStore that serves in-memory images instead of downloading."""

    def __init__(self, tmp_path, images):
        super().__init__(tmp_path / "img")
        self.images = images

    def get(self, entity_id, url, allowed):
        if not url:
            self.report.missing += 1
            return None
        if not allowed:
            self.report.skipped_rights += 1
            return None
        self.report.loaded += 1
        return self.images[entity_id]


def _row(eid, title, colour, pattern, creator, year, inst, image=True, caption_colour=None):
    cc = caption_colour or colour  # the colour the *metadata* claims
    return {
        "entity_id": eid,
        "title": title,
        "titles": [title],
        "creators": [creator],
        "year_start": year,
        "year_end": year,
        "date_approx": False,
        "object_type": "painting",
        "materials": [],
        "subjects": [cc, f"stripes{pattern}"],
        "institution": inst,
        "city": None,
        "image_url": f"https://img.test/{eid}.jpg" if image else None,
        "rights": "http://creativecommons.org/publicdomain/mark/1.0/",
        "sources": ["wikidata"],
        "source_records": [f"wikidata:{eid}"],
        "themes": [],
        "inhouse_id": None,
        "wikidata_qid": None,
        "image_rights_cleared": None,
    }


@pytest.fixture
def collection(tmp_path):
    rows, images = [], {}
    insts = ["Museum A", "Museum B"]
    k = 0
    for colour, creator, base_year in [
        ("red", "Painter Red", 1500),
        ("green", "Painter Green", 1700),
        ("blue", "Painter Blue", 1900),
    ]:
        for j in range(6):
            eid = f"e{k:02d}"
            p = j % N_PATTERNS
            rows.append(
                _row(
                    eid,
                    f"{colour.title()} study {j}",
                    colour,
                    p,
                    creator,
                    base_year + j,
                    insts[j % 2],
                )
            )
            images[eid] = make_image(colour, p)
            k += 1
    # the same red study twice under another id: a duplicate the metadata missed
    rows.append(_row("dup", "Rode studie 0", "red", 0, "Painter Red", 1500, "Museum A"))
    images["dup"] = make_image("red", 0)
    # a record whose image is blue but whose caption says green: wrong image attached
    rows.append(
        _row(
            "bad",
            "Green study wrong image",
            "blue",
            7,
            "Painter Green",
            1710,
            "Museum A",
            caption_colour="green",
        )
    )
    images["bad"] = make_image("blue", 7)
    # a later artist's free variation on red study 1: a visual echo, not a duplicate
    rows.append(_row("echo", "Homage in red", "red", 5, "Later Copyist", 1850, "Museum B"))
    images["echo"] = make_image("red", 1, 5)
    # an object with no image at all
    rows.append(
        _row("noimg", "Lost blue study", "blue", 2, "Painter Blue", 1901, "Museum B", image=False)
    )
    path = tmp_path / "gold.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path, DictImageStore(tmp_path, images)
