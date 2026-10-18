# Project conventions for coding assistants

## What this repo is
Stage 3 of the heritage pipeline. Input: heritage-lake's `gold.parquet`. Output: an
embedding space plus clusters, relationships, evaluation and an HTML explorer.

## Rules
- Tests never download models or images. Use `ToyAlignedEmbedder` and `DictImageStore`
  from `tests/conftest.py`; `HashEmbedder` for smoke runs on the real sample.
- Every relationship type must have a planted example in the synthetic collection
  (`tests/conftest.py::collection`) and a test asserting it is found, and ideally one
  asserting a near-miss is not.
- Relationship rows must carry human-readable evidence. A researcher reads them.
- `image_allowed` is the only gate for downloading images. Do not bypass it; the
  `--allow-all-images` flag exists for private experiments and stays off by default.
- Checks that compare images with captions are only valid for aligned models; keep the
  `aligned` guard.
- `explorer_template.html` must stay self-contained (no CDN scripts) and escape all data.

## Commands
- `pytest -q`
- `ruff check . && ruff format --check .`
- `atlas build --embedder hash --no-images` – fast smoke run on the committed sample

## Before finishing a task
Run the tests and the smoke build, and report changes in `evaluation.json` metrics if a
change could affect the space.
