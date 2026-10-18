"""Fetch, cache and validate images for embedding.

Images are cached by entity id, failures are recorded rather than raised (a
collection always has some dead links), and nothing is downloaded unless
`documents.image_allowed` says the licence permits reuse.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from PIL import Image, UnidentifiedImageError

log = logging.getLogger(__name__)
UA = "heritage-atlas/0.2 (+https://github.com/jaogye/heritage-atlas; research use)"


@dataclass
class ImageReport:
    loaded: int = 0
    cached: int = 0
    skipped_rights: int = 0
    missing: int = 0
    failed: dict[str, str] = field(default_factory=dict)


class ImageStore:
    def __init__(
        self,
        cache_dir: Path,
        max_side: int = 512,
        min_interval: float = 0.5,
        client: httpx.Client | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_side = max_side
        self.min_interval = min_interval
        self.client = client or httpx.Client(
            headers={"User-Agent": UA}, timeout=30, follow_redirects=True
        )
        self.report = ImageReport()
        self._last = 0.0

    def path_for(self, entity_id: str) -> Path:
        return self.cache_dir / f"{entity_id}.jpg"

    def get(self, entity_id: str, url: str | None, allowed: bool) -> Image.Image | None:
        if not url:
            self.report.missing += 1
            return None
        if not allowed:
            self.report.skipped_rights += 1
            return None
        path = self.path_for(entity_id)
        if path.exists():
            self.report.cached += 1
            return Image.open(path).convert("RGB")
        try:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            resp = self.client.get(url)
            self._last = time.monotonic()
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except (httpx.HTTPError, UnidentifiedImageError, OSError) as exc:
            self.report.failed[entity_id] = f"{type(exc).__name__}: {exc}"[:200]
            log.warning("image %s failed: %s", entity_id, exc)
            return None
        img.thumbnail((self.max_side, self.max_side))
        img.save(path, "JPEG", quality=90)
        self.report.loaded += 1
        return img
