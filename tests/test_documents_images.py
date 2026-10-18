import io

import httpx
from PIL import Image

from atlas.documents import caption, image_allowed
from atlas.images import ImageStore


def test_caption_leads_with_visual_content_and_skips_institution():
    row = {
        "title": "The Fall of the Rebel Angels",
        "titles": ["De val van de opstandige engelen", "The Fall of the Rebel Angels"],
        "object_type": "painting",
        "creators": ["Pieter Bruegel the Elder"],
        "year_start": 1562,
        "year_end": 1562,
        "date_approx": False,
        "subjects": ["Archangel Michael", "demon"],
        "materials": ["oil paint", "oak panel"],
        "institution": "Royal Museums of Fine Arts of Belgium",
    }
    c = caption(row)
    assert c.startswith("The Fall of the Rebel Angels (De val van de opstandige engelen)")
    assert "by Pieter Bruegel the Elder 1562." in c
    assert "Depicts Archangel Michael, demon." in c
    assert "Royal Museums" not in c


def test_caption_handles_anonymous_and_ranges():
    c = caption(
        {
            "title": "Portrait",
            "creators": ["anonymous"],
            "year_start": 1500,
            "year_end": 1599,
            "date_approx": True,
        }
    )
    assert c == "Portrait by an anonymous artist 1500–1599."


def test_image_rights():
    assert image_allowed({"rights": "http://creativecommons.org/publicdomain/mark/1.0/"})
    assert image_allowed(
        {"rights": "http://rightsstatements.org/vocab/InC/1.0/", "image_rights_cleared": True}
    )
    assert not image_allowed({"rights": "http://rightsstatements.org/vocab/InC/1.0/"})
    assert image_allowed({"rights": None}, allow_all=True)


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (1200, 800), (10, 20, 30)).save(buf, "PNG")
    return buf.getvalue()


def test_image_store_downloads_resizes_caches_and_records_failures(tmp_path):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if "dead" in str(request.url):
            return httpx.Response(404)
        if "html" in str(request.url):
            return httpx.Response(200, content=b"<html>not an image</html>")
        return httpx.Response(200, content=_png())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    store = ImageStore(tmp_path, max_side=256, min_interval=0, client=client)
    img = store.get("e1", "https://img.test/ok.png", allowed=True)
    assert max(img.size) == 256
    assert store.get("e1", "https://img.test/ok.png", allowed=True) is not None  # from cache
    assert store.get("e2", "https://img.test/dead.png", allowed=True) is None
    assert store.get("e3", "https://img.test/page.html", allowed=True) is None
    assert store.get("e4", "https://img.test/x.png", allowed=False) is None
    assert store.get("e5", None, allowed=True) is None
    r = store.report
    assert (r.loaded, r.cached, r.skipped_rights, r.missing) == (1, 1, 1, 1)
    assert set(r.failed) == {"e2", "e3"}
    assert len(calls) == 3  # cached and disallowed images were never requested
