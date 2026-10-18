import numpy as np
import pytest

from atlas.space import Space


def toy_space(image_weight=0.5):
    text = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    image = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]], np.float32)
    has = np.array([True, True, True, False])
    return Space.build(["a", "b", "c", "d"], text, image, has, "toy", image_weight)


def test_fusion_falls_back_to_text_without_image():
    s = toy_space()
    np.testing.assert_allclose(s.fused[3], [0, 0, 1], atol=1e-6)
    assert np.allclose(np.linalg.norm(s.fused, axis=1), 1)


def test_image_weight_extremes():
    np.testing.assert_allclose(toy_space(0.0).fused[0], [1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(toy_space(1.0).fused[0], [0, 1, 0], atol=1e-6)
    with pytest.raises(ValueError):
        toy_space(1.5)


def test_similar_excludes_self_and_ranks():
    s = toy_space(0.0)
    res = s.similar("a", k=2, view="text")
    assert res[0][0] == "b" and all(eid != "a" for eid, _ in res)


def test_image_search_ignores_objects_without_image():
    s = toy_space()
    res = s.search(np.array([0, 0, 1.0]), k=4, view="image")
    assert "d" not in [eid for eid, sc in res if np.isfinite(sc)]


def test_knn_matches_brute_force():
    rng = np.random.default_rng(0)
    v = rng.standard_normal((50, 8))
    s = Space.build([str(i) for i in range(50)], v, np.zeros_like(v), np.zeros(50, bool), "r")
    nbrs, sims = s.knn(5, "text")
    full = s.text @ s.text.T
    np.fill_diagonal(full, -np.inf)
    np.testing.assert_array_equal(nbrs[:, 0], full.argmax(axis=1))
    assert (np.diff(sims, axis=1) <= 1e-6).all()


def test_save_load_roundtrip(tmp_path):
    s = toy_space()
    s.save(tmp_path / "s.npz")
    t = Space.load(tmp_path / "s.npz")
    assert t.ids == s.ids and t.model == "toy" and t.image_weight == 0.5
    np.testing.assert_allclose(t.fused, s.fused)
