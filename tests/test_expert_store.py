"""The expert store (`atlas/chartop/expert_store.py`; the coolant plan's section 8): a card
for every trained model, failures included, each saying whether its weights are here; the
studio reads the catalog for the networks it offers."""

from __future__ import annotations

import json
import os

import pytest

from atlas.chartop import expert_store as es

ROOT = es.ROOT
BACKFILLED = {"B", "Bp", "C1", "C2-free", "C2-sup", "C3", "C4", "C5-0.3M", "C5-3M", "C6", "C7",
              "S", "Sp-s0", "Sp-s1", "W-s0", "W-s1", "P-s0", "P-s1", "PG-s0", "PG-s1",
              "TL-s0", "TL-s1"}


@pytest.fixture(scope="module")
def cat():
    c = es.catalog()
    if not c["experts"]:
        pytest.skip("no catalog: run scripts/expert_store_backfill.py")
    return c


def test_every_trained_model_has_a_card_failures_included(cat):
    ids = {r["id"] for r in cat["experts"]}
    assert BACKFILLED <= ids
    by = {r["id"]: r for r in cat["experts"]}
    assert by["B"]["verdict"] == "failed" and by["TL-s0"]["verdict"] == "failed"
    assert by["C3"]["verdict"] == "comparison" and by["W-s0"]["verdict"] == "comparison"
    for r in cat["experts"]:
        card = es.load_card(r)
        assert es.check_card(card) == [], r["id"]
        assert card["did"]["evaluation"].startswith("out/chartop/evaluation-")
        assert os.path.isfile(os.path.join(ROOT, card["did"]["evaluation"]))


def test_a_card_quotes_its_record_never_retypes_it(cat):
    for r in cat["experts"]:
        card = es.load_card(r)
        rec = json.load(open(os.path.join(ROOT, card["did"]["evaluation"]), encoding="utf-8"))
        assert card["weights"]["sha256"] == rec["weights"][card["id"]]


def test_a_card_never_claims_a_file_it_cannot_hash(cat, tmp_path):
    """Presence is checked now, on this machine; a stored flag is only a dated note."""
    for r in cat["experts"]:
        card = es.load_card(r)
        w = card["weights"]
        assert set(w) == {"path", "sha256"} and "present" not in w
        path = os.path.join(ROOT, w["path"])
        st = es.present(r)
        if st["present"]:
            assert os.path.isfile(path) and es.sha256_file(path) == w["sha256"]
        else:
            assert not os.path.isfile(path) or es.sha256_file(path) != w["sha256"]
    # a file that differs reads as not present, with the sha256 it has
    f = tmp_path / "w.pt"
    f.write_bytes(b"not the weights")
    st = es.weights_status(str(f.relative_to(tmp_path)), "0" * 64, root=str(tmp_path))
    assert st["present"] is False and st["why"] == "the file here has another sha256"
    assert es.weights_status("missing.pt", "0" * 64, root=str(tmp_path))["present"] is False


def test_an_incomplete_card_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no built"):
        es.write_card({"card": 1, "id": "x", "family": "CO-T", "class": "G0′", "what": {},
                       "did": {}, "verdict": "failed", "weights": {}}, store=str(tmp_path))
    with pytest.raises(ValueError, match="verdict"):
        es.write_card({"card": 1, "id": "x", "family": "CO-T", "class": "G0′", "what": {}, "built": {},
                       "did": {}, "verdict": "great", "weights": {}}, store=str(tmp_path))


def test_the_studio_offers_what_the_catalog_says(cat):
    from atlas.studio import experts as X
    keys = [k for k, *_ in X.learned()]
    assert keys == [r["studio"]["key"] for r in cat["experts"] if r.get("studio")]
    assert set(keys) == {"TL-s0", "TL-s1", "Sp-s0", "Sp-s1"}
    rows = {r["key"]: r for r in X.registry()}
    for k in keys:
        present = es.present(next(r for r in cat["experts"] if r["id"] == k))["present"]
        assert rows[k]["offered"] == (present and rows[k].get("why") is None)
