"""The expert store: one catalog, a card for every trained model, failures included
([[chart-operator-stage-0b-and-1-plan]] section 8).

    out/experts/
      catalog.json                 one row per expert: id, family, class, verdict, card path
      <family>/<expert-id>/
        card.json                  what it is, how it was built, how it did
        weights.pt                 a new model's weights (git-ignored until publishing is decided)

**A card** holds:
- ``what``: the PDE, its guarantee class, the pieces and ports, the envelope it is valid for;
- ``built``: the code commit and module, the architecture and training settings, the data
  (seed base and count, or a frozen dataset's sha256) and the seed;
- ``did``: its registration, its evaluation record, every verdict, and its measured price;
- ``verdict``: ``failed`` (a registered accuracy gate not met), ``met`` (every registered
  gate met), or ``comparison`` (an arm that answered a comparison, not a gate);
- ``weights``: where the file is and its recorded sha256.  Whether the file here has that
  sha256 is a fact about one machine, so it is never trusted from a stored file: `present`
  checks it whenever it is asked (a missing file, or another sha256, is not present).  A
  card written on a machine notes what that machine had (``present_when_written``, dated),
  and nothing reads that note as true elsewhere.

Models trained before the store existed (Stage 0's eleven arms, the forms and prior runs,
TL-s0 and TL-s1) are backfilled from their evaluation records
(`scripts/expert_store_backfill.py`); their weights stay where training left them, under
``out/chartop/runs/`` (git-ignored), and the card points there.  A card with a ``studio``
section is one the studio offers (`atlas/studio/experts.py`), when its weights are present.
"""

from __future__ import annotations

import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE = os.path.join(ROOT, "out", "experts")
CARD_VERSION = 1
VERDICTS = ("met", "failed", "comparison")

_sha: dict = {}


def sha256_file(path: str) -> str:
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _sha:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        _sha[key] = h.hexdigest()
    return _sha[key]


def weights_status(path_rel: str | None, recorded: str | None, root: str = ROOT) -> dict:
    """Whether the weights a card names are here: the file, its sha256 against the one
    recorded, and why not when not."""
    out = {"path": path_rel, "sha256": recorded, "present": False}
    if not path_rel:
        out["why"] = "no weights file is recorded"
        return out
    path = os.path.join(root, path_rel)
    if not os.path.isfile(path):
        out["why"] = "the file is not on this machine"
        return out
    got = sha256_file(path)
    if recorded is None:
        out["why"] = "no sha256 is recorded to check it against"
        out["sha256_here"] = got
        return out
    if got != recorded:
        out["why"] = "the file here has another sha256"
        out["sha256_here"] = got
        return out
    out["present"] = True
    return out


def present(row_or_card: dict, root: str = ROOT) -> dict:
    """The weights' status now, on this machine: a catalog row or a card."""
    w = row_or_card.get("weights") or {}
    if not w and "card" in row_or_card and isinstance(row_or_card["card"], str):
        w = load_card(row_or_card, root).get("weights") or {}
    return weights_status(w.get("path"), w.get("sha256"), root)


def card_path(family: str, eid: str, store: str = STORE) -> str:
    return os.path.join(store, family, eid, "card.json")


def check_card(card: dict) -> list[str]:
    """What a card lacks (empty when it is complete)."""
    out = []
    for k in ("card", "id", "family", "class", "what", "built", "did", "verdict", "weights"):
        if k not in card:
            out.append(f"no {k}")
    if card.get("verdict") not in VERDICTS:
        out.append(f"verdict {card.get('verdict')!r} is not one of {VERDICTS}")
    w = card.get("weights") or {}
    if w.get("path") and not w.get("sha256"):
        out.append("weights named with no sha256")
    return out


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, path)


def write_card(card: dict, store: str = STORE) -> str:
    bad = check_card(card)
    if bad:
        raise ValueError(f"card {card.get('id')!r}: " + "; ".join(bad))
    path = card_path(card["family"], card["id"], store)
    _write_json(path, card)
    return path


def rebuild_catalog(store: str = STORE, root: str = ROOT) -> dict:
    """Scan the cards, write catalog.json: one row per card, with its weights' path and
    sha256 and what this machine had when it wrote the catalog (a dated note only)."""
    import datetime as _dt
    when = _dt.date.today().isoformat()
    rows = []
    for fam in sorted(os.listdir(store)) if os.path.isdir(store) else []:
        fdir = os.path.join(store, fam)
        if not os.path.isdir(fdir):
            continue
        for eid in sorted(os.listdir(fdir)):
            cp = os.path.join(fdir, eid, "card.json")
            if not os.path.isfile(cp):
                continue
            card = json.load(open(cp, encoding="utf-8"))
            w = card.get("weights") or {}
            status = weights_status(w.get("path"), w.get("sha256"), root)
            rows.append({"id": card["id"], "family": card["family"], "class": card["class"],
                         "verdict": card["verdict"], "summary": card.get("did", {}).get("summary"),
                         "card": os.path.relpath(cp, root).replace(os.sep, "/"),
                         "weights": {"path": w.get("path"), "sha256": w.get("sha256")},
                         "present_when_written": {"date": when, "present": status["present"]},
                         "studio": card.get("studio")})
    cat = {"catalog": CARD_VERSION, "experts": rows}
    _write_json(os.path.join(store, "catalog.json"), cat)
    return cat


def catalog(store: str = STORE) -> dict:
    path = os.path.join(store, "catalog.json")
    if not os.path.isfile(path):
        return {"catalog": CARD_VERSION, "experts": []}
    return json.load(open(path, encoding="utf-8"))


def load_card(row: dict, root: str = ROOT) -> dict:
    return json.load(open(os.path.join(root, row["card"]), encoding="utf-8"))


__all__ = ["STORE", "VERDICTS", "catalog", "card_path", "check_card", "load_card", "present",
           "rebuild_catalog", "sha256_file", "weights_status", "write_card"]
