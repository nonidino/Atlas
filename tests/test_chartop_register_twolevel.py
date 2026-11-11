"""The registration script after the Stage 0 rework (`scripts/chartop_register_twolevel.py`): the
test split's content hash, the pin's check (one byte added to a pinned copy fails it), and the
refusals before the dry run.  Nothing here draws a registered seed: the hash is tested on seeds
no registration draws."""
from __future__ import annotations

import os
import shutil
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: F401
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import chartop_register_twolevel as R  # noqa: E402
from atlas.chartop import gates_twolevel as GT  # noqa: E402

SEEDS = (9_000_100, 9_000_101, 9_000_102)


def test_the_test_split_hash_is_the_generators_content_in_seed_order():
    a = R.test_split_sha256(SEEDS)
    assert a == R.test_split_sha256(SEEDS)                               # deterministic
    assert a["count"] == 3 and a["seeds"] == [SEEDS[0], SEEDS[-1]] and sum(a["by_n"].values()) == 3
    assert R.test_split_sha256(SEEDS[::-1])["sha256"] != a["sha256"]       # order matters
    assert R.test_split_sha256(SEEDS[:2])["sha256"] != a["sha256"]
    n, d = R.example_digest(SEEDS[0])
    assert n in (32, 64, 128) and len(d) == 64


def _pinned_copy(tmp_path, rels):
    root = tmp_path / "root"
    for rel in rels:
        dst = root.joinpath(*rel.split("/"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(os.path.join(ROOT, *rel.split("/")), dst)
    reg = tmp_path / "registered.txt"
    reg.write_text("\n".join(f"label  {rel}  sha256 {R.file_hash(str(root.joinpath(*rel.split('/'))))}"
                             for rel in rels) + "\n", encoding="ascii")
    return root, reg


def test_one_byte_added_to_a_pinned_copy_fails_the_check(tmp_path):
    rels = [R.PINNED["gate module"], R.PINNED["the coarse level"], R.ALREADY_PINNED["S's form"]]
    root, reg = _pinned_copy(tmp_path, rels)
    assert set(R.check_pin(str(reg), str(root), rels)) == set(rels)
    target = root.joinpath(*rels[1].split("/"))
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(SystemExit):
        R.check_pin(str(reg), str(root), rels)
    with pytest.raises(SystemExit):                                         # a file it does not pin
        R.check_pin(str(reg), str(root), rels + [R.PINNED["the expert"]])
    with pytest.raises(SystemExit):
        R.check_pin(str(tmp_path / "missing.txt"), str(root), rels)


def test_the_line_endings_do_not_move_the_hash(tmp_path):
    p = tmp_path / "a.py"
    p.write_bytes(b"x = 1\r\ny = 2\r\n")
    q = tmp_path / "b.py"
    q.write_bytes(b"x = 1\ny = 2\n")
    assert R.file_hash(str(p)) == R.file_hash(str(q))


def test_the_scope_exists_and_agrees_with_the_earlier_pins():
    for rel in [*R.PINNED.values(), *R.ALREADY_PINNED.values(), *R.RECORDED_BESIDE.values()]:
        assert os.path.exists(os.path.join(ROOT, *rel.split("/"))), rel
    why = R.refusals_before_pin()
    assert not any("earlier registrations pinned" in w for w in why), why


def test_the_passing_dry_run_is_the_one_recorded():
    """The dry run the pin records passed on the pricing CPU, at the registered settings, on the
    code the price runs execute now; with it the scope may be pinned."""
    import json
    run = json.load(open(os.path.join(ROOT, *R.DRY_RUN.split("/")), encoding="utf-8"))
    assert run["passes"] is True and run["on_price_cpu"] and run["n"] == 64 and run["reps"] == 5
    assert len(run["runs"]) == 6 and run["aa"]["holds"] and set(run["tally"].values()) == {6}
    assert R.refusals_before_pin() == []


def test_the_pin_refuses_before_the_dry_run_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "DRY_RUN", None)
    with pytest.raises(SystemExit):
        R.pin(path=str(tmp_path / "registered-twolevel.txt"))
    assert not (tmp_path / "registered-twolevel.txt").exists()
    existing = tmp_path / "exists.txt"
    existing.write_text("a registration", encoding="ascii")
    with pytest.raises(SystemExit):
        R.write_pin(str(existing))
    assert existing.read_text(encoding="ascii") == "a registration"


def test_the_registration_is_pinned():
    """The pin holds: every file it records has the recorded sha256 (line endings normalised).
    Editing any of them fails this test; that is the point."""
    assert "accepted by the owner on 2026-11-10" in GT.REGISTERED and "hash-pinned" in GT.REGISTERED
    got = R.check_pin()
    assert set(got) == set(R.every_path()) and len(got) == 31
    text = open(R.REGISTRATION, encoding="ascii").read()
    assert R.DRY_RUN in text and "735dbc76dde1fd81" in text and GT.INIT_SHA256 in text
