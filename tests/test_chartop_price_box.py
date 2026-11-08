"""The squares price protocol's box payload (`scripts/chartop_price_box_payload.py`) and its
runner (`scripts/chartop_price_box_run.sh`): one commit, byte for byte; the runner LF and
executable; nothing the box must not carry (`gates_squares.REVISED`, section 11.18)."""
from __future__ import annotations

import os
import subprocess
import sys
import tarfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import chartop_price_box_payload as PB  # noqa: E402


def _git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", ROOT, "-c", "core.autocrlf=false", *args], check=True,
                          capture_output=True).stdout


def _has_history() -> bool:
    try:
        _git("cat-file", "-e", f"HEAD:{PB.RUNNER}")
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


needs_history = pytest.mark.skipif(not _has_history(), reason="needs the git history with the runner committed")


@needs_history
def test_the_payload_is_one_commit_byte_for_byte(tmp_path):
    r = PB.build("HEAD", str(tmp_path))
    head = _git("rev-parse", "HEAD").decode().strip()
    assert r["commit"] == head and os.path.basename(r["path"]) == f"price-squares-{head[:7]}.tar.gz"
    with tarfile.open(r["path"]) as t:
        names = t.getnames()
        run = t.getmember(f"{PB.TOP}/run.sh")
        runner = t.extractfile(run).read()
        identity = t.extractfile(f"{PB.TOP}/IDENTITY").read().decode().strip()
        shipped = t.extractfile(f"{PB.TOP}/scripts/chartop_price_squares.py").read()
    assert run.mode & 0o111 and b"\r" not in runner and runner.startswith(b"#!/usr/bin/env bash\n")
    assert identity == head
    assert shipped == _git("show", "HEAD:scripts/chartop_price_squares.py")
    assert all(n == PB.TOP or n.startswith(PB.TOP + "/") for n in names)
    assert all(f"{PB.TOP}/{rec}" in names for rec in PB.RECORDS)
    assert not [n for n in names if n.endswith(PB.REFUSED_SUFFIXES) or os.path.basename(n) in PB.REFUSED_NAMES]


@needs_history
def test_the_payload_refuses_what_the_box_must_not_carry(tmp_path, monkeypatch):
    monkeypatch.setattr(PB, "REFUSED_SUFFIXES", PB.REFUSED_SUFFIXES + (".py",))     # the failing control
    with pytest.raises(SystemExit, match="must not"):
        PB.build("HEAD", str(tmp_path))


def test_the_runner_installs_the_laptops_versions_and_stops_itself():
    text = open(os.path.join(ROOT, *PB.RUNNER.split("/")), encoding="utf-8").read()
    for pin in ("torch==2.7.1", "numpy==1.26.4", "scipy==1.13.1", "download.pytorch.org/whl/cpu",
                "threadpoolctl==3.5.0"):
        assert pin in text
    constraints = open(os.path.join(ROOT, "constraints.txt"), encoding="utf-8").read().splitlines()
    for name in ("pydantic", "pydantic_core", "annotated-types", "typing_extensions"):   # atlas.workbench's
        pinned = next(ln.strip() for ln in constraints if ln.strip().startswith(name + "=="))
        assert pinned in text, pinned
    assert "vastai stop instance" in text and 'sleep "$DEADLINE_S"' in text and "ALL.done" in text
    assert "chartop_price_squares.py --dry-run > dryrun.log" in text
