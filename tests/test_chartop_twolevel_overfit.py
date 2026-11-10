"""Phase 3's overfit probe (`scripts/chartop_twolevel_overfit.py`), on two post-hoc validation
pieces at n = 32 for a few steps, from zero heads: it starts at TL0 (rho 0 exactly, the
relative objective 1 per term), it learns on its own pieces, its ledger and controls are
recorded, the dead-network stop fires on a network that cannot move, and on the CPU a rerun
is bit for bit the same."""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import chartop_twolevel_overfit as P  # noqa: E402

ARGS = ["--start", "scratch", "--n", "32", "--pieces", "2", "--n-c", "8", "--threads", "1"]


def _run(tmp_path, *extra):
    out = str(tmp_path)
    assert P.main(ARGS + ["--out", out, *extra]) == 0
    return json.load(open(os.path.join(out, "probe-scratch.json"), encoding="utf-8"))


def test_the_probe_starts_at_TL0_learns_and_records_its_ledger(tmp_path):
    rec = _run(tmp_path, "--steps", "60", "--every", "30", "--lr", "3e-3", "--repeat-check", "2", "--no-dead-stop")
    t0, last = rec["trajectory"][0], rec["trajectory"][-1]
    assert t0["step"] == 0 and t0["rho"] == [0.0, 0.0] and t0["rho_p"] == [0.0, 0.0]
    assert all(k is None for k in t0["kappa_kill"])           # no network term at TL0: no ratio
    # on its own two pieces it climbs (0.06 by step 50 in the development run), and the same
    # weights read with the other piece's inputs do worse
    assert last["step"] == 60 and last["rho_summary"]["median"] > 0.02
    assert last["control_shuffled_inputs"]["rho_median"] < last["rho_summary"]["median"]
    assert set(last["rho_groups"]) == {"hats", "j1-3", "j4-8"}
    assert all(0.0 <= k <= 1.0 for k in last["kappa_kill"] if k is not None)
    assert "rho_median" in last["control_shuffled_inputs"]
    assert rec["repeat_check"]["bit_for_bit"] is True                          # the CPU is deterministic
    assert rec["cost"]["coarse_level_s_per_piece"] is not None and rec["device"] == "cpu"
    assert rec["test_split_read"] is False and rec["easy"]["modes"] == [False, False]


def test_the_dead_network_stop_ends_a_probe_that_cannot_move(tmp_path):
    rec = _run(tmp_path, "--steps", "10", "--every", "2", "--lr", "0")
    assert rec["stopped"]["step"] == 2 and "dead network" in rec["stopped"]["why"]
    assert rec["trajectory"][-1]["rho_summary"]["median"] == 0.0


def test_the_payload_ships_the_probe_and_everything_it_imports(monkeypatch, tmp_path):
    import re
    import tarfile
    import chartop_box_payload as payload
    shipped = set(payload.SCRIPTS + payload.PROBE_SCRIPTS)
    for script in payload.PROBE_SCRIPTS:
        src = open(os.path.join(ROOT, "scripts", script), encoding="utf-8").read()
        for mod in re.findall(r"^\s*(?:import|from) (chartop_\w+)", src, re.M):
            assert mod + ".py" in shipped, f"{script} imports {mod}, which the payload does not ship"
    out = tmp_path / "payload.tar.gz"
    monkeypatch.setattr(payload, "OUT", str(out))
    monkeypatch.setattr(sys, "argv", ["chartop_box_payload.py", "--with-probe"])
    assert payload.main() == 0
    with tarfile.open(out) as tar:
        names = tar.getnames()
        run = tar.getmember("chartop/run.sh")
        assert run.mode == 0o755 and b"\r" not in tar.extractfile(run).read()
    for must in ("chartop/scripts/chartop_twolevel_overfit.py", "chartop/atlas/chartop/twolevel.py",
                 "chartop/atlas/chartop/coarse.py", "chartop/scripts/chartop_forms_diagnose.py"):
        assert must in names, must
    assert not any(n.endswith((".pt", ".npz", ".npy")) for n in names)
    # without the option the payload is as before
    monkeypatch.setattr(sys, "argv", ["chartop_box_payload.py"])
    assert payload.main() == 0
    with tarfile.open(out) as tar:
        assert "chartop/scripts/chartop_twolevel_overfit.py" not in tar.getnames()


def test_the_runner_has_the_probe_stage():
    text = open(os.path.join(ROOT, "scripts", "chartop_box_run.sh"), encoding="utf-8").read()
    start = text.index("    probe)")
    stage = text[start:text.index(";;", start)]
    assert "chartop_twolevel_overfit.py" in stage and "--device cuda" in stage
    assert 'probe-$start.done' in stage and "probe.done" in stage and "--repeat-check" in stage
