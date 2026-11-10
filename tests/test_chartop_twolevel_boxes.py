"""The rented boxes' side of the registration after the Stage 0 rework: the training plan's check
against `gates_twolevel` (`scripts/chartop_train.py`), the price payload built from a commit
(`scripts/chartop_price_twolevel_payload.py`), and the runners' stages.  No box is touched."""
from __future__ import annotations

import copy
import os
import subprocess
import sys
import tarfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import chartop_train as CT  # noqa: E402
from atlas.chartop import gates_twolevel as GT  # noqa: E402


def _plan(tmp_path):
    w = tmp_path / "S" / "best.pt"
    w.parent.mkdir(parents=True, exist_ok=True)
    w.write_bytes(b"stand-in for S's weights")
    arms = []
    for seed in GT.SEEDS:
        d = {k: v for k, v in GT.arm_config().items() if k != "m"}
        d.update(name=f"TL-s{seed}", registered="TL", seed=seed, init_from=str(w),
                 batch={str(k): v for k, v in {32: 32, 64: 32, 128: 8}.items()})
        arms.append(d)
    return {"registration": "gates_twolevel", "arms": arms}


def _ok(path):
    return GT.INIT_SHA256


def test_the_registered_plan_passes(tmp_path):
    CT.check_twolevel_plan(_plan(tmp_path), sha256=_ok)
    arm = CT.arm_from(_plan(tmp_path)["arms"][0])
    assert (arm.coarse, arm.corners, arm.lr, arm.steps, arm.head_init) == (16, 8, 3e-4, 37_500, "loaded")


@pytest.mark.parametrize("change", ["lr", "corners", "steps", "seed", "drop a seed", "a third run", "weights",
                                   "no weights", "registered", "limit"])
def test_a_plan_that_is_not_the_registration_is_refused(tmp_path, change):
    plan = _plan(tmp_path)
    a = plan["arms"][0]
    sha = _ok
    if change == "lr":
        a["lr"] = 1e-3
    elif change == "corners":
        a["corners"] = None
    elif change == "steps":
        a["steps"] = 150_000
    elif change == "seed":
        a["seed"] = 5
    elif change == "drop a seed":
        plan["arms"].pop()
    elif change == "a third run":
        plan["arms"].append(copy.deepcopy(a))
    elif change == "weights":
        sha = lambda path: "0" * 64              # noqa: E731
    elif change == "no weights":
        a["init_from"] = None
    elif change == "registered":
        a["registered"] = "S"
    elif change == "limit":
        a["limit"] = 100
    with pytest.raises(SystemExit):
        CT.check_twolevel_plan(plan, sha256=sha)


def _head_has(rel: str) -> bool:
    return subprocess.run(["git", "-C", ROOT, "cat-file", "-e", f"HEAD:{rel}"], capture_output=True).returncode == 0


def test_the_price_payload_is_one_commit_and_nothing_else(tmp_path):
    if not _head_has("scripts/chartop_price_twolevel_payload.py"):
        pytest.skip("HEAD does not hold the payload builder yet")
    import chartop_price_twolevel_payload as P
    r = P.build("HEAD", str(tmp_path))
    with tarfile.open(r["path"]) as tar:
        names = tar.getnames()
        ident = tar.extractfile("chartop/IDENTITY").read().decode().strip()
        run = tar.getmember("chartop/run.sh")
        body = tar.extractfile(run).read()
    assert ident == r["commit"] and len(ident) == 40
    assert run.mode == 0o755 and b"\r\n" not in body and b"chartop_price_twolevel.py --dry-run" in body
    assert not any(n.endswith((".pt", ".npz", ".npy")) for n in names)
    assert "chartop/atlas/chartop/coarse_patches.py" in names and "chartop/atlas/workbench/__init__.py" in names


def test_the_written_plan_is_the_registration(tmp_path):
    """``out/chartop/plan-twolevel.json`` passes `check_twolevel_plan` once its weights path
    names S's weights (here a stand-in whose hash is taken as S's)."""
    import json
    plan = json.load(open(os.path.join(ROOT, "out", "chartop", "plan-twolevel.json"), encoding="utf-8"))
    assert plan["registration"] == "gates_twolevel" and plan["data"] == {"train": 50000, "val": 600}
    assert all(a["init_from"] == "weights/S/best.pt" for a in plan["arms"])
    w = tmp_path / "best.pt"
    w.write_bytes(b"S")
    for a in plan["arms"]:
        a["init_from"] = str(w)
    CT.check_twolevel_plan(plan, sha256=_ok)


def test_the_budget_projects_tl_from_its_own_price():
    import json
    import chartop_budget as B
    plan = json.load(open(os.path.join(ROOT, "out", "chartop", "plan-twolevel.json"), encoding="utf-8"))
    rows = [{"arm": "TL", "n": n, "batch": b, "step_s": s, "forward_s": f}
            for n, b, s, f in ((32, 32, 0.10, 0.20), (64, 32, 0.20, 0.40), (128, 8, 0.30, 0.30))]
    price = {"twolevel": {"rows": rows, "projection": {"step_mix": {"32": 0.3, "64": 0.4, "128": 0.3}}},
             "generator": {"examples_per_s": 50.0}, "gpu": "test"}
    rec = B.project_twolevel(plan, price, 0.2, set(), 0.5)
    mean = 0.3 * 0.10 + 0.4 * 0.20 + 0.3 * 0.30
    assert all(abs(r["mean_step_s"] - mean) < 1e-12 and r["validations"] == 15 for r in rec["runs"])
    data_h = B.LABELLED_COST * 50_600 / 50.0 / 3600.0
    assert abs(rec["data_hours_per_box"] - data_h) < 1e-12
    hours = sum(r["hours"] for r in rec["runs"]) + data_h + B.FIXED_HOURS_PER_BOX
    assert abs(rec["registration"]["usd"] - (hours * 0.2 + 0.5)) < 1e-12
    bad = json.loads(json.dumps(plan))
    bad["arms"][0]["registered"] = "S"
    with pytest.raises(SystemExit):
        B.project_twolevel(bad, price, 0.2, set(), 0.0)


def test_the_runners_carry_the_registrations_stages():
    price = open(os.path.join(ROOT, "scripts", "chartop_price_twolevel_box_run.sh"), encoding="utf-8").read()
    for s in ("smoke)", "dry-run)", "GT.INIT_SHA256", "export WEIGHTS", "vastai stop instance", "threadpoolctl==3.5.0"):
        assert s in price, s
    box = open(os.path.join(ROOT, "scripts", "chartop_box_run.sh"), encoding="utf-8").read()
    assert 'DATA_CORNERS' in box and "$CF" in box and "vastai stop instance" in box
    assert "DATA_LIGHT" in box and "$LL" in box and "chartop_price.py --twolevel" in box
