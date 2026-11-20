"""P6: the chart-operator records (integrated-ui-plan §7, §8).

Done-when: each existing evaluation's gates and prices on screen match its JSON record,
checked by a test that reads both; and a Stage 0b registration file, once it exists,
appears with no change to the page's code.  The JSON is read here by explicit paths per
record, written out again, not through `records.py`'s rules.
"""

from __future__ import annotations

import json
import os
import shutil

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest  # noqa: E402

from atlas.studio import protocol, records as R  # noqa: E402
from atlas.studio.session import Session  # noqa: E402
from atlas.workbench.spec import example_case  # noqa: E402

CH = R.CHARTOP


def _json(name):
    with open(os.path.join(CH, name), encoding="utf-8") as fh:
        return json.load(fh)


def _page_records():
    s = Session(spec=example_case("wall-2"), key="x")
    reply, _ = protocol.handle(s, {"id": 1, "type": "records.list"})
    assert reply["ok"]
    return {e["file"]: e for e in reply["records"] if e.get("file")}


def _met(v):
    return v.get("met") if isinstance(v, dict) else None


def test_every_evaluations_gates_on_the_page_are_its_records():
    page = _page_records()
    # where each record keeps its verdicts, written out by hand
    where = {
        "evaluation-twolevel-20261111-004223.json": lambda r, k: _met(r["gates"].get(k)),
        "evaluation-forms-20261019-205923.json": lambda r, k: _met(r["gates"].get(k)),
        "evaluation-20261018-203833.json": lambda r, k: _met(r["gates"].get(k)),
        "evaluation-prior-20261107-172219.json": lambda r, k: _met(r["gates"]["P"].get(k)),
        "evaluation-squares-20261108-141654.json": lambda r, k: _met(r.get(k)),
        "evaluation-hybrid-20261107-231105.json": lambda r, k: _met(r.get(k)),
    }
    present = [f for f in where if os.path.isfile(os.path.join(CH, f))]
    assert present, "no evaluation records on this machine"
    for f in present:
        rec = _json(f)
        e = page[f]
        assert e["gates"], f
        for g in e["gates"]:
            assert g["met"] == where[f](rec, g["key"]), (f, g["key"])
            assert g["title"] and g["bar"], (f, g["key"])     # the registered title and bar
    tl = page.get("evaluation-twolevel-20261111-004223.json")
    if tl:
        assert {g["key"]: g["met"] for g in tl["gates"]}["E4"] is False      # missed, as recorded
        assert {g["key"]: g["met"] for g in tl["gates"]}["CX"] is True


def test_every_price_on_the_page_is_its_records_and_says_its_problem_set():
    page = _page_records()
    f = "evaluation-twolevel-20261111-004223.json"
    if f in page:
        gm = _json(f)["gates"]["PR"]["geometric_mean"]
        shown = {(p["arm"], p["set"]): p["value"] for p in page[f]["prices"]}
        assert shown == {(a, s): v for a, sets in gm.items() for s, v in sets.items()}
        for p in page[f]["prices"]:
            assert p["label"] in ("all priced", "held-out 6", "squares (R2's 16)")
    f = "evaluation-squares-20261108-141654.json"
    if f in page:
        rec = _json(f)
        assert [(p["gate"], p["value"]) for p in page[f]["prices"]] == [("PR2", rec["PR2"]["geometric_mean"])]


def test_rho_on_the_page_is_the_records():
    page = _page_records()
    f = "evaluation-twolevel-20261111-004223.json"
    if f not in page or not page[f]["rho"]:
        pytest.skip("the two-level record's test rows are not on this machine")
    nc = _json(f)["gates"]["NC"]
    r = page[f]["rho"]
    assert r["pieces"] == nc["pieces"] == 2400
    assert r["median"] == pytest.approx(nc["median"], rel=1e-12)
    assert r["p10"] == pytest.approx(nc["p10"], rel=1e-12)
    assert sum(r["bins"]) == r["pieces"]


def test_the_coolant_registration_appears_with_no_change_to_the_code(tmp_path):
    for f in os.listdir(CH):                     # the existing records, as they are
        if f.startswith(("evaluation-", "registration-")) and f.endswith(".json"):
            shutil.copy(os.path.join(CH, f), tmp_path / f)
    before = R.evaluations(str(tmp_path))
    assert before[0]["name"] == "Coolant (CO-T)" and before[0]["status"] == "stopped, not registered"
    assert before[0]["arms"] == ["TL0", "arm P", "arm R", "classical"] and "CL-A" in before[0]["note"]
    assert "stopped" in before[0]["note"] and "CAP" in before[0]["note"]
    assert not any(e["kind"] == "stage0b" for e in before)      # the placeholder it replaced
    reg = {"registration": "a test's stand-in for the coolant's registration",
           "arms": {"TL0": {}, "arm P": {}, "arm R": {}, "classical": {}},
           "gates": [{"key": "CL-A", "title": "meets E4 where TL0 misses it", "bar": "every evaluation layout"}]}
    (tmp_path / "registration-cot.json").write_text(json.dumps(reg))
    after = R.evaluations(str(tmp_path))
    cot = [e for e in after if e["kind"] == "cot"]
    assert len(cot) == 1 and cot[0]["status"] == "registered, not run"
    assert cot[0]["name"] == "Coolant (CO-T)" and cot[0]["arms"] == ["TL0", "arm P", "arm R", "classical"]
    # the gates are read from atlas/chartop/gates_cot.py (the draft), not from the file's own list
    from atlas.chartop import gates_cot as GC
    assert [g["key"] for g in cot[0]["gates"]] == [g.key for g in GC.GATES]
    title = {g.key: g.title for g in GC.GATES}
    # its evaluation, once written, is read the same way
    (tmp_path / "evaluation-cot-20261201-120000.json").write_text(json.dumps(
        {"written": "2026-12-01", "registered_run": True, "code": "abc1234", "main": "arm P",
         "gates": {"CL-A": {"met": True}}}))
    ev = [e for e in R.evaluations(str(tmp_path)) if e["kind"] == "cot"]
    assert len(ev) == 1 and ev[0]["status"] == "evaluated"
    got = {g["key"]: (g["met"], g["title"]) for g in ev[0]["gates"]}
    assert got["CL-A"] == (True, title["CL-A"]) and got["E4"] == (None, title["E4"])
    # a Stage 0b registration, once it exists, is still read by rule
    (tmp_path / "registration-stage0b.json").write_text(json.dumps({"registration": "stand-in", "arms": ["TL0"]}))
    s0b = [e for e in R.evaluations(str(tmp_path)) if e["kind"] == "stage0b"]
    assert len(s0b) == 1 and s0b[0]["name"] == "Stage 0b" and s0b[0]["status"] == "registered, not run"
