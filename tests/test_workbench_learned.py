"""The learned case (demo item 1.5): its registration, its split, its arms and its judgement.

The gate was registered before any data existed (`out/learned-case/registered.txt`,
`atlas/workbench/learned_gate.py`); these tests keep the registration and the code
from drifting apart, and check the machinery the evaluation runs on: the farm run
with a scaled band and per-rotor induction reproduces the workbench's run to the
bit, the split is deterministic and never touches the held-out layout, the
judgement fails where it should, and a learned step with a zero network is the
blend of unchanged windows, projected.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from atlas.workbench import learned_gate as G                          # noqa: E402

REGISTERED = os.path.join(HERE, "out", "learned-case", "registered.txt")
MODULE = os.path.join(HERE, "atlas", "workbench", "learned_gate.py")


def _module_hash() -> str:
    import hashlib
    with open(MODULE, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


# ---------------------------------------------------------------------------
# the registration
# ---------------------------------------------------------------------------


def test_the_registration_and_the_gate_module_agree():
    """A bar changed after registration fails here (the vault's memory 'pre-registered
    criteria drift in code'): the file records the module's hash and the split's."""
    if not os.path.isfile(REGISTERED):
        pytest.skip("the registration lives in the repository's out/learned-case/, which "
                    "this copy does not carry")
    text = open(REGISTERED, encoding="ascii").read()
    m = re.search(r"learned_gate\.py\s+sha256 ([0-9a-f]{64})", text)
    assert m and m.group(1) == _module_hash()
    m = re.search(r"THE SPLIT\s+\(sha256 ([0-9a-f]{64})\)", text)
    assert m and m.group(1) == G.split_hash()
    for g in G.GATE:
        assert g.bar in " ".join(text.split()), g.key


def test_the_split_is_deterministic_and_never_the_held_out_layout():
    assert not G.held_out_ok(G.HELD_OUT)
    assert G.sample_layout(1000) == G.sample_layout(1000)
    assert set(G.TRAIN_SEEDS).isdisjoint(G.VALIDATION_SEEDS)
    for s in G.TRAIN_SEEDS[:20] + G.VALIDATION_SEEDS:
        lay = G.sample_layout(s)
        assert G.N_ROTORS[0] <= len(lay.rotors) <= G.N_ROTORS[1]
        assert G.held_out_ok(lay.rotors)
        pts = np.array([(x, y) for x, y, _a in lay.rotors])
        d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
        assert d[np.triu_indices(len(pts), 1)].min() >= G.MIN_SPACING


# ---------------------------------------------------------------------------
# the judgement
# ---------------------------------------------------------------------------


def _results(eL=0.010, eF=0.010, eEp=0.012, eCc=0.050, tL=0.5, on_ac=True):
    def arm(e, t, energy):
        return {"step_seconds": [t] * G.STEPS, "power": [1.0] * G.STEPS,
                "energy": [energy] * G.STEPS, "div": [1e-15] * G.STEPS,
                "errors": {str(h): {"P": e, "V": e} for h in G.HORIZONS}}
    return {"on_ac": on_ac, "held_out_verified": True,
            "arms": {"F": arm(eF, 6.0, 1.0), "Ep": arm(eEp, 1.3, 1.0),
                     "L": arm(eL, tL, 1.02), "Cc": arm(eCc, 0.4, 1.0)}}


def test_the_judgement_passes_a_case_that_meets_every_bar():
    j = G.judge(_results())
    assert j["all"] and all(j[g]["passed"] for g in ("G1", "G2", "G3", "G4", "G5", "G6"))


def test_the_judgement_fails_where_it_should():
    assert G.judge(_results(tL=1.0))["G1"]["passed"] is False         # 1.3x is under 1.5
    assert G.judge(_results(on_ac=False))["G1"]["passed"] is None     # battery: not judged
    assert G.judge(_results(eL=0.0135))["G2"]["passed"] is False      # over 1.1 x 0.012
    assert G.judge(_results(eL=0.0131))["G2"]["passed"] is True
    assert G.judge(_results(eCc=0.009))["G5"]["passed"] is False      # the coarse wins
    r = _results()
    r["arms"]["L"]["energy"] = [1.2] * G.STEPS                         # 20% off Ep's
    assert G.judge(r)["G4"]["passed"] is False
    r = _results()
    r["arms"]["L"]["div"][7] = 1e-6
    assert G.judge(r)["G3"]["passed"] is False
    assert not G.judge(_results(eCc=0.009))["all"]


def test_the_velocity_error_is_read_on_the_d16_grid():
    f = np.random.default_rng(0).normal(size=(464, 688))
    t = np.kron(f, np.ones((2, 2)))                  # the same field at D/64
    assert G.error_velocity((f, 0 * f), 32, (t, 0 * t), 64) == pytest.approx(0.0, abs=1e-15)
    assert G.error_velocity((f + 0.1, 0 * f), 32, (t, 0 * t), 64) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# the arms
# ---------------------------------------------------------------------------


def test_the_farm_run_with_defaults_is_the_workbenchs_to_the_bit():
    from atlas.workbench import learned_arms as LA
    from atlas.workbench.families import windfarm as wf
    from atlas.workbench.spec import example_case
    s = example_case("wake-array-3")
    a, b = wf.build(s, arms=("serial",), threads=1), LA.FarmRun(s, arms=("serial",), threads=1)
    sa, sb = a.initial("serial"), b.initial("serial")
    for _ in range(2):
        sa, sb = a.step("serial", sa), b.step("serial", sb)
    assert b.band_cells == wf.BAND
    assert np.array_equal(sa.u, sb.u) and np.array_equal(sa.v, sb.v)


def test_the_scaled_cases_keep_the_physics_where_it_is():
    from atlas.workbench import learned_arms as LA
    from atlas.workbench.spec import example_case
    s = example_case(G.CASE)
    for f in (0.5, 2.0):
        c = LA.scaled_spec(s, f)
        assert c.domain.nx * c.domain.dx == pytest.approx(s.domain.nx * s.domain.dx)
        assert [(d.x, d.y) for d in c.devices] == [(d.x, d.y) for d in s.devices]
        assert c.windows[1].x0 == s.windows[1].x0 * f
    with pytest.raises(ValueError):
        LA.scaled_spec(s, 0.3)


# ---------------------------------------------------------------------------
# the case in the page
# ---------------------------------------------------------------------------


@pytest.fixture()
def wb(tmp_path, monkeypatch):
    pytest.importorskip("panel")
    from atlas.workbench import compile as compile_
    from atlas.workbench import runner
    from atlas.workbench.app import Workbench
    monkeypatch.setattr(runner.CaseRun, "start", lambda self: None)
    monkeypatch.setattr(compile_.CompileJob, "start", lambda self: None)
    return Workbench(cases_dir=str(tmp_path))


def test_the_learned_case_is_first_under_the_farms_examples_and_fixed(wb):
    from atlas.workbench import learned_case as LC
    wb.dispatch("case:type:incompressible-2d")
    items = wb._example_items()
    assert items[0][1] == "file:example:%s" % LC.KEY
    wb.dispatch("file:example:%s" % LC.KEY)
    assert wb.read_only and wb.case_key == LC.KEY
    before = wb.spec.to_json()
    assert wb.edit(lambda c: setattr(c.run, "steps", 3), "an edit") is False
    assert wb.spec.to_json() == before
    assert any(LC.READ_ONLY in line for line in wb.log_lines)
    assert set(wb.run_arms) == {"parallel", "full", "learned", "coarse"}
    wb.dispatch("case:type:conduction-2d")                 # another kind: editable again
    assert not wb.read_only
    assert not any(i[1].endswith(LC.KEY) for i in wb._example_items())
    wb.dispatch("edit:undo")                               # Undo restores the fixed case
    assert wb.read_only


def test_the_learned_case_marches_four_arms_and_measures_them_against_the_truth(
        tmp_path, monkeypatch):
    """A zero network and a truth that is the full domain's own fields, at a horizon
    of 2 macro-steps: the full domain's error is then exactly zero, and every arm
    is measured, checked and drawn."""
    pytest.importorskip("torch")
    from atlas.workbench import learned_case as LC
    from atlas.workbench import runner
    from atlas.workbench.learned_net import WindowUNet, save_window_net
    from atlas.workbench.runview import learned_card
    from atlas.workbench.spec import example_case
    net = WindowUNet(width=G.NET_WIDTH, depth=G.NET_DEPTH)
    for p in net.parameters():
        p.data.zero_()
    weights = str(tmp_path / "w.pt")
    save_window_net(net, weights)
    monkeypatch.setattr(LC, "WEIGHTS", weights)
    monkeypatch.setattr(G, "HORIZONS", (2, 40))
    spec = example_case(LC.KEY)
    spec.run.steps = 2
    from atlas.workbench import learned_arms as LA
    f = LA.FarmRun(spec, arms=("full",), threads=1)
    s, power = f.initial("full"), []
    for _ in range(2):
        s = f.step("full", s)
        power.append(f.power(s.rec))
    truth = tmp_path / "truth.npz"
    # the stored truth lives on the D/16 grid it is compared on
    np.savez(truth, power=np.array(power + [power[-1]] * 38),
             u2=G.block_mean(s.u, 2), v2=G.block_mean(s.v, 2))
    monkeypatch.setattr(LC, "TRUTH", str(truth))
    r = runner.CaseRun(spec, arms=LC.ARMS, steps=2, threads=2).run_blocking()
    assert r.status == "done", r.progress().error
    assert r.arms == LC.ARMS
    e = r.results["metrics"]["errors_vs_truth"]
    assert set(e) == set(LC.ARMS)
    assert e["full"]["V"] == pytest.approx(0.0, abs=1e-12)
    assert e["full"]["P"] == pytest.approx(0.0, abs=1e-12)
    by = {c["key"]: c for c in r.results["checks"]}
    assert by["mass"]["passed"] is True
    assert by["competitor"]["passed"] in (True, False)
    assert r.results["arm_labels"]["learned"] == LC.ARM_LABELS["learned"]
    import html as html_
    page = learned_card(r.results).object
    assert html_.escape(LC.CAVEAT) in page and "Velocity" in page and "Farm power" in page
    # the headline speed is the learned arm's, not the coarse competitor's
    from atlas.workbench.runview import summary_cards
    speed = summary_cards(r.results)[0].object
    assert html_.escape(LC.ARM_LABELS["learned"].lower()) in speed


def test_a_learned_step_with_a_zero_network_is_the_projected_blend():
    torch = pytest.importorskip("torch")
    from atlas.workbench import learned_arms as LA
    from atlas.workbench.learned_net import WindowUNet
    from atlas.workbench.spec import example_case
    net = WindowUNet(width=4, depth=2)
    for p in net.parameters():
        p.data.zero_()
    s = example_case("wake-array-3")
    run = LA.LearnedFarmRun(s, net, arms=("learned",), threads=1)
    st = run.step("learned", run.initial("learned"))
    assert float(np.abs(st.u - 1.0).max()) < 1e-12 and float(np.abs(st.v).max()) < 1e-12
    assert run.mass_measure("learned", st) <= G.G3_DIV
    assert torch.get_num_threads() == 1
