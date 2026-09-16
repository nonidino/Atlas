"""Tier 73 -- the certified mode on the screen: section 12's criterion 4.

[[poc3-racelab-certified-screen]].

Tier 72 built the mechanism and priced it; nobody could see it. This pins the
switch, the two prices it must keep apart, and the three things the page is
obliged to show rather than drop.

The column is exercised on a CHEAP flow through a stand-in for `CarUnion`, so
the suite pays for the switch and not for a 377,267-unknown composite. The
car's own numbers are in `out/racelab22/racelab22.json`.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

import atlas.demo_racelab.bodyfitted as BF
from atlas.cases import overset_ns as NS

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "atlas", "demo_racelab", "static", "bodyfitted.html")


class _Union:
    """Just enough of `CarUnion` for the column's step to run on a bare flow."""

    def __init__(self, flow):
        self.flow = flow
        self.trace = {}
        self.coolant = None
        self.envelope = None
        self.outside_steps = 0

    def step(self):
        return self.flow.step()


@pytest.fixture(scope="module")
def col():
    """A column past its impulsive start.

    **The pre-march is not decoration.** `cylinder_flow` starts impulsively, and
    the certified sweep does not converge through that transient -- it returns
    `fallback_unconverged` and the next step blows up. The real column never
    marches from rest either: it releases from a state settled to t = 12, which
    is exactly what W293 is about. So the fixture settles first, in the
    classical mode, and the switch is exercised on a flow that is actually
    marching.
    """
    flow = NS.cylinder_flow(h=1.0 / 14, ni=112, nj=23, dt=0.03)
    flow.precond = "ilu"
    c = BF.BodyFittedColumn()
    c.flow = flow
    c.union = _Union(flow)
    c.coverage = {"body_grid_frac": 0.61, "windows_learned": 0}
    for _ in range(5):
        c.step()
    return c


# ---------------------------------------------------------------------------
# section 4.2's switch
# ---------------------------------------------------------------------------


def test_the_column_starts_classical_with_no_solver_installed(col):
    assert col.mode == "classical"
    assert col.flow.momentum_solver is None


def test_flipping_to_certified_installs_the_solver_and_back_removes_it(col):
    col.set_mode("certified")
    assert col.mode == "certified"
    assert col.flow.momentum_solver is not None
    col.set_mode("classical")
    assert col.flow.momentum_solver is None, "the classical mode must be the CLASSICAL solve"


def test_learned_is_refused_with_BOTH_of_its_reasons(col):
    """Section 4.3's rule, applied to a mode: an option with no shippable
    expert is refused **with the reason**, never hidden.

    Two INDEPENDENT blocks, and the second was only found when the checkpoint
    was actually asked to run (Tier 72, W294). Naming only the spatial one
    would leave a reader thinking a better tiling could fix it.
    """
    with pytest.raises(ValueError) as exc:
        col.set_mode("learned")
    why = str(exc.value)
    assert "128" in why and "61.0%" in why, "the SPATIAL block is not named"
    assert "0.1" in why and "out of distribution" in why, "the TIME block is not named"
    assert col.mode != "learned"


def test_an_unknown_mode_is_refused(col):
    with pytest.raises(ValueError):
        col.set_mode("nonsense")


# ---------------------------------------------------------------------------
# section 5.2's certified row
# ---------------------------------------------------------------------------


def test_the_certified_step_reports_what_section_5_2_asks_for(col):
    col.set_mode("certified")
    col.set_verify(False)
    for _ in range(2):
        row = col.step()
    c = row["certified"]
    for key in ("outer_iterations", "inner_cheap_calls", "residual"):
        assert key in c and c[key] is not None, key
    assert c["outer_iterations"] >= 1 and c["inner_cheap_calls"] >= 1
    assert c["status"] in ("converged", "fallback_converged")
    # the two residuals are in different currencies and must both be carried
    assert c["algebraic_residual"] != c["residual"]


def test_the_error_is_None_until_verify_pays_for_it(col):
    """**The residual is free and the error is not.**

    The residual is computed every outer iteration whatever else is on.
    MEASURING the distance to the classical answer needs the classical solve
    run beside the certified one, which is what `verify` buys. Reporting a
    number there with verify off would be reporting something nobody computed.
    """
    col.set_mode("certified")
    col.set_verify(False)
    row = col.step()
    assert row["certified"]["error_to_classical"] is None
    assert row["certified"]["classical_iterations"] == []

    col.set_verify(True)
    row = col.step()
    c = row["certified"]
    assert c["error_to_classical"] is not None
    assert np.isfinite(c["error_to_classical"])
    assert c["classical_iterations"], "verify must actually run the classical solve"
    # and the certificate must bracket what it claims to bound
    assert c["error_to_classical"] < 1e-4


# ---------------------------------------------------------------------------
# the price, and the two prices that must not be confused
# ---------------------------------------------------------------------------


def test_a_verified_certified_step_is_its_own_arm(col):
    """With `verify` on a full classical solve runs beside the certified one
    every step. Mixing those samples into the plain certified arm would put the
    verification's price inside the mode's, which is the confound Tier 72 found
    in its own instrument."""
    col.set_mode("certified")
    col.set_verify(False)
    col.step()
    n_plain = len(col.mode_steps["certified"])
    n_ver = len(col.mode_steps["certified+verify"])
    col.set_verify(True)
    col.step()
    assert len(col.mode_steps["certified"]) == n_plain, "a verified step landed in the plain arm"
    assert len(col.mode_steps["certified+verify"]) == n_ver + 1


def test_the_ratio_is_withheld_until_each_arm_has_replicates():
    """Section 5.4's rule 5: never show a single-draw number as an effect size."""
    c = BF.BodyFittedColumn()
    c.mode_steps = {"classical": [0.5] * (BF.BodyFittedColumn.MIN_SAMPLES - 1),
                    "certified": [0.6] * 20, "certified+verify": []}
    c.mode = "certified"
    assert c.cost()["ratio_mode_only"] is None, "a ratio was quoted on too few samples"
    c.mode_steps["classical"] = [0.5] * BF.BodyFittedColumn.MIN_SAMPLES
    out = c.cost()
    assert out["ratio_mode_only"] == pytest.approx(1.2)
    # ... and it always comes with its counts and its range
    for arm in ("classical", "certified"):
        a = out["arms"][arm]
        assert a["n"] > 0 and a["lo_s"] is not None and a["hi_s"] is not None


def test_the_mode_price_and_the_verification_price_are_reported_separately():
    """Quoting only the compared arm's ratio while verify is on prices the
    INSTRUMENT and calls it the mode."""
    c = BF.BodyFittedColumn()
    c.mode_steps = {"classical": [1.0] * 10, "certified": [1.1] * 10,
                    "certified+verify": [1.6] * 10}
    c.mode, c.verify = "certified", True
    out = c.cost()
    assert out["ratio_mode_only"] == pytest.approx(1.1)
    assert out["ratio_with_verify"] == pytest.approx(1.6)
    assert out["comparing"] == "certified+verify"


# ---------------------------------------------------------------------------
# the page -- because an engine that is right and a page that is blank is the
# defect Tier 71 shipped and caught by opening it
# ---------------------------------------------------------------------------


def test_the_page_draws_the_switch_and_every_number_behind_it():
    text = open(STATIC, encoding="utf-8").read()
    for ident in ('id="modes"', 'id="cert"', 'id="verify"', 'id="certPanel"',
                  'id="certNote"', 'id="notPerWindow"', 'id="modeWhat"'):
        assert ident in text, ident
    for label in ("outer iterations", "inner cheap calls", "residual",
                  "error to classical", "cost of the mode", "cost with verify",
                  "BiCGSTAB iterations"):
        assert label in text, label
    # the switch must SEND, not just render
    assert 'kind: "mode"' in text and 'kind: "verify"' in text


def test_the_page_never_shows_one_arm_s_numbers_under_another_s_name():
    """**The defect this panel was built making.**

    Reading `arms[comparing]` put the classical arm's 0.516 s under a
    "certified step" label whenever the mode was classical -- a number that was
    real and a label that was wrong. Each arm is now rendered under its own key.
    """
    text = open(STATIC, encoding="utf-8").read()
    assert 'A[cost.comparing]' not in text, \
        "the panel is reading the compared arm again instead of each arm by name"
    for key in ('["classical", "classical step"]',
                '["certified", "certified step"]',
                '["certified+verify", "certified + verify step"]'):
        assert key in text, key


def test_the_page_carries_the_reason_the_mode_is_not_per_window():
    """Section 5.2 asks for per-window certified telemetry and this column
    cannot give it (Tier 65, W295). The page says so rather than faking a
    split or leaving the absence unexplained."""
    text = open(STATIC, encoding="utf-8").read()
    assert "notPerWindow" in text
    assert "not one per window" in BF.CERTIFIED_NOT_PER_WINDOW
    assert "W295" in BF.CERTIFIED_NOT_PER_WINDOW
    assert "porous column" in BF.CERTIFIED_NOT_PER_WINDOW


def test_the_server_offers_the_mode_and_verify_routes():
    pytest.importorskip("fastapi")
    import atlas.demo_racelab.bodyfitted_server as S

    src = open(S.__file__, encoding="utf-8").read()
    assert '@app.post("/api/mode")' in src
    assert '@app.post("/api/verify")' in src
    # and the worker must handle them, or the route queues into nothing
    assert 'kind == "mode"' in src and 'kind == "verify"' in src
    # the payload has to carry the ENGINE's mode back, so a refused flip
    # cannot leave the page showing a mode the march is not running
    assert '"mode": self.col.mode' in src


def test_the_meta_names_the_modes_and_the_refusal():
    pytest.importorskip("fastapi")
    from atlas.demo_racelab.bodyfitted_server import BodyFittedEngine

    m = BodyFittedEngine(autostart=False).meta()
    assert list(m["modes"]) == list(BF.MODES)
    assert "learned" in m["mode_refused"]
    assert m["certified_not_per_window"]
    assert m["certified_what"]
