"""Tier 32 -- W149, virtual control: uncontrollable, or merely unconfined?

`scripts/w149_control_observability.py` and `out/w149/w149.json`.  Six groups:

  * **the instrument** -- the field march equals `WindowAgent.respond`'s flux
    BITWISE, and the periodic positive control returns a control-to-jump map that
    is exactly zero.  These are live and cheap, and if either fails nothing else
    in the tier means anything;
  * **gate 0** -- the objective's own consistency check, which splits on the
    elliptic axis: the reference trace strictly reduces J on the R10-admissible
    column at all three dt, and strictly INCREASES it on the column R10 refuses,
    by more as dt grows;
  * **gate 1's spectra** -- `beta_ctrl`, the conditioning, the rank, and the
    predicted rank-one null space, which is asserted here rather than described;
  * **the controls** -- the finite-difference step, the probe-state horizon, and
    the fact that the three states are inside the expert's declared envelope;
  * **the negative half** -- solving the objective drives `sigma` UP, and the
    monolith's own datum in the same control space drives it down.  Both are
    assertions, because the favourable half would be misread without them;
  * **what the write-up may not claim** -- the page does not say `tau` moved, does
    not call any seam certified, and does not claim a speed-up.

The artifact-backed tests re-derive the published numbers from `out/w149/w149.json`
and skip when it is absent.  The live tests build seams and probe them, which
costs about fifteen seconds; nothing here re-runs the full measurement.
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ARTIFACT = os.path.join(_ROOT, "out", "w149", "w149.json")
_SCRIPT = os.path.join(_ROOT, "scripts", "w149_control_observability.py")
_PAGE = os.path.join(_ROOT, "wiki", "concepts", "Atlas 0.1", "common",
                     "control-observability.md")
_STATE = os.path.join(_ROOT, "out", "tier0_verify", "s0_state.npz")


@pytest.fixture(scope="module")
def art():
    if not os.path.isfile(_ARTIFACT):
        pytest.skip("out/w149/w149.json is not here")
    with open(_ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def w149():
    """The measurement module, loaded by path -- `scripts/` is not a package."""
    if not os.path.isfile(_SCRIPT):
        pytest.skip("scripts/w149_control_observability.py is not here")
    spec = importlib.util.spec_from_file_location("_w149", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def state():
    if not os.path.isfile(_STATE):
        pytest.skip("out/tier0_verify/s0_state.npz is not here")
    d = np.load(_STATE)
    return d["u"], d["v"], d["fx"]


def _case(art, key):
    c = art.get("gate1", {}).get("cases", {}).get(key)
    if c is None or "unreachable" in c:
        pytest.skip(f"gate 1 case {key!r} is not in the artifact")
    return c


def _settled(c):
    """The state the vault's other measurements are reported at."""
    return c["states"][-1]


# ---------------------------------------------------------------------------
# 1. the instrument
# ---------------------------------------------------------------------------


def test_the_field_march_and_the_published_flux_response_agree_bitwise(w149, state):
    """`Side.march` reimplements `respond`'s ring so it can return a field.

    The one thing that would silently invalidate every number in gate 1 is the
    two drifting apart, so the flux read off the march is compared with
    `WindowAgent.respond`'s under `np.array_equal` rather than an allclose.
    """
    u, v, fx = state
    out = w149.check_flux_agrees(u, v, fx)
    assert out["bitwise_equal"], out
    assert out["max_abs_difference"] == 0.0
    assert out["flux_scale"] > 0.0, "a zero flux would make the check vacuous"


def test_a_periodic_window_returns_a_control_to_jump_map_that_is_exactly_zero(
        w149, state):
    """The positive control, live, with a known answer.

    `probed-dtn-coupling` 4.5's `Xi` already reproduces as exactly zero on this
    configuration with 34 solver calls.  A periodic window has no boundary to
    impose a datum on, so the control-to-jump map must be zero -- not small.  A
    probe that manufactured a small operator out of roundoff, an initial-
    condition leak or an off-by-one in the ring index would still produce
    plausible numbers on the Dirichlet agent, and this is the only measurement
    that can catch it.
    """
    u, v, fx = state
    A, B, tiling = w149.spectral_sides(u, v, fx, 0.05)
    hold = w149._lagged_hold(A, B, tiling)
    T, j0 = w149.assemble_T(A, B, 1e-2, m=4, hold=hold)
    assert np.all(T == 0.0), f"max |T| = {np.abs(T).max()}"
    sp = w149.spectrum(T)
    assert sp["exactly_zero"] is True
    assert sp["beta_ctrl"] == 0.0
    assert sp["rank_machine"] == 0
    # and the probe genuinely ran, rather than returning zero by not being called
    assert A.n_calls + B.n_calls >= 2 * (4 + 1)
    assert np.linalg.norm(j0) > 0.0, (
        "the two windows must still disagree; a zero jump would make the zero "
        "operator uninformative")


def test_the_seam_geometry_is_the_declared_overlap(w149, state):
    u, v, fx = state
    A, B, _tiling = w149.window_sides(u, v, fx, 0.05, True)
    a0, a1 = A.overlap_cols(B)
    b0, b1 = B.overlap_cols(A)
    from atlas.cases import window_ns as W
    assert a1 - a0 == b1 - b0 == W.DEFAULT_TILING.halo == 21
    assert (A.seam_face, B.seam_face) == ("xhi", "xlo")


# ---------------------------------------------------------------------------
# 2. gate 0 -- the objective, before anything else
# ---------------------------------------------------------------------------


def test_gate_0_passes_on_the_column_r10_admits(art):
    """J(reference) < J(lagged), strictly, at dt = 0.05, 0.01 and 0.002.

    Three points because three points are what exposed flux balance's
    dt-scaling: that condition's reference-trace ratio sat at 1.002 and its
    overshoot GREW as dt refined, which is what named it a steady-state
    condition.  This one falls at all three.
    """
    g0 = art["gate0"]
    assert g0["passes_admissible"] is True
    ratios = g0["by_column"]["exposed"]["ratios"]
    assert len(ratios) == 3
    for r in ratios:
        assert r < 1.0, ratios
    assert max(ratios) < 0.995, ratios


def test_gate_0_fails_on_the_column_r10_refuses_and_fails_worse_at_larger_dt(art):
    """The failure is R10's, and it carries R10's signature.

    A defect that grows with dt is not a dt-scaling defect of the objective --
    flux balance's got WORSE as dt shrank.  This one vanishes as dt shrinks and
    is largest at the coarsest step, which is what a per-window elliptic solve
    accumulating over ten sub-steps looks like.
    """
    g0 = art["gate0"]
    ratios = g0["by_column"]["embedded"]["ratios"]      # dt = 0.05, 0.01, 0.002
    assert ratios[0] > 1.0, ratios
    assert ratios == sorted(ratios, reverse=True), (
        "the embedded column's defect must shrink as dt shrinks", ratios)
    assert g0["passes"] is False, (
        "the literal form of the gate -- every row -- does not hold, and the "
        "artifact must keep saying so")


def test_j_is_monotone_along_the_line_from_the_lagged_trace_to_the_true_one(art):
    """Monotone in both directions, which is what makes the split unambiguous.

    A ratio near one could be noise.  A ratio near one at the end of a five-point
    monotone line is a direction.
    """
    for row in art["gate0"]["rows"]:
        if row["scope"] != "all":
            continue
        js = [row["J"][k] for k in sorted(row["J"], key=float)]
        d = np.diff(js)
        if row["elliptic"] == "exposed":
            assert np.all(d < 0), (row["dt"], js)
        else:
            assert np.all(d > 0), (row["dt"], js)


# ---------------------------------------------------------------------------
# 3. gate 1 -- the spectra
# ---------------------------------------------------------------------------


def test_the_periodic_control_is_zero_in_the_artifact_too(art):
    c = _case(art, "spectral_periodic")
    for st in c["states"]:
        assert st["exactly_zero"] is True
        assert st["beta_ctrl"] == 0.0
        assert st["max_abs"] == 0.0
        assert st["rank_machine"] == 0
        assert st["Xi_control"] == 0.0


def test_the_embedded_column_has_the_predicted_rank_one_null_space(art):
    """`probed-dtn-coupling` 2.1's n_0 = 1, asserted rather than described.

    With incompressibility INSIDE `Lambda_i` the trace is constrained to zero net
    flux and the operator is singular in one direction; with the elliptic part
    exposed it is not.  The direction is the constant mode, and it is shared
    equally by the two sides -- a net-flux constraint is a statement about the
    seam, not about either window -- so the split is asserted too.  Without it a
    coincidental single small singular value would pass.
    """
    c = _case(art, "windowns_embedded")
    st = _settled(c)
    assert st["null_dim_gap"] == 1, st.get("bottom_gap_index")
    assert st["bottom_gap_index"] == st["shape"][1] - 1, (
        "the gap must be at the BOTTOM of the spectrum, not in the middle")
    assert st["bottom_gap_ratio"] < 0.25, st["bottom_gap_ratio"]
    n = st["null"]
    assert n["overlap_with_constant_modes"] > 0.95, n
    assert abs(n["share_side_a"] - n["share_side_b"]) < 0.05, (
        "a net-flux null direction belongs to the seam and must be carried "
        "equally by both sides", n)


def test_the_exposed_column_has_no_such_null_space(art):
    """The control for the test above: the same measurement, elliptic exposed.

    Exposed, `Lambda_i` is the pure advection-diffusion DtN and is nonsingular.
    If the gap appeared here too it would be an artifact of the instrument.
    """
    c = _case(art, "windowns_exposed")
    st = _settled(c)
    assert st["rank_machine"] == st["shape"][1]
    assert st["bottom_gap_ratio"] > 0.4, st["bottom_gap_ratio"]
    assert st["null"]["overlap_with_constant_modes"] < 0.5, st["null"]


def test_poseidon_t_is_observable_from_a_seam(art):
    """The number the tier exists for.

    Full rank, and better conditioned than the classical agent with its elliptic
    part embedded.  `support_reach`'s 128-of-128 says the response is dense;
    dense is not well conditioned, and this is the part that was missing.
    """
    p = _case(art, "poseidon")
    e = _case(art, "windowns_embedded")
    for st in p["states"]:
        assert st["beta_ctrl"] > 0.0
        assert st["rank_machine"] == st["shape"][1] == 32
        assert np.isfinite(st["kappa"])
    assert _settled(p)["kappa"] < _settled(e)["kappa"], (
        "the checkpoint's control map is better conditioned than the classical "
        "agent's when that agent embeds its elliptic solve")
    assert 0.0 < _settled(p)["Xi_control"] < 1.0, (
        "the checkpoint reproduces part of the reference agent's control "
        "authority, and part is not all")


def test_the_declared_port_is_narrower_than_the_datum_and_it_costs(art):
    """The MECH port carries the normal component; a ring carries two.

    The widened control is the same seam, the same states, the same probe, one
    more component -- and it reaches most of the jump where the declared one
    reaches a third of it.  That is a fact about the port declaration rather than
    about the physics, which is why it gets a test rather than a sentence.
    """
    narrow = _settled(_case(art, "windowns_exposed"))
    wide = _settled(_case(art, "windowns_exposed_2c"))
    assert wide["shape"][1] == 2 * narrow["shape"][1]
    assert wide["controllable_fraction"] > 0.9
    assert narrow["controllable_fraction"] < 0.5
    # widening the control barely moves the conditioning, so the restriction is
    # buying nothing back
    assert abs(wide["kappa"] / narrow["kappa"] - 1.0) < 0.3


# ---------------------------------------------------------------------------
# 4. the controls
# ---------------------------------------------------------------------------


def test_beta_ctrl_is_reported_across_three_states_that_are_three_states(art):
    """`positive-controls-need-a-horizon`, applied to this tier's own headline.

    The settled state is a FIXED POINT -- 200 macro-steps move it 6.03e-4 and it
    saturates by step 40 -- so a spread quoted along it would be a spread across
    one state wearing three labels.  The separation is asserted before the
    stability is.
    """
    tr = art["controls"]["trajectory"]
    assert tr["max_separation"] > 0.05, tr
    assert len(tr["states"]) == 3
    for r in tr["states"]:
        assert r["inside_declared_envelope"] is True, (
            "a spread bought with an out-of-envelope state is not a spread", r)
    assert tr["reproduces_artifact"] == 0.0, (
        "the last mark must reproduce the supplied state exactly, which is what "
        "says this march is the one that made it")
    for key in ("windowns_exposed", "windowns_embedded", "poseidon"):
        c = art["gate1"]["cases"].get(key)
        if c is None or "unreachable" in c:
            continue
        assert c["beta_ctrl_spread"] < 1.05, (key, c["beta_ctrl_spread"])


def test_the_finite_difference_step_does_not_set_the_answer(art):
    """Four decades for the solvers, and the checkpoint's own floor for it.

    `window_ns` declares a machine-epsilon probe floor because its expert is a
    deterministic solver.  The checkpoint's is 1e-6, so its sweep is expected to
    break at the small end -- and it does, which is the control working rather
    than failing.
    """
    sw = art["controls"]["eps_sweep"]
    for key in ("windowns_exposed", "windowns_embedded"):
        assert sw[key]["beta_ctrl_spread"] < 1.01, (key, sw[key])
    if "poseidon" in sw:
        p = sw["poseidon"]
        assert p["beta_ctrl_spread_dropping_smallest_step"] < 1.01, p
        assert p["beta_ctrl_spread"] > p[
            "beta_ctrl_spread_dropping_smallest_step"], (
            "the smallest step must be the one that moves; if it were not, the "
            "checkpoint's declared 1e-6 floor would be unobserved here", p)


def test_the_sigma_instrument_returns_exactly_zero_against_itself(art):
    """sigma(exact datum) is the composed run compared with itself.

    Not a floor (W106) -- it is an identity, and a nonzero value would mean the
    composed step is not reproducible, which would invalidate every other sigma
    on the page.
    """
    for key in ("windowns_exposed", "windowns_embedded"):
        asm = _settled(_case(art, key))["solve"]["assembly"]
        assert asm["sigma_all_exact"] == 0.0, (key, asm["sigma_all_exact"])
        assert asm["sigma_lagged"] > 0.0


def test_t_is_checked_as_a_derivative_rather_than_assumed_to_be_one(art):
    """The change in J that T predicts, against the change a march produces."""
    for key in ("windowns_exposed", "windowns_embedded", "windowns_exposed_2c"):
        sv = _settled(_case(art, key))["solve"]
        assert abs(sv["derivative_check"] - 1.0) < 0.01, (key, sv)


# ---------------------------------------------------------------------------
# 5. the negative half
# ---------------------------------------------------------------------------


def test_the_channel_carries_the_correction(art):
    """The monolith's own datum, in the control's own coordinates, reduces sigma.

    This is what separates 'the seam cannot be corrected from here' from 'this
    objective does not find the correction'.  With the widened control the
    reference datum reaches essentially the floor a perfect control could reach.
    """
    wide = _settled(_case(art, "windowns_exposed_2c"))["solve"]["assembly"]
    lag = wide["sigma_lagged"]
    floor = wide["sigma_seam_exact"]
    ref = wide["sigma_reference_control"]
    assert floor < lag, "a perfect datum on this seam must have something to buy"
    assert ref < lag
    # the reference control gets within a few percent of the floor
    assert (ref - floor) / (lag - floor) < 0.05, (floor, ref, lag)


def test_minimising_the_objective_drives_sigma_the_wrong_way(art):
    """And the tier says so, because the favourable half would be misread.

    Gate 0 passes, T is a derivative to 0.1%, and the map is full rank -- and the
    unregularised solve still moves sigma up by an order of magnitude.  Under the
    most favourable reading the data supports -- the best point over THREE
    regularisation families, selected on sigma itself rather than on J -- it
    captures a fraction of what a perfect datum on the same seam would.
    """
    for key in ("windowns_exposed", "windowns_embedded", "windowns_exposed_2c"):
        asm = _settled(_case(art, key))["solve"]["assembly"]
        assert asm["solved_over_lagged"] > 2.0, (key, asm["solved_over_lagged"])
        assert asm["reduction_captured"] < asm["reduction_available"], (
            key, asm)
    wide = _settled(_case(art, "windowns_exposed_2c"))["solve"]["assembly"]
    assert wide["reduction_captured"] < 0.5 * wide["reduction_available"], wide


def test_the_solved_control_overshoots_the_true_datum(art):
    """`probed-dtn-coupling` 2.3's own diagnostic, on the new objective.

    Flux balance's solve moved the trace 41x / 75x / 150x past the truth.  This
    one overshoots too, which is the honest reason the tier does not claim an
    optimiser is buildable.
    """
    for key in ("windowns_exposed", "windowns_embedded", "windowns_exposed_2c"):
        sv = _settled(_case(art, key))["solve"]
        assert sv["overshoot"] > 5.0, (key, sv["overshoot"])


# ---------------------------------------------------------------------------
# 6. what the write-up may not claim
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.path.isfile(_PAGE), reason="the page is not here")
def test_the_page_does_not_claim_more_than_the_measurement():
    """Four claims this tier is forbidden to make, checked against the prose.

    Not that neural operators are composable -- this attacks sigma and gamma and
    `tau` is untouched.  Not that Pi = 1 is a failure -- it is a certificate
    becoming unavailable.  Not that any of it is faster.  Not that R12's global
    projection and this are the same move.
    """
    with open(_PAGE, encoding="utf-8") as fh:
        text = fh.read().lower()
    # Vocabulary with no legitimate disclaiming use. A bare word-ban on
    # "composable" would have banned the DISCLAIMER too, which is how a prose
    # test starts writing worse prose than it prevents -- so the required list
    # below carries the disclaimers verbatim instead.
    for bad in ("breakthrough", "unprecedented", "revolutionary", "guaranteed",
                "green seam", "certified seam", "solves the coupling problem",
                "speedup", "speed-up"):
        assert bad not in text, f"the page uses the word {bad!r}"
    for needed in ("does not make neural operators composable",
                   "tau", "not faster", "admit-uncertified",
                   "is not a failure"):
        assert needed in text, f"the page never says {needed!r}"


@pytest.mark.skipif(not os.path.isfile(_PAGE), reason="the page is not here")
def test_the_headline_numbers_on_the_page_are_re_derivable_from_the_artifact(art):
    """Every figure below is FORMATTED from `out/w149/w149.json` and then looked
    for in the prose, so a page that drifted from its own artifact fails here
    rather than being believed.  The renderings are the ones the page uses.
    """
    with open(_PAGE, encoding="utf-8") as fh:
        page = fh.read()

    g0 = art["gate0"]["by_column"]
    want = [f"{r:.4f}" for r in g0["exposed"]["ratios"]]
    want += [f"{r:.4f}" for r in g0["embedded"]["ratios"]]

    pos = _settled(_case(art, "poseidon"))
    emb = _settled(_case(art, "windowns_embedded"))
    exp = _case(art, "windowns_exposed")
    want += [
        f"{pos['kappa']:.2f}",                     # 4.48
        f"{emb['kappa']:.1f}",                     # 19.4
        f"{min(s['kappa'] for s in exp['states']):.2f}",   # 1.95
        f"{emb['null']['overlap_with_constant_modes']:.4f}",
        f"{emb['bottom_gap_ratio']:.3f}",
        f"{art['controls']['trajectory']['max_separation'] * 100:.1f}",
    ]
    wide = _settled(_case(art, "windowns_exposed_2c"))["solve"]["assembly"]
    want += [
        f"{wide['sigma_reference_control'] / wide['sigma_lagged']:.3f}",
        f"{wide['sigma_seam_exact'] / wide['sigma_lagged']:.3f}",
    ]

    missing = [w for w in want if w not in page]
    assert not missing, (
        "the page quotes numbers the artifact no longer carries, or the artifact "
        f"moved and the page did not: {missing}")
