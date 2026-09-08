"""Tier 33 -- W153, the interaction horizon and the weighted envelope.

`scripts/w153_influence_envelope.py` and `out/w153/w153.json`.  Nine groups:

  * **the instrument** -- the reused march still matches `WindowAgent.respond`
    bitwise, and a periodic agent's influence field is exactly zero.  Live, and
    if either fails nothing else here means anything;
  * **Gate A** -- `sigma` re-measured reproduces the record exactly, `Pi_w <= Pi`
    everywhere, `Pi_w == Pi` EXACTLY where the cutoff is hard, and the re-fitted
    constant's spread is no worse than the indicator's;
  * **Gate B** -- the local explicit agent's influence is bitwise zero at its
    declared `rho * s` and the embedded one's does not decay at all;
  * **the rank-one claim, falsified** -- asserted rather than described, because
    the page rests on its being false and a later reader will want the check;
  * **Gate C** -- Poseidon-T has no finite horizon, and its globality is
    structurally unlike an embedded elliptic solve's;
  * **the envelope** -- `Pi` is 1 for every agent at this cadence and `Pi_w` is
    not, which is the whole of what the tier buys;
  * **the pin** -- it buys a few percent, not a collapse;
  * **W152** -- the ring declaration reads back, and the prediction it came with
    is confirmed from Tier 32's own artifact;
  * **the prose** -- the page does not claim more than the measurement.

Artifact-backed tests skip when `out/w153/w153.json` is absent.  The live tests
cost a few seconds; nothing here re-runs the measurement.
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ARTIFACT = os.path.join(_ROOT, "out", "w153", "w153.json")
_SCRIPT = os.path.join(_ROOT, "scripts", "w153_influence_envelope.py")
_PAGE = os.path.join(_ROOT, "wiki", "concepts", "Atlas 0.1", "common",
                     "interaction-horizon.md")
_STATE = os.path.join(_ROOT, "out", "tier0_verify", "s0_state.npz")


@pytest.fixture(scope="module")
def art():
    if not os.path.isfile(_ARTIFACT):
        pytest.skip("out/w153/w153.json is not here")
    with open(_ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def w153():
    if not os.path.isfile(_SCRIPT):
        pytest.skip("scripts/w153_influence_envelope.py is not here")
    spec = importlib.util.spec_from_file_location("_w153", _SCRIPT)
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
    c = art.get("gateB", {}).get("cases", {}).get(key)
    if c is None:
        pytest.skip(f"gate B case {key!r} is not in the artifact")
    return c["states"][-1]


# ---------------------------------------------------------------------------
# 1. the instrument
# ---------------------------------------------------------------------------


def test_the_reused_march_still_matches_the_published_flux_response(w153, state):
    """Tier 33 reuses Tier 32's `Side` rather than reimplementing the ring.

    Reuse is only safe while the thing reused still behaves, so the bitwise
    control is re-run here rather than inherited on trust.
    """
    u, v, fx = state
    out = w153._w149.check_flux_agrees(u, v, fx)
    assert out["bitwise_equal"], out
    assert out["max_abs_difference"] == 0.0


def test_a_periodic_agent_has_an_influence_field_that_is_exactly_zero(w153, state):
    """The positive control, live, with a known answer.

    A periodic window has no boundary to impose a datum on, so its influence is
    zero -- not small.  This is the only measurement that can catch a probe
    manufacturing an operator out of roundoff or an off-by-one in the ring.
    """
    u, v, fx = state
    side, faces, tiling, k = w153._spectral_case()(u, v, fx)
    Jc, _lab = w153.influence_jacobian(side, faces, 1e-2, m=4)
    assert np.all(Jc == 0.0), f"max |J| = {np.abs(Jc).max()}"
    pr = w153.profile_of(Jc, w153.b_local(tiling, k))
    assert pr["peak"] == 0.0
    assert side.n_calls >= 2 * 4 + 1, "the probe must actually have run"


# ---------------------------------------------------------------------------
# 2. Gate A
# ---------------------------------------------------------------------------


def test_sigma_is_re_measured_and_reproduces_the_record_exactly(art):
    """The recorded rows become a positive control on the harness.

    `out/l6/w49_sigma.json` was an unchecked input to every quotation of
    ``C_mu``.  Re-deriving it with the same harness and getting the same bits is
    what licenses re-fitting the constant against a different envelope.
    """
    g = art["gateA"]
    assert g["recorded_rows_reproduced"] >= 10
    assert g["sigma_reproduction_worst"] == 0.0, g["sigma_reproduction_worst"]


def test_the_weighted_envelope_never_exceeds_the_indicator(art):
    """``Pi_w <= Pi`` is a property of the construction, not a hope."""
    for r in art["gateA"]["rows"]:
        assert r["Pi_w"] <= r["Pi"] * (1.0 + 1e-12), r


def test_the_refinement_returns_the_indicator_where_the_cutoff_is_hard(art):
    """At halo 1 the overlap is narrower than the domain of dependence.

    Nothing decays inside it, so the measured sensitivity and the indicator must
    agree EXACTLY.  This is the property that lets `Pi_w` be adopted without
    re-auditing a single past verdict, so it is asserted exactly rather than to
    a tolerance.
    """
    rows = [r for r in art["gateA"]["rows"] if r["halo"] == 1]
    assert rows, "the halo-1 configuration is not in the artifact"
    for r in rows:
        assert r["Pi"] == 1.0
        assert r["Pi_w"] == r["Pi"], (r["Pi_w"], r["Pi"])


def test_gate_a_passes_the_constant_is_no_worse_behaved(art):
    """A constant that moves less while the quantity spans 4.3e4 is more of one."""
    g = art["gateA"]
    assert g["passes"] is True
    assert g["C_mu_weighted"]["spread"] <= g["C_mu_indicator"]["spread"]
    assert g["sigma_span"] > 1e4, "the sweep must still span what it used to"


# ---------------------------------------------------------------------------
# 3. Gate B -- the reduction check
# ---------------------------------------------------------------------------


def test_the_local_agent_reduces_to_its_declared_domain_of_dependence(art):
    """**The single most important assertion in the tier.**

    An explicit agent with stencil radius `rho` and `s` sub-steps per exchange
    cannot move information further than `rho * s` cells, so its influence must
    be zero there -- and it is, bitwise, not approximately.  If this failed,
    `Pi_w` would not be a refinement of `Pi` and nothing else would follow.
    """
    from atlas.cases import window_ns as W
    d = W.STENCIL_RADIUS * W.SUBSTEPS
    st = _case(art, "windowns_exposed")
    assert st["d_eff"]["0.0"] == d, (st["d_eff"], d)
    assert st["profile_max"][d] == 0.0
    assert st["far_field"]["exactly_zero"] is True
    assert st["far_field"]["n_far_cells"] > 1000


def test_the_embedded_agent_does_not_decay_at_all(art):
    """R10's "flat in distance from the cut", measured rather than derived.

    Same solver as the test above, same state, same probe -- one difference.
    """
    st = _case(art, "windowns_embedded")
    for theta in ("0.1", "0.01", "0.001", "1e-06", "1e-10", "0.0"):
        assert st["d_eff"][theta] is None, (theta, st["d_eff"])
    assert st["profile_max"][60] > 0.05
    assert st["far_field"]["exactly_zero"] is False


def test_the_periodic_control_is_zero_in_the_artifact_too(art):
    st = _case(art, "spectral_periodic")
    assert st["peak"] == 0.0
    assert st["far_field"]["exactly_zero"] is True


# ---------------------------------------------------------------------------
# 4. the rank-one claim, falsified
# ---------------------------------------------------------------------------


def test_the_non_decaying_part_is_not_rank_one(art):
    """The derivation the tier was designed around, and it is false.

    The claim was that a boundary mode decays like ``exp(-pi k r / L)`` so every
    mode decays except ``k = 0``, making the flat part the rank-one gauge mode.
    The constant mode IS the largest single contributor -- and it is not the
    whole of it.  Asserted because the page rests on this being false.
    """
    ff = _case(art, "windowns_embedded")["far_field"]
    assert ff["effective_rank"] > 2.0, ff["effective_rank"]
    assert ff["gap_s1_over_s2"] < 1.5, (
        "a rank-one far field would show a large gap", ff["gap_s1_over_s2"])
    assert ff["energy_in_mode_1"] < 0.75, ff["energy_in_mode_1"]
    pm = _case(art, "windowns_embedded")["per_mode"]
    assert pm["far_energy_fraction_in_constant_mode"] > 0.3, (
        "the gauge mode must still be the largest single contributor", pm)
    assert pm["modes_within_10x_of_largest_far"] > 1, pm


def test_the_two_mode_families_decay_by_different_laws(art):
    """Even-about-the-midpoint decays exponentially; odd decays algebraically.

    The exponential family is the strip estimate working.  The algebraic one is
    what the derivation does not predict, and it is the reason the flat part is
    not rank one.
    """
    rows = [r for r in _case(art, "windowns_embedded")["per_mode"]["rows"]
            if r["face"] == "xhi"]
    cos = sorted([r for r in rows if r["parity"] == "cos"],
                 key=lambda r: r["wavenumber"])
    sin = sorted([r for r in rows if r["parity"] == "sin"],
                 key=lambda r: r["wavenumber"])
    assert len(cos) >= 5 and len(sin) >= 5
    # the exponential family falls by orders across the sweep; the algebraic one
    # falls by less than one order over the same wavenumbers
    assert cos[0]["far_over_near"] / cos[4]["far_over_near"] > 100.0, cos
    assert sin[0]["far_over_near"] / sin[4]["far_over_near"] < 20.0, sin
    # and at every shared wavenumber the odd family reaches further
    for a, b in zip(cos, sin):
        if a["wavenumber"] == b["wavenumber"]:
            assert b["far_over_near"] > a["far_over_near"], (a, b)


# ---------------------------------------------------------------------------
# 5. Gate C -- Poseidon-T, and what W60's field cannot say
# ---------------------------------------------------------------------------


def test_poseidon_has_no_finite_interaction_horizon(art):
    g = art.get("gateC")
    if not g or "unreachable" in g:
        pytest.skip("gate C is not in the artifact")
    for st in g["states"]:
        for theta in ("0.1", "0.01", "0.001", "1e-06", "1e-10", "0.0"):
            assert st["d_eff"][theta] is None, (st["probe_state"], theta)
        assert st["profile_max"][20] > 0.3
        assert st["far_field"]["exactly_zero"] is False


def test_poseidons_globality_is_not_elliptic_in_character(art):
    """Both agents are global; they are global by different mechanisms.

    An embedded elliptic solve is gauge-dominated.  The checkpoint is not, by an
    order of magnitude in effective rank and by a factor of seven in the
    constant mode's share -- which is why a field that names a MECHANISM could
    never be declared for it, and why what the rules consume is REACH.
    """
    g = art.get("gateC")
    if not g or "unreachable" in g:
        pytest.skip("gate C is not in the artifact")
    pos = g["states"][-1]["far_field"]
    ell = _case(art, "windowns_embedded")["far_field"]
    assert pos["effective_rank"] > 4.0 * ell["effective_rank"], (
        pos["effective_rank"], ell["effective_rank"])
    pos_share = g["states"][-1]["per_mode"]["far_energy_fraction_in_constant_mode"]
    ell_share = _case(art, "windowns_embedded")["per_mode"][
        "far_energy_fraction_in_constant_mode"]
    assert pos_share < 0.25 < ell_share, (pos_share, ell_share)


def test_the_horizon_is_stable_across_three_probe_states(art):
    """`positive-controls-need-a-horizon`, applied to a measurement about horizons."""
    tr = art["controls"]["trajectory"]
    assert tr["max_separation"] > 0.05, tr
    for r in tr["states"]:
        assert r["inside_declared_envelope"] is True, r
    g = art.get("gateC")
    if g and "unreachable" not in g:
        ranks = [s["far_field"]["effective_rank"] for s in g["states"]]
        assert max(ranks) / min(ranks) < 1.10, ranks


# ---------------------------------------------------------------------------
# 6. the envelope, and 7. the pin
# ---------------------------------------------------------------------------


def test_the_indicator_cannot_tell_the_agents_apart_and_the_measurement_can(art):
    """The whole of what the tier buys, in one assertion.

    At one exchange per macro-step the domain of dependence covers the overlap,
    so `Pi` is 1 for everything and the certificate is unavailable.  `Pi_w` ranks
    the three agents in the order the physics says.
    """
    side = art.get("side")
    if not side:
        pytest.skip("the side experiment is not in the artifact")
    rows = {(r["agent"], r["variant"]): r for r in side["rows"]}
    for r in side["rows"]:
        assert r["Pi"] == 1.0, ("the indicator must be vacuous here", r)
    ex = rows[("WindowNS", "exposed")]["Pi_w"]
    em = rows[("WindowNS", "embedded")]["Pi_w"]
    assert ex < em < 1.0, (ex, em)
    if ("Poseidon-T", "as declared") in rows:
        po = rows[("Poseidon-T", "as declared")]["Pi_w"]
        assert em < po < 1.0, (
            "the checkpoint must sit above the embedded elliptic solve and "
            "still below the vacuous value", ex, em, po)


def test_pinning_one_scalar_buys_a_few_percent_and_not_a_collapse(art):
    """The practical consequence of the rank-one claim being false."""
    side = art.get("side")
    if not side:
        pytest.skip("the side experiment is not in the artifact")
    assert 0.0 < side["pin_buys"] < 0.25, side["pin_buys"]
    assert side["pin_collapses_the_gap"] is False, side
    assert side["gap_to_exposed_after"] > 0.8 * side["gap_to_exposed_before"], side


# ---------------------------------------------------------------------------
# 8. W152, and the confound
# ---------------------------------------------------------------------------


def test_the_ring_declaration_is_present_and_undeclared_stays_the_default():
    """W152's fields, on `ResponseHalf`'s template.

    `UNDECLARED` by default matters: a record that has not thought about it must
    not be read as claiming the narrow answer, and `actuated_components` returns
    `None` rather than 1 so a caller has to notice.
    """
    from atlas.capability import PortDecl
    from atlas.ports import PortType, RingActuation

    bare = PortDecl(name="xhi:MECH", port_type=PortType.MECH)
    assert bare.actuation is RingActuation.UNDECLARED
    assert bare.actuated_components is None
    pair = PortDecl(name="x", port_type=PortType.MECH, ring_components=2,
                    actuation=RingActuation.CONJUGATE_PAIR)
    assert pair.actuated_components == 1
    full = PortDecl(name="x", port_type=PortType.MECH, ring_components=2,
                    actuation=RingActuation.FULL_RING)
    assert full.actuated_components == 2


def test_the_fluid_case_studies_declare_two_carried_and_one_driven(art):
    ports = art["w152"]["declared"]
    assert ports
    for p in ports:
        assert p["ring_components"] == 2
        assert p["actuated_components"] == 1
        assert p["actuation"] == "conjugate-pair"


def test_a_richer_control_space_makes_the_objective_worse(art):
    """W152's prediction, recorded before it was checked, and confirmed.

    A control that reaches more of the jump gives the optimiser more room to
    absorb the agents' defect difference, so `J`'s pathology should grow.  It
    does -- and that is W150's theorem showing its face.
    """
    jp = art["w152"].get("j_pathology")
    if not jp:
        pytest.skip("out/w149/w149.json is not here")
    assert jp["full_ring"]["control_dim"] == 2 * jp["declared_pair"]["control_dim"]
    assert jp["full_ring"]["controllable_fraction"] > 0.9
    assert jp["declared_pair"]["controllable_fraction"] < 0.5
    assert jp["worse_with_full_ring"] is True
    assert jp["worse_by"] > 2.0, jp["worse_by"]
    assert jp["full_ring"]["overshoot"] > jp["declared_pair"]["overshoot"]


def test_the_tier_32_venue_was_looser_than_the_scheme_it_names(art):
    """The confound, settled -- and it cuts against the objection, not for it."""
    c = art["confound"]
    if not c.get("available"):
        pytest.skip("out/w149/w149.json is not here")
    assert c["carried_by"].startswith("exposed"), c["carried_by"]
    assert c["exposed_venue_over_published_split_step"] > 5.0, c
    assert abs(c["embedded_venue_over_published_as_built"] - 1.0) < 0.02, (
        "the embedded row must reproduce the published as-built sigma, which is "
        "what validates the instrument for free", c)


# ---------------------------------------------------------------------------
# 9. what the page may not claim
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.path.isfile(_PAGE), reason="the page is not here")
def test_the_page_does_not_claim_more_than_the_measurement():
    with open(_PAGE, encoding="utf-8") as fh:
        text = fh.read().lower()
    for bad in ("breakthrough", "unprecedented", "revolutionary", "guaranteed",
                "green seam", "certified seam", "solves the coupling problem",
                "speedup", "speed-up"):
        assert bad not in text, f"the page uses the word {bad!r}"
    for needed in ("does not make neural operators composable",
                   "is not a new bound", "r10 is right", "tau",
                   "nothing here is faster", "admit-uncertified"):
        assert needed in text, f"the page never says {needed!r}"


@pytest.mark.skipif(not os.path.isfile(_PAGE), reason="the page is not here")
def test_the_headline_numbers_are_re_derivable_from_the_artifact(art):
    """Formatted from the artifact and then looked for in the prose."""
    with open(_PAGE, encoding="utf-8") as fh:
        page = fh.read()
    g = art["gateA"]
    side = {(r["agent"], r["variant"]): r for r in art["side"]["rows"]}
    want = [
        f"{g['C_mu_indicator']['spread']:.2f}",
        f"{g['C_mu_weighted']['spread']:.2f}",
        f"{side[('WindowNS', 'exposed')]['Pi_w']:.3f}",
        f"{side[('WindowNS', 'embedded')]['Pi_w']:.3f}",
        f"{_case(art, 'windowns_embedded')['far_field']['effective_rank']:.2f}",
    ]
    if ("Poseidon-T", "as declared") in side:
        want.append(f"{side[('Poseidon-T', 'as declared')]['Pi_w']:.3f}")
    g4 = art.get("gateC")
    if g4 and "unreachable" not in g4:
        want.append(f"{g4['states'][-1]['far_field']['effective_rank']:.2f}")
    missing = [w for w in want if w not in page]
    assert not missing, (
        "the page quotes numbers the artifact no longer carries, or the artifact "
        f"moved and the page did not: {missing}")
