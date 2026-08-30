"""Tier 15 -- W74's class, W75, W76, W77.

Tier 14 fixed W74's instance (the probe linearized about the zero trace) and
left the class open: nothing detected that a port spec's own choice of bond had
invalidated a probe assumption.  Chasing the class found the field was on the
wrong OBJECT -- `probe_base` is per expert, and an interface state is per seam --
and the same instrument then inverted W75 and closed W77.

As in Tier 14, a rule is tested against the algebra it comes from wherever that
is possible, and only then through a case study.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from atlas.capability import (
    EllipticSubsolve,
    ExpertCapabilities,
    TimeDiscretization,
    linear_response,
    port_decl,
)
from atlas.composition import certify_substitution
from atlas.ports import PortType, ResponseHalf
from atlas.probe import (
    BASE_SENSITIVITY_FLOOR,
    SUPPORT_GLOBAL_FRACTION,
    SupportReach,
    base_disagreement,
    base_sensitivity,
    support_reach,
)
from atlas.transfer import InterfaceSpace, identity_prolongation
from atlas.verdict import ADMIT, ADMIT_UNCERTIFIED, REFUSE

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

N = 32


def _caps(respond, **kw):
    base = dict(
        expert_id="t",
        ports=[port_decl(name="p:MECH", port_type=PortType.MECH,
                         nondim={"velocity": 1.0, "traction": 1.0, "power_area": 1.0},
                         response_half=ResponseHalf.EFFORT,
                         effective_resolution=N)],
        substeps_per_macro_step=1,
        stencil_radius=1,
        boundary_response=respond,
        dt_native=1.0,
        governing_family="test",
    )
    base.update(kw)
    return ExpertCapabilities(**base)


def _banded(n, half):
    """A matrix with a finite domain of dependence: exact zeros past `half`."""
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - half), min(n, i + half + 1)):
            A[i, j] = 1.0 / (1 + abs(i - j))
    return A


def _dense_decay(n, ell):
    """A dense matrix: every entry nonzero, like the inverse of a sparse SPD."""
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return np.exp(-np.abs(i - j) / ell)


# ---------------------------------------------------------------------------
# W75 -- the support gate, against the algebra
# ---------------------------------------------------------------------------


def test_a_banded_response_is_not_global_and_a_dense_one_is():
    """The gate, at the two ends it is defined by.

    An implicit macro-step inverts a sparse SPD matrix and the inverse of a
    sparse SPD matrix is DENSE; an explicit march's response is exactly zero
    past `radius * substeps`.  Nothing here is calibrated -- the distinction is
    arithmetic zero against arithmetic nonzero.
    """
    band = support_reach(linear_response(_banded(N, 3)), "p", np.zeros(N))
    assert band.nonzero == 7
    assert band.reach == 3
    assert not band.is_global
    assert band.consistent_with == ("exposed", "none")

    dense = support_reach(linear_response(_dense_decay(N, 8.0)), "p", np.zeros(N))
    assert dense.nonzero == N
    assert dense.fraction == 1.0
    assert dense.is_global
    assert dense.consistent_with == ("embedded",)


def test_the_support_gate_reads_a_lower_bound_not_a_measurement_of_compactness():
    """The one way it fails, and it fails in the conservative direction.

    A genuinely dense response whose far tail UNDERFLOWS reads as compact.  This
    is not hypothetical: the same implicit shell reads 48/48 at dt = 0.05 s and
    42/48 at the matched clock dt = 1e-4 s, where sqrt(alpha dt) is 60x below one
    seam cell.  So only the GLOBAL reading is a positive measurement.
    """
    A = _dense_decay(N, 0.015)         # exp(-16/0.015) underflows; exp(-1/0.015) does not
    r = support_reach(linear_response(A), "p", np.zeros(N))
    assert not np.all(A == 0.0)        # the matrix really is dense as written
    assert r.nonzero < N               # and the probe cannot see that
    assert not r.is_global


def test_the_support_fraction_floor_sits_between_the_measured_calibration_points():
    """0.986 (embedded) against 0.210 (exposed), on window_ns's two modes."""
    assert 0.210 < SUPPORT_GLOBAL_FRACTION < 0.986


def test_support_reach_of_a_zero_response_is_empty_rather_than_global():
    r = support_reach(lambda _p, t: np.zeros_like(t), "p", np.zeros(N))
    assert r.peak == 0.0
    assert r.nonzero == 0
    assert not r.is_global


# ---------------------------------------------------------------------------
# W74's class -- base sensitivity, and the disagreement across a seam
# ---------------------------------------------------------------------------


def test_an_affine_response_has_no_base_dependence_at_all():
    """The check whose PASSING promotes: if the response is affine the base is
    provably free and every base-related decertification goes away."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((N, N))
    P = np.eye(N)
    for base in (np.zeros(N), np.full(N, 400.0), rng.standard_normal(N)):
        r = base_sensitivity(linear_response(A, bias=rng.standard_normal(N)),
                             "p", base, P, 1e-3, shift=0.5)
        assert r["affine"]
        assert r["relative_change"] < BASE_SENSITIVITY_FLOOR


def test_a_reciprocal_response_is_base_dependent_and_that_is_the_therm_bond():
    """PORT_SPECS[THERM] pairs (T, q_n/T) so effort x flow is a power.

    Dividing by T is exactly what makes the response nonlinear in the effort --
    the pseudo-bond (T, q_n) would have been affine.  Two correct choices with
    incompatible hidden assumptions, which is W74's mechanism stated in four
    lines of arithmetic.
    """
    def q_over_T(_port, trace):
        return 1.0e4 / np.maximum(np.asarray(trace, float), 1.0)

    def q_plain(_port, trace):
        return 5.0 * np.asarray(trace, float)

    P = np.eye(N)
    nonlinear = base_sensitivity(q_over_T, "p", np.full(N, 400.0), P, 1e-3, shift=0.01)
    affine = base_sensitivity(q_plain, "p", np.full(N, 400.0), P, 1e-3, shift=0.01)
    assert not nonlinear["affine"]
    assert nonlinear["relative_change"] > 1e-3
    assert affine["affine"]


def test_base_disagreement_is_exact_when_the_two_sides_agree():
    b = np.full(8, 371.9667)
    chk = base_disagreement({"gas": b, "shell": b.copy()})
    assert chk["consistent"]
    assert chk["spread"] == 0.0


def test_base_disagreement_catches_two_sides_linearized_at_different_states():
    """thermal_seam's actual defect, in the algebra.

    The gas declares the wall temperature it sees (400 K) and the shell the gas
    temperature it sees (900 K).  Each record is honest; they are 500 K apart on
    ONE interface variable, and the sum of two Jacobians taken at different
    points is not a Jacobian.
    """
    chk = base_disagreement({"gas": np.full(8, 400.0), "shell": np.full(8, 900.0)})
    assert not chk["consistent"]
    assert chk["spread"] == pytest.approx(500.0 * np.sqrt(8), rel=1e-12)
    assert chk["relative_spread"] == pytest.approx(1.25, rel=1e-12)


def test_a_single_sided_seam_cannot_disagree_with_itself():
    assert base_disagreement({"only": np.zeros(4)})["consistent"]


# ---------------------------------------------------------------------------
# W76 -- what the assembly hides, and the substitution blind spot
# ---------------------------------------------------------------------------


def test_the_assembled_operator_can_be_perfect_while_a_block_is_singular():
    """Non-sufficiency, constructed so it is a fact and not a measurement.

    No function of ``S_A + S_B`` can recover ``beta(S_A)``: the two matrices
    below sum to the identity, whose every diagnostic is ideal, and one of them
    is singular to 1e-12.
    """
    SA = np.diag([1.0, 1e-12])
    SB = np.diag([0.0, 1.0])
    assert np.allclose(SA + SB, np.diag([1.0, 1.0 + 1e-12]))
    assert np.linalg.svd(SA + SB, compute_uv=False)[-1] == pytest.approx(1.0, abs=1e-9)
    assert np.linalg.svd(SA, compute_uv=False)[-1] == pytest.approx(1e-12, rel=1e-6)


def test_a_substitution_is_blind_exactly_when_the_block_is_below_the_margin():
    """The inequality, which is where the rule comes from.

    A replacement that under-responds moves the seam by at most the swapped
    agent's own block, so if ``||S_i|| < beta - beta_min`` no such failure can
    be caught -- including an expert that ignores its boundary data entirely.
    """
    caps = _caps(linear_response(np.eye(N)))
    S = np.eye(4)
    for block_norm, beta, beta_min, expect in (
        (0.0696, 0.3486, 0.05, True),     # window_ns sx0, the one-sided seam
        (0.0696, 0.3486, 0.30, False),
        (0.1860, 0.2773, 0.05, True),     # sy0, balanced, still blind at low beta_min
        (0.1860, 0.2773, 0.10, False),
    ):
        c = certify_substitution(
            agent_id="a", old_caps=caps, new_caps=caps, S_old=S, S_new=S,
            beta=beta, beta_min=beta_min, block_norm=block_norm,
        )
        assert c.blind is expect, (block_norm, beta_min)


def test_a_blind_pass_is_downgraded_from_admit_to_uncertified():
    """A certificate that could not have failed must not be reported as a pass."""
    caps = _caps(linear_response(np.eye(N)))
    S = np.eye(4)
    blind = certify_substitution(
        agent_id="a", old_caps=caps, new_caps=caps, S_old=S, S_new=S,
        beta=0.3486, beta_min=0.05, block_norm=0.0696)
    assert blind.passes and blind.verdict is ADMIT_UNCERTIFIED
    seeing = certify_substitution(
        agent_id="a", old_caps=caps, new_caps=caps, S_old=S, S_new=S,
        beta=0.3486, beta_min=0.05, block_norm=0.9)
    assert seeing.passes and seeing.verdict is ADMIT


def test_an_unsupplied_block_norm_leaves_the_old_behaviour_untouched():
    """Back-compat is not an accident here: every pre-2026-08-29 caller omits it."""
    caps = _caps(linear_response(np.eye(N)))
    c = certify_substitution(agent_id="a", old_caps=caps, new_caps=caps,
                             S_old=np.eye(4), S_new=np.eye(4),
                             beta=0.3, beta_min=0.05)
    assert c.blind is None
    assert c.verdict is ADMIT


# ---------------------------------------------------------------------------
# W75 -- the field test, on synthetic experts with known internals
# ---------------------------------------------------------------------------


def _elliptic_test(caps):
    from atlas.conformance import _test_elliptic_subsolve

    space = InterfaceSpace(seam_id="s", dim=N)
    prol = identity_prolongation("t", "p:MECH", N)
    return _test_elliptic_subsolve(caps, space, prol)


def test_a_global_response_declared_none_is_refused():
    """'none' switches R10 off, so this is the silent-wrongness case."""
    t = _elliptic_test(_caps(linear_response(_dense_decay(N, 8.0)),
                             elliptic_subsolve=EllipticSubsolve.NONE))
    assert t.verdict is REFUSE
    assert t.silent_if_false


def test_a_global_response_declared_embedded_is_corroborated():
    t = _elliptic_test(_caps(linear_response(_dense_decay(N, 8.0)),
                             elliptic_subsolve=EllipticSubsolve.EMBEDDED))
    assert t.verdict is ADMIT


def test_a_compact_response_declared_exposed_is_corroborated():
    t = _elliptic_test(_caps(linear_response(_banded(N, 3)),
                             elliptic_subsolve=EllipticSubsolve.EXPOSED))
    assert t.verdict is ADMIT


def test_a_compact_response_declared_embedded_is_uncertified_not_refused():
    """The conservative direction: L2 refuses on 'embedded' anyway, and the
    fraction is a lower bound, so a contradiction would overclaim."""
    t = _elliptic_test(_caps(linear_response(_banded(N, 3)),
                             elliptic_subsolve=EllipticSubsolve.EMBEDDED))
    assert t.verdict is ADMIT_UNCERTIFIED


def test_unknown_is_resolved_by_the_reach_which_is_what_w60_asked_for():
    """W68 refused to promote from the SPECTRUM; this promotes from the reach.

    The difference is not strictness: reach measures the property the enum is
    defined by -- an infinite domain of dependence -- while kappa and the
    asymmetry measure a correlate that comes apart for a self-adjoint operator.
    """
    t = _elliptic_test(_caps(linear_response(_dense_decay(N, 8.0)),
                             elliptic_subsolve=EllipticSubsolve.UNKNOWN))
    assert t.verdict is ADMIT
    assert "RESOLVES" in t.message
    assert "embedded" in t.message


def test_the_field_test_declines_when_no_delta_can_be_imposed():
    t = _elliptic_test(_caps(lambda _p, tr: np.zeros_like(tr),
                             elliptic_subsolve=EllipticSubsolve.NONE))
    assert t.verdict is ADMIT_UNCERTIFIED


# ---------------------------------------------------------------------------
# the case studies -- the rules on the real objects
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def window_split():
    from atlas import compile_scheme
    from atlas.cases import window_ns as W

    st = np.load(os.path.join(_ROOT, "out", "tier0b", "s0_state.npz"))
    g, _ = W.build(st["u"], st["v"], mode="split-step")
    return compile_scheme(g, probe_state="s0")


def test_window_ns_vertical_seams_are_one_sided_and_the_horizontal_ones_are_not(
        window_split):
    """The measurement L4/block-share fires on, and the contrast it rests on."""
    r = window_split
    for sid in ("sx0", "sx1"):
        op = r.seam_operators[sid]
        worst = max(b.kappa for b in op.blocks.values())
        assert worst / op.kappa > 1000.0
        assert op.one_sided < 0.20
        assert float(np.median(op.mode_shares)) > 0.95
    for sid in ("sy0", "sy1"):
        op = r.seam_operators[sid]
        worst = max(b.kappa for b in op.blocks.values())
        assert worst / op.kappa < 2.0
        assert op.one_sided > 0.45
        assert float(np.median(op.mode_shares)) < 0.60


def test_block_share_decertifies_the_vertical_seams_only(window_split):
    fired = {d.subject for d in window_split.decisions._decisions
             if d.rule == "block-share" and d.verdict is ADMIT_UNCERTIFIED}
    assert fired == {"sx0", "sx1"}


def test_window_ns_split_step_still_has_no_refusals(window_split):
    """The composition is untouched: sigma and L rest on the assembled beta,
    which is sound. What Tier 15 costs this case is the clean `admit`, and it
    costs it for a measured reason rather than a cautious one."""
    assert not [d for d in window_split.decisions._decisions if d.verdict is REFUSE]


def test_the_two_sides_of_the_thermal_seam_are_linearized_500_kelvin_apart():
    from atlas import compile_scheme
    from atlas.cases import thermal_seam as T

    g, _ = T.build(mode="split-step", clocks="matched")
    r = compile_scheme(g, probe_state="duct, T_hot=900 K, T_wall=400 K")
    op = r.seam_operators["cht"]
    assert not op.base_check["consistent"]
    assert op.base_check["relative_spread"] == pytest.approx(1.25, rel=1e-6)
    means = {a: b.base_summary[0] for a, b in op.blocks.items()}
    assert means["gas"] == pytest.approx(400.0)
    assert means["shell"] == pytest.approx(900.0)
    assert op.probe_state_agrees is False
    assert "INCONSISTENT" in op.derived_probe_state
    assert [d for d in r.decisions._decisions if d.rule == "probe-base"]


def test_the_derived_probe_state_is_physical_and_not_a_fourier_coefficient():
    """A 400 K uniform trace reduces to 11.18 in a measure-weighted Fourier
    basis. M is the right space to COMPARE in and the wrong one to read."""
    from atlas import compile_scheme
    from atlas.cases import thermal_seam as T

    g, _ = T.build(mode="split-step", clocks="matched")
    op = compile_scheme(g, probe_state="x").seam_operators["cht"]
    assert "gas@400" in op.derived_probe_state
    assert "shell@900" in op.derived_probe_state


def test_a_seam_base_overrides_both_sides_and_makes_them_consistent():
    """The structural fix: an interface state belongs to a SEAM, not an expert."""
    from atlas import compile_scheme
    from atlas.cases import thermal_seam as T
    from atlas.probe import assemble_seam

    g, _ = T.build(mode="split-step", clocks="matched")
    conn = g.connections[0]
    transfer = compile_scheme(g, probe_state="x").transfers["cht"]
    P = transfer.prolongations["gas"]
    lam = np.linalg.lstsq(P.matrix, np.full(P.matrix.shape[0], 371.9667), rcond=None)[0]
    op = assemble_seam(g, conn, transfer, seam_base=lam)
    assert op.base_check["consistent"]
    for b in op.blocks.values():
        assert b.base_summary[0] == pytest.approx(371.9667, rel=1e-6)


def test_the_reported_beta_is_not_attainable_at_any_admissible_base():
    """The sharpest form of the W74 class defect.

    The interface temperature is trapped between the reservoirs, and over that
    whole interval a COMMON base gives beta in [0.4200, 1.7689].  The probe
    reports 0.3757, which is below the entire range -- so it is not the value at
    the wrong point, it is the value at no point.
    """
    from atlas.cases import thermal_seam as T

    gas, shell = T.GasAgent(dt=T.DT_GAS), T.ShellAgent(dt=T.DT_SHELL)
    P = T.fourier_basis(T.N_SEAM, T.M_EFF)

    def block(agent, port, base, eps=1e-3):
        f0 = np.asarray(agent.respond(port, base), float).ravel()
        cols = [(np.asarray(agent.respond(port, base + eps * P[:, j]), float).ravel()
                 - f0) / eps for j in range(P.shape[1])]
        return P.T @ (T.H_SEAM * np.stack(cols, axis=1))

    def beta(S):
        return float(np.linalg.svd(S, compute_uv=False)[-1])

    mismatched = beta(block(gas, "wall:THERM", gas.base_trace())
                      + block(shell, "inner:THERM", shell.base_trace()))
    at = []
    for lam in (T.T_OUT, 400.0, 650.0, T.T_HOT):
        b = np.full(T.N_SEAM, lam)
        at.append(beta(block(gas, "wall:THERM", b) + block(shell, "inner:THERM", b)))
    assert mismatched < min(at)
    assert mismatched == pytest.approx(0.3757, rel=2e-3)
    assert min(at) == pytest.approx(0.4200, rel=2e-3)


def test_the_spectral_signature_ranks_the_known_pair_backwards_at_long_cadence():
    """W75's negative result, and the reason the gate had to change instruments.

    Same shell, two solvers, everything else fixed, at a cadence where W68's
    precondition is satisfied: the EXPLICIT solver reads the higher kappa.
    """
    from atlas.cases import thermal_seam as T
    from atlas.cases.poseidon import elliptic_signature

    P = T.fourier_basis(T.N_SEAM, T.M_EFF)
    kap = {}
    for expose in (False, True):
        ag = T.ShellAgent(dt=100.0, expose_elliptic=expose)
        base = np.full(T.N_SEAM, 371.9667)
        f0 = np.asarray(ag.respond("inner:THERM", base), float).ravel()
        cols = [(np.asarray(ag.respond("inner:THERM", base + 1e-3 * P[:, j]),
                            float).ravel() - f0) / 1e-3 for j in range(P.shape[1])]
        S = P.T @ (T.H_SEAM * np.stack(cols, axis=1))
        kap["explicit" if expose else "implicit"] = elliptic_signature(S)["kappa"]
    assert kap["explicit"] > kap["implicit"]


def test_the_support_gate_gets_every_known_declaration_right():
    """Five points, two of them one shell under two solvers with all else fixed."""
    from atlas.cases import thermal_seam as T
    from atlas.cases import window_ns as W

    lam = np.full(T.N_SEAM, 371.9667)
    assert support_reach(T.ShellAgent(dt=5e-2).respond, "inner:THERM", lam).is_global
    assert not support_reach(
        T.ShellAgent(dt=5e-2, expose_elliptic=True).respond, "inner:THERM", lam).is_global
    gas = T.GasAgent(dt=T.DT_GAS)
    assert not support_reach(gas.respond, "wall:THERM", gas.base_trace()).is_global

    st = np.load(os.path.join(_ROOT, "out", "tier0b", "s0_state.npz"))
    for mode, expect in (("as-built", True), ("split-step", False)):
        _g, experts = W.build(st["u"], st["v"], mode=mode)
        ag = experts["W00"]
        assert support_reach(ag.respond, "xhi", np.zeros(ag.n)).is_global is expect


def test_the_w78_script_reproduces():
    """From cold, and it writes the artifact the wiki section cites."""
    import subprocess

    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
    out = os.path.join(_ROOT, "out", "w78_test")
    p = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "scripts", "w78_base_blocks_and_reach.py"),
         "--out", out, "--stages", "base", "state"],
        capture_output=True, text=True, env=env, timeout=1800,
    )
    assert p.returncode == 0, p.stderr[-3000:]
    assert "NO -- it is below the entire range" in p.stdout
    assert "gas@400; shell@900 INCONSISTENT" in p.stdout
