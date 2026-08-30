"""Tier 12: the topology items -- W6, W33/W63, and what the port algebra misses.

Three rows had stood open with the same shape: *the machinery exists and has
never been pointed at anything real*.

    W6    a cross-point rule, written and conditional on the decomposition, with
          a definition of done -- *"a test on a 4-agent junction showing beta
          does not collapse"* -- that had never been run on any graph
    W33   a conformance suite, every field's test implemented, never run on a
          real expert
    W5    ADVEC passengers, a ROT port and a field-to-lumped seam, fixture-tested
          only

Running them produced two live defects (**W63**, **W66**) and one rule
interaction (**W65**), and every one of the three was found by pointing existing
machinery at a real object rather than by building anything new.

Written against the rules and the algebra, so the file runs without the build
repo.  The measured numbers in the docstrings come from
`scripts/w6_cross_point.py`, `scripts/w33_conformance.py` and
`atlas/cases/wind_farm_real.py`; see `tier0-measurements` §12.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas.capability import (
    BCChannel,
    ClaimType,
    Differentiable,
    Direction,
    EllipticSubsolve,
    ExpertCapabilities,
    MotionClass,
    TimeDiscretization,
    linear_response,
    port_decl,
)
from atlas.conformance import run_conformance
from atlas.ports import PortType, check_scales
from atlas.transfer import InterfaceSpace, identity_prolongation
from atlas.verdict import ADMIT, ADMIT_UNCERTIFIED, REFUSE

M = 8


# ---------------------------------------------------------------------------
# W63 -- the conformance verdict on `storage` belongs to the seam
# ---------------------------------------------------------------------------


def _expert(name, response, **kw):
    d = dict(
        expert_id=name,
        ports=[port_decl(name="p:MECH", port_type=PortType.MECH, geometry="p",
                         direction=Direction.BIDIRECTIONAL,
                         nondim={"stress": 1.0, "velocity": 1.0,
                                 "power_area": 1.0},
                         effective_resolution=M, motion_class=MotionClass.STATIC)],
        bc_channel=BCChannel.DIRICHLET,
        differentiable=Differentiable.NONE,
        elliptic_subsolve=EllipticSubsolve.EXPOSED,
        time_discretization=TimeDiscretization.EXPLICIT,
        stencil_radius=2, substeps_per_macro_step=1,
        dt_native=1e-2, governing_family="g",
        boundary_response=response,
        claim_types=frozenset({ClaimType.TRAJECTORY}),
    )
    d.update(kw)
    return ExpertCapabilities(**d)


def test_w63_the_block_and_the_seam_can_disagree_and_the_seam_decides():
    """The measured case: block defect 3.218e-03, seam defect 0.

    Poseidon-T's `xhi` block is not passive and the seam it assembles into is.
    Before W63 the suite refused the record on the block; the theory page had
    said since W48 that the verdict belongs to the seam.
    """
    A = np.diag([2.0, 1.0, -0.5] + [1.0] * (M - 3))
    caps = _expert("c", linear_response(A), storage=lambda u: 1.0)
    args = (InterfaceSpace("s", dim=M), identity_prolongation("c", "p:MECH", M))

    on_block = run_conformance(caps, *args)
    on_seam = run_conformance(caps, *args, seam_defect=0.0)
    refused = run_conformance(caps, *args, seam_defect=0.5)

    def storage(c):
        return [t for t in c.tests if t.field_name == "storage"][0]

    assert storage(on_block).verdict is ADMIT_UNCERTIFIED
    assert storage(on_seam).verdict is ADMIT
    assert storage(refused).verdict is REFUSE


def test_w33_a_false_bc_channel_label_is_refused_at_admission():
    """The W33 row's definition of done, on the smallest possible example.

    An expert with no boundary channel, declaring one. Nothing else in the
    package can tell the two records apart -- that is `plug-in-composition-
    theorems` §3's whole argument -- and this refuses it.
    """
    from atlas.capability import zero_response

    caps = _expert("liar", zero_response, bc_channel=BCChannel.DIRICHLET)
    cert = run_conformance(caps, InterfaceSpace("s", dim=M),
                           identity_prolongation("l", "p:MECH", M))
    bad = [t for t in cert.contradictions if t.field_name == "bc_channel"]
    assert bad and bad[0].silent_if_false
    assert cert.verdict is REFUSE


def test_w33_the_same_expert_declaring_none_is_confirmed():
    """The control. `bc_channel = none` is a true statement about it."""
    from atlas.capability import zero_response

    caps = _expert("honest", zero_response, bc_channel=BCChannel.NONE)
    cert = run_conformance(caps, InterfaceSpace("s", dim=M),
                           identity_prolongation("h", "p:MECH", M))
    bc = [t for t in cert.tests if t.field_name == "bc_channel"][0]
    assert bc.verdict is ADMIT


# ---------------------------------------------------------------------------
# W66 -- the scale set is checked and the callable is not
# ---------------------------------------------------------------------------


ADVEC_SCALES = {
    "enthalpy": 1.0, "mass_flux": 1.0, "power_area": 1.0,
    "h0_effort": 4.0, "h0_flow": 2.0, "h0_power": 8.0,
}


def test_w66_check_scales_validates_the_declaration_not_the_callable():
    """**The gap, stated as an assertion.**

    `check_scales` reads the `nondim` dictionary and confirms
    ``s_e * s_f = s_P``. It never calls `boundary_response`, so a port whose
    callable returns the *power* where the scale set says *effort* passes every
    check in the package.

    Measured consequence, on `cases/wind_farm_real.py`: returning ``(u.n) h0``
    instead of ``h0`` from an ADVEC port made **E7 fail** with a passivity defect
    of 3.915e-02 on the assembled seam `e5a`, and E7 stamps `holds` the moment
    the pairing is made conjugate. The number was real and it was measuring the
    wrong thing -- a **false alarm** rather than a false pass, which is the
    silent-wrongness class inverted and no better for it.
    """
    assert ADVEC_SCALES["h0_effort"] * ADVEC_SCALES["h0_flow"] == \
        ADVEC_SCALES["h0_power"]

    port = port_decl(name="f:ADVEC", port_type=PortType.ADVEC, geometry="f",
                     direction=Direction.BIDIRECTIONAL, nondim=dict(ADVEC_SCALES),
                     passengers=("h0",), effective_resolution=M,
                     motion_class=MotionClass.STATIC)
    # The declaration is complete and power-consistent...
    chk = check_scales(PortType.ADVEC, port.nondim, port.passengers)
    assert chk.ok, chk

    # ...and its inputs are the declaration and nothing else. `check_scales`
    # takes a port type, a scale dictionary and a passenger tuple; it has no
    # access to the callable and the record has no field naming which half of the
    # conjugate pair the callable returns.
    import inspect

    params = set(inspect.signature(check_scales).parameters)
    assert params == {"port_type", "scales", "passengers"}
    assert not hasattr(port, "response_variable")


def test_w66_a_non_conjugate_pairing_is_not_detectable_from_the_operator():
    """Why a check is needed rather than a smarter diagnostic.

    Scaling the response by any positive function of the trace's own magnitude
    leaves ``S`` symmetric-positive or not, essentially at random, and there is
    no property of ``S`` alone that says which physical variable produced it.
    So the pairing has to be **declared**; it cannot be inferred.
    """
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, M))
    A = A @ A.T + np.eye(M)                       # a passive (SPD) operator
    assert np.linalg.eigvalsh(0.5 * (A + A.T))[0] > 0.0

    # the same operator with one row rescaled -- what a wrong flow variable does
    D = np.diag(np.linspace(0.1, 3.0, M))
    B = D @ A
    assert np.linalg.eigvalsh(0.5 * (B + B.T))[0] < 0.0
    # both are "a matrix the probe assembled"; nothing about either says which
    # physical quantity its rows are in
    assert A.shape == B.shape


# ---------------------------------------------------------------------------
# W65 -- R10's plausibility guard and an algebraic closure
# ---------------------------------------------------------------------------


def test_w65_r10_challenges_an_algebraic_closure_that_is_telling_the_truth():
    """Two rules that are each right, interacting on one record.

    An `ActuatorDisk` is a closed-form zero-parameter law, so
    ``elliptic_subsolve = none`` is **true**. It must declare
    ``governing_family = "incompressible-navier-stokes-2d"`` or E3 fails at every
    rotor face and tau goes UNDEFINED there -- `CASE-STUDY-GUIDE` says so
    explicitly, *"put there after getting it wrong once"*. R10's guard challenges
    exactly that pair.

    The guard is right about solvers and this is not a solver, so the
    decertification is a false positive that the guide's own instruction walks
    every lumped closure into. Asserted here so the interaction is recorded
    rather than rediscovered.
    """
    from atlas import Agent, CaseGraph, Connection, Decomposition, compile_scheme

    caps = _expert("disk", linear_response(np.eye(M) * 0.4),
                   elliptic_subsolve=EllipticSubsolve.NONE,
                   governing_family="incompressible-navier-stokes-2d",
                   storage=lambda u: 1.0, validity=lambda s, c=None: True)
    other = _expert("flow", linear_response(np.eye(M) * 0.5),
                    elliptic_subsolve=EllipticSubsolve.NONE,
                    governing_family="incompressible-navier-stokes-2d",
                    storage=lambda u: 1.0, validity=lambda s, c=None: True)
    g = CaseGraph(
        name="t", agents=[Agent("disk", caps), Agent("flow", other)],
        connections=[Connection(seam_id="s", a=("disk", "p:MECH"),
                                b=("flow", "p:MECH"), port_type=PortType.MECH,
                                derive_space=True, expected_null_dim=0)],
        decomposition=Decomposition.OVERLAPPING, overlap_cells=8, macro_dt=1e-2)
    r = compile_scheme(g)
    hit = [d for d in r.decisions.decertifications if d.rule == "R10"]
    assert hit
    assert "almost always contains a pressure solve" in hit[0].message
    # and it is a decertification rather than a refusal, which is the part that
    # keeps it survivable: the record is challenged, not rejected
    assert r.verdict is not REFUSE


# ---------------------------------------------------------------------------
# W6 -- the cross-point, as a property of the interface space
# ---------------------------------------------------------------------------


def test_w6_a_shared_cell_is_multivalued_under_per_seam_truncation():
    """**W6's real defect, and it is not a beta collapse.**

    Each seam carries its own multipliers over its own ring, so a cell lying on
    two seams gets two values -- and they differ because each seam truncates a
    *different* function to the same number of modes. Measured on the four-window
    junction: 1.6% of the trace scale in the streamwise component and **66.7%**
    in the transverse one, where the two seams disagree about the sign.

    It is a property of the **declared interface space**, not of the expert, so
    no better probe removes it and every expert on a given tiling has the same
    one. Reproduced here in miniature, with no solver at all.
    """
    n, m = 64, 8
    x = (np.arange(n) + 0.5) / n
    B = np.column_stack([np.ones(n)] + [
        f(2 * np.pi * k * x) * np.sqrt(2.0)
        for k in range(1, m) for f in (np.cos,)][:m - 1])

    # two seams crossing at one cell, carrying two different functions that
    # happen to agree there
    shared = 17
    f1 = np.sin(3 * np.pi * x) + 0.3 * np.cos(11 * np.pi * x)
    f2 = np.exp(-((x - x[shared]) ** 2) / 0.02)
    f2 = f2 - f2[shared] + f1[shared]              # equal AT the shared cell
    assert f1[shared] == pytest.approx(f2[shared])

    r1 = B @ np.linalg.lstsq(B, f1, rcond=None)[0]
    r2 = B @ np.linalg.lstsq(B, f2, rcond=None)[0]
    # after truncation they no longer agree there
    spread = abs(r1[shared] - r2[shared])
    scale = float(np.linalg.norm(f1) / np.sqrt(n))
    assert spread / scale > 1e-3, "truncation must break the agreement"


def test_w6_the_primal_constraint_is_what_removes_it():
    """Declaring a single-valued corner DOF: the constraint, as linear algebra.

    Requiring ``k`` seams to agree at one cell is ``k - 1`` constraints, and the
    primal multiplier space is the orthogonal complement of their span. This
    asserts the dimension count the treatment costs -- 3 for a four-seam junction
    -- which is what `scripts/w6_cross_point.py` applies to the real operator.
    """
    dim, k = 64, 4
    rng = np.random.default_rng(1)
    V = rng.standard_normal((k, dim))              # each row: one seam's value map
    D = np.vstack([V[i] - V[i + 1] for i in range(k - 1)])
    Q, _r = np.linalg.qr(D.T)
    P = np.eye(dim) - Q @ Q.T
    assert Q.shape[1] == k - 1 == 3
    assert np.linalg.matrix_rank(P, tol=1e-10) == dim - (k - 1)
    # and on the primal space the k values genuinely agree
    w = P @ rng.standard_normal(dim)
    vals = V @ w
    assert np.allclose(vals, vals[0], atol=1e-9)
