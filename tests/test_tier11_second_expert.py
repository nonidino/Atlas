"""Tier 11: what a second and a third expert found -- W61, W56 and the rules.

W55 stood open from 2026-08-28's first session: every constant in the vault was
measured against `reference.WindowNS`, and *"a rule that only holds for one
solver's internals is not a rule"*.  Two more experts were put through the same
stack -- `reference.ChannelNS`, a second discretization, and Poseidon-T, a frozen
20.8M-parameter neural operator -- and the rules held.  **What did not hold was a
rule's implementation**, and it was found by the first expert honest enough to
declare that it did not know its own time discretization.

Every test here is written against the rule and the algebra rather than against
either expert, so the file runs without the build repo checked out and without
torch.  The measured numbers quoted in the docstrings come from
`scripts/w55_second_expert.py`; see `tier0-measurements` §11.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas import (
    ADMIT_UNCERTIFIED,
    Agent,
    BCChannel,
    CaseGraph,
    Connection,
    Decomposition,
    Differentiable,
    Direction,
    EllipticSubsolve,
    ExpertCapabilities,
    MeasuredConstants,
    PortType,
    TimeDiscretization,
    compile_scheme,
)
from atlas.capability import linear_response, port_decl
from atlas.scheme import Transmission

M = 8


def _caps(name, td, **kw):
    """An agent that CAN support probed-DtN, so R2 wants to lift the rung."""
    rng = np.random.default_rng(3)
    A = rng.standard_normal((M, M)) * 0.05
    A = A + A.T + 2.0 * np.eye(M)
    d = dict(
        expert_id=name,
        ports=[port_decl(name=f"{f}:MECH", port_type=PortType.MECH, geometry=f,
                         direction=Direction.BIDIRECTIONAL,
                         nondim={"stress": 1.0, "velocity": 1.0,
                                 "power_area": 1.0},
                         effective_resolution=M)
               for f in ("xlo", "xhi")],
        bc_channel=BCChannel.ROBIN,
        differentiable=Differentiable.NONE,
        dt_native=1e-2,
        governing_family="g",
        boundary_response=linear_response(A, np.full(M, 0.05)),
        elliptic_subsolve=EllipticSubsolve.NONE,
        time_discretization=td,
        stencil_radius=2,
        substeps_per_macro_step=1,
        storage=lambda u: 0.5 * float(np.dot(np.ravel(u), np.ravel(u))),
        validity=lambda state, cond=None: True,
    )
    d.update(kw)
    return ExpertCapabilities(**d)


def _pou():
    """Two subdomains of an 8-cell line, convex, declaring their contamination."""
    from atlas.assembly import PartitionOfUnity

    n = 8
    R1, R2 = np.eye(n)[:5], np.eye(n)[3:]
    pou = PartitionOfUnity(
        n, {"A": R1, "B": R2},
        {"A": np.array([1.0, 1.0, 1.0, 0.6, 0.3]),
         "B": np.array([0.4, 0.7, 1.0, 1.0, 1.0])})
    pou.contaminated = {"A": np.array([0, 0, 0, 1, 1], bool),
                        "B": np.array([1, 1, 0, 0, 0], bool)}
    return pou


def _graph(td, **kw):
    return CaseGraph(
        name="t",
        agents=[Agent(a, _caps(a, td, **kw)) for a in ("A", "B")],
        connections=[Connection(seam_id="s", a=("A", "xhi:MECH"),
                                b=("B", "xlo:MECH"), port_type=PortType.MECH,
                                derive_space=True, expected_null_dim=0)],
        decomposition=Decomposition.OVERLAPPING,
        overlap_cells=8, macro_dt=1e-2,
    )


# ---------------------------------------------------------------------------
# W61 -- an undecided premise is not a favourable one
# ---------------------------------------------------------------------------


def test_w61_unknown_time_discretization_holds_the_rung_down():
    """**The bug a learned expert found.**

    R2b was written to stop R2 lifting the rung to probed-DtN for an agent that
    poses no boundary-value problem over a macro-step -- because §5 measured that
    scheme at **8.9x worse than doing nothing** with beta, kappa, the passivity
    spectrum and the probe residual all reading healthy.  For ``explicit`` it
    corrected the rung.  For ``unknown`` it emitted a decertification whose text
    ended *"It is not assumed to"* -- **and left the rung lifted**, which is
    assuming it.

    The whole vault had never noticed because `window_ns` and `channel_ns` both
    declare ``explicit``, and the two fixtures got their lift from ``unknown``
    being the field's DEFAULT.  Poseidon-T is the first expert for which
    ``unknown`` is the honest answer: a frozen one-shot map is neither explicit
    nor implicit.
    """
    r = compile_scheme(_graph(TimeDiscretization.UNKNOWN))
    assert r.scheme.transmission is Transmission.DIRICHLET
    assert r.scheme.decomposition is Decomposition.OVERLAPPING
    hit = [d for d in r.decisions.decertifications if d.rule == "R2b/W46"]
    assert hit, "the uncertainty must still be recorded, not silently resolved"
    assert "held at dirichlet" in hit[0].message


def test_w61_explicit_still_corrects_the_rung_as_an_admit():
    """The two branches differ, and the difference is the whole point.

    With ``explicit`` we KNOW probed-DtN is wrong, so correcting it is an
    `admit`.  With ``unknown`` we know only that it is unestablished, so
    correcting it is conservative and the claim stays uncertified.
    """
    from atlas.verdict import ADMIT

    r = compile_scheme(_graph(TimeDiscretization.EXPLICIT))
    assert r.scheme.transmission is Transmission.DIRICHLET
    hit = [d for d in r.decisions.of_verdict(ADMIT) if d.rule == "R2b/W46"]
    assert hit
    assert not [d for d in r.decisions.decertifications if d.rule == "R2b/W46"]


def test_w61_implicit_still_lifts_the_rung():
    """The fix must not disable R2. An implicit macro-step DOES pose a BVP."""
    r = compile_scheme(_graph(TimeDiscretization.IMPLICIT))
    assert r.scheme.transmission is Transmission.PROBED_DTN
    assert r.scheme.decomposition is Decomposition.NON_OVERLAPPING


def test_w61_one_unknown_agent_holds_the_whole_graphs_rung_down():
    """The rung is decided over all agents, so the weakest declaration governs.

    That is not new -- `achievable_rung` has always been a minimum over the
    capability records -- but it is now visible in a second field, and it is why
    declaring `time_discretization` on *every* agent of the wind-farm fixture was
    needed rather than only on the fluid ones.
    """
    g = CaseGraph(
        name="t",
        agents=[Agent("A", _caps("A", TimeDiscretization.IMPLICIT)),
                Agent("B", _caps("B", TimeDiscretization.UNKNOWN))],
        connections=[Connection(seam_id="s", a=("A", "xhi:MECH"),
                                b=("B", "xlo:MECH"), port_type=PortType.MECH,
                                derive_space=True, expected_null_dim=0)],
        decomposition=Decomposition.OVERLAPPING, overlap_cells=8, macro_dt=1e-2)
    r = compile_scheme(g)
    assert r.scheme.transmission is Transmission.DIRICHLET


def test_w61_the_rung_decision_is_the_only_thing_that_changed():
    """A graph that declares its time discretization is unaffected by W61.

    Guards against the fix having been a behaviour change rather than a bug fix:
    everything an `explicit` graph does must be what it did before.
    """
    r = compile_scheme(_graph(TimeDiscretization.EXPLICIT))
    assert r.verdict is ADMIT_UNCERTIFIED
    assert r.scheme.decomposition is Decomposition.OVERLAPPING
    assert len(r.seam_operators) == 1


# ---------------------------------------------------------------------------
# W60 -- the elliptic signature, for an expert whose internals are unknown
# ---------------------------------------------------------------------------


def test_r10_already_challenges_an_incompressible_agent_declaring_no_elliptic_part():
    """The guard that partly pre-empts W60, asserted so it is not lost.

    `EllipticSubsolve` has no ``unknown``, so a black-box checkpoint's record
    cannot be honest.  What the compiler does have is a **plausibility check**:
    an incompressible solver almost always contains a pressure solve, so
    declaring ``none`` alongside an incompressible `governing_family` is
    challenged rather than believed.
    """
    r = compile_scheme(_graph(TimeDiscretization.EXPLICIT,
                              governing_family="incompressible-navier-stokes-2d",
                              elliptic_subsolve=EllipticSubsolve.NONE))
    hit = [d for d in r.decisions.decertifications if d.rule == "R10"]
    assert hit
    assert "almost always contains a pressure solve" in hit[0].message


def test_elliptic_signature_separates_the_two_calibration_points():
    """W60's measurement: classify a black box from the probe alone.

    §8.3 left two calibration points on the same instrument, same basis, same
    state -- `WindowNS` with its elliptic part embedded (kappa 21.73, asymmetry
    0.153) and exposed (kappa 1.196, asymmetry 0.002).  A statistic that cannot
    separate those two is not a statistic.

    Measured on Poseidon-T: kappa 9.41, asymmetry 0.272 -- and the asymmetry is
    what decides it.
    """
    from atlas.cases.poseidon import elliptic_signature

    exposed = np.diag([1.196, 1.0]) + 0.001 * np.array([[0.0, 1.0], [-1.0, 0.0]])
    embedded = np.diag([21.73, 1.0]) + 0.9 * np.array([[0.0, 1.0], [-1.0, 0.0]])
    assert "EXPOSED" in elliptic_signature(exposed)["verdict"]
    assert "EMBEDDED" in elliptic_signature(embedded)["verdict"]
    assert elliptic_signature(np.zeros((2, 2)))["verdict"] == "empty operator"


def test_elliptic_signature_says_indeterminate_rather_than_guessing():
    """Between the calibration points it must decline, not interpolate."""
    from atlas.cases.poseidon import elliptic_signature

    mid = np.diag([5.0, 1.0]) + 0.05 * np.array([[0.0, 1.0], [-1.0, 0.0]])
    assert "INDETERMINATE" in elliptic_signature(mid)["verdict"]


# ---------------------------------------------------------------------------
# W55 -- a constant measured on one expert is not another expert's
# ---------------------------------------------------------------------------


def test_a_second_expert_may_not_inherit_the_first_ones_constants():
    """`channel_ns.build` and `poseidon.build` default `measured` to None.

    Not an oversight and not a TODO: `C_mu = 1.2` is what sixteen configurations
    of `WindowNS` measured, and handing it to another solver is the W56 bug
    committed on purpose.  The compile must decertify until this expert's own
    numbers exist -- which for Poseidon-T is **permanent**, because `tau` and
    `sigma` need a same-class monolith and the checkpoint is fixed at 128x128.
    """
    import inspect

    from atlas.cases import channel_ns, poseidon

    for mod in (channel_ns, poseidon):
        sig = inspect.signature(mod.build)
        assert sig.parameters["measured"].default is None, mod.__name__


def test_the_unmeasured_backstop_catches_a_borrowed_constant_being_absent():
    """W56, on a graph with no constants at all: the verdict cannot be `admit`."""
    r = compile_scheme(_graph(TimeDiscretization.EXPLICIT))
    assert r.verdict is ADMIT_UNCERTIFIED
    hit = [d for d in r.decisions.decertifications if d.rule == "W56"]
    assert hit and r.unmeasured
    assert set(hit[0].evidence["unmeasured"]) == set(r.unmeasured)


def test_declaring_another_experts_constants_is_expressible_and_visible():
    """The record carries provenance, which is what makes borrowing detectable.

    Nothing stops a case study declaring `C_mu = 1.2` on a different solver --
    the compiler cannot know. What it can do is carry `probe_state`, `scheme` and
    `depth` beside the number, so the borrowing is on the artefact rather than in
    somebody's head. This asserts the provenance survives into the decision.
    """
    m = MeasuredConstants(L=0.98, tau=1e-6, sigma=1e-8, gamma=0.0, norm_A=1.0,
                          C_mu=1.2, cut_defect_bound=2.6e-7,
                          cut_defect_bound_form="chi-weighted",
                          probe_state="BORROWED from WindowNS",
                          scheme="split-step", depth=0)
    g = _graph(TimeDiscretization.EXPLICIT)
    g.measured = m
    # L2/C2 reaches its `admit` branch -- the one that records provenance -- only
    # on an overlapping graph with a convex partition of unity, so the graph has
    # to carry one for this assertion to be about provenance rather than about a
    # missing declaration.
    g.partition_of_unity = _pou()
    r = compile_scheme(g)
    hit = [d for d in r.decisions if d.rule == "C2" and d.evidence]
    assert hit
    assert hit[0].evidence["provenance"][0] == "BORROWED from WindowNS"
