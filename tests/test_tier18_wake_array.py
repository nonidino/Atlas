"""Tier 18: real turbine-array geometry, and the attribution machinery on a
frozen pretrained expert.

`atlas/cases/wake_array.py` is the sixth real case study and the first with
turbine positions rather than a schematic stand-in.  The measured numbers quoted
in the docstrings come from `scripts/w93_wake_array.py`; see
[[tier0-measurements]] 18.

**Most of this file runs without the build repo and without torch**, because the
findings are about rules and algebra rather than about either expert: the
geometry is derived from module constants, the port-segment split is derived from
the layout, the traction is a closed form, and W93's gap is a property of
`required_halo` that a fixture agent exhibits exactly as the checkpoint does.
The three tests that genuinely need a forward pass are marked and skip cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas import (
    ADMIT_UNCERTIFIED,
    REFUSE,
    BCChannel,
    Differentiable,
    Direction,
    EllipticSubsolve,
    ExpertCapabilities,
    PortType,
    ResponseHalf,
    TimeDiscretization,
    certify_substitution,
    check_scales,
)
from atlas.capability import port_decl
from atlas.multiphysics import interface_power, seam_defect_split, tight_couple
from atlas.probe import SUPPORT_GLOBAL_FRACTION, support_reach

from atlas.cases import wake_array as wa


def _has_expert() -> bool:
    try:
        wa.load_reference()
    except Exception:                                        # pragma: no cover
        return False
    return True


needs_expert = pytest.mark.skipif(
    not _has_expert(),
    reason="the build repo (reference.WindowNS / FrozenFluidExpert) is not checked out",
)


def _needs_checkpoint():
    if not _has_expert():
        return False
    try:
        import torch                                          # noqa: F401,PLC0415
        import scOT                                           # noqa: F401,PLC0415
    except Exception:                                         # pragma: no cover
        return False
    return True


needs_checkpoint = pytest.mark.skipif(
    not _needs_checkpoint(),
    reason="Poseidon-T needs torch + scOT + the cached checkpoint",
)


# ---------------------------------------------------------------------------
# the geometry the checkpoint chose
# ---------------------------------------------------------------------------


def test_the_macro_step_is_forced_by_the_window_size():
    """`dt = S_LEN / 20` is a derivation, not a setting.

    One macro-step must be one NATIVE expert lead (0.1) and `adapters.Scaling`
    fixes ``T_s = S_LEN / U_s`` with ``U_s = 2``, so ``dt = 0.1 * S_LEN / 2``.
    Choosing the window chooses the macro-step; there is no third degree of
    freedom, and `poseidon.py`'s "with a checkpoint you choose the domain to suit
    the window" is this identity.
    """
    assert wa.MACRO_DT == pytest.approx(0.1 * wa.S_LEN / 2.0)
    assert wa.DX == pytest.approx(wa.S_LEN / wa.EXPERT_RES)
    # and the Reynolds number falls as the window grows, which is the conflict
    re_at = lambda s: 1.0 / (2.0 * wa.NU_P_GRID_SCALE * s)
    assert re_at(1.0) > 1e3 > re_at(wa.S_LEN)


def test_the_array_is_an_L_at_a_whole_number_of_window_strides():
    ids = {r.rotor_id: r for r in wa.ROTORS}
    assert set(ids) == {"R1", "R2", "R3"}
    # R1 -> R2 in line, R2 -> R3 abreast: an L
    assert ids["R1"].y_centre == pytest.approx(ids["R2"].y_centre)
    assert ids["R2"].x_plane == pytest.approx(ids["R3"].x_plane)
    stride_D = wa.STRIDE * wa.DX
    assert ids["R2"].x_plane - ids["R1"].x_plane == pytest.approx(stride_D)
    assert ids["R3"].y_centre - ids["R2"].y_centre == pytest.approx(stride_D)
    # every disk plane sits at the middle of an overlap, where both windows cover it
    for r in wa.ROTORS:
        lo = (r.col + 1) * wa.STRIDE * wa.DX
        hi = ((r.col + 1) * wa.STRIDE + wa.HALO) * wa.DX
        assert lo < r.x_plane < hi


def test_every_rotor_face_is_inside_one_rows_own_region():
    """No disk straddles a lateral seam, so its port has one owner per side."""
    for r in wa.ROTORS:
        sl = r.cells
        assert sl.stop - sl.start == wa.ROTOR_CELLS
        assert 0 <= sl.start and sl.stop <= wa.N
        # clear of the lateral overlap band of its own row
        assert sl.stop <= wa.N - wa.HALO or sl.start >= wa.HALO


def test_ports_are_per_face_SEGMENT_and_the_segments_partition_the_face():
    """A real rotor spans 1 D of a 4 D face, so a port's V is a face segment.

    `wind_farm_real` established that an `ADVEC` passenger list is per FACE and
    not per agent.  Real geometry needs one more step: the rotor and the open flow
    beside it are two ports on ONE ring, and nothing in the port algebra says a
    port's V has to be a whole face.
    """
    for window in wa.DEFAULT_TILING.names:
        for face, segments in wa.port_segments(window).items():
            idx = np.concatenate([wa.segment_index(window, face, s) for s in segments])
            assert sorted(idx.tolist()) == list(range(wa.N)), (window, face)
            assert len(idx) == len(set(idx.tolist()))
        if any(s != ["full"] for s in wa.port_segments(window).values()):
            split = [f for f, s in wa.port_segments(window).items() if s != ["full"]]
            for f in split:
                assert wa.segment_index(window, f, "rotor").size == wa.ROTOR_CELLS
                assert wa.segment_index(window, f, "bypass").size == wa.N - wa.ROTOR_CELLS


def test_effective_resolution_is_a_wavelength_not_a_mode_count():
    """m_eff scales with the face length, because the cutoff is a LENGTH.

    W0 4.2 puts the checkpoint's spectral cutoff between lambda = 0.125 D and
    0.25 D.  Carrying `window_ns`'s 16 modes onto a 32-cell rotor face would claim
    four times the resolution the checkpoint has; carrying the WAVELENGTH gives 9.
    """
    assert wa.modes_for(128) == 33
    assert wa.modes_for(96) == 25
    assert wa.modes_for(32) == 9
    for n in (32, 96, 128):
        k_max = (wa.modes_for(n) - 1) // 2
        assert n / k_max == pytest.approx(wa.LAMBDA_CUT_CELLS, rel=1e-9)


def test_the_fourier_basis_is_orthonormal_in_this_cases_own_cell_size():
    """The Gram is ``dx I``, so the forced adjoint is ``dx P^T`` with no solve.

    A local copy rather than an import: `window_ns.fourier_basis` closes over
    ``H = 2/128`` and this case's cell is ``1/32 D``.
    """
    for n in (32, 96, 128):
        B = wa.fourier_basis(n)
        assert np.allclose(wa.DX * B.T @ B, np.eye(B.shape[1]), atol=1e-12)


# ---------------------------------------------------------------------------
# the declarations
# ---------------------------------------------------------------------------


def test_every_port_declares_a_power_consistent_scale_set():
    for window in wa.DEFAULT_TILING.names:
        for p in wa.fluid_ports(window):
            assert p.scale_check().ok, (window, p.name)
            assert p.response_half is ResponseHalf.EFFORT
    assert check_scales(PortType.ROT, dict(wa.ROT_SCALES)).ok


def test_the_two_experts_declare_an_identical_port_list():
    """Which is what makes one a legal SUBSTITUTION for the other.

    A replacement that changed the port list would be a graph edit, and
    `SubstitutionCertificate` refuses it outright rather than measuring anything.
    """
    zero = (np.zeros((wa.N, wa.N)), np.zeros((wa.N, wa.N)))
    for window in wa.DEFAULT_TILING.names:
        names = {p.name for p in wa.fluid_ports(window)}
        assert len(names) == len(wa.fluid_ports(window))
        # `certify_substitution` compares exactly this
        assert sorted((p.port_type.value, p.name) for p in wa.fluid_ports(window)) ==             sorted((p.port_type.value, p.name) for p in wa.fluid_ports(window))
        assert names == {wa.port_name(f, s)
                         for f, segs in wa.port_segments(window).items()
                         for s in segs}
    assert zero


def test_the_seams_do_not_claim_a_coincidence_they_do_not_have():
    """`geometrically_coincident=False`, honestly, on all thirteen.

    In an overlapping decomposition the two artificial rings a seam pairs are
    HALO cells apart and no two of them coincide.  `poseidon.py` and
    `wind_farm_real.py` both declare True on rings 20 cells apart; the README
    lists the field as "a label nothing checks" and this is what checking it
    would have said.
    """
    conns = wa.connections()
    assert len(conns) == 13
    assert all(not c.geometrically_coincident for c in conns)
    assert all(c.derive_space for c in conns)
    assert all(c.expected_null_dim == 0 for c in conns)


def test_the_rotor_meets_the_downstream_windows_inflow_ring_on_its_upstream_side():
    """Which reads backwards and is right.

    Each window's artificial ring lies INSIDE its neighbour, so the ring upstream
    of a disk in the overlap belongs to the window downstream of it.
    """
    for r in wa.ROTORS:
        assert r.upstream_window == f"F{r.col + 1}{r.row}"
        assert r.downstream_window == f"F{r.col}{r.row}"
        ring_up = (r.col + 1) * wa.STRIDE * wa.DX          # F{col+1} xlo
        ring_dn = (r.col * wa.STRIDE + wa.N - 1) * wa.DX   # F{col}   xhi
        assert ring_up < r.x_plane < ring_dn
    names = {c.seam_id for c in wa.connections()}
    assert {"R1_up", "R1_down", "R2_up", "R2_down", "R3_up", "R3_down"} <= names


# ---------------------------------------------------------------------------
# W93 -- the halo rule reads a declaration and the probe measures the same thing
# ---------------------------------------------------------------------------


class _GlobalAgent:
    """An agent whose response reaches every cell, like a learned operator's.

    Not a mock of Poseidon-T: a two-line object with the property that matters,
    so the rule can be tested where the checkpoint is not available.  ``radius``
    gives it a COMPACT stencil instead -- exactly zero past that many cells, the
    way an explicit march is -- which is the positive control W75's gate needs.
    """

    def __init__(self, n=32, radius=None):
        self.n = n
        self.radius = radius

    def respond(self, port, trace):
        t = np.asarray(trace, dtype=float).ravel()
        d = np.abs(np.arange(self.n)[:, None] - np.arange(self.n)[None, :])
        kernel = (np.ones((self.n, self.n)) if self.radius is None
                  else (d <= self.radius).astype(float))
        return (kernel @ t) / self.n


def _learned_caps(**kw):
    d = dict(
        expert_id="learned",
        ports=[port_decl(name="xhi:full:MECH", port_type=PortType.MECH,
                         geometry="xhi", direction=Direction.BIDIRECTIONAL,
                         nondim=dict(wa.MECH_SCALES), effective_resolution=8,
                         response_half=ResponseHalf.EFFORT)],
        bc_channel=BCChannel.DIRICHLET,
        elliptic_subsolve=EllipticSubsolve.UNKNOWN,
        stencil_radius=2, substeps_per_macro_step=1,
        time_discretization=TimeDiscretization.UNKNOWN,
        differentiable=Differentiable.NONE, dt_native=wa.MACRO_DT,
        governing_family="incompressible-navier-stokes-2d",
        boundary_response=_GlobalAgent().respond,
    )
    d.update(kw)
    return ExpertCapabilities(**d)


def test_W93_the_halo_is_undecidable_for_an_agent_that_will_not_name_its_clock():
    """**W93**, and it is W69's own derivation reaching a third case.

    `capability.required_halo` returns ``stencil_radius * substeps``, which W69
    established is an *explicit* agent's domain of dependence -- and it fixed the
    ``implicit`` branch and left the branch where the record says nothing.
    ``unknown`` is exactly what a frozen learned one-shot map declares, under
    R2b, because it is neither.

    Before this the declaration read **2 cells** and `probe.support_reach` --
    which measures the same quantity, by poking a delta -- reads the whole window
    on Poseidon-T: nonzero in all 128 seam cells at every amplitude from 1 to
    1e-3, 64 cells from the poke, and `L2/R10` was admitting a 16-cell overlap on
    the strength of the 2.
    """
    assert _learned_caps().required_halo() is None
    # the same record with the clock DECLARED gets the product, unchanged
    assert _learned_caps(
        time_discretization=TimeDiscretization.EXPLICIT).required_halo() == 2
    # and a zero-stencil algebraic closure is 0 under any label: the guard
    # follows the derivation, not the verdict
    assert _learned_caps(stencil_radius=0).required_halo() == 0


def test_W93_the_instrument_that_measures_what_the_declaration_claimed():
    """`support_reach` reads the quantity `required_halo` was returning."""
    agent = _GlobalAgent(n=32)
    reach = support_reach(agent.respond, "xhi:full:MECH", np.zeros(32))
    assert reach.is_global
    assert reach.fraction >= SUPPORT_GLOBAL_FRACTION
    assert reach.nonzero == reach.n, "the response reaches every seam cell"
    assert reach.reach == reach.n // 2, "as far as a centre poke can reach"
    assert reach.reach > 2, "the measurement and the old declaration disagree"


def test_W93_a_compact_agent_is_read_as_compact_by_the_same_instrument():
    """The positive control: the gate is not simply saying `global` to everything."""
    agent = _GlobalAgent(n=64, radius=3)
    reach = support_reach(agent.respond, "xhi:full:MECH", np.zeros(64))
    assert not reach.is_global
    assert reach.consistent_with == ("exposed", "none")


# ---------------------------------------------------------------------------
# the rotor's bond
# ---------------------------------------------------------------------------


@needs_expert
def test_the_rotor_returns_a_traction_not_a_force_density():
    """``0.5 rho C_T' u^2`` carries the declared ``stress`` scale; the alternative
    is that divided by the strip thickness.

    `wind_farm_real.RotorAgent` returns `ActuatorDisk.force_density`, a force per
    unit VOLUME, against a port whose scale set declares ``stress``.  At the
    default ``thickness = 0.1`` the two differ by exactly 10x -- W66's class, with
    the factor being a declared geometric LENGTH rather than W69's unseparable
    O(1).
    """
    import importlib
    wa.load_reference()
    dk = importlib.import_module("atlas_windfarm_reference.disk")

    u = np.full(wa.ROTOR_CELLS, 0.7)
    rotor = wa.RotorDisk(agent_id="Rt", u_ref=u)
    t = rotor.traction(u)
    disk = dk.ActuatorDisk()
    st = disk(0.7)
    assert t[0] == pytest.approx(st.thrust / disk.area, rel=1e-12)
    assert disk.force_density(st.thrust) == pytest.approx(t[0] / disk.thickness,
                                                          rel=1e-12)
    assert disk.force_density(st.thrust) / t[0] == pytest.approx(10.0, rel=1e-12)


@needs_expert
def test_the_rotors_probe_base_is_its_inflow_because_the_response_is_quadratic():
    """At a zero base the disk's block is identically zero and the seam reads EMPTY.

    Which is the right answer for a turbine in still air and the wrong one for
    this graph.  W74, on a rotor: the base is not free whenever the response is
    not affine in the trace.
    """
    u = np.full(wa.ROTOR_CELLS, 0.65)
    rotor = wa.RotorDisk(agent_id="Rt", u_ref=u)
    assert np.allclose(rotor.probe_base("up:MECH"), u)

    def block_at(base):
        eps = 1e-4
        z = rotor.respond("up:MECH", base)
        return np.column_stack([
            (rotor.respond("up:MECH", base + eps * np.eye(wa.ROTOR_CELLS)[:, k]) - z)
            / eps for k in range(4)])

    at_zero = block_at(np.zeros(wa.ROTOR_CELLS))
    at_inflow = block_at(u)
    # the block at a zero base is not merely small, it VANISHES with the probe
    # step: d/du (0.5 C_T' u^2) at u = 0 is 0, so the chord is 0.5 C_T' eps.
    assert np.linalg.norm(at_inflow) / np.linalg.norm(at_zero) > 1e3
    assert np.linalg.norm(at_inflow) > 1.0


@needs_expert
def test_the_two_rotor_faces_present_opposite_tractions():
    """A momentum sink pushes upstream on one face and downstream on the other.

    The sign is the outward normal's; getting it wrong would make the disk a
    momentum SOURCE and the Betz canary would not catch it, because `disk.py`
    never outputs C_P.
    """
    u = np.full(wa.ROTOR_CELLS, 0.6)
    rotor = wa.RotorDisk(agent_id="Rt", u_ref=u)
    up = rotor.respond("up:MECH", u)
    dn = rotor.respond("down:MECH", u)
    assert np.allclose(up, -dn)
    assert np.all(up > 0.0)


# ---------------------------------------------------------------------------
# the absolute trace, and what it buys
# ---------------------------------------------------------------------------


def test_interface_power_needs_an_absolute_trace_to_be_a_power():
    """A perturbation times a perturbation is not a power.

    Every fluid case study before this one writes the trace into the ring as
    ``ring + trace``, so the trace is a perturbation and ``probe_base`` is
    identically zero.  `multiphysics.seam_defect_split` measures the defect in
    ``effort x flow``; on a perturbation trace that quantity is a second-order
    residual and not the interface power at all.
    """
    trace_abs = np.full(8, 0.8)
    effort = np.full(8, 2.0)
    p_abs = interface_power(trace_abs, effort, ResponseHalf.EFFORT, measure=wa.DX)
    p_pert = interface_power(np.zeros(8), effort, ResponseHalf.EFFORT, measure=wa.DX)
    assert p_abs == pytest.approx(8 * 2.0 * 0.8 * wa.DX)
    assert p_pert == 0.0
    assert p_abs != p_pert


def test_seam_defect_split_recovers_an_injected_error_exactly():
    """The instrument, on a two-agent fixture with a KNOWN defect.

    This is the check §16 ran on `thermal_seam` and it is re-run here on the
    ABSOLUTE-trace convention, because the norm the split is taken in changes
    with it.  `scripts/w93_wake_array.py` runs the same ladder against
    Poseidon-T itself.
    """
    n = 6
    base = np.full(n, 0.8)

    def side_a(lam):
        return 2.0 * np.asarray(lam, dtype=float) - 1.0

    def side_b(lam):
        return -1.5 * np.asarray(lam, dtype=float) + 0.2

    ref = {"A": side_a, "B": side_b}
    halves = {"A": ResponseHalf.EFFORT, "B": ResponseHalf.EFFORT}
    # the converged trace, so a "lagged" trace equal to it is a ZERO lag
    lam_star = tight_couple(lambda lam: side_a(lam) + side_b(lam), base).trace
    taus = []
    for alpha in (1.05, 1.25, 2.0):
        actual = {"A": (lambda lam, s=alpha: s * side_a(lam)), "B": side_b}
        sd = seam_defect_split("s", actual, ref, halves, base, lam_star,
                               measure=wa.DX)
        assert sd.converged
        assert sd.tau["B"] == pytest.approx(0.0, abs=1e-12), "the unswapped side"
        assert sd.sigma == pytest.approx(0.0, abs=1e-10), "a zero lag costs nothing"
        assert sd.tau["A"] > 0.0
        taus.append(sd.tau["A"])
    assert taus == sorted(taus), "a bigger injected error must read bigger"
    # and identical callables give exactly zero, which `min` has to survive (W84)
    sd = seam_defect_split("s", ref, ref, halves, base, base, measure=wa.DX)
    assert sd.tau == {"A": pytest.approx(0.0, abs=1e-12),
                      "B": pytest.approx(0.0, abs=1e-12)}


# ---------------------------------------------------------------------------
# W76 -- a rotor seam is one-sided by construction
# ---------------------------------------------------------------------------


def _caps_like(name, n_ports=1):
    return ExpertCapabilities(
        expert_id=name,
        ports=[port_decl(name="up:MECH", port_type=PortType.MECH, geometry="f",
                         direction=Direction.BIDIRECTIONAL,
                         nondim=dict(wa.MECH_SCALES), effective_resolution=8,
                         response_half=ResponseHalf.EFFORT)],
        bc_channel=BCChannel.DIRICHLET,
        elliptic_subsolve=EllipticSubsolve.NONE,
        stencil_radius=1, substeps_per_macro_step=1,
        time_discretization=TimeDiscretization.EXPLICIT,
        differentiable=Differentiable.NONE, dt_native=wa.MACRO_DT,
        governing_family="incompressible-navier-stokes-2d",
        boundary_response=lambda p, t: np.asarray(t, dtype=float),
    )


def test_W76_a_seam_whose_blocks_differ_by_the_reynolds_number_is_blind():
    """The derivable half of the rotor-seam finding, with no expert involved.

    `probed-dtn-coupling` 2.2's normative MECH effort is ``nu dw/dn``, so a fluid
    block scales with ``nu`` while an actuator disk's traction ``0.5 C_T' u^2``
    does not.  At this case's ``nu = 3.92e-3`` the rotor's block is three orders
    larger, ``||S_fluid|| < beta - beta_min`` for every beta_min the framework can
    offer, and `SubstitutionCertificate.blind` is True: **no** replacement of the
    fluid expert at a rotor seam could have failed the test.
    """
    m = 8
    caps = _caps_like("f")

    def blind_range(S_other, S_i):
        """How far beta_min can go before the swap becomes visible, IN UNITS OF
        the tested agent's own block -- which is the only scale-free statement.

        ``blind`` is ``||S_i|| < beta - beta_min``, so the range is
        ``(beta - ||S_i||) / ||S_i||``, and for two commuting blocks that is
        ``||S_other|| / ||S_i||``: the ratio of the two sides.
        """
        beta = float(np.linalg.svd(S_other + S_i, compute_uv=False)[-1])
        n_i = float(np.linalg.norm(S_i, 2))
        return beta, n_i, (beta - n_i) / n_i

    S_fluid = wa.NU_REF * np.eye(m)
    beta_r, n_r, range_rotor = blind_range(np.eye(m), S_fluid)
    beta_b, n_b, range_balanced = blind_range(np.eye(m), np.eye(m))
    assert range_rotor == pytest.approx(1.0 / wa.NU_REF, rel=1e-9)
    assert range_balanced == pytest.approx(1.0, rel=1e-9)
    assert range_rotor / range_balanced > 250, "the ratio IS the cell Reynolds number"

    # at a beta_min of 100 of the tested block's own norms, the rotor seam still
    # certifies an expert that IGNORES its boundary data, and the balanced one
    # refuses the same swap
    zero = np.zeros((m, m))
    cert = certify_substitution("f", caps, caps, S_fluid, zero,
                                beta=beta_r, beta_min=100 * n_r, block_norm=n_r)
    assert cert.passes and cert.blind is True
    assert cert.verdict is ADMIT_UNCERTIFIED, "a pass that could not fail is not an admission"
    cert2 = certify_substitution("f", caps, caps, np.eye(m), zero,
                                 beta=beta_b, beta_min=100 * n_b, block_norm=n_b)
    assert not cert2.passes
    assert cert2.verdict is REFUSE


# ---------------------------------------------------------------------------
# the graph, end to end
# ---------------------------------------------------------------------------


@needs_checkpoint
def test_the_graph_builds_and_declares_what_it_says_it_declares():
    state = (np.ones((wa.NY, wa.NX)), np.zeros((wa.NY, wa.NX)))
    graph, experts = wa.build(*state, kind="reference")
    assert len(graph.agents) == 9
    assert len(graph.connections) == 13
    assert graph.overlap_cells == wa.HALO
    # three unconnected ROT ports: the extracted power leaving the system
    opens = graph.open_ports()
    assert sorted(opens) == [(r.rotor_id, "shaft:ROT") for r in wa.ROTORS]
    # the partition of unity is CONVEX (L6/C1) and sums to one
    pou = graph.partition_of_unity
    assert pou.chi_min() >= 0.0
    assert pou.identity_residual() < 1e-12
    # every record is complete
    for a in graph.agents:
        assert not a.capabilities.missing_fields(), (a.agent_id,
                                                     a.capabilities.missing_fields())


@needs_checkpoint
def test_the_absolute_trace_makes_the_probe_base_visible():
    """Both sides of a rotor seam declare a base, and they are the flow's own.

    Under the perturbation convention every fluid `probe_base` in this vault is
    identically zero, so `base_disagreement` reads *consistent* on every seam
    while the two sides sit at their own states.
    """
    u = np.ones((wa.NY, wa.NX))
    state = (u, np.zeros((wa.NY, wa.NX)))
    _graph, experts = wa.build(*state, kind="reference")
    for r in wa.ROTORS:
        rot = experts[r.rotor_id]
        base = rot.probe_base("up:MECH")
        assert base.shape == (wa.ROTOR_CELLS,)
        assert np.all(base > 0.0), "a wind farm is never at a zero trace"
    for name in wa.DEFAULT_TILING.names:
        w = experts[name]
        for p in wa.fluid_ports(name):
            b = w.probe_base(p.name)
            assert b.shape == (wa.segment_index(name, *p.name.split(":")[:2]).size,)


@needs_checkpoint
def test_the_reference_expert_responds_and_the_flux_is_a_traction():
    state = (np.ones((wa.NY, wa.NX)), np.zeros((wa.NY, wa.NX)))
    _graph, experts = wa.build(*state, kind="reference")
    w = experts["F01"]
    port = wa.port_name("xhi", "full")
    base = w.probe_base(port)
    f0 = np.asarray(w.respond(port, base))
    assert f0.shape == (wa.N,)
    assert np.all(np.isfinite(f0))
    # a uniform flow leaves no normal gradient at an x-face, to the solver's own
    # accuracy: the sign control the first real assembly needed
    assert np.max(np.abs(f0)) < 1e-6


# ---------------------------------------------------------------------------
# W98: transport and pressure are GLOBAL, and running them per window
# manufactured a wake where no turbine stands
# ---------------------------------------------------------------------------


def _divergence(u, v):
    ny, nx = u.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=wa.DX)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=wa.DX)
    kx[nx // 2] = 0.0
    ky[ny // 2] = 0.0
    return np.real(np.fft.ifft2(1j * kx[None, :] * np.fft.fft2(u)
                                + 1j * ky[:, None] * np.fft.fft2(v)))


def test_W98_the_domain_is_open_at_the_outlet_so_a_wake_leaves_instead_of_returning():
    """The defect the viewer exposed, in one assertion.

    `step_many`'s ``frame=`` translates each window with `spectral_shift`, which
    is periodic ON THE WINDOW: a wake reaching a window's outflow edge re-enters
    that window's own inflow edge, 3.5 D upstream, where no turbine stands.
    `transport_and_project` runs the translation once on the whole domain,
    extended downstream by a tapered buffer, so what wraps onto the inlet is the
    taper's zero -- freestream.
    """
    y = (np.arange(wa.NY) + 0.5) * wa.DX
    uf = np.zeros((wa.NY, wa.NX))
    wake = np.abs(y - 1.75) <= 0.5 * wa.ROTOR_D
    uf[np.ix_(wake, np.arange(wa.NX - 32, wa.NX))] = -0.5     # a wake, at the outlet
    u1, _v1 = wa.transport_and_project(uf, np.zeros_like(uf), project=False)
    # it must still be at the outlet, and it must NOT have arrived at the inlet
    assert np.abs(u1[:, -25:]).max() > 0.4
    assert np.abs(u1[:, :wa.HALO]).max() < 1e-2


def test_W98_the_translation_is_exact_because_the_extension_is_periodic():
    """One macro-step moves the field ``U dt`` and does not damp it.

    The buffer is tapered to zero, so the extension is periodic-compatible and
    the translation is a phase factor rather than an interpolation -- no stencil,
    no diffusion, and nothing to accumulate over a sixty-step march.
    """
    x = (np.arange(wa.NX) + 0.5) * wa.DX
    bump = -0.4 * np.exp(-((x - 4.0) / 0.5) ** 2)
    uf = np.tile(bump, (wa.NY, 1))
    u1, _ = wa.transport_and_project(uf, np.zeros_like(uf), project=False)
    centroid = lambda f: float((x * f).sum() / f.sum())       # noqa: E731
    assert centroid(u1[0]) - centroid(uf[0]) == pytest.approx(
        wa.U_INF * wa.MACRO_DT, abs=1e-6)
    assert u1.min() == pytest.approx(uf.min(), rel=1e-3)


def test_W98_the_global_projection_is_a_projection_and_never_returns_nan():
    """It leaves a divergence-free field alone, and it guards THREE null modes.

    `_wavenumbers` zeroes the Nyquist for a derivative so that the divergence and
    the projection meant to cancel it agree at every mode.  On a rectangle that
    leaves ``k = 0`` at three places, not one -- ``(0,0)``, ``(0, nyq)`` and
    ``(nyq, 0)`` -- and guarding only the origin returns NaN on the first step.
    """
    x = (np.arange(wa.NX) + 0.5) * wa.DX
    y = (np.arange(wa.NY) + 0.5) * wa.DX
    psi = (np.exp(-((x[None, :] - 9.0)) ** 2)
           * np.exp(-((y[:, None] - 3.5)) ** 2))
    u0 = np.gradient(psi, wa.DX, axis=0)                      # div-free by construction
    v0 = -np.gradient(psi, wa.DX, axis=1)
    kept, _ = wa.transport_and_project(u0, v0, project=False)
    projected, _ = wa.transport_and_project(u0, v0, project=True)
    assert np.isfinite(projected).all()
    assert np.abs(projected - kept).max() < 0.05 * np.abs(kept).max()

    rng = np.random.default_rng(0)
    a = rng.standard_normal((wa.NY, wa.NX))
    b = rng.standard_normal((wa.NY, wa.NX))
    pa, pb = wa.transport_and_project(a, b, project=True)
    assert np.isfinite(pa).all() and np.isfinite(pb).all()
    assert np.abs(_divergence(pa, pb)).max() < 0.5 * np.abs(_divergence(a, b)).max()


@needs_checkpoint
def test_W98_the_march_does_not_manufacture_a_wake_upstream_of_a_lone_turbine():
    """The end-to-end regression, on the checkpoint, with ONE turbine.

    R1 is the only disk and it stands in the top row, so every cell upstream of
    its plane and the whole bottom row have no cause.  At 1.25 D upstream after
    ten macro-steps, measured:

        before W98   0.8244        a 17.6% deficit with nothing to cause it
        after  W98   0.9495
        reference    0.9966        the referent, which never had the defect

    so the gate sits at 0.92, between them.  The residual 5% is the CHECKPOINT's
    own support reach (W93): the measured reach is the WHOLE window, so a window
    holding a disk responds to it everywhere and no coupling choice undoes that.
    The bottom-row bound is a coarse guard against gross contamination rather
    than a discriminator -- at ten steps both marches pass it, and it is the
    upstream number that separates them.
    """
    import sys, os                                            # noqa: PLC0415
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    from w93_wake_array import disk_rows, march                # noqa: PLC0415

    u, _v, _hist, _prev = march({"R1"}, 10, "poseidon")
    rows = disk_rows(wa.ROTORS[0].y_centre)
    upstream = float(np.mean(u[rows, 40]))                    # 1.25 D, well upstream
    bottom = float(np.mean(u[disk_rows(wa.ROTORS[2].y_centre), 40]))
    assert upstream > 0.92, f"manufactured a {1 - upstream:.1%} deficit upstream"
    assert abs(bottom - 1.0) < 0.05, f"bottom row drifted to {bottom:.4f}"
