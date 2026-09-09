"""PoC 2 -- CS-10's graph and CS-12's graph, assembled into one.

**This is an ASSEMBLY, not a thirteenth case study.**  Every part of it has been
built and measured somewhere else:

  * the fluid column is `ground_effect`'s, unchanged through `wing_fsi` -- six
    exposed `reference.WindowNS` windows of 80x80 at halo 16 over 208x144 cells,
    a `ProjectedAssembly` with one global spectral Leray projection per exchange,
    and the cuts placed 8 cells clear of the plate in x and 12 in y (W124);
  * the wing is `wing_fsi.FlexWing`, the porous inclined plate at 20 degrees with
    ``C_N = 20``, unchanged;
  * the structure is `wing_fsi.WingStructure` -- `ThermoStruct2D.solve_mechanical`
    on a 32x2 Q1 plane-stress cantilever clamped at the leading edge -- with the
    thickness promoted from a module constant to a field, which is additive;
  * the suspension is `ground_effect.Suspension`, ``k (h0 - h) = L``, two lines.

What is new is that **both seams are live at once**.  CS-10's wing rides at a
height its spring chooses and does not bend; CS-12's wing bends and its mount
does not move.  Here the plate's stations sit at

    y_k  =  h  +  s_k sin(alpha)  +  delta_k cos(alpha),
    x_k  =  x_LE  +  s_k cos(alpha)  -  delta_k sin(alpha),

with ``h`` the suspension's ride height and ``delta`` the structure's own
generalized deflection -- **two degrees of freedom in two different directions,
neither of which can stand in for the other**, and each one's agent has to be
told what the other did.

Three things follow and each is a claim this module has to make good on:

1. **The interface problem is one system of ``N_STATION + 1`` equations, not two
   systems solved in turn.**  `general-coupling-scheme` section 4.2's rule --
   write the residual in the port's own conjugate variables and solve it there --
   applied to two seams at once.  The Jacobian is analytic and needs one field
   evaluation per Newton step regardless of how many seams there are, because
   neither agent's response depends on the other's velocity through the flow.
2. **Each parent is a limit of it, and both limits are asserted.**  Send the
   spring rate to infinity and the mount cannot move: the remaining 32 equations
   are `wing_fsi.FSIRollout.solve_interface` exactly.  Send the structure's
   stiffness to infinity and the plate cannot bend: the remaining scalar equation
   is `ground_effect.GroundRollout`'s.  A composition that does not reduce to its
   parts is not an assembly.
3. **The compiler's verdict is per seam, and the two seams get different ones.**
   That is the whole content of the demo's new indicator, and it is not a display
   choice -- it is what `compile_scheme` already reports and nothing had read.

**W114 carries into the assembly and it has to be checked rather than assumed.**
R10 refuses an ``EMBEDDED`` agent only when another agent shares its
`governing_family`.  `Suspension` declares the FLOW's family -- correctly, since
an algebraic closure inside a continuum problem is not a different continuum
problem -- so adding it leaves `STRUCT` the sole agent of
``plane-stress-elasticity-2d`` and the narrowing still clears.  A graph that
declared the suspension as its own family would move nothing about R10; one that
declared the STRUCTURE with the flow's family brings the refusal straight back,
and that control is run in the driver.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

import numpy as np
import torch

from ..capability import ExpertCapabilities, MotionClass
from ..graph import (Agent, CaseGraph, Connection, Decomposition, GlobalField,
                     MeasuredConstants)
from ..ports import PortType
from . import ground_effect as G
from . import wing_fsi as W

# ---------------------------------------------------------------------------
# what is inherited verbatim
# ---------------------------------------------------------------------------

DX = W.DX
NX, NY = W.NX, W.NY
MACRO_DT = W.MACRO_DT
EXCHANGES = W.EXCHANGES
N_STATION = W.N_STATION
CHORD = W.CHORD
ALPHA = W.ALPHA
C_N = W.C_N
NU = W.NU
U_INF = W.U_INF
TORCH_DTYPE = W.TORCH_DTYPE
E_REF = W.E_REF
DELTA_MAX = W.DELTA_MAX
Y_MOUNT = W.Y_MOUNT

WingTiling = W.WingTiling
DEFAULT_TILING = W.DEFAULT_TILING
SINGLE_TILING = W.SINGLE_TILING

# ---------------------------------------------------------------------------
# the design space
# ---------------------------------------------------------------------------

#: **The reference design, and it is each parent's own operating point.**
#: ``E*`` and the thickness are CS-12's; ``k`` and ``h0`` are CS-10's.  Nothing
#: here is tuned: the point of starting at the parents' values is that the
#: assembled march at the reference design can be read against two published
#: rollouts rather than against nothing.
E_STAR_REF = W.E_STAR                  # 5.0e4
TC_REF = W.THICK / W.CHORD             # 0.04
K_REF = G.K_SPRING                     # 2.5
H0_REF = G.H0_REF                      # 0.32

#: The box the search runs in, one axis at a time, and each bound is a
#: MEASUREMENT rather than a preference.
#:
#: ``E*``   below about 3e4 the equilibrium deflection leaves the structural
#:          expert's own small-strain envelope -- CS-12 section 7's table has
#:          ``-0.0633`` at ``2e4`` against a bound of ``0.05`` -- so the lower
#:          bound is where the expert stops answering, not where the optimiser
#:          stops liking the answer.
#: ``t/c``  the lower bound is where the Q1 mesh's SHEAR LOCKING inverts the
#:          stress trend: measured in the driver's `stage_thickness`, the peak
#:          von Mises rises with thickness below ``t/c ~ 0.026`` and falls above
#:          it, and a design search run through the inversion optimises an
#:          element formulation rather than a wing.  See `THICKNESS_SCOPE`.
#: ``k``    the spring has to hold the wing off the floor at the heaviest load
#:          the box can produce and stay in compression, which is
#:          `Suspension.validity`.
#: ``h0``   CS-10's own measured ground-effect curve runs from 0.075 to 0.75 and
#:          is monotone with no turnover; the box sits inside it with the floor
#:          clear of the plate's own smearing.
DESIGN_BOX: dict[str, tuple[float, float]] = {
    "e_star": (3.0e4, 2.0e5),
    "tc": (0.032, 0.046),
    "k": (1.5, 6.0),
    "h0": (0.20, 0.45),
}
DESIGN_KEYS = ("e_star", "tc", "k", "h0")

#: **The thickness's UPPER bound is the flow model's own neglect, and it was
#: found by a search running into it.**  The porous plate has *no thickness in
#: the flow* -- it is a line of stations, and CS-12 section 3.3 says so -- so the
#: thickness knob costs the aerodynamics nothing, and a first run with the bound
#: at ``t/c = 0.080`` drove straight to it: the optimum sat on all four box
#: bounds with both ceilings inactive (stress 86 of 600, deflection 0.0019 of
#: 0.040).  That is a box search wearing the word "constrained", and the cause is
#: that stiffening and thickening the wing raise the downforce AND lower both the
#: stress and the deflection, so the objective and both constraints pull the same
#: way on the two structural knobs.
#:
#: The bound that is defensible is the one the flow model implies: a thickness
#: the grid would have to resolve is a thickness the flow model may not ignore.
#: At ``dx = 1/64`` and a chord of ``0.5``, ``t/c = 0.046`` is ``1.47`` cells and
#: ``t/c = 0.080`` is ``2.56``.  **The plate may be about a cell thick and not
#: three**, and the first run's optimum was outside that.
THICKNESS_BOUND_NOTE = (
    "t/c <= 0.046 is 1.47 cells at dx = 1/64 on a chord of 0.5. The flow model "
    "carries the plate as a line of stations with no thickness at all, so a "
    "plate the grid would resolve is one the flow model may not neglect; the "
    "first design run drove to t/c = 0.080, which is 2.56 cells, and its optimum "
    "is therefore outside the flow model's own scope rather than merely at a box "
    "bound."
)

#: **The two ceilings, and they constrain different things.**
#:
#: A stress is set by the LOAD and the GEOMETRY -- raising ``E`` at a fixed
#: traction shrinks the displacement and the strain in the same proportion and
#: leaves ``D eps`` where it was -- so the stress ceiling is a constraint on the
#: thickness and on how hard the wing is worked, and it does not see ``E*`` at
#: all.  A deflection is set by the load and the BENDING STIFFNESS, so the
#: deflection ceiling sees both.  Two constraints on four knobs, and they pull
#: the two structural knobs in different directions: that is what makes this a
#: constrained design problem rather than a box-clipped one.
#:
#: Both numbers are declared at a level the reference design sits INSIDE and the
#: downforce-seeking direction runs INTO, which is the only property a ceiling
#: has to have for a constrained search to be about the constraint.  Neither is a
#: statement about an alloy -- see `THICKNESS_SCOPE`.
#: **Both ceilings are set where they BIND, and the first run is why.**
#:
#: A ceiling the search never reaches is not a constraint, and the first run's
#: pair (600 and 0.040) were never reached: at its optimum the stress was 86 and
#: the deflection 0.0019.  Measured over the structural box at the aggressive
#: ride height the search chooses, the reachable ranges are **85 to 308** in the
#: stress and **0.0019 to 0.102** in the deflection, so:
#:
#:  * ``SIGMA_CEIL = 200`` is inside the reachable range and **above** the
#:    reference design's own 165, so the start is feasible and the
#:    downforce-seeking direction runs into it;
#:  * ``DELTA_CEIL = 0.035`` is **above** the reference design's own 0.0304 and
#:    **below** what the reference structure gives at the aggressive ride height
#:    (0.0432), so the deflection ceiling is what stops the ride height coming
#:    down until the structure is stiffened -- which is the trade this design
#:    problem is about.
#:
#: Neither is a statement about an alloy; see `THICKNESS_SCOPE`.
SIGMA_CEIL = 2.0e2
DELTA_CEIL = 0.035

#: CS-12's own settled fixed-shape load, quoted from `out/w136/w136.json`
#: (``setup.settled_load``).  It sets the release height and nothing else.
LOAD_REF = 0.22725455823530805

#: `WingStructure.validity`'s own bound, which is the expert declining rather
#: than the design being rejected.  `DELTA_CEIL` sits inside it deliberately: a
#: search that drives the deflection constraint to its own boundary must not
#: simultaneously drive the expert out of its envelope, or the two failures are
#: indistinguishable from outside a `try` block -- CS-12 section 7.1's rule.
assert DELTA_CEIL < DELTA_MAX

THICKNESS_SCOPE = (
    "The plate is a 32x2 Q1 plane-stress cantilever and Q1 elements LOCK in "
    "bending, so the model is stiffer than its own beam theory and the locking "
    "gets worse as the elements get more slender.  Measured over the design box: "
    "the tip deflection under a fixed traction is monotone in the thickness "
    "everywhere and its power-law exponent runs -0.72 at t/c = 0.02 to -2.74 at "
    "t/c = 0.08 against beam theory's -3, and the peak von Mises RISES with "
    "thickness below t/c ~ 0.026 -- the wrong way -- before falling with an "
    "exponent -1.04 to -1.76 against beam theory's -2.  The design box's lower "
    "thickness bound is placed above the inversion for that reason.  Inside the "
    "box every trend has the physical sign and the wrong magnitude by up to a "
    "factor of two, so a thickness reported here is a knob on THIS model and no "
    "stress figure is a statement about an alloy.  ||S_e||_2 is not the quantity "
    "to read: it is set by the stiffest mode, which is a locked shear mode, and "
    "it is non-monotone in the thickness while every response quantity is not."
)

R10_ASSEMBLY_SCOPE = (
    "W114's narrowing clears STRUCT in this graph for the same reason it clears "
    "it in CS-12: STRUCT is the only agent declaring plane-stress-elasticity-2d, "
    "so nothing has cut its domain.  Adding SUSP does not change that, because a "
    "lumped algebraic closure inside a continuum problem declares the FLOW's "
    "governing family -- which is right on its own terms and is also what keeps "
    "the predicate's answer the same.  The predicate remains a PROXY for `is this "
    "agent's domain cut`, and the control that shows it still fires is in the "
    "driver."
)


# ---------------------------------------------------------------------------
# the structure, with thickness on the tape
# ---------------------------------------------------------------------------


def surface_operator(tc: float) -> dict[str, Any]:
    """CS-12's own operator at a thickness expressed as ``t/c``."""
    return W._surface_operator(thick=float(tc) * CHORD)


class _OperatorOfThickness(torch.autograd.Function):
    """``t/c -> (S_e at E_ref, the stress map)``, differentiable in the thickness.

    **The rollout is exactly differentiable in ``E*`` and finite-differenced in
    the thickness, and the split is not a compromise -- it is where each is
    exact.**  Linear elasticity is exactly linear in ``E`` at fixed geometry, so
    ``S_e(E*) = (E*/E_ref) S_e(E_ref)`` holds to the bit (CS-12 section 3.2
    asserts it) and autograd carries ``E*`` with no help.  The thickness moves the
    MESH, so there is no such identity: the derivative of the operator is a
    derivative of an FE assembly.

    What makes a difference legitimate here rather than a shrug is that the thing
    being differenced is **cheap, smooth and small**: one operator build is 32
    solves of a 198-degree-of-freedom system, and it is a function of ONE scalar,
    evaluated away from any rollout.  So the difference is taken on the operator
    and the exact adjoint is taken through everything else -- the whole march, the
    blend, the projection and both seams' Newton solves.  The step is declared
    rather than tuned and the driver walks it over three decades against the
    composed objective's own central difference, which is the only check that
    means anything.
    """

    #: Relative central-difference step on ``t/c``.  Chosen in the driver's
    #: `stage_thickness` from a step sweep, on the truncation branch with the
    #: cancellation branch not reached.
    STEP = 1.0e-4

    @staticmethod
    def forward(ctx, tc):
        t = float(tc.detach() if torch.is_tensor(tc) else tc)
        op = surface_operator(t)
        S = torch.as_tensor(op["S_e"], dtype=TORCH_DTYPE)
        M = torch.as_tensor(op["sigma_map"], dtype=TORCH_DTYPE)
        h = _OperatorOfThickness.STEP * t
        op_p, op_m = surface_operator(t + h), surface_operator(t - h)
        ctx.save_for_backward(
            torch.as_tensor((op_p["S_e"] - op_m["S_e"]) / (2.0 * h),
                            dtype=TORCH_DTYPE),
            torch.as_tensor((op_p["sigma_map"] - op_m["sigma_map"]) / (2.0 * h),
                            dtype=TORCH_DTYPE))
        return S, M

    @staticmethod
    def backward(ctx, gS, gM):
        dS, dM = ctx.saved_tensors
        return (gS * dS).sum() + (gM * dM).sum()


def operator_of(tc) -> tuple[torch.Tensor, torch.Tensor]:
    """``(S_e at E_ref, sigma_map)`` as tensors, differentiable in ``tc``."""
    if not torch.is_tensor(tc):
        tc = torch.tensor(float(tc), dtype=TORCH_DTYPE)
    return _OperatorOfThickness.apply(tc)


def von_mises(sig):
    """Plane stress, from `solve_mechanical`'s own ``(s_xx, s_yy, s_xy)``."""
    sx, sy, sxy = sig[..., 0], sig[..., 1], sig[..., 2]
    q = sx * sx - sx * sy + sy * sy + 3.0 * sxy * sxy
    if torch.is_tensor(q):
        return torch.sqrt(torch.clamp(q, min=0.0))
    return np.sqrt(np.maximum(q, 0.0))


def stress_of(sigma_map, traction):
    """The element stress field under a station traction.  Linear, exactly.

    ``sigma_map[k]`` is the expert's own answer to a unit traction at station
    ``k``, so this is the expert's `solve_mechanical` result without re-entering
    it -- asserted against a direct solve at ``1.8e-13``.
    """
    return torch.einsum("k,kea->ea", traction, sigma_map)


# ---------------------------------------------------------------------------
# the agents
# ---------------------------------------------------------------------------


@dataclass
class RidingFlowWindow(W.FSIFlowWindow):
    """CS-12's flow window with CS-10's mounting port back on it.

    The window that carries the wing now exposes TWO surface ports on the same
    plate: ``wet:MECH`` against the structure, whose trace is the wetted
    surface's own normal velocity relative to the mount, and ``mount:MECH``
    against the suspension, whose trace is the mount's vertical velocity.  They
    are different halves of the same physical face and they are not redundant --
    one carries the plate's SHAPE and the other its POSITION.

    **The two ports are declared separately rather than as one 33-dimensional
    port, and that is a statement about the port algebra.**  A single port would
    have to declare one prolongation and one interface space for two quantities
    whose agents are different, whose null spaces are different, and whose
    substitution certificates are separately meaningful.  `port-algebra` section
    3.2's rule for `ADVEC` -- the passenger list is per FACE, not per agent -- is
    the same rule: what shares a face need not share a port.
    """

    ride_height: float = Y_MOUNT

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.has_wing:
            #: **The wing sits at the RIDE HEIGHT, not at CS-12's mount.**  CS-12
            #: parks the plate at ``Y_MOUNT = 0.625`` because it is not about
            #: ground effect; CS-10's ``h`` is the leading edge's height above the
            #: rolling road and runs 0.20-0.45.  They are the same coordinate, so
            #: the assembly is the CS-12 wing evaluated at the CS-10 height, and
            #: the window's own corner is subtracted from both (W130).
            self._wing = W.FlexWing(
                x_le=W.X_LE - self.ox * DX,
                y_mount=float(self.ride_height) - self.oy * DX,
                device=self.device)

    def probe_base(self, name: str) -> np.ndarray:
        if name == "mount:MECH":
            #: The mount's own vertical velocity, ABSOLUTELY -- zero is the wing
            #: at rest, which is a real physical state.  CS-10's `Suspension`
            #: declares the plate's current heave rate on its side, so once the
            #: wing is moving the two disagree and `L4/probe-base` says so
            #: (W74's class), which is reported rather than hidden.
            return np.zeros(N_STATION)
        return super().probe_base(name)

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(port, trace) -> flux.  ``mount:MECH`` is the VERTICAL half."""
        if port_name != "mount:MECH":
            return super().respond(port_name, trace)
        if not self.has_wing:
            raise ValueError(f"{self.agent_id} carries no wing")
        trace = np.asarray(trace, dtype=float).reshape(-1)
        if trace.shape[0] != N_STATION:
            raise ValueError(f"mount trace has {trace.shape[0]} stations, "
                             f"expected {N_STATION}")
        opt = dict(dtype=TORCH_DTYPE, device=self.device)
        ny = float(self._wing.n_hat[1])
        #: the trace is a VERTICAL velocity; the plate's normal velocity is its
        #: projection, which is the geometric factor CS-10's seam carried and
        #: CS-12's did not
        vp = torch.as_tensor(trace, **opt) * ny
        fx, fy, _, _ = self._wing.forcing(self._u, self._v, self._d, vp,
                                          self.ny_c, self.nx_c)
        u1, v1 = self._step(self._u, self._v, force=(fx[None], fy[None]))
        w, _um, _vm, _us, _vs = self._wing.station_normal(
            u1, v1, self._d, vp, self.ny_c, self.nx_c)
        #: the VERTICAL traction on the plate, which is the half a spring returns
        return (self._wing.normal_traction(w) * ny).detach().cpu().numpy()


def flow_ports(expert: RidingFlowWindow, motion: MotionClass) -> list:
    ports = W.flow_ports(expert, motion)
    if expert.has_wing:
        ports.append(W.port_decl(
            name="mount:MECH", port_type=PortType.MECH,
            geometry=f"the wing's mounting face, {N_STATION} stations inside "
                     f"{expert.agent_id}",
            direction=W.Direction.BIDIRECTIONAL, nondim=dict(W.MECH_SCALES),
            effective_resolution=W.modes_for(N_STATION),
            motion_class=motion,
            response_half=W.ResponseHalf.EFFORT,
            prolongation=W._prolongation(expert.agent_id, "mount:MECH",
                                         N_STATION),
            note="the VERTICAL traction on the wing: the MECH EFFORT against an "
                 "imposed mount vertical velocity (the FLOW). CS-10's seam, on "
                 "the window that already carries CS-12's"))
    return ports


def flow_capabilities(expert: RidingFlowWindow,
                      motion: MotionClass = MotionClass.STATIC,
                      substeps: int = EXCHANGES) -> ExpertCapabilities:
    caps = W.flow_capabilities(expert, motion, substeps)
    return replace(caps, ports=flow_ports(expert, motion))


def structure_capabilities(expert: W.WingStructure,
                           motion: MotionClass = MotionClass.STATIC
                           ) -> ExpertCapabilities:
    """CS-12's record with the thickness in the weight hash.

    The hash names the expert a certificate was earned on, and PoC 2 moves the
    thickness, so two designs that differ only in it are two different experts as
    far as any reuse claim is concerned -- which is W55's whole point.
    """
    caps = W.structure_capabilities(expert, motion)
    return replace(caps,
                   weight_hash=f"{caps.weight_hash}-t{expert.thick:.6g}")


def suspension_capabilities(expert: G.Suspension,
                            motion: MotionClass = MotionClass.STATIC
                            ) -> ExpertCapabilities:
    return G.suspension_capabilities(expert, motion)


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------


def connections(tiling: WingTiling = DEFAULT_TILING) -> list[Connection]:
    """CS-12's seams, plus CS-10's, on the same window."""
    conns = W.connections(tiling)
    conns.append(Connection(
        seam_id="mount", a=(tiling.wing_window(), "mount:MECH"),
        b=("SUSP", "mount:MECH"),
        port_type=PortType.MECH, derive_space=True,
        geometrically_coincident=True,
        # field <-> lumped: the lumped side responds in exactly the direction
        # incompressibility leaves open, so n_0 = 0 -- CS-10's row, unchanged,
        # and it is a DIFFERENT row of the table from the `wet` seam beside it,
        # which is fluid-solid MECH on a CLAMPED body and is 0 for a third reason.
        expected_null_dim=0,
        # **W138**, the `wet` seam's declaration on this seam's own physics.
        # `solve_seams` writes `k(h0 - h - dt v_mount) - L`: the spring's
        # reaction and the aerodynamic load, both positive in the same vertical
        # direction, subtracted.  A lumped agent has no domain and so no outward
        # normal of its own; naming SUSP says the shared direction is the one
        # `orientation` points AWAY from the fluid, which is a direction whether
        # or not the named side is a continuum.
        effort_normal="SUSP",
        note="CS-10's seam: field-to-lumped MECH carrying the wing's POSITION, "
             "beside the field-to-field one carrying its SHAPE"))
    return conns


def make_experts(u_full: np.ndarray, v_full: np.ndarray,
                 tiling: WingTiling = DEFAULT_TILING,
                 delta: np.ndarray | None = None,
                 design: dict | None = None,
                 flux_mode: str = "reaction",
                 nu: float = NU,
                 h: float | None = None) -> dict[str, Any]:
    d = dict(DESIGN_REF, **(design or {}))
    h = H0_REF if h is None else h
    experts: dict[str, Any] = {}
    for k, name in enumerate(tiling.names):
        ox, oy = tiling.offsets[k]
        experts[name] = RidingFlowWindow(
            agent_id=name,
            u0=u_full[oy:oy + tiling.wy, ox:ox + tiling.wx],
            v0=v_full[oy:oy + tiling.wy, ox:ox + tiling.wx],
            shared_faces=tiling.artificial_faces(ox, oy),
            has_wing=(name == tiling.wing_window()),
            ox=ox, oy=oy, delta=delta, nu=nu, flux_mode=flux_mode,
            ride_height=h)
    experts["STRUCT"] = W.WingStructure(e_star=d["e_star"],
                                        thick=d["tc"] * CHORD, delta=delta)
    experts["SUSP"] = G.Suspension(k=d["k"], h0=d["h0"], h=h)
    return experts


def build(u_full: np.ndarray,
          v_full: np.ndarray,
          motion: bool = False,
          tiling: WingTiling = DEFAULT_TILING,
          delta: np.ndarray | None = None,
          design: dict | None = None,
          flux_mode: str = "reaction",
          nu: float = NU,
          h: float | None = None,
          measured: MeasuredConstants | None = "default",
          experts: dict[str, Any] | None = None,
          struct_family: str | None = None,
          ) -> tuple[CaseGraph, dict[str, Any]]:
    """The assembled graph at a state.

    ``struct_family`` is the **control**: declare the structure with the fluid's
    `governing_family` and `L2/R10` fires again, which is what shows W114's
    narrowing is a premise check rather than the rule being switched off.  It is
    a parameter of `build` rather than a separate module so that the control runs
    the same code path as the graph it is a control for.
    """
    mc = MotionClass.SOLUTION_DEPENDENT if motion else MotionClass.STATIC
    if measured == "default":
        measured = W.MEASURED_MOVING if motion else W.MEASURED_STATIC
    experts = experts or make_experts(u_full, v_full, tiling, delta, design,
                                      flux_mode, nu, h)
    agents = [Agent(n, flow_capabilities(experts[n], mc),
                    domain=f"fluid window {n}")
              for n in tiling.names]
    s_caps = structure_capabilities(experts["STRUCT"], mc)
    if struct_family is not None:
        s_caps = replace(s_caps, governing_family=struct_family)
    agents.append(Agent("STRUCT", s_caps,
                        domain="Omega_solid: the whole wing section",
                        role="structure"))
    agents.append(Agent("SUSP", suspension_capabilities(experts["SUSP"], mc),
                        domain="the mounting face: a lumped suspension",
                        role="suspension"))
    return (
        CaseGraph(
            name=f"front-wing-{tiling.n_col}x{tiling.n_row}-"
                 f"{'riding' if motion else 'fixed-shape'}",
            agents=agents,
            connections=connections(tiling),
            decomposition=Decomposition.OVERLAPPING,
            overlap=tiling.halo * DX,
            overlap_cells=tiling.halo,
            partition_of_unity=W.projected_assembly(tiling),
            global_fields=[GlobalField(
                "pressure",
                # W117: produced by every FLUID agent.  Neither the structure nor
                # the suspension is in the list -- an elastic body has no pressure
                # field of the flow's kind and a spring has no field at all -- and
                # `applies_to` is the same set, which is what keeps
                # `L3/global-field` clear.
                produced_by=tuple(tiling.names),
                applies_to=tuple(tiling.names),
                note="one global spectral Leray projection on the assembled "
                     "fluid field, once per exchange"),
            ],
            cross_points=() if tiling.is_single else ("mid",),
            macro_dt=MACRO_DT,
            measured=measured,
            note=("PoC 2: CS-10's ground-effect graph assembled with CS-12's "
                  "FSI graph. Two surface seams live at once on one plate -- a "
                  "field-to-lumped MECH carrying the ride height and a "
                  "field-to-field MECH carrying the deflection")),
        experts,
    )


DESIGN_REF: dict[str, float] = {
    "e_star": E_STAR_REF, "tc": TC_REF, "k": K_REF, "h0": H0_REF,
}


# ---------------------------------------------------------------------------
# the march
# ---------------------------------------------------------------------------


@dataclass
class FrontWingResult:
    """What one assembled march returns.  Every trace is per macro-step."""

    load: np.ndarray                  # downforce per unit span, positive down
    drag: np.ndarray                  # streamwise force on the plate
    h: np.ndarray                     # ride height
    tip: np.ndarray                   # trailing-edge normal deflection
    delta: np.ndarray                 # [steps, N_STATION]
    vm_max: np.ndarray                # peak von Mises over the plate
    strain_energy: np.ndarray
    spring_energy: np.ndarray
    interface_power: np.ndarray
    half_step: np.ndarray
    u_max: np.ndarray
    substeps: np.ndarray
    inner_residual: np.ndarray
    inner_residual_0: np.ndarray
    #: The strain energy AND the spring energy at RELEASE, so `R` differences the
    #: marched history against a real starting value rather than a one-sided
    #: stencil.  **Both are needed and forgetting the second is not a small
    #: error**: the strain energy at release is identically zero because the
    #: plate starts flat, so it is easy to carry one and not the other -- and the
    #: spring is already loaded at release, by construction (`release_height`),
    #: so its energy at release is the largest single number in the balance.
    #: Measured, with it missing: the first macro-step's residual came out at
    #: 1.81 against a static accounting's 1.00, i.e. the gate FAILED, on a march
    #: whose every other step closed to 5e-8.
    energy_0: float = 0.0
    spring_energy_0: float = 0.0
    u: Any = None
    v: Any = None
    delta_final: Any = None
    h_final: float = 0.0
    wall_s: float = 0.0
    fields: dict = field(default_factory=dict)

    def settled(self, frac: float = 0.25) -> dict:
        n = max(1, int(round(frac * self.load.size)))
        return dict(load=float(self.load[-n:].mean()),
                    drag=float(self.drag[-n:].mean()),
                    h=float(self.h[-n:].mean()),
                    tip=float(self.tip[-n:].mean()),
                    vm_max=float(self.vm_max[-n:].max()),
                    delta_max=float(np.abs(self.delta[-n:]).max()))


class FrontWingRollout(W.FSIRollout):
    """The composed march, differentiable in all four design knobs.

    **It SUBCLASSES `FSIRollout` and the four composition-layer operators are
    inherited rather than rewritten.**  `cut`, `blend`, `project` and `band` --
    the tiling, the partition of unity, the one global spectral Leray projection
    per exchange and the boundary band -- are CS-12's, which are CS-10's, and the
    only way to say that credibly is to not have a second copy of them.  What is
    overridden is the interface system, the exchange and the march.  The parent's
    own `solve_interface` is left reachable on purpose: it is the ``k -> inf``
    limit this system has to reduce to, and the test asserts it does.

    One code path, three configurations, exactly as `FSIRollout` and
    `GroundRollout` are:

      ``WingTiling()``        the composed column -- six windows, fourteen seams
      ``WingTiling.single()`` the REFERENT -- one window over the whole domain,
                              a partition of unity identically one, cut and blend
                              the identity, and the same global projection applied
                              to a field nothing was blended into.  The two differ
                              by the CUT and by nothing else.
      ``coupling='tight'``    the 33-equation interface system solved at EVERY
                              exchange; ``'lagged'`` once every ``lag``
                              macro-steps; ``'split'`` the two seams solved in
                              TURN rather than together, which is the scheme this
                              assembly exists to have an alternative to.
    """

    def __init__(self, tiling: WingTiling = DEFAULT_TILING,
                 design: dict | None = None,
                 nu: float = NU, device: str = "cpu",
                 coupling: str = "tight", motion: bool = True,
                 lag: int = 1, n_inner: int = 3,
                 checkpoint: bool = True) -> None:
        if coupling not in ("tight", "lagged", "split", "staggered"):
            raise ValueError(coupling)
        self.design = dict(DESIGN_REF, **(design or {}))
        super().__init__(tiling=tiling, e_star=self.design["e_star"], nu=nu,
                         device=device,
                         coupling="tight" if coupling == "split" else coupling,
                         motion=motion, lag=lag, n_inner=n_inner,
                         checkpoint=checkpoint)
        #: `split` is this assembly's own ablation and the parent has no member
        #: for it, so it is set after the parent has validated the three it does
        #: know.  Every branch that reads it is in this class.
        self.coupling = coupling
        self._opt = dict(dtype=TORCH_DTYPE, device=device)
        self.n_hat_y = float(self.wing.n_hat[1])
        self.n_hat_x = float(self.wing.n_hat[0])

    # -- the design, as tensors -------------------------------------------

    def knobs(self, design=None):
        """``(e_star, tc, k, h0)`` as tensors, leaves if they were passed as such."""
        d = dict(self.design, **(design or {}))
        out = {}
        for key in DESIGN_KEYS:
            x = d[key]
            out[key] = (x if torch.is_tensor(x)
                        else torch.tensor(float(x), **self._opt))
        return out

    def structure(self, e_star, tc):
        """``(S_e, sigma_map)`` at this design.  Both are on the tape."""
        S_ref, M = operator_of(tc)
        return S_ref * (e_star / E_REF), M

    @staticmethod
    def release_height(h0, k):
        """The QUASI-STATIC equilibrium under CS-12's own settled load.

        **Not ``h0``, and the difference is a refusal rather than a preference.**
        At ``h = h0`` the spring carries nothing, so the wing falls under the
        whole load from rest; and a release ABOVE ``h0`` puts a linear compression
        spring in tension, which is what `Suspension.validity` declines -- CS-10
        was caught by exactly that.  Releasing at ``h0 - L_ref/k`` starts the
        march where the spring is already carrying the load it will settle at, so
        what the transient shows is the COUPLING adjusting rather than a body
        being dropped.  ``L_ref`` is CS-12's measured settled load at the
        reference design; it is a starting point and not a prediction, and every
        column in a comparison is released from the same one.
        """
        return h0 - LOAD_REF / k

    # -- the interface system ---------------------------------------------

    def solve_seams(self, u, v, delta, h, w_delta, v_mount, S_e, k, h0,
                    interval=None):
        """**Both seams, one system, in the ports' own conjugate variables.**

        ``general-coupling-scheme`` section 4.2's rule with two seams instead of
        one.  The unknowns are the two ports' FLOW halves --

            ``w_delta``  [S]  the wetted surface's normal velocity RELATIVE to
                              the mount: the structure's port trace
            ``v_mount``  [1]  the mount's vertical velocity: the suspension's

        -- and the surface's total normal velocity, which is what the fluid sees,
        is ``w_plate = w_delta + v_mount * n_y``.  With ``w = w_ext - w_plate``
        the relative flow and ``f = 1/2 C_N |w| w`` the aero traction,

            R1 = S_e (delta + dt w_delta)  -  f                       [S]
            R2 = k (h0 - h - dt v_mount)   -  L,   L = -sum f n_y ds  [1]

        both of the form *reaction minus load*, which is the sign convention
        W138 is about: the two responses are each other's negative in orientation
        and the residual SUBTRACTS them where ``Lambda_M`` adds.

        **The Jacobian is analytic and one field evaluation serves the whole
        system**, for CS-10's reason carried to two seams: ``w_ext`` does not
        depend on either unknown, so with ``g = C_N |w|``

            dR1/dw_delta = dt S_e + diag(g)      dR1/dv_mount = g n_y
            dR2/dw_delta = -g n_y ds             dR2/dv_mount = -k dt - sum g n_y^2 ds

        and the ``(S+1)``-square system is assembled and solved directly.  It is
        NOT symmetric, and that is the orientation again rather than a defect: the
        two residuals are written from opposite sides of the same face.

        **Each parent is a limit.**  Drop the last row and column and fix
        ``v_mount = 0`` and this is `FSIRollout.solve_interface` exactly; drop the
        first ``S`` rows and columns and fix ``w_delta = 0`` and it is CS-10's
        scalar Newton step, including its analytic derivative
        ``-k dt - C_N sum |w| cos^2 alpha ds``.  Both are asserted.

        Three steps at a FIXED count, because a reverse-mode tape has to have the
        same shape at every design, and the residual is returned at both ends.
        """
        wing = self.wing
        dt = self.dt_ex if interval is None else interval
        ny = self.n_hat_y
        S = N_STATION
        w_ext = wing.external_normal(u, v, delta, self.ny, self.nx, h=h)
        eye = torch.eye(S, dtype=u.dtype, device=u.device)
        r0 = None
        for _ in range(self.n_inner):
            w_plate = w_delta + v_mount * ny
            w = w_ext - w_plate
            f = wing.normal_traction(w)
            load = -(f * ny * self.ds).sum()
            r1 = S_e @ (delta + dt * w_delta) - f
            r2 = (k * (h0 - h - dt * v_mount) - load).reshape(1)
            r = torch.cat((r1, r2))
            if r0 is None:
                r0 = torch.linalg.vector_norm(r)
            g = C_N * torch.abs(w)
            J11 = dt * S_e + g[:, None] * eye
            J12 = (g * ny).reshape(S, 1)
            J21 = (-g * ny * self.ds).reshape(1, S)
            J22 = (-k * dt - (g * ny * ny * self.ds).sum()).reshape(1, 1)
            J = torch.cat((torch.cat((J11, J12), dim=1),
                           torch.cat((J21, J22), dim=1)), dim=0)
            step = torch.linalg.solve(J, r)
            w_delta = w_delta - step[:S]
            v_mount = v_mount - step[S]
        w_plate = w_delta + v_mount * ny
        w = w_ext - w_plate
        f = wing.normal_traction(w)
        load = -(f * ny * self.ds).sum()
        res = torch.linalg.vector_norm(torch.cat((
            S_e @ (delta + dt * w_delta) - f,
            (k * (h0 - h - dt * v_mount) - load).reshape(1))))
        return w_delta, v_mount, res, r0

    def solve_seams_split(self, u, v, delta, h, w_delta, v_mount, S_e, k, h0,
                          interval=None):
        """The two seams solved in TURN -- the ablation, and it is the point.

        A Gauss-Seidel sweep over the seams: solve the structure's 32 equations
        holding the mount's velocity at its last value, then solve the spring's
        one holding the deflection rate at what just came out.  This is what a
        composition that treats each seam as its own interface problem does, and
        it is what the joint solve is being compared against -- the assembly's
        own ablation, in the sense `poc1a-frozen-expert-results` section 6 means
        it: is the COUPLING between the two seams doing any work?
        """
        wing = self.wing
        dt = self.dt_ex if interval is None else interval
        ny = self.n_hat_y
        w_ext = wing.external_normal(u, v, delta, self.ny, self.nx, h=h)
        eye = torch.eye(N_STATION, dtype=u.dtype, device=u.device)
        r0 = None
        for _ in range(self.n_inner):
            w = w_ext - (w_delta + v_mount * ny)
            f = wing.normal_traction(w)
            load = -(f * ny * self.ds).sum()
            if r0 is None:
                r0 = torch.linalg.vector_norm(torch.cat((
                    S_e @ (delta + dt * w_delta) - f,
                    (k * (h0 - h - dt * v_mount) - load).reshape(1))))
            g = C_N * torch.abs(w)
            r1 = S_e @ (delta + dt * w_delta) - f
            w_delta = w_delta - torch.linalg.solve(dt * S_e + g[:, None] * eye, r1)
            w = w_ext - (w_delta + v_mount * ny)
            f = wing.normal_traction(w)
            load = -(f * ny * self.ds).sum()
            g = C_N * torch.abs(w)
            r2 = k * (h0 - h - dt * v_mount) - load
            j2 = -k * dt - (g * ny * ny * self.ds).sum()
            v_mount = v_mount - r2 / j2
        w = w_ext - (w_delta + v_mount * ny)
        f = wing.normal_traction(w)
        load = -(f * ny * self.ds).sum()
        res = torch.linalg.vector_norm(torch.cat((
            S_e @ (delta + dt * w_delta) - f,
            (k * (h0 - h - dt * v_mount) - load).reshape(1))))
        return w_delta, v_mount, res, r0

    # -- one exchange ------------------------------------------------------

    def exchange(self, u, v, delta, h, w_plate):
        fx, fy, w, load = self.wing.forcing(u, v, delta, w_plate, self.ny,
                                            self.nx, h=h)
        us, vs = self.cut(u), self.cut(v)
        fxs, fys = self.cut(fx), self.cut(fy)
        u1, v1 = self.solver.step_batch(us, vs, self.dt_ex, bc0=None,
                                        force=(fxs, fys))
        self.substep_log.append(int(self.solver.last_substeps))
        au, av = self.blend(u1, v1)
        au, av = self.project(au, av)
        bu, bv = self.band(au, av)
        f = self.wing.normal_traction(w)
        drag = (f * self.n_hat_x * self.ds).sum()
        return bu, bv, load, f, drag

    def macro_step(self, u, v, delta, h, w_delta, v_mount,
                   e_star, tc, k, h0, step: int = 0):
        S_e, sig_map = self.structure(e_star, tc)
        ny = self.n_hat_y
        zero = torch.zeros((), dtype=u.dtype, device=u.device)
        res, r0 = zero, zero
        load = drag = None
        f = None
        solve_now = self.motion and self.coupling in ("tight", "split")
        lagged_now = (self.motion and self.coupling == "lagged"
                      and step % self.lag == 0)
        hold = self.dt_ex if solve_now else self.lag * MACRO_DT
        solver = (self.solve_seams_split if self.coupling == "split"
                  else self.solve_seams)
        work = zero
        half = zero
        for i in range(EXCHANGES):
            if solve_now or (lagged_now and i == 0):
                w_delta, v_mount, res, r0 = solver(
                    u, v, delta, h, w_delta, v_mount, S_e, k, h0, interval=hold)
            w_plate = w_delta + v_mount * ny
            u, v, load, f, drag = self.exchange(u, v, delta, h, w_plate)
            #: The interface work over the exchange, accumulated exchange by
            #: exchange -- CS-12 section 6's rule, and it is the SUM of the two
            #: seams' powers because there are two seams.
            work = work + (f * w_plate).sum() * self.ds * self.dt_ex
            if self.motion and self.coupling != "staggered":
                #: **The half-step energy term, and BOTH receivers have one.**
                #: At a converged interface solve each agent's effort is its
                #: reaction at the END of the exchange while its energy gain is
                #: taken at the MIDDLE, and the difference is second order in the
                #: increment: ``1/2 d_delta^T S_e d_delta ds`` for the structure
                #: and ``1/2 k dh^2`` for the spring, with the SAME sign.  CS-12
                #: accumulated the first; carrying it without the second would
                #: correct one receiver's accounting and not the other's on a
                #: balance that adds them.
                d_delta = self.dt_ex * w_delta
                d_h = self.dt_ex * v_mount
                half = (half + 0.5 * (d_delta @ (S_e @ d_delta)) * self.ds
                        + 0.5 * k * d_h * d_h)
                delta = delta + d_delta
                h = h + d_h
        if self.motion and self.coupling == "staggered":
            new = torch.linalg.solve(S_e, f)
            w_delta = (new - delta) / MACRO_DT
            delta = new
            h_new = h0 - load / k
            v_mount = (h_new - h) / MACRO_DT
            h = h_new
        vm = von_mises(stress_of(sig_map, f)).max()
        return (u, v, delta, h, w_delta, v_mount, load, drag, f, vm,
                res, r0, work / MACRO_DT, half / MACRO_DT)

    # -- the march ---------------------------------------------------------

    def run(self, design=None, steps: int = 40, u0=None, v0=None,
            delta0=None, h0_state=None, grad: bool = False,
            keep_fields: Sequence[int] = (),
            progress: Callable[[int, float], None] | None = None
            ) -> FrontWingResult:
        import time
        opt = self._opt
        kn = self.knobs(design)
        e_star, tc, k, h0 = (kn["e_star"], kn["tc"], kn["k"], kn["h0"])
        u = (torch.full((self.ny, self.nx), U_INF, **opt) if u0 is None
             else torch.as_tensor(np.asarray(u0), **opt))
        v = (torch.zeros((self.ny, self.nx), **opt) if v0 is None
             else torch.as_tensor(np.asarray(v0), **opt))
        delta = (torch.zeros(N_STATION, **opt) if delta0 is None
                 else torch.as_tensor(np.asarray(delta0), **opt))
        h = (self.release_height(h0, k) if h0_state is None
             else torch.as_tensor(float(h0_state), **opt))
        w_delta = torch.zeros(N_STATION, **opt)
        v_mount = torch.zeros((), **opt)
        self.substep_log = []
        S_e0, _ = self.structure(e_star, tc)
        e0 = float((0.5 * (delta @ (S_e0 @ delta)) * self.ds).detach())
        s0 = float((0.5 * k * (h0 - h) ** 2).detach())

        loads, drags, hs, tips, deltas, vms = [], [], [], [], [], []
        energies, springs, powers, halves = [], [], [], []
        umax, ress, r0s = [], [], []
        fields, keep = {}, set(keep_fields)
        t0 = time.perf_counter()
        for s in range(steps):
            if s in keep:
                fields[s] = (u.detach().cpu().numpy().copy(),
                             v.detach().cpu().numpy().copy(),
                             delta.detach().cpu().numpy().copy(),
                             float(h.detach()))
            args = (u, v, delta, h, w_delta, v_mount, e_star, tc, k, h0, s)
            if grad and self.checkpoint:
                out = torch.utils.checkpoint.checkpoint(
                    self.macro_step, *args, use_reentrant=False)
            else:
                out = self.macro_step(*args)
            (u, v, delta, h, w_delta, v_mount, load, drag, f, vm,
             res, r0, p_mean, p_half) = out
            dv = delta.detach()
            hv = float(h.detach())
            self.check_envelopes(s, dv, hv, float(h0.detach()))
            um = float(torch.max(torch.hypot(u.detach(), v.detach())))
            if not np.isfinite(um):
                raise RuntimeError(f"the field is not finite at macro-step {s}")
            loads.append(load); drags.append(drag); hs.append(h)
            tips.append(delta[-1]); deltas.append(dv.cpu().numpy().copy())
            vms.append(vm)
            energies.append(0.5 * (delta @ (S_e0 @ delta)) * self.ds)
            springs.append(0.5 * k * (h0 - h) ** 2)
            powers.append(p_mean); halves.append(p_half)
            umax.append(um); ress.append(res); r0s.append(r0)
            if progress is not None:
                progress(s, um)

        def _np(xs):
            return np.array([float(x.detach()) if torch.is_tensor(x) else float(x)
                             for x in xs])

        return FrontWingResult(
            load=_np(loads), drag=_np(drags), h=_np(hs), tip=_np(tips),
            delta=np.array(deltas), vm_max=_np(vms),
            strain_energy=_np(energies), spring_energy=_np(springs),
            interface_power=_np(powers), half_step=_np(halves),
            u_max=np.array(umax), substeps=np.array(self.substep_log),
            inner_residual=_np(ress), inner_residual_0=_np(r0s),
            energy_0=e0, spring_energy_0=s0,
            u=u, v=v, delta_final=delta, h_final=float(h.detach()),
            wall_s=time.perf_counter() - t0, fields=fields,
            # the taped tail, so an objective can be built from a march
        )

    # -- the objective -----------------------------------------------------

    def check_envelopes(self, s: int, delta_v, h_v: float, h0_v: float) -> None:
        """Both experts' declared envelopes, at one macro-step.

        **This exists as a method because it was inlined in `run` and absent from
        `objective`, and the design search runs on `objective`.** So the search
        optimised its way out of the suspension's declared floor and nothing
        fired: the reported optimum leaves it at macro-step 4 of 120, and the
        discovery was the *ablation* crashing when it marched the same design
        through `run`. A declination the framework can make and does not make on
        the path a search uses is worse than no declination, because the search
        is what goes looking for the edge. Opened as **W145**.

        `_evaluate` in the driver already catches `RuntimeError` here and
        classifies it as an envelope decline; it had never had one to catch.
        """
        if not self.motion:
            return
        d_max = float(torch.max(torch.abs(delta_v)))
        if d_max > DELTA_MAX:
            raise RuntimeError(
                f"the wing left the structural expert's declared envelope at "
                f"macro-step {s}: max|delta| = {d_max:.6g} against a "
                f"small-strain bound of {DELTA_MAX:.6g}")
        #: **Two bounds, two sentences.**  The condition below is unchanged --
        #: the same set of states raises, in the same order -- but it used to
        #: report both violations with one message naming both numbers, and the
        #: two are physically different: below `H_FLOOR` the lumped suspension
        #: has run out of travel, and above `h0` the spring is in TENSION, which
        #: it is not declared to model either.  A demo that shows a decline has
        #: to be able to say which one, and reading "against a floor of 0.07"
        #: under a height of 0.35 is a sentence that makes the framework look
        #: wrong when it is right.
        if not h_v > G.H_FLOOR:
            raise RuntimeError(
                f"the wing left the suspension expert's declared envelope at "
                f"macro-step {s}: the ride height {h_v:.6g} is at or below the "
                f"declared floor of {G.H_FLOOR:.6g}, where the lumped "
                f"suspension has run out of travel")
        if not 0.0 < h0_v - h_v:
            raise RuntimeError(
                f"the wing left the suspension expert's declared envelope at "
                f"macro-step {s}: the ride height {h_v:.6g} is at or above the "
                f"free height {h0_v:.6g}, so the spring is in tension, which it "
                f"is not declared to model")

    def objective(self, design, steps: int, u0=None, v0=None, delta0=None,
                  h_state=None, frac: float = 0.25, grad: bool = True):
        """``(J, g_stress, g_delta)`` with the tape attached.

        ``(J, g_sigma, g_delta, true_sigma, true_delta)``.

        ``J`` is the settled downforce over the last ``frac`` of the march; the
        two constraints are the settled peak von Mises and the settled peak
        deflection, each returned as a **margin** ``value / ceiling - 1`` so that
        a feasible design is non-positive on both and the two are commensurable
        without a weight nobody measured.  The first pair is the SMOOTH maximum,
        which is what the penalty and therefore the gradient is built on; the
        second is the TRUE maximum, which is what feasibility is judged by.
        """
        opt = self._opt
        kn = self.knobs(design)
        e_star, tc, k, h0 = kn["e_star"], kn["tc"], kn["k"], kn["h0"]
        u = torch.as_tensor(np.asarray(u0), **opt)
        v = torch.as_tensor(np.asarray(v0), **opt)
        delta = (torch.zeros(N_STATION, **opt) if delta0 is None
                 else torch.as_tensor(np.asarray(delta0), **opt))
        h = (self.release_height(h0, k) if h_state is None
             else torch.as_tensor(float(h_state), **opt))
        w_delta = torch.zeros(N_STATION, **opt)
        v_mount = torch.zeros((), **opt)
        n_tail = max(1, int(round(frac * steps)))
        loads, vms, dmax = [], [], []
        for s in range(steps):
            args = (u, v, delta, h, w_delta, v_mount, e_star, tc, k, h0, s)
            if grad and self.checkpoint:
                out = torch.utils.checkpoint.checkpoint(
                    self.macro_step, *args, use_reentrant=False)
            else:
                out = self.macro_step(*args)
            (u, v, delta, h, w_delta, v_mount, load, drag, f, vm,
             _res, _r0, _p, _hs) = out
            #: the SAME check `run` makes, on the path the design search uses --
            #: see `check_envelopes` and W145.  It reads detached values, so it
            #: adds nothing to the tape and cannot change a gradient.
            self.check_envelopes(s, delta.detach(), float(h.detach()),
                                 float(h0.detach() if torch.is_tensor(h0)
                                       else h0))
            if s >= steps - n_tail:
                loads.append(load)
                vms.append(vm)
                dmax.append(torch.max(torch.abs(delta)))
        J = torch.stack(loads).mean()
        #: **A smooth maximum for the GRADIENT and the true one for FEASIBILITY,
        #: and the gap between them is returned rather than assumed negligible.**
        #:
        #: A hard `max` over the tail has a subgradient that jumps between
        #: macro-steps and an optimiser stepping on it chatters, so the penalty
        #: is built on `logsumexp`.  But `temp * logsumexp(x/temp)` overshoots the
        #: true maximum by up to `temp * log(n)` -- at a temperature of 2% of the
        #: ceiling over a 30-step tail that is **14%** of a stress reading, which
        #: is far too much to leave inside a constraint and call it smoothing.
        #: Two things follow and both are declared rather than tuned: the
        #: temperature is **0.5%** of each ceiling, and the caller is handed both
        #: numbers so that the search is DRIVEN by the smooth one and JUDGED by
        #: the true one.  The overshoot is conservative -- it reports a peak
        #: higher than the real one -- which is the safe direction for a ceiling
        #: and is still not a reason to leave it unmeasured.
        sig = torch.stack(vms)
        dl = torch.stack(dmax)
        g_sig = _softmax_over(sig, SIGMA_CEIL * SOFTMAX_FRAC) / SIGMA_CEIL - 1.0
        g_del = _softmax_over(dl, DELTA_CEIL * SOFTMAX_FRAC) / DELTA_CEIL - 1.0
        t_sig = sig.max() / SIGMA_CEIL - 1.0
        t_del = dl.max() / DELTA_CEIL - 1.0
        return J, g_sig, g_del, t_sig, t_del


#: The `logsumexp` temperature, as a fraction of each ceiling.  At 2% the smooth
#: maximum overshoots the true one by ``temp * log(n_tail)`` = 14% of a stress
#: reading over a 30-step tail, which is a constraint error wearing the word
#: "smoothing"; at 0.5% it is 3.4%, and `objective` returns both numbers so the
#: gap is measured at the optimum rather than argued about here.
SOFTMAX_FRAC = 0.005


def _softmax_over(x, temp: float):
    return temp * torch.logsumexp(x / temp, dim=0)


#: The penalty weight on each constraint margin.  Declared, not tuned: the
#: margins are already normalised by their own ceiling, so a weight of 1 makes a
#: 1% ceiling violation cost the same as a 1% loss of downforce measured against
#: J itself.  The driver runs the search at more than one weight so the answer's
#: dependence on it is a measurement rather than a hope.
PENALTY = 40.0


def penalised(J, g_sigma, g_delta, weight: float = PENALTY):
    """``J - w (max(0, g_sigma)^2 + max(0, g_delta)^2) |J|``.

    **It lives here rather than in the driver so the demo and the results page
    optimise the SAME objective.**  It was in the driver first, and the demo
    reached into `front_wing` for a `_penalised` that was never there -- caught
    by a timing probe rather than by a user, which is the good outcome of a bad
    arrangement.  One definition, two callers.

    Two properties, both declared: it is **smooth** at the constraint boundary,
    which a hinge is not and which an optimiser stepping on a subgradient
    notices; and it is **scaled by |J|**, so the weight is dimensionless and the
    same number means the same thing at every design point.  Feasibility is
    judged on the TRUE margins `objective` returns beside these, never on these.
    """
    pen = (torch.clamp(g_sigma, min=0.0) ** 2
           + torch.clamp(g_delta, min=0.0) ** 2)
    return J - weight * pen * torch.abs(J.detach())


#: kept as a private alias: the demo and an earlier driver both spell it this way
_penalised = penalised


def referent_rollout(**kw) -> FrontWingRollout:
    kw.setdefault("tiling", SINGLE_TILING)
    kw.setdefault("coupling", "tight")
    return FrontWingRollout(**kw)


def composed_rollout(**kw) -> FrontWingRollout:
    kw.setdefault("tiling", DEFAULT_TILING)
    return FrontWingRollout(**kw)
