"""The body-fitted column, declared: a volumetric device gets a port at last.

PoC 3, Tier 65.  [[poc3-racelab-car-graph]].

Tier 64 marched the radiator core and the recovery turbine in the car's duct on
solid walls and measured the three joins.  Nothing about that march was
DECLARED: the devices are body forces inside one solver, the joins are solved in
`car_union`, and no `CaseGraph` exists for the column, so `atlas.compiler` has
never seen it and gate clause P1 cannot be stated on it.  This module writes the
declaration, and the shape it had to take is the finding.

Why the obvious declarations do not work
----------------------------------------

**One fluid expert with no ports is refused by the record itself.**  The
composite's couplings are a rolling road, an inlet, a top and an outflow -- all
domain boundaries, which carry no port (W200's standing hole) -- plus two
devices, which are body forces.  That is zero ports, and
`capability.ExpertCapabilities` raises *"expert 'FLUID' declares no ports"*
before any compile.  `no_port_record` measures that refusal rather than
asserting it.

**One agent per grid, with the overset fringes as seams, would be a lie.**  The
porous column declares its fluid-fluid overlaps as ordinary artificial
boundaries because each window is marched by its own expert and the results are
blended.  The overset composite is ONE implicit solve: the interpolation
equations are rows of the same matrix as the momentum equations, so a grid
cannot be stepped alone and there is nothing to exchange, lag or blend.
`cross_grid_rows` counts the rows that make this true.  That answers **W268**
with a decision -- an overset overlap is not a seam, it is inside the expert --
rather than leaving it open.

**The two faces of a device would land on the same agent.**  On the porous
column the tiling is cut AT each device plane, so `integration_union` connects
the device's up and down faces to two different windows.  Nothing cuts the
composite there, so the same declaration would be a seam from the fluid to
itself; `self_seam_graph` builds exactly that and the record says what the
compiler does with it.

What does work, and it is W94's slot
------------------------------------

A device applied as a body force over a strip of thickness ``Delta`` presents
the traction its force integrates to,

    effort  =  T / W        [the MECH effort, a traction]
    flow    =  <u> over the strip,

and the product is the power the force takes out of the fluid, which Tier 64
measured against the shaft's claim to 3.4e-4.  So the strip IS expressible as a
MECH port -- the bond `wake_array`'s disk has never had against a frozen
operator (**W94**, open since Tier 18) -- **because this host accepts a body
force and reports the work it does**.  The fluid declares one such port per
device (`composite_capabilities`), each device declares the matching face
(`strip_core_capabilities`, `strip_rotor_capabilities`), and the seam between
them is an ordinary two-agent MECH connection.

What the declaration still cannot say is named in `GRAPH_HOLES` and on the page:
the port's geometry is a strip and not a surface, so its "outward normal" is a
direction rather than a boundary; the domain's own boundaries still carry no
ports; and the fluid's response to a traction on a strip is a solver call this
module does not make at compile time.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from ..capability import (BCChannel, ClaimType, Differentiable, Direction, EllipticSubsolve,
                          ExpertCapabilities, MotionClass, RecordError, TimeDiscretization,
                          port_decl)
from ..graph import Agent, CaseGraph, Connection, Decomposition, FluxMatching
from ..ports import PortType, ResponseHalf
from . import cooling_loop as CL
from . import car_solids as CS
from . import car_union as CU
from . import ground_effect as GE
from . import integration_union as IU
from . import powertrain as PT
from . import racelab as RL
from . import vehicle_march as VM
from . import wing_fsi as W

__all__ = [
    "STRIP_CELLS", "composite_capabilities", "strip_core_capabilities",
    "strip_rotor_capabilities", "connections", "build", "no_port_record",
    "self_seam_graph", "cross_grid_rows", "GRAPH_HOLES",
]

#: The duct's open band, in cells: what a device's strip spans (`car_union`).
STRIP_CELLS = 29

#: How many samples the trace across that band carries.  **It is the DEVICE's
#: number, not the grid's**: `integration_union.DEVICE_CELLS` is 32 and both
#: `CoreRadiator` and `wake_array.RotorDisk` refuse any other length, while the
#: band the strip spans is 29 grid cells wide.  So the port is non-conforming by
#: construction -- 32 device cells over 29 grid cells -- which is what
#: `effective_resolution` and the declared prolongation are for, and what the
#: porous column never had to say because its band was 32 cells of its own
#: lattice.
STRIP_SAMPLES = IU.DEVICE_CELLS

#: What this declaration still cannot say, carried with the graph so a page
#: cannot quietly drop one.
GRAPH_HOLES: tuple[str, ...] = (
    "the port's geometry is a STRIP, not a surface: its 'outward normal' is the "
    "streamwise direction and its measure is the band's width, so the seam's "
    "orientation is declared rather than read off a boundary",
    "the domain's own boundaries -- the rolling road, the inlet, the top and the "
    "outflow -- carry no ports, so no power crosses them in the declaration and "
    "the global balance W200 named is still not assemblable",
    "the fluid's response at a strip port is a solver call (apply the traction, "
    "step, read the band's mean velocity); this module declares it and does not "
    "call it at compile time, so no seam operator here is probed",
    "the eleven body grids and the background are inside ONE expert, so the "
    "overlap has no declaration at all -- which is the answer to W268 and not a "
    "measurement of it",
)


# ---------------------------------------------------------------------------
# the fluid
# ---------------------------------------------------------------------------


def _strip_port(name: str, label: str, cells: int = STRIP_SAMPLES) -> Any:
    return port_decl(
        name=name, port_type=PortType.MECH,
        geometry=f"{label}: a strip of {cells} device cells over the duct's "
                 f"{STRIP_CELLS} open cells, and "
                 f"{2 * CU.HALF_WIDTH_CELLS:g} cells along it, where the device's "
                 "body force acts",
        direction=Direction.BIDIRECTIONAL, nondim=dict(W.MECH_SCALES),
        effective_resolution=W.modes_for(cells), motion_class=MotionClass.STATIC,
        response_half=ResponseHalf.EFFORT,
        prolongation=W._prolongation("FLUID", name, cells),
        note="given a velocity across the band, the traction that holds it there "
             "over one step -- rho (u - u_now) Delta / dt, direct forcing. Its "
             "product with the velocity is the power the device's force takes out "
             "of the fluid, which is the quantity Tier 64 measured")


#: The band velocity the composite's record is written AT: Tier 64's registered
#: arm, mean over its window at the turbine plane.  A capability record describes
#: an expert at a state (W74), and this is the state.
U_BAND = 0.08351150680759814


def _band_of(band: Any) -> np.ndarray:
    if band is None:
        return np.full(STRIP_SAMPLES, U_BAND)
    b = np.asarray(band, dtype=float).reshape(-1)
    if b.size != STRIP_SAMPLES:
        raise ValueError(f"a strip band has {b.size} samples, expected {STRIP_SAMPLES}")
    return b


def composite_capabilities(n_unknowns: int | None = None, nu: float = GE.NU,
                           fingerprint: str | None = None,
                           band: Any = None, dt: float = GE.MACRO_DT,
                           rho: float = 1.0) -> ExpertCapabilities:
    """The overset composite as ONE expert, with one port per device.

    ``elliptic_subsolve`` is EMBEDDED and that is the honest value: the pressure
    system is factored once over the whole composite and solved inside the step,
    so R10 has no cut to place -- where the porous column EXPOSES its elliptic
    part to a global projection in the composition layer because its windows cut
    it.  ``time_discretization`` is IMPLICIT (BDF2 with implicit advection) and
    ``substeps_per_macro_step`` is 1: the macro-step IS the step.
    """
    n = RL.RNX * (RL.RNY + 1) if n_unknowns is None else int(n_unknowns)
    u_band = _band_of(band)
    thickness = 2.0 * CU.HALF_WIDTH_CELLS * GE.DX

    def respond(port_name: str, trace: Any) -> np.ndarray:
        #: **Direct forcing, and it is the march's own arithmetic read backwards.**
        #: `car_union` hands the fluid a force and reads the velocity that follows;
        #: a MECH port is asked the other way round, so the response is the force
        #: that would put the band at the trace within one step, integrated across
        #: the strip to a traction.  It is explicit -- it does not solve for the
        #: pressure the change would raise -- which is named in `GRAPH_HOLES`.
        if port_name not in ("core:MECH", "rotor:MECH"):
            raise KeyError(f"the composite has no port {port_name!r}")
        u = np.asarray(trace, dtype=float).reshape(-1)
        if u.size != u_band.size:
            raise ValueError(f"port {port_name} takes {u_band.size} cells, got {u.size}")
        return rho * (u - u_band) * thickness / float(dt)

    return ExpertCapabilities(
        expert_id="FLUID",
        ports=[_strip_port("core:MECH", "the radiator core's plane"),
               _strip_port("rotor:MECH", "the recovery turbine's plane")],
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        elliptic_subsolve=EllipticSubsolve.EMBEDDED,
        time_discretization=TimeDiscretization.IMPLICIT,
        #: the composite's widest reach per step: a three-point interpolation
        #: stencil either side of a fringe point, and a nine-point Laplacian
        stencil_radius=2,
        substeps_per_macro_step=1,
        differentiable=Differentiable.NONE,
        dt_native=GE.MACRO_DT,
        L_native=RL.RNX * GE.DX,
        storage=lambda: 3.0 * n,
        equivariances=(),
        validity=None,
        governing_family=IU.FLUID,
        lambda_ref="itself: the body-fitted overset composite, exposed, at this cell",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash=f"overset-composite-nu{nu:.6g}-n{n}"
                    + (f"-{fingerprint[:12]}" if fingerprint else ""),
        boundary_response=respond,
        probe_base=lambda _port, _u=u_band: _u.copy(),
        reproducibility_floor=np.finfo(float).eps,
        deterministic=True,
        note="one implicit solve over a Cartesian background and eleven "
             "body-fitted grids; the car's walls are no-slip surfaces INSIDE it "
             "and the two devices are body forces on strips of it")


def cross_grid_rows(ov) -> dict[str, Any]:
    """How many equations of the composite couple one grid to another.

    This is the measurement behind *"a grid cannot be stepped alone"*: every
    interpolation row of the momentum and pressure systems reads unknowns that
    belong to a different grid, so there is no sub-problem to hand an expert.
    """
    rows = 0
    entries = 0
    per: dict[str, int] = {}
    for name, spec in ov.donors.items():
        for e in spec["entries"]:
            k = int(np.asarray(e["j"]).size)
            rows += k
            entries += int(np.asarray(e["flat"]).size)
            per[f"{name}<-{e['donor']}"] = per.get(f"{name}<-{e['donor']}", 0) + k
    return {"interpolation_rows": rows, "donor_entries": entries,
            "unknowns": int(ov.n_unknowns), "by_pair": per,
            "fraction_of_unknowns": rows / max(int(ov.n_unknowns), 1)}


# ---------------------------------------------------------------------------
# the devices, each with ONE face
# ---------------------------------------------------------------------------


def _device_strip_port(agent_id: str, label: str, what: str,
                       cells: int = STRIP_SAMPLES) -> Any:
    return port_decl(
        name="strip:MECH", port_type=PortType.MECH,
        geometry=f"{label}, {cells} device cells over {STRIP_CELLS} grid cells",
        direction=Direction.BIDIRECTIONAL, nondim=dict(W.MECH_SCALES),
        effective_resolution=W.modes_for(cells), motion_class=MotionClass.STATIC,
        response_half=ResponseHalf.EFFORT,
        prolongation=W._prolongation(agent_id, "strip:MECH", cells),
        note=what)


def strip_core_capabilities(expert: IU.CoreRadiator, caps: ExpertCapabilities
                            ) -> ExpertCapabilities:
    """RAD's record with ONE face: the strip its loss acts on.

    `integration_union.core_capabilities` declares two faces because the porous
    column's tiling is cut at the core's plane and the two faces land on two
    windows.  Here the core is a body force inside one expert, so it has one
    face and the seam has two agents.  The response is the same law --
    ``1/2 K u^2`` against the air through it -- read as a traction.
    """
    inner = caps.boundary_response
    port = _device_strip_port(
        expert.agent_id, "the radiator core's strip in the body-fitted duct",
        "the core's pressure drop 1/2 K u^2 against the absolute air velocity "
        "through it, as the traction its body force integrates to")

    def respond(port_name, trace):
        if port_name == "strip:MECH":
            return expert.traction(np.asarray(trace, dtype=float).reshape(-1))
        return inner(port_name, trace)

    return replace(caps, ports=list(caps.ports) + [port], boundary_response=respond,
                   weight_hash=f"{caps.weight_hash}-strip-K{expert.k_core:.6g}")


def strip_rotor_capabilities(expert, caps: ExpertCapabilities, width: float,
                             elements: dict | None = None) -> ExpertCapabilities:
    """ROTOR's record with ONE face: the strip its thrust acts on.

    The response solves the union's shared operating point at the trace it is
    handed (`vehicle_march.operating_point`, decision 6) and returns the disk's
    thrust over the band's width.  That is the same number `car_union` applies
    to the fluid, so the declaration and the march cannot drift apart.
    """
    inner = caps.boundary_response
    port = _device_strip_port(
        expert.agent_id, "the recovery turbine's strip in the body-fitted duct",
        "the disk's thrust over the band's width at the operating point the "
        "union solves for the trace it is handed")

    def respond(port_name, trace):
        if port_name == "strip:MECH":
            ring = np.asarray(trace, dtype=float).reshape(-1)
            _res, _els, disk = VM.operating_point(ring, width=width, scale=width,
                                                  elements=elements)
            return np.full(ring.size, float(disk.thrust) / width)
        return inner(port_name, trace)

    return replace(caps, ports=list(caps.ports) + [port], boundary_response=respond,
                   weight_hash=f"{caps.weight_hash}-strip-w{width:.6g}")


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------


def connections(joins: Sequence[str] = ("J1", "J2", "J3")) -> list[Connection]:
    """The body-fitted column's seams: one per device, and J2's.

    There are no fluid-fluid seams: the composite is one expert (W268).
    """
    out: list[Connection] = []
    if "J1" in joins:
        out.append(Connection(
            seam_id="J1_core_strip", a=("RAD", "strip:MECH"), b=("FLUID", "core:MECH"),
            port_type=PortType.MECH, derive_space=True,
            geometrically_coincident=True, expected_null_dim=0,
            #: **W308, corrected at Tier 82.**  This named `RAD`, which made the
            #: assembled operator NEGATIVE definite -- passive only up to a
            #: global sign, which `S lambda = chi` is indifferent to but which
            #: `L4/E7/passivity` is not.  Naming the other side makes it
            #: positive definite with **every singular value bitwise unchanged**
            #: (measured: max relative difference 0.0), so beta and kappa are
            #: untouched and only E7's test moves.  `J2_heat` already named the
            #: right agent, which is the control that made this a finding rather
            #: than a blanket flip.
            effort_normal="FLUID",
            note="J1: the radiator core's loss as a traction on the strip its body "
                 "force acts over, against the band's mean velocity"))
    if "J2" in joins:
        out.append(Connection(
            seam_id="J2_heat", a=("MGU", "case:THERM"), b=("BLOCK", "outer:THERM"),
            port_type=PortType.THERM, derive_space=True,
            geometrically_coincident=True, expected_null_dim=0,
            effort_normal="BLOCK",
            note="J2: the machine's I^2 R into the block's dry face through the "
                 "mount conductance -- unchanged from the porous column"))
    if "J3" in joins:
        out.append(Connection(
            seam_id="J3_rotor_strip", a=("ROTOR", "strip:MECH"), b=("FLUID", "rotor:MECH"),
            port_type=PortType.MECH, derive_space=True,
            geometrically_coincident=True, expected_null_dim=0,
            #: **W308, corrected at Tier 82** -- the same inversion as `J1`, and
            #: see its note. Together the two corrections take this graph from
            #: 191 decisions to 189: `L4/E7/passivity` stops firing on both strip
            #: seams, the 156 admits and the one refusal are unchanged, and the
            #: refusal was never these seams' -- it is the clocks.
            effort_normal="FLUID",
            note="J3: the turbine's thrust over its band as a traction on the strip "
                 "its body force acts over, against the band's mean velocity"))
    return out


def build(joins: Sequence[str] = ("J1", "J2", "J3"), clocks: str = "native",
          u_host: float | None = None, width: float | None = None,
          n_unknowns: int | None = None, fingerprint: str | None = None,
          band: Any = None, flux_matching: FluxMatching = FluxMatching.POINTWISE,
          name: str | None = None) -> tuple[CaseGraph, dict[str, Any]]:
    """The body-fitted column as a `CaseGraph`: one fluid expert, the devices,
    the coolant circuit and the powertrain.

    ``joins=()`` is the disjoint union, the control every join is a difference
    against; ``clocks="reconciled"`` puts every agent on the fluid's step, which
    is what L7/R9 refuses the native version for.
    """
    joins = tuple(j for j in ("J1", "J2", "J3") if j in tuple(joins))
    if clocks not in ("native", "reconciled"):
        raise ValueError(f"clocks is 'native' or 'reconciled', not {clocks!r}")
    width = (STRIP_CELLS * GE.DX) if width is None else float(width)
    u_host = 0.08293773923977797 if u_host is None else float(u_host)
    info: dict[str, Any] = {"joins": joins, "clocks": clocks, "width": width,
                            "u_host": u_host, "holes": list(GRAPH_HOLES)}

    agents = [Agent("FLUID", composite_capabilities(n_unknowns, fingerprint=fingerprint,
                                                    band=band),
                    domain="the whole fluid domain: a Cartesian background and "
                           "eleven body-fitted grids, solved together",
                    role="fluid")]

    # -- the powertrain, sized for the duct the car delivers (decisions 2 and 8)
    elements = RL.machine_for_host(u_host, scale=width)
    pt = dict(PT.make_elements())
    pt.update(elements)
    ring = np.full(CU.RING_SAMPLES, u_host)
    if "J3" in joins:
        pt["ROTOR"] = RL.WA.RotorDisk(agent_id="ROTOR", u_ref=ring)
        op = VM.SizedCircuitSolve(width=width, rotor=pt["ROTOR"], elements=pt,
                                  u_ref=float(np.mean(ring))).solve()
        info["operating_point"] = op.as_dict()
    else:
        pt["ROTOR"] = PT._rotor()
    q_machine = None
    if "J2" in joins:
        p_ref, _i = IU.calibrated_p_ref(PT.MachineAgent())
        current = float(info.get("operating_point", {}).get("current")
                        or pt["MGU"].current_at(pt["MGU"].omega))
        q_machine = current * current * pt["MGU"].resistance * p_ref
        info.update(p_ref=p_ref, current_op=current, q_machine=q_machine)

    # -- the coolant circuit -------------------------------------------------
    dt_c = GE.MACRO_DT if clocks == "reconciled" else CL.MACRO_DT
    cl = dict(CL.make_legs(4))
    if "J1" in joins:
        cl["RAD"] = IU.CoreRadiator(u_air=ring)
    for leg in cl.values():
        leg.dt = dt_c
    block = (IU.MountedBlock(dt=dt_c, q_machine=q_machine) if "J2" in joins
             else CL.BlockAgent(dt=dt_c))
    cl["BLOCK"] = block
    g_cl, _ = CL.build(experts=cl, measured=None)
    for a in g_cl.agents:
        if a.agent_id == "RAD" and "J1" in joins:
            a = replace(a, capabilities=strip_core_capabilities(
                cl["RAD"], IU.core_capabilities(cl["RAD"], a.capabilities)))
        if a.agent_id == "BLOCK" and "J2" in joins:
            a = replace(a, capabilities=IU.mounted_block_capabilities(block, a.capabilities))
        agents.append(a)

    # -- the powertrain ------------------------------------------------------
    dt_p = GE.MACRO_DT if clocks == "reconciled" else PT.MACRO_DT
    if "J2" in joins:
        base = pt["MGU"]
        pt["MGU"] = IU.CooledMachine(current_op=info["current_op"], p_ref=info["p_ref"],
                                     t_case_ref=float(block.t_case), omega=base.omega)
        pt["MGU"].k_t, pt["MGU"].k_e = base.k_t, base.k_e
        pt["MGU"].resistance = base.resistance
        info["t_case_ref"] = float(block.t_case)
    for e in pt.values():
        e.dt = dt_p
    g_pt, _ = PT.build(experts=pt, measured=None)
    for a in g_pt.agents:
        if a.agent_id == "ROTOR" and "J3" in joins:
            a = replace(a, capabilities=strip_rotor_capabilities(
                pt["ROTOR"], IU.host_rotor_capabilities(pt["ROTOR"], a.capabilities),
                width, pt))
        if a.agent_id == "MGU" and "J2" in joins:
            a = replace(a, capabilities=IU.cooled_machine_capabilities(pt["MGU"], a.capabilities))
        agents.append(a)

    conns = connections(joins) + list(g_cl.connections) + list(g_pt.connections)
    families = {a.agent_id: a.capabilities.governing_family for a in agents}
    #: **NON_OVERLAPPING, and that is the change.**  The porous column declares
    #: OVERLAPPING because its fourteen windows share halos; this column's fluid
    #: is one expert, so no seam in the graph has an overlap at all.
    graph_axis = Decomposition.NON_OVERLAPPING

    def axis(c):
        fa, fb = families[c.a[0]], families[c.b[0]]
        return None if fa == fb else None

    conns = [replace(c, cut_axis=axis(c)) for c in conns]
    graph = CaseGraph(
        name=name or ("racelab-body-fitted-"
                      + ("+".join(joins) if joins else "disjoint")
                      + ("" if clocks == "native" else f"-clocks-{clocks}")),
        agents=agents, connections=conns, decomposition=graph_axis,
        overlap={}, overlap_cells={}, partition_of_unity={},
        global_fields=[], cross_points=(),
        loop_gains=tuple(lg for g in (g_cl, g_pt) for lg in g.loop_gains),
        macro_dt=GE.MACRO_DT, flux_matching=flux_matching, measured=None,
        note=("PoC 3 phase 1 on body-fitted grids: ONE fluid expert -- a Cartesian "
              "background and eleven curvilinear grids solved together -- with the "
              "radiator core and the recovery turbine on strip MECH ports, joined by "
              + (", ".join(joins) if joins else "nothing")))
    info["agent_count"] = len(agents)
    info["seam_count"] = len(conns)
    return graph, {"info": info, "experts": {**cl, **pt}}


# ---------------------------------------------------------------------------
# the two controls
# ---------------------------------------------------------------------------


def no_port_record() -> dict[str, Any]:
    """Build the honest zero-port record and report what the record layer says."""
    try:
        composite_capabilities().__class__(expert_id="FLUID", ports=[])
    except RecordError as exc:
        return {"refused": True, "message": str(exc),
                "why": "the composite's only couplings are domain boundaries, which "
                       "carry no port, and two body forces"}
    return {"refused": False, "message": None}       # pragma: no cover


def self_seam_graph(joins: Sequence[str] = ("J1", "J2", "J3")):
    """The declaration the porous column's shape would suggest: the device's two
    faces on the fluid, which here is ONE agent -- a seam from it to itself."""
    graph, aux = build(joins=joins)
    conns = list(graph.connections) + [Connection(
        seam_id="J3_rotor_self", a=("FLUID", "core:MECH"), b=("FLUID", "rotor:MECH"),
        port_type=PortType.MECH, derive_space=True, geometrically_coincident=False,
        expected_null_dim=0,
        note="the porous column's shape, on a column nothing cuts at the device "
             "plane: the fluid on both sides of the device is the same agent")]
    return replace(graph, connections=conns, name=graph.name + "-self-seam"), aux
