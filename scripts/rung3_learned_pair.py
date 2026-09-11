"""Rung 3 -- the first Atlas graph holding TWO learned experts, of different families.

`f1-pathmap-and-end-goal` section 3 defines rung 3 as *"a second learned expert joins an
existing graph; first true multi-family coupling"*, and section 3.3 records that no graph in
`atlas/cases` has ever held two.  The substitution campaign tests a different
arrangement -- one learned expert replacing one classical one, seam by seam -- so
neither its negatives nor its positives answer this.  This script builds the
configuration rung 3 actually names and compiles it.

**The template the brief suggested is not coherent, and that is stage 0.**
`thermal_seam` couples a gas to a shell across a `THERM` port.  Neither learned
expert has one: Poseidon-T declares four `face:MECH` ports and returns
``nu du/dn``; NeuberNet takes displacements and returns tractions.  Both are pure
mechanics and neither carries a temperature, so a learned `thermal_seam` has no
counterpart to declare.  `MECH` is the only bond available, which makes the
rung-3 graph a learned fluid-structure seam -- `wing_fsi`'s shape, not
`thermal_seam`'s.

**And the seam is a geometric fiction, deliberately and in the open.**
NeuberNet's ring at ``5 R_n`` is an INTERNAL CUT in a solid: the rest of the body
imposes displacement there.  A fluid face is a WETTED SURFACE.  Handing fluid
traction to an internal cut is physically wrong, and `PortDecl.geometry` is free
text, so nothing in the schema can tell the two apart -- which is `Agent.domain`'s
free-text problem (R10's docstring) one object down.  That silence is a finding
and it is reported as one; it is not a property the fixture earns.

**The 2x2, so the result is attributable.**  Four graphs at ONE declared
interface space, differing only in which side is learned:

                    solid: NeuberNet      solid: classical patch
    fluid: Poseidon-T     LL  (rung 3)         LC
    fluid: WindowNS       CL                   CC  (the classical incumbent)

The fluid pair sits on the checkpoint's own 128-cell geometry, which
`neural_interface` built precisely so a learned and a classical column are
comparable.  The solid pair sits on CS-S2's 29-sensor ring with the identical
port, trace and Galerkin flux.  Every cell is compiled and the four decision
records are diffed.

Writes ``out/rung3/rung3.json``.  Needs torch, the Poseidon-T checkpoint in the
HuggingFace cache, and NeuberNet in ``~/.cache/neubernet`` -- which is
UNLICENSED and local-only: nothing derived from it beyond measured numbers is
written here.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import dataclasses
import io
import json
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

# Ampere+ turns TF32 on for float32 matmuls by default, which silently drops a
# checkpoint's mantissa. Asserted rather than trusted -- both flags, because
# cudnn and matmul carry separate ones.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert torch.backends.cuda.matmul.allow_tf32 is False
assert torch.backends.cudnn.allow_tf32 is False
# One thread beats eight on this machine for tensors this small.
torch.set_num_threads(1)

from atlas import Budget, compile_scheme                              # noqa: E402
from atlas.capability import (                                        # noqa: E402
    BCChannel,
    ClaimType,
    Differentiable,
    Direction,
    EllipticSubsolve,
    ExpertCapabilities,
    MotionClass,
    TimeDiscretization,
    port_decl,
)
from atlas.cases import neural_interface as NI                        # noqa: E402
from atlas.cases import poseidon as PO                                # noqa: E402
from atlas.cases.window_ns import H, fourier_basis                    # noqa: E402
from atlas.graph import Agent, CaseGraph, Connection, Decomposition   # noqa: E402
from atlas.ports import PortType, ResponseHalf                        # noqa: E402
from atlas.probe import ProbeBudget                                   # noqa: E402
from atlas.transfer import InterfaceSpace, Prolongation               # noqa: E402
from scripts import cs_s2_elastic_patch as EP                         # noqa: E402
from scripts import cs_s2_neubernet as NB                             # noqa: E402

OUT = os.path.join(HERE, "out", "rung3")

#: CS-S2's geometry, unchanged, so the donor is the object that study measured.
ALPHA, R, SY_E, ET_E, NU_S = 30.0, 50.0, 3e-3, 1e-2, 0.3

#: The probe amplitude CS-S2 used on this ring.
AMPLITUDE = 1.0e-2


def _stage(res, name, t0):
    res.setdefault("stage_seconds", {})[name] = time.perf_counter() - t0
    return time.perf_counter()


# ===========================================================================
# 0 -- coherence: what bond can these two experts possibly share?
# ===========================================================================


def coherence(pos_caps, nn_ports) -> dict:
    """Read the port types off the declarations. No forward passes.

    The brief's suggestion was `thermal_seam` -- Poseidon-T on the gas side and
    NeuberNet on the shell side, across a `THERM` port. This is the check that
    says whether that is constructible, and it is decidable before anything is
    loaded.
    """
    poseidon_types = sorted({p.port_type.value for p in pos_caps.ports})
    return {
        "template_suggested": "thermal_seam (Compressible2D vs ThermoStruct2D, THERM)",
        "thermal_seam_port_type": "THERM",
        "poseidon_port_types": poseidon_types,
        "poseidon_ports": [p.name for p in pos_caps.ports],
        "neubernet_port_types": ["MECH"],
        "neubernet_ports": nn_ports,
        "neubernet_trace": "displacement at the material sensors",
        "neubernet_flux": "traction, hat-weighted (the Galerkin dual)",
        "either_declares_THERM": False,
        "verdict": (
            "NOT constructible. thermal_seam's bond is THERM -- temperature "
            "against entropy flux -- and neither learned expert carries a "
            "temperature field at all. Poseidon-T is incompressible NS and "
            "returns nu du/dn; NeuberNet is quasi-static mechanics and returns "
            "a traction. MECH is the only port type both can declare, so the "
            "rung-3 graph is a learned FLUID-STRUCTURE seam -- wing_fsi's shape "
            "-- and not thermal_seam's."
        ),
        "geometry_caveat": (
            "NeuberNet's ring at 5 R_n is an INTERNAL CUT in a solid; a fluid "
            "face is a WETTED SURFACE. Coupling them is physically wrong and "
            "PortDecl.geometry is free text, so no rule can see it. Declared "
            "here in the open and reported as a finding about the schema."
        ),
    }


# ===========================================================================
# 1 -- the solid side: two experts on one ring, and the donor's port measured
# ===========================================================================


def solid_experts(res) -> tuple:
    """NeuberNet and the classical patch, on CS-S2's ring at CS-S2's base."""
    t0 = time.perf_counter()
    model = NB.load_model()
    assert torch.set_flush_denormal(True), "denormals not flushed"
    big = EP.ElasticPatch(ALPHA, R, NU_S, r_out=25.0, h_tip=0.04, grad=0.03, h_max=0.8)
    patch = EP.ElasticPatch(ALPHA, R, NU_S, r_out=EP.RL, h_tip=0.04, grad=0.03,
                            h_max=0.2, agent_id="SOLID")
    t0 = _stage(res, "solid_build", t0)

    # CS-S2's tension-elastic base, rebuilt the same way: a classical far field on
    # a disc of 25 R_n, interpolated onto the 36 sensors and scaled so the elastic
    # von Mises reaches 0.7 sigma_y.
    th36 = np.radians(EP.SENSOR_ANGLES)
    pts36 = np.stack([EP.RL * np.cos(th36), EP.RL * np.sin(th36)], axis=1)
    u_ex, ut_ex = EP.homogeneous_field(big.nodes, R, NU_S, axial=1.0, shear=0.0)
    u, ut, _t, _f = big.solve_nodal(u_ex[big.B, 0], u_ex[big.B, 1], ut_ex[big.B])
    unit_t = np.nan_to_num(big.interpolate(np.column_stack([u[:, 0], u[:, 1], ut]),
                                           pts36))
    probe0 = NB.NeuberNetPatch(model, ALPHA, R, SY_E, ET_E, NU_S)
    y0 = probe0.regime(unit_t)["elastic_von_mises_over_sy"]
    base36 = (0.7 / y0) * unit_t
    nn = NB.NeuberNetPatch(model, ALPHA, R, SY_E, ET_E, NU_S, base36=base36,
                           agent_id="SOLID")
    res["solid"] = {
        "base": "tension-elastic, CS-S2's construction, elastic von Mises 0.7 sigma_y",
        "regime": nn.regime(base36),
        "material_sensors": int(nn.M),
        "classical_nodes": int(patch.nodes.shape[0]),
    }
    _stage(res, "solid_base", t0)
    return nn, patch, base36


def measure_port(responder, label, n, base=None) -> dict:
    """The 29x29 `ring:ux` block, and the spectral cutoff read off it.

    `CASE-STUDY-GUIDE`: ``effective_resolution`` is *"the expert's own measured
    spectral cutoff, not a chosen number"*.  Nobody had measured NeuberNet's, so
    this does -- one column per sensor, CS-S2's amplitude, CS-S2's Galerkin flux.
    """
    t0 = time.perf_counter()
    f0 = np.asarray(responder.respond("ring:ux", np.zeros(n)), dtype=float)
    cols = []
    for j in range(n):
        e = np.zeros(n)
        e[j] = AMPLITUDE
        cols.append((np.asarray(responder.respond("ring:ux", e), dtype=float) - f0)
                    / AMPLITUDE)
    S = np.column_stack(cols)
    sv = np.linalg.svd(S, compute_uv=False)
    total = float(np.sum(sv ** 2))
    csum = np.cumsum(sv ** 2) / total
    rank99 = int(np.searchsorted(csum, 0.99) + 1)
    rank999 = int(np.searchsorted(csum, 0.999) + 1)

    # Reproducibility: the same trace twice, through the same object.
    a = np.asarray(responder.respond("ring:ux", np.zeros(n)), dtype=float)
    b = np.asarray(responder.respond("ring:ux", np.zeros(n)), dtype=float)
    denom = float(np.linalg.norm(a)) or 1.0
    repeat = float(np.linalg.norm(a - b) / denom)

    return {
        "label": label,
        "n": int(n),
        "norm2": float(np.linalg.norm(S, 2)),
        "singular_values": [float(x) for x in sv[:10]],
        "effective_rank_99": rank99,
        "effective_rank_999": rank999,
        "repeat_call_relative": repeat,
        "seconds": time.perf_counter() - t0,
    }


# ===========================================================================
# 2 -- the capability records
# ===========================================================================

#: Nondimensionalization on the solid side. CS-S2's trace is u E/(sigma_y R_n)
#: and its flux is sigma.n/sigma_y, so both are already dimensionless and the
#: three scales are 1 by construction -- which satisfies s_e * s_f = s_P exactly.
SOLID_SCALES = {"stress": 1.0, "velocity": 1.0, "power_area": 1.0}


def solid_capabilities(agent_id, responder, learned: bool, port_res: int,
                       floor: float, prolongation) -> ExpertCapabilities:
    """One record for either solid-side expert. Only four fields differ.

    The four that differ are exactly the four rung 3 is about: whether the
    expert is a checkpoint, what it can be a referent for, what its elliptic
    content is, and what its reproducibility floor is.
    """
    return ExpertCapabilities(
        expert_id=agent_id,
        ports=[port_decl(
            name="ring:ux", port_type=PortType.MECH,
            geometry=("the material arc of the ring at 5 R_n around the notch, "
                      f"{responder.M} sensors -- an INTERNAL CUT in the solid, "
                      "not a wetted surface. The schema cannot say that"),
            direction=Direction.BIDIRECTIONAL, nondim=dict(SOLID_SCALES),
            effective_resolution=port_res, motion_class=MotionClass.STATIC,
            response_half=ResponseHalf.EFFORT,
            prolongation=prolongation,
            note="quasi-static: displacement in, hat-weighted traction out")],
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=False,
        # **Measured, not declared.** CS-S2's `support_reach` on this port reads
        # the response nonzero in all 29 sensors, `is_global: True`,
        # `consistent_with: ['embedded']`. The classical patch is a global
        # Dirichlet-to-Neumann solve on its disc, which is EMBEDDED in R10's
        # exact sense and is what `wing_fsi`'s STRUCT declares.
        elliptic_subsolve=EllipticSubsolve.EMBEDDED,
        # A learned one-shot map is neither explicit nor implicit (poseidon.py's
        # own reasoning); the classical patch is a quasi-static linear solve.
        time_discretization=(TimeDiscretization.UNKNOWN if learned
                             else TimeDiscretization.IMPLICIT),
        stencil_radius=1,
        substeps_per_macro_step=1,
        differentiable=Differentiable.NONE,
        dt_native=PO.MACRO_DT,
        L_native=EP.RL,
        storage=None,
        equivariances=(),
        validity=None,
        governing_family="elastic-plastic-notch-2d",
        # **W95, and it is the whole point of the arm.** CS-S2 section 9: NeuberNet is
        # defined on ONE disc, so there is no larger or undecomposed domain to
        # evaluate it on and it cannot be its own referent. The classical patch
        # is its own referent on the same disc at a finer mesh.
        lambda_ref=(None if learned else
                    "the same patch at h_tip 0.02 / h_max 0.1; the real solver "
                    "is its own reference"),
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash=("neubernet-grossIt-cce93244-production" if learned
                     else "cs_s2_elastic_patch/1-linear-elastic-h0.04"),
        boundary_response=responder.respond,
        probe_base=(lambda _p, _n=responder.M: np.zeros(_n)),
        reproducibility_floor=floor,
        deterministic=True,
        note=("NeuberNet, frozen, 22.45M parameters, production mode; UNLICENSED "
              "and local-only, nothing but measured numbers leaves the machine"
              if learned else
              "classical linear-elastic patch on the same ring, CS-S2's control"),
    )


def solid_prolongation(agent_id, n, dim_m, hat_mass) -> Prolongation:
    """M -> the 29 sensors, a real Fourier basis in the sensor index.

    ``gram_V`` is the rho-weighted hat mass the adapter already computes, which
    is the inner product that makes trace times flux the interface power -- so
    the forced adjoint is the Galerkin reduction rather than a hand-written one.
    """
    return Prolongation(
        agent_id=agent_id, port_name="ring:ux",
        matrix=fourier_basis(n, dim_m),
        gram_V=np.diag(np.asarray(hat_mass, dtype=float)),
        label=f"{dim_m}-mode real Fourier basis on the material arc",
    )


def fluid_prolongation(agent_id, dim_m) -> Prolongation:
    """M -> the 128 face cells. `poseidon.face_prolongation` at a general dim."""
    return Prolongation(
        agent_id=agent_id, port_name="xhi:MECH",
        matrix=fourier_basis(PO.EXPERT_RES, dim_m),
        gram_V=H * np.eye(PO.EXPERT_RES),
        label=f"{dim_m}-mode real Fourier basis (probed-dtn-coupling 2.2)",
    )


# ===========================================================================
# 3 -- the 2x2
# ===========================================================================


def build_cell(fluid_caps, solid_caps, dim_m, name) -> CaseGraph:
    """Two agents, one MECH seam, a declared non-conforming transfer.

    ``derive_space`` is unavailable here and correctly so: the two sides'
    effective resolutions differ, so the conforming special case does not apply
    and the amendment requires a declared M with one prolongation per side.
    """
    space = InterfaceSpace(seam_id="fsi", dim=dim_m,
                           note="declared: a non-conforming learned FSI seam")
    return CaseGraph(
        name=name,
        agents=[Agent("FLUID", fluid_caps, domain="one 128-cell window"),
                Agent("SOLID", solid_caps, domain="the notch patch inside 5 R_n")],
        connections=[Connection(
            seam_id="fsi",
            a=("FLUID", "xhi:MECH"),
            b=("SOLID", "ring:ux"),
            port_type=PortType.MECH,
            space=space,
            geometrically_coincident=False,
            note="a fluid face against an internal material cut: NOT coincident, "
                 "and not physically a seam either. Declared in the open",
        )],
        # Two agents owning their own regions and their own equations -- the
        # multiphysics arrangement, not a tiling. `wing_fsi` declares the same.
        decomposition=Decomposition.NON_OVERLAPPING,
        macro_dt=PO.MACRO_DT,
        measured=None,
        note="rung 3: the first graph holding two learned experts of different "
             "governing families",
    )


def decisions(result):
    return [(d.layer, d.rule, d.verdict.value, d.subject)
            for d in result.decisions.decisions]


def compile_cell(graph) -> dict:
    r = compile_scheme(graph, Budget(), ProbeBudget(fd_step=AMPLITUDE))
    op = r.seam_operators.get("fsi")
    return {
        "verdict": r.verdict.value,
        "runnable": bool(r.runnable),
        "envelope": list(r.envelope.tuple()),
        "n_decisions": len(r.decisions.decisions),
        "decisions": [list(x) for x in decisions(r)],
        # Every message, so a page can quote a rule rather than paraphrase it and
        # a test can assert the quote. Tier 39's discipline: a figure in a page and
        # a figure in a rule must not be able to drift apart.
        "messages": {f"{d.layer}/{d.rule}/{d.subject}": d.message
                     for d in r.decisions.decisions},
        "refusals": [[d.layer, d.rule, d.subject, d.message[:400]]
                     for d in r.decisions.decisions if d.verdict.value == "refuse"],
        "decertifications": sorted({f"{d.layer}/{d.rule}"
                                    for d in r.decisions.decisions
                                    if d.verdict.value == "admit-uncertified"}),
        "tau_undefined_seams": list(r.tau_undefined_seams),
        "unmeasured": list(r.unmeasured),
        "beta": None if op is None else float(op.beta),
        "kappa": None if op is None else float(op.kappa),
        "norm_S": None if op is None else float(np.linalg.norm(op.S, 2)),
        "passivity_defect": None if op is None else float(op.passivity_defect),
        "blocks": {} if op is None else {
            k: float(np.linalg.norm(b.S, 2)) for k, b in op.blocks.items()},
    }


# ===========================================================================
# 4 -- R10's positive control
# ===========================================================================


def r10_positive_control() -> dict:
    """Same shape, same rule, one declaration changed: does R10 still refuse?

    Rung 3's zero refusals rest on W114's premise not being met -- each agent is
    the sole one of its `governing_family`, so the decomposition does not cut
    either. A rule that did not fire is evidence only beside a cell where it
    does. Here both agents declare ONE family and one declares EMBEDDED, which
    is the arrangement R10 was derived on, and nothing else changes.
    """
    from atlas.capability import linear_response

    dim = 4

    def caps(agent_id, family, elliptic):
        return ExpertCapabilities(
            expert_id=agent_id,
            ports=[port_decl(
                name="f:MECH", port_type=PortType.MECH, geometry="one face",
                direction=Direction.BIDIRECTIONAL, nondim=dict(SOLID_SCALES),
                effective_resolution=dim, motion_class=MotionClass.STATIC,
                response_half=ResponseHalf.EFFORT)],
            bc_channel=BCChannel.DIRICHLET, bc_time_varying=True,
            elliptic_subsolve=elliptic,
            time_discretization=TimeDiscretization.IMPLICIT,
            stencil_radius=1, substeps_per_macro_step=1,
            differentiable=Differentiable.NONE, deterministic=True,
            reproducibility_floor=float(np.finfo(float).eps),
            dt_native=PO.MACRO_DT, L_native=1.0,
            storage=lambda u: 0.5 * float(np.dot(np.ravel(u), np.ravel(u))),
            validity=lambda state, cond=None: True,
            governing_family=family,
            lambda_ref="itself",
            claim_types=frozenset({ClaimType.TRAJECTORY}),
            weight_hash=f"control/{agent_id}",
            boundary_response=linear_response(2.0 * np.eye(dim)),
            note="R10 positive control, declaration only")

    def graph(fam_b, name):
        return CaseGraph(
            name=name,
            agents=[Agent("A", caps("A", "one-family", EllipticSubsolve.EMBEDDED)),
                    Agent("B", caps("B", fam_b, EllipticSubsolve.NONE))],
            connections=[Connection(
                seam_id="s", a=("A", "f:MECH"), b=("B", "f:MECH"),
                port_type=PortType.MECH, derive_space=True,
                geometrically_coincident=True)],
            decomposition=Decomposition.NON_OVERLAPPING,
            macro_dt=PO.MACRO_DT)

    out = {}
    for label, fam in (("same_family", "one-family"), ("different_family", "other-family")):
        r = compile_scheme(graph(fam, f"r10-control-{label}"), Budget(), ProbeBudget())
        out[label] = {
            "verdict": r.verdict.value,
            "runnable": bool(r.runnable),
            "refusing_rules": sorted({f"{d.layer}/{d.rule}"
                                      for d in r.decisions.decisions
                                      if d.verdict.value == "refuse"}),
        }
    out["reads"] = (
        "R10 refuses the EMBEDDED agent when a second agent of its own family is "
        "present and admits it when there is not. Rung 3's graph is the second "
        "case, so its zero refusals are the premise clearing rather than the rule "
        "being absent.")
    return out


# ===========================================================================


def main() -> None:
    t_start = time.perf_counter()
    os.makedirs(OUT, exist_ok=True)
    res = {"what": "rung 3 -- two learned experts of different families in one graph",
           "date": "2026-09-10",
           "torch": {"version": torch.__version__,
                     "threads": torch.get_num_threads(),
                     "flush_denormal": True,
                     "tf32_matmul": torch.backends.cuda.matmul.allow_tf32}}

    # -- the solid side --------------------------------------------------
    nn, patch, _base = solid_experts(res)

    t0 = time.perf_counter()
    res["port"] = {
        "neubernet": measure_port(nn, "NeuberNet ring:ux", nn.M),
        "classical": measure_port(patch, "classical patch ring:ux", patch.M),
    }
    _stage(res, "port_measurement", t0)

    # -- the fluid side --------------------------------------------------
    t0 = time.perf_counter()
    u_full, v_full = NI.load_state(HERE)
    tiling = PO.DEFAULT_TILING
    sub = tiling.mono_n
    us, vs = tiling.cut(np.asarray(u_full)[:sub, :sub]), tiling.cut(np.asarray(v_full)[:sub, :sub])
    pos = PO.PoseidonAgent(agent_id="FLUID", u0=us[0], v0=vs[0],
                           shared_faces=("xhi",), dt=PO.MACRO_DT, nu=PO.NU)
    win = NI.make_experts(np.asarray(u_full)[:sub, :sub],
                          np.asarray(v_full)[:sub, :sub],
                          tiling, PO.MACRO_DT, PO.NU, True)[tiling.names[0]]
    win.agent_id = "FLUID"
    _stage(res, "fluid_build", t0)

    pos_caps = PO.poseidon_capabilities(pos)
    res["coherence"] = coherence(pos_caps, ["ring:ux", "ring:uy", "ring:ut", "ring:all"])

    # -- dim M, and what it would be per cell ----------------------------
    nn_res = res["port"]["neubernet"]["effective_rank_99"]
    cl_res = res["port"]["classical"]["effective_rank_99"]
    dim_m = int(min(PO.M_EFF, nn_res, cl_res))
    res["dim_M"] = {
        "declared_for_every_cell": dim_m,
        "poseidon_m_eff": int(PO.M_EFF),
        "windowns_m_eff": int(PO.M_EFF),
        "neubernet_m_eff_measured": nn_res,
        "classical_m_eff_measured": cl_res,
        "per_cell_if_each_declared_its_own": {
            "LL": int(min(PO.M_EFF, nn_res)),
            "LC": int(min(PO.M_EFF, cl_res)),
            "CL": int(min(PO.M_EFF, nn_res)),
            "CC": int(min(PO.M_EFF, cl_res)),
        },
        "why_one_value": (
            "dim M = min_i m_i_eff would differ between cells, so the diff would "
            "conflate 'the solid side is learned' with 'the interface space is "
            "smaller'. Every cell is declared at the smallest honest value so "
            "the only thing that varies is which expert answers."),
    }

    # -- the four cells --------------------------------------------------
    fluid_p = fluid_prolongation("FLUID", dim_m)
    solid_p_nn = solid_prolongation("SOLID", nn.M, dim_m, nn.hat_mass)
    solid_p_cl = solid_prolongation("SOLID", patch.M, dim_m, patch.hat_mass)

    pos_caps = PO.poseidon_capabilities(pos)
    pos_caps.ports[0].effective_resolution = PO.M_EFF
    pos_caps.ports[0].prolongation = fluid_p
    # **`lambda_ref`, and it is a correction rather than a convenience.**
    # `window_ns.window_capabilities` leaves it None, which is right for a
    # TILING: the referent there is the monolith the windows were cut from.
    # At a MULTIPHYSICS seam the same solver is its own referent, and
    # `wing_fsi` declares exactly that on this expert (line 969). Without it
    # every cell has tau UNDEFINED and the 2x2 cannot attribute it.
    win_caps = dataclasses.replace(
        NI.window_capabilities(win),
        lambda_ref='itself: reference.WindowNS, exposed, at this cell '
                   '(wing_fsi declares the same on the same solver)')
    win_caps.ports[0].prolongation = fluid_p

    nn_caps = solid_capabilities("SOLID", nn, True, nn_res,
                                 max(res["port"]["neubernet"]["repeat_call_relative"],
                                     float(np.finfo(np.float32).eps)), solid_p_nn)
    cl_caps = solid_capabilities("SOLID", patch, False, cl_res,
                                 float(np.finfo(float).eps), solid_p_cl)

    cells = {
        "LL": (pos_caps, nn_caps, "rung3-poseidon-neubernet"),
        "LC": (pos_caps, cl_caps, "rung3-poseidon-classical"),
        "CL": (win_caps, nn_caps, "rung3-windowns-neubernet"),
        "CC": (win_caps, cl_caps, "rung3-windowns-classical"),
    }
    res["cells"] = {}
    for key, (fc, sc, nm) in cells.items():
        t0 = time.perf_counter()
        res["cells"][key] = compile_cell(build_cell(fc, sc, dim_m, nm))
        res["cells"][key]["seconds"] = time.perf_counter() - t0
        res["cells"][key]["fluid"] = fc.expert_id
        res["cells"][key]["name"] = nm
        print("%-3s %-22s %-18s refusals %d  decert %d"
              % (key, nm, res["cells"][key]["verdict"],
                 len(res["cells"][key]["refusals"]),
                 len(res["cells"][key]["decertifications"])))

    # -- the diff --------------------------------------------------------
    base = res["cells"]["CC"]["decisions"]
    res["diff"] = {
        k: {
            "identical_to_CC": res["cells"][k]["decisions"] == base,
            "verdict_same_as_CC": res["cells"][k]["verdict"] == res["cells"]["CC"]["verdict"],
            "refusing_rules": sorted({f"{r[0]}/{r[1]}" for r in res["cells"][k]["refusals"]}),
            "only_here": sorted(
                {f"{d[0]}/{d[1]}/{d[2]}" for d in res["cells"][k]["decisions"]}
                - {f"{d[0]}/{d[1]}/{d[2]}" for d in base}),
            "missing_here": sorted(
                {f"{d[0]}/{d[1]}/{d[2]}" for d in base}
                - {f"{d[0]}/{d[1]}/{d[2]}" for d in res["cells"][k]["decisions"]}),
        }
        for k in ("LL", "LC", "CL")
    }
    # -- the elliptic axis, W60's own instruction ------------------------
    # `poseidon_capabilities` defaults `elliptic_subsolve` to NONE and W60
    # records that as a declaration nobody measured. W93 measured it --
    # support_reach 64 of 128, elliptic_signature agreeing -- so EMBEDDED is
    # the measured value. R10 refuses an EMBEDDED agent only when the graph
    # holds ANOTHER agent of the same governing_family (W114). Here there is
    # none, so the prediction is that the measured declaration changes
    # nothing. Compiled rather than predicted.
    res["elliptic_axis"] = {}
    for label, ell in (("none", EllipticSubsolve.NONE),
                       ("embedded", EllipticSubsolve.EMBEDDED),
                       ("unknown", EllipticSubsolve.UNKNOWN)):
        pc = PO.poseidon_capabilities(pos, elliptic=ell)
        pc.ports[0].prolongation = fluid_p
        cell = compile_cell(build_cell(pc, nn_caps, dim_m,
                                       f"rung3-LL-elliptic-{label}"))
        res["elliptic_axis"][label] = {
            "verdict": cell["verdict"], "runnable": cell["runnable"],
            "refusals": cell["refusals"],
            "decertifications": cell["decertifications"],
            "same_decisions_as_LL": cell["decisions"] == res["cells"]["LL"]["decisions"],
        }
        print("elliptic=%-9s %-18s refusals %d"
              % (label, cell["verdict"], len(cell["refusals"])))

    res["r10_positive_control"] = r10_positive_control()
    print("R10 control: same family ->", res["r10_positive_control"]["same_family"]["verdict"],
          "| different family ->", res["r10_positive_control"]["different_family"]["verdict"])

    res["expert_calls"] = {"poseidon": int(pos.n_calls), "neubernet": int(nn.calls)}
    res["elapsed_seconds"] = time.perf_counter() - t_start

    path = os.path.join(OUT, "rung3.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print()
    print("dim M declared for every cell:", dim_m,
          "(poseidon %d, neubernet measured %d, classical measured %d)"
          % (PO.M_EFF, nn_res, cl_res))
    for k in ("LL", "LC", "CL"):
        print("%-3s vs CC: identical=%s  only here: %s"
              % (k, res["diff"][k]["identical_to_CC"], res["diff"][k]["only_here"]))
    print("tau undefined seams, LL:", res["cells"]["LL"]["tau_undefined_seams"])
    print("wrote", path, "in %.1f s" % res["elapsed_seconds"])


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    main()
