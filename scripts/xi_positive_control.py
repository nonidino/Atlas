"""Section 4.5's positive control: a periodic window must score Xi = 0 exactly.

`probed-dtn-coupling` section 4.5 defines the composability index

    Xi_i := ||Lambda_i^expert|| / ||Lambda_i^ref||

and predicts **exactly zero** for a frozen periodic-window checkpoint, because a
periodic window has no boundary to impose a datum on -- ``Lambda == 0``, not
"small".  Section 9's falsifiable table row 3 says "anything nonzero would mean
the periodic window has a boundary channel nobody has found, which would be a
larger finding than this page".

`tier0-measurements` section 8.3 measured the FIRST nonzero Xi in this vault
(0.19-0.53, embedded against exposed) and recorded that the positive control was
still unrun.  This runs it.

**Why it is worth the hour.** Every nonzero Xi rests on the probe reporting zero
when there is nothing to report.  A probe that manufactured a small operator out
of roundoff, an initial-condition leak, or an off-by-one in the ring index would
still produce plausible numbers on the Dirichlet agent, and nothing measured so
far would catch it.  This is the only measurement that can.

The expert is `reference.SpectralNS` -- the same vorticity-streamfunction solver
the frozen-checkpoint harness uses -- whose ``step`` takes no boundary argument
at all.  The whole graph goes through `compile_scheme`, not a hand-rolled probe,
so what is under test is the path a real case study uses.

Run:  python scripts/xi_positive_control.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas import ProbeBudget, compile_scheme                      # noqa: E402
from atlas.capability import (                                     # noqa: E402
    BCChannel, Differentiable, Direction, EllipticSubsolve, ExpertCapabilities,
    TimeDiscretization, port_decl,
)
from atlas.cases import window_ns as W                             # noqa: E402
from atlas.capability import MotionClass                           # noqa: E402
from atlas.graph import Agent, CaseGraph, Connection, Decomposition  # noqa: E402
from atlas.ports import PortType                                   # noqa: E402

M_EFF = W.M_EFF


@dataclass
class PeriodicWindowAgent:
    """`SpectralNS` on one window, offered a boundary trace it cannot accept.

    ``respond`` is written to be as close as possible to `window_ns.WindowAgent`'s
    -- same signature, same flux, same ring indices, same macro-step -- and
    differs in exactly one way: the solver's ``step`` has no ``bc`` parameter, so
    the trace has nowhere to go.  That single difference is what the control is
    about, and making everything else identical is what makes it a control.
    """

    agent_id: str
    u0: np.ndarray
    v0: np.ndarray
    dt: float = W.MACRO_DT
    nu: float = W.NU
    n_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        ref = W.load_reference()
        self.u0 = np.asarray(self.u0, float)
        self.v0 = np.asarray(self.v0, float)
        self.n = int(self.u0.shape[0])
        self.h = (self.n * W.H) / self.n
        self._solver = ref.SpectralNS(nu=self.nu, length=self.n * W.H, n=self.n,
                                      cfl=0.4)

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        face = port_name.split(":", 1)[0]
        ring, interior = {
            "xlo": ((slice(None), 0), (slice(None), 1)),
            "xhi": ((slice(None), -1), (slice(None), -2)),
            "ylo": ((0, slice(None)), (1, slice(None))),
            "yhi": ((-1, slice(None)), (-2, slice(None))),
        }[face]
        trace = np.asarray(trace, float).reshape(-1)
        if trace.shape[0] != self.n:
            raise ValueError(f"trace on {port_name} has length {trace.shape[0]}")

        # The trace IS constructed, exactly as the Dirichlet agent constructs it.
        # There is simply no argument to hand it to: a periodic window has no
        # boundary. Building it and then dropping it is the honest depiction --
        # the agent is offered the datum and has nowhere to put it.
        _unused_ring_datum = trace

        u1, v1 = self._solver.step(self.u0[None], self.v0[None], self.dt)
        self.n_calls += 1
        w = u1[0] if face in ("xlo", "xhi") else v1[0]
        return self.nu * (w[ring] - w[interior]) / self.h


def capabilities(expert: PeriodicWindowAgent) -> ExpertCapabilities:
    """The record a periodic window honestly declares.

    ``bc_channel = NONE`` is the load-bearing one, and CASE-STUDY-GUIDE's fifth
    common mistake is expecting a nonzero coupling from an agent that declares it.
    """
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        # The SAME declared prolongation the Dirichlet case study uses, so the
        # probe hands this agent raw face values exactly as it hands them to
        # `WindowAgent`. Anything else would make the control test the transfer
        # layer rather than the boundary channel.
        ports=[port_decl(name=f"{f}:MECH", port_type=PortType.MECH,
                         geometry=f"{f} face of {expert.agent_id}",
                         direction=Direction.BIDIRECTIONAL,
                         nondim=dict(W.MECH_SCALES),
                         effective_resolution=M_EFF,
                         motion_class=MotionClass.STATIC,
                         prolongation=W.face_prolongation(
                             expert.agent_id, f"{f}:MECH", expert.n))
               for f in ("xlo", "xhi", "ylo", "yhi")],
        bc_channel=BCChannel.NONE,
        bc_time_varying=False,
        differentiable=Differentiable.NONE,
        deterministic=True,
        reproducibility_floor=np.finfo(float).eps,
        dt_native=expert.dt,
        governing_family="incompressible-navier-stokes-2d",
        boundary_response=expert.respond,
        elliptic_subsolve=EllipticSubsolve.EXPOSED,
        time_discretization=TimeDiscretization.EXPLICIT,
        stencil_radius=2,
        substeps_per_macro_step=W.SUBSTEPS,
        storage=lambda u: 0.5 * float(np.dot(np.ravel(u), np.ravel(u))),
        validity=lambda state, cond=None: True,
    )


def build(u_full, v_full, n=138):
    experts = {}
    for k, name in enumerate(("P00", "P10")):
        ox = k * (W.MONO_N - n)
        experts[name] = PeriodicWindowAgent(name, u_full[0:n, ox:ox + n],
                                            v_full[0:n, ox:ox + n])
    agents = [Agent(nm, capabilities(experts[nm]), domain=f"tile {nm}")
              for nm in experts]
    conn = Connection(seam_id="sx0", a=("P00", "xhi:MECH"), b=("P10", "xlo:MECH"),
                      port_type=PortType.MECH, derive_space=True,
                      geometrically_coincident=True, expected_null_dim=0,
                      note="a periodic window has no boundary to impose this on")
    return CaseGraph(name="periodic-window-control", agents=agents,
                     connections=[conn], decomposition=Decomposition.OVERLAPPING,
                     overlap_cells=n * 2 - W.MONO_N, macro_dt=W.MACRO_DT,
                     note="section 4.5's positive control"), experts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default=os.path.join("out", "tier0_verify", "s0_state.npz"))
    ap.add_argument("--out", default=os.path.join("out", "xi"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    d = np.load(a.state)
    graph, experts = build(d["u"], d["v"])
    result = compile_scheme(graph, probe_budget=ProbeBudget(),
                            probe_state="developed wake t=5, periodic control")

    print(f"verdict: {result.verdict}   refusals: {len(result.decisions.refusals)}")
    for x in result.decisions.refusals:
        print(f"  REFUSE [{x.layer}/{x.rule}] {x.message[:150]}")
    print(f"refused claims: {result.refused_claims}")

    out = {"verdict": str(result.verdict),
           "refusals": [x.as_dict() for x in result.decisions.refusals],
           "refused_claims": result.refused_claims, "seams": {}}

    for seam, op in result.seam_operators.items():
        S = np.asarray(op.S, float)
        nrm = float(np.linalg.norm(S, 2))
        blocks = {k: float(np.abs(np.asarray(b.S, float)).max())
                  for k, b in op.blocks.items()}
        print(f"\nseam {seam}:")
        print(f"  ||Lambda||_2         = {nrm:.6e}")
        print(f"  max |S_ij|           = {float(np.abs(S).max()):.6e}")
        print(f"  S is EXACTLY zero    = {bool(np.all(S == 0.0))}")
        print(f"  is_empty             = {op.is_empty}")
        print(f"  beta                 = {op.beta}")
        print(f"  null dim / expected  = {op.null_dim} / {op.expected_null_dim}")
        print(f"  per-block max |S|    = {blocks}")
        out["seams"][seam] = {
            "norm": nrm, "max_abs": float(np.abs(S).max()),
            "exactly_zero": bool(np.all(S == 0.0)), "is_empty": bool(op.is_empty),
            "beta": op.beta, "null_dim": op.null_dim, "blocks_max_abs": blocks,
        }

    # Xi against the Dirichlet agent of the same class at the same state, which
    # is the ||Lambda^ref|| section 4.5's ratio needs.
    ref_norm = 0.41686153742542914      # out/tier0_verify/w2.json, seam sx0, exposed
    xi = {s: v["norm"] / ref_norm for s, v in out["seams"].items()}
    out["Xi"] = xi
    out["Lambda_ref_norm"] = ref_norm
    print(f"\nXi = ||Lambda^periodic|| / ||Lambda^exposed|| = "
          f"{ {k: f'{v:.3e}' for k, v in xi.items()} }")
    print(f"  (||Lambda^exposed|| = {ref_norm:.6f}, from out/tier0_verify/w2.json)")
    print(f"\nsolver calls: {sum(e.n_calls for e in experts.values())}")

    with open(os.path.join(a.out, "xi_control.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"wrote {os.path.join(a.out, 'xi_control.json')}")


if __name__ == "__main__":
    main()
