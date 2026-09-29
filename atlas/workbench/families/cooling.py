"""The conjugate heat-transfer family, run from a case file: style C at a seam
between two physics.

A heated block cooled by channel flow (showcase case 8), steady.  Two families
meet at one seam:

* the **coolant** is advection and diffusion of temperature: a plug flow along
  x at the case's speed, carrying ``rho cp u`` through every face of its rows,
  entering at a given temperature and leaving with what it holds (the transport
  family's physics, `families/plume.py`, with temperature for concentration);
* the **block** is conduction, with heat generated in any material that has a
  volumetric source (the conduction family's physics).

Both are `fv.py`'s operator, so the full domain is one system on every cell --
the coolant's rows with their flow, the block's without -- solved directly.

**Style C at the seam.**  The two windows are the two physics: the coolant
channel and the block, meeting along the channel's wall.  Dirichlet-Neumann
(`styles.dirichlet_neumann`): the piece with the lower conductivity (the
coolant) is given the wall's temperatures, the block the heat the coolant takes
through the wall, relaxed (the case's first factor, then Aitken's rule) until
the wall temperature agrees.  No flow crosses the wall, which is what lets a
face-cut method couple the two (fv.py refuses a cut face that carries flow).

**The balance** is the whole point of a cooled block: in the steady state every
watt generated leaves in the coolant, ``P = sum over the outlet of rho cp u T
- sum over the inlet of rho cp u T_in``.  The coolant's bulk temperature rise is
``P / (mdot cp)``.

**What the showcase does not claim.**  The coolant is a plug flow and the cells
are coarse against its thermal boundary layer, so the wall's heat transfer is
the grid's, not a correlation's; both arms carry exactly this discretization,
which is what they are compared on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import fv, styles
from .. import geometry as geo
from ..checks import CheckSpec, judge
from .conduction import dirichlet_side, rho_1d, window_cells

FAMILY = "conjugate-heat-2d"
STYLE = "C"
ARMS = ("serial", "full")
FIELD_LABEL = "temperature (K)"
LENGTH_UNIT = "m"
SERIES: dict[str, str] = {}

REGISTERED = "2026-09-29, before the cooled-block family's first run"

CHECKS: tuple[CheckSpec, ...] = (
    CheckSpec(
        key="energy", title="Heat generated = heat the coolant carries out", kind="balance",
        tolerance=1e-6, registered=REGISTERED,
        measure=("|heat generated + heat into the domain through its edges (the coolant's "
                 "advection in and out, and any conduction)| over the heat generated, "
                 "every repeat, worst arm"),
        why=("the full domain closes to the direct solver's round-off. The Dirichlet-"
             "Neumann pair closes to its iteration's floor: the wall temperature moves by "
             "less than 1e-10 of the temperature scale between the last iterations, and "
             "the wall's heat flow is that over a half-cell times the conductivity, far "
             "under 1e-6 of the power")),
    CheckSpec(
        key="reference", title="Temperatures agree with the full domain", kind="reference",
        tolerance=1e-6, registered=REGISTERED,
        measure=("max |T_decomposed - T_full| over the full domain's rise above the "
                 "inlet, at the last repeat"),
        why=("both solve the same finite volumes; the converged Dirichlet-Neumann pair is "
             "the full-domain solution to the iteration's tolerance (its converged wall "
             "flow is the harmonic-mean face exactly)")),
)

_KIND = {"coolant-inlet": fv.INLET, "coolant-outlet": fv.OUTLET, "insulated": fv.NO_FLUX,
         "fixed-temperature": fv.FIXED, "heat-flux": fv.FLUX}


def material_grid(spec, prop: str) -> np.ndarray:
    d = spec.domain
    owner = geo.region_owner(spec.regions, d.nx, d.ny)
    out = np.zeros((d.ny, d.nx))
    for i, r in enumerate(spec.regions):
        out[owner == i] = float(spec.materials[r.material][prop])
    return out


def coolant_rows(spec) -> tuple[np.ndarray, str | None]:
    """The rows the coolant runs in, and a reason when its cells are not whole rows
    (a plug flow along x must run from the left edge to the right)."""
    flows = material_grid(spec, "flows") > 0.5
    rows = np.flatnonzero(flows.all(axis=1))
    if np.any(flows.any(axis=1) & ~flows.all(axis=1)):
        return rows, ("the coolant must fill whole rows, from the left edge to the right: a "
                      "plug flow along x has nowhere to go at a solid across its path")
    return rows, None


def field_from_case(spec) -> fv.Field:
    """The block and its coolant as one `fv.Field`: ``k`` and the source per cell,
    ``rho cp u`` through the x-faces of the coolant's rows."""
    d = spec.domain
    k = material_grid(spec, "k")
    src = material_grid(spec, "heat")
    rows, _why = coolant_rows(spec)
    fx = np.zeros((d.ny, d.nx + 1))
    if rows.size:
        rc = material_grid(spec, "rho") * material_grid(spec, "cp")
        u = spec.physics.get("u_coolant")
        fx[rows, 1:-1] = u * 0.5 * (rc[rows, :-1] + rc[rows, 1:])
        fx[rows, 0] = u * rc[rows, 0]
        fx[rows, -1] = u * rc[rows, -1]
    bc = {}
    for e in fv.EDGES:
        n = geo.edge_length(e, d.nx, d.ny)
        kd, vv = np.zeros(n, dtype=np.int8), np.zeros(n)
        for b in spec.boundaries:
            if b.edge == e:
                stop = n if b.stop is None else b.stop
                kd[b.start:stop] = _KIND[b.kind]
                vv[b.start:stop] = 0.0 if b.value is None else float(b.value)
        bc[e] = (kd, vv)
    return fv.Field(d.nx, d.ny, float(d.dx), k, source=src, fx=fx, bc=bc)


def inlet_temperature(spec) -> float | None:
    vals = [float(b.value) for b in spec.boundaries if b.kind == "coolant-inlet"
            and b.value is not None]
    return min(vals) if vals else None


@dataclass
class CoolState:
    u: np.ndarray
    iterations: int = 0
    converged: bool = True
    history: list[float] = field(default_factory=list)
    relaxation: list[float] = field(default_factory=list)


class CoolingRun:
    family = FAMILY

    def __init__(self, spec, arms=ARMS, threads: int = 1):
        self.spec = spec
        self.f = field_from_case(spec)
        self.nx, self.ny = self.f.nx, self.f.ny
        self.power = fv.total_source(self.f)
        self.T_in = inlet_temperature(spec)
        fixed = [float(b.value) for b in spec.boundaries if b.value is not None
                 and b.kind in ("coolant-inlet", "fixed-temperature")]
        self.scale = max([abs(v) for v in fixed] + [1.0])
        self.tol = float(spec.coupling.tolerance)
        self.max_it = int(spec.coupling.max_iterations)
        self.arms = tuple(a for a in ARMS if a in arms)
        self.full_sys = fv.assemble(self.f)
        self.full_lu = styles.Factor(self.full_sys.A) if "full" in self.arms else None
        d_id, n_id = dirichlet_side(spec, self.f)
        box = {w.id: (w.x0, w.y0, w.nx, w.ny) for w in spec.windows}
        self.d_id, self.n_id = d_id, n_id
        self.sd = fv.assemble(self.f, window_cells(self.nx, box[d_id]), "dirichlet")
        self.sn = fv.assemble(self.f, window_cells(self.nx, box[n_id]), "neumann")
        self.fd, self.fn = styles.Factor(self.sd.A), styles.Factor(self.sn.A)
        self.pairing = styles.pair_faces(self.sd, self.sn)
        self.rho_1d = rho_1d(spec, d_id, n_id)
        rows, _why = coolant_rows(spec)
        rc = material_grid(spec, "rho") * material_grid(spec, "cp")
        self.mdot_cp = float(spec.physics.get("u_coolant") * np.sum(rc[rows, 0])
                             * self.f.dx)

    def initial(self, arm: str) -> CoolState:
        return CoolState(np.full(self.f.n, self.T_in if self.T_in is not None else 0.0))

    def step(self, arm: str, s: CoolState) -> CoolState:
        """The whole steady solve again (a timed repeat)."""
        if arm == "full":
            return CoolState(self.full_lu.solve(self.full_sys.b), 1, True)
        lam0 = np.full(self.sd.face_rows.size, self.T_in if self.T_in is not None else 0.0)
        it = styles.dirichlet_neumann(self.sd, self.fd, self.sn, self.fn, lam0, self.tol,
                                      self.max_it, self.scale,
                                      theta0=self.spec.coupling.relaxation,
                                      aitken=self.spec.coupling.aitken, pairing=self.pairing)
        return CoolState(it.u, it.iterations, it.converged, it.history, it.relaxation)

    def observe(self, arm: str, s: CoolState, prev: CoolState | None = None) -> dict:
        q = fv.boundary_inflow(self.f, s.u)
        net = float(sum(q.values())) + self.power
        return {"balance": abs(net) / max(abs(self.power), 1e-300),
                "carried_out_W": -float(sum(q.values())),
                "max_temperature": float(np.max(s.u)),
                "iterations": float(s.iterations), "converged": float(s.converged),
                "convergence": list(s.history), "relaxation": list(s.relaxation)}

    def bitwise_equal(self, a: CoolState, b: CoolState) -> bool:
        return bool(np.array_equal(a.u, b.u))

    def field(self, s: CoolState) -> np.ndarray:
        return s.u.reshape(self.ny, self.nx)

    def close(self) -> None:
        pass

    def compare(self, states, history, bitwise):
        metrics: dict[str, Any] = {}
        for arm, rows in history.items():
            if not rows:
                continue
            metrics[arm] = {"power_W": self.power, "carried_out_W": rows[-1]["carried_out_W"],
                            "max_temperature_K": rows[-1]["max_temperature"],
                            "balance_max": max(r["balance"] for r in rows),
                            "iterations_last": rows[-1]["iterations"],
                            "all_converged": all(r["converged"] for r in rows)}
            if self.T_in is not None and self.mdot_cp > 0:
                metrics[arm]["bulk_rise_K"] = rows[-1]["carried_out_W"] / self.mdot_cp
        checks = [judge(CHECKS[0], max((m["balance_max"] for m in metrics.values()),
                                       default=None),
                        "; ".join(f"{a}: {m['balance_max']:.3g}" for a, m in metrics.items())
                        + f" (power {self.power:.6g} W per metre of depth)")]
        if "serial" in metrics and "full" in metrics:
            uf = states["full"].u
            rise = float(np.max(uf)) - (self.T_in if self.T_in is not None else 0.0)
            dev = float(np.max(np.abs(states["serial"].u - uf)))
            metrics["serial"]["max_temperature_difference_K"] = dev
            checks.append(judge(CHECKS[1], dev / rise if rise > 0 else dev,
                                f"max |dT| = {dev:.3g} K against a rise of {rise:.4g} K"))
        else:
            checks.append(judge(CHECKS[1], None, "needs the pieces and the full domain"))
        return metrics, checks

    def notes(self, done: int) -> list[str]:
        out = [f"Dirichlet side: {self.d_id}; Neumann side: {self.n_id}. The 1-D contraction "
               f"factor rho = k_D L_N / (k_N L_D) = {self.rho_1d:.3g}"
               + (" (under 1: the unrelaxed iteration would contract)" if self.rho_1d < 1
                  else " (over 1: unrelaxed it would diverge; the relaxation carries it)")
               + f". First relaxation factor {self.spec.coupling.relaxation:g}, "
               + ("then Aitken's rule." if self.spec.coupling.aitken else "held fixed.")]
        if self.mdot_cp > 0:
            out.append(f"The block generates {self.power:.6g} W per metre of depth; the "
                       f"coolant carries mdot cp = {self.mdot_cp:.6g} W/K per metre, so its "
                       f"bulk temperature rises {self.power / self.mdot_cp:.4g} K.")
        out.append("Steady: every repeat is the whole solve again, repeated to time it.")
        return out

    def describe(self) -> dict[str, Any]:
        return {"style": "C", "mode": "steady", "cells": self.f.n, "power_W": self.power,
                "mdot_cp_W_per_K": self.mdot_cp, "dirichlet": self.d_id,
                "neumann": self.n_id, "rho_1d": self.rho_1d, "tolerance": self.tol,
                "relaxation0": self.spec.coupling.relaxation,
                "aitken": self.spec.coupling.aitken,
                "interface_faces": int(self.sd.face_rows.size)}


def step_label(spec) -> str:
    return "timed repeat"


def available_arms(spec) -> tuple[tuple[str, ...], dict[str, str]]:
    return ARMS, {"parallel": "Dirichlet-Neumann is sequential by construction: the block "
                              "needs the heat the coolant takes through the wall"}


def build(spec, arms=ARMS, threads: int = 1) -> CoolingRun:
    return CoolingRun(spec, arms=arms, threads=threads)


__all__ = ["FAMILY", "STYLE", "ARMS", "CHECKS", "CoolingRun", "CoolState", "build",
           "field_from_case", "coolant_rows", "material_grid", "step_label",
           "available_arms"]
