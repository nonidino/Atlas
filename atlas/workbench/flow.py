"""A flow that follows a drawn domain: the potential flow from its inlets to its outlets.

The two families that carry something along a given flow -- the river's pollutant
(`families/plume.py`) and the cooled block's coolant (`families/cooling.py`) --
were given their flow per row: the river runs along +x at ``q / h``, the coolant
fills whole rows from the left edge to the right.  A drawn river bends, and a
drawn channel winds through its block, so a flow along +x would cross the banks.

**The flow here is solved, not given.**  Over the cells that carry it, the
potential ``phi`` is harmonic: 1 on the inlet faces, 0 on the outlet faces, and no
flux through every other face -- the banks, a wall to a solid, the grid's edge
(`fv.assemble` with a unit conductivity, one sparse solve).  The flow through each
face is the potential's drop across it, scaled so the inlets deliver the case's
total.  So:

* no flow crosses a bank (the Neumann condition is exact, face by face);
* every cell's flows sum to zero to the solve's round-off: the discrete flow is
  divergence-free in the finite volumes' own sense, which is what an upwinded
  advection needs to carry a uniform concentration unchanged;
* in a straight channel from one grid edge to the other the potential is linear
  and every face carries the same flow: the plug flow the families had.

**What it is not.**  A potential flow has no viscosity and no inertia: it slips
along the banks and turns corners without separating.  It is the depth-averaged
flow of a slowly varying river and the bulk flow of a channel, good enough to
carry a scalar the right way round a bend; it is not a Navier-Stokes solution,
and the families say so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from . import fv
from . import geometry as geo


class FlowError(ValueError):
    """The flow cannot be solved as the case stands, and why."""


@dataclass
class Flow:
    """A solved flow on the grid, as `fv.Field` takes it: the volume flux through
    each x-face ``fx`` ``(ny, nx + 1)`` and y-face ``fy`` ``(ny + 1, nx)`` per unit
    area of the face (a speed), positive towards +x and +y."""

    fx: np.ndarray
    fy: np.ndarray
    phi: np.ndarray             # the potential on the cells (NaN off the flow)
    inflow: float               # the total volume flow in, per unit depth
    residual: float             # the largest |sum of a cell's flows|, over the inflow

    def scaled(self, face_x: np.ndarray, face_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The flux times a property per face (a heat capacity per volume, say)."""
        return self.fx * face_x, self.fy * face_y


def potential_flow(carrier: np.ndarray, bf: geo.BoundaryFaces, kinds: np.ndarray,
                   total: float, dx: float) -> Flow:
    """The potential flow through the cells of ``carrier`` (a ``(ny, nx)`` mask).

    ``bf`` are the domain's boundary faces and ``kinds`` says, per face, ``"in"``,
    ``"out"`` or anything else (closed); only the faces of carrier cells take part.
    ``total`` is the volume flow per unit depth the inlets deliver together.
    """
    carrier = np.asarray(carrier, dtype=bool)
    ny, nx = carrier.shape
    on = carrier.ravel()[bf.cell]
    inlet = on & (kinds == "in")
    outlet = on & (kinds == "out")
    if not inlet.any():
        raise FlowError("nothing lets the flow in: no inlet face on the cells that carry it")
    if not outlet.any():
        raise FlowError("nothing lets the flow out: no outlet face on the cells that carry "
                        "it")
    from scipy.ndimage import label
    lab, n_parts = label(carrier)
    for part in range(1, n_parts + 1):
        here = (lab.ravel()[bf.cell] == part)
        if not (here & inlet).any() or not (here & outlet).any():
            raise FlowError("a separate piece of the flow has no inlet or no outlet of its "
                            "own, so nothing can run through it")
    # the potential: 1 on the inlet faces, 0 on the outlet faces, closed elsewhere,
    # by the finite volumes themselves (grid-edge faces through `bc`, the rest as
    # faces to the "void", which here is every cell that does not carry the flow)
    bc = {}
    for e in fv.EDGES:
        n = ny if e in ("left", "right") else nx
        bc[e] = (np.zeros(n, dtype=np.int8), np.zeros(n))
    void_cell, void_out, void_dir, void_kind, void_val = [], [], [], [], []
    k_axis = {"left": 1, "right": 0, "bottom": 3, "top": 2}
    for c, d, a, is_in, is_out, ok in zip(bf.cell.tolist(), bf.direction.tolist(),
                                          bf.along.tolist(), inlet.tolist(), outlet.tolist(),
                                          on.tolist()):
        if not ok or not (is_in or is_out):
            continue
        val = 1.0 if is_in else 0.0
        if a >= 0:                                   # a face on the grid's own edge
            e = {v: k for k, v in k_axis.items()}[d]
            bc[e][0][a], bc[e][1][a] = fv.FIXED, val
        else:
            di, dj = geo.DIRECTIONS[d]
            void_cell.append(c)
            void_out.append(c + dj * nx + di)
            void_dir.append(d)
            void_kind.append(fv.FIXED)
            void_val.append(val)
    f = fv.Field(nx, ny, dx, np.where(carrier, 1.0, 0.0), bc=bc, active=carrier,
                 void=(np.asarray(void_cell, dtype=np.int64),
                       np.asarray(void_out, dtype=np.int64),
                       np.asarray(void_dir, dtype=np.int8),
                       np.asarray(void_kind, dtype=np.int8),
                       np.asarray(void_val, dtype=float),
                       np.array([""] * len(void_cell), dtype=object)))
    s = fv.assemble(f)
    phi = np.full(ny * nx, np.nan)
    phi[s.idx] = spla.spsolve(s.A.tocsc(), s.b)
    P = phi.reshape(ny, nx)
    # raw flows (unit conductance): interior faces between two carrier cells, and
    # the inlet and outlet faces through the half-cell (conductance 2)
    gx = np.where(carrier[:, :-1] & carrier[:, 1:], 1.0, 0.0)
    qx = np.zeros((ny, nx + 1))
    qx[:, 1:-1] = np.where(gx > 0, np.nan_to_num(P[:, :-1] - P[:, 1:]), 0.0)
    gy = np.where(carrier[:-1, :] & carrier[1:, :], 1.0, 0.0)
    qy = np.zeros((ny + 1, nx))
    qy[1:-1, :] = np.where(gy > 0, np.nan_to_num(P[:-1, :] - P[1:, :]), 0.0)
    raw_in = 0.0
    for c, d, is_in, is_out in zip(bf.cell.tolist(), bf.direction.tolist(), inlet.tolist(),
                                   outlet.tolist()):
        if not (is_in or is_out):
            continue
        j, i = divmod(c, nx)
        out = 2.0 * (phi[c] - (1.0 if is_in else 0.0))     # flow leaving the domain
        if is_in:
            raw_in -= out
        if d == 0:
            qx[j, i + 1] = out
        elif d == 1:
            qx[j, i] = -out
        elif d == 2:
            qy[j + 1, i] = out
        else:
            qy[j, i] = -out
    if raw_in <= 0.0:                                          # pragma: no cover
        raise FlowError("the potential carries no flow in")
    scale = float(total) / raw_in
    fx, fy = qx * scale / dx, qy * scale / dx
    # continuity, face by face: each carrier cell's flows out sum to zero
    net = (fx[:, 1:] - fx[:, :-1] + fy[1:, :] - fy[:-1, :]) * dx
    resid = float(np.max(np.abs(net[carrier]))) / max(abs(float(total)), 1e-300)
    return Flow(fx, fy, P, float(total), resid)


def case_faces(spec, inlet_kind: str, outlet_kind: str):
    """(boundary faces, per face "in" / "out" / "") from the case's boundaries."""
    d = spec.domain
    bf = geo.boundary_faces(d)
    owner = geo.face_conditions(spec.boundaries, bf, d.nx, d.ny)
    kinds = np.array([""] * bf.n, dtype=object)
    for i, b in enumerate(spec.boundaries):
        if b.kind == inlet_kind:
            kinds[owner == i] = "in"
        elif b.kind == outlet_kind:
            kinds[owner == i] = "out"
    return bf, kinds


__all__ = ["FlowError", "Flow", "potential_flow", "case_faces"]
