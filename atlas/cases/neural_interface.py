"""CS-S1 -- a learned operator on the INTERFACE, with classical solvers owning
the subdomains: the hybrid built, and the half of it that does not work measured.

The proposal this file tests
----------------------------

Every substitution certificate in this vault has refused Poseidon-T, and R10 is
why: a neural operator's domain of dependence is the whole window (W93 measured
64 cells against a declared 2, nonzero in all 128 seam cells), so cutting the
domain cuts the operator and no halo repairs it.  The refusal is correct and it
is not about accuracy -- fine-tuning the checkpoint to a one-step error of 1e-9
would not move it.

So put the learned operator where its globality is an asset instead of a
liability.  **Let it predict the interface trace and let classical solvers own
the subdomains.**  Three things follow, and only the third is a hypothesis:

1. It cannot introduce a silent error.  A starting guess for a *convergent*
   iteration is a preconditioner: a wrong guess costs iterations, not
   correctness.
2. It needs no certificate, because it is not an agent.  The refusal machinery
   is about experts that PRODUCE the answer, and this one produces a guess.
3. It should be *good* at this, because the interface trace genuinely is
   determined by the whole domain -- which is what Schwarz needs many sweeps to
   propagate and what a global operator has for free.  Tier 32 measured the
   supporting half from the other side: Poseidon's control-to-jump map is full
   rank 32/32 at ``kappa = 4.48``, better conditioned than the classical solver
   carrying its own elliptic part at 19.4.  **Global receptivity cost it Pi and
   did not cost it observability.**

What was measured
-----------------

**1 and 2 hold.**  The fixed point is independent of where the iteration starts,
including from a deliberately wrong start (`AmplitudeMatchedNoise`), to the
iteration's own floor.  `predictor_is_not_an_agent` checks the structural half
mechanically: the predictor appears in no capability record on the graph.

**3 is false, and it is false for a reason that is not about Poseidon.**  A
linearly convergent iteration needs

    m  =  log(d_0 / tol) / log(1 / rho)

sweeps.  **A predictor moves d_0.  Only the operator moves rho.**  The ceiling is
therefore not an opinion, and it is measurable: `OneSweep` is the first iterate,
built from the very solver that owns the subdomains, so no prediction of the new
field can be much better -- and it saves **exactly one sweep**, in both
arrangements, at every tolerance, because it *is* one sweep.  It also costs
exactly the sweep it saves, so ``total_solves`` comes out identical to the
incumbent's to the unit.  **A field predictor's entire budget is one sweep.**

Measured against that ceiling, on the arrangement the compiler admits:

    hold old ring (shipped)      skill  0.000   2 sweeps     0 solves
    one classical sweep          skill +0.998   1 sweep      4 solves
    poseidon-T                   skill -1.201   2 sweeps   244 solves
    amplitude-matched noise      skill -1.597   2 sweeps     0 solves
    oracle (the fixed point)     skill +1.000   1 sweep      -

Poseidon is **61x the break-even** in the admitted arrangement and 7x it in the
refused one.  The two differ because the as-built window step costs ~9x the
split-step one -- it runs a Poisson solve per sub-step -- so the same checkpoint
buys more sweeps there; both are single-shot timing ratios measured in-process
and neither is quoted more finely than its order.  It is too expensive by an
order of magnitude at either reading, **and its skill is negative besides**:
-1.20 against amplitude-matched noise's -1.60 in the admitted arrangement, and
-3.25 against noise's -3.68 in the refused one.  The gap between the checkpoint
and pure noise of the same size is 0.4 of a skill point in both.

**And the rate lever is worth a factor.**  Three are measured here, in expert
calls rather than in adjectives, and two were sitting in the vault unused:

    exposing the elliptic part      74.8 -> 0.50 sweeps per decade: 149x
    alpha_star as a Robin           kappa 22.7 -> 7.3, and 594 -> 172 calls
      preconditioner, DAMPED        (undamped it diverges: rho(D^-1 S) = 2.98)
    the probed S as a coarse        594 -> 44 calls at tol 1e-6 ||r(0)||
      space (Newton)

**The first of those is R10's own mechanism, and it has never been quoted as a
cost.**  `L2/R10` refuses an agent with an embedded pressure solve because the
composed *error* is elliptic.  The same refusal is worth **149x in coupling
sweeps**, and the compiler picks the fast arrangement out of the declaration
before a single sweep runs.  A control separates it from W100's other half: with
the global projection switched off the rate is the same to within 17%, so the
149x is the *exposure* and not the projection.

**And the halo rule turns out to be a convergence condition -- for the embedded
agent, and only for it.**  Swept from halo 4 to 48:

    as-built     diverges up to halo 18, converges from halo 20
    split-step   converges at EVERY halo from 4 up, at 7-8 sweeps

The agent's declared reach is ``STENCIL_RADIUS * SUBSTEPS = 20`` and the as-built
bracket lands on it exactly.  So R10's two branches are one fact: **the embedded
elliptic solve is what makes the overlap load-bearing.**  Take it out and the
overlap stops deciding whether the coupling converges -- it still decides
accuracy, which is a different measurement and is `assembly.py`'s.

The split-step contraction is flat at 0.011 from halo 8 to 48 and that is
reported as a **floor, not a null**: the iteration reaches round-off in 7-8
sweeps, so no asymptote is resolvable in that arrangement.  Halo 4 is resolvably
worse at 0.089 over 12 sweeps, which is what says the instrument can tell the
rows apart at all.

What this file is not
---------------------

**It is not a rung on the f1 ladder** -- hence `CS-S1` and not a number from
the ladder's own sequence, which reserves CS-16 for `vehicle.py` at rung 9 --
and it declares no new capability record:
the agents are `window_ns.WindowAgent` unchanged and the geometry is
`poseidon.PoseidonTiling` unchanged, which is the affordability claim being
*used* rather than restated.  What is new is a **scheme**, and the graph exists
so that the scheme's claims can be checked against a compile.

**No constant is measured on this graph.**  ``tau``, ``sigma``, ``C_mu`` and
``||A||`` stay `unmeasured`; the quantities here are properties of the
ITERATION, not of the composition error, and the two are different questions.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..graph import Agent, CaseGraph, Connection, Decomposition, GlobalField
from ..ports import PortType
from . import poseidon as PO
from . import wake_array as WA
from . import window_ns as W
from .window_ns import (
    H,
    MACRO_DT,
    NU,
    STENCIL_RADIUS,
    SUBSTEPS,
    WindowAgent,
    load_reference,
    window_capabilities,
)

#: The agent's own reach over one macro-step, in cells: how far a datum on an
#: artificial face can travel into the window before the step ends.  R10's halo
#: rule compares the overlap against this, and section 4 measures the iterated
#: scheme's convergence threshold against the same number.
DOMAIN_OF_DEPENDENCE = STENCIL_RADIUS * SUBSTEPS

#: 2 * 128 - 21.  The geometry is `poseidon.PoseidonTiling`'s and is not re-derived
#: here: 128-cell windows because that is the only size the checkpoint has, and a
#: 21-cell overlap because that is the smallest one that covers the reach above.
DEFAULT_MONO_N = PO.MONO_N

#: Which cells of a window array are its ring, per face.
RING = {
    "xlo": (slice(None), 0),
    "xhi": (slice(None), -1),
    "ylo": (0, slice(None)),
    "yhi": (-1, slice(None)),
}


@dataclass(frozen=True)
class HybridTiling(PO.PoseidonTiling):
    """`PoseidonTiling` with classical windows in it, so the names say so.

    Subclassed rather than copied: the overlap arithmetic, the ramped partition
    of unity and the contaminated sets are the same geometry and there is no
    version of this study in which they should be allowed to drift apart from
    the checkpoint's.  Only the agent identities change.
    """

    @property
    def names(self) -> list[str]:
        return ["C00", "C10", "C01", "C11"]


DEFAULT_TILING = HybridTiling(mono_n=DEFAULT_MONO_N)

#: seam_id -> ((agent, face), (agent, face)), the same four seams
#: `window_ns.SEAMS` and `poseidon.SEAMS` declare, on this file's agent names.
SEAMS = {
    "sx0": (("C00", "xhi"), ("C10", "xlo")),
    "sx1": (("C01", "xhi"), ("C11", "xlo")),
    "sy0": (("C00", "yhi"), ("C01", "ylo")),
    "sy1": (("C10", "yhi"), ("C11", "ylo")),
}


def load_state(root: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The developed wake at ``t = 5`` every Tier 0 measurement was taken on.

    The same file `w55_second_expert.load_state` reads, so this study's state is
    the one the probe floor, `Xi` and the elliptic signature were measured at
    rather than a fresh one nobody has a prior for.
    """
    here = root or os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    d = np.load(os.path.join(here, "out", "tier0b", "s0_state.npz"))
    return d["u"], d["v"]


# ===========================================================================
# 1 -- the exchange: overlapping Schwarz at the FIELD level
# ===========================================================================


@dataclass
class SchwarzExchange:
    """The coupling as it actually runs, iterated to its fixed point.

    **What the unknown is, and why it took two arrangements to find.**
    `solve.py` poses the interface problem in the multiplier space: a common
    16-mode ``lambda`` imposed on both sides' rings, whose root balances two
    outward fluxes.  On an OVERLAPPING decomposition those two rings are 20 cells
    apart, and `MultiplierSeam` below measures the consequence -- the root is 22x
    the size of the field's own change over the macro-step, so it is not a trace
    that any field predictor predicts, and every one of them scores negative
    against a zero start including the classical one.

    The overlapping Schwarz transmission condition is a different object, and it
    is the one the composed rollout runs on.  `reference.WindowNS.step_batch`
    reads the ring at BOTH ends of the macro-step and interpolates across the
    sub-steps; the shipped scheme passes ``bc1=None``, which **holds the old ring
    across the step**.  The unknown is therefore the ring at ``t + dt``, and the
    fixed point is where every window's artificial-face data equals the assembled
    field there.  That is a physical trace, and it is what a field predictor
    predicts.

    **A face that is the domain boundary is not iterated.**  It carries a real
    boundary condition and the monolith holds its own ring there, so this class
    holds it too, exactly on the faces `Tiling.artificial_faces` does not name.
    Iterating it would let the domain edge drift and would make the fixed point a
    property of the tiling.
    """

    u0: np.ndarray                       # the assembled field at t_n, [mono_n, mono_n]
    v0: np.ndarray
    tiling: Any = DEFAULT_TILING
    dt: float = MACRO_DT
    nu: float = NU
    #: W100's split: `expose_elliptic` hands the pressure solve to the composition
    #: layer, which is what `probed-dtn-coupling` 2.1 assumes and what L2/R10
    #: requires.  False is the agent as built, which R10 refuses.
    expose_elliptic: bool = True
    #: The composition layer's own operator (L6/C2), applied ONCE to the assembled
    #: field per sweep.  `wake_array.project_assembled` is the declared one.
    project: bool = True
    solves: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        n, m = self.tiling.n, self.tiling.mono_n
        self.u0 = np.ascontiguousarray(np.asarray(self.u0, dtype=float)[:m, :m])
        self.v0 = np.ascontiguousarray(np.asarray(self.v0, dtype=float)[:m, :m])
        if self.u0.shape != (m, m):
            raise ValueError(
                f"the tiling covers {m}x{m} and the field is {self.u0.shape}")
        self.us = self.tiling.cut(self.u0)
        self.vs = self.tiling.cut(self.v0)
        self.faces = [self.tiling.artificial_faces(*o) for o in self.tiling.offsets]
        self._edge = [self._domain_edge_mask(k)
                      for k in range(len(self.tiling.names))]
        cls = (W._no_projection_class() if self.expose_elliptic
               else load_reference().WindowNS)
        self._solvers = [cls(nu=self.nu, length=n * H, n=n, cfl=0.4,
                             transmission="dirichlet")
                         for _ in self.tiling.names]

    def _domain_edge_mask(self, k: int) -> np.ndarray:
        """Cells of window ``k`` that lie on the GLOBAL domain boundary.

        **An artificial face's ring can end on the domain edge**, and its corner
        cell is then a boundary cell belonging to an interior face.  C00's `yhi`
        ring runs the full width of the window and its first cell sits at global
        ``x = 0``, which is the inlet.  Writing the iterate there lets the domain
        boundary drift: measured before this mask existed, wrecking the edge by
        7.0 moved the sweep's output by 9.2e-06, a leak of 1.3e-06 -- small, and
        exactly the kind of small that a fixed point quietly becomes a property
        of the tiling through.

        A cell on the domain boundary is a domain boundary cell whichever face of
        whichever window it belongs to, so the mask is over cells and not faces.
        """
        ox, oy = self.tiling.offsets[k]
        n, m = self.tiling.n, self.tiling.mono_n
        xs = np.arange(ox, ox + n)
        ys = np.arange(oy, oy + n)
        return (((ys == 0) | (ys == m - 1))[:, None]
                | ((xs == 0) | (xs == m - 1))[None, :])

    # -- one sweep ---------------------------------------------------------

    def sweep(self, gu: np.ndarray, gv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Every window steps once, reading its artificial ring from ``(gu, gv)``.

        One sweep costs one subdomain solve per window, and `solves` counts them.
        That is the currency the whole study is quoted in: in this arrangement a
        sweep IS a subdomain solve, so "how expensive is the coupling" and "how
        many sweeps" are the same question.
        """
        n = self.tiling.n
        ou, ov = [], []
        for k, (ox, oy) in enumerate(self.tiling.offsets):
            b1u, b1v = self.us[k].copy(), self.vs[k].copy()
            su = gu[oy:oy + n, ox:ox + n]
            sv = gv[oy:oy + n, ox:ox + n]
            edge = self._edge[k]
            for f in self.faces[k]:
                idx = RING[f]
                keep = ~edge[idx]
                b1u[idx] = np.where(keep, su[idx], b1u[idx])
                b1v[idx] = np.where(keep, sv[idx], b1v[idx])
            u1, v1 = self._solvers[k].step_batch(
                self.us[k][None], self.vs[k][None], self.dt,
                bc0=(self.us[k][None], self.vs[k][None]),
                bc1=(b1u[None], b1v[None]))
            self.solves += 1
            ou.append(u1[0])
            ov.append(v1[0])
        au, av = self.tiling.assemble(np.stack(ou), np.stack(ov))
        if self.project:
            au, av = WA.project_assembled(au, av)
        return au, av

    # -- the iteration -----------------------------------------------------

    def march(self, gu: np.ndarray, gv: np.ndarray, *, cap: int = 400,
              tol: float = 0.0,
              star: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, Any]:
        """Sweep until the change (or the distance to ``star``) falls under ``tol``.

        Returns the history rather than a verdict.  **A stalled march is a
        blow-up**: a diverging Schwarz iteration on an explicit solver overflows
        rather than stalling, so non-finiteness is detected every sweep and
        reported as divergence instead of being marched into ``nan``.
        """
        gu = np.array(gu, dtype=float, copy=True)
        gv = np.array(gv, dtype=float, copy=True)
        changes: list[float] = []
        dists: list[float] = []
        diverged = False
        for _ in range(cap):
            nu_, nv_ = self.sweep(gu, gv)
            if not (np.all(np.isfinite(nu_)) and np.all(np.isfinite(nv_))):
                diverged = True
                break
            changes.append(_rms(nu_ - gu, nv_ - gv))
            gu, gv = nu_, nv_
            if star is not None:
                dists.append(_rms(gu - star[0], gv - star[1]))
            if tol > 0.0:
                if (dists[-1] if star is not None else changes[-1]) <= tol:
                    break
        return {
            "u": gu, "v": gv,
            "sweeps": len(changes),
            "changes": changes,
            "distances": dists,
            "diverged": diverged,
            "contraction": _tail_ratio(changes),
        }

    def fixed_point(self, cap: int = 400,
                    tol: float = 1e-13) -> tuple[np.ndarray, np.ndarray, dict]:
        """March from the cold start until the change stops, and report how far it got.

        The referent for every skill number below.  It is an *approximate* fixed
        point and the record says which: `march` returns the last change, and a
        contraction near 1 means the last change is not a bound on the distance.
        """
        out = self.march(self.u0, self.v0, cap=cap, tol=tol)
        return out["u"], out["v"], out


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a * a + b * b)))


def _tail_ratio(hist: Sequence[float], n: int = 10) -> float:
    """Geometric mean of the last ``n`` per-sweep ratios: the asymptotic rate.

    The early sweeps of an overlapping Schwarz iteration contract much faster
    than the asymptote -- the first one is nearly free because the initial datum
    is exactly wrong in a way one exchange fixes -- so an average over the whole
    history flatters the method.  The tail is what decides the sweep count at a
    tight tolerance.
    """
    if len(hist) < 3:
        return float("nan")
    lo = max(1, len(hist) - n)
    r = [hist[i] / max(hist[i - 1], 1e-300) for i in range(lo, len(hist))]
    return float(np.exp(np.mean(np.log(np.maximum(r, 1e-300)))))


# ===========================================================================
# 2 -- the predictors
# ===========================================================================


class Predictor:
    """Something that supplies the iteration's FIRST ITERATE and nothing else.

    **A predictor is not an agent, and that is the whole architectural claim.**
    It has no `ExpertCapabilities`, no ports, no ``boundary_response``, and it
    appears in no `CaseGraph`.  It cannot be substituted into a seam, so no
    substitution certificate is owed on it, and the reason is not a convention:
    the iteration it seeds is convergent, so its output cannot change the answer.
    `start_independence` measures that rather than asserting it.

    ``cost_solves`` is what this predictor costs in the same currency as a sweep
    -- subdomain solves -- so that "does it pay for itself" is a subtraction.
    """

    name = "predictor"
    cost_solves = 0.0

    def initial(self, ex: SchwarzExchange) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class HoldOldRing(Predictor):
    """The incumbent: hold the ring at its value at ``t_n``.

    This is what the shipped scheme does -- ``step_batch(bc1=None)`` -- so it is
    the arm every other one has to beat, and it is free.
    """

    name = "hold old ring (the shipped scheme)"

    def initial(self, ex):
        return ex.u0.copy(), ex.v0.copy()


class OneSweep(Predictor):
    """One classical sweep, used as a prediction.

    **The ceiling for the whole class of field predictors, and the harness's own
    control.**  It is built from the same solver that owns the subdomains, so no
    predictor of the new field can be much better; and because it *is* the first
    iterate, it must save exactly one sweep.  A harness in which it saves two, or
    none, is measuring something other than what it says.
    """

    name = "one classical sweep"

    def __init__(self, tiling=None):
        self._n = 4 if tiling is None else len(tiling.names)
        self.cost_solves = float(self._n)

    def initial(self, ex):
        return ex.sweep(ex.u0, ex.v0)


class Neural(Predictor):
    """Poseidon-T's prediction of the field at ``t + dt``, assembled by the PoU.

    Every window is stepped by the checkpoint from its own restriction of the
    field at ``t_n``, and the four predictions are blended by **the graph's own
    declared partition of unity** rather than by anything this class invents.  So
    the prediction is a global object: it uses the whole domain's state, which is
    the property the proposal rests on.

    ``cost_solves`` is measured on the FIRST prediction, as a RATIO against the
    classical subdomain solve on the same machine in the same process, because a
    millisecond figure rots and a ratio does not.  Not on construction: building
    a predictor must not load 20.8M parameters, or `predictor_is_not_an_agent` --
    a question about a NAME -- would cost fourteen seconds to answer.
    """

    name = "poseidon-T"

    def __init__(self, expert=None, threads: int = 1, calibrate: bool = True):
        # Lazy: constructing a predictor must not load a 20.8M-parameter
        # checkpoint, because `predictor_is_not_an_agent` is a question about
        # the NAME and answering it should not cost fourteen seconds.
        self._expert = expert
        self._threads = threads
        self.cost_solves = float("nan")
        self._calibrate = calibrate

    @property
    def expert(self):
        if self._expert is None:
            self._expert = PO.load_expert(threads=self._threads)
        return self._expert

    def _measure_cost(self, ex: SchwarzExchange) -> float:
        """The predictor's cost in SUBDOMAIN SOLVES, measured as a ratio.

        A millisecond figure in a document is wrong within hours; a ratio against
        an operation measured in the same process on the same machine is not.
        One sweep is ``n`` solves, so a predictor that takes ``t_pred`` against a
        sweep's ``t_sweep`` costs ``n * t_pred / t_sweep`` solves.  The sweep run
        here is not charged to the arm: `predictor_skill` reads the counter after
        `initial` returns.
        """
        n = len(ex.tiling.names)
        t0 = time.perf_counter()
        self.expert.step_many(np.ascontiguousarray(ex.us),
                              np.ascontiguousarray(ex.vs), ex.dt)
        t_pred = time.perf_counter() - t0
        t1 = time.perf_counter()
        ex.sweep(ex.u0, ex.v0)
        t_sweep = time.perf_counter() - t1
        return n * t_pred / max(t_sweep, 1e-12)

    def initial(self, ex):
        if self._calibrate and not np.isfinite(self.cost_solves):
            self.cost_solves = self._measure_cost(ex)
        pu, pv = self.expert.step_many(np.ascontiguousarray(ex.us),
                                       np.ascontiguousarray(ex.vs), ex.dt)
        return ex.tiling.assemble(pu, pv)


class AmplitudeMatchedNoise(Predictor):
    """The floor: a random field of the same amplitude as the arm it controls.

    **A predictor that only has the right MAGNITUDE must not score.**  Without
    this control a prediction whose direction is uninformative but whose size is
    right can look like skill, and on the first arrangement measured here the
    two were within 0.07 of each other -- which is the finding, and is only
    visible because the control was marched as far as the thing it controls.
    """

    name = "amplitude-matched noise"

    def __init__(self, amplitude: float, seed: int = 20260909):
        self.amplitude = float(amplitude)
        self.seed = int(seed)

    def initial(self, ex):
        rng = np.random.default_rng(self.seed)
        du = rng.standard_normal(ex.u0.shape)
        dv = rng.standard_normal(ex.v0.shape)
        s = self.amplitude / max(_rms(du, dv), 1e-300)
        return ex.u0 + s * du, ex.v0 + s * dv


class Oracle(Predictor):
    """The ceiling: start at the answer.  Must converge in one sweep and no fewer.

    One rather than zero, because the loop measures the change AFTER a sweep and
    a fixed point still costs the sweep that shows it is one.
    """

    name = "oracle (the fixed point)"
    cost_solves = float("inf")

    def __init__(self, star):
        self.star = (np.array(star[0], copy=True), np.array(star[1], copy=True))

    def initial(self, ex):
        return self.star[0].copy(), self.star[1].copy()


def predictor_is_not_an_agent(graph: CaseGraph, predictor: Predictor) -> bool:
    """The structural half of the claim, checked rather than asserted.

    True when nothing about ``predictor`` appears in the graph: not as an agent
    id, not as a port, not in any capability record's weight hash.  A predictor
    that DID appear would be an agent, would owe a substitution certificate, and
    would be refused -- which is exactly the situation this arrangement exists to
    get out of.
    """
    tokens = {predictor.name.lower(), type(predictor).__name__.lower()}
    for a in graph.agents:
        blob = " ".join([
            a.agent_id, a.capabilities.expert_id or "",
            a.capabilities.weight_hash or "", a.capabilities.note or "",
            " ".join(p.name for p in a.capabilities.ports),
        ]).lower()
        if any(t in blob for t in tokens if t):
            return False
    return True


# ---------------------------------------------------------------------------
# what a predictor is worth
# ---------------------------------------------------------------------------


def predictor_skill(ex: SchwarzExchange, predictor: Predictor,
                    star: tuple[np.ndarray, np.ndarray],
                    tol: float, cap: int = 400) -> dict[str, Any]:
    """Skill, sweeps and total cost for one arm, against the cold start.

    ``skill = 1 - ||g_0 - g*|| / ||g_cold - g*||`` -- one for a perfect start,
    zero for the incumbent, negative for a start that is worse than doing
    nothing.  ``total_solves`` adds the predictor's own cost to the sweeps it
    leaves, which is the only comparison that can come out against the predictor
    and is therefore the only one worth making.
    """
    g0 = predictor.initial(ex)
    d0 = _rms(g0[0] - star[0], g0[1] - star[1])
    d_cold = _rms(ex.u0 - star[0], ex.v0 - star[1])
    before = ex.solves
    out = ex.march(g0[0], g0[1], cap=cap, tol=tol, star=star)
    n_win = len(ex.tiling.names)
    return {
        "predictor": predictor.name,
        "d0": d0,
        "d_cold": d_cold,
        # How far this start moved from the cold one -- the amplitude the noise
        # control has to be matched to, and the only honest way to match it.
        "perturbation": _rms(g0[0] - ex.u0, g0[1] - ex.v0),
        "skill": 1.0 - d0 / max(d_cold, 1e-300),
        "sweeps": out["sweeps"],
        "diverged": out["diverged"],
        "final_distance": out["distances"][-1] if out["distances"] else float("nan"),
        "sweep_solves": ex.solves - before,
        "cost_solves": predictor.cost_solves,
        "total_solves": (ex.solves - before) + predictor.cost_solves,
        "fixed_point": (out["u"], out["v"]),
        "contraction": out["contraction"],
        "n_windows": n_win,
    }


def start_independence(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """**The safety claim, measured.**  How far apart are the arms' answers?

    Every arm iterated the same operator to the same tolerance from a different
    start, including one deliberately wrong.  If a start could change the answer,
    the predictor would be an agent and would owe a certificate.  The number
    returned is the largest disagreement between any two arms' converged fields.

    **The bound to read it against is 2 * tol, not tol.**  Each arm stops when it
    is within ``tol`` of the limit, so two arms can be ``2 * tol`` apart by the
    triangle inequality and still be converging to the same point; asserting
    ``<= tol`` would be asserting something the stopping rule does not promise,
    and it fails for exactly that reason on the slowly-contracting arrangement
    (6.95e-06 against a 5.20e-06 tolerance -- 1.34x, inside the bound).  An
    agreement far *below* the tolerance would be the suspicious result: it would
    mean the arms had not actually started apart.
    """
    fps = [r["fixed_point"] for r in rows if not r["diverged"]]
    worst = 0.0
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            worst = max(worst, _rms(fps[i][0] - fps[j][0], fps[i][1] - fps[j][1]))
    return {"worst_pair_disagreement": worst, "arms": len(fps)}


# ===========================================================================
# 3 -- the multiplier space, and the accelerator ladder
# ===========================================================================


@dataclass
class MultiplierSeam:
    """`solve.py`'s interface problem on one seam, evaluated MATRIX-FREE.

    ``r(lambda) = sum_i P_i^* F_i(P_i lambda)``, with ``F_i`` the agent's actual
    flux rather than a probed linearization, so **one residual evaluation is two
    expert calls** and an iteration count converts to a cost.  That is the
    arrangement in which "is the coupling or the subdomain solve the bottleneck"
    has an answer, and it is not the arrangement `solve.py` runs: that one
    assembles ``S`` first and then iterates on the matrix, where an iteration is
    a 16x16 matvec and costs nothing.

    Both are legitimate and they answer different questions.  The probe is 34
    expert calls; the ladder below prices every accelerator against it.
    """

    a: WindowAgent
    b: WindowAgent
    face_a: str = "xhi"
    face_b: str = "xlo"
    m_eff: int = W.M_EFF
    calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.a.n != self.b.n:
            raise ValueError("the two sides present different numbers of face cells")
        self.B = W.fourier_basis(self.a.n, self.m_eff)

    def _reduce(self, e_V: np.ndarray) -> np.ndarray:
        """The forced adjoint ``R = G_M^-1 P^T G_V = h P^T``.

        Written out rather than routed through `transfer.Prolongation` because
        this class is called tens of thousands of times in a sweep count; it is
        the same map `window_ns.face_prolongation` declares, and
        `test_tier38` checks the two against each other rather than trusting it.
        """
        return H * (self.B.T @ np.asarray(e_V, dtype=float))

    def residual(self, lam: np.ndarray) -> np.ndarray:
        self.calls += 2
        trace = self.B @ np.asarray(lam, dtype=float)
        return (self._reduce(self.a.respond(f"{self.face_a}:MECH", trace))
                + self._reduce(self.b.respond(f"{self.face_b}:MECH", trace)))

    # -- the probe ---------------------------------------------------------

    def probe(self, eps: float = 1e-4) -> dict[str, Any]:
        """``S``, and the three numbers the accelerators are chosen from.

        ``alpha_star`` is ``diag(S)``, which `probe.py` has emitted since Tier 0
        with the comment that in a Fourier interface basis it IS the measured
        symbol mode by mode -- the optimal Robin coefficient, read off rather
        than derived, which for a frozen expert is the only available route.
        **`mode_shares` is the only thing in the vault that has ever read it, and
        it reads it as a ratio.**  Nothing has used it as a transmission
        condition, which is what it is.
        """
        S = np.zeros((self.m_eff, self.m_eff))
        blocks = {}
        n0 = self.calls
        for ag, face in ((self.a, self.face_a), (self.b, self.face_b)):
            z = ag.respond(f"{face}:MECH", np.zeros(ag.n))
            self.calls += 1
            cols = []
            for k in range(self.m_eff):
                cols.append((ag.respond(f"{face}:MECH", eps * self.B[:, k]) - z) / eps)
                self.calls += 1
            blocks[face] = self._reduce(np.column_stack(cols))
            S += blocks[face]
        sv = np.linalg.svd(S, compute_uv=False)
        alpha = np.diag(S).copy()
        D = np.diag(alpha)
        DS = np.linalg.solve(D, S)
        svp = np.linalg.svd(DS, compute_uv=False)
        return {
            "S": S, "blocks": blocks,
            "norm_S": float(sv[0]), "beta": float(sv[-1]),
            "kappa": float(sv[0] / max(sv[-1], 1e-300)),
            "alpha_star": alpha,
            "off_diagonal_share": float(np.linalg.norm(S - D) / np.linalg.norm(S)),
            "kappa_preconditioned": float(svp[0] / max(svp[-1], 1e-300)),
            # Richardson converges when the preconditioned spectrum sits inside
            # (0, 2); a better condition number does not put it there, which is
            # why the undamped arm below diverges while cutting kappa by 3x.
            "rho_D_inv_S": float(np.max(np.abs(np.linalg.eigvals(DS)))),
            "probe_calls": self.calls - n0,
        }


def accelerator_ladder(seam: MultiplierSeam, probe: dict[str, Any],
                       tol_fracs: Sequence[float] = (1e-2, 1e-3, 1e-6),
                       cap: int = 600) -> list[dict[str, Any]]:
    """Every accelerator, priced in expert calls INCLUDING the probe each needs.

    The probe is 34 calls and three of the four arms cannot be built without it,
    so quoting sweeps alone would hide the cost that decides the comparison.  The
    fourth -- plain Richardson -- needs only ``||S||`` for its step size, which
    is still the probe; a genuinely probe-free Richardson would have to estimate
    it, and that is charged here as the full probe rather than as zero, which is
    the reading least favourable to the conclusion.
    """
    S = probe["S"]
    D = np.diag(probe["alpha_star"])
    rho = probe["rho_D_inv_S"]
    pc = probe["probe_calls"]
    m = seam.m_eff

    def richardson(Minv, tol):
        lam = np.zeros(m)
        for k in range(cap):
            r = seam.residual(lam)
            nr = float(np.linalg.norm(r))
            if not np.isfinite(nr):
                return k, nr
            if nr <= tol:
                return k, nr
            lam = lam - Minv @ r
        return cap, float(np.linalg.norm(seam.residual(lam)))

    def newton(tol):
        lam = np.zeros(m)
        for k in range(cap):
            r = seam.residual(lam)
            nr = float(np.linalg.norm(r))
            if nr <= tol:
                return k, nr
            lam = lam - np.linalg.solve(S, r)
        return cap, float(np.linalg.norm(seam.residual(lam)))

    arms = [
        ("richardson, theta = 1/||S||",
         lambda t: richardson(np.eye(m) / probe["norm_S"], t)),
        ("richardson, alpha_star undamped",
         lambda t: richardson(np.linalg.inv(D), t)),
        ("richardson, alpha_star damped",
         lambda t: richardson(np.linalg.inv(D) / rho, t)),
        ("newton on the probed S",
         newton),
    ]
    seam.calls = 0
    r0 = float(np.linalg.norm(seam.residual(np.zeros(m))))
    out = []
    for frac in tol_fracs:
        for label, fn in arms:
            seam.calls = 0
            iters, nr = fn(frac * r0)
            out.append({
                "tol_frac": frac, "accelerator": label, "iterations": iters,
                "sweep_calls": seam.calls, "probe_calls": pc,
                "total_calls": seam.calls + pc,
                "final_residual": nr, "converged": bool(np.isfinite(nr) and nr <= frac * r0),
                "r0": r0,
            })
    return out


# ===========================================================================
# 4 -- the halo threshold
# ===========================================================================


def halo_convergence(u_full: np.ndarray, v_full: np.ndarray,
                     halos: Sequence[int] = (4, 8, 12, 16, 18, 20, 21, 22, 24, 32, 48),
                     *, expose_elliptic: bool = True, project: bool = True,
                     cap: int = 60, ramp: int = 8) -> list[dict[str, Any]]:
    """Sweep the overlap and report where the iteration's contraction crosses one.

    **R10's halo rule is stated as an accuracy condition on a one-sweep scheme**:
    a halo narrower than the agent's reach over the exchange interval carries
    stale data into the interior, and `assembly.py` charges the contaminated
    cells for it.  Nothing says what it does to an ITERATED scheme, because until
    this study nothing in the vault iterated one.

    **Measured, it is a convergence condition -- and only for the embedded
    agent.**  With ``expose_elliptic=False`` the contraction crosses 1 between
    halo 18 and halo 20 and the declared reach is 20; with it True the iteration
    converges at every halo from 4 up.  So run it in both arrangements or it says
    the opposite of what it means.

    The window count and size are held fixed and only the domain shrinks, so the
    overlap is the single variable: every other quantity the rate could depend on
    -- window size, ramp, dt, the state, the number of seams -- is the same at
    every row.
    """
    rows = []
    for halo in halos:
        mono_n = 2 * PO.EXPERT_RES - halo
        if mono_n > min(u_full.shape[0], u_full.shape[1]):
            continue
        til = HybridTiling(mono_n=mono_n, ramp=ramp)
        ex = SchwarzExchange(u_full, v_full, tiling=til,
                             expose_elliptic=expose_elliptic, project=project)
        out = ex.march(ex.u0, ex.v0, cap=cap, tol=1e-14)
        g = out["contraction"]
        rows.append({
            "halo": til.halo, "mono_n": mono_n,
            "overlap_length": til.halo * H,
            "sweeps": out["sweeps"], "diverged": out["diverged"],
            "contraction": g,
            "converging": bool(out["diverged"] is False and np.isfinite(g) and g < 1.0),
            "covers_reach": bool(til.halo >= DOMAIN_OF_DEPENDENCE),
        })
    return rows


def halo_threshold(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The largest diverging halo and the smallest converging one, and the gap.

    Reported as a bracket rather than a number, because the sweep is on a grid
    and a bracket is what a grid measures.  ``declared_reach`` is beside it so
    the comparison is in the record rather than in a sentence about it.
    """
    div = [r["halo"] for r in rows if not r["converging"]]
    con = [r["halo"] for r in rows if r["converging"]]
    return {
        "largest_diverging": max(div) if div else None,
        "smallest_converging": min(con) if con else None,
        "declared_reach": DOMAIN_OF_DEPENDENCE,
        "bracket_contains_reach": bool(
            div and con and max(div) < DOMAIN_OF_DEPENDENCE <= min(con)),
    }


# ===========================================================================
# 5 -- the graph
# ===========================================================================


def make_experts(u_full, v_full, tiling=DEFAULT_TILING, dt: float = MACRO_DT,
                 nu: float = NU, expose_elliptic: bool = True):
    """Four `window_ns.WindowAgent`s on this tiling.  **No new record is declared.**

    The capability record is `window_ns.window_capabilities` unchanged, which is
    the affordability claim being used rather than restated: a study that needed
    a new L1 record for a new *scheme* would be a study about the record.
    """
    m = tiling.mono_n
    u = np.ascontiguousarray(np.asarray(u_full, dtype=float)[:m, :m])
    v = np.ascontiguousarray(np.asarray(v_full, dtype=float)[:m, :m])
    us, vs = tiling.cut(u), tiling.cut(v)
    return {
        name: WindowAgent(agent_id=name, u0=us[k], v0=vs[k],
                          shared_faces=tiling.artificial_faces(*tiling.offsets[k]),
                          dt=dt, nu=nu, expose_elliptic=expose_elliptic)
        for k, name in enumerate(tiling.names)
    }


def build(u_full, v_full, tiling=DEFAULT_TILING, dt: float = MACRO_DT,
          nu: float = NU, expose_elliptic: bool = True, experts=None):
    """The classical four-window graph on the checkpoint's own geometry.

    **This geometry has never carried a classical graph.**  `window_ns` tiles
    255 with 138-cell windows; `poseidon` tiles 235 with 128-cell ones because
    the checkpoint is fixed at 128.  Putting `WindowNS` windows on the SECOND
    geometry gives the checkpoint's graph a same-shaped classical column -- same
    domain, same overlap, same ramp, same four seams, same prolongations, a
    different expert -- which is what makes the predictor arm and the
    subdomain-owner arm comparable at all.

    ``measured`` is None and stays None.  Every constant in `MeasuredConstants`
    is about composition ERROR; this study measures the ITERATION, and carrying
    another graph's numbers to make the compile look better would be exactly the
    move the audit exists to catch.
    """
    experts = experts or make_experts(u_full, v_full, tiling, dt, nu, expose_elliptic)
    agents = [Agent(n, window_capabilities(experts[n]), domain=f"tile {n}")
              for n in tiling.names]
    connections = [
        Connection(
            seam_id=seam_id,
            a=(a_id, f"{a_face}:MECH"),
            b=(b_id, f"{b_face}:MECH"),
            port_type=PortType.MECH,
            derive_space=True,
            geometrically_coincident=True,
            expected_null_dim=0,
            note="seam presented through a declared 16-mode Fourier prolongation",
        )
        for seam_id, ((a_id, a_face), (b_id, b_face)) in SEAMS.items()
    ]
    return (
        CaseGraph(
            name="neural-interface-2x2",
            agents=agents,
            connections=connections,
            decomposition=Decomposition.OVERLAPPING,
            overlap=tiling.halo * H,
            overlap_cells=tiling.halo,
            partition_of_unity=tiling.partition_of_unity(),
            global_fields=[
                GlobalField(
                    "pressure",
                    produced_by=tuple(tiling.names),
                    note=("the elliptic part. Global when expose_elliptic, which is "
                          "the arrangement whose Schwarz contraction is 0.011; "
                          "per-window otherwise, which R10 refuses and whose "
                          "contraction is 0.904"),
                ),
            ],
            cross_points=("centre",),
            macro_dt=dt,
            measured=None,
            note=(f"four reference.WindowNS windows of {tiling.n} cells tiling "
                  f"{tiling.mono_n}x{tiling.mono_n}, halo {tiling.halo}: the "
                  f"checkpoint's own geometry with classical subdomain owners. "
                  f"The interface PREDICTOR is not an agent and appears nowhere "
                  f"in this graph, which is the point"),
        ),
        experts,
    )
