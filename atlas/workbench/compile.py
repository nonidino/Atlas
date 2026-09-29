"""Compile a case with the Atlas compiler: the case as a `CaseGraph`, a verdict per seam.

Plan step 2's graph half of [[outcome-c4-path-to-declarative-cases]], and step 6
of [[showcase-library-plan]].  A case becomes a graph through its family: every
family's adapter module has a ``case_graph(spec)`` (the registry names the
module), which declares one agent per window or piece with a capability record,
one connection per seam, and the decomposition around them.  The compiler
(`atlas/compiler.py`) then derives the scheme and returns its decision record,
and this module reads that record **per seam**: each seam's verdict is the worst
of the decisions about it -- admit, admit-uncertified or refuse -- with the rules
that set it.  Nothing here grades the case; the verdicts and their rules are the
compiler's.

**Seams are derived from the geometry, not listed.**  Two windows form a seam
when a cut face of one has its outside cell in the other: across an overlap (the
overlapping styles), or across a shared face (style C).  Each side's port is its
own cut faces there, ordered along the interface, and each port declares its
prolongation from a small modal space (an orthonormal Fourier basis, the
vault's `window_ns` construction) to those faces; the connection asks the
compiler to derive the common space (``derive_space``).

**Cross-points are declared, never left to the proxy (W162).**  ``cross_points``
names every place three or more windows overlap (`geometry.analyse_windows`),
and is ``()`` -- the declaration that there are none -- when there are none.

**The agents' responses are the families' own arithmetic.**  The compiler's L4
probe perturbs each port's trace and reads the response, so an agent's
``boundary_response`` is a real local solve: a finite-volume window given the
temperatures (or potentials) on its cut faces returns the flow through them
(`FVAgent`), an explicit window the flux its step would send, a structure's
window its reaction to prescribed displacements.  A family whose coupling the
port algebra cannot express is refused before the compiler, by the package's
own vocabulary, and the page says so.
"""

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import fv, styles
from . import geometry as geo

#: modes each port's prolongation carries (fewer on a short interface)
M_MODES = 8


class CompileRefused(RuntimeError):
    """The case cannot be put to the compiler as it stands; the message says why."""


# ---------------------------------------------------------------------------
# interfaces: modal prolongations over a port's faces
# ---------------------------------------------------------------------------


def modes_for(n: int) -> int:
    """How many modes a port of ``n`` faces carries: M_MODES, or half the faces on a
    short interface (so every mode is resolved), and one on a single face."""
    return max(1, min(M_MODES, n // 2))


def fourier_basis(n: int, h: float, m: int) -> np.ndarray:
    """(n, m) real Fourier modes on ``n`` faces of length ``h``, orthonormal in the
    h-weighted pairing: ``sum_j phi_k phi_l h = delta_kl`` on the midpoints
    (`cases/window_ns.fourier_basis`'s construction)."""
    length = n * h
    y = (np.arange(n) + 0.5) * h
    cols = [np.full(n, 1.0 / np.sqrt(length))]
    k = 1
    while len(cols) < m:
        w = 2.0 * np.pi * k * y / length
        cols.append(np.sqrt(2.0 / length) * np.cos(w))
        if len(cols) < m:
            cols.append(np.sqrt(2.0 / length) * np.sin(w))
        k += 1
    return np.column_stack(cols[:m])


def face_prolongation(agent_id: str, port: str, n: int, h: float):
    """P : M -> V, from ``modes_for(n)`` Fourier modes to the port's ``n`` faces."""
    from atlas.transfer import Prolongation
    m = modes_for(n)
    return Prolongation(agent_id=agent_id, port_name=port, matrix=fourier_basis(n, h, m),
                        gram_V=h * np.eye(n),
                        label=f"{m}-mode real Fourier basis over {n} faces "
                              f"(atlas/workbench/compile.py)")


# ---------------------------------------------------------------------------
# seams, from the geometry
# ---------------------------------------------------------------------------


@dataclass
class FaceSeam:
    """A seam between two windows: each side's cut faces there, ordered along it."""

    seam_id: str
    a: str
    b: str
    faces: dict[str, np.ndarray]           # window id -> indices into its cut-face list


def cell_order(sys_: fv.LocalSystem, faces: np.ndarray, nx: int) -> np.ndarray:
    """Faces ordered along the interface: by the inside cell's row, then column."""
    cells = sys_.idx[sys_.face_rows[faces]]
    return faces[np.lexsort((cells % nx, cells // nx))]


def face_seams(windows: list[tuple[str, tuple]], systems: dict[str, fv.LocalSystem],
               nx: int) -> list[FaceSeam]:
    """Every pair of windows one of whose cut faces opens into the other, in window
    order.  Across an overlap each side's port is its own ring inside the other;
    across a shared face the two ports are the same faces seen from each side."""
    cells = {n: set(systems[n].idx.tolist()) for n, _b in windows}
    out = []
    for i, (na, _ba) in enumerate(windows):
        for nb, _bb in windows[i + 1:]:
            sa, sb = systems[na], systems[nb]
            fa = np.array([k for k, c in enumerate(sa.face_outside.tolist()) if c in cells[nb]],
                          dtype=np.int64)
            fb = np.array([k for k, c in enumerate(sb.face_outside.tolist()) if c in cells[na]],
                          dtype=np.int64)
            if fa.size and fb.size:
                out.append(FaceSeam(f"{na}|{nb}", na, nb,
                                    {na: cell_order(sa, fa, nx), nb: cell_order(sb, fb, nx)}))
    return out


def cross_points(spec) -> tuple[str, ...]:
    """W162: the geometric cross-points, named, or ``()`` when there are none."""
    d = spec.domain
    an = geo.analyse_windows(d.nx, d.ny, [(w.id, (w.x0, w.y0, w.nx, w.ny))
                                          for w in spec.windows], spec.coupling.ramp_cells)
    return tuple("+".join(names) for names, _box in an.cross_points)


# ---------------------------------------------------------------------------
# a finite-volume window as an agent
# ---------------------------------------------------------------------------


@dataclass
class CutFaces:
    """A set of cells' faces that leave it: ``idx`` the cells (ascending),
    ``face_rows`` the inside cell's position in ``idx``, ``face_outside`` the
    outside cell -- `fv.LocalSystem`'s names, without assembling anything."""

    idx: np.ndarray
    face_rows: np.ndarray
    face_outside: np.ndarray


def cut_faces(f: fv.Field, cells: np.ndarray) -> CutFaces:
    cells = np.asarray(np.unique(cells), dtype=np.int64)
    inside = np.zeros(f.n, dtype=bool)
    inside[cells] = True
    loc = np.full(f.n, -1, dtype=np.int64)
    loc[cells] = np.arange(cells.size)
    P, Q, _G, _F = fv.interior_faces(f)
    rows, outs = [], []
    for mine, other in ((P, Q), (Q, P)):
        sel = inside[mine] & ~inside[other]
        rows.append(loc[mine[sel]])
        outs.append(other[sel])
    return CutFaces(cells, np.concatenate(rows), np.concatenate(outs))


class FVAgent:
    """One window or piece of an `fv.py` family, as the compiler's agent.

    ``implicit``: the window's step is a solve (steady, or backward Euler with
    ``diag_add`` and the state's storage in ``extra``), so its response to the
    face values on one port -- every other cut face held at the probe base -- is
    the flow through that port's faces after the solve: the window's own
    Dirichlet-to-Neumann map.  ``implicit=False``: the step is explicit, so the
    response is the flow its step sends through those faces with the trace as
    the outside value, at the probe state (one exchange per step).

    ``per_effort``: divide the flow per unit area by the face value, which makes
    a heat flux THERM's declared flow, the entropy flux ``q_n / T``.
    """

    def __init__(self, agent_id: str, f: fv.Field, cells: np.ndarray, *, implicit: bool,
                 base: float, diag_add: np.ndarray | None = None,
                 extra: np.ndarray | None = None, state: np.ndarray | None = None,
                 per_effort: bool = False):
        self.agent_id = agent_id
        self.f = f
        self.implicit = implicit
        self.per_effort = per_effort
        self.base = float(base)
        self.sys = fv.assemble(f, cells, "dirichlet" if implicit else "neighbour",
                               diag_add=diag_add)
        #: the cut faces: the Dirichlet side's own list for a solve; for an
        #: explicit step (whose cut faces may carry the flow, which a face-cut
        #: assembly refuses) read straight off the face list
        self.cut = self.sys if implicit else cut_faces(f, cells)
        self.extra = extra
        self.lu = styles.Factor(self.sys.A) if implicit else None
        self.state = (np.full(f.n, self.base) if state is None
                      else np.asarray(state, dtype=float))
        self.ports: dict[str, np.ndarray] = {}
        self.calls = 0

    def respond(self, port: str, trace: np.ndarray) -> np.ndarray:
        self.calls += 1
        idx = self.ports[port]
        trace = np.asarray(trace, dtype=float).ravel()
        s = self.cut
        if self.implicit:
            lam = np.full(s.face_rows.size, self.base)
            lam[idx] = trace
            rhs = s.b.copy() if self.extra is None else s.b + self.extra[s.idx]
            np.add.at(rhs, s.face_rows, s.face_g * lam)
            u = self.lu.solve(rhs)
            q = s.face_g[idx] * (trace - u[s.face_rows[idx]])
        else:
            # the explicit step's flow through each cut face: the full face
            # conductance and the upwinded advective flow, the outside value the trace
            u_in = self.state[s.idx[s.face_rows[idx]]]
            q = self._explicit_flow(idx, u_in, trace)
        #: the flow INTO the window, per unit area: the Steklov-Poincare
        #: convention (each side against its own outward normal, each block
        #: positive), so raising the effort raises the response.  The first
        #: draft returned the flow out and the compiler read a passivity defect
        #: of 299 on the wall's seam (seen, 2026-09-29)
        flow = q / self.f.dx
        return flow / trace if self.per_effort else flow

    def _explicit_flow(self, idx, u_in, trace):
        """The flow into the window through each face: diffusion from the outside
        value, and the upwinded advective flow."""
        G, flow = self._faces(idx)
        up = np.where(flow > 0.0, u_in, trace)
        return G * (trace - u_in) - flow * up

    def _faces(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(the full conductance, the advective flow inside -> outside) of each face in
        ``idx``, looked up once per port."""
        key_ = idx.tobytes()
        if key_ not in getattr(self, "_face_cache", {}):
            s = self.cut
            P, Q, G, F = fv.interior_faces(self.f)
            where = {}
            for k, (p, q_) in enumerate(zip(P.tolist(), Q.tolist())):
                where[(p, q_)] = (k, 1.0)
                where[(q_, p)] = (k, -1.0)
            ks, signs = zip(*[where[(ci, co)] for ci, co in
                              zip(s.idx[s.face_rows[idx]].tolist(),
                                  s.face_outside[idx].tolist())])
            ks, signs = np.array(ks), np.array(signs)
            self.__dict__.setdefault("_face_cache", {})[key_] = (G[ks], signs * F[ks])
        return self._face_cache[key_]

    def probe_base(self, port: str) -> np.ndarray:
        return np.full(self.ports[port].size, self.base)

    def storage(self, state=None) -> float:
        u = self.state if state is None else np.asarray(state, dtype=float)
        return float(0.5 * np.sum(u[self.sys.idx] ** 2))


# ---------------------------------------------------------------------------
# the graph of a finite-volume family
# ---------------------------------------------------------------------------


def fv_graph(spec, f: fv.Field, *, family: str, port_type: str, scales: dict[str, float],
             implicit: bool, base: float, per_effort: bool, lambda_ref: str,
             diag_add: np.ndarray | None = None, extra: np.ndarray | None = None,
             state: np.ndarray | None = None, passengers: tuple[str, ...] = (),
             response_note: str = "", name: str | None = None,
             validity: Callable[..., bool] | None = None,
             families: dict[str, str] | None = None):
    """The case's windows as `FVAgent`s, its seams from the geometry, and the graph.

    ``validity`` is the family's own predicate for where its arithmetic holds
    (a classical solver has no training distribution; its validity is its
    stability and the case's own rules), called by the compiler as C8 asks.
    ``families`` gives a window its own governing family where the pieces are
    different physics (the cooled block: a coolant and a solid)."""
    from atlas.capability import (BCChannel, ClaimType, Direction, EllipticSubsolve,
                                  ExpertCapabilities, MotionClass, TimeDiscretization,
                                  port_decl)
    from atlas.graph import Agent, CaseGraph, Connection, Decomposition
    from atlas.ports import PortType, ResponseHalf

    d = spec.domain
    style = spec.coupling.style
    overlapping = style in ("A", "B")
    windows = [(w.id, (w.x0, w.y0, w.nx, w.ny)) for w in spec.windows]
    from .families.conduction import window_cells
    agents_fv = {n: FVAgent(n, f, window_cells(d.nx, b), implicit=implicit, base=base,
                            diag_add=diag_add, extra=extra, state=state,
                            per_effort=per_effort)
                 for n, b in windows}
    seams = face_seams(windows, {n: a.cut for n, a in agents_fv.items()}, d.nx)
    if not seams:
        raise CompileRefused("no two windows meet or overlap, so there is no seam to "
                             "compile: one window is not a composition")
    pt = PortType(port_type)
    ports: dict[str, list] = {n: [] for n, _b in windows}
    conns = []
    for s in seams:
        pname = {}
        for side in (s.a, s.b):
            faces = s.faces[side]
            name_ = f"{s.seam_id}:{port_type}"
            agents_fv[side].ports[name_] = faces
            pname[side] = name_
            ports[side].append(port_decl(
                name=name_, port_type=pt,
                geometry=f"{faces.size} cut faces of {side} at the seam with "
                         f"{s.b if side == s.a else s.a}",
                direction=Direction.BIDIRECTIONAL, nondim=dict(scales),
                passengers=passengers, effective_resolution=modes_for(faces.size),
                motion_class=MotionClass.STATIC, response_half=ResponseHalf.FLOW,
                prolongation=face_prolongation(side, name_, faces.size, d.dx),
                note=response_note))
        conns.append(Connection(
            seam_id=s.seam_id, a=(s.a, pname[s.a]), b=(s.b, pname[s.b]), port_type=pt,
            derive_space=True, geometrically_coincident=not overlapping,
            expected_null_dim=0,
            cut_axis=Decomposition.OVERLAPPING if overlapping
            else Decomposition.NON_OVERLAPPING,
            note=("across an overlap: each side's ring inside the other" if overlapping
                  else "across a shared face: the same faces seen from each side")))
    agents = []
    for n, _b in windows:
        a = agents_fv[n]
        caps = ExpertCapabilities(
            expert_id=n, ports=ports[n],
            bc_channel=BCChannel.DIRICHLET, bc_time_varying=True,
            elliptic_subsolve=EllipticSubsolve.EMBEDDED if implicit else EllipticSubsolve.NONE,
            time_discretization=(TimeDiscretization.IMPLICIT if implicit
                                 else TimeDiscretization.EXPLICIT),
            stencil_radius=1, substeps_per_macro_step=1,
            dt_native=float(spec.run.macro_dt), storage=a.storage, validity=validity,
            governing_family=(families or {}).get(n, family), lambda_ref=lambda_ref,
            claim_types=frozenset({ClaimType.TRAJECTORY}),
            weight_hash=f"workbench-fv/{family}",
            boundary_response=a.respond, probe_base=a.probe_base,
            reproducibility_floor=float(np.finfo(float).eps), deterministic=True,
            note="atlas/workbench/fv.py: the window's own finite volumes")
        agents.append(Agent(n, caps, domain=f"window {n}"))
    kw: dict[str, Any] = {}
    if overlapping:
        from .tiling import RectangleTiling
        tiling = RectangleTiling(d.nx, d.ny, windows, spec.coupling.ramp_cells)
        ov = geo.analyse_windows(d.nx, d.ny, windows, spec.coupling.ramp_cells).overlaps
        cells_ov = min((min(b[2], b[3]) for _a, _b, b in ov), default=0)
        kw = dict(overlap=cells_ov * d.dx, overlap_cells=int(cells_ov),
                  partition_of_unity=tiling.partition_of_unity())
    graph = CaseGraph(
        name=name or f"workbench-{spec.name}",
        agents=agents, connections=conns,
        decomposition=Decomposition.OVERLAPPING if overlapping
        else Decomposition.NON_OVERLAPPING,
        cross_points=cross_points(spec), macro_dt=float(spec.run.macro_dt),
        note=f"the workbench case {spec.name!r}, family {family}, style {style}", **kw)
    return graph, agents_fv


# ---------------------------------------------------------------------------
# compiling, and reading the record per seam
# ---------------------------------------------------------------------------


@dataclass
class SeamVerdict:
    seam: str
    between: tuple[str, str]
    port_type: str
    verdict: str
    rules: list[dict[str, str]] = field(default_factory=list)


@dataclass
class CompileSummary:
    """What the page shows and the record keeps."""

    case: str
    family: str
    verdict: str                        # admit / admit-uncertified / refuse
    seams: list[SeamVerdict]
    other: list[dict[str, str]]         # decisions about the graph, agents, the run
    seconds: float
    agents: int
    cross_points: tuple[str, ...] | None
    refused_before: str | None = None   # the reason the case never reached the compiler
    report: str = ""
    when: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"case": self.case, "family": self.family, "verdict": self.verdict,
                "seams": [vars(s) for s in self.seams], "other": self.other,
                "seconds": self.seconds, "agents": self.agents,
                "cross_points": list(self.cross_points) if self.cross_points is not None
                else None, "refused_before": self.refused_before, "when": self.when}


def _row(d) -> dict[str, str]:
    return {"layer": d.layer, "rule": d.rule, "verdict": d.verdict.value,
            "subject": d.subject or "", "message": d.message}


def summarize(spec, graph, result, seconds: float) -> CompileSummary:
    from atlas.verdict import ADMIT
    seam_ids = [c.seam_id for c in graph.connections]
    by: dict[str, list] = {}
    other = []
    for d in result.decisions:
        if d.subject in seam_ids:
            by.setdefault(d.subject, []).append(d)
        elif d.verdict is not ADMIT:
            other.append(_row(d))
    seams = []
    for c in graph.connections:
        ds = by.get(c.seam_id, [])
        worst = max((d.verdict for d in ds), key=lambda v: v.severity, default=ADMIT)
        seams.append(SeamVerdict(c.seam_id, c.agents, c.port_type.value, worst.value,
                                 [_row(d) for d in ds if d.verdict is not ADMIT]))
    return CompileSummary(
        case=spec.name, family=spec.physics.family, verdict=result.verdict.value,
        seams=seams, other=other, seconds=seconds, agents=len(graph.agents),
        cross_points=graph.cross_points, report=result.report(),
        when=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))


def case_graph(spec):
    """The case as a `CaseGraph`, through its family's adapter."""
    from .runner import adapter_for
    module = adapter_for(spec.physics.family)
    builder: Callable | None = getattr(module, "case_graph", None)
    if builder is None:
        raise CompileRefused(f"the {spec.physics.family} family declares no graph")
    return builder(spec)


def compile_case(spec) -> CompileSummary:
    """Build the case's graph and compile it.  A case the port algebra cannot express
    comes back refused before the compiler, with the package's own reason."""
    from atlas.compiler import compile_scheme
    from atlas.holes import NamedHoleError
    t0 = time.perf_counter()
    try:
        built = case_graph(spec)
        graph = built[0] if isinstance(built, tuple) else built
    except (CompileRefused, NamedHoleError) as exc:
        return CompileSummary(case=spec.name, family=spec.physics.family, verdict="refuse",
                              seams=[], other=[], seconds=time.perf_counter() - t0,
                              agents=0, cross_points=None,
                              refused_before=f"{type(exc).__name__}: {exc}",
                              when=datetime.datetime.now().astimezone().isoformat(
                                  timespec="seconds"))
    result = compile_scheme(graph)
    return summarize(spec, graph, result, time.perf_counter() - t0)


class CompileJob:
    """One compile of one committed case, in a background thread (the 21-rotor farm's
    graph takes about a minute and a half to probe) or the caller's.  The case is
    copied when the job is made, so an edit made while it compiles cannot reach
    it; the record goes beside the case file, ``compile-<time>.json``."""

    def __init__(self, spec, results_dir: str | None = None,
                 on_finish: Callable[["CompileJob"], None] | None = None):
        import threading
        self.spec = spec.copy_deep()
        self.committed_json = self.spec.to_json()
        self.results_dir = results_dir
        self.on_finish = on_finish
        self.status = "created"
        self.summary: CompileSummary | None = None
        self.error: str | None = None
        self.record_path: str | None = None
        self.started = time.time()
        self._thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        return self.status in ("starting", "compiling")

    def start(self) -> None:
        import threading
        self.status = "starting"
        self._thread = threading.Thread(target=self._run, name="workbench-compile",
                                        daemon=True)
        self._thread.start()

    def run_blocking(self) -> "CompileJob":
        self._run()
        return self

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        import traceback
        self.status = "compiling"
        try:
            self.summary = compile_case(self.spec)
            if self.results_dir:
                from .runner import write_record
                rec = {"schema": "atlas-workbench/compile@1", **self.summary.as_dict(),
                       "report": self.summary.report,
                       "case": __import__("json").loads(self.committed_json)}
                self.record_path = write_record(rec, self.results_dir, prefix="compile-")
            self.status = "done"
        except Exception as exc:                          # the page says what broke
            self.error = f"{type(exc).__name__}: {exc}"
            self.trace = traceback.format_exc()
            self.status = "failed"
        finally:
            if self.on_finish is not None:
                try:
                    self.on_finish(self)
                except Exception:                         # pragma: no cover
                    pass


__all__ = ["M_MODES", "CompileRefused", "modes_for", "fourier_basis", "face_prolongation",
           "FaceSeam", "face_seams", "cross_points", "FVAgent", "fv_graph", "SeamVerdict",
           "CompileSummary", "summarize", "case_graph", "compile_case", "CompileJob"]
