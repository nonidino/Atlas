"""CS-8 -- is a substitution certificate a property of the EXPERT, or of the
STATE it was probed at?

The eighth real case study, and the second of `case-study-ladder-to-f1` Phase A.
It is the only rung of `f1-pathmap-and-end-goal`'s ladder that tests the
FOUNDATION-MODEL claim rather than the coupling claim, and section 3.1 flags it
as *the one most likely to be skipped by accident*.

Why it decides something
------------------------

A twenty-agent car carries roughly twenty conformance certificates and forty
seam certificates.  `composition.certify_substitution` issues one per swap per
seam, and every one of them is issued at a probe state -- because `probe_block`
linearizes about `probe_base`, and for a nonlinear expert the block it returns
is the TANGENT map there (`atlas-and-standard-dd-theory` section on nonlinear
substructuring).  Nothing in the record says how far that reading travels.

  * If a certificate is a property of the EXPERT, it is issued once per expert
    and reused across the design space, and the economic argument for a reusable
    expert library survives: search wide, certify once.
  * If it is a property of the STATE, the plug-in claim is **per-design rather
    than per-expert**, and every candidate design re-pays the certification cost.

This file does not build a new physics graph and does not need one.  CS-7's
ladder already supplies developed states at several sizes, and `wake_array`
already supplies the geometry; what CS-8 adds is a **probe design** over them --
a seam taxonomy derived from the layout, a state schedule, the two controls that
make a spread readable, and the extraction of the five quantities a certificate
actually discriminates on.  That is a declaration, and it lives here for the
same reason the tiling does: so no driver invents it.

The design, and both controls
-----------------------------

Two factors crossed, plus two controls, and the controls are the point.

  factor STATE     one seam, one graph, the same expert pair, probed at
                   macro-steps 0, 10, 30, 60 and 110 of ONE trajectory -- so
                   the flow develops from a uniform stream to a settled farm
                   and nothing else moves.
  factor SEAM      one state, one graph, the same expert pair, probed at seams
                   in four flow regimes: near-freestream, a shallow single
                   wake, a deep multi-turbine wake, and a bypass region.
  factor GEOMETRY  the same seam IDS in a 6-window graph and a 12-window one,
                   which is the cross-geometry half of the question.

  control ZERO     the FREESTREAM state.  Every window holds the same uniform
                   field, every ``_full`` seam has the same port on the same
                   face, so the four regime seams must return the SAME operator
                   -- bit for bit, not to a tolerance.  A seam-to-seam spread
                   there would mean the regime labels are picking up the port
                   and not the flow, and every number in the SEAM factor would
                   be uninterpretable.  This is `tier0-measurements` section 8's
                   own lesson: run the control that could redirect the search.
  control FLOOR    a re-probe of an identical (seam, state) pair with freshly
                   constructed experts, from the state reloaded off disk.  It is
                   the reproducibility floor of the whole pipeline, and it is
                   what every spread below is quoted against.  A movement is a
                   result only if it is larger than this.

What is held fixed
------------------

Everything CS-7 held fixed -- dx, dt_macro, overlap 16, ramp 8, nu_ref, the disk
model -- plus the two this case study adds:

  the trajectory      every probe state comes off ONE march per graph, the
                      R10-compliant classical column: `wake_array
                      .exposed_reference_solver` (elliptic part EXPOSED) with
                      `assembly.ProjectedAssembly` applied once after the blend.
                      Tier 19 section 19.6 is why: the embedded classical column
                      is not finite past macro-step 82 at six windows, and a
                      probe taken on a diverging trajectory measures the
                      divergence.  **Tier 18's published wake-array state is
                      excluded for exactly that reason** (section 19.6's last
                      row).
  the interface base  both experts are linearized about the SAME ``seam_base``,
                      the mean of the two sides' declared probe bases, so a
                      difference between the two blocks is the expert and not
                      the base.  `assemble_seam`'s ``seam_base`` argument.

W105 is asserted rather than assumed
------------------------------------

A graph can clear `R10` and `R10b` and never apply the elliptic part at all --
measured, and it is `gap-worklist` W105.  So `build` below refuses to return a
graph whose assembly is not a `ProjectedAssembly`, and `assert_projected` is
called on every graph this case study compiles.  The rule cannot check it yet;
the case study can, and a case study that relied on the projection without
checking for it would be the W105 hole with a case study in it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..assembly import ProjectedAssembly
from ..capability import EllipticSubsolve
from ..graph import CaseGraph, MeasuredConstants
from . import scaling_ladder as sl
from . import wake_array as wa
from .wake_array import DX, HALO, MACRO_DT, N, NU_REF, ROTOR_D, STRIDE

__all__ = [
    "REGIMES", "SeamChoice", "seam_menu", "regime_of", "shared_seams",
    "PROBE_STEPS", "PROBE_RUNGS", "build", "assert_projected",
    "ProbeStateRefused", "probe_state_health", "CertificateReading",
    "read_certificate", "spread", "travel_verdict", "FLOOR_MULTIPLE",
    "verdict_at", "BETA_MIN_AXIS",
]


# ---------------------------------------------------------------------------
# 1. the seam taxonomy -- a RULE over the layout, not a list
# ---------------------------------------------------------------------------

#: The four flow regimes this case study distinguishes, and what each means at a
#: seam.  They are derived from one number -- how many rotors of the seam's own
#: window-row lie upstream of it -- because `scaling_ladder.rotor_motif` places
#: every rotor of a row at the SAME ``y_centre``, so a row's rotors are in line
#: and a row's wake is the only wake crossing that row's faces.
REGIMES = {
    "near-freestream": "no rotor upstream in this row: the seam sees the "
                       "undisturbed stream plus whatever has diffused in",
    "shallow-wake": "one rotor upstream, 3.5 D away: a single wake that has not "
                    "yet recovered and has not yet met another",
    "deep-wake": "two or more in-line rotors upstream: the multi-turbine deficit "
                 "the array loss is made of",
    "bypass-clean": "the open flow beside the FIRST rotor of a row, whose own "
                    "rotor cells the port excludes: clean inflow, split face",
    "bypass-wake": "the open flow beside a rotor that already stands in a wake: "
                   "the same port declaration in a different flow",
}

#: The regime a `_full` seam is in, by upstream in-row rotor count.
_FULL_REGIME = {0: "near-freestream", 1: "shallow-wake"}


@dataclass(frozen=True)
class SeamChoice:
    """One seam, with everything that decides whether two of them are comparable.

    ``segment`` and ``dim_M`` are carried because **a spread across seams of
    different port declarations is not a measurement of the state**: a ``full``
    face is 128 cells and 33 interface modes, a ``bypass`` face is 96 cells and
    25, and the two operators do not live in the same space.  Every comparison
    this case study makes is WITHIN a ``segment`` group, and `seam_menu` groups
    them so a driver cannot accidentally cross the boundary.
    """

    seam_id: str
    regime: str
    segment: str
    row: int
    col: int
    n_cells: int
    dim_M: int
    upstream_rotors: int
    face_x_D: float
    band_y_D: tuple[float, float]

    @property
    def comparable_key(self) -> str:
        """Seams with the same key differ only in the flow they sit in."""
        return f"{self.segment}:{self.dim_M}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "seam_id": self.seam_id, "regime": self.regime,
            "segment": self.segment, "row": self.row, "col": self.col,
            "n_cells": self.n_cells, "dim_M": self.dim_M,
            "upstream_rotors": self.upstream_rotors,
            "face_x_D": self.face_x_D, "band_y_D": list(self.band_y_D),
            "comparable_key": self.comparable_key,
        }


def _parse_x_seam(seam_id: str) -> tuple[int, int, str] | None:
    """``x{col}r{row}_{segment}`` -> (col, row, segment); None for anything else.

    The y-seams and the rotor seams are deliberately not in the menu.  A y-seam
    pairs two ROWS and its normal component is ``v``, so it is a different
    operator wearing the same port type; a rotor seam is field-to-lumped and is
    blind at every admissible beta_min by the cell Reynolds number (**W97**), so
    it has no thresholds to move.
    """
    if not seam_id.startswith("x") or "_" not in seam_id:
        return None
    head, segment = seam_id.split("_", 1)
    if "r" not in head:
        return None
    col_s, row_s = head[1:].split("r", 1)
    if not (col_s.isdigit() and row_s.isdigit()):
        return None
    return int(col_s), int(row_s), segment


def regime_of(tiling: wa.ArrayTiling, seam_id: str) -> str | None:
    """The flow regime of one seam, from the rotor layout alone.

    No state is read.  That is the point: the label has to be assignable before
    the march runs, or it is a description of the answer rather than a factor of
    the design.
    """
    parsed = _parse_x_seam(seam_id)
    if parsed is None:
        return None
    col, row, segment = parsed
    upstream = sum(1 for r in tiling.rotors if r.row == row and r.col < col)
    if segment == "full":
        return _FULL_REGIME.get(upstream, "deep-wake")
    if segment == "bypass":
        return "bypass-clean" if upstream == 0 else "bypass-wake"
    return None


def seam_menu(tiling: wa.ArrayTiling) -> list[SeamChoice]:
    """Every fluid-fluid x-seam of a tiling, classified and measured.

    Ordered by ``(segment, upstream_rotors, seam_id)`` so the deep-wake pair a
    layout provides twice lands adjacent -- which is the replicate control: two
    seams the RULE calls the same regime, in different rows, whose disagreement
    is the label's own error bar.
    """
    out: list[SeamChoice] = []
    for conn in wa.connections(tiling):
        parsed = _parse_x_seam(conn.seam_id)
        if parsed is None:
            continue
        col, row, segment = parsed
        regime = regime_of(tiling, conn.seam_id)
        if regime is None:
            continue
        n_cells = N if segment == "full" else N - int(round(ROTOR_D / DX))
        # the xhi ring of the upstream window, which is where the trace is set
        face_x = (col * STRIDE + N - 0.5) * DX
        y_lo = row * STRIDE * DX
        out.append(SeamChoice(
            seam_id=conn.seam_id, regime=regime, segment=segment,
            row=row, col=col, n_cells=n_cells, dim_M=wa.modes_for(n_cells),
            upstream_rotors=sum(1 for r in tiling.rotors
                                if r.row == row and r.col < col),
            face_x_D=float(face_x), band_y_D=(float(y_lo), float(y_lo + N * DX)),
        ))
    return sorted(out, key=lambda s: (s.segment, s.upstream_rotors, s.seam_id))


def shared_seams(a: wa.ArrayTiling, b: wa.ArrayTiling) -> list[str]:
    """Seam IDs present in BOTH tilings -- the cross-geometry comparison set.

    A seam ID encodes a column, a row and a segment, so the same ID in two
    tilings is the same declared port on the same face of the same window at the
    same place in the layout.  What differs is how much graph is around it, and
    that is the geometry factor stated as a difference of one thing.
    """
    ids_b = {s.seam_id for s in seam_menu(b)}
    return [s.seam_id for s in seam_menu(a) if s.seam_id in ids_b]


# ---------------------------------------------------------------------------
# 2. the state schedule
# ---------------------------------------------------------------------------

#: Macro-steps of the trajectory at which the probe is taken.
#:
#: ``0`` is the freestream, which is control ZERO and is where the answer is
#: known.  ``110`` is inside the horizon Tier 19 marched the exposed classical
#: column over (120) and outside the 60 Tier 18 published at -- section 19.6's
#: rule that *a repair marched no further than the failure it repairs is not a
#: repair* applies to a probe state as much as to a scheme.  The three between
#: are spaced so the wake is still forming at 10 and settled by 60.
PROBE_STEPS = (0, 10, 30, 60, 110)

#: The rungs this case study probes, and why not the others.  ``N24`` is absent
#: for the reason `tier0-measurements` section 19.6 gives: it was never marched
#: at the long horizon, at 6.1 s per macro-step per classical column, and a
#: probe state has to come off a marched trajectory.  ``N1`` is absent because a
#: single window has no artificial face, so no port, so no seam and no
#: certificate -- `capability.ExpertCapabilities` raises before the compiler.
PROBE_RUNGS = ("N6", "N12")

#: Steps past which the divergence tail of the exposed+projected column is not
#: yet known to have settled -- Tier 19 measured the tail trend at 1.0441 at
#: N=12, still rising by 4% over the last thirty macro-steps.  A probe state
#: reached in more steps than this carries its ``div_rms`` in the record and is
#: compared against the same rung's freestream value rather than trusted.
SETTLED_STEPS = 60


# ---------------------------------------------------------------------------
# 3. the graph -- and the W105 assertion
# ---------------------------------------------------------------------------


class ProbeStateRefused(RuntimeError):
    """A probe state that is not admissible to probe at, and why."""


def assert_projected(graph: CaseGraph) -> ProjectedAssembly:
    """Refuse a graph whose assembly is not a `ProjectedAssembly` (**W105**).

    `R10` moves the elliptic part out of the agent and `R10b` fixes the cadence
    the composition layer must apply it at; neither checks that it is applied,
    and `R12` is silent on an all-exposed graph because `L6/C2`'s hypothesis
    genuinely fails there.  Measured, the gap is real: the exposed classical
    column with NO global projection runs all 120 macro-steps without leaving
    the band, at ``||div u||_rms = 1.25`` against the projected column's 0.066.
    Stable, and not incompressible.

    Every probe state in this case study is reached through the assembly this
    returns, so a graph that does not declare one is a graph whose trajectory
    this file cannot vouch for.
    """
    proj = graph.assembly_projection
    if proj is None:
        raise ProbeStateRefused(
            f"graph {graph.name!r} declares no constraint projection on its "
            "assembly. CS-8 probes only states reached through an "
            "assembly.ProjectedAssembly, because a graph can clear R10 and R10b "
            "and never apply the elliptic part at all (W105) -- and the column "
            "that does is stable and not incompressible")
    pou = graph.partition_of_unity
    if not isinstance(pou, ProjectedAssembly):
        raise ProbeStateRefused(
            f"graph {graph.name!r} carries a projection on a "
            f"{type(pou).__name__}, which is not the object assemble_conservative "
            "runs. The declaration and the step have to be the same object or "
            "the record can disagree with the driver, which is what W100 was")
    return pou


def build(u_full: np.ndarray, v_full: np.ndarray, rung: sl.LadderRung,
          kind: str = "reference_exposed", dt: float = MACRO_DT,
          nu: float = NU_REF, measured: MeasuredConstants | None = None,
          experts: dict[str, Any] | None = None) -> tuple[CaseGraph, dict]:
    """CS-7's graph at a CS-8 probe state, with the projection asserted.

    ``kind`` defaults to ``reference_exposed`` rather than ``reference``, which
    is the whole of what CS-8 changes about the classical column: the elliptic
    part is out of the agent, `EllipticSubsolve.EXPOSED` is on the record, and
    it is the first classical column in this vault `R10` does not refuse.  It is
    also the only classical column whose trajectory reaches macro-step 110.
    """
    graph, experts = sl.build(u_full, v_full, rung, kind=kind, dt=dt, nu=nu,
                              elliptic=EllipticSubsolve.UNKNOWN,
                              measured=measured, experts=experts,
                              assembly_projection=True)
    assert_projected(graph)
    graph.name = f"reuse-probe-{rung.label}-{kind}"
    return graph, experts


def probe_state_health(u: np.ndarray, v: np.ndarray, step: int,
                       tiling: wa.ArrayTiling, band: float = 3.0,
                       div_ref: float | None = None,
                       div_factor: float = 10.0) -> dict[str, Any]:
    """Is this state one a certificate read at it would mean anything?

    Three checks, and the third is the one Tier 19 asks for by name.  The state
    must be finite; ``|u|`` must be inside the band the whole ladder is judged
    by; and ``||div u||_rms`` must not have run away from the same column's own
    settled value -- because at N=12 the divergence tail was still rising by 4%
    at macro-step 120, so a state reached in more than `SETTLED_STEPS` steps is
    reported with its divergence rather than assumed to be on the attractor.

    Returns the record.  Raises `ProbeStateRefused` only for the two failures
    that make the reading meaningless; a rising divergence is FLAGGED, not
    refused, because the flag is a measurement and the refusal would hide it.
    """
    finite = bool(np.all(np.isfinite(u)) and np.all(np.isfinite(v)))
    if not finite:
        raise ProbeStateRefused(
            f"probe state at macro-step {step} is not finite: every certificate "
            "read at it would be a number about a blow-up")
    u_max = float(np.max(np.abs(u)))
    div = float(wa.divergence_rms(u, v))
    if u_max > band:
        raise ProbeStateRefused(
            f"probe state at macro-step {step} has |u|_max = {u_max:.4g}, outside "
            f"the band {band}. Tier 18 published a state on a diverging "
            "trajectory and section 19.6 published a repair on one; this is the "
            "check that says so before the probe rather than after")
    out = {"step": step, "u_max": u_max, "div_rms": div, "finite": True,
           "settled_horizon": step <= SETTLED_STEPS, "div_ref": div_ref,
           "div_ratio": None, "div_flag": False}
    if div_ref is not None and div_ref > 0:
        out["div_ratio"] = div / div_ref
        out["div_flag"] = bool(out["div_ratio"] > div_factor)
    return out


# ---------------------------------------------------------------------------
# 4. the certificate, reduced to what it discriminates on
# ---------------------------------------------------------------------------

#: A quantity has MOVED, rather than merely been re-measured, when its spread
#: across the factor exceeds this multiple of the reproducibility floor measured
#: on a re-probe of the identical pair.  Ten is a choice and it is declared here
#: rather than buried in a driver; every table reports the raw ratio beside the
#: verdict so a reader can move it.
FLOOR_MULTIPLE = 10.0


@dataclass
class CertificateReading:
    """One substitution certificate, and the five things it discriminates on.

    The swap is always the same one -- `reference.WindowNS` -> Poseidon-T at one
    fluid agent of one seam -- so anything that moves between two readings is
    the probe state, the seam, or the graph, and nothing else.
    """

    rung: str
    seam_id: str
    regime: str
    segment: str
    step: int
    target: str
    #: sigma_min of the ASSEMBLED reference seam operator: the inf-sup constant
    #: the whole plug-in bound divides by.
    beta: float | None = None
    #: ||S_i|| of the reference expert's own block -- the ``visible_above``
    #: threshold's other term.
    block_norm: float | None = None
    #: ||Lambda_expert - Lambda_ref||, `probe.substitution_delta`. The prompt's
    #: quantity, and the numerator of every plug-in degradation bound.
    delta_norm: float | None = None
    #: The composability index at the TARGET BLOCK, which is
    #: `probed-dtn-coupling` 4.5's own definition -- a per-agent quantity.
    Xi_block: float | None = None
    #: CS-7's seam-level ratio ||S_poseidon|| / ||S_reference||, carried so this
    #: tier's numbers can be set beside section 19.8's without a conversion.
    Xi_seam: float | None = None
    visible_above: float | None = None
    fails_above: float | None = None
    #: ``||Delta|| / beta`` -- the MARGIN by which the verdict is decided.
    #:
    #: Both thresholds are ``beta`` minus a norm, so on the admissible axis
    #: ``beta_min >= 0`` the verdict is `refuse` exactly when ``fails_above < 0``,
    #: which is exactly when this ratio exceeds 1.  It is therefore the one
    #: derived number that says how CLOSE the certificate came to answering
    #: differently -- and a ratio of 1.003 and a ratio of 1.26 are the same
    #: verdict with two very different amounts of room.
    delta_over_beta: float | None = None
    verdict: str | None = None
    passivity_preserved: bool | None = None
    same_port_list: bool | None = None
    #: the seam operator's own diagnostics, for context rather than comparison
    kappa_ref: float | None = None
    norm_ref: float | None = None
    norm_new: float | None = None
    one_sided_ref: float | None = None
    base_spread: float | None = None
    dim_M: int | None = None
    n_solves: int | None = None
    tau_total: float | None = None
    sigma: float | None = None
    eps_tol: float | None = None
    #: Why tau has no value here, when it has none.  `seam_defect_split` refuses
    #: a RELATIVE defect where no power crosses the reference interface -- L4's
    #: empty-seam case -- and a deep-wake seam can land there while every
    #: threshold beside it is well defined.  Recorded as a reason rather than as
    #: a zero, because zero is a measurement and this is not one.
    tau_undefined: str | None = None
    state: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    #: The fields a CS-8 comparison is made on.  Named here so the driver, the
    #: tests and the write-up cannot drift apart about what "the certificate's
    #: discriminating quantities" means.
    DISCRIMINATING = ("beta", "block_norm", "delta_norm", "Xi_block", "Xi_seam",
                      "visible_above", "fails_above", "delta_over_beta")

    def as_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items()}
        out["discriminating"] = {k: getattr(self, k)
                                 for k in self.DISCRIMINATING}
        return out


def read_certificate(rung_label: str, choice: SeamChoice, step: int,
                     target: str, op_ref: Any, op_new: Any, cert: Any,
                     state: dict[str, Any] | None = None) -> CertificateReading:
    """Reduce two probed seam operators and one certificate to a comparable row.

    ``op_ref`` and ``op_new`` are `probe.SeamOperator`s assembled at the SAME
    ``seam_base`` from the same graph topology with the two experts, and ``cert``
    is `composition.certify_substitution` on the target block of that pair.
    Nothing is recomputed here; this is the projection onto the axes the
    comparison is made on, and it exists so the projection is declared once.
    """
    nr = float(np.linalg.norm(op_ref.S, 2))
    nn = float(np.linalg.norm(op_new.S, 2))
    b_ref = op_ref.blocks[target]
    b_new = op_new.blocks[target]
    nb_ref = float(np.linalg.norm(b_ref.S, 2))
    nb_new = float(np.linalg.norm(b_new.S, 2))
    return CertificateReading(
        rung=rung_label, seam_id=choice.seam_id, regime=choice.regime,
        segment=choice.segment, step=step, target=target,
        beta=op_ref.beta, block_norm=nb_ref,
        delta_norm=cert.delta_norm,
        Xi_block=(nb_new / nb_ref) if nb_ref > 0 else None,
        Xi_seam=(nn / nr) if nr > 0 else None,
        visible_above=cert.visible_above, fails_above=cert.fails_above,
        delta_over_beta=((cert.delta_norm / op_ref.beta)
                         if op_ref.beta not in (None, 0) else None),
        verdict=str(getattr(cert.verdict, "value", cert.verdict)),
        passivity_preserved=cert.passivity_preserved,
        same_port_list=cert.same_port_list,
        kappa_ref=op_ref.kappa, norm_ref=nr, norm_new=nn,
        one_sided_ref=op_ref.one_sided, dim_M=op_ref.dim_M,
        base_spread=(None if op_ref.base_check is None
                     else op_ref.base_check.get("spread")),
        n_solves=sum(b.n_solves for b in op_ref.blocks.values()),
        state=dict(state or {}),
    )


# ---------------------------------------------------------------------------
# 5. how much did it move -- the travel statistic
# ---------------------------------------------------------------------------


def spread(values: Sequence[float | None]) -> dict[str, Any]:
    """Absolute and relative spread of one quantity over one factor.

    Both are reported and neither is called *the* spread.  ``range_over_mean`` is
    scale-free and is what a reader compares across quantities; ``abs_range`` is
    what gets compared with the reproducibility floor, which is an absolute
    number.  A quantity whose mean is near zero -- and ``visible_above`` is
    negative at every seam Tier 19 measured -- makes the relative form
    meaningless, so ``mean_abs`` is carried and the relative form is None when
    the mean is not usable.
    """
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "min": None, "max": None, "mean": None,
                "abs_range": None, "range_over_mean": None, "sd": None,
                "max_over_min": None}
    lo, hi = min(vals), max(vals)
    mean = float(np.mean(vals))
    out = {
        "n": len(vals), "min": lo, "max": hi, "mean": mean,
        "mean_abs": float(np.mean(np.abs(vals))),
        "abs_range": hi - lo,
        "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "range_over_mean": ((hi - lo) / abs(mean)) if abs(mean) > 0 else None,
        "max_over_min": (hi / lo) if lo > 0 and hi > 0 else None,
    }
    return out


def travel_verdict(factor_spread: dict[str, Any], floor: float | None,
                   multiple: float = FLOOR_MULTIPLE) -> dict[str, Any]:
    """Did the quantity TRAVEL, or is its spread the instrument?

    The comparison is absolute, between the factor's range and the range the
    same quantity shows when the identical pair is probed twice.  A quantity
    whose spread across the factor is inside the floor is a property of the
    expert as far as this measurement can see; one outside it by ``multiple`` is
    a property of the state.  Between the two is `undecided`, and it is reported
    as that rather than rounded to whichever side is convenient.
    """
    rng = factor_spread.get("abs_range")
    if rng is None or floor is None:
        return {"verdict": "unmeasured", "ratio_to_floor": None,
                "abs_range": rng, "floor": floor}
    if floor <= 0.0:
        # a deterministic expert re-probed identically CAN return exactly zero;
        # then any nonzero range is a movement and saying "infinite" is honest
        ratio = math.inf if rng > 0 else 0.0
    else:
        ratio = rng / floor
    verdict = ("travels" if ratio <= 1.0
               else "state-dependent" if ratio >= multiple
               else "undecided")
    return {"verdict": verdict, "ratio_to_floor": (None if math.isinf(ratio)
                                                   else ratio),
            "ratio_is_infinite": bool(math.isinf(ratio)),
            "abs_range": rng, "floor": floor, "multiple": multiple}


#: The admissible risk tolerances a verdict is asked at.  ``beta_min`` is a
#: NON-NEGATIVE risk parameter -- a tolerance below which an interface
#: perturbation is not worth seeing -- so the axis starts at zero, and Tier 19's
#: own sweep is a subset of this one.
BETA_MIN_AXIS = (0.0, 1e-12, 1e-6, 1e-3, 0.01, 0.1, 0.25, 0.5)


def verdict_at(visible_above: float | None, fails_above: float | None,
               beta_min: float) -> str | None:
    """The certificate's verdict at one ``beta_min``, from the thresholds alone.

    **W81 made the verdict function two numbers**, and this is that function
    written down: `blind` and `passes` are each monotone in ``beta_min``, so

        beta_min <= visible_above  ->  blind
        beta_min >  fails_above    ->  refuse
        otherwise                  ->  admit

    Nothing is re-probed.  That matters here for a reason beyond cost: it means
    the whole beta_min axis can be swept at every cell of the grid, so the
    question *does the VERDICT move with the probe state* is answered on the
    same footing as *do the NUMBERS move*, and not only at the one tolerance a
    driver happened to pass.
    """
    if visible_above is None or fails_above is None:
        return None
    if beta_min <= visible_above:
        return "blind"
    if beta_min > fails_above:
        return "refuse"
    return "admit"


def design_report(rungs: Sequence[sl.LadderRung]) -> dict[str, Any]:
    """The design, before a single solve -- the artifact's header."""
    return {
        "question": ("is a substitution certificate a property of the expert, "
                     "or of the state it was probed at?"),
        "regimes": dict(REGIMES),
        "probe_steps": list(PROBE_STEPS),
        "settled_steps": SETTLED_STEPS,
        "floor_multiple": FLOOR_MULTIPLE,
        "discriminating": list(CertificateReading.DISCRIMINATING),
        "held_fixed": {
            "dx_D": DX, "macro_dt": MACRO_DT, "overlap_cells": HALO,
            "ramp_cells": wa.RAMP, "nu_ref": NU_REF, "window_cells": N,
            "trajectory": "exposed_reference_solver + ProjectedAssembly (R10 + "
                          "R10b + R12), one march per rung, from the freestream",
            "seam_base": "the mean of the two sides' declared probe bases, "
                         "imposed on both experts",
            "swap": "reference.WindowNS -> Poseidon-T at one fluid agent",
        },
        "rungs": {r.label: {"n_windows": r.n_windows,
                            "n_overlaps": r.n_overlaps,
                            "seams": [s.as_dict() for s in seam_menu(r.tiling)]}
                  for r in rungs},
    }
