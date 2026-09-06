"""Conformance -- nothing checks that a declaration is true, until this runs.

plug-in-composition-theorems §3.  The architecture's central principle is that
everything it concludes about a composition is inherited from labels, never from
inspecting the box.  The corollary nobody wrote: **a false label produces a
guarantee that does not hold, with no error and no failed gate.**  For a single
hand-built case study that is tolerable, because the person who declared the
record built the expert.  For a plug-in architecture it is the foundational hole,
because the point of plug-in is that those are different people.

The result that makes conformance affordable: the three silent fields with the
largest consequences -- ``bc_channel``, ``storage``, ``equivariances`` -- are all
certified by machinery the probe already builds.  Conformance is not a new
subsystem; it is a report over the probe that is already budgeted.

And some fields admit no test at all.  ``validity`` is falsifiable but not
verifiable: certifying it requires knowing where the expert is wrong, which
requires the reference the predicate exists to make unnecessary.  So the
certificate records *"not falsified on suite X"*, never *"valid"*.

**The count, audited 2026-08-30 (W69).**  This paragraph used to end "the
architecture rests on exactly one unverifiable declaration".  W66 made it two by
adding ``response_half``, and the audit that followed found it had been **three**
the whole time -- ``governing_family`` is a free-form string, compared across a
seam by E3, read by nothing else, and it had no FieldTest at all.  The three are
not equivalent and the difference is the split rule:

    validity          false -> decertifies.      The correct verdict already.
    response_half     false -> L3/C9 refuses IF only one side is wrong; a
                      matching pair of wrong halves passes (W66's three legs).
    governing_family  false -> E3 reads `holds` and tau is attributed under a
                      hypothesis that does not hold.  It PROMOTES.

The third is the one with teeth, and naming it is all this module can do about
it.  What the audit *did* close is the adjacent set -- fields that were untested
but perfectly testable, which is a different failure and a fixable one:
``deterministic`` and ``stencil_radius`` now have tests, at one and two solves.

**Still three, 2026-08-29 (W75/W77), and one candidate was removed rather than
added.**  ``elliptic_subsolve`` joins the tested set: `_test_elliptic_subsolve`
measures the property the enum is *defined* by -- an infinite domain of
dependence -- by poking a delta, and it is the only test here that can RESOLVE an
``unknown`` rather than merely fail to contradict a declaration.  That is the
promotion W60 asked for and W68 refused to grant from the spectrum; the
difference is that reach measures the definition where kappa measures a correlate
that comes apart for a self-adjoint operator.

And ``probe_state`` was on its way to being a fourth.  It is a caller-supplied
string that nothing checked, and W74 proved it can be wrong while everything else
is right -- it read "duct, T_hot=900 K, T_wall=400 K" while the probe linearized
at 0 K.  It never needed declaring: the probe knows its base and the base IS the
state, so `SeamOperator.derived_probe_state` computes it and the count stays at
three.  **A declaration that can be derived is not an unverifiable field, it is
an unwritten derivation**, which is worth checking before adding one.

**Still three, 2026-08-29 (W87), and that check paid a second time.**  Tier 16
made ``lambda_ref`` load-bearing -- it decides whether tau is ``UNDEFINED`` at a
multiphysics seam -- and it was logged as a *fourth* declaration of the
unverifiable kind, on the grounds that it is a free-form string.  It is not.  The
string is not what the rule consumes; the rule consumes the claim that a
**reference pair converges**, and `multiphysics.tight_couple` settles that in one
solve.  `_test_lambda_ref` runs it.  So W77's lesson generalizes further than
"derive it": **a declaration that names an experiment is not unverifiable,
whatever its type is**, and the question to ask of a new field is what experiment
would falsify it rather than what it is made of.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .capability import (
    BCChannel,
    EllipticSubsolve,
    ExpertCapabilities,
    TimeDiscretization,
)
from .multiphysics import tight_couple
from .probe import (
    SUPPORT_GLOBAL_FRACTION,
    ProbeBudget,
    ProbedBlock,
    probe_block,
    support_reach,
)
from .transfer import InterfaceSpace, Prolongation
from .verdict import ADMIT, ADMIT_UNCERTIFIED, REFUSE, FailureClass, Verdict


@dataclass
class FieldTest:
    """One field's declared value against its measured one."""

    field_name: str
    declared: Any
    measured: Any
    residual: float | None
    verdict: Verdict
    silent_if_false: bool
    message: str
    cost: str = ""

    def as_dict(self) -> dict[str, Any]:
        from .verdict import _jsonable

        return {
            "field": self.field_name,
            "declared": _jsonable(self.declared),
            "measured": _jsonable(self.measured),
            "residual": self.residual,
            "verdict": self.verdict.value,
            "silent_if_false": self.silent_if_false,
            "cost": self.cost,
            "message": self.message,
        }


@dataclass
class ConformanceCertificate:
    """Bound to a weight hash and a probe state, because both matter.

    ``expert_id`` binds the certificate to a weight hash: a retrained expert is a
    different expert and inherits nothing, which is the whole reason plug-in
    needs a certificate rather than a habit.  ``probe_state`` records that every
    measured quantity is local to a state -- a certificate is valid at a state and
    a regime, never globally.
    """

    expert_id: str
    weight_hash: str
    tests: list[FieldTest] = field(default_factory=list)
    probe_state: str = "unspecified"
    suite: tuple[str, ...] = ()          # the states validity was NOT falsified on
    date: str = field(default_factory=lambda: _dt.date.today().isoformat())
    note: str = ""

    @property
    def verdict(self) -> Verdict:
        v = ADMIT
        for t in self.tests:
            v = v.worse_of(t.verdict)
        return v

    @property
    def contradictions(self) -> list[FieldTest]:
        """Fields whose measured value contradicts the declaration."""
        return [t for t in self.tests if t.verdict is REFUSE]

    def valid_at(self, regime: Any) -> bool:
        """A certificate quoted outside its probe regime is not a certificate."""
        return str(regime) == self.probe_state or self.probe_state == "unspecified"

    def as_dict(self) -> dict[str, Any]:
        return {
            "expert_id": self.expert_id,
            "weight_hash": self.weight_hash,
            "probe_state": self.probe_state,
            "suite": list(self.suite),
            "date": self.date,
            "verdict": self.verdict.value,
            "tests": [t.as_dict() for t in self.tests],
            "note": self.note,
        }

    def report(self) -> str:
        lines = [f"ConformanceCertificate {self.expert_id} [{self.weight_hash}] -> {self.verdict.value}"]
        for t in self.tests:
            flag = "silent" if t.silent_if_false else "loud  "
            lines.append(f"  {t.verdict.value:<18} {flag} {t.field_name}: {t.message}")
        return "\n".join(lines)


def run_conformance(
    caps: ExpertCapabilities,
    space: InterfaceSpace | None = None,
    prolongation: Prolongation | None = None,
    reference: ExpertCapabilities | None = None,
    budget: ProbeBudget | None = None,
    equivariance_check: Callable[[str], float] | None = None,
    validity_suite: Sequence[str] = (),
    validity_falsified_on: Sequence[str] = (),
    probe_state: str = "unspecified",
    block: ProbedBlock | None = None,
    seam_defect: float | None = None,
    reference_pair: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    reference_trace0: np.ndarray | None = None,
) -> ConformanceCertificate:
    """Certify a record against the expert it describes.

    Every consequential test rides on one probe.  If no probe can be run (no
    interface space, or no boundary response) the probe-borne fields come back
    ``admit-uncertified`` rather than silently passing.

    ``seam_defect`` is the passivity defect of the **assembled seam** the record's
    port participates in.  Supply it and `storage` is certified; omit it and
    `storage` decertifies with the block number as a diagnostic, because **the
    verdict belongs to the seam** (W48/W63) and a block can be non-passive while
    the seam it assembles into is passive.  It is a parameter rather than
    something this function computes because a seam involves two records and this
    function certifies one.

    ``reference_pair`` and ``reference_trace0`` are the same shape and there for
    the same reason: ``lambda_ref`` claims that a reference *pair* converges, and
    a pair is two records.  Supply them and the claim is run; omit them and it
    decertifies, because a load-bearing declaration whose experiment was never
    run is not the same as one that passed (W87).
    """
    cert = ConformanceCertificate(
        expert_id=caps.expert_id,
        weight_hash=caps.weight_hash or "unhashed",
        probe_state=probe_state,
        suite=tuple(validity_suite),
    )

    cert.tests.append(_test_ports(caps))
    cert.tests.append(_test_nondim(caps))

    if block is None and space is not None and prolongation is not None and caps.boundary_response:
        block = probe_block(
            caps, caps.ports[0], space, prolongation, space.seam_id, budget, reference, probe_state
        )

    cert.tests.append(_test_bc_channel(caps, block))
    cert.tests.append(_test_deterministic(caps, space, prolongation))
    cert.tests.append(_test_support_radius(caps, space, prolongation))
    cert.tests.append(_test_elliptic_subsolve(caps, space, prolongation))
    cert.tests.append(_test_governing_family(caps))
    cert.tests.append(_test_lambda_ref(caps, reference_pair, reference_trace0))
    cert.tests.append(_test_storage(caps, block, seam_defect))
    cert.tests.append(_test_equivariances(caps, equivariance_check))
    cert.tests.append(_test_differentiable(caps, space, prolongation))
    cert.tests.append(_test_validity(caps, validity_suite, validity_falsified_on))
    return cert


# ---------------------------------------------------------------------------


def _test_ports(caps: ExpertCapabilities) -> FieldTest:
    """Structural, free, and loud on failure."""
    missing = caps.missing_fields()
    if missing:
        return FieldTest(
            "ports/schema", "complete record", f"missing {missing}", None, REFUSE, False,
            f"record is missing fields a downstream layer needs: {missing}", "free",
        )
    return FieldTest(
        "ports/schema", "complete record", "complete", 0.0, ADMIT, False,
        "schema check against the port algebra passes", "free",
    )


def _test_nondim(caps: ExpertCapabilities) -> FieldTest:
    """Completeness is decidable from the port list; a wrong value is not attributable."""
    problems = []
    worst = 0.0
    for p in caps.ports:
        chk = p.scale_check()
        if not chk.complete:
            problems.append(f"{p.name}: missing {list(chk.missing)}")
        elif not chk.ok:
            problems.append(f"{p.name}: {chk.detail}")
        if chk.power_identity_residual is not None:
            worst = max(worst, chk.power_identity_residual)
    if problems:
        return FieldTest(
            "nondim", "complete and power-preserving", "; ".join(problems), worst, REFUSE, True,
            "; ".join(problems), "free",
        )
    return FieldTest(
        "nondim", "complete and power-preserving", "complete", worst, ADMIT, True,
        "scale sets are complete per port type and satisfy s_e * s_f = s_P. A wrong "
        "VALUE would show in the power residual but is not attributable, so this test "
        "is only partly a conformance test",
        "free",
    )


def _test_bc_channel(caps: ExpertCapabilities, block: ProbedBlock | None) -> FieldTest:
    """Impose a non-constant trace and check the response exceeds the probe floor.

    This is a failed test the vault has already run by accident: a frozen
    checkpoint scoring Xi = 0 IS a failed bc_channel conformance test, discovered
    late, expensively, and framed as a finding about a checkpoint.
    """
    declared = caps.bc_channel.value
    if block is None:
        return FieldTest(
            "bc_channel", declared, None, None, ADMIT_UNCERTIFIED, True,
            "no probe available, so the declaration is unverified. This is the field "
            "one probe certifies and it is the one that failed silently before",
            "the probe itself",
        )
    responds = not block.is_empty
    if caps.bc_channel is BCChannel.NONE:
        if responds:
            return FieldTest(
                "bc_channel", declared, "responds", block.Xi, REFUSE, True,
                "declares no boundary channel but the probe measures a nonzero response; "
                "the declaration and the expert disagree", "the probe itself",
            )
        return FieldTest(
            "bc_channel", declared, "no response (Xi = 0)", 0.0, ADMIT, True,
            "declaration confirmed: Lambda == 0, so the interface problem is empty rather "
            "than ill-conditioned. Correct as a declaration and disqualifying as a plug",
            "the probe itself",
        )
    if not responds:
        return FieldTest(
            "bc_channel", declared, "no response (Xi = 0)", 0.0, REFUSE, True,
            f"declares bc_channel {declared} but the probe measures Xi = 0: the agent's "
            "output does not depend on imposed boundary data. The interface problem is "
            "empty, every trace is equally consistent, and the word 'coupled' is refused "
            "on the output", "the probe itself",
        )
    return FieldTest(
        "bc_channel", declared, f"responds (Xi = {block.Xi})", block.Xi, ADMIT, True,
        "the probe measures a response above the floor", "the probe itself",
    )


def _trace_length(
    caps: ExpertCapabilities,
    space: InterfaceSpace | None,
    prolongation: Prolongation | None,
) -> int | None:
    """How long a trace this expert's first port accepts.

    ``InterfaceSpace.dim`` is M's dimension and a trace lives in V, so the length
    comes from the prolongation -- the same route `probe_block` takes, rather
    than a second one that could disagree with it.
    """
    port = caps.ports[0] if caps.ports else None
    if port is None:
        return None
    sp = port.interface_space or space
    pr = port.prolongation or prolongation
    if pr is not None and getattr(pr, "matrix", None) is not None:
        # A port may declare a prolongation and leave the space to the seam --
        # `thermal_seam` does exactly that -- and the matrix's row count IS V.
        return int(np.asarray(pr.matrix).shape[0])
    if sp is None or pr is None:
        return None
    try:
        return int(np.asarray(pr.prolong(np.zeros(sp.dim), sp)).size)
    except Exception:                                                 # noqa: BLE001
        return None


def _test_deterministic(
    caps: ExpertCapabilities,
    space: InterfaceSpace | None = None,
    prolongation: Prolongation | None = None,
) -> FieldTest:
    """Two calls on one trace. **W69, 2026-08-30** -- one solve, never run before.

    ``deterministic`` sets the probe's own finite-difference step through
    ``reproducibility_floor``, so a false value corrupts every block on the seam
    and does it silently: the columns come back with noise in them and nothing
    downstream can tell that from physics.  Every record in the vault declares
    ``True`` and none of them had ever been asked.
    """
    declared = caps.deterministic
    if caps.boundary_response is None or not caps.ports:
        return FieldTest(
            "deterministic", declared, None, None, ADMIT_UNCERTIFIED, True,
            "no boundary response, so the repeat cannot be run", "one solve",
        )
    port = caps.ports[0]
    n = _trace_length(caps, space, prolongation)
    if n is None:
        return FieldTest(
            "deterministic", declared, None, None, ADMIT_UNCERTIFIED, True,
            "no interface space and prolongation pair reaches this record, so no trace "
            "of the right length can be built", "one solve",
        )
    try:
        trace = np.zeros(n)
        a = np.asarray(caps.boundary_response(port.name, trace), dtype=float)
        b = np.asarray(caps.boundary_response(port.name, trace), dtype=float)
    except Exception as exc:                                          # noqa: BLE001
        return FieldTest(
            "deterministic", declared, f"call failed: {type(exc).__name__}", None,
            ADMIT_UNCERTIFIED, True,
            f"the repeat could not be run: {exc}", "one solve",
        )
    scale = float(np.linalg.norm(a))
    rel = float(np.linalg.norm(a - b) / scale) if scale > 0 else 0.0
    if declared and rel > caps.reproducibility_floor:
        return FieldTest(
            "deterministic", True, f"repeat differs by {rel:.3e}", rel, REFUSE, True,
            f"declares deterministic but two identical calls differ by {rel:.3e}, above "
            f"the declared reproducibility floor {caps.reproducibility_floor:.3e}. The "
            "floor sets the probe's finite-difference step, so this contaminates every "
            "column of every block on every seam this expert joins", "one solve",
        )
    if not declared and rel == 0.0:
        return FieldTest(
            "deterministic", False, "repeat is bit-identical", 0.0,
            ADMIT_UNCERTIFIED, False,
            "declares non-deterministic and the repeat is bit-identical. Not a "
            "contradiction -- one trace is not the state space -- but the conservative "
            "declaration is costing a regression probe route for nothing visible here",
            "one solve",
        )
    return FieldTest(
        "deterministic", declared, f"repeat differs by {rel:.3e}", rel, ADMIT, True,
        f"two identical calls agree to {rel:.3e}, at or below the declared floor",
        "one solve",
    )


def _test_support_radius(
    caps: ExpertCapabilities,
    space: InterfaceSpace | None = None,
    prolongation: Prolongation | None = None,
) -> FieldTest:
    """A delta on the seam: how far does the response reach?  **W69, 2026-08-30.**

    ``stencil_radius`` and ``substeps_per_macro_step`` multiply into
    ``required_halo()``, which `_halo_rule` compares against the declared overlap
    and **refuses** below.  A too-small radius therefore buys a passing halo
    check for a contaminated blend -- the silent-wrongness class, on two fields
    that no test had ever touched.

    Poking a delta at one interface node and measuring the along-seam spread of
    the response is a **lower bound** on the reach: influence that travels along
    the seam had to travel through the stencil.  So this can falsify an
    under-declaration and can never verify one, which is `validity`'s shape and
    is stated that way rather than dressed up.

    Measured 2026-08-30: `thermal_seam`'s gas block spreads **0** cells -- the
    response is exactly diagonal, bit-zero off the poked cell -- and its shell
    spreads **5**, an exponential tail from 0.471 at the pole to 1.6e-3 one cell
    out.  The shell declares ``stencil_radius=1, substeps=1``, so its
    ``required_halo()`` is 1 against a measured 5, and that is not a wrong
    constant: see `ExpertCapabilities.required_halo`, where the product is an
    *explicit* agent's domain of dependence and the shell is implicit.
    """
    declared = caps.required_halo()
    if caps.boundary_response is None or not caps.ports:
        return FieldTest(
            "stencil_radius", declared, None, None, ADMIT_UNCERTIFIED, True,
            "no boundary response, so the delta cannot be imposed", "two solves",
        )
    port = caps.ports[0]
    n = _trace_length(caps, space, prolongation)
    if n is None:
        return FieldTest(
            "stencil_radius", declared, None, None, ADMIT_UNCERTIFIED, True,
            "no interface space and prolongation pair reaches this record, so no delta "
            "of the right length can be built", "two solves",
        )
    try:
        base = np.zeros(n)
        f0 = np.asarray(caps.boundary_response(port.name, base), dtype=float).ravel()
        d = np.zeros(n)
        j0 = n // 2
        d[j0] = 1.0
        f1 = np.asarray(caps.boundary_response(port.name, base + d), dtype=float).ravel()
    except Exception as exc:                                          # noqa: BLE001
        return FieldTest(
            "stencil_radius", declared, f"call failed: {type(exc).__name__}", None,
            ADMIT_UNCERTIFIED, True, f"the delta could not be imposed: {exc}",
            "two solves",
        )
    g = np.abs(f1 - f0)
    peak = float(g.max()) if g.size else 0.0
    if peak <= 0.0:
        return FieldTest(
            "stencil_radius", declared, "no response to a delta", None,
            ADMIT_UNCERTIFIED, True,
            "a delta on the seam moves nothing, so the reach is unmeasurable here. "
            "That is bc_channel's finding, not this one's", "two solves",
        )
    above = np.nonzero(g > 1e-6 * peak)[0]
    spread = int(np.max(np.abs(above - min(j0, g.size - 1)))) if above.size else 0
    if declared is None:
        implicit = (caps.time_discretization is TimeDiscretization.IMPLICIT
                    and int(caps.stencil_radius) >= 1)
        if implicit:
            product = (int(caps.stencil_radius) * int(caps.substeps_per_macro_step)
                       if caps.substeps_per_macro_step is not None else None)
            return FieldTest(
                "stencil_radius", f"radius {caps.stencil_radius} (implicit)", spread,
                None, ADMIT_UNCERTIFIED, True,
                f"the response reaches {spread} cells along the seam. The record is "
                f"implicit with a nonzero stencil, so required_halo() is None by "
                f"derivation (W69) and there is no declared number to falsify -- one "
                f"macro-step inverts an operator coupling every cell to every other. "
                + (f"Reading stencil_radius x substeps as a halo would have given "
                   f"{product} against the {spread} measured here. " if product is not None
                   else "")
                + "The measured spread is the diagnostic, not a verdict",
                "two solves",
            )
        return FieldTest(
            "stencil_radius", None, spread, None, ADMIT_UNCERTIFIED, True,
            f"the response reaches {spread} cells along the seam and the record declares "
            "no substeps_per_macro_step, so there is nothing to compare it against",
            "two solves",
        )
    if spread > declared:
        return FieldTest(
            "stencil_radius", declared, spread, float(spread - declared), REFUSE, True,
            f"a delta at one seam node moves the response {spread} cells away, and the "
            f"record's stencil_radius x substeps is {declared}. Along-seam travel is a "
            "LOWER bound on the stencil's reach, so the declaration is falsified. "
            "required_halo() feeds L2/R10/halo, which REFUSES an overlap below it -- so "
            "an under-declaration here buys a passing halo check for a contaminated "
            "blend", "two solves",
        )
    return FieldTest(
        "stencil_radius", declared, spread, None, ADMIT, True,
        f"a delta reaches {spread} cells along the seam, within the declared "
        f"{declared}. Not verified: along-seam spread bounds the reach from below "
        "only, so this can falsify an under-declaration and never confirm one",
        "two solves",
    )


def _test_elliptic_subsolve(
    caps: ExpertCapabilities,
    space: InterfaceSpace | None = None,
    prolongation: Prolongation | None = None,
) -> FieldTest:
    """**W75, closed 2026-08-29.**  The field W60 could not declare, measured.

    `EllipticSubsolve` is defined as *"whether the agent's step contains a solve
    with an INFINITE domain of dependence"*, and that is a statement about
    SUPPORT.  Every attempt to settle it so far went through the spectrum --
    `poseidon.elliptic_signature` -- and W68 had to give that statistic a
    precondition because a film-dominated block has no operator in it to look
    at.  W75 asked whether clearing the precondition would let it decide the
    field.  **It does not.**  On the same shell under two solvers, once ``omega``
    is above the floor:

        dt = 1     implicit kappa 1.0214   explicit kappa 1.0210
        dt = 10    implicit kappa 1.2311   explicit kappa 1.2649
        dt = 100   implicit kappa 3.1966   explicit kappa 9.6688

    indistinguishable at the first cadence and **ranked backwards** at the last,
    with the asymmetry flat at 5e-6 throughout.

    The reason is the probe basis, not the statistic: a Fourier mode is global by
    construction, so a block assembled from smooth modes cannot report whether
    the operator behind it was local.  **Poke a delta instead** and the answer is
    immediate and needs no threshold on any spectral quantity -- one implicit
    macro-step inverts ``(M/dt + K)``, and the inverse of a sparse SPD matrix is
    dense, so every seam cell responds.  An explicit march's response is exactly
    zero past its domain of dependence.  Five known points, two of them one shell
    under two solvers with all else fixed, separate 0.986 from 0.210.

    It costs **two solves**, the same two `_test_support_radius` already spends,
    and it is the cheapest gate in the framework.  What it cannot do is split
    ``exposed`` from ``none`` -- both have taken the global solve out of the
    agent's own step, and that is all the reach can see.  Neither R10 nor L2
    needs that split, so the measurement decides exactly what the rules consume.
    """
    declared = caps.elliptic_subsolve
    unknown = declared is EllipticSubsolve.UNKNOWN
    if caps.boundary_response is None or not caps.ports:
        return FieldTest(
            "elliptic_subsolve", declared.value, None, None, ADMIT_UNCERTIFIED, True,
            "no boundary response, so no delta can be imposed and the reach is "
            "unmeasurable", "two solves",
        )
    n = _trace_length(caps, space, prolongation)
    if n is None:
        return FieldTest(
            "elliptic_subsolve", declared.value, None, None, ADMIT_UNCERTIFIED, True,
            "no interface space and prolongation pair reaches this record", "two solves",
        )
    try:
        reach = support_reach(caps.boundary_response, caps.ports[0].name, np.zeros(n))
    except Exception as exc:                                          # noqa: BLE001
        return FieldTest(
            "elliptic_subsolve", declared.value, f"call failed: {type(exc).__name__}",
            None, ADMIT_UNCERTIFIED, True, f"the delta could not be imposed: {exc}",
            "two solves",
        )
    if reach.peak <= 0.0:
        return FieldTest(
            "elliptic_subsolve", declared.value, "no response to a delta", None,
            ADMIT_UNCERTIFIED, True,
            "a delta on the seam moves nothing, so the reach is unmeasurable here",
            "two solves",
        )
    frac = reach.fraction
    measured = f"{reach.nonzero}/{reach.n} cells respond (fraction {frac:.3f})"
    consistent = reach.consistent_with

    if unknown:
        # W60 asked for exactly this and W68 refused to supply it from the
        # spectrum. A direct measurement of the DEFINING property is a different
        # thing from a correlate, so it may resolve the field -- and it is
        # reported, never written back into the record, because promotion by
        # side effect is what the three-verdict split exists to prevent.
        return FieldTest(
            "elliptic_subsolve", "unknown", measured, None, ADMIT, False,
            f"declared unknown and the reach RESOLVES it: the response is "
            f"{'global' if reach.is_global else 'not shown global'}, consistent with "
            f"{' or '.join(consistent)}. This is the promotion W60 asked for and "
            f"W68 refused to grant from the spectrum -- the difference is that "
            f"reach measures the property the enum is DEFINED by rather than a "
            f"correlate of it. Reported, not written back: nothing here edits the "
            f"record", "two solves",
        )
    if reach.is_global and declared is not EllipticSubsolve.EMBEDDED:
        return FieldTest(
            "elliptic_subsolve", declared.value, measured, float(frac), REFUSE, True,
            f"the record declares {declared.value!r} and the response to a delta at one "
            f"seam cell reaches {reach.nonzero} of {reach.n} cells -- an infinite domain "
            f"of dependence, which is what EMBEDDED means. 'none' switches R10 OFF and "
            f"'exposed' tells L2 the elliptic part is the composition layer's to run, so "
            f"either way a decomposition silently changes the operator and the resulting "
            f"error is elliptic: it does not decay with distance from the cut and no halo "
            f"removes it", "two solves",
        )
    if not reach.is_global and declared is EllipticSubsolve.EMBEDDED:
        return FieldTest(
            "elliptic_subsolve", "embedded", measured, float(frac), ADMIT_UNCERTIFIED,
            False,
            f"the record declares 'embedded' and the reach does not establish globality "
            f"({measured}). This is NOT a measurement of compactness: the fraction is a "
            f"LOWER bound on the reach, because a genuinely dense response whose far "
            f"tail underflows to zero is indistinguishable from a compact one. Measured "
            f"here on the same implicit shell under two clocks -- 48/48 at dt = 0.05 s "
            f"and 42/48 at dt = 1e-4 s, where sqrt(alpha dt) is 60x below one seam cell "
            f"and exp(-d/ell) underflows past about 21 cells. So the gate has an "
            f"operating range in dt, bounded below by underflow and above by an explicit "
            f"competitor filling the seam anyway (0.81 at dt = 1 s), and it is only the "
            f"GLOBAL reading that is a positive measurement. Left uncertified rather "
            f"than contradicted for exactly that reason, and it is the conservative "
            f"direction regardless: L2 refuses on 'embedded'", "two solves",
        )
    return FieldTest(
        "elliptic_subsolve", declared.value, measured, float(frac), ADMIT, True,
        f"the reach corroborates {declared.value!r}: the response is "
        f"{'global' if reach.is_global else 'not global'} ({measured}, floor "
        f"{SUPPORT_GLOBAL_FRACTION:g}), consistent with {' or '.join(consistent)}. "
        f"Only the GLOBAL reading is a positive measurement; the other direction is a "
        f"lower bound, since a dense tail can underflow",
        "two solves",
    )


def _test_governing_family(caps: ExpertCapabilities) -> FieldTest:
    """**W69, 2026-08-30.**  The third unverifiable declaration, and the one with teeth.

    ``governing_family`` is a free-form string.  E3 compares it across a seam and
    nothing else in the architecture reads it, so a *false* value is invisible
    and a *matching pair* of false values makes E3 read ``holds`` -- which is the
    hypothesis tau is attributed under.  ``validity`` and ``response_half`` are
    unverifiable too, but a false ``validity`` decertifies and a false
    ``response_half`` disagrees at L3/C9 whenever only one side is wrong.  This
    one silently promotes.

    Nothing here can test it.  What this entry does is stop the gap being
    invisible: an untested field with no FieldTest looks identical, in the
    certificate, to a field nobody thought about.
    """
    declared = caps.governing_family
    if declared is None:
        return FieldTest(
            "governing_family", None, "not declared", None, ADMIT_UNCERTIFIED, True,
            "governing_family undeclared, so E3 cannot be stamped and tau is UNDEFINED "
            "at every seam this expert joins", "none exists",
        )
    return FieldTest(
        "governing_family", declared, "no test exists", None, ADMIT_UNCERTIFIED, True,
        f"declared {declared!r} and unverified. The string is compared across a seam by "
        "E3 and read by nothing else, so a false value is invisible and two agents "
        "sharing one false value make E3 read 'holds' -- the hypothesis tau is "
        "attributed under. This is the architecture's THIRD unverifiable declaration "
        "after validity and response_half, and the only one whose falsity promotes "
        "rather than decertifies", "none exists",
    )


def _test_lambda_ref(
    caps: ExpertCapabilities,
    reference_pair: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    reference_trace0: np.ndarray | None = None,
) -> FieldTest:
    """Build the referent, or fail to.  **W87, 2026-08-29.**

    ``lambda_ref`` became load-bearing in Tier 16 and had no test.  It names the
    reference this agent is scored against, and since W88 it is what decides
    whether tau is ``UNDEFINED`` at a seam whose two sides declare different
    governing families: declared on both sides, `L1/E3` **admits** and reports a
    measured tau; absent, tau is emitted as ``UNDEFINED``.  So a *false*
    declaration promotes -- ``governing_family``'s shape, which is the class with
    teeth -- and reports tau as measurable at a seam where no referent exists.

    The reason it looked unverifiable is that it is a *string*, and no string can
    be checked.  But the string is not what the rule consumes.  What the rule
    consumes is the **claim that a reference pair can be converged**, and that is
    an experiment: `multiphysics.tight_couple` either converges or it does not,
    and `seam_defect_split` already refuses on exactly that outcome.  So this is
    the same move W77 made on ``probe_state`` -- **a declaration that names an
    experiment is not an unverifiable field, it is an unrun experiment** -- and
    the count of unverifiable declarations stays at three rather than becoming
    four.

    **What it can and cannot do, measured rather than asserted.**  Run on
    `thermal_seam` against four deliberately wrong reference pairs, **four of the
    five converge**:

        the real pair          admit    lambda* = 372.1377 K
        shell sign-flipped     admit    lambda* = 425.6904 K
        shell blind            admit    lambda* = 900.0067 K
        gas blind              admit    lambda* = 374.1082 K
        both blind             REFUSE   residual 2.317e+02 after 50 iterations

    A monotone residual has a root whichever way its two halves lean, so a
    reference that ignores its boundary datum -- and even one on the wrong side
    of its own bond -- still balances, at a different trace.  What converging
    falsifies is therefore exactly one claim: **that the interface problem is not
    empty.**  That is `CASE-STUDY-GUIDE` mistake 6 arriving through a declaration,
    and it is the claim `L1/E3` actually promotes on.

    The admissible interval is not a second discriminator, and that was measured
    too rather than assumed: the one wrong pair it separates (`shell blind`)
    lands $0.0067$ K outside a $650$ K reservoir span, a relative margin of
    $1.0\\times10^{-5}$, because a constant shell flux is balanced only where the
    gas transmits that constant -- which is the gas's own inlet temperature.  The
    sign-flipped pair is not separated at all.

    **The cost of the miss, and the one reassuring thing in it.**  Scoring the
    *real* pair against each wrong referent, where the true tau is zero:

        reference = shell sign-flipped     tau_shell = 2.0000
        reference = shell blind            tau_shell = 1.3179e+08
        reference = gas blind              tau_gas   = 7.3161e-02

    every defect lands on the agent whose referent was corrupted and the other
    side stays identically zero.  So a false ``lambda_ref`` corrupts the
    *magnitude* of an attribution and not its *localization* -- which is worth
    knowing, and is not a reason to trust the magnitude.

    The pair is a parameter for `_test_storage`'s reason: a referent involves two
    records and this function certifies one.
    """
    declared = caps.lambda_ref
    cost = "one tight_couple solve on the reference pair"
    if declared is None:
        return FieldTest(
            "lambda_ref", None, "not declared", None, ADMIT_UNCERTIFIED, True,
            "no lambda_ref, so at a seam whose two sides declare different governing "
            "families tau is emitted as UNDEFINED for both. That is the conservative "
            "reading and there is nothing here to falsify -- the failure mode this test "
            "exists for is a declaration that is present and false", cost,
        )
    if reference_pair is None or reference_trace0 is None:
        return FieldTest(
            "lambda_ref", declared, None, None, ADMIT_UNCERTIFIED, True,
            f"declares the reference {declared!r} and the pair was not supplied, so the "
            "referent was not built. The declaration gates whether tau is UNDEFINED at "
            "a multiphysics seam (L1/E3) and a false one PROMOTES, so an unrun test "
            "here is not the same as a passing one", cost,
        )
    if caps.expert_id not in reference_pair:
        return FieldTest(
            "lambda_ref", declared, f"pair names {sorted(reference_pair)}", None,
            ADMIT_UNCERTIFIED, True,
            f"the supplied reference pair names {sorted(reference_pair)} and this record "
            f"is {caps.expert_id!r}, so the pair is not this agent's referent and the "
            "test was not run", cost,
        )

    def residual(lam):
        return sum(
            np.asarray(reference_pair[a](lam), dtype=float).ravel()
            for a in sorted(reference_pair)
        )

    try:
        tc = tight_couple(residual, np.asarray(reference_trace0, dtype=float).ravel())
    except Exception as exc:                                          # noqa: BLE001
        return FieldTest(
            "lambda_ref", declared, f"solve failed: {type(exc).__name__}", None,
            ADMIT_UNCERTIFIED, True,
            f"the reference pair could not be run: {exc}", cost,
        )
    if not tc.converged:
        return FieldTest(
            "lambda_ref", declared, f"did not converge in {tc.iterations} iterations",
            tc.residual_norm, REFUSE, True,
            f"declares the reference {declared!r}, and converging that pair leaves a "
            f"residual of {tc.residual_norm:.4e} after {tc.iterations} iterations. **No "
            "interface state balances the bond**, so there is no reference trajectory and "
            "tau has no referent -- and the declaration is what makes L1/E3 report tau as "
            "MEASURED at this seam rather than UNDEFINED. That is the silent-wrongness "
            "class: the compile promotes on a declaration whose own experiment fails",
            cost,
        )
    return FieldTest(
        "lambda_ref", declared, f"converged in {tc.iterations} iterations",
        tc.residual_norm, ADMIT, True,
        f"the declared reference pair converges to a residual of {tc.residual_norm:.4e} "
        f"in {tc.iterations} iterations, so a reference trajectory exists and tau is "
        "measurable against it. **Narrower than it sounds**: measured on thermal_seam, "
        "four of five deliberately wrong pairs converge too -- a monotone residual has a "
        "root whichever way its halves lean, so a blind or sign-flipped reference "
        "balances at a different trace. What this falsifies is that the interface "
        "problem is EMPTY, which is the claim L1/E3 promotes on, and nothing more. A "
        f"referent that converges to the wrong root still corrupts tau's magnitude (2.0 "
        "for a sign flip) while leaving its localization correct", cost,
    )


def _test_storage(
    caps: ExpertCapabilities,
    block: ProbedBlock | None,
    seam_defect: float | None = None,
) -> FieldTest:
    """Passivity is positive-realness of the probed matrix: one eigenvalue.

    And this is the property that matters most for plug-in, not only for accuracy:
    passivity is the only property in the framework that survives an expert swap
    without re-certification, because L <= 1 follows from a per-part property plus
    the interconnection structure and references no global quantity.

    **W63, and the first run of this suite is what found it.** W48 was closed on
    2026-08-28 with the statement *"the conformance suite certifies on the seam
    and reports blocks as diagnostics"*, and this function refused on the
    **block**.  The distinction is not academic and the vault had already measured
    it twice: §2.3 found an assembled seam at ``kappa = 19`` whose own block was
    ``2260``, and §8.5 found ``1.196`` against ``7061`` -- *"any per-block gate
    would have condemned a seam that is in fact well conditioned"*.

    Then the suite ran, and the first thing it did was that.  **Poseidon-T's
    ``xhi`` block measures a passivity defect of 3.218e-03 while its assembled
    seam measures mu = +1.534e-03 and pi = 0** -- so the record was refused for
    not being passive by a suite whose own theory page says the verdict belongs
    to the seam.

    So: ``seam_defect`` supplied -> the verdict is the seam's and the block is a
    diagnostic beside it.  ``seam_defect`` absent -> the block number is reported
    and the field **decertifies**, because a block verdict is not the seam's and
    refusing on one is the error W48 named.
    """
    declared = caps.declares_storage
    if block is None:
        return FieldTest(
            "storage", declared, None, None, ADMIT_UNCERTIFIED, True,
            "no probe available; the passivity certificate is an eigenvalue of a matrix "
            "the probe already assembles", "eigenvalue of the probed matrix",
        )
    pi = block.passivity_defect
    if not declared:
        return FieldTest(
            "storage", False, f"passivity defect {pi}", pi, ADMIT_UNCERTIFIED, True,
            "no storage function declared, so the structural route to L <= 1 is "
            "unavailable and L must be fitted. The defect is measured anyway, and its "
            "eigenvector names which interface mode is amplified",
            "eigenvalue of the probed matrix",
        )

    if seam_defect is None:
        detail = "unmeasured" if pi is None else f"{pi:.3e}"
        return FieldTest(
            "storage", True, f"block passivity defect {detail}", pi,
            ADMIT_UNCERTIFIED, True,
            f"the block's passivity defect is {detail} and **the verdict belongs to the "
            "seam** (W48): a block can be non-passive while the assembled seam is "
            "passive, measured at kappa 1.196 against 7061 on the same seam (§8.5) and "
            "at a block defect of 3.218e-03 against a seam defect of 0 on Poseidon-T. "
            "Pass `seam_defect` to certify. The block number is a diagnostic",
            "eigenvalue of the probed matrix",
        )

    if seam_defect > 0.0:
        return FieldTest(
            "storage", True, f"seam passivity defect {seam_defect:.3e}", seam_defect,
            REFUSE, True,
            f"declares a storage function and the ASSEMBLED SEAM's symmetric part has a "
            f"negative mode (defect {seam_defect:.3e}); the composition is not "
            f"incrementally passive and the declared certificate is false. The block's "
            f"own defect is {pi!r}, reported as a diagnostic",
            "eigenvalue of the probed matrix",
        )
    return FieldTest(
        "storage", True, f"seam positive real (block defect {pi!r})", seam_defect,
        ADMIT, True,
        "the symmetric part of the ASSEMBLED SEAM is positive semidefinite: incremental "
        "passivity confirmed, and it survives any substitution of a passive expert for a "
        "passive expert with no re-measurement. The block's own defect is reported "
        "beside it and is not the verdict (W48)", "eigenvalue of the probed matrix",
    )


def _test_equivariances(
    caps: ExpertCapabilities,
    check: Callable[[str], float] | None,
) -> FieldTest:
    """One extra solve per generator. Also already failed once, the slow way."""
    if not caps.equivariances:
        return FieldTest(
            "equivariances", (), "none declared", None, ADMIT, True,
            "no equivariance declared, so none is claimed", "free",
        )
    if check is None:
        return FieldTest(
            "equivariances", list(caps.equivariances), None, None, ADMIT_UNCERTIFIED, True,
            "no equivariance check supplied; the test is one extra solve per generator and "
            "the composition layer relies on the declaration being true",
            "one solve per generator",
        )
    worst = 0.0
    failures = []
    for g in caps.equivariances:
        r = float(check(g))
        worst = max(worst, r)
        if r > 1e-8:
            failures.append(f"{g}: {r:.3e}")
    if failures:
        return FieldTest(
            "equivariances", list(caps.equivariances), f"residual {worst:.3e}", worst,
            REFUSE, True,
            f"declared equivariances do not hold: {failures}. Symmetry averaging is exact "
            "only for a group the expert actually has", "one solve per generator",
        )
    return FieldTest(
        "equivariances", list(caps.equivariances), f"residual {worst:.3e}", worst, ADMIT, True,
        "declared equivariances confirmed", "one solve per generator",
    )


def _test_differentiable(
    caps: ExpertCapabilities,
    space: InterfaceSpace | None,
    prolongation: Prolongation | None,
) -> FieldTest:
    """JVP against a finite difference. Disagreement is loud, not silent."""
    if not caps.differentiable.has_jvp:
        return FieldTest(
            "differentiable", caps.differentiable.value, "no jvp claimed", None, ADMIT, False,
            "no JVP claimed, so none is tested; the probe falls back to finite differences "
            f"({caps.probe_class()})", "cheap",
        )
    if caps.boundary_response_jvp is None:
        return FieldTest(
            "differentiable", caps.differentiable.value, "absent", None, REFUSE, False,
            "declares a JVP and supplies no boundary_response_jvp", "free",
        )
    if space is None or prolongation is None or caps.boundary_response is None:
        return FieldTest(
            "differentiable", caps.differentiable.value, None, None, ADMIT_UNCERTIFIED, False,
            "cannot compare the JVP against a finite difference without an interface space",
            "cheap",
        )
    port = caps.ports[0].name
    d = prolongation.prolong(np.eye(space.dim)[:, 0], space)
    base = np.zeros_like(d)
    exact = np.asarray(caps.boundary_response_jvp(port, base, d), dtype=float)
    eps = max(1e-6, 100.0 * caps.reproducibility_floor)
    fd = (
        np.asarray(caps.boundary_response(port, base + eps * d), dtype=float)
        - np.asarray(caps.boundary_response(port, base), dtype=float)
    ) / eps
    scale = max(float(np.linalg.norm(exact)), float(np.linalg.norm(fd)), 1e-30)
    resid = float(np.linalg.norm(exact - fd) / scale)
    ok = resid < 1e-4
    return FieldTest(
        "differentiable", caps.differentiable.value, f"jvp vs fd residual {resid:.3e}",
        resid, ADMIT if ok else REFUSE, False,
        "JVP agrees with a finite difference" if ok
        else f"JVP disagrees with a finite difference at {resid:.3e}", "cheap",
    )


def _test_validity(
    caps: ExpertCapabilities,
    suite: Sequence[str],
    falsified_on: Sequence[str],
) -> FieldTest:
    """The one axiom. Falsifiable, never verifiable."""
    if not caps.declares_validity:
        return FieldTest(
            "validity", None, "not declared", None, ADMIT_UNCERTIFIED, True,
            "no validity predicate, so the run cannot abstain and every bound is vacuous "
            "wherever the expert is out of distribution, with nothing saying where that "
            "is. The declination rate p is undefined, so the abstention horizon cannot be "
            "reported", "none exists",
        )
    if falsified_on:
        return FieldTest(
            "validity", "declared", f"falsified on {list(falsified_on)}", None, REFUSE, True,
            f"the predicate admits states where the expert is measurably wrong: "
            f"{list(falsified_on)}. validity can only be falsified, and it has been",
            "none exists",
        )
    return FieldTest(
        "validity", "declared", f"not falsified on suite of {len(suite)}", None,
        ADMIT_UNCERTIFIED, True,
        f"not falsified on suite {list(suite)}. Never 'valid': verifying the predicate "
        "requires knowing where the expert is wrong, which requires the reference the "
        "predicate exists to make unnecessary. This is the architecture's single "
        "unverifiable declaration, and the correct verdict on it is admit-uncertified",
        "none exists",
    )
