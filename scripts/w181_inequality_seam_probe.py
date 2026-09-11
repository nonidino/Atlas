"""W181 -- what the compiler does at a seam whose response has a KINK.

Rung 8's admissibility question, measured from the DECLARATION alone.  No
contact expert exists and none is written here: the fixture is three otherwise
identical two-agent graphs differing only in the shape of one side's
``boundary_response``.

    smooth        f(t) = A t + b                       -- the control
    kink-at-base  f(t) = A t + b + c relu(w . t) w     -- kink AT the probe base
    kink-at-gap   f(t) = A t + b + c relu(w . t - g) w -- kink a gap g away

``relu`` is the shape a complementarity condition takes once it is solved for
one of its two variables: a penalty normal law is ``p = k max(0, -g)``, and a
regularised stick-slip law is a ``min`` of the same family.  Nothing here is
Hertzian, nothing is Coulomb, and nothing solves a contact problem.  The
question is what the nine layers SEE.

Four things are measured, in order of how much they matter.

1.  **Does the compile differ at all?**  Compile all three and diff the decision
    records rule-for-rule.  If a kinked seam compiles to the same verdict under
    the same rules as a linear one, the framework cannot tell them apart, which
    is the whole admissibility question in one experiment.

2.  **R7's route, and whether any rule says so.**  R7 reads *"the probing method
    follows differentiable: jvp -> exact JVP; deterministic and SMOOTH -> finite
    differences with the step set by the reproducibility floor; otherwise
    regularized regression"*.  ``capability.probe_route`` checks ``has_jvp``,
    then ``deterministic``.  There is no third clause.  Counted here: how many
    decisions in the whole compile cite R7, and what the selected step actually
    is.

3.  **The epsilon sweep, with a horizon.**  Nine decades, ``1e-8`` to ``1e0``,
    on all three fixtures under one instrument.  W165 predicts the returned
    block "depends on the probe amplitude in a way no epsilon sweep will
    settle".  That prediction is checked rather than assumed, and the control is
    that every epsilon sweep in this vault so far has come back stable.

4.  **The step census.**  ``probe_block`` sets ``step = budget.fd_step or
    max(1e-2, 100 * reproducibility_floor)``.  For which of the vault's declared
    agents does the reproducibility floor set the step, as R7's sentence says it
    does?

Writes ``out/w181/w181.json``.  Runs in seconds on numpy alone; no torch, no
checkpoint, no network.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import io
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from atlas import Budget, compile_scheme                              # noqa: E402
from atlas.capability import (                                        # noqa: E402
    BCChannel,
    ClaimType,
    Differentiable,
    Direction,
    EllipticSubsolve,
    ExpertCapabilities,
    TimeDiscretization,
    port_decl,
)
from atlas.graph import Agent, CaseGraph, Connection, Decomposition   # noqa: E402
from atlas.ports import PortType, ResponseHalf                        # noqa: E402
from atlas.probe import ProbeBudget                                   # noqa: E402

OUT = os.path.join(HERE, "out", "w181")

#: One face, eight modes.  Small on purpose: the finding is structural and a
#: bigger seam would only make the same matrices slower.
M_EFF = 8

#: The fixture's ruler.  Satisfies s_e * s_f = s_P so C4 passes and the run is
#: not stopped by something irrelevant to the question.
RHO, U = 1.0, 1.0
MECH_SCALES = {"stress": RHO * U**2, "velocity": U, "power_area": RHO * U**3}

#: The gap the offset kink sits behind, in trace units.  Deliberately smaller
#: than the probe's own default step of 1e-2, because that is the arrangement a
#: contact seam is actually in: the gap is small and the probe is not.
GAP = 1.0e-3

#: How hard the kink bites, as a fraction of the linear operator's scale.
KINK_GAIN = 0.6


# --------------------------------------------------------------------------
# the three responses.  Everything else about the two agents is identical.
# --------------------------------------------------------------------------


def _operator(seed: int, n: int = M_EFF) -> np.ndarray:
    """A symmetric positive definite block: passive, so E7 has nothing to say."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) * 0.05
    return A + A.T + 2.0 * np.eye(n)


def smooth_response(A: np.ndarray, b: np.ndarray):
    def respond(_port: str, trace) -> np.ndarray:
        return A @ np.asarray(trace, dtype=float) + b
    return respond


def kinked_response(A: np.ndarray, b: np.ndarray, w: np.ndarray,
                    gain: float, gap: float):
    """Linear, plus a rectified term along one direction.

    The rectifier is positively homogeneous when ``gap == 0``.  That is not a
    detail: ``relu(eps x) == eps relu(x)``, so a one-sided difference quotient
    taken from the kink is INDEPENDENT of eps, and an amplitude sweep across it
    is flat.  With ``gap > 0`` the homogeneity is broken and the quotient
    depends on whether the step reaches the gap.
    """
    w = np.asarray(w, dtype=float)

    def respond(_port: str, trace) -> np.ndarray:
        t = np.asarray(trace, dtype=float)
        return A @ t + b + gain * max(0.0, float(w @ t) - gap) * w
    return respond


def _caps(agent_id: str, respond) -> ExpertCapabilities:
    """One capability record.  The ONLY thing that varies is ``respond``.

    Every field below is chosen so that nothing except the kink can explain a
    difference between the three compiles: an explicit march, no elliptic part,
    a declared storage and validity, one governing family, one clock.
    """
    return ExpertCapabilities(
        expert_id=agent_id,
        ports=[port_decl(
            name="face:MECH",
            port_type=PortType.MECH,
            geometry="one flat face",
            direction=Direction.BIDIRECTIONAL,
            nondim=dict(MECH_SCALES),
            effective_resolution=M_EFF,
            response_half=ResponseHalf.EFFORT,
        )],
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        elliptic_subsolve=EllipticSubsolve.NONE,
        time_discretization=TimeDiscretization.EXPLICIT,
        stencil_radius=1,
        substeps_per_macro_step=1,
        # **The triple R7's finite-difference branch is selected by.**  No JVP,
        # deterministic, and a floor at one ulp.  `brake_thermal` declares the
        # same three and its own comment quotes R7's sentence back.
        differentiable=Differentiable.NONE,
        deterministic=True,
        reproducibility_floor=float(np.finfo(float).eps),
        dt_native=1.0e-2,
        storage=lambda u: 0.5 * float(np.dot(np.ravel(u), np.ravel(u))),
        validity=lambda state, cond=None: True,
        governing_family="fixture-mech-1d",
        lambda_ref="the same fixture uncut; this agent is its own reference",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash=f"w181/{agent_id}",
        boundary_response=respond,
        note="declaration-only fixture for W181. Not an expert and not a case study",
    )


def build(shape: str) -> CaseGraph:
    """Two agents, one MECH seam.  ``shape`` picks agent B's response only."""
    A_a, A_b = _operator(11), _operator(12)
    b_a = np.full(M_EFF, 0.05)
    b_b = np.full(M_EFF, -0.03)
    w = np.zeros(M_EFF)
    w[0] = 1.0                       # the kink direction: one interface mode

    left = _caps("A", smooth_response(A_a, b_a))
    if shape == "smooth":
        right = _caps("B", smooth_response(A_b, b_b))
    elif shape == "kink-at-base":
        right = _caps("B", kinked_response(A_b, b_b, w, KINK_GAIN, 0.0))
    elif shape == "kink-at-gap":
        right = _caps("B", kinked_response(A_b, b_b, w, KINK_GAIN, GAP))
    else:
        raise ValueError(shape)

    return CaseGraph(
        name=f"w181-{shape}",
        agents=[Agent("A", left, domain="A"), Agent("B", right, domain="B")],
        connections=[Connection(
            seam_id="s",
            a=("A", "face:MECH"),
            b=("B", "face:MECH"),
            port_type=PortType.MECH,
            geometrically_coincident=True,
            derive_space=True,
        )],
        decomposition=Decomposition.NON_OVERLAPPING,
        macro_dt=1.0e-2,
        note="W181: a seam whose interface condition would be an INEQUALITY, "
             "declared as if it were an equality, because nothing can declare "
             "otherwise",
    )


# --------------------------------------------------------------------------
# 1 + 2 -- the compile, and what it says about R7
# --------------------------------------------------------------------------


def _rules(result) -> list[list[str]]:
    """Every decision as (layer, rule, verdict), in order.  The comparable part."""
    out = []
    for d in result.decisions.decisions:
        out.append([d.layer, d.rule, d.verdict.value, d.subject])
    return out


def compile_all() -> dict:
    shapes = ["smooth", "kink-at-base", "kink-at-gap"]
    per: dict[str, dict] = {}
    for s in shapes:
        g = build(s)
        r = compile_scheme(g, Budget(), ProbeBudget())
        op = r.seam_operators.get("s")
        per[s] = {
            "verdict": r.verdict.value,
            "runnable": bool(r.runnable),
            "envelope": list(r.envelope.tuple()),
            "rules": _rules(r),
            "n_decisions": len(r.decisions.decisions),
            "beta": None if op is None else float(op.beta),
            "kappa": None if op is None else float(op.kappa),
            "norm_S": None if op is None else float(np.linalg.norm(op.S, 2)),
            "passivity_defect": None if op is None else float(op.passivity_defect),
            "probe_route": {a.agent_id: a.capabilities.probe_route()
                            for a in g.agents},
        }
    base = per["smooth"]["rules"]
    for s in shapes:
        per[s]["rules_identical_to_smooth"] = (per[s]["rules"] == base)
        per[s]["verdict_identical_to_smooth"] = (
            per[s]["verdict"] == per["smooth"]["verdict"])
    return per


def r7_citations() -> dict:
    """Does any rule in the package CITE R7, and does any check smoothness?"""
    import inspect

    from atlas import capability, compiler, conformance, probe, scheme

    mods = {"compiler": compiler, "probe": probe, "capability": capability,
            "scheme": scheme, "conformance": conformance}
    cites = {}
    for name, mod in mods.items():
        src = inspect.getsource(mod)
        cites[name] = {
            "mentions_R7": src.count('"R7"') + src.count("'R7'") + src.count("R7 --"),
            "mentions_smooth": src.lower().count("smooth"),
        }
    # The decisive one: is there any decision, on any graph, whose rule is R7?
    emitted = []
    for s in ("smooth", "kink-at-base", "kink-at-gap"):
        r = compile_scheme(build(s), Budget(), ProbeBudget())
        emitted += [d.rule for d in r.decisions.decisions if "R7" in str(d.rule)]
    return {
        "source_mentions": cites,
        "decisions_citing_R7": emitted,
        "n_decisions_citing_R7": len(emitted),
        "probe_route_source": inspect.getsource(
            capability.ExpertCapabilities.probe_route),
        "rule_text": scheme.RULES["R7"],
    }


def step_actually_used() -> dict:
    """``step = budget.fd_step or max(1e-2, 100 * reproducibility_floor)``.

    R7's sentence says the step is set by the reproducibility floor.  For which
    declared agents in this package is that true?
    """
    import importlib

    names = ["brake_thermal", "channel_ns", "cooling_loop", "front_wing",
             "ground_effect", "powertrain", "thermal_seam", "thermal_strain",
             "wind_farm", "wind_farm_real", "window_ns", "wing_fsi"]
    rows, floor_sets_it, total = [], 0, 0
    for n in names:
        try:
            mod = importlib.import_module(f"atlas.cases.{n}")
            graph = mod.build()
            if isinstance(graph, tuple):              # a few builders return extras
                graph = next(g for g in graph if hasattr(g, "agents"))
        except Exception as exc:                      # a case needing an arg or a donor
            rows.append({"case": n, "skipped": str(exc)[:120]})
            continue
        for a in graph.agents:
            c = a.capabilities
            f = float(c.reproducibility_floor)
            step = max(1e-2, 100.0 * f)
            by_floor = bool(100.0 * f > 1e-2)
            total += 1
            floor_sets_it += int(by_floor)
            rows.append({"case": n, "agent": a.agent_id, "route": c.probe_route(),
                         "floor": f, "step": step, "step_set_by_floor": by_floor})
    return {"rows": rows, "n_agents": total, "n_step_set_by_floor": floor_sets_it,
            "source": _floor_declarations()}


def _floor_declarations() -> dict:
    """Every declared ``reproducibility_floor`` in the package, statically.

    Six case builders need live field state and cannot be constructed here, so
    the built-graph census above misses them.  The DECLARATION is what decides
    the step, and it is in the source, so this reads it there instead and covers
    the package.
    """
    import ast
    import glob

    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "atlas", "cases", "*.py"))):
        src = io.open(path, encoding="utf-8").read()
        try:
            tree = ast.parse(src)
        except SyntaxError:                            # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.keyword) or node.arg != "reproducibility_floor":
                continue
            text = (ast.get_source_segment(src, node.value) or "").strip()
            value = None
            if "finfo" in text:
                value = float(np.finfo(float).eps)
            else:
                try:
                    value = float(ast.literal_eval(node.value))
                except Exception:
                    value = None
            rows.append({
                "file": os.path.basename(path), "text": text, "value": value,
                "step_set_by_floor": (None if value is None
                                      else bool(100.0 * value > 1e-2)),
            })
    known = [r for r in rows if r["value"] is not None]
    return {
        "rows": rows,
        "n_declarations": len(rows),
        "n_resolved": len(known),
        "n_step_set_by_floor": sum(1 for r in known if r["step_set_by_floor"]),
        "floor_needed_to_set_the_step": 1e-4,
    }


# --------------------------------------------------------------------------
# 3 -- the epsilon sweep, marched nine decades
# --------------------------------------------------------------------------


def epsilon_sweep() -> dict:
    """One instrument, three fixtures, nine decades.

    The quantity is the probed block itself, not a scalar summary of it: two
    matrices can share a norm and a beta and differ everywhere.  ``drift`` is
    ``||S(eps) - S(eps_ref)||_2 / ||S(eps_ref)||_2`` against the smallest step
    in the sweep, so a flat column means the instrument returned the SAME
    operator over nine decades.
    """
    decades = [10.0 ** k for k in range(-8, 1)]
    out: dict[str, dict] = {}
    for s in ("smooth", "kink-at-base", "kink-at-gap"):
        g = build(s)
        blocks, rows = {}, []
        for eps in decades:
            r = compile_scheme(g, Budget(), ProbeBudget(fd_step=eps))
            op = r.seam_operators["s"]
            blocks[eps] = op.S.copy()
            rows.append({
                "eps": eps,
                "norm_S": float(np.linalg.norm(op.S, 2)),
                "beta": float(op.beta),
                "kappa": float(op.kappa),
                "passivity_defect": float(op.passivity_defect),
            })
        ref = blocks[decades[0]]
        nref = float(np.linalg.norm(ref, 2))
        for row in rows:
            d = blocks[row["eps"]] - ref
            row["drift"] = float(np.linalg.norm(d, 2) / nref)
        drifts = [row["drift"] for row in rows]
        # How many digits every step in the sweep agrees to.  A sweep that is
        # stable to seven digits is what this vault has seen every time so far.
        worst = max(drifts)
        out[s] = {
            "rows": rows,
            "max_drift": worst,
            "digits_stable": (16 if worst == 0.0
                              else int(np.floor(-np.log10(worst)))),
            "monotone_in_eps": bool(all(
                drifts[i] <= drifts[i + 1] + 1e-15 for i in range(len(drifts) - 1))),
        }
    return out


def truth_at_the_kink() -> dict:
    """What the block WOULD be on each side of the kink, computed exactly.

    Not a probe: the fixture's response is known in closed form, so the two
    one-sided Jacobians are available and the chord can be compared against
    both.  This is the horizon the sweep alone does not have -- a sweep that is
    flat is only evidence if something independent says what it is flat AT.
    """
    A_b = _operator(12)
    w = np.zeros(M_EFF)
    w[0] = 1.0
    J_minus = A_b                                     # w . t < gap: the rectifier is off
    J_plus = A_b + KINK_GAIN * np.outer(w, w)         # w . t > gap: it is on
    jump = float(np.linalg.norm(J_plus - J_minus, 2) / np.linalg.norm(J_minus, 2))

    # The one-sided quotient the probe actually forms, at the base, per decade.
    rows = []
    for shape, gap in (("kink-at-base", 0.0), ("kink-at-gap", GAP)):
        f = kinked_response(A_b, np.full(M_EFF, -0.03), w, KINK_GAIN, gap)
        base = np.zeros(M_EFF)
        f0 = f("face:MECH", base)
        for k in range(-8, 1):
            eps = 10.0 ** k
            cols = [(f("face:MECH", base + eps * np.eye(M_EFF)[:, j]) - f0) / eps
                    for j in range(M_EFF)]
            J = np.column_stack(cols)
            rows.append({
                "shape": shape, "eps": eps,
                "to_J_minus": float(np.linalg.norm(J - J_minus, 2)
                                    / np.linalg.norm(J_minus, 2)),
                "to_J_plus": float(np.linalg.norm(J - J_plus, 2)
                                   / np.linalg.norm(J_plus, 2)),
                "straddles_gap": bool(eps > gap),
            })
    return {"jump_relative": jump, "gap": GAP, "gain": KINK_GAIN, "rows": rows}


# --------------------------------------------------------------------------
# 5 -- what happens when the seam is declared TRUTHFULLY
# --------------------------------------------------------------------------


def declared_truthfully() -> dict:
    """A contact patch's boundary MOVES with the solution. Declare that.

    `MotionClass.SOLUTION_DEPENDENT` is the only field in the whole schema that
    a contact seam can truthfully set and that any rule reads.  Setting it is
    what a careful declarer would do, and the question this answers is whether
    the compiler then asks the admissibility question or something else.
    """
    from atlas.capability import MotionClass

    out = {}
    for shape in ("smooth", "kink-at-gap"):
        g = build(shape)
        for a in g.agents:
            for port in a.capabilities.ports:
                port.motion_class = MotionClass.SOLUTION_DEPENDENT
        r = compile_scheme(g, Budget(), ProbeBudget())
        rules = [[d.layer, d.rule] for d in r.decisions.decisions
                 if d.verdict.value == "refuse"]
        out[shape] = {
            "verdict": r.verdict.value,
            "runnable": bool(r.runnable),
            "refusing_rules": rules,
            "envelope": list(r.envelope.tuple()),
        }
    return out


# --------------------------------------------------------------------------
# 6 -- the residual the framework reports, against the residual of the
#      problem it actually posed
# --------------------------------------------------------------------------


def residual_blindness() -> dict:
    """Solve the seam the way L5 solves it, then evaluate the TRUE condition.

    `InterfaceProblem.residual` is ``S lam - chi``, and for affine agents that is
    identically ``sum_i P_i^* f_i(P_i lam)`` -- the port-matching condition
    itself.  With a kinked agent the two part company: the framework drives its
    own residual to the solver's tolerance and the condition it was standing in
    for is not satisfied.  Nothing in the run reports the second number, so this
    computes it here.
    """
    from atlas.scheme import Accelerator
    from atlas.solve import build_interface_problem, solve_interface

    out = {}
    for shape in ("smooth", "kink-at-base", "kink-at-gap"):
        g = build(shape)
        r = compile_scheme(g, Budget(), ProbeBudget())
        conn = g.connections[0]
        transfer = r.transfers[conn.seam_id]
        op = r.seam_operators[conn.seam_id]
        problem = build_interface_problem(g, conn, transfer, op)
        lam, iters, history = solve_interface(problem, Accelerator.DIRECT_SCHUR)

        def exact(vec):
            total = np.zeros(transfer.space.dim)
            for agent_id, port_name in (conn.a, conn.b):
                caps = g.agent(agent_id).capabilities
                P = transfer.prolongations[agent_id]
                flux = np.asarray(
                    caps.boundary_response(port_name, P.prolong(vec, transfer.space)),
                    dtype=float)
                total += P.reduce(flux, transfer.space)
            return total

        reported = float(np.linalg.norm(problem.residual(lam)))
        truth = float(np.linalg.norm(exact(lam)))
        scale = float(np.linalg.norm(problem.chi)) or 1.0
        out[shape] = {
            "reported_residual": reported,
            "true_residual": truth,
            "chi_norm": scale,
            "reported_relative": reported / scale,
            "true_relative": truth / scale,
            "ratio_true_over_reported": (float("inf") if reported == 0.0
                                         else truth / reported),
            "iterations": int(iters),
            "lam_norm": float(np.linalg.norm(lam)),
        }
    return out


# --------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    results = {
        "what": "W181 -- an inequality-shaped seam through the nine layers, "
                "measured from the declaration alone",
        "date": "2026-09-10",
        "compiles": compile_all(),
        "r7": r7_citations(),
        "step_census": step_actually_used(),
        "sweep": epsilon_sweep(),
        "truth": truth_at_the_kink(),
        "declared_truthfully": declared_truthfully(),
        "residual": residual_blindness(),
    }
    results["elapsed_seconds"] = time.time() - t0

    path = os.path.join(OUT, "w181.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)

    c = results["compiles"]
    print("=== 1. the compile ===")
    for s, row in c.items():
        print("  %-13s verdict %-18s rules identical to smooth: %s  beta %.6f"
              % (s, row["verdict"], row["rules_identical_to_smooth"], row["beta"]))
    print("=== 2. R7 ===")
    print("  decisions citing R7 anywhere:", results["r7"]["n_decisions_citing_R7"])
    print("  agents whose step is set by their floor: %d of %d built, %d of %d declared"
          % (results["step_census"]["n_step_set_by_floor"],
             results["step_census"]["n_agents"],
             results["step_census"]["source"]["n_step_set_by_floor"],
             results["step_census"]["source"]["n_resolved"]))
    print("=== 3. the sweep, nine decades ===")
    for s, row in results["sweep"].items():
        print("  %-13s max drift %.3e  stable to %d digits"
              % (s, row["max_drift"], row["digits_stable"]))
    print("=== 4. the jump the sweep is blind to ===")
    print("  ||J+ - J-|| / ||J-|| = %.4f" % results["truth"]["jump_relative"])
    print("=== 5. the seam declared truthfully (motion_class) ===")
    for shape, row in results["declared_truthfully"].items():
        print("  %-13s %-8s refusing rules: %s"
              % (shape, row["verdict"],
                 sorted({r[0] + "/" + str(r[1]) for r in row["refusing_rules"]})))
    print("=== 6. the residual reported, against the residual posed ===")
    for shape, row in results["residual"].items():
        print("  %-13s reported %.3e   true %.3e   ratio %.3e"
              % (shape, row["reported_relative"], row["true_relative"],
                 row["ratio_true_over_reported"]))
    print("wrote", path, "in %.1f s" % results["elapsed_seconds"])


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    main()
