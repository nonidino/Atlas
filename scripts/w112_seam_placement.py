"""W112 -- CS-9*, the seam-placement search: where to cut, decided by an algorithm.

    python scripts/w112_seam_placement.py [--out out/w112] [--quick]

`poc1-retrospective-and-hybrid-roadmap` 4 proposes this case study and W112 is
the row.  `atlas/cases/seam_placement.py`'s module docstring carries the design
and the correction to W112's own premise (Tier 10 already falsified Q as a
ranking function on ONE cut of a fixed decomposition; what has never happened
is a SEARCH over decompositions, under a cost budget, on real geometry,
against a hand-chosen baseline).  This is the driver.

Stages
-------

  0  controls        ZERO (one window, exactly zero defect), REPRODUCE
                     (`from_array_tiling` reproduces `wake_array`'s tiling to
                     the bit), NOISE (the instrument's own floor, measured
                     rather than assumed).
  A  CS-7 N=6        the wake array's own domain and developed state.  Enumerate
                     candidates, score every one, and run the search under two
                     cost budgets against the hand-chosen 3x2 tiling.
  B  CS-7 N=12       the same at the next rung, which is the other hand-chosen
                     tiling this vault has measured.
  C  KNOWN           the one-rotor state: a single localized wake in an
                     otherwise near-freestream domain, where the right answer
                     is known before the run.
  D  BASIS           Q's orbit under an admissible re-declaration of the same
                     interface space, per seam, on every geometry -- 9's own
                     flagged weakness, stress-tested where a search wants to go.
  E  HORIZON         re-rank every criterion after K exchange intervals, because
                     a criterion true as far as it was marched is a criterion
                     true as far as it was marched.
  F  CS-9            the shared thermal-strain domain: one spatial cut, two
                     families, and whether they want it in the same place.

Everything lands in ``out/w112/w112.json`` and every number in the wiki pages
is quoted from it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from atlas.cases import seam_placement as sp                       # noqa: E402
from atlas.cases import wake_array as wa                           # noqa: E402


def say(msg: str) -> None:
    print(msg, flush=True)


def _j(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# ---------------------------------------------------------------------------
# the candidate space
# ---------------------------------------------------------------------------

#: Per-cut offsets, in cells.  Each cut moves INDEPENDENTLY by one of these, so
#: a seam can step off a wake centreline while its neighbours stay put.  The
#: span is +/- 32 cells = 1 D = one rotor diameter, which is the length scale
#: the decision is actually about.
OFFSETS = (-32, 0, 32)

#: Common shifts, applied to a whole cut set at once.  These reach FURTHER than
#: `OFFSETS` and cost one candidate per value rather than ``|OFFSETS|^k``, so
#: they buy the wide excursions cheaply -- and they are the only family that
#: survives at four or more cuts on an axis.
SHIFTS = (-64, -48, 48, 64)

#: Per-side halos for the halo sweep.  `wake_array.HALO` is 16 cells of
#: OVERLAP, so 8 is the hand-chosen one and 4, 16 and 24 bracket it.  The halo
#: is swept on its OWN axis rather than inside the placement pool: CS-7 nails
#: it down explicitly so that Pi does not move, and a search that varied
#: placement and overlap together would be measuring neither.
HALOS = (4, 8, 16, 24)

#: How many exchange intervals stage E marches.  Eight is `w16_cut_policy`'s
#: own horizon and it is what makes the ranking a statement about a rollout
#: rather than about a single application of the operator.
HORIZON = 8

QUICK = {"offsets": (0, 32), "shifts": (), "halos": (8,)}

#: The N=6 geometry is the gate, so it gets the widest net: every independent
#: per-cut move plus the wide common excursions, and the halo axis swept.
FULL = {"offsets": OFFSETS, "shifts": SHIFTS, "halos": HALOS}

#: The KNOWN control gets the independent moves only: it is asking whether the
#: search avoids a known-bad cut, and the answer lives in its own `WAKE` sweep
#: rather than in the width of the placement pool.
REDUCED = {"offsets": OFFSETS, "shifts": (), "halos": (8,)}

#: N=12 has twelve windows and six factorizations, so the full net is 494
#: candidates on a domain that is 1.9x larger per solve.  It is asking one
#: narrower question -- *does the N=6 answer hold at the next rung* -- so its
#: pool is restricted to the hand-chosen shape and its transpose, moved by a
#: common shift on each axis.  **That restriction is a property of this run and
#: is on the artifact**: a search that did not look at 2x6 cannot report that
#: 2x6 is not better.
LADDER = {"offsets": (), "shifts": (-64, -32, 0, 32, 64), "halos": (8,),
          "shapes": [(4, 3), (3, 4)]}


def _factor_pairs(n: int) -> list[tuple[int, int]]:
    """Every ``(n_col, n_row)`` whose product is ``n``."""
    return [(c, n // c) for c in range(1, n + 1) if n % c == 0]


def pools(nx: int, ny: int, baseline: sp.RectDecomposition,
          space: dict) -> dict[str, dict]:
    """The three pools, each a well-posed question with its own budget.

    **PLACE** is the gate.  ``n_windows`` is fixed at the hand-chosen tiling's
    own count and the halo at its own width, so the only thing that varies is
    *where the cuts are* -- which is the question `f1-pathmap-and-end-goal`'s
    fourth capability names and the one no case study in this vault has ever
    asked.  Every ``(n_col, n_row)`` with the right product is included, so a
    3x2 tiling can lose to a 2x3 or a 6x1 one.

    **COUNT** sweeps the agent count at the evenly-spaced placement, and it is
    where an upper-bound budget's degeneracy is shown rather than described:
    the argmin over this pool is the smallest admissible count, every time,
    for every criterion.

    **HALO** sweeps the overlap at the hand-chosen placement.  It is the axis
    CS-7 deliberately froze, and freezing it is why `master-error-bound`'s Pi
    is comparable across that ladder -- so it is swept here, alone, where
    nothing else is moving.
    """
    c = baseline.cost
    shapes = space.get("shapes") or _factor_pairs(c.n_windows)
    return {
        "PLACE": {
            "budget": f"n_windows == {c.n_windows}, halo == {baseline.halo}, "
                      f"shapes {sorted(shapes)}",
            "note": "the placement question: same agent count, same overlap, "
                    "different cuts. THE GATE",
            "candidates": sp.enumerate_candidates(
                nx, ny,
                n_cols=sorted({a for a, _b in shapes}),
                n_rows=sorted({b for _a, b in shapes}),
                halos=(baseline.halo,), offsets=space["offsets"],
                shifts=space["shifts"],
                keep=lambda d: (d.n_col, d.n_row) in set(shapes)),
        },
        "SHAPE": {
            "budget": f"n_col == {baseline.n_col}, n_row == {baseline.n_row}, "
                      f"halo == {baseline.halo}",
            "note": "PLACE with the SHAPE held too: the person's own tiling "
                    "shape, only the cuts moved. Separates 'a better shape was "
                    "available' from 'the cuts were in the wrong place'",
            "candidates": sp.enumerate_candidates(
                nx, ny, n_cols=(baseline.n_col,), n_rows=(baseline.n_row,),
                halos=(baseline.halo,), offsets=space["offsets"],
                shifts=space["shifts"]),
        },
        "COUNT": {
            "budget": f"n_windows <= {c.n_windows}, evenly spaced",
            "note": "the degeneracy: an upper-bound budget on agent count is "
                    "not a placement question and the search proves it",
            "candidates": sp.enumerate_candidates(
                nx, ny, n_cols=range(1, c.n_windows + 1),
                n_rows=range(1, c.n_windows + 1), halos=(baseline.halo,),
                offsets=(0,), shifts=(0,),
                keep=lambda d: d.n_windows <= c.n_windows),
        },
        "HALO": {
            "budget": f"the hand-chosen placement, halo in {space['halos']}",
            "note": "the axis CS-7 froze, swept alone",
            "candidates": [
                sp.RectDecomposition(nx=nx, ny=ny, x_cuts=baseline.x_cuts,
                                     y_cuts=baseline.y_cuts, halo=h,
                                     ramp=baseline.ramp)
                for h in space["halos"]],
        },
    }


# ---------------------------------------------------------------------------
# stage 0 -- the controls
# ---------------------------------------------------------------------------


def stage_controls(u, v, dt, tiling) -> dict:
    """ZERO, REPRODUCE and NOISE.  Each would invalidate the run if it failed."""
    out = {}

    # ZERO: one window, no interface, and the composed step IS the monolith.
    one = sp.RectDecomposition(nx=tiling.nx, ny=tiling.ny)
    z = sp.composed_defect(one, u, v, dt)
    out["ZERO"] = {
        "label": one.label, "n_seams": len(one.seams()),
        "n_overlap_pairs": len(one.overlap_pairs()),
        "composed_defect": z["defect"],
        "exactly_zero": z["defect"] == 0.0,
        "note": "chi is identically one, cutting and assembling are the identity, "
                "so the composed step is the monolith bit for bit -- not to a "
                "tolerance. CS-7's N=1 rung, repeated because it is free",
    }
    say(f"  0 ZERO      one window: composed defect = {z['defect']!r} "
        f"(exactly zero: {z['defect'] == 0.0})")

    # REPRODUCE: this file's parameterization against wake_array's own tiling.
    dec = sp.from_array_tiling(tiling)
    boxes_ok = ([tuple(b) for b in dec.boxes]
                == [(ox, ox + wa.N, oy, oy + wa.N) for ox, oy in tiling.offsets])
    w_gap = max(float(np.max(np.abs(a - b)))
                for a, b in zip(tiling.weights(), dec.weights()))
    ov = len(dec.overlap_pairs())
    out["REPRODUCE"] = {
        "label": dec.label, "boxes_identical": bool(boxes_ok),
        "weights_max_abs_difference": w_gap,
        "weights_bitwise_identical": w_gap == 0.0,
        "partition_of_unity_residual": float(
            np.max(np.abs(np.sum(dec.weights(), axis=0) - 1.0))),
        "n_overlap_pairs": ov,
        "n_overlap_pairs_cs7": len(wa.ArrayTiling(
            n_col=tiling.n_col, n_row=tiling.n_row).weights()) and ov,
        "n_seams": len(dec.seams()),
        "note": "the hand-chosen tiling is a candidate in this file's search "
                "space, and it is the SAME object -- a search graded against a "
                "baseline it cannot express is graded against nothing",
    }
    say(f"  0 REPRODUCE boxes identical: {boxes_ok}, weights differ by {w_gap!r}, "
        f"{ov} overlapping pairs")

    # NOISE: the instrument's own floor.  Two readings -- the same call twice
    # (determinism) and the probe at a different finite-difference step, which
    # is the floor that actually bounds a ranking.
    a = sp.composed_defect(dec, u, v, dt)
    b = sp.composed_defect(dec, u, v, dt)
    r1 = sp.seam_scores(dec, u, v, dt)
    r2 = [dict(r) for r in sp.seam_scores(dec, u, v, dt)]
    q1 = sp.aggregate_seams(r1)
    q2 = sp.aggregate_seams(r2)
    # the same seam probed at half the step: an equally admissible reading of
    # the same operator, and the spread between them is Q's real resolution
    half = [{"Q_ai_inference": None} for _ in dec.seams()]
    qs_half = []
    for s in dec.seams():
        Sa = sp._block(dec, s.a, s.a_face, u, v, dt, step=sp.PROBE_STEP / 2)
        Sb = sp._block(dec, s.b, s.b_face, u, v, dt, step=sp.PROBE_STEP / 2)
        S = Sa + Sb
        beta = float(np.linalg.svd(S, compute_uv=False)[-1])
        qs_half.append(sp.cut_score(S, beta))
    q_declared = [r["Q_ai_inference"] for r in r1]
    rel = [abs(x - y) / max(abs(y), 1e-300) for x, y in zip(qs_half, q_declared)]
    out["NOISE"] = {
        "defect_repeat_identical": a["defect"] == b["defect"],
        "defect_repeat_relative_gap":
            abs(a["defect"] - b["defect"]) / max(a["defect"], 1e-300),
        "Q_repeat_identical": q1["Q_ai_inference__max"] == q2["Q_ai_inference__max"],
        "Q_at_half_fd_step": qs_half,
        "Q_at_declared_fd_step": q_declared,
        "Q_fd_step_relative_spread_max": max(rel),
        "Q_fd_step_relative_spread_mean": float(np.mean(rel)),
        "fd_step": sp.PROBE_STEP,
        "note": "the pipeline is deterministic, so a repeat is bitwise. The floor "
                "that bounds a RANKING is the probe's own: halving the "
                "finite-difference step is an equally admissible reading of the "
                "same operator, and Q moves by the amount reported here. A "
                "ranking difference smaller than this is not a ranking difference",
    }
    say(f"  0 NOISE     defect repeat bitwise: {out['NOISE']['defect_repeat_identical']}; "
        f"Q moves {max(rel):.3e} (max) under a halved FD step")
    return out


# ---------------------------------------------------------------------------
# stages A / B / C -- score a geometry, then search it
# ---------------------------------------------------------------------------


def score_geometry(tag: str, u, v, tiling, space: dict, march: bool,
                   extra: dict[str, list] | None = None) -> dict:
    """Score every candidate of every pool, then run each criterion's search."""
    nx, ny = tiling.nx, tiling.ny
    dt = sp.exchange_interval(u, v)
    baseline = sp.from_array_tiling(tiling)
    pl = pools(nx, ny, baseline, space)
    for name, block in (extra or {}).items():
        pl[name] = block

    # The hand-chosen tiling is NOT evenly spaced on this domain -- 128-cell
    # windows at a 112 stride do not divide 352 -- so it is not produced by the
    # enumeration and is added to every pool it belongs in, explicitly.
    for block in pl.values():
        if baseline.label not in {c.label for c in block["candidates"]}:
            block["candidates"] = list(block["candidates"]) + [baseline]

    uniq: dict[str, sp.RectDecomposition] = {}
    for block in pl.values():
        for c in block["candidates"]:
            uniq.setdefault(c.label, c)
    say(f"  {tag}: domain {nx}x{ny}, dt = {dt:.6e}, baseline {baseline.label} "
        f"(cost {baseline.cost.as_dict()})")
    for name, block in pl.items():
        say(f"      pool {name:6s} {len(block['candidates']):4d} candidates "
            f"[{block['budget']}]")
    say(f"      {len(uniq)} distinct candidates to score")

    scored: dict[str, sp.CandidateScore] = {}
    t0 = time.perf_counter()
    for n, (label, c) in enumerate(uniq.items()):
        scored[label] = sp.score_candidate(
            c, u, v, dt, probe=True, march=(HORIZON,) if march else ())
        if (n + 1) % 10 == 0 or n + 1 == len(uniq):
            say(f"      scored {n + 1}/{len(uniq)}  "
                f"({time.perf_counter() - t0:.0f} s)")
    base_score = scored[baseline.label]

    searches = {}
    for name, block in pl.items():
        pool = [scored[c.label] for c in block["candidates"]]
        for crit in sp.CRITERIA:
            rep = sp.search(pool, crit, block["budget"], base_score).report()
            rep["pool"] = name
            rep["pool_note"] = block["note"]
            searches[f"{name}::{crit}"] = rep

    return {
        "tag": tag, "nx": nx, "ny": ny, "dt": dt,
        "hand_chosen": baseline.as_dict(),
        "hand_chosen_is_uniform": baseline.is_uniform,
        "n_candidates_scored": len(uniq),
        "candidate_space": {k: list(v) for k, v in space.items()},
        "pools": {k: {"budget": v["budget"], "note": v["note"],
                      "n_candidates": len(v["candidates"]),
                      "labels": [c.label for c in v["candidates"]]}
                  for k, v in pl.items()},
        "searches": searches,
        "scores": [s.as_dict() for s in scored.values()],
        "_scores": list(scored.values()),        # not serialized; stages D/E use it
    }


def disagreement_table(block: dict) -> dict:
    """Every place a criterion and the measured defect disagree about a ranking.

    Two readings of "disagree", because they are different failures:

      * **sign** -- the rank correlation is negative, i.e. the criterion
        systematically prefers worse cuts.  That is `tier0-measurements`
        10.2's finding about Q, and whether it survives on real geometry is
        the question this file was built to answer.
      * **argmin** -- the criterion's chosen cut is not the best available.
        A criterion can correlate positively and still pick badly, and it is
        the argmin that a search actually acts on.
    """
    rows = []
    for key, rep in block["searches"].items():
        bud, crit = key.split("::", 1)
        rc = rep.get("rank_correlation_vs_defect")
        rows.append({
            "geometry": block["tag"], "budget": bud, "criterion": crit,
            "rank_correlation": rc,
            "sign_disagrees": (rc is not None and rc < 0.0),
            "argmin_disagrees": rep["chosen"] != rep["best_available"],
            "penalty_vs_best": rep["penalty_vs_best"],
            "fraction_of_available_range_lost":
                rep["fraction_of_available_range_lost"],
            "chosen_over_hand_chosen": rep.get("chosen_over_hand_chosen"),
        })
    return {"rows": rows,
            "n_sign_disagreements": sum(1 for r in rows if r["sign_disagrees"]),
            "n_argmin_disagreements": sum(1 for r in rows if r["argmin_disagrees"])}


# ---------------------------------------------------------------------------
# stage D -- the basis orbit, on real decompositions
# ---------------------------------------------------------------------------


def stage_basis(blocks: dict, n_frames: int = 64, max_seams: int = 40) -> dict:
    """9's flagged weakness, measured per seam of a real decomposition.

    `tier0-measurements` 10.2 measured a 361x orbit on ONE seam of a linear
    strip model.  What 9 says is that the Fourier basis is a locality measure
    *by assumption*; a placement search actively seeks out the geometries where
    that assumption is weakest, so the orbit is measured here on the seams the
    search actually visits.
    """
    rows = []
    for tag, block in blocks.items():
        seen = 0
        for sc in block["_scores"]:
            for r in sc.seams:
                if seen >= max_seams:
                    break
                orb = sp.basis_orbit(r["S"], n_frames=n_frames)
                rows.append({"geometry": tag, "decomposition": sc.dec.label,
                             "seam_id": r["seam_id"], **orb})
                seen += 1
            if seen >= max_seams:
                break
    ratios = [r["orbit_ratio"] for r in rows]
    betas = [r["beta_invariance_rel"] for r in rows]
    return {
        "rows": rows, "n_seams_measured": len(rows),
        "orbit_ratio_min": float(min(ratios)), "orbit_ratio_max": float(max(ratios)),
        "orbit_ratio_median": float(np.median(ratios)),
        "beta_invariance_worst": float(max(betas)),
        "note": "S -> U^T S U for orthogonal U re-declares the SAME interface "
                "space (range(PU) = range(P)), so the scheme is unchanged. beta "
                "and ||S|| are invariant; Q is not. Tier 10 measured 361x on one "
                "seam of a strip model -- these are real seams of real "
                "decompositions of a developed wake field",
    }


# ---------------------------------------------------------------------------
# stage E -- the horizon
# ---------------------------------------------------------------------------


def stage_horizon(blocks: dict) -> dict:
    """Re-rank every criterion against the defect after `HORIZON` intervals."""
    out = {}
    for tag, block in blocks.items():
        scores = [s for s in block["_scores"]
                  if f"defect_{HORIZON}_intervals" in s.marched]
        if len(scores) < 3:
            continue
        truth1 = [s.defect["defect"] for s in scores]
        truthN = [s.marched[f"defect_{HORIZON}_intervals"] for s in scores]
        rows = {}
        for crit in sp.CRITERIA:
            # A criterion that cannot express a candidate is not ranked on it:
            # the one-window decomposition has no seams and no cuts, so every
            # seam- and cut-derived criterion drops it rather than imputing a
            # value. `CandidateScore.scorable` is the same rule `SearchResult`
            # applies, used here so the two cannot disagree.
            pool = [s for s in scores if s.scorable(crit)]
            if len(pool) < 3:
                continue
            rows[crit] = {
                "n_ranked": len(pool),
                "vs_one_interval": sp.spearman(
                    [s.value(crit) for s in pool],
                    [s.defect["defect"] for s in pool]),
                f"vs_{HORIZON}_intervals": sp.spearman(
                    [s.value(crit) for s in pool],
                    [s.marched[f"defect_{HORIZON}_intervals"] for s in pool]),
            }
        finite = [x for x in truthN if np.isfinite(x)]
        out[tag] = {
            "n_scored": len(scores),
            "one_interval_vs_marched": sp.spearman(truth1, truthN),
            "marched_defect_min": float(min(finite)) if finite else None,
            "marched_defect_max": float(max(finite)) if finite else None,
            "n_nonfinite": int(sum(1 for x in truthN if not np.isfinite(x))),
            "criteria": rows,
        }
    return out


# ---------------------------------------------------------------------------
# stage F -- CS-9's shared domain
# ---------------------------------------------------------------------------


def stage_shell(quick: bool) -> dict:
    """One spatial cut of CS-9's shell, and both families' opinion of it."""
    from atlas.cases.thermal_strain import L_Z, NZ, W_STREAK

    T_prev, T = sp.shell_state()
    say(f"  F CS-9: shell {NZ} elements, streak at z = {0.5 * L_Z:.3f} m of "
        f"width {W_STREAK} m, developed field T in "
        f"[{float(T.min()):.1f}, {float(T.max()):.1f}] K")

    segs = sp.shell_candidates(NZ, (1, 2) if quick else (1, 2, 3))
    say(f"  F CS-9: {len(segs)} candidate segmentations (exhaustive)")
    rows, t0 = [], time.perf_counter()
    for n, s in enumerate(segs):
        rows.append(sp.score_shell(s, (T_prev, T)))
        if (n + 1) % 25 == 0 or n + 1 == len(segs):
            say(f"      scored {n + 1}/{len(segs)}  "
                f"({time.perf_counter() - t0:.0f} s)")

    multi = [r for r in rows if r["n_segments"] > 1]
    two = [r for r in rows if r["n_segments"] == 2]
    ctrl = next((r for r in rows if r["n_segments"] == 1), None)
    out = {
        "n_z": NZ, "streak_z_m": 0.5 * L_Z, "streak_width_m": W_STREAK,
        "halo_elements": sp.SHELL_HALO,
        "T_range": [float(T.min()), float(T.max())],
        "n_candidates": len(rows), "rows": rows,
        "one_segment_control": ctrl,
        "by_conduction": sp.shell_search(multi, "conduction_defect"),
        "by_elasticity": sp.shell_search(multi, "elasticity_defect"),
    }
    if ctrl is not None:
        # The ZERO control on this domain: one segment is the monolith in both
        # families, so both defects must be machine zero. The elasticity half
        # is the one that had to be got right -- a constrained solve on a
        # ring-free segment returns a rigid-body mode and reports it as a
        # decomposition error.
        out["ZERO_shell"] = {
            "conduction_defect": ctrl["conduction_defect"],
            "elasticity_defect": ctrl["elasticity_defect"],
            "both_machine_zero": (ctrl["conduction_defect"] < 1e-9
                                  and ctrl["elasticity_defect"] < 1e-9),
        }
        say(f"  F ZERO: one segment gives conduction "
            f"{ctrl['conduction_defect']:.3e}, elasticity "
            f"{ctrl['elasticity_defect']:.3e}")

    if two:
        # The KNOWN answer on this domain, stated without a threshold: does the
        # conduction defect fall MONOTONICALLY as the cut retreats from the
        # streak?  A rank correlation over the whole sweep is the statement;
        # picking a "near" and a "far" band would be choosing the answer.
        d = sorted(two, key=lambda r: r["nearest_cut_to_streak"])
        out["two_segment_scan"] = [
            {"z_cut": r["z_cuts"][0], "z_cut_m": r["z_cuts_m"][0],
             "widths_from_streak": r["nearest_cut_to_streak"],
             "conduction_defect": r["conduction_defect"],
             "elasticity_defect": r["elasticity_defect"]} for r in d]
        dist = [r["nearest_cut_to_streak"] for r in d]
        out["KNOWN_shell"] = {
            "distance_vs_conduction_defect_rank": sp.spearman(
                dist, [r["conduction_defect"] for r in d]),
            "distance_vs_elasticity_defect_rank": sp.spearman(
                dist, [r["elasticity_defect"] for r in d]),
            "conduction_at_the_streak": d[0]["conduction_defect"],
            "conduction_farthest_from_it": d[-1]["conduction_defect"],
            "conduction_penalty_for_cutting_through_it":
                d[0]["conduction_defect"] / max(d[-1]["conduction_defect"], 1e-300),
            "elasticity_at_the_streak": d[0]["elasticity_defect"],
            "elasticity_farthest_from_it": d[-1]["elasticity_defect"],
            "note": "the known answer is 'do not cut along the feature'. A "
                    "STRONGLY NEGATIVE conduction rank confirms it: the defect "
                    "falls as the cut retreats from the streak. Elasticity has "
                    "no local feature and is not expected to agree",
        }
        k = out["KNOWN_shell"]
        say(f"  F KNOWN: conduction defect vs distance from the streak ranks at "
            f"{k['distance_vs_conduction_defect_rank']}; cutting through it "
            f"costs {k['conduction_penalty_for_cutting_through_it']:.1f}x")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "out", "w112"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-n12", action="store_true")
    ap.add_argument("--skip-shell", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    space = QUICK if args.quick else FULL
    space_ladder = QUICK if args.quick else LADDER
    space_solo = QUICK if args.quick else REDUCED
    t_start = time.perf_counter()

    report: dict = {
        "row": "W112",
        "case_study": "CS-9* -- seam placement, decided by an algorithm",
        "code": ["atlas/cases/seam_placement.py",
                 "scripts/w112_seam_placement.py",
                 "tests/test_tier25_seam_placement.py"],
        "candidate_space": {"N6": {k: list(v) for k, v in space.items()},
                            "N12": {k: list(v) for k, v in space_ladder.items()},
                            "SOLO": {k: list(v) for k, v in space_solo.items()}},
        "horizon_intervals": HORIZON,
        "premise_correction": (
            "W112 and poc1-retrospective 4 both state that Q has never been "
            "used to choose a cut and has never been falsified. tier0-measurements "
            "10.2 (2026-08-28) falsified it as a ranking function on twelve cut "
            "placements of a fixed two-strip decomposition: rank correlation "
            "-0.853, a 361x basis orbit, constant to 8e-11 where the truth "
            "spreads 2.29x. probe.cut_score has read RETIRED since. What has "
            "never happened, and is what W112's own 'done when' asks for, is a "
            "SEARCH over decompositions under a cost budget on real geometry "
            "against a hand-chosen baseline. That is what this run is"),
    }

    # -- state -----------------------------------------------------------
    d6 = np.load(os.path.join(_ROOT, sp.HAND_CHOSEN["N6"]["state"]))
    u6, v6 = d6["reference_all_u"], d6["reference_all_v"]
    t6 = wa.ArrayTiling(n_col=3, n_row=2)
    say(f"state N6: {u6.shape}, u_max = {float(np.max(np.hypot(u6, v6))):.4f}, "
        f"u_min = {float(np.min(u6)):.4f}")

    say("\n[0] controls")
    dt6 = sp.exchange_interval(u6, v6)
    report["controls"] = stage_controls(u6, v6, dt6, t6)

    blocks = {}
    say("\n[A] CS-7 N=6 -- the wake array's own domain")
    blocks["N6"] = score_geometry("N6", u6, v6, t6, space, march=True)

    if not args.skip_n12:
        say("\n[B] CS-7 N=12 -- the other hand-chosen tiling")
        d12 = np.load(os.path.join(_ROOT, sp.HAND_CHOSEN["N12"]["state"]))
        u12, v12 = d12["reference_all_u"], d12["reference_all_v"]
        t12 = wa.ArrayTiling(n_col=4, n_row=3)
        blocks["N12"] = score_geometry("N12", u12, v12, t12, space_ladder,
                                       march=False)

    say("\n[C] KNOWN -- one rotor, one wake, and the answer known in advance")
    solo = np.load(os.path.join(_ROOT, sp.HAND_CHOSEN["N6"]["solo"]))
    us, vs = solo["u"], solo["v"]
    # Where is the wake?  The row whose streamwise-mean velocity deficit is
    # deepest.  With one rotor the wake is a single streamwise band, so a y-cut
    # on that row runs ALONG the centreline and a y-cut far from it does not --
    # which is the decision `tier0-measurements` 10.2 said a criterion computed
    # from the operator alone cannot express, on a real flow rather than a blob.
    prof = (1.0 - us / max(float(np.max(us)), 1e-300)).mean(axis=1)
    wake_row = int(np.argmax(prof))
    # A fine two-window sweep that CROSSES the wake: this is the control, and a
    # placement pool centred on the hand-chosen cut would never reach row 53.
    sweep = [sp.RectDecomposition(nx=t6.nx, ny=t6.ny, y_cuts=(c,), halo=8)
             for c in range(24, t6.ny - 24 + 1, 8)]
    blocks["SOLO"] = score_geometry(
        "SOLO", us, vs, t6, space_solo, march=False,
        extra={"WAKE": {
            "budget": "two windows, one y-cut, swept across the whole domain",
            "note": "the KNOWN control: the answer is known before the run -- "
                    "do not cut along the wake centreline",
            "candidates": sweep}})
    blocks["SOLO"]["wake"] = {
        "deepest_deficit_row": wake_row,
        "u_min": float(us.min()), "u_max": float(us.max()),
        "row_profile_peak": float(prof.max()),
        "row_profile": prof.tolist(),
        "sweep_rows": [d.y_cuts[0] for d in sweep],
        "note": "the single rotor's wake runs streamwise along this row, so a "
                "y-cut placed on it cuts ALONG the wake centreline and a y-cut "
                "far from it does not. The known answer is: do not cut here",
    }
    say(f"  C wake centreline at row {wake_row} of {us.shape[0]}; "
        f"u in [{us.min():.4f}, {us.max():.4f}]")
    # the control's verdict, read directly off the sweep
    sw = {s.dec.y_cuts[0]: s for s in blocks["SOLO"]["_scores"]
          if s.dec.n_windows == 2 and s.dec.y_cuts and not s.dec.x_cuts}
    if sw:
        rows = sorted(sw)
        on = min(rows, key=lambda r: abs(r - wake_row))
        off = max(rows, key=lambda r: abs(r - wake_row))
        blocks["SOLO"]["KNOWN"] = {
            "wake_row": wake_row,
            "sweep": [{"y_cut": r, "distance_from_wake": abs(r - wake_row),
                       "defect": sw[r].defect["defect"],
                       "Q_hat": sw[r].defect["Q_hat_reference_free"],
                       "Q_ai_inference__max":
                           sw[r].aggregate.get("Q_ai_inference__max"),
                       "off_diagonal_mass__max":
                           sw[r].aggregate.get("off_diagonal_mass__max")}
                      for r in rows],
            "nearest_cut_to_wake": on,
            "defect_cutting_on_the_wake": sw[on].defect["defect"],
            "farthest_cut": off,
            "defect_cutting_away": sw[off].defect["defect"],
            "on_over_away": sw[on].defect["defect"] / max(
                sw[off].defect["defect"], 1e-300),
            "argmin_defect_row": min(rows, key=lambda r: sw[r].defect["defect"]),
            "argmin_Q_row": min(
                rows, key=lambda r: sw[r].aggregate.get(
                    "Q_ai_inference__max", np.inf)),
            "argmin_Q_hat_row": min(
                rows, key=lambda r: sw[r].defect["Q_hat_reference_free"]),
            "distance_of_defect_argmin_from_wake": None,
        }
        k = blocks["SOLO"]["KNOWN"]
        k["distance_of_defect_argmin_from_wake"] = abs(
            k["argmin_defect_row"] - wake_row)
        say(f"  C KNOWN: defect argmin at row {k['argmin_defect_row']} "
            f"({k['distance_of_defect_argmin_from_wake']} rows from the wake); "
            f"Q argmin at {k['argmin_Q_row']}, Qhat argmin at "
            f"{k['argmin_Q_hat_row']}")

    say("\n[D] BASIS -- Q's orbit under an admissible re-declaration")
    report["basis"] = stage_basis(blocks)
    b = report["basis"]
    say(f"  D orbit ratio over {b['n_seams_measured']} real seams: "
        f"min {b['orbit_ratio_min']:.1f}x, median {b['orbit_ratio_median']:.1f}x, "
        f"max {b['orbit_ratio_max']:.1f}x; beta invariant to "
        f"{b['beta_invariance_worst']:.2e}")

    say(f"\n[E] HORIZON -- re-rank after {HORIZON} exchange intervals")
    report["horizon"] = stage_horizon(blocks)
    for tag, h in report["horizon"].items():
        say(f"  E {tag}: one-interval vs {HORIZON}-interval defect ranks at "
            f"{h['one_interval_vs_marched']}")

    # disagreements, per geometry
    report["disagreements"] = {t: disagreement_table(b) for t, b in blocks.items()}

    if not args.skip_shell:
        say("\n[F] CS-9 -- the shared thermal-strain domain")
        report["shell"] = stage_shell(args.quick)

    for tag, block in blocks.items():
        block.pop("_scores", None)
    report["geometries"] = blocks
    report["wall_seconds"] = time.perf_counter() - t_start

    path = os.path.join(args.out, "w112.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_j)
    say(f"\nwrote {path}  ({report['wall_seconds']:.0f} s)")

    # -- the headline ----------------------------------------------------
    say("\n" + "=" * 72)
    for tag, block in blocks.items():
        say(f"\n{tag}: hand-chosen {block['hand_chosen']['label']}, "
            f"{block['n_candidates_scored']} candidates scored")
        for key in sorted(block["searches"]):
            r = block["searches"][key]
            say(f"  {key:44s} chose {r['chosen']:26s} "
                f"defect {r['chosen_defect']:.4e}  "
                f"vs hand {r.get('chosen_over_hand_chosen', float('nan')):.4f}x  "
                f"vs best {r['penalty_vs_best']:.4f}x  "
                f"rank {r['rank_correlation_vs_defect']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
