"""W177 -- what the substitution campaign has actually returned, censused.

`case-study-ladder-to-f1` section 7 names a stopping rule and a checkpoint at
which to stand on it:

    "If the substitution campaign returns `refuse` or `blind` at EVERY seam of
     EVERY case study, the composed model cannot be certified with learned
     experts at all ... it is worth naming a checkpoint at which it is declared
     rather than approached asymptotically: AFTER CS-12."

CS-12 is built and so are CS-13, CS-14 and CS-S1, so the checkpoint has arrived.
Standing on it needs the campaign's verdicts **counted**, and the figure in
circulation -- "nine of nine seams have refused Poseidon-T on R10" ([[gap-worklist]]
W166, [[log]]'s Tier 33 entry, and both demo READMEs) -- is quoted without the
condition it was measured under.  This counts them.

Two graphs, because the figure is about one and the campaign is about both:

  * `front_wing` (CS-12 + CS-10 + CS-9, nine seams) with the fluid expert
    RE-DESCRIBED three ways.  This is the demo's own beat 1 and it is where the
    nine comes from.
  * `poseidon-t-2x2`, the graph that actually holds the checkpoint, whose four
    agents ARE `PoseidonAgent` and whose responses are the live weights.

`refuse`, `admit-uncertified` and `admit` are counted separately and not rolled
together, because the stopping rule's trigger is `refuse` OR `blind` and section
7 says in terms that "`admit-uncertified` is not failure".

Outputs `out/w177/w177.json`.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import io
import json
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas import Budget, compile_scheme  # noqa: E402
from atlas.cases import front_wing as F  # noqa: E402
from atlas.cases import neural_interface as NI  # noqa: E402
from atlas.cases import poseidon as PO  # noqa: E402
from atlas.demo_frontwing import substitution as SUB  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "out", "w177")


def seam_tally(graph, res):
    """Per-seam refusals and decertifications, from the decision record."""
    seam_ids = [c.seam_id for c in graph.connections]
    out = {}
    for sid in seam_ids:
        ref, dec = [], []
        for d in res.decisions:
            if d.subject != sid:
                continue
            tag = f"{d.layer}/{d.rule}"
            if d.verdict.value == "refuse":
                ref.append(tag)
            elif d.verdict.value == "admit-uncertified":
                dec.append(tag)
        out[sid] = {"refusals": sorted(set(ref)),
                    "decertifications": sorted(set(dec)),
                    "colour": "red" if ref else ("amber" if dec else "green")}
    return out


def summarise(name, graph, res):
    seams = seam_tally(graph, res)
    n_red = sum(1 for m in seams.values() if m["colour"] == "red")
    n_amber = sum(1 for m in seams.values() if m["colour"] == "amber")
    n_green = sum(1 for m in seams.values() if m["colour"] == "green")
    row = {
        "name": name,
        "graph": graph.name,
        "n_seams": len(seams),
        "graph_verdict": res.verdict.value,
        "seams_refused": n_red,
        "seams_uncertified_only": n_amber,
        "seams_clean": n_green,
        "graph_level_refusals": sorted({f"{d.layer}/{d.rule}"
                                        for d in res.decisions.refusals}),
        "per_seam": seams,
    }
    print("  %-46s %-18s seams %d: %d red / %d amber / %d green"
          % (name[:46], res.verdict.value, len(seams), n_red, n_amber, n_green))
    if row["graph_level_refusals"]:
        print("      refusing rules anywhere in the record: "
              + ", ".join(row["graph_level_refusals"]))
    return row


def certificates(u_full, v_full, m_eff=PO.M_EFF, eps=1e-4):
    """One `SubstitutionCertificate` per seam per side, for the live swap.

    ``S_old`` is `window_ns.WindowAgent` with its elliptic part exposed -- the
    incumbent, and the arrangement `L2/R10` admits -- and ``S_new`` is
    `PoseidonAgent` on the same window at the same state.  ``beta`` is
    ``sigma_min`` of the ASSEMBLED two-sided block on the declared 16-mode
    Fourier interface space, which is the seam's own amplifier and the quantity
    section 4 of `master-error-bound` divides by.

    ``beta_min`` is passed as ``None`` deliberately: W81 derives it from
    ``eps_tol = min(tau, sigma)`` and this graph has neither, so passing a
    number would be importing another expert's constant to make a verdict
    appear.  The certificate then reports `visible_above` and `fails_above` --
    the two thresholds that WOULD make it a verdict -- which is the result.
    """
    import numpy as np

    from atlas.composition import certify_substitution
    from atlas.cases.window_ns import WindowAgent

    tiling = PO.DEFAULT_TILING
    sub = tiling.mono_n
    uf = np.asarray(u_full)[:sub, :sub]
    vf = np.asarray(v_full)[:sub, :sub]
    us, vs = tiling.cut(uf), tiling.cut(vf)
    ex = PO.load_expert()
    names = list(tiling.names)

    def blocks(k, face):
        u0 = np.ascontiguousarray(us[k])
        v0 = np.ascontiguousarray(vs[k])
        pa = PO.PoseidonAgent(agent_id=names[k], u0=u0, v0=v0,
                              shared_faces=(face,), expert=ex)
        wa = WindowAgent(agent_id=names[k], u0=u0, v0=v0, shared_faces=(face,),
                         dt=pa.dt, nu=pa.nu, expose_elliptic=True)
        new = PO.probe_columns(pa, u0, v0, pa.dt, face=face, eps=eps, m_eff=m_eff)
        B = PO.fourier_basis(PO.EXPERT_RES, m_eff)
        z = wa.respond(f"{face}:MECH", np.zeros(PO.EXPERT_RES))
        cols = [(wa.respond(f"{face}:MECH", eps * B[:, j]) - z) / eps
                for j in range(m_eff)]
        old = PO.H * (B.T @ np.column_stack(cols))
        return old, new, pa, wa

    idx = {n: i for i, n in enumerate(names)}
    rows = []
    for seam_id, ((a_name, a_face), (b_name, b_face)) in PO.SEAMS.items():
        oa, na, pa, wa = blocks(idx[a_name], a_face)
        ob, nb, _, _ = blocks(idx[b_name], b_face)
        S_old = oa + ob
        beta = float(np.linalg.svd(S_old, compute_uv=False)[-1])
        for who, old_i, new_i, other in [(a_name, oa, na, ob),
                                         (b_name, ob, nb, oa)]:
            cert = certify_substitution(
                agent_id=who,
                old_caps=wa_caps(wa), new_caps=PO.poseidon_capabilities(pa),
                S_old=old_i + other, S_new=new_i + other,
                beta=beta, beta_min=None,
                block_norm=float(np.linalg.norm(old_i, 2)),
            )
            row = {
                "seam": seam_id, "agent": who,
                "verdict": cert.verdict.value,
                "passes": cert.passes,
                "blind": cert.blind,
                "delta_norm": cert.delta_norm,
                "beta": cert.beta,
                "block_norm": cert.block_norm,
                "visible_above": cert.visible_above,
                "fails_above": cert.fails_above,
                "beta_min": cert.beta_min,
                "same_port_list": cert.same_port_list,
                # W76's own bound, measured rather than assumed: the largest
                # move ANY swap of this agent can make is ||S_i||, because a
                # replacement that ignores its boundary data entirely removes
                # the block and nothing else.  So ||Delta|| / ||S_i|| says how
                # close this particular replacement is to that null one.
                "new_block_norm": float(np.linalg.norm(new_i, 2)),
                "delta_over_block": (float(np.linalg.norm(old_i - new_i, 2))
                                     / float(np.linalg.norm(old_i, 2))),
                "new_over_old_block": (float(np.linalg.norm(new_i, 2))
                                       / float(np.linalg.norm(old_i, 2))),
            }
            # the verdict FUNCTION: sweep the tolerance the graph cannot supply.
            # The midpoint of (visible_above, fails_above) is included when that
            # window is non-empty and positive: it is the ONLY place an
            # informative `admit` can live, so a sweep that misses it would
            # report "never admits" for a swap that does.
            lo, hi = cert.visible_above, cert.fails_above
            grid = [0.0, 1e-6, 1e-4, 1e-2, 0.1, 0.5]
            if lo is not None and hi > lo and hi > 0.0:
                grid.append(0.5 * (max(lo, 0.0) + hi))
            sweep = []
            for bm in sorted(grid):
                c2 = certify_substitution(
                    agent_id=who, old_caps=wa_caps(wa),
                    new_caps=PO.poseidon_capabilities(pa),
                    S_old=old_i + other, S_new=new_i + other,
                    beta=beta, beta_min=bm,
                    block_norm=float(np.linalg.norm(old_i, 2)))
                sweep.append({"beta_min": bm, "verdict": c2.verdict.value,
                              "blind": c2.blind, "passes": c2.passes})
            row["beta_min_sweep"] = sweep
            rows.append(row)
            print("  %-5s %-4s  %-18s ||D|| %.4g  beta %.4g  vis>%.4g  "
                  "fails>%.4g  ||D||/||Si|| %.4f  ||Snew||/||Sold|| %.4f"
                  % (seam_id, who, row["verdict"],
                     row["delta_norm"], row["beta"],
                     row["visible_above"], row["fails_above"],
                     row["delta_over_block"], row["new_over_old_block"]))
    tally = {}
    for r in rows:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    print("  certificate tally:", tally)
    return {"rows": rows, "tally": tally,
            "beta_min": None,
            "why_no_beta_min": "eps_tol = min(tau, sigma) is the derivation W81 "
                               "gives, and this graph declares neither; both are "
                               "structurally unmeasurable for a checkpoint fixed "
                               "at 128x128, which has no same-class monolith"}


def window_resolution(a, b):
    """Is the informative-admit window wider than the probe's own disagreement?

    ``window_width`` is ``fails_above - visible_above`` -- the whole interval on
    which the certificate both passes and could have failed.  ``endpoint_shift``
    is how far those two endpoints move when the probe amplitude changes by
    ten.  A window narrower than the shift is not a verdict.
    """
    rows = []
    by_key = {(r["seam"], r["agent"]): r for r in b["rows"]}
    for r in a["rows"]:
        o = by_key.get((r["seam"], r["agent"]))
        if o is None:
            continue
        width = r["fails_above"] - r["visible_above"]
        shift = max(abs(r["visible_above"] - o["visible_above"]),
                    abs(r["fails_above"] - o["fails_above"]))
        rows.append({
            "seam": r["seam"], "agent": r["agent"],
            "window_width": width,
            "endpoint_shift": shift,
            "window_over_beta": width / r["beta"],
            "shift_over_width": (shift / width) if width > 0 else float("inf"),
            "admits_somewhere": bool(width > 0 and r["fails_above"] > 0),
            "resolvable": bool(width > 0 and r["fails_above"] > 0 and shift < width),
        })
    return {"rows": rows, "n_rows": len(rows),
            "n_admitting": sum(1 for r in rows if r["admits_somewhere"]),
            "n_resolvable": sum(1 for r in rows if r["resolvable"]),
            "note": "an informative admit exists only on (visible_above, "
                    "fails_above); a window narrower than the endpoint shift "
                    "between two probe amplitudes is the instrument, not a verdict"}


def wa_caps(wa):
    from atlas.cases.window_ns import window_capabilities

    return window_capabilities(wa)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    u, v = NI.load_state()
    results = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": []}

    # --- 1. front_wing, the graph the "nine" is counted on ---------------
    # Both shapes: `motion=False` is the configuration Tier 39 certified and
    # `motion=True` is the one the demo shows live.  Quoting one without the
    # other is how a nine gets into circulation in the first place.
    for motion, shape in [(False, "fixed shape"), (True, "riding")]:
        print("=== front_wing (CS-12), %s, fluid expert re-described" % shape)
        base, _ = F.build(u, v, motion=motion, tiling=F.DEFAULT_TILING)
        for c in SUB.CANDIDATES:
            g = SUB.patch_fluid(base, c.fields, F.DEFAULT_TILING)
            r = compile_scheme(g)
            row = summarise(f"{c.name} [{shape}]", g, r)
            row["candidate"] = f"{c.key}_{'fixed' if not motion else 'riding'}"
            row["shape"] = shape
            row["declaration"] = {k: getattr(x, "value", x)
                                  for k, x in (c.fields or {}).items()}
            row["declared_not_measured"] = bool(c.caveat and "DECLARED" in c.caveat)
            row["caveat"] = c.caveat
            row["is_incumbent"] = c.is_incumbent
            results["rows"].append(row)

    # --- 2. the graph that actually holds the checkpoint ------------------
    print("=== poseidon-t-2x2, the live weights in the subdomains")
    from atlas.capability import EllipticSubsolve

    for ell, lab in [(EllipticSubsolve.NONE, "as poseidon_capabilities defaults"),
                     (EllipticSubsolve.UNKNOWN, "elliptic_subsolve=unknown"),
                     (EllipticSubsolve.EMBEDDED,
                      "elliptic_subsolve=embedded [DECLARED, not measured -- W60]")]:
        g, _ = PO.build(u, v, elliptic=ell)
        r = compile_scheme(g, budget=Budget(allow_probe=False))
        row = summarise("poseidon-t-2x2, " + lab, g, r)
        row["candidate"] = "poseidon_live_" + ell.value
        row["declaration"] = {"elliptic_subsolve": ell.value}
        row["declared_not_measured"] = ell is EllipticSubsolve.EMBEDDED
        row["eps_tol"] = None if r.scheme is None else r.scheme.eps_tol
        row["unmeasured"] = list(r.unmeasured)
        results["rows"].append(row)

    # --- 2b. the campaign's OWN instrument, per seam ----------------------
    # Section 5 step 3 is `certify_substitution` at every seam at a declared
    # beta_min, and step 4 records refuse / admit / blind per seam.  A compile
    # verdict is not that instrument.  Build the pair the certificate wants --
    # the same 16-mode Fourier block for the classical incumbent and for the
    # checkpoint, on the same window and the same state -- and issue one per
    # seam per side.
    print("=== substitution certificates, the campaign's own instrument")
    results["certificates"] = certificates(u, v, eps=1e-4)

    # **Can the instrument resolve its own verdict?**  An informative `admit`
    # lives only on the interval (visible_above, fails_above), and those windows
    # come out 0.2% to 1.1% of beta wide.  The probe that produces them uses a
    # finite difference at `eps`, and W173 section 5.3 measured this checkpoint's
    # differencing floor rising as 1/eps.  So repeat the whole construction at a
    # ten-times-larger amplitude: if the window's ENDPOINTS move by more than the
    # window's WIDTH, the admit is the instrument's noise straddling a threshold
    # and not a verdict.
    print("=== the same certificates at eps = 1e-3 (probe-resolution control)")
    results["certificates_eps1e-3"] = certificates(u, v, eps=1e-3)
    results["window_resolution"] = window_resolution(
        results["certificates"], results["certificates_eps1e-3"])
    for r in results["window_resolution"]["rows"]:
        print("  %-5s %-5s  window %9.3g   endpoint shift %9.3g   resolvable %s"
              % (r["seam"], r["agent"], r["window_width"],
                 r["endpoint_shift"], r["resolvable"]))
    print("  resolvable windows: %d of %d"
          % (results["window_resolution"]["n_resolvable"],
             results["window_resolution"]["n_rows"]))

    # --- 3. the stopping rule's trigger, evaluated ------------------------
    honest = [r for r in results["rows"]
              if not r["declared_not_measured"] and not r.get("is_incumbent")]
    results["stopping_rule"] = {
        "trigger": "refuse or blind at EVERY seam of EVERY case study",
        "rows_counted": [r["candidate"] for r in honest],
        "excluded_declared_not_measured": [
            r["candidate"] for r in results["rows"] if r["declared_not_measured"]],
        "total_seams": sum(r["n_seams"] for r in honest),
        "seams_refused": sum(r["seams_refused"] for r in honest),
        "seams_uncertified_only": sum(r["seams_uncertified_only"] for r in honest),
        "seams_clean": sum(r["seams_clean"] for r in honest),
    }
    s = results["stopping_rule"]
    s["every_seam_refused"] = (s["seams_refused"] == s["total_seams"])
    print("=== stopping rule: %d of %d seams refused across the honestly "
          "declared rows -- trigger met: %s"
          % (s["seams_refused"], s["total_seams"], s["every_seam_refused"]))

    results["elapsed_seconds"] = time.perf_counter() - t0
    path = os.path.join(OUT, "w177.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("wrote", path, "in %.1f s" % results["elapsed_seconds"])


if __name__ == "__main__":
    main()
