"""W137 -- is the fluid's substitution certificate at `wing_fsi`'s wetted seam blind?

W137 (Tier 26) says yes, by five orders: *"the fluid's block is 1.44e-5 of the
assembled operator, so `SubstitutionCertificate` cannot see any replacement of
the flow expert"*.  That sentence reads the block's SHARE of the operator,
``||S_F|| / ||S||``.  The certificate never reads the share.  It reads
``||S_i||`` against ``beta``, and ``beta`` is the SMALLEST singular value of the
assembled operator where ``||S||`` is the largest:

    blind  <=>  ||S_i|| < beta - beta_min

So the two readings can disagree, and at a seam whose other side is a stiff but
ill-conditioned structure they are expected to.  This script measures both, on
today's code, at W136's own settled field, for all three flux modes -- and runs
the certificate the campaign would actually issue for a NULL replacement of the
fluid (one that ignores its boundary data, removing its block and nothing else)
across a beta_min sweep, beside a replacement that under-responds by 10%.

**A confound is controlled rather than assumed away.**  W136 probed on
2026-09-04; W138 declared `effort_normal="STRUCT"` on this seam on 2026-09-08,
which negates the fluid's contribution to the assembled operator.  W136's beta is
therefore the unoriented sum's.  Both are reported.

**And the compiler's own reading is recorded**: `L4/block-share` on the `wet`
seam of the default graph, which states in terms whether the certificate is
blind there and below which beta_min.

Outputs `out/w137/w137.json`.  No rule is written.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atlas import compile_scheme  # noqa: E402
from atlas.cases import wing_fsi as W  # noqa: E402
from atlas.composition import SubstitutionCertificate  # noqa: E402
from scripts import w136_wing_fsi as S136  # noqa: E402

OUT = os.path.join(ROOT, "out", "w137")


def _rel(A, B):
    nb = float(np.linalg.norm(B))
    return float(np.linalg.norm(A - B) / nb) if nb else float(np.linalg.norm(A))


def _smin(A):
    return float(np.linalg.svd(A, compute_uv=False)[-1])


def _cert_row(label, delta, beta, beta_min, block, fluid):
    c = SubstitutionCertificate(agent_id=fluid, old_expert="window_ns (the incumbent)",
                                new_expert=label, delta_norm=float(delta),
                                beta=float(beta), beta_min=beta_min,
                                same_port_list=True, block_norm=float(block))
    return {"replacement": label, "beta_min": beta_min, "verdict": c.verdict.value,
            "passes": c.passes, "blind": c.blind, "visible_above": c.visible_above,
            "fails_above": c.fails_above, "decision_margin": c.decision_margin,
            "null_replacement_ratio": c.null_replacement_ratio,
            "block_over_beta": c.block_over_beta}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return getattr(x, "value", str(x))


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    u, v = S136._settled({}, os.path.join(ROOT, "out", "w136"))
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "state": "out/w136/settled.npz -- W136's own settled fixed-shape field",
           "modes": {}}
    with open(os.path.join(ROOT, "out", "w136", "w136.json"), encoding="utf-8") as fh:
        stored = json.load(fh)
    stored_os = stored.get("probe", {}).get("one_sided") or stored.get("one_sided")

    for mode in W.FLUX_MODES:
        g, _e = W.build(u, v, motion=False, flux_mode=mode)
        conn = g.connection("wet")
        op, _tr = S136._probe_seam(g, "wet", f"settled fixed-shape field, {mode}")
        names = list(op.blocks)
        fluid = [k for k in names if k != "STRUCT"][0]
        B = {k: np.asarray(op.blocks[k].S, dtype=float) for k in names}
        S = np.asarray(op.S, dtype=float)
        signs = {k: (1.0 if (not conn.effort_normal or k == conn.effort_normal) else -1.0)
                 for k in names}
        recon_plain = _rel(sum(B.values()), S)
        recon_signed = _rel(sum(signs[k] * B[k] for k in names), S)
        signed = recon_signed <= recon_plain
        contrib = {k: (signs[k] if signed else 1.0) * B[k] for k in names}
        nF = float(np.linalg.norm(B[fluid], 2))
        nS = float(np.linalg.norm(B["STRUCT"], 2))
        nAll = float(np.linalg.norm(S, 2))
        beta = _smin(S)
        beta_unoriented = _smin(B[fluid] + B["STRUCT"])
        beta_flipped = _smin(B["STRUCT"] - B[fluid])
        null_new = S - contrib[fluid]
        under_new = S - 0.1 * contrib[fluid]
        d_null = float(np.linalg.norm(null_new - S, 2))
        d_under = float(np.linalg.norm(under_new - S, 2))
        grid = [None, 0.0, 1e-3, 1e-1, 0.5 * beta, 0.9 * beta, 0.99 * beta]
        rows = [_cert_row("null replacement (ignores its boundary data)", d_null, beta,
                          bm, nF, fluid) for bm in grid]
        rows += [_cert_row("under-responds by 10%", d_under, beta, bm, nF, fluid)
                 for bm in grid]
        # the same null replacement against the UNORIENTED beta, as W136 had it
        rows_unor = [_cert_row("null replacement, unoriented beta (pre-W138)", nF,
                               beta_unoriented, bm, nF, fluid)
                     for bm in [None, 0.0, 1e-3, 1e-1, 0.5 * beta_unoriented]]
        res["modes"][mode] = {
            "fluid_agent": fluid,
            "effort_normal": conn.effort_normal,
            "assembled_is_signed_sum": bool(signed),
            "reconstruction_rel_error": {"plain_sum": recon_plain, "signed_sum": recon_signed},
            "op_beta": None if op.beta is None else float(op.beta),
            "beta_sigma_min": beta,
            "beta_unoriented": beta_unoriented,
            "beta_fluid_flipped": beta_flipped,
            "norm_fluid": nF, "norm_structure": nS, "norm_assembled": nAll,
            "share_fluid": nF / nAll,
            "block_over_beta_fluid": nF / beta,
            "block_over_beta_fluid_unoriented": nF / beta_unoriented,
            "structure_over_fluid": nS / nF,
            "certificates": rows, "certificates_unoriented": rows_unor,
            "w136_stored": None if not stored_os else stored_os.get(mode)}
        m = res["modes"][mode]
        print("%-10s share %.3g   ||S_F||/beta %.4g (oriented)  %.4g (unoriented)   "
              "null verdict at beta_min=0: %s"
              % (mode, m["share_fluid"], m["block_over_beta_fluid"],
                 m["block_over_beta_fluid_unoriented"],
                 next(r["verdict"] for r in rows if r["beta_min"] == 0.0)))

    # --- the compiler's own reading, on the default graph ------------------
    g, _e = W.build(u, v, motion=False)
    r = compile_scheme(g)
    bs = [d for d in r.decisions if d.rule == "block-share" and d.subject == "wet"]
    res["compile"] = {
        "flux_mode": "reaction (the default)",
        "graph_verdict": r.verdict.value,
        "block_share_on_wet": [_jsonable({k: v for k, v in vars(d).items()})
                               for d in bs]}
    for d in bs:
        print("compile L4/block-share on wet:", d.verdict.value, "--", d.message[:160])

    res["elapsed_seconds"] = time.perf_counter() - t0
    path = os.path.join(OUT, "w137.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(res), fh, indent=1)
    print("wrote", path, "in %.1f s" % res["elapsed_seconds"])


if __name__ == "__main__":
    main()
