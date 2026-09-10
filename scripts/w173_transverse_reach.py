"""W173, part 2 -- the TRANSVERSE reach: how far into the window a ring datum goes.

`w173_epsilon_halo.py` measures the ALONG-SEAM operator, because that is what
`probe.support_reach` measures and what W93 compared against the declaration:
poke one seam cell, see which seam cells move, 64 of them against a declared 2.

But the quantity `L2/R10/halo` actually compares the overlap against is the
distance a stale artificial-boundary datum travels *into the window* during one
macro-step -- the rule's own sentence is "an artificial boundary's influence
travels stencil_radius cells per internal sub-step".  That is TRANSVERSE, and no
page in this vault has measured it on a learned expert.

The two can differ.  A local stencil is isotropic and they agree; an operator
that mixes along the seam through attention and only diffuses transversely would
be global in one direction and compact in the other -- and that operator would
have a derivable epsilon-halo even though `support_reach` reads global.  So the
negative in part 1 is not complete until this is measured, and if this one
decays the epsilon-halo is back on.

Method: poke one ring cell, step once, and profile the whole field response
against distance from the poked face.  Two solves per row.

Outputs `out/w173/transverse.json`.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import io
import json
import sys
import time

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.cases import neural_interface as NI  # noqa: E402
from atlas.cases import poseidon as PO  # noqa: E402
from atlas.cases import window_ns as W  # noqa: E402
from scripts.w173_epsilon_halo import fit_laws, load_checkpoint  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "out", "w173")

#: face -> (axis, index of the ring, direction of increasing depth)
_DEPTH = {
    "xhi": ("x", -1, -1),
    "xlo": ("x", 0, +1),
    "yhi": ("y", -1, -1),
    "ylo": ("y", 0, +1),
}


def step_fields(agent, face, trace):
    """Run the agent's own step with the ring written, and return (u1, v1).

    Deliberately re-implements the two lines `respond` uses to write the ring
    rather than calling it, because `respond` throws the interior away and the
    interior is the whole subject here.  The ring-writing itself is taken from
    the agent so the two cannot disagree about what a trace means.
    """
    ring, _ = agent._ring_index(face)
    r_u, r_v = agent.u0.copy(), agent.v0.copy()
    if face.startswith("x"):
        r_u[ring] = r_u[ring] + trace
    else:
        r_v[ring] = r_v[ring] + trace
    if isinstance(agent, PO.PoseidonAgent):
        return agent.step(r_u, r_v)
    u1, v1 = agent._solver.step_batch(
        agent.u0[None], agent.v0[None], agent.dt,
        bc0=(r_u[None], r_v[None]), bc1=None)
    return u1[0], v1[0]


def transverse_profile(agent, face, j0, amplitude):
    """max and rms |delta| at each transverse depth d from the poked face."""
    n = agent.n
    zero = np.zeros(n)
    u0, v0 = step_fields(agent, face, zero)
    d = np.zeros(n)
    d[j0] = float(amplitude)
    u1, v1 = step_fields(agent, face, d)
    g = np.sqrt((u1 - u0) ** 2 + (v1 - v0) ** 2) / float(amplitude)
    axis, _, direction = _DEPTH[face]
    # depth index 0 is the poked face itself
    if axis == "x":
        lines = [g[:, -1 - k] if direction < 0 else g[:, k] for k in range(n)]
    else:
        lines = [g[-1 - k, :] if direction < 0 else g[k, :] for k in range(n)]
    mx = [float(np.max(line)) for line in lines]
    rms = [float(np.sqrt(np.mean(line ** 2))) for line in lines]
    nz = [int(np.count_nonzero(line)) for line in lines]
    reach = max((k for k, c in enumerate(nz) if c > 0), default=0)
    return {"face": face, "j0": int(j0), "amplitude": float(amplitude),
            "n": int(n), "max_by_depth": mx, "rms_by_depth": rms,
            "nonzero_by_depth": nz, "nonzero_reach": int(reach),
            "peak": float(mx[0]) if mx else 0.0}


def depth_star(profile, key, targets):
    """Smallest depth at which the response has fallen below each target.

    Relative to the response AT the face, which is the quantity a halo has to
    outrun: the halo exists so the stale datum's influence is below tolerance by
    the time it reaches the blended region.
    """
    y = np.asarray(profile[key], dtype=float)
    peak = float(y[0]) if y.size and y[0] > 0 else float(np.max(y))
    out = []
    for t in targets:
        thr = t * peak
        hit = next((int(k) for k in range(len(y)) if y[k] <= thr), None)
        out.append({"target_relative": float(t), "threshold": float(thr),
                    "depth_star": hit,
                    "trivial": hit is not None and hit >= len(y) - 1})
    return {"peak": peak, "rows": out}


def run_subject(label, agent, face, subject, global_by, declared, amps=(1e-2,),
                js=(64,)):
    rows = []
    for a in amps:
        for j0 in js:
            t0 = time.perf_counter()
            p = transverse_profile(agent, face, j0, a)
            p["seconds"] = time.perf_counter() - t0
            p["depth_star_max"] = depth_star(p, "max_by_depth",
                                             [1e-1, 1e-2, 1e-3, 1e-4, 1e-6])
            p["depth_star_rms"] = depth_star(p, "rms_by_depth",
                                             [1e-1, 1e-2, 1e-3, 1e-4, 1e-6])
            y = np.asarray(p["max_by_depth"], dtype=float)
            lim = min(len(y) - 1, agent.n // 2)
            p["decay_fit"] = fit_laws(np.arange(1, lim + 1), y[1:lim + 1])
            rows.append(p)
            ds = p["depth_star_max"]["rows"]
            print("  %-40s a=%-7g j0=%-3d reach %3d  d*(1e-2)=%-5s d*(1e-4)=%-5s"
                  % (label, a, j0, p["nonzero_reach"],
                     ds[1]["depth_star"], ds[3]["depth_star"]))
    return {"label": label, "subject": subject, "global_by": global_by,
            "declared_reach": declared, "agent": agent.agent_id, "face": face,
            "rows": rows}


def main():
    os.makedirs(OUT, exist_ok=True)
    t_all = time.perf_counter()
    u_full, v_full = NI.load_state()
    results = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "state": "out/tier0b/s0_state.npz, developed wake t = 5",
               "what": "transverse reach: how far a ring datum travels INTO the "
                       "window in one macro-step. R10/halo's own quantity.",
               "subjects": []}

    ptil = PO.DEFAULT_TILING
    wtil = NI.DEFAULT_TILING

    from scripts.w173_epsilon_halo import poseidon_subjects, window_subjects

    print("Poseidon-T")
    pag = poseidon_subjects(u_full, v_full, ptil)
    results["subjects"].append(run_subject(
        "poseidon-T P00 xhi", pag["P00"], "xhi", "poseidon-t", "architecture", 2,
        amps=(1.0, 1e-2, 1e-4), js=(32, 64, 96)))
    results["subjects"].append(run_subject(
        "poseidon-T P11 xlo", pag["P11"], "xlo", "poseidon-t", "architecture", 2,
        amps=(1e-2,), js=(64,)))

    print("Poseidon-B")
    try:
        exB = load_checkpoint("camlab-ethz/Poseidon-B")
        pagB = poseidon_subjects(u_full, v_full, ptil, expert=exB)
        results["subjects"].append(run_subject(
            "poseidon-B P00 xhi", pagB["P00"], "xhi", "poseidon-b",
            "architecture", 2, amps=(1e-2,), js=(32, 64, 96)))
    except Exception as exc:                                  # pragma: no cover
        results["poseidon_b"] = {"unavailable": repr(exc)}
        print("  unavailable:", exc)

    for expose, sub, gb in [
        (False, "windowns-as-built", "physics (embedded pressure solve)"),
        (True, "windowns-split-step", "nothing -- the elliptic part is exposed"),
    ]:
        print(sub)
        wag = window_subjects(u_full, v_full, wtil, expose)
        results["subjects"].append(run_subject(
            f"{sub} C00 xhi", wag["C00"], "xhi", sub, gb,
            NI.DOMAIN_OF_DEPENDENCE, amps=(1e-2,), js=(32, 64, 96)))
        results["subjects"].append(run_subject(
            f"{sub} C11 xlo", wag["C11"], "xlo", sub, gb,
            NI.DOMAIN_OF_DEPENDENCE, amps=(1e-2,), js=(64,)))

    results["elapsed_seconds"] = time.perf_counter() - t_all
    path = os.path.join(OUT, "transverse.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("wrote", path, "in %.1f s" % results["elapsed_seconds"])


if __name__ == "__main__":
    main()
