"""W33 -- the conformance suite, run for the first time, on real experts.

`gap-worklist` Tier 6 has carried W33 as `in progress` since it was opened: *"the
conformance suite is built and every field's test is implemented; it has never
been run on a real expert."*  Its two blockers are both gone -- the probe is cheap
(W2) and the flux convention is pinned (W47, §2.2's form normative) -- and
[[tier0-measurements]] §9.8 records that *"the blocker is gone and the work is not
done"*.

This is the work.  Five real records go through `run_conformance`:

    WindowNS   embedded   the agent as built; R10 refuses its decomposition
    WindowNS   exposed    the split-step agent, the one that reaches `admit`
    ChannelNS  exposed    W55's second discretization
    Poseidon-T            W55's third expert, a frozen neural operator
    SpectralNS            the zero-channel control, whose Lambda is bitwise zero

plus **one record that lies**, because a suite that only ever sees true
declarations has not been shown to catch a false one.  `plug-in-composition-
theorems` §3's whole argument is that *"a false label produces a guarantee that
does not hold, with no error and no failed gate"*, and the definition of done on
the W33 row is that such a record is **refused at admission**.

Run:  python scripts/w33_conformance.py [--out out/w33]
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

from atlas.capability import BCChannel, ExpertCapabilities        # noqa: E402
from atlas.cases import channel_ns as C                           # noqa: E402
from atlas.cases import window_ns as W                            # noqa: E402
from atlas.conformance import run_conformance                     # noqa: E402
from atlas.probe import ProbeBudget                               # noqa: E402
from atlas.transfer import InterfaceSpace                         # noqa: E402
from atlas.verdict import ADMIT, ADMIT_UNCERTIFIED, REFUSE        # noqa: E402


def _j(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _space(caps, n_cells):
    """The interface space the suite probes on: the record's own declaration."""
    port = caps.ports[0]
    pro = port.prolongation
    return InterfaceSpace(seam_id="conformance", dim=pro.matrix.shape[1],
                          note="the record's own declared prolongation"), pro


def seam_defect_of(agent, faces=("xhi", "xlo"), m_eff=None, eps=1e-6):
    """The assembled seam's passivity defect: pi of S = Lambda_A + Lambda_B.

    **W48/W63.** The conformance verdict on `storage` belongs to the seam, and a
    seam involves two blocks, so it cannot be computed inside `run_conformance`
    -- which certifies one record. It is computed here and passed in.

    Both blocks are taken from the SAME agent on opposite faces, which is what a
    self-similar tiling's seam is and is how §9.5 and §11.4 measured theirs.
    """
    import numpy as _np

    from atlas.cases.window_ns import fourier_basis as _fb

    m_eff = m_eff or W.M_EFF
    n = agent.n
    B = _fb(n, m_eff)
    blocks = []
    for face in faces:
        port = f"{face}:MECH"
        z = agent.respond(port, _np.zeros(n))
        cols = [(agent.respond(port, eps * B[:, k]) - z) / eps for k in range(m_eff)]
        blocks.append(W.H * (B.T @ _np.column_stack(cols)))
    S = blocks[0] + blocks[1]
    mu = float(_np.linalg.eigvalsh(0.5 * (S + S.T))[0])
    floor = 1e-12 * float(_np.linalg.norm(S))
    return max(0.0, -mu) if abs(mu) > floor else 0.0


def certify(name, caps, n_cells, probe_state, budget=None, **kw):
    space, pro = _space(caps, n_cells)
    cert = run_conformance(caps, space=space, prolongation=pro,
                           budget=budget or ProbeBudget(fd_step=1e-6),
                           probe_state=probe_state, **kw)
    rows = {t.field_name: t for t in cert.tests}
    say(f"  {name:24s} verdict={cert.verdict.value:18s} "
        + "  ".join(f"{k}={v.verdict.value[:6]}" for k, v in rows.items()
                    if k in ("bc_channel", "storage", "validity")))
    for t in cert.tests:
        if t.verdict is REFUSE:
            say(f"     REFUSE {t.field_name}: {t.message[:130]}")
    return cert


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "out", "w33"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    d = np.load(os.path.join(_ROOT, "out", "tier0b", "s0_state.npz"))
    u, v = d["u"], d["v"]
    tiling = W.DEFAULT_TILING
    n = tiling.n
    us, vs = tiling.cut(u), tiling.cut(v)
    state = "developed wake, t = 5, dt = 0.05, nu = 1/255"
    out = {"probe_state": state, "certificates": {}}

    say("W33 -- the conformance suite on real experts")

    # -- 1, 2: WindowNS, both modes ---------------------------------------
    for mode, expose in (("embedded", False), ("exposed", True)):
        ag = W.WindowAgent(agent_id=f"windowns-{mode}", u0=us[0], v0=vs[0],
                           shared_faces=("xhi", "xlo"), expose_elliptic=expose)
        caps = W.window_capabilities(ag)
        c = certify(f"WindowNS {mode}", caps, n, state,
                    seam_defect=seam_defect_of(ag))
        out["certificates"][f"windowns-{mode}"] = c.as_dict()

    # -- 3: ChannelNS ------------------------------------------------------
    ag = C.ChannelAgent(agent_id="channelns-exposed", u0=us[0], v0=vs[0],
                        shared_faces=("xhi", "xlo"), expose_elliptic=True)
    caps = C.channel_capabilities(ag)
    c = certify("ChannelNS exposed", caps, n, state, seam_defect=seam_defect_of(ag))
    out["certificates"]["channelns-exposed"] = c.as_dict()

    # -- 4: Poseidon-T -----------------------------------------------------
    try:
        from atlas.cases import poseidon as P                      # noqa: PLC0415

        pu = np.ascontiguousarray(u[:P.EXPERT_RES, :P.EXPERT_RES])
        pv = np.ascontiguousarray(v[:P.EXPERT_RES, :P.EXPERT_RES])
        pag = P.PoseidonAgent(agent_id="poseidon-t", u0=pu, v0=pv,
                              shared_faces=("xhi", "xlo"))
        pcaps = P.poseidon_capabilities(pag, reproducibility_floor=1e-4)
        c = certify("Poseidon-T", pcaps, P.EXPERT_RES, state,
                    budget=ProbeBudget(fd_step=3.16e-3),
                    seam_defect=seam_defect_of(pag, eps=3.16e-3))
        out["certificates"]["poseidon-t"] = c.as_dict()
    except Exception as exc:                                       # noqa: BLE001
        say(f"  Poseidon-T SKIPPED: {type(exc).__name__}: {exc}")
        out["certificates"]["poseidon-t"] = {"skipped": str(exc)}

    # -- 5: the zero-channel control, declared honestly --------------------
    #
    # `SpectralNS` has no boundary argument at all, so its record says
    # `bc_channel = none` and its Lambda is bitwise zero (§9.5). A suite that
    # passes this is agreeing that "no channel" is a true statement about it.
    zero_caps = _spectral_caps(u, v, honest=True)
    c = certify("SpectralNS (honest)", zero_caps, P_RES, state)
    out["certificates"]["spectralns-honest"] = c.as_dict()

    # -- 6: THE RECORD THAT LIES -------------------------------------------
    #
    # The same expert, the same weights, one field changed: `bc_channel` now
    # claims DIRICHLET. Nothing about the expert has changed and nothing about
    # the graph would notice -- which is precisely the failure mode
    # `plug-in-composition-theorems` §3 names, and the one the W33 row says must
    # be **refused at admission**.
    liar = _spectral_caps(u, v, honest=False)
    c = certify("SpectralNS (LIES: bc)", liar, P_RES, state)
    out["certificates"]["spectralns-lying"] = c.as_dict()
    bc = [t for t in c.tests if t.field_name == "bc_channel"][0]
    out["the_false_label_is_refused"] = bool(bc.verdict is REFUSE)
    say(f"  -> the false bc_channel label is refused at admission: "
        f"{out['the_false_label_is_refused']}")

    # -- the summary the row asked for -------------------------------------
    out["summary"] = _summary(out["certificates"])
    say("")
    say("  field-by-field, across every real expert:")
    for fld, counts in out["summary"]["by_field"].items():
        say(f"    {fld:16s} " + "  ".join(f"{k}={n_}" for k, n_ in counts.items()))

    path = os.path.join(a.out, "w33.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=_j)
    say(f"wrote {path}")
    return out


P_RES = 128


def _spectral_caps(u, v, honest=True):
    """`reference.SpectralNS` as an agent, with `bc_channel` told two ways.

    The expert is byte-identical between the two records. Only the label moves,
    which is the whole point: conformance is the only thing in the package that
    can tell them apart.
    """
    ref = W.load_reference()
    sol = ref.SpectralNS(nu=W.NU, length=P_RES * W.H, n=P_RES, cfl=0.4)
    u0 = np.ascontiguousarray(u[:P_RES, :P_RES])
    v0 = np.ascontiguousarray(v[:P_RES, :P_RES])
    h = P_RES * W.H / P_RES

    def respond(port_name, trace):
        trace = np.asarray(trace, dtype=float).reshape(-1)
        r_u = u0.copy()
        r_u[:, -1] = r_u[:, -1] + trace
        # SpectralNS.step takes NO boundary argument: the ring cannot be held,
        # and on a periodic window it is not even a boundary. This is what
        # `bc_channel = none` means, and the response below is what it produces.
        w = sol.step(r_u, v0, W.MACRO_DT)[0] if isinstance(
            sol.step(r_u, v0, W.MACRO_DT), tuple) else sol.step(r_u, v0, W.MACRO_DT)
        return W.NU * (w[:, -1] - w[:, -2]) / h

    from atlas.capability import (
        ClaimType, Differentiable, Direction, EllipticSubsolve, MotionClass,
        TimeDiscretization, port_decl, zero_response)

    return ExpertCapabilities(
        expert_id="spectralns" + ("" if honest else "-mislabelled"),
        ports=[port_decl(
            name="xhi:MECH", port_type=__import__(
                "atlas.ports", fromlist=["PortType"]).PortType.MECH,
            geometry="xhi", direction=Direction.BIDIRECTIONAL,
            nondim=dict(W.MECH_SCALES), effective_resolution=W.M_EFF,
            motion_class=MotionClass.STATIC,
            prolongation=W.face_prolongation("spectralns", "xhi:MECH", P_RES))],
        bc_channel=(BCChannel.NONE if honest else BCChannel.DIRICHLET),
        bc_time_varying=False,
        elliptic_subsolve=EllipticSubsolve.EMBEDDED,
        time_discretization=TimeDiscretization.EXPLICIT,
        differentiable=Differentiable.NONE,
        stencil_radius=2, substeps_per_macro_step=10,
        dt_native=W.MACRO_DT, L_native=P_RES * W.H,
        storage=lambda a=None, b=None: 0.5 * float(np.sum(u0**2 + v0**2)) * h**2,
        validity=lambda s=None, c=None: True,
        governing_family="incompressible-navier-stokes-2d",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash="spectralns-periodic",
        boundary_response=zero_response if honest else zero_response,
        reproducibility_floor=np.finfo(float).eps, deterministic=True,
        note="periodic spectral solver; no boundary channel exists (section 9.5)",
    )


def _summary(certs):
    by_field: dict[str, dict[str, int]] = {}
    for _name, c in certs.items():
        if "tests" not in c:
            continue
        for t in c["tests"]:
            d = by_field.setdefault(t["field"], {})
            d[t["verdict"]] = d.get(t["verdict"], 0) + 1
    return {"by_field": by_field, "n_experts": len(certs)}


if __name__ == "__main__":
    main()
