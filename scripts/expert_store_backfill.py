"""expert_store_backfill -- cards for every model trained before the expert store existed.

[[chart-operator-stage-0b-and-1-plan]] section 8: "Stage 0's eleven arms, the forms runs and
TL-s0/s1 are backfilled from their existing records.  A card says whether its weights are
present locally."  Read from the evaluation records and the runs' own ``arm.json``; nothing
is trained, evaluated or re-measured, and no number is retyped: each card quotes its
record.  Weights stay where training left them (``out/chartop/runs/``, git-ignored).

    python scripts/expert_store_backfill.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from atlas.chartop import expert_store as es  # noqa: E402

CHARTOP = os.path.join(HERE, "out", "chartop")


def _load(name: str) -> dict:
    return json.load(open(os.path.join(CHARTOP, name), encoding="utf-8"))


def _arm(rel: str) -> dict:
    p = os.path.join(HERE, rel)
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}


def _repo_at(pin: str) -> str | None:
    p = os.path.join(CHARTOP, pin)
    if not os.path.isfile(p):
        return None
    for line in open(p, encoding="utf-8"):
        if line.startswith("repository at"):
            return line.split()[2]
    return None


def _scalars(d: dict) -> dict:
    return {k: v for k, v in (d or {}).items() if isinstance(v, (bool, int, float, str)) or v is None}


CO_D = {"pde": "2-D linear diffusion (heat), steady and one backward-Euler step: "
               "rho c dT/dt = div(k grad T) + q, pulled back to the unit square",
        "pieces": "curved four-sided pieces, one chart each (Winslow, Coons or conformal), "
                  "bilinear elements on the reference grid (decision D7)",
        "ports": ["THERM"], "port_basis": "corner hats and 15 side sines per side (D3)",
        "envelope": {"sup_abs_mu": 0.65, "logJ_range": "ln 8", "grad_logJ_plus_grad_mu": 40.0,
                     "abs_log_kappa_hat": 2.3, "fourier_number": [0.01, 100], "steady": True}}


def stage0() -> list:
    ev = _load("evaluation-20261018-203833.json")
    commit = _repo_at("registered.txt")
    comp = ev.get("comparisons", {})
    role = {"B": "main", "Bp": "B', the comparisons' default", "C1": "C1", "C2-free": "C2",
            "C2-sup": "C2", "C3": "C3", "C4": "C4", "C5-0.3M": "C5", "C5-3M": "C5",
            "C6": "C6", "C7": "C7"}
    out = []
    for aid, sha in ev["weights"].items():
        arm = _arm(f"out/chartop/runs/{aid}/arm.json")
        gates = {k: v.get("met") for k, v in ev["gates"].items()} if aid == "B" else None
        r = role.get(aid, "")
        c = comp.get(r.split()[0]) if r and r[0] == "C" else None
        if aid == "B":
            verdict, summary = "failed", ("Stage 0's main arm: E1, E4 and KT not met; E3, SE1-SE3 "
                                          "and DUAL met (the guarantees held, the modes did not learn)")
        else:
            verdict = "comparison"
            summary = (f"comparison {r}: {c.get('verdict', c.get('choice'))}" if c else
                       f"comparison arm ({r})")
        out.append({
            "card": es.CARD_VERSION, "id": aid, "family": "CO-D", "class": "G0", "what": CO_D,
            "built": {"commit": commit, "module": "atlas/chartop/model.py, atlas/chartop/train.py",
                      "architecture": arm.get("config"), "training": arm.get("arm"),
                      "data": {"generator": "atlas/chartop/data.py", "split": (arm.get("arm") or {}).get("data"),
                               "seed_base": 1000000}, "seed": (arm.get("arm") or {}).get("seed")},
            "did": {"registration": "out/chartop/registered.txt (Stage 0, W355)",
                    "evaluation": "out/chartop/evaluation-20261018-203833.json",
                    "verdicts": gates, "comparison": c, "summary": summary,
                    "price": "E5 in the evaluation record: no arm faster at comparable accuracy"},
            "verdict": verdict,
            "weights": es.weights_status(f"out/chartop/runs/{aid}/best.pt", sha)})
    return out


def forms() -> list:
    ev = _load("evaluation-forms-20261019-205923.json")
    commit = _repo_at("registered-forms.txt")
    out = []
    for aid, sha in ev["weights"].items():
        arm = _arm(f"out/chartop/runs/forms/{aid}/arm.json")
        verd = {k: (v.get("runs", {}).get(aid) if isinstance(v.get("runs"), dict) else v.get("met"))
                for k, v in ev["gates"].items() if isinstance(v, dict) and "met" in v}
        is_w = aid.startswith("W")
        out.append({
            "card": es.CARD_VERSION, "id": aid, "family": "CO-D", "class": "G0", "what": CO_D,
            "built": {"commit": commit, "module": "atlas/chartop/forms.py, atlas/chartop/train.py",
                      "architecture": arm.get("config"), "training": arm.get("arm"),
                      "data": {"generator": "atlas/chartop/data.py", "split": (arm.get("arm") or {}).get("data"),
                               "seed_base": 1000000}, "seed": (arm.get("arm") or {}).get("seed")},
            "did": {"registration": "out/chartop/registered-forms.txt (W356-W358)",
                    "evaluation": "out/chartop/evaluation-forms-20261019-205923.json",
                    "verdicts": verd, "comparison": ev.get("CW") if is_w else None,
                    "summary": ("W358's control for comparison CW (a tie)" if is_w else
                                "the kernel-exact forms: K, D and L met; E1, E4 and KT not met"),
                    "price": "E5 in the evaluation record"},
            "verdict": "comparison" if is_w else "failed",
            "weights": es.weights_status(f"out/chartop/runs/forms/{aid}/best.pt", sha)})
    return out


def prior() -> list:
    ev = _load("evaluation-prior-20261107-172219.json")
    commit = _repo_at("registered-prior.txt")
    out = []
    for aid, sha in ev["weights"].items():
        arm = _arm(f"out/chartop/runs/prior/{aid}/arm.json")
        reg = aid.split("-")[0]
        g = ev["gates"].get(reg, {})
        verd = {k: v.get("met") for k, v in g.items() if isinstance(v, dict) and "met" in v}
        e4 = g.get("E4", {}).get("per_problem", {})
        out.append({
            "card": es.CARD_VERSION, "id": aid, "family": "CO-D", "class": "G0", "what": CO_D,
            "built": {"commit": commit, "module": "atlas/chartop/forms.py, atlas/chartop/train.py",
                      "architecture": arm.get("config"), "training": arm.get("arm"),
                      "data": {"generator": "atlas/chartop/data.py", "split": (arm.get("arm") or {}).get("data"),
                               "seed_base": 1000000}, "seed": (arm.get("arm") or {}).get("seed")},
            "did": {"registration": "out/chartop/registered-prior.txt",
                    "evaluation": "out/chartop/evaluation-prior-20261107-172219.json",
                    "verdicts": verd,
                    "summary": f"arm {reg}: E4 met on {sum(1 for v in e4.values() if v)} of {len(e4)} held-out "
                               "problems (registered arm, as evaluated)",
                    "price": "E5 in the evaluation record"},
            "verdict": "failed",
            "weights": es.weights_status(f"out/chartop/runs/prior/{aid}/best.pt", sha)})
    return out


def twolevel() -> list:
    ev = _load("evaluation-twolevel-20261111-004223.json")
    out = []
    for aid, sha in ev["weights"].items():
        arm = _arm(f"out/chartop/runs/twolevel/{aid}/arm.json")
        verd = {k: v.get("met") for k, v in ev["gates"].items() if isinstance(v, dict) and "met" in v}
        e4 = ev["gates"]["E4"].get("per_problem", {})
        card = {
            "card": es.CARD_VERSION, "id": aid, "family": "CO-D", "class": "G0", "what": CO_D,
            "built": {"commit": ev.get("code"), "module": "atlas/chartop/twolevel.py, atlas/chartop/coarse.py, "
                                                         "atlas/chartop/train.py",
                      "architecture": arm.get("config"), "training": arm.get("arm"),
                      "data": {"generator": "atlas/chartop/data.py --corners --light-labels",
                               "seed_base": 1000000, "count": 50000},
                      "seed": (ev.get("runs", {}).get(aid) or {}).get("seed")},
            "did": {"registration": "out/chartop/registered-twolevel.txt (" + str(ev.get("registration"))[:120] + "...)",
                    "evaluation": "out/chartop/evaluation-twolevel-20261111-004223.json",
                    "verdicts": verd, "nc": _scalars(ev["gates"].get("NC")),
                    "summary": f"two-level, coarse-exact: E4 met on {sum(1 for v in e4.values() if v)} of {len(e4)} "
                               "held-out problems and NC not met; the coarse level carries the coupled gain",
                    "price": {"record": "out/chartop/evaluation-twolevel-20261111-004223-price.json",
                              "pr": _scalars(ev["gates"].get("PR"))}},
            "verdict": "failed",
            "weights": es.weights_status(f"out/chartop/runs/twolevel/{aid}/best.pt", sha)}
        card["studio"] = {"key": aid, "label": f"{aid}, the two-level network (seed {card['built']['seed']})",
                          "loader": "twolevel", "record": "evaluation-twolevel-20261111-004223.json", "name": aid,
                          "checkpoint": f"runs/twolevel/{aid}/best.pt"}
        out.append(card)
    return out


def main() -> int:
    import datetime as _dt
    cards = stage0() + forms() + prior() + twolevel()
    when = _dt.date.today().isoformat()
    for c in cards:
        st = c["weights"]
        c["weights"] = {"path": st["path"], "sha256": st["sha256"]}
        c["present_when_written"] = {"date": when, "present": st["present"], "why": st.get("why")}
        if c["id"] in ("Sp-s0", "Sp-s1"):
            c["studio"] = {"key": c["id"], "label": f"S′-{c['id'][-2:]}, the forms' network (seed {c['id'][-1]})",
                           "loader": "form", "record": "evaluation-forms-20261019-205923.json", "name": c["id"],
                           "checkpoint": f"runs/forms/{c['id']}/best.pt"}
        es.write_card(c)
    cat = es.rebuild_catalog()
    here = [es.present(r)["present"] for r in cat["experts"]]
    print(f"{len(cat['experts'])} cards; weights present and matching here for {sum(here)}")
    for r, h in zip(cat["experts"], here):
        print(f"  {r['family']}/{r['id']:8s} {r['verdict']:10s} weights {'here' if h else 'NOT here'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
