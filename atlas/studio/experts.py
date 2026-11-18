"""The experts a piece can be given (plan §5), each offered only when it can run here.

* **Classical**: the piece's exact response, solved (`host.ClassicalExpert`).
* **H2r**: the classical response shared by every square of one material at one
  resolution, step and source (`reuse.ReuseExpert`, arm H2r of `gates_squares`);
  admitted only on such squares.
* **TL0**: the two-level expert's untrained floor: the U form of `gates_hybrid.U_FORM`
  (its registered seed), its networks replaced by zeros, inside the coarse correction at
  ``n_c = 16`` with the corner treatment B, as the two-level evaluation builds its TL0
  (``scripts/chartop_evaluate_twolevel.py``), from `atlas.chartop` alone.
* **The trained networks**, from the registered evaluations' weights: TL-s0 and TL-s1
  (two-level), S′-s0 and S′-s1 (the forms' reused S′).  **Each checkpoint's sha256 is
  computed and compared with its evaluation record before it is offered**; a file that
  is missing or does not match is listed as withheld, with why.

NeuberNet is not a chart-operator expert and is never listed (it is unlicensed and
local-only; the plan's refusal table).  Every learned expert carries the CO-D card's
envelope (`data.ENVELOPE`): the host refuses a piece outside it, naming the quantity.
Nothing here trains or changes a network, and nothing under `atlas/chartop` is edited.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHARTOP = os.path.join(ROOT, "out", "chartop")

#: the trained networks the page offers: (key, label, record, name in it, checkpoint, loader)
LEARNED = (
    ("TL-s0", "TL-s0, the two-level network (seed 0)", "evaluation-twolevel-20261111-004223.json",
     "TL-s0", os.path.join("runs", "twolevel", "TL-s0", "best.pt"), "twolevel"),
    ("TL-s1", "TL-s1, the two-level network (seed 1)", "evaluation-twolevel-20261111-004223.json",
     "TL-s1", os.path.join("runs", "twolevel", "TL-s1", "best.pt"), "twolevel"),
    ("Sp-s0", "S′-s0, the forms' network (seed 0)", "evaluation-forms-20261019-205923.json",
     "Sp-s0", os.path.join("runs", "forms", "Sp-s0", "best.pt"), "form"),
    ("Sp-s1", "S′-s1, the forms' network (seed 1)", "evaluation-forms-20261019-205923.json",
     "Sp-s1", os.path.join("runs", "forms", "Sp-s1", "best.pt"), "form"),
)
N_C, CORNERS = 16, 8                  # gates_twolevel.N_C, gates_twolevel.CORNERS

_sha_cache: dict = {}


def sha256_file(path: str) -> str:
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _sha_cache:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        _sha_cache[key] = h.hexdigest()
    return _sha_cache[key]


def _torch_ok() -> str | None:
    try:
        import torch  # noqa: F401
        return None
    except Exception as exc:                       # torch is optional
        return f"torch is not installed here ({type(exc).__name__})"


def registry(chartop: str = CHARTOP) -> list[dict]:
    """Every expert, with whether it is offered and, when not, why."""
    out = [{"key": "classical", "label": "Classical", "kind": "classical", "offered": True,
            "tip": "The piece's exact response, solved: the floor every other expert is measured against"},
           {"key": "H2r", "label": "H2r", "kind": "reuse", "offered": True,
            "tip": "The classical response shared by every square of one material: one solve per class "
                   "of squares (arm H2r). Admitted only on squares of one material"}]
    no_torch = _torch_ok()
    out.append({"key": "TL0", "label": "TL0", "kind": "tl0", "offered": no_torch is None,
                "why": no_torch, "tip": "The two-level expert with its networks at zero: the untrained "
                "floor at n_c = 16, what a network must beat"})
    for key, label, record, name, rel, loader in LEARNED:
        row = {"key": key, "label": label, "kind": loader, "record": record, "offered": False}
        path = os.path.join(chartop, rel)
        rec_path = os.path.join(chartop, record)
        if no_torch:
            row["why"] = no_torch
        elif not os.path.isfile(path):
            row["why"] = f"its weights are not on this machine ({rel.replace(os.sep, '/')})"
        elif not os.path.isfile(rec_path):
            row["why"] = f"its record {record} is not on this machine"
        else:
            want = (json.load(open(rec_path, encoding="utf-8")).get("weights") or {}).get(name)
            got = sha256_file(path)
            row.update(path=path, sha256=got, recorded=want)
            if want is None:
                row["why"] = f"{record} records no weights for {name}"
            elif got != want:
                row["why"] = f"its weights' sha256 {got[:12]}… is not the one {record} records ({want[:12]}…)"
            else:
                row["offered"] = True
        row["tip"] = (f"{label}: weights checked against {record} (sha256 {row.get('sha256', '')[:12]}…)"
                      if row["offered"] else f"{label}: withheld, {row.get('why')}")
        out.append(row)
    return out


def is_square(p) -> bool:
    """A square piece of one material, as H2r's premise needs (an affine square chart)."""
    X = p.chart.X
    c = np.array([X[0, 0], X[0, -1], X[-1, -1], X[-1, 0]])
    e = np.roll(c, -1, axis=0) - c
    L = np.hypot(e[:, 0], e[:, 1])
    if np.ptp(L) > 1e-9 * L.max():
        return False
    dots = [abs(float(e[k] @ e[(k + 1) % 4])) for k in range(4)]
    if max(dots) > 1e-9 * L.max() ** 2:
        return False
    n = X.shape[0] - 1
    t = np.arange(n + 1) / n
    XI, ET = np.meshgrid(t, t)
    aff = c[0] + XI[..., None] * e[0] + ET[..., None] * (c[3] - c[0])
    if np.abs(X - aff).max() > 1e-9 * L.max():
        return False
    return bool(np.abs(p.log_kappa_hat).max() <= 1e-12)


def build(key: str, chartop: str = CHARTOP, rows: list | None = None):
    """The host expert for a key that is offered."""
    from atlas.chartop import host as hs
    if key == "classical":
        return hs.ClassicalExpert()
    if key == "H2r":
        from atlas.chartop.reuse import ReuseExpert
        return ReuseExpert(is_square)
    rows = rows or registry(chartop)
    row = next((r for r in rows if r["key"] == key), None)
    if row is None or not row["offered"]:
        raise ValueError(f"{key} is not offered here: {row.get('why') if row else 'no such expert'}")
    import torch
    from atlas.chartop import coarse as co
    from atlas.chartop import data
    from atlas.chartop.expert import LearnedExpert
    if key == "TL0":
        from atlas.chartop import forms as fm
        from atlas.chartop import gates_hybrid as GH
        from atlas.chartop import twolevel as tl
        cfg = dict(GH.U_FORM)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.pop("seed"))
            u = fm.KernelExpert(**cfg).eval()
        z = copy.deepcopy(u)
        # the networks' output shapes, read from one forward on a plain piece
        x = torch.zeros(1, 5, 65, 65)
        s = torch.zeros(1, 2)
        with torch.no_grad():
            z.net = tl._Zeros(int(u.net(x[:, :4], s).shape[1]))
            if getattr(u, "pnet", None) is not None:
                z.pnet = tl._Zeros(int(u.pnet(x, s).shape[1]))
        return co.CoarseCorrected(LearnedExpert(z.eval(), envelope=data.ENVELOPE), N_C,
                                  corners=co.CornerTreatment("B", frac=CORNERS))
    if row["kind"] == "twolevel":
        from atlas.chartop import twolevel as tl
        return tl.host_expert(tl.load(row["path"], "cpu"), envelope=data.ENVELOPE)
    from atlas.chartop import train as tr
    return LearnedExpert(tr.load_expert(row["path"], "cpu"), envelope=data.ENVELOPE)


class Mixed:
    """One expert per piece, as the host reads one expert: each group of pieces is
    answered by its own expert, in the pieces' order."""
    name = "mixed"
    envelope = None

    def __init__(self, assign: list[str], experts: dict):
        self.assign = assign
        self.experts = experts
        self.calls = 0

    def respond(self, problems, pb_of):
        out = [None] * len(problems)
        for key in dict.fromkeys(self.assign):
            idx = [i for i, k in enumerate(self.assign) if k == key]
            ex = self.experts[key]
            c0 = getattr(ex, "calls", 0)
            resp = ex.respond([problems[i] for i in idx], pb_of)
            self.calls += getattr(ex, "calls", 0) - c0 or len(idx)
            for i, r in zip(idx, resp):
                out[i] = r
        return out


def admitted(key: str, problem, envelope) -> str | None:
    """Why an expert is refused on a piece (its card, or H2r's premise), or None."""
    from atlas.chartop import certificate as cert
    if key == "classical":
        return None
    if key == "H2r":
        return None if is_square(problem) else "H2r needs a square of one material (reuse.py's premise)"
    m = cert.measure(problem.chart, problem.log_kappa_hat, None if problem.steady else problem.fo)
    r = cert.refusals(m, envelope)
    return "; ".join(r) if r else None


__all__ = ["LEARNED", "Mixed", "admitted", "build", "is_square", "registry", "sha256_file"]
