"""Phase 3 of the Stage 0 rework: the network on TL0's residual -- the two-level expert in
torch, its relative objective, and the ledger of what the network adds.

[[chart-operator-stage-0-rework]] sections 3.3-3.4 and the handoff's sections 5-6.

**The expert.**  `TwoLevelExpert` is S's form (`forms.KernelExpert`: split, bubble, strips,
the kernel projection) followed by the coarse correction ``C`` of `coarse`: the modes against
their traces, the particular field against the load, and on a step the level field against
``b_1 = (M 1)_I / s0``.  It adds no parameter, so S's weights load into it unchanged, and with
``n_c=None`` it is S bit for bit.  With zero heads (the U-Net's head is zero at
initialisation) it is TL0.  Two paths for ``C``:
- ``arrays=(a11, a12, a22, sigma)`` (a training batch's element arrays): torch, float64,
  differentiable (`coarse.apply_M`, `coarse.coarse_matrix_torch`, `coarse.TorchCoarseSpace`);
- ``ops`` (each example's ``(A_I, M_IG)``, as `expert.LearnedExpert` passes them): numpy
  per example (`coarse.CoarseSpace`), no gradient.  The two agree to rounding (tested).
`call` is `forms.call` for this expert (that function is pinned): it passes the arrays.

**The relative objective** (the handoff's method 1).  Per piece, over the learned columns
``L`` (the hats and ``j = 1-8``),

    L_i = (tr L~_theta - tr L) / (tr L~_0 - tr L)  +  (E_p(theta) + ||u_p||^2) / (E_p(0) + ||u_p||^2),

``tr L~`` the label-free energies of the trained answer's modes (T26), ``tr L`` the exact ones
(labels), the subscript 0 TL0's answer, and ``E_p(v) = v'Av - 2 b'v`` the particular field's
label-free energy, whose minimum is ``-||u_p||^2``.  Each denominator is TL0's error energy:
it does not depend on the weights, so the gradient is T26's label-free gradient divided by
it, the exact fields stay the minimiser, and "no help" reads exactly 1 per term.  A term
whose TL0 error is below `EASY` of the piece's own energy has nothing to learn: it is left
out (method 3) and counted.

**The ledger** (the handoff's section 5), pure functions of measured energies:
``rho = 1 - eps_theta / eps_0`` per piece and column group (the network's share of the error
TL0 leaves; 0 no help, negative harm, 1 exact), with the guard; the coupled attribution
between U, TL0 and TL against the floor; the equivalent coarse size ``n_c*``; and
``kappa_kill``, the fraction of the network's raw contribution that ``C`` discards.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from . import coarse as co
from . import fe
from . import forms as fm
from . import model as md
from . import ports as pt

#: a piece (or column group) whose TL0 error energy is below this fraction of its own exact
#: energy has nothing to learn: it is counted, never divided by
EASY = 1e-8
#: the coupled attribution: TL0 within this (relative energy, or the rounding if larger) of
#: the floor leaves the network nothing to do
AT_FLOOR = 1e-4
#: the dead-network stop: the validation median rho below this after `DEAD_AFTER` of the steps
DEAD_RHO, DEAD_AFTER = 0.05, 0.2
GROUPS = ("hats", "j1-3", "j4-8")


def learned_groups(kind: str = "sine-hat", m: int = 16) -> dict:
    """{group: columns} over the learned columns: the hats, ``j = 1-3`` and ``j = 4-8``."""
    cols = pt.basis(kind, m, m).columns
    out = {"hats": [k for k, c in enumerate(cols) if c.what == "vertex"],
           "j1-3": [k for k, c in enumerate(cols) if c.what == "side" and 1 <= c.j <= 3],
           "j4-8": [k for k, c in enumerate(cols) if c.what == "side" and 4 <= c.j <= 8]}
    learned = set(fm.learned_columns(kind, m))
    if set(sum(out.values(), [])) != learned:
        raise ValueError("the groups do not partition the learned columns")
    return out


# ---------------------------------------------------------------------------
# the expert
# ---------------------------------------------------------------------------


class TwoLevelExpert(fm.KernelExpert):
    """S's form, then ``C`` at coarse size ``n_c`` (``None``: S itself).  Kept after a call:
    ``last_uncorrected`` (the form's answer before ``C``: modes, particular field, level
    field), ``last_level`` (the corrected level field), and, on the torch path,
    ``last_coarse`` (the batch's `coarse.TorchCoarseSpace`) and ``last_arrays``."""

    def __init__(self, form: str = "split", backbone: str = "unet", size: str = "1M",
                 kind: str = "sine-hat", m: int = 16, p_size: str | None = None,
                 bubble: str = "square", strips: bool = False, hats: str = "learned",
                 n_c: int | None = None):
        super().__init__(form, backbone, size, kind, m, p_size, bubble, strips, hats)
        if n_c is not None and (n_c < 2 or n_c & (n_c - 1)):
            raise ValueError(f"n_c {n_c!r}: a power of two, at least 2")
        self.n_c = n_c

    @property
    def needs_operator(self) -> bool:
        return super().needs_operator or self.n_c is not None

    def config(self) -> dict:
        c = super().config()
        c["n_c"] = self.n_c
        return c

    def forward(self, x, s, s0, w=None, g=None, ops=None, strip_solve: bool = True, arrays=None):
        form_ops = ops if (self.strips or self.hats == "classical") else None
        H, up = super().forward(x, s, s0, w, g, form_ops, strip_solve)
        self.last_uncorrected = (H.detach(), up.detach(), self.last_level)
        if self.n_c is None:
            return H, up
        n = x.shape[-1] - 1
        if self.n_c > n:
            raise ValueError(f"n_c = {self.n_c} on a grid of n = {n}")
        s0 = s0.to(torch.float64)
        if arrays is not None:
            return self._torch_path(H, up, s0, w, g, arrays, n)
        if ops is None or len(ops) != x.shape[0]:
            raise ValueError("the coarse correction needs each example's arrays or operator")
        return self._numpy_path(H, up, s0, w, g, ops, n)

    def _load(self, w, g, like):
        if g is None:
            return torch.zeros_like(like)
        return md._load(w.to(torch.float64), g.to(torch.float64))[:, 1:-1, 1:-1]

    def _torch_path(self, H, up, s0, w, g, arrays, n):
        a11, a12, a22, sig = (a.to(torch.float64) for a in arrays)
        ps = md.plain_square(n, self.kind, self.m, str(H.device))
        tcs = co.TorchCoarseSpace(co.coarse_matrix_torch(a11, a12, a22, sig, self.n_c), n, self.n_c)
        Hc = tcs.correct(H, co.residual_torch(a11, a12, a22, sig, H, ps.QB))
        b = self._load(w, g, up)
        upc = tcs.correct(up, co.residual_torch(a11, a12, a22, sig, up[:, None], rhs=b[:, None])[:, 0])
        u1 = self.last_level
        b1, transient = co.level_rhs_torch(a11, a12, a22, sig, s0)
        u1c = tcs.correct(u1, co.residual_torch(a11, a12, a22, sig, u1[:, None], rhs=b1[:, None])[:, 0])
        self.last_level = torch.where(transient[:, None, None], u1c, u1).detach()
        self.last_coarse, self.last_arrays = tcs, (a11, a12, a22, sig)
        return Hc, upc

    def _numpy_path(self, H, up, s0, w, g, ops, n):
        Q = pt.basis(self.kind, n, self.m).Q
        K = H.shape[1]
        b = self._load(w, g, up).detach().cpu().numpy().reshape(H.shape[0], -1)
        Hn = H.detach().cpu().numpy()
        upn = up.detach().cpu().numpy()
        u1n = self.last_level.detach().cpu().numpy()
        outH, outp, out1 = np.empty_like(Hn), np.empty_like(upn), np.empty_like(u1n)
        for k, (A_I, M_IG) in enumerate(ops):
            A = A_I.tocsr()
            cs = co.CoarseSpace(A, n, self.n_c)
            X = Hn[k].reshape(K, -1).T
            outH[k] = cs.correct(X, -(A @ X) - np.asarray(M_IG @ Q)).T.reshape(K, n - 1, n - 1)
            xp = upn[k].reshape(-1)
            outp[k] = cs.correct(xp, b[k] - A @ xp).reshape(n - 1, n - 1)
            s0k = float(s0[k])
            if s0k > 0.0:
                b1 = (A @ np.ones(A.shape[0]) + np.asarray(M_IG @ np.ones(M_IG.shape[1]))) / s0k
                x1 = u1n[k].reshape(-1)
                out1[k] = cs.correct(x1, b1 - A @ x1).reshape(n - 1, n - 1)
            else:
                out1[k] = u1n[k]
        dev = H.device
        self.last_level = torch.tensor(out1, dtype=torch.float64, device=dev)
        self.last_coarse, self.last_arrays = None, None
        return (torch.tensor(outH, dtype=torch.float64, device=dev),
                torch.tensor(outp, dtype=torch.float64, device=dev))


def make(backbone: str = "unet", size: str = "1M", form: str = "split", kind: str = "sine-hat",
         m: int = 16, p_size: str | None = None, bubble: str = "corner", strips: bool = True,
         hats: str = "learned", n_c: int | None = None) -> TwoLevelExpert:
    return TwoLevelExpert(form, backbone, size, kind, m, p_size, bubble, strips, hats, n_c)


def from_kernel(expert: fm.KernelExpert, n_c: int | None) -> TwoLevelExpert:
    """A two-level expert with ``expert``'s configuration and weights (S's, for instance)."""
    c = expert.config()
    tl = TwoLevelExpert(c["form"], c["backbone"], c["size"], c["kind"], c["m"], c["p_size"],
                        c["bubble"], c["strips"], c["hats"], n_c)
    tl.load_state_dict(expert.state_dict(), strict=True)
    return tl.to(next(expert.parameters()).device)


class _Zeros(torch.nn.Module):
    """A network that answers zeros of its head's shape: TL0's networks, at no cost."""

    def __init__(self, cout: int):
        super().__init__()
        self.cout = cout

    def forward(self, x, s):
        return torch.zeros(x.shape[0], self.cout, x.shape[-2], x.shape[-1], dtype=torch.float32,
                           device=x.device)


def zero_heads_twin(expert: TwoLevelExpert) -> TwoLevelExpert:
    """TL0 in ``expert``'s configuration: its networks replaced by ones answering zeros, which
    is what zero heads answer (``H0 + beta 0 = H0`` bit for bit), without running a backbone.
    It holds no weights, so one twin serves a whole training run.  It is built under a
    forked random state, so making it draws nothing from anyone's seeding."""
    c = expert.config()
    with torch.random.fork_rng(devices=[]):
        twin = TwoLevelExpert(c["form"], c["backbone"], c["size"], c["kind"], c["m"], c["p_size"],
                              c["bubble"], c["strips"], c["hats"], c["n_c"])
    for name in ("net", "pnet"):
        net = getattr(expert, name)
        if net is None:
            continue
        if not hasattr(net, "head"):
            raise ValueError(f"{type(net).__name__} has no output head to stand in for")
        setattr(twin, name, _Zeros(net.head.out_channels))
    return twin.to(next(expert.parameters()).device).eval()


def call(expert, batch: dict, strip_solve: bool = True):
    """`forms.call` for the two-level expert: the batch's element arrays go to ``C``, and
    the strips' operators when they are solved (or the classical hats need them)."""
    if not isinstance(expert, TwoLevelExpert) or expert.n_c is None:
        return fm.call(expert, batch, strip_solve)
    need_ops = (expert.strips and strip_solve) or expert.hats == "classical"
    ops = fm.batch_operators(batch) if need_ops else None
    arrays = (batch["a11"], batch["a12"], batch["a22"], batch["sigma"])
    return expert(batch["inputs"], batch["scalars"], batch["s0"], w=batch["w"], g=batch["g"], ops=ops,
                  strip_solve=strip_solve, arrays=arrays)


def load(path: str, device: str = "cpu") -> TwoLevelExpert:
    st = torch.load(path, map_location=device, weights_only=False)
    c = st["config"]
    e = TwoLevelExpert(c["form"], c["backbone"], c["size"], c["kind"], c["m"], c.get("p_size"),
                       c.get("bubble", "square"), c.get("strips", False), c.get("hats", "learned"),
                       c.get("n_c"))
    e.load_state_dict(st["model"])
    return e.to(device).eval()


# ---------------------------------------------------------------------------
# energies per column and the relative objective
# ---------------------------------------------------------------------------


def column_energies(batch: dict, H, kind: str = "sine-hat", m: int = 16) -> torch.Tensor:
    """``Lambda~_kk`` [B, 4m]: the label-free energy of each mode's full field (its traces
    ``Q_k``), float64, differentiable."""
    n = H.shape[-1] + 1
    ps = md.plain_square(n, kind, m, str(H.device))
    return md.energy(batch["a11"], batch["a12"], batch["a22"], batch["sigma"], md.full(ps, H))


def particular_energy(batch: dict, up) -> torch.Tensor:
    """``E_p(u) = u'Au - 2 b'u`` [B], label-free; its minimum is ``-||u_p||^2``."""
    Up = md.full_p(up)
    return (md.energy(batch["a11"], batch["a12"], batch["a22"], batch["sigma"], Up)[:, 0]
            - 2 * md.mass_form(batch["w"], batch["g"], Up)[:, 0])


def exact_column_energies(batch: dict, labels: dict, kind: str = "sine-hat", m: int = 16) -> torch.Tensor:
    """``Lambda_kk`` [B, 4m] of the exact modes: the labels' ``lam`` when they carry it (the
    port matrix's diagonal, light enough for a training set), else from their modes ``H``."""
    if "lam" in labels:
        return labels["lam"].to(torch.float64)
    with torch.no_grad():
        return column_energies(batch, labels["H"], kind, m)


def reference(twin: TwoLevelExpert, batch: dict, labels: dict, strip_solve: bool = False,
              easy: float = EASY) -> dict:
    """TL0's error energies on a batch, the relative objective's denominators: per piece,
    over the learned columns (and per group) ``eps_0 = sum_k (Lambda~_kk(TL0) - Lambda_kk)``,
    and ``eps_p0 = E_p(TL0) + ||u_p||^2``; with the exact energies, and which terms are easy
    (below ``easy`` of the piece's own energy).  ``twin`` is `zero_heads_twin` of the trained
    expert, built once."""
    with torch.no_grad():
        H0, up0 = call(twin, batch, strip_solve)
        lam_t = column_energies(batch, H0, twin.kind, twin.m)
        lam = exact_column_energies(batch, labels, twin.kind, twin.m)
        ep0 = particular_energy(batch, up0) + labels["up_energy"]
    G = learned_groups(twin.kind, twin.m)
    L = sorted(sum(G.values(), []))
    eps0 = (lam_t - lam)[:, L].sum(-1)
    trace = labels["trace_Lambda"].to(torch.float64)
    up_energy = labels["up_energy"].to(torch.float64)
    return {"eps0": eps0, "eps0_groups": {g: (lam_t - lam)[:, c].sum(-1) for g, c in G.items()},
            "lam": lam, "lam_L": lam[:, L].sum(-1), "eps_p0": ep0, "up_energy": up_energy,
            "easy_H": eps0 < easy * trace,
            "easy_p": ep0 <= easy * torch.clamp(up_energy, min=0.0),
            "H0": H0, "up0": up0}


def relative_objective(batch: dict, H, up, ref: dict, kind: str = "sine-hat", m: int = 16) -> dict:
    """Per piece ``L_i`` (see the module's docstring) and its two terms; easy terms are
    zero and flagged.  ``ref`` is `reference` of the same batch."""
    L = sorted(sum(learned_groups(kind, m).values(), []))
    lam_t = column_energies(batch, H, kind, m)
    eps = lam_t[:, L].sum(-1) - ref["lam_L"]
    ep = particular_energy(batch, up) + ref["up_energy"]
    termH = torch.where(ref["easy_H"], torch.zeros_like(eps), eps / torch.where(ref["easy_H"], torch.ones_like(eps), ref["eps0"]))
    termP = torch.where(ref["easy_p"], torch.zeros_like(ep), ep / torch.where(ref["easy_p"], torch.ones_like(ep), ref["eps_p0"]))
    return {"loss": termH + termP, "term_H": termH, "term_p": termP, "eps": eps.detach(), "eps_p": ep.detach()}


# ---------------------------------------------------------------------------
# the ledger
# ---------------------------------------------------------------------------


def rho(eps_theta, eps_0, energy, easy: float = EASY) -> np.ndarray:
    """``1 - eps_theta / eps_0`` per piece, ``nan`` where ``eps_0 < easy * energy`` (nothing to
    learn: counted by the caller, never divided by)."""
    et, e0, en = (np.asarray(v, float) for v in (eps_theta, eps_0, energy))
    out = np.full(e0.shape, np.nan)
    ok = e0 >= easy * en
    out[ok] = 1.0 - et[ok] / e0[ok]
    return out


def summarise_rho(r) -> dict:
    """Median and the 10th and 90th percentiles over the pieces with something to learn,
    and how many had nothing."""
    r = np.asarray(r, float)
    ok = r[np.isfinite(r)]
    if not ok.size:
        return {"median": None, "p10": None, "p90": None, "pieces": 0, "nothing_to_learn": int(r.size)}
    return {"median": float(np.median(ok)), "p10": float(np.percentile(ok, 10)),
            "p90": float(np.percentile(ok, 90)), "pieces": int(ok.size),
            "nothing_to_learn": int(r.size - ok.size)}


def attribution(e_U: float, e_TL0: float, e_TL: float, e_floor: float, rounding: float = 0.0) -> dict:
    """The coupled gain, coarse level first and then the network, against the floor: the
    coarse share ``(e_U - e_TL0) / (e_U - e_floor)``, the network's ``(e_TL0 - e_TL) /
    (e_U - e_floor)``, and the network's share of what TL0 left ``(e_TL0 - e_TL) /
    (e_TL0 - e_floor)``.  With TL0 within ``max(AT_FLOOR, rounding)`` of the floor there is
    nothing for the network to do, and no ratio is formed."""
    out = {"order": "coarse first, then the network", "e_U": e_U, "e_TL0": e_TL0, "e_TL": e_TL,
           "e_floor": e_floor}
    gain = e_U - e_floor
    left = e_TL0 - e_floor
    if left < max(AT_FLOOR, rounding):
        out.update(at_floor=True, note="TL0 at the floor: nothing for the network to do",
                   coarse_share=(e_U - e_TL0) / gain if gain > max(AT_FLOOR, rounding) else None,
                   network_share=None, network_share_of_left=None)
        return out
    out.update(at_floor=False, coarse_share=(e_U - e_TL0) / gain, network_share=(e_TL0 - e_TL) / gain,
               network_share_of_left=(e_TL0 - e_TL) / left)
    return out


def equivalent_nc(n_cs, e_TL0s, e_TL: float) -> dict:
    """``n_c*``: where TL0's error, interpolated linearly in ``log e`` against ``log n_c``,
    equals TL's.  Outside the measured range it is extrapolated from the two nearest points
    and flagged; a TL0 curve that does not fall is refused."""
    x = np.log(np.asarray(n_cs, float))
    y = np.log(np.asarray(e_TL0s, float))
    if np.any(np.diff(x) <= 0) or np.any(np.diff(y) >= 0):
        raise ValueError("n_c must rise and TL0's error must fall along it")
    t = math.log(e_TL)
    if y[-1] <= t <= y[0]:
        k = int(np.searchsorted(-y, -t))
        k = min(max(k, 1), len(x) - 1)
        extrap = False
    else:
        k = 1 if t > y[0] else len(x) - 1
        extrap = True
    x0, x1, y0, y1 = x[k - 1], x[k], y[k - 1], y[k]
    xs = x0 + (t - y0) * (x1 - x0) / (y1 - y0)
    return {"n_c_star": float(math.exp(xs)), "extrapolated": extrap}


def kappa_kill(tl: TwoLevelExpert, delta) -> torch.Tensor:
    """``||(I - Q_c) delta||^2_A / ||delta||^2_A`` per piece for the network's raw contribution
    ``delta`` [B, K, n-1, n-1] (the form's answer less TL0's, before ``C``), summed over its
    columns: the fraction ``C`` discards.  Needs the torch path's last call (its arrays and
    coarse space); ``nan`` where ``delta`` is zero."""
    if tl.last_coarse is None:
        raise ValueError("kappa_kill reads the torch path's coarse space: call with arrays")
    a11, a12, a22, sig = tl.last_arrays
    B, K, m1 = delta.shape[0], delta.shape[1], delta.shape[-1]
    D = torch.zeros(B, K, m1 + 2, m1 + 2, dtype=torch.float64, device=delta.device)
    D[..., 1:-1, 1:-1] = delta
    AD = co.apply_M(a11, a12, a22, sig, D)[..., 1:-1, 1:-1]
    kept = tl.last_coarse.correction(AD)               # (I - Q_c) delta = P A_c^-1 P^T A delta
    num = (kept * AD).sum((-3, -2, -1))                # ||(I - Q_c) d||^2 = <(I - Q_c) d, A d>
    den = (delta * AD).sum((-3, -2, -1))
    return torch.where(den > 0, num / torch.where(den > 0, den, torch.ones_like(den)),
                       torch.full_like(den, float("nan")))


def dead(step: int, steps: int, median_rho) -> bool:
    """The dead-network stop: past `DEAD_AFTER` of the steps with a median ``rho`` under `DEAD_RHO`."""
    return step >= DEAD_AFTER * steps and median_rho is not None and median_rho < DEAD_RHO
