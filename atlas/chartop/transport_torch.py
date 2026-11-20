"""The CO-T expert as a differentiable map, and its label-free loss (the coolant plan's
capacity probe and Phase 5; [[chart-operator-stage-0b-and-1-plan]] sections 5, 6 and 15).

`transport_model.ArmR` is the expert for the host, in numpy.  This module is the same
map in torch, batched over pieces of one resolution, so a network can be trained through
it.  Nothing here is trained; no pinned module is edited (`coarse` is only called).

**The operators, as images.**  A piece's interior block K_II couples each interior node
to its eight neighbours only, so it is nine coefficient images (`transport_coarse.stencil`
on the interior grid), applied by shifts; K_II^T by the transposed shifts.  The coarse
level is the pinned bilinear prolongation (`coarse.prolongation_torch`, `restrict`,
`prolong`), and A_c = P^T K_II P is factorised once per piece by LU (it is not
symmetric).  So, for a batch,

    C(X)  = X + P A_c^-1  P^T (F  - K_II X),
    C*(Y) = Y + P A_c^-T  P^T (F* - K_II^T Y),
    C_p(u) = u + P A_c^-1 P^T (b_I - K_II u),

each linear in its input with fixed matrices, so gradients pass through exactly.

**Arm R** (`ArmR`): the network's outputs, on the inputs and on the reversed inputs
(`transport_model.inputs`, b -> -b), give the trial and adjoint corrections and the
particular field's:

    H = C(H0 + c(N)),   G = C*(H0 + c(N_rev)),   u_p = C_p(u0 + c(N_p) * scale),

with H0, u0 the plain square's (`principal`), so zero heads are TL0 exactly.  Two
correction forms, ``form``:
- ``"bubble"``: c(N) = beta N, beta = 16 xi (1 - xi) eta (1 - eta) (CO-D's form, as
  `transport_model.ArmR` drafts it);
- ``"free"``: c(N) = N at the interior nodes.  An interior field never touches the
  traces, which are Q, so the traces are exact in either form.  Added because TL0's
  error on fast pieces lies in boundary layers one or two cells thick (measured on four
  pieces at n = 128 and Pe 150-194: 32-41% of its error energy in the first interior
  layer, about half within two), where the bubble is of order h and its network output
  must be of order 1/h.

**The loss is label-free.**  The compact labels of `transport_data` (Lambda = W^T K V and
chi) cannot see a trial mode's error: with exact adjoint modes, G^T K V~ = Lambda for
every V~ with exact traces (G^T K_II = -Q^T K_GI), and the coupled field inside a piece
is first order in the modes' error.  Full fields at n = 128 are about 8 MB a piece in
float32.  So the loss reads the residual, as CO-D's T26 energy needed no labels:

    l_k = || F_k - K_II H_k ||^2_{(A^0)^-1},

the dual norm of the plain square's operator A^0 at the piece's reaction, which two
type-I sine transforms diagonalise (`principal`): r^T (A^0)^-1 r = sum_pq (S R S)_pq^2 /
lambda_pq with S the orthonormal DST-I matrix.  Its minimiser is the exact mode (K_II is
invertible), and it is equivalent to the error in the diffusion norm with constants set
by the chart (A^0 against the piece's symmetric part) and by the flow (K against it,
which grows with the Peclet number and weights smooth errors more).  **[AI Inference]:**
that tilt is toward the low modes the coupled answer reads, the opposite of P6; check 5
of the plan's section 6 measures whether the loss ranks networks as the coupled error does.

`objective` is relative, per group, as CO-D's method 1: each group's residual over TL0's
on the same piece, trial, adjoint and particular, averaged with equal weight, so no group
can hide behind another's energy (P6).  ``weights`` (k x k, PSD) instead weights the
columns by port traces (Phase 5's loss), l = tr(R^T (A^0)^-1 R T).

**Errors, when labels exist** (`errors`): each group's error in the diffusion norm of
the piece's own symmetric part D (`TransportPiece.diffusion_matrix`), over TL0's: the
quantity the capacity gate CAP reads.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as Fn

from . import coarse as co
from . import fe
from . import ports as pt
from . import principal as pr
from . import transport as tr
from . import transport_coarse as tco
from . import transport_model as tm

FORMS = ("bubble", "free")
OFFSETS = tuple((dj, di) for dj in (-1, 0, 1) for di in (-1, 0, 1))


def groups(m: int = 16, kind: str = "sine-hat") -> dict:
    """{group: columns} over all 4m port columns (CO-T has no strips, so every column is
    learned): the corner hats, j = 1-3, j = 4-8 and j = 9-15."""
    cols = pt.basis(kind, m, m).columns
    out = {"hats": [k for k, c in enumerate(cols) if c.what == "vertex"],
           "j1-3": [k for k, c in enumerate(cols) if c.what == "side" and 1 <= c.j <= 3],
           "j4-8": [k for k, c in enumerate(cols) if c.what == "side" and 4 <= c.j <= 8],
           "j9-15": [k for k, c in enumerate(cols) if c.what == "side" and 9 <= c.j]}
    if sorted(sum(out.values(), [])) != list(range(len(cols))):
        raise ValueError("the groups do not partition the port columns")
    return out


def dst_matrix(n: int) -> np.ndarray:
    """The orthonormal type-I sine transform on the n-1 interior nodes (symmetric)."""
    i = np.arange(1, n)
    return math.sqrt(2.0 / n) * np.sin(np.outer(i, i) * math.pi / n)


# ---------------------------------------------------------------------------
# stencils
# ---------------------------------------------------------------------------


def interior_stencil(K, n: int) -> np.ndarray:
    """K_II's nine coefficient images [9, n-1, n-1] (OFFSETS order): entry [d, j, i] is
    K_II[g, g + dj (n-1) + di] for interior node g = (j, i), zero off the interior."""
    loop, inner = fe.grid(n)
    K = K.tocsr()
    st = tco.stencil(K[inner][:, inner], n - 2)
    return np.stack([st[o].reshape(n - 1, n - 1) for o in OFFSETS])


def _shift(X, dj: int, di: int):
    """out[..., j, i] = X[..., j + dj, i + di], zero off the grid."""
    N = X.shape[-1]
    Xp = Fn.pad(X, (1, 1, 1, 1))
    return Xp[..., 1 + dj:1 + dj + N, 1 + di:1 + di + N]


def apply_stencil(c, X):
    """K_II X for coefficient images c [B, 9, N, N] and fields X [B, k, N, N]."""
    out = None
    for d, (dj, di) in enumerate(OFFSETS):
        t = c[:, d:d + 1] * _shift(X, dj, di)
        out = t if out is None else out + t
    return out


def apply_stencil_t(c, X):
    """K_II^T X: (K^T X)[g] = sum_d c_d[g - d] X[g - d]."""
    out = None
    for d, (dj, di) in enumerate(OFFSETS):
        t = _shift(c[:, d:d + 1] * X, -dj, -di)
        out = t if out is None else out + t
    return out


def _img(a: np.ndarray, n: int) -> np.ndarray:
    """((n-1)^2, k) or ((n-1)^2,) interior values -> (k, n-1, n-1) or (n-1, n-1) images."""
    a = np.asarray(a, np.float64)
    if a.ndim == 1:
        return a.reshape(n - 1, n - 1)
    return a.T.reshape(a.shape[1], n - 1, n - 1)


# ---------------------------------------------------------------------------
# a batch of pieces
# ---------------------------------------------------------------------------


class PieceBatch:
    """The fixed operators of pieces of one resolution n, as float64 tensors, with the
    network's inputs (float32) and TL0's residuals per group.  ``labels``: an optional list
    of `transport_data.labels(..., full=True)` dicts, for `errors`."""

    def __init__(self, pieces: list, n_c: int = 16, m: int = 16, labels: list | None = None,
                 device: str = "cpu", reference: bool = True):
        if not pieces:
            raise ValueError("an empty batch")
        n = pieces[0].n
        if any(p.n != n for p in pieces):
            raise ValueError("a batch holds pieces of one resolution")
        n_c = min(n_c, n)
        co.check_sizes(n, n_c)
        self.n, self.n_c, self.m, self.B = n, n_c, m, len(pieces)
        self.device = torch.device(device)
        dev = self.device
        Q = pt.basis("sine-hat", n, m).Q
        self.k = Q.shape[1]
        loop, inner = fe.grid(n)
        f64 = dict(dtype=torch.float64, device=dev)
        Cs, Ds, Fs_, Fa_, bs, X0s, u0s, syms, LUs, pivs = [], [], [], [], [], [], [], [], [], []
        xs, xrs, ss, scs = [], [], [], []
        P = co.prolongation(n, n_c)
        for p in pieces:
            K = p.matrix().tocsr()
            Cs.append(interior_stencil(K, n))
            if labels is not None:
                Ds.append(interior_stencil(p.diffusion_matrix(), n))
            K_IG = K[inner][:, loop]
            K_GI = K[loop][:, inner]
            Fs_.append(_img(-np.asarray(K_IG @ Q), n))
            Fa_.append(_img(-np.asarray(K_GI.T @ Q), n))
            b_I = p.load()[inner]
            bs.append(_img(b_I, n))
            X0s.append(_img(pr.modes(n, Q, p.s0), n))
            u0s.append(_img(pr.solve(n, b_I, p.s0), n))
            syms.append(1.0 / pr.symbol(n, p.s0))
            Ac = (P.T @ K[inner][:, inner] @ P).toarray()
            LU, piv = torch.linalg.lu_factor(torch.tensor(Ac, **f64))
            LUs.append(LU)
            pivs.append(piv)
            x, s, sc = tm.inputs(p, False)
            xr, _sr, _ = tm.inputs(p, True)
            xs.append(x)
            xrs.append(xr)
            ss.append(s)
            scs.append(sc)
        T = lambda a: torch.tensor(np.stack(a), **f64)                    # noqa: E731
        self.C = T(Cs)
        self.D = T(Ds) if Ds else None
        self.F, self.Fa, self.b = T(Fs_), T(Fa_), T(bs)
        self.X0, self.u0 = T(X0s), T(u0s)
        self.inv_symbol = T(syms)
        self.LU = torch.stack(LUs)
        self.piv = torch.stack(pivs)
        self.p = co.prolongation_torch(n, n_c, dev)
        self.S = torch.tensor(dst_matrix(n), **f64)
        self.x = torch.tensor(np.stack(xs), dtype=torch.float32, device=dev)
        self.x_rev = torch.tensor(np.stack(xrs), dtype=torch.float32, device=dev)
        self.s = torch.tensor(np.stack(ss), dtype=torch.float32, device=dev)
        self.scale = torch.tensor(scs, **f64)
        self.beta = torch.tensor(tm.bubble(n)[1:n, 1:n], **f64)
        self.steady = torch.tensor([p.s0 == 0.0 for p in pieces], device=dev)
        self.flowing = torch.tensor([bool(p.flows) for p in pieces], device=dev)
        self.groups = groups(m)
        self.labels = None
        if labels is not None:
            self.labels = {"H": T([_img(l["H"], n) for l in labels]),
                           "G": T([_img(l["G"], n) for l in labels]),
                           "up": T([_img(l["up"], n) for l in labels])}
        self.res0 = self.err0 = self.cols0 = self.data = None
        if reference:                     # TL0's residuals (the objective) and errors (labels)
            with torch.no_grad():
                H, G, up = self.tl0()
                self.cols0 = self.residuals(H, G, up)
                self.res0 = self._by_group(*self.cols0)
                self.err0 = self.group_errors(H, G, up) if self.labels is not None else None
                self.data = (self.dual(self.F), self.dual(self.Fa), self.dual(self.b))

    # -- the coarse level ---------------------------------------------------------

    def _coarse(self, r, adjoint: bool = False):
        """P A_c^-1 P^T r (A_c^-T with ``adjoint``) for r [B, k, N, N] or [B, N, N]."""
        squeeze = r.dim() == 3
        if squeeze:
            r = r[:, None]
        B, k = r.shape[0], r.shape[1]
        mc = self.n_c - 1
        rc = co.restrict(r, self.p).reshape(B, k, mc * mc).transpose(1, 2)
        v = torch.linalg.lu_solve(self.LU, self.piv, rc, adjoint=adjoint)
        out = co.prolong(v.transpose(1, 2).reshape(B, k, mc, mc), self.p)
        return out[:, 0] if squeeze else out

    def correct(self, X):
        return X + self._coarse(self.F - apply_stencil(self.C, X))

    def correct_adjoint(self, Y):
        return Y + self._coarse(self.Fa - apply_stencil_t(self.C, Y), adjoint=True)

    def correct_particular(self, u):
        return u + self._coarse(self.b - apply_stencil(self.C, u[:, None])[:, 0])

    def tl0(self):
        """TL0 for CO-T (`transport_coarse.TL0TransportExpert`), on the batch."""
        return self.correct(self.X0), self.correct_adjoint(self.X0), self.correct_particular(self.u0)

    # -- residuals and errors -----------------------------------------------------

    def dual(self, R):
        """Per column, r^T (A^0)^-1 r for R [B, k, N, N] -> [B, k] (or [B, N, N] -> [B])."""
        squeeze = R.dim() == 3
        if squeeze:
            R = R[:, None]
        Rh = self.S @ R @ self.S
        out = (Rh * Rh * self.inv_symbol[:, None]).sum((-2, -1))
        return out[:, 0] if squeeze else out

    def residuals(self, H, G, up):
        """The dual-norm residual of every column: (trial [B, k], adjoint [B, k],
        particular [B])."""
        return (self.dual(self.F - apply_stencil(self.C, H)),
                self.dual(self.Fa - apply_stencil_t(self.C, G)),
                self.dual(self.b - apply_stencil(self.C, up[:, None])[:, 0]))

    def _by_group(self, t, a, p):
        cols = list(self.groups.values())
        trial = torch.stack([t[:, c].sum(1) for c in cols], 1)
        adj = torch.stack([a[:, c].sum(1) for c in cols], 1)
        return torch.cat([trial, adj, p[:, None]], 1)       # [B, 2 * groups + 1]

    def group_residuals(self, H, G, up):
        return self._by_group(*self.residuals(H, G, up))

    def group_errors(self, H, G, up):
        """Each group's squared error in the piece's diffusion norm (labels needed)."""
        L = self.labels
        en = lambda E: (E * apply_stencil(self.D, E)).sum((-2, -1))      # noqa: E731
        return self._by_group(en(H - L["H"]), en(G - L["G"]),
                              en((up - L["up"])[:, None])[:, 0])

    def column_names(self) -> list:
        g = list(self.groups)
        return [f"trial {x}" for x in g] + [f"adjoint {x}" for x in g] + ["particular"]


def relative(now, start, floor: float = 1e-30):
    """sqrt(now / start) per entry: the group's norm over its starting norm."""
    return torch.sqrt(now / torch.clamp(start, min=floor))


#: a term whose TL0 residual is below this fraction of its data's is one TL0 already solves
#: (a square piece of one material, say): it is left out of the loss and counted (CO-D's method 3)
EASY_FLOOR = 1e-20
#: the realigned loss's shares (check 5, 2026-11-19): the adjoint kept at a small weight
ADJOINT_SHARE = 0.1


def easy_terms(batch: PieceBatch, floor: float = EASY_FLOOR):
    """(trial, adjoint, particular) masks [B] of the terms TL0 already solves."""
    t0, a0, p0 = batch.cols0
    td, ad, pd = batch.data
    return (t0.sum(1) <= floor * td.sum(1), a0.sum(1) <= floor * ad.sum(1), p0 <= floor * torch.clamp(pd, min=1e-300))


def objective(batch: PieceBatch, H, G, up, weights=None, particular: float = 0.0,
              adjoint: float = ADJOINT_SHARE, floor: float = EASY_FLOOR):
    """The label-free loss, relative to TL0 on each piece.

    ``weights`` None: each of the nine groups' dual-norm residual over TL0's, the mean over
    groups and pieces (the capacity probe's loss).  Groups TL0 already solves are left out.

    ``weights`` given (the realigned loss, after check 5): ``{"flowing": w, "solid": w}`` or one
    w, k column weights from the training layouts' port traces (`trace_weights`).  On each
    piece, the trial term is sum_k w_k l_k / sum_k w_k l0_k, the adjoint term likewise, and the
    particular term l_p / l0_p; they are combined with the shares (1 - adjoint - particular,
    adjoint, particular), each term TL0 already solves left out and the shares renormalised;
    a piece with no term left is left out.  The mean over pieces."""
    if weights is None:
        now = batch.group_residuals(H, G, up)
        g = len(batch.groups)
        t0, a0, p0 = batch.cols0
        td, ad, pd = batch.data
        cols = list(batch.groups.values())
        den = torch.cat([torch.stack([t0[:, c].sum(1) for c in cols], 1), torch.stack([a0[:, c].sum(1) for c in cols], 1),
                         p0[:, None]], 1)
        dat = torch.cat([torch.stack([td[:, c].sum(1) for c in cols], 1), torch.stack([ad[:, c].sum(1) for c in cols], 1),
                         pd[:, None]], 1)
        keep = den > floor * torch.clamp(dat, min=1e-300)
        assert now.shape[1] == 2 * g + 1
        r = torch.where(keep, now / torch.clamp(den, min=1e-300), torch.zeros_like(now))
        n = keep.sum()
        return r.sum() / n if int(n) else (H * 0.0).sum()
    w = _piece_weights(batch, weights, H.dtype)
    t, a, p = batch.residuals(H, G, up)
    t0, a0, p0 = batch.cols0
    et, ea, ep = easy_terms(batch, floor)
    terms = [((w * t).sum(1) / torch.clamp((w * t0).sum(1), min=1e-300), ~et, 1.0 - adjoint - particular),
             ((w * a).sum(1) / torch.clamp((w * a0).sum(1), min=1e-300), ~ea, adjoint),
             (p / torch.clamp(p0, min=1e-300), ~ep, particular)]
    num = sum(torch.where(m, s * v, torch.zeros_like(v)) for v, m, s in terms)
    den = sum(m.to(H.dtype) * s for _v, m, s in terms)
    ok = den > 0
    if not bool(ok.any()):
        return (H * 0.0).sum()
    return (num[ok] / den[ok]).mean()


def _piece_weights(batch: PieceBatch, weights, dtype):
    if isinstance(weights, dict):
        wf = torch.as_tensor(np.asarray(weights["flowing"]), dtype=dtype, device=batch.device)
        ws = torch.as_tensor(np.asarray(weights["solid"]), dtype=dtype, device=batch.device)
        return torch.where(batch.flowing[:, None], wf[None], ws[None])
    w = torch.as_tensor(np.asarray(weights), dtype=dtype, device=batch.device)
    return w[None].expand(batch.B, -1)


def trace_weights(path: str | None = None) -> dict:
    """The realigned loss's column weights and particular share, from
    ``out/chartop/cot/phase3-trace-weights.json`` (`scripts/cot_phase3_traces.py`)."""
    import json
    import os
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "out", "chartop", "cot", "phase3-trace-weights.json")
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    return {"weights": {k: v["w"] for k, v in rec["weights"].items()},
            "particular": float(rec["particular_share"]["mean"])}


# ---------------------------------------------------------------------------
# arm R
# ---------------------------------------------------------------------------


class ArmR(torch.nn.Module):
    """Arm R as a differentiable map: the network's corrections on TL0's fine part, then
    the coarse correction (see the module's docstring).  With zero heads it is TL0."""

    def __init__(self, size: str = "1M", form: str = "free", seed: int = 0, kernel: bool = True):
        super().__init__()
        if form not in FORMS:
            raise ValueError(f"form {form!r}: one of {FORMS}")
        torch.manual_seed(seed)
        self.net = tm.network(size, "R")
        self.form, self.size, self.kernel = form, size, bool(kernel)

    def _kernel(self, batch: PieceBatch, y):
        """Gate K whatever the weights: on a steady piece the four hat corrections are made
        zero-sum, so the hats' sum stays the plain square's, the uniform temperature, and
        the coarse correction keeps it (Lemma B).  The exact modes need no such sum there
        (their hats also sum to 1), so nothing the exact answer needs is removed."""
        if not self.kernel or not bool(batch.steady.any()):
            return y
        hats = batch.groups["hats"]
        mean = y[:, hats].mean(1, keepdim=True)
        y = y.clone()
        y[:, hats] = y[:, hats] - batch.steady[:, None, None, None].to(y.dtype) * mean
        return y

    def raw(self, batch: PieceBatch, x=None, x_rev=None, s=None):
        """The network's outputs on the interior, float64: (trial, adjoint) [B, cout, N, N]."""
        x = batch.x if x is None else x
        x_rev = batch.x_rev if x_rev is None else x_rev
        s = batch.s if s is None else s
        y = self.net(torch.cat([x, x_rev]), torch.cat([s, s]))
        n = batch.n
        y = y[:, :, 1:n, 1:n].double()
        return y[:batch.B], y[batch.B:]

    def forward(self, batch: PieceBatch, x=None, x_rev=None, s=None):
        yt, ya = self.raw(batch, x, x_rev, s)
        yt, ya = self._kernel(batch, yt), ya
        k = batch.k
        c = (lambda y: batch.beta * y) if self.form == "bubble" else (lambda y: y)   # noqa: E731
        H = batch.correct(batch.X0 + c(yt[:, :k]))
        G = batch.correct_adjoint(batch.X0 + c(ya[:, :k]))
        up = batch.correct_particular(batch.u0 + c(yt[:, k]) * batch.scale[:, None, None])
        return H, G, up


class HostExpert:
    """A torch arm as an expert for the transport host (`transport_host.TransportLayout`):
    each piece's batch is built at the network's coarse size, n_c = min(``n_c``, n/2), and
    its outputs are returned as a `transport.TransportResponse`."""

    def __init__(self, arm: ArmR, n_c: int = 16, name: str | None = None):
        self.arm, self.n_c = arm, n_c
        self.name = name or f"arm-R({arm.form}, {arm.size})"
        self.envelope = None
        self.calls = 0

    def respond(self, problems, pb_of) -> list:
        out = []
        self.arm.eval()
        for p in problems:
            Q = pb_of(p.n).Q
            b = PieceBatch([p], n_c=min(self.n_c, p.n // 2), reference=False)
            if Q.shape != (4 * p.n, b.k) or not np.array_equal(Q, pt.basis("sine-hat", p.n, b.m).Q):
                raise ValueError("the host's port basis is not the arm's (sine-hat, 16 per side)")
            with torch.no_grad():
                H, G, up = self.arm(b)
            out.append(tr.TransportResponse(to_interior(H[0]), to_interior(G[0]), to_interior(up[0])))
            self.calls += 1
        return out


def to_interior(X: torch.Tensor) -> np.ndarray:
    """[k, N, N] or [N, N] images back to ((n-1)^2, k) or ((n-1)^2,) arrays."""
    a = X.detach().cpu().numpy()
    if a.ndim == 2:
        return a.ravel()
    return a.reshape(a.shape[0], -1).T


__all__ = ["ADJOINT_SHARE", "ArmR", "EASY_FLOOR", "FORMS", "HostExpert", "PieceBatch", "apply_stencil",
           "apply_stencil_t", "dst_matrix", "easy_terms", "groups", "interior_stencil", "objective", "relative",
           "to_interior", "trace_weights"]
