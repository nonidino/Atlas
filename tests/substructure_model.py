"""A steady advection-diffusion torus, SUBSTRUCTURED into two non-overlapping blocks.

**A model problem for a rule, not a fixture and not a case study** --- the same
third category `strip_model.py` declares, and for the same reason.  It earns
nothing on Tier 0 and measures no constant of any real expert.  What it has is an
**exactly computable reference** and a **freely movable cut**, on the one branch
`strip_model.py` cannot reach: `strip_model` is overlapping with a partition of
unity, which is L2/C2's hypothesis, and **W57 is about the branch that has
neither**.

Why steady, and why that is the honest choice rather than a convenience
-----------------------------------------------------------------------

`master-error-bound` 4's sigma bound

    sigma <= (C_mu / beta) * ||Lambda - Lambda~|| * ||lambda*||

is derived for a scheme whose transmission error enters through an **interface
solve**, and its own 2026-08-28 box says applying it to an overlapping scheme is
"not conservative but uninformative" --- there is no interface solve there, so
``1/beta`` is an amplifier imported from a factorization that does not apply.
That box is also why section 10.2's falsification of the old cut score does
**not** transfer here: the measurements that inverted the ranking were taken on
the overlapping branch, where ``1/beta`` was never derived.

So the model problem has to be one where the interface equation is genuinely the
right condition.  For an unsteady explicit march it is not --- that is section 5's
whole finding, and R2b gates the rung on exactly this.  A **steady** problem
poses a boundary-value problem by construction, flux balance is its exact
interface condition, and ``1/beta`` is the amplifier by derivation rather than by
analogy.  Making the model unsteady-and-explicit would measure section 5 again
instead of W57.

    a . grad u  -  div(nu grad u)  +  c u  =  f        on the unit torus

centred second order, ``nu`` at faces by arithmetic mean.  The reaction ``c > 0``
makes the monolithic operator nonsingular on the torus, so "the exact answer" is
a single dense solve and not a projection.

The decomposition is the **exact algebraic Schur complement**, which matters
---------------------------------------------------------------------------

The interface ``Gamma`` is two whole rows of cells (a torus cut in two has two
seams).  The two interiors touch each other only through them, so

    A = [[A_II, A_IG], [A_GI, A_GG]],   S = A_GG - sum_i A_GI_i A_II_i^-1 A_I_iG

and ``S`` is a genuine **sum of per-agent contributions** --- one Dirichlet-to-
Neumann block per block, which is exactly the ``Lambda_i`` the package probes.
Nothing is approximated: with the FULL interface space this reproduces the
monolith to machine precision, and `test_substructuring_is_exact_at_full_resolution`
asserts it.  A ghost-cell formulation cannot do that, because a cell-centred
stencil's out-of-block neighbour is a *different cell* for each of the two sides
and one shared unknown per seam cannot be both.

Where the defect comes from
---------------------------

Substructuring with the full interface space is a direct method.  The defect this
model measures is therefore entirely the one this vault's architecture actually
creates: the interface space is **declared** at ``m`` modes with ``m < nx``, so
the trace the scheme can represent is a truncation of the trace the monolith has.
That is the same source section 12.2 found the cross-point defect in ("a property
of the declared interface space, not of the expert"), and it is cut-dependent,
which is what a cut criterion has to rank.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# the physics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SteadyPhysics:
    """Steady advection-diffusion-reaction on a torus.

    ``nu_y_amp`` is load-bearing for the same reason it is in `strip_model`: it
    makes two cut placements differ in the OPERATOR and not only in the solution.
    ``nu_x_amp`` keeps the seam operator off-diagonal in the declared Fourier
    basis --- a translation-invariant seam would make Fourier the eigenbasis and
    drive the truncation defect to something no criterion could be wrong about.
    """

    nx: int = 32
    ny: int = 32
    a_x: float = 0.6
    a_y: float = 0.35
    nu0: float = 6.0e-3
    nu_y_amp: float = 0.8
    nu_x_amp: float = 0.3
    nu_y_waves: float = 2.0
    c_reac: float = 1.0

    @property
    def hx(self) -> float:
        return 1.0 / self.nx

    @property
    def hy(self) -> float:
        return 1.0 / self.ny

    def nu_field(self) -> np.ndarray:
        y = (np.arange(self.ny) + 0.5) * self.hy
        x = (np.arange(self.nx) + 0.5) * self.hx
        fy = 1.0 + self.nu_y_amp * np.cos(2.0 * np.pi * self.nu_y_waves * y)
        fx = 1.0 + self.nu_x_amp * np.cos(2.0 * np.pi * x)
        return self.nu0 * np.outer(fy, fx)

    def source(self, seed: int = 0, k_max: int = 3) -> np.ndarray:
        """A smooth band-limited forcing, fixed per seed."""
        rng = np.random.default_rng(seed)
        y = (np.arange(self.ny) + 0.5) * self.hy
        x = (np.arange(self.nx) + 0.5) * self.hx
        f = np.zeros((self.ny, self.nx))
        for ky in range(-k_max, k_max + 1):
            for kx in range(-k_max, k_max + 1):
                if ky == 0 and kx == 0:
                    continue
                amp = rng.normal() / (1.0 + ky * ky + kx * kx)
                ph = rng.uniform(0.0, 2.0 * np.pi)
                f += amp * np.cos(2.0 * np.pi * (ky * y[:, None] + kx * x[None, :]) + ph)
        return f


def monolith_matrix(phys: SteadyPhysics) -> np.ndarray:
    """The dense (ny*nx, ny*nx) operator on the whole torus, row-major in (j, i)."""
    ny, nx, hx, hy = phys.ny, phys.nx, phys.hx, phys.hy
    nu = phys.nu_field()
    A = np.zeros((ny * nx, ny * nx))
    for j in range(ny):
        for i in range(nx):
            r = j * nx + i
            ip, im = (i + 1) % nx, (i - 1) % nx
            jp, jm = (j + 1) % ny, (j - 1) % ny
            nu_xp = 0.5 * (nu[j, i] + nu[j, ip])
            nu_xm = 0.5 * (nu[j, i] + nu[j, im])
            nu_yp = 0.5 * (nu[j, i] + nu[jp, i])
            nu_ym = 0.5 * (nu[j, i] + nu[jm, i])
            A[r, j * nx + ip] += -phys.a_x / (2.0 * hx) - nu_xp / hx**2
            A[r, j * nx + im] += +phys.a_x / (2.0 * hx) - nu_xm / hx**2
            A[r, jp * nx + i] += -phys.a_y / (2.0 * hy) - nu_yp / hy**2
            A[r, jm * nx + i] += +phys.a_y / (2.0 * hy) - nu_ym / hy**2
            A[r, r] += ((nu_xp + nu_xm) / hx**2 + (nu_yp + nu_ym) / hy**2
                        + phys.c_reac)
    return A


def monolith_solve(phys: SteadyPhysics, f: np.ndarray) -> np.ndarray:
    return np.linalg.solve(monolith_matrix(phys), f.ravel()).reshape(phys.ny, phys.nx)


def fourier_basis(n: int, m: int, h: float) -> np.ndarray:
    """(n, m) real Fourier modes, orthonormal in the h-weighted pairing.

    The same declared basis every case study in this vault uses.
    """
    length = n * h
    y = (np.arange(n) + 0.5) * h
    cols = [np.full(n, 1.0 / np.sqrt(length))]
    k = 1
    while len(cols) < m:
        w = 2.0 * np.pi * k * y / length
        cols.append(np.sqrt(2.0 / length) * np.cos(w))
        if len(cols) < m:
            cols.append(np.sqrt(2.0 / length) * np.sin(w))
        k += 1
    return np.column_stack(cols[:m])


# ---------------------------------------------------------------------------
# the decomposition -- NON-OVERLAPPING, two interiors, two interface rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubstructureDecomposition:
    """A torus cut in two, rotated by ``cut``, with the seams as interface rows.

    Interface rows are global ``cut`` and ``cut + ny/2``.  Interior 0 is the rows
    strictly between them going up; interior 1 is the rest.  ``cut`` is the only
    free parameter: block sizes, the agent and the declared interface dimension
    are identical for every value of it.

    There is **no overlap and no partition of unity**, which is exactly the
    branch L2/C2 does not cover.  Assembly is concatenation.
    """

    phys: SteadyPhysics = field(default_factory=SteadyPhysics)
    m_eff: int = 8
    cut: int = 0

    def __post_init__(self) -> None:
        if self.phys.ny % 4:
            raise ValueError("ny must be divisible by 4")
        if self.m_eff > self.phys.nx:
            raise ValueError("declared interface space larger than the face")

    @property
    def half(self) -> int:
        return self.phys.ny // 2

    def seam_rows(self) -> tuple[int, int]:
        ny = self.phys.ny
        return (self.cut % ny, (self.cut + self.half) % ny)

    def interior_rows(self, i: int) -> np.ndarray:
        """Global rows of interior ``i``, strictly between the two seam rows."""
        ny = self.phys.ny
        start = self.cut + i * self.half + 1
        return (np.arange(self.half - 1) + start) % ny

    def gamma_dofs(self) -> np.ndarray:
        """Global cell indices of the interface, seam 0 first then seam 1."""
        nx = self.phys.nx
        return np.concatenate([np.arange(nx) + r * nx for r in self.seam_rows()])

    def interior_dofs(self, i: int) -> np.ndarray:
        nx = self.phys.nx
        return np.concatenate([np.arange(nx) + r * nx for r in self.interior_rows(i)])

    def basis(self) -> np.ndarray:
        """Block-diagonal prolongation: ``m_eff`` modes on each of the two seams."""
        P1 = fourier_basis(self.phys.nx, self.m_eff, self.phys.hx)
        Z = np.zeros_like(P1)
        return np.block([[P1, Z], [Z, P1]])


# ---------------------------------------------------------------------------
# the exact Schur complement, as a sum of per-agent DtN blocks
# ---------------------------------------------------------------------------


def schur(dec: SubstructureDecomposition, f: np.ndarray) -> dict:
    """``S = A_GG - sum_i A_GI_i A_II_i^-1 A_I_iG`` and ``chi``, plus the pieces.

    The per-block term ``A_GI_i A_II_i^-1 A_I_iG`` **is** agent ``i``'s
    Dirichlet-to-Neumann operator: give it a trace on Gamma, it solves its own
    interior, and it returns the flux that trace costs.  That the assembled
    interface operator is a SUM of them is not a modelling choice here, it is the
    block factorization, which is why this model can test a rule about
    ``Lambda~ = sum_i P_i^* Lambda_i P_i`` rather than merely illustrate one.
    """
    A = monolith_matrix(dec.phys)
    g = dec.gamma_dofs()
    fv = f.ravel()
    S = A[np.ix_(g, g)].copy()
    chi = fv[g].copy()
    blocks, pieces = {}, {}
    for i in (0, 1):
        d = dec.interior_dofs(i)
        A_II = A[np.ix_(d, d)]
        A_GI = A[np.ix_(g, d)]
        A_IG = A[np.ix_(d, g)]
        X = np.linalg.solve(A_II, A_IG)          # A_II^-1 A_IG
        y = np.linalg.solve(A_II, fv[d])
        blocks[i] = A_GI @ X
        S -= blocks[i]
        chi -= A_GI @ y
        pieces[i] = dict(dofs=d, X=X, y=y)
    return dict(A=A, S=S, chi=chi, gamma=g, blocks=blocks, pieces=pieces)


def solve_substructured(dec: SubstructureDecomposition, sc: dict,
                        u_gamma: np.ndarray) -> np.ndarray:
    """Given the interface values, recover the interiors and concatenate."""
    out = np.zeros(dec.phys.ny * dec.phys.nx)
    out[sc["gamma"]] = u_gamma
    for i in (0, 1):
        p = sc["pieces"][i]
        out[p["dofs"]] = p["y"] - p["X"] @ u_gamma
    return out.reshape(dec.phys.ny, dec.phys.nx)


# ---------------------------------------------------------------------------
# W57 -- the derived criterion, in both of its forms
# ---------------------------------------------------------------------------


def w57_criterion(S_M: np.ndarray, chi_M: np.ndarray, a_star: np.ndarray,
                  S_full: np.ndarray | None = None,
                  u_gamma_star: np.ndarray | None = None,
                  P: np.ndarray | None = None) -> dict:
    """L2/C3, the substructuring cut criterion, and the two ways to measure it.

    From `master-error-bound` 4, with ``lam_dag`` solving ``S_M lam = chi_M`` on
    the declared space and ``a_star`` the exact trace's coordinates there,

        S_M (lam_dag - a_star) = chi_M - S_M a_star =: -r

    so

        ||lam_dag - a_star||  <=  ||r|| / beta        exactly, beta = sigma_min(S_M)

    with **no operator-mismatch step at all**.  That is the TIGHT form,

        Q_sub(Gamma)  =  ||S_M a_star - chi_M|| / beta                    (tight)

    Section 4's own product form is the same quantity after one further bound,
    ``||r|| <= ||Lambda - Lambda~|| ||lambda*||``:

        Q_sub_product(Gamma)  =  ||Lambda - Lambda~|| ||lambda*|| / beta   (loose)

    The two-form structure is exactly L2/C2's on the overlapping branch, where
    the cellwise form is tight to 0.2% and the max form is 220x loose --- so the
    parallel is not decoration; it is the same inequality chain used twice.

    ``C_mu`` is deliberately absent: it is the agents' sensitivity to interface
    data, it is not a function of the cut, and a constant cannot change a
    ranking.
    """
    beta = float(np.linalg.svd(S_M, compute_uv=False)[-1])
    r = S_M @ a_star - chi_M
    lam_dag = np.linalg.solve(S_M, chi_M)
    out = {
        "beta": beta,
        "residual": float(np.linalg.norm(r)),
        "q_tight": float(np.linalg.norm(r) / max(beta, 1e-300)),
        "trace_error": float(np.linalg.norm(lam_dag - a_star)),
        "norm_S": float(np.linalg.norm(S_M, 2)),
        "kappa": float(np.linalg.cond(S_M)),
    }
    if S_full is not None and u_gamma_star is not None and P is not None:
        # the product form: ||Lambda - Lambda~|| with Lambda~ = P (P^T S P) P^+
        # lifted back, i.e. the part of the exact operator the declared space
        # cannot represent.
        mismatch = S_full - P @ np.linalg.solve(P.T @ P, P.T @ S_full)
        out["op_mismatch"] = float(np.linalg.norm(mismatch, 2))
        out["norm_lam_star"] = float(np.linalg.norm(u_gamma_star))
        out["q_product"] = float(
            out["op_mismatch"] * out["norm_lam_star"] / max(beta, 1e-300))
    return out


def measure(dec: SubstructureDecomposition, f: np.ndarray,
            u_star: np.ndarray) -> dict:
    """Everything one cut placement produces: the criterion and the truth."""
    sc = schur(dec, f)
    P = dec.basis()
    S, chi = sc["S"], sc["chi"]
    W = dec.phys.hx                                  # the face measure
    S_M = P.T @ S @ P * W
    chi_M = P.T @ chi * W
    u_g_star = u_star.ravel()[sc["gamma"]]
    a_star = P.T @ u_g_star * W                      # orthonormal in the W pairing

    crit = w57_criterion(S_M, chi_M, a_star, S_full=S, u_gamma_star=u_g_star, P=P)
    lam_dag = np.linalg.solve(S_M, chi_M)
    u_comp = solve_substructured(dec, sc, P @ lam_dag)
    denom = float(np.linalg.norm(u_star))
    crit["composed_defect"] = float(np.linalg.norm(u_comp - u_star) / denom)
    u_exact = solve_substructured(dec, sc, P @ a_star)
    crit["defect_with_exact_trace"] = float(np.linalg.norm(u_exact - u_star) / denom)
    crit["cut"] = dec.cut
    crit["m_eff"] = dec.m_eff
    return crit


def exact_check(phys: SteadyPhysics, f: np.ndarray, u_star: np.ndarray,
                cut: int = 0) -> float:
    """Substructuring with the FULL interface space must reproduce the monolith."""
    dec = SubstructureDecomposition(phys=phys, m_eff=phys.nx, cut=cut)
    sc = schur(dec, f)
    u_g = np.linalg.solve(sc["S"], sc["chi"])
    u = solve_substructured(dec, sc, u_g)
    return float(np.linalg.norm(u - u_star) / np.linalg.norm(u_star))
