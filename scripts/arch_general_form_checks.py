"""arch_general_form_checks -- sanity checks for the generalized chart operator (2026-10-10).

Three statements of the design note [[chart-operator-general-form]], checked with classical
solves standing in for networks. A "learned" output is the exact one plus a smooth random
perturbation of relative size eps, as in `arch_superelement_gram.py`. These are checks of
algebra and of convergence claims, not measurements of any network.

A. **Non-symmetric pieces** (steady advection-diffusion, upwind cell-centred finite volumes,
   face-hybridised, 16 cosine modes per side). With primal *and* adjoint constraint modes the
   host's Petrov-Galerkin port matrix W~^T M V~ errs by exactly E_W^T A_II E_V, quadratic in
   the field errors. With primal modes alone (Galerkin, V~^T M V~) the error is first order
   once advection breaks the symmetry.
B. **Two separately trained single-field experts on one piece** (linear thermoelasticity, a
   1-D bar of steel, one backward-Euler step). The host's coupled system over the product of
   the two experts' trial spaces is symmetric quasi-definite, so it is nonsingular for any
   expert output; its solution is within (1 + gamma) of the best approximation in the block
   energy norm; a staggered sweep contracts the error by gamma^2; and gamma^2 is bounded by the
   thermoelastic coupling number delta = beta^2 T0 / ((lambda + 2 mu) rho c).
C. **A nonlinear convex piece** (-Lap u + c u^3 = 0 with Dirichlet data on the unit square).
   The host minimises the true energy over x + span{network proposal, classical
   Jacobi-preconditioned gradient}. The energy decreases at every step whatever the proposal,
   the iteration converges at no worse than the classical rate, and good proposals make it
   as fast as Newton's method.

Writes ``out/arch/general_form_checks.txt`` and ``.json``. Pure numpy/scipy; a few seconds.

    set PYTHONIOENCODING=utf-8
    python scripts/arch_general_form_checks.py
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "out", "arch")
RNG = np.random.default_rng(20261010)
EPS = (0.3, 0.1, 0.03, 0.01)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def smooth_field(n: int, rng, kmax: int = 6) -> np.ndarray:
    """A smooth random field on the n x n cell centres of the unit square."""
    s = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(s, s)
    f = np.zeros((n, n))
    for a in range(kmax + 1):
        for b in range(kmax + 1):
            w = math.exp(-(a * a + b * b) / 8.0)
            f += w * rng.standard_normal() * np.cos(a * math.pi * X + rng.uniform(0, 2 * math.pi)) \
                * np.cos(b * math.pi * Y + rng.uniform(0, 2 * math.pi))
    return f.ravel()


def perturb_columns(H: np.ndarray, eps: float, n: int, rng) -> np.ndarray:
    """Each column plus a smooth random field of the same Euclidean norm times eps."""
    E = np.empty_like(H)
    for k in range(H.shape[1]):
        r = smooth_field(n, rng)
        E[:, k] = eps * np.linalg.norm(H[:, k]) * r / np.linalg.norm(r)
    return E


def slope(eps, err) -> float:
    x, y = np.log(np.asarray(eps)), np.log(np.asarray(err))
    return float(np.polyfit(x, y, 1)[0])


# ---------------------------------------------------------------------------
# A. non-symmetric pieces: Petrov-Galerkin against Galerkin
# ---------------------------------------------------------------------------


def adv_diff_piece(n: int, b: tuple[float, float]):
    """Upwind cell-centred finite volumes for -div(grad u) + b . grad u on the unit square,
    hybridised on the 4n boundary faces. Rows of cells: outward flux balance; rows of faces:
    the flux into the piece through that face. Returns the four blocks."""
    h = 1.0 / n
    N = n * n
    faces = [(i, (0.0, -1.0)) for i in range(n)]                       # bottom
    faces += [(j * n + n - 1, (1.0, 0.0)) for j in range(n)]           # right
    faces += [((n - 1) * n + i, (0.0, 1.0)) for i in range(n)]         # top
    faces += [(j * n, (-1.0, 0.0)) for j in range(n)]                  # left
    F = len(faces)
    AII = np.zeros((N, N)); AIG = np.zeros((N, F)); AGI = np.zeros((F, N)); AGG = np.zeros((F, F))
    bx, by = b

    def interior(c, d, nv):
        bn = bx * nv[0] + by * nv[1]
        AII[c, c] += 1.0; AII[c, d] -= 1.0; AII[d, d] += 1.0; AII[d, c] -= 1.0
        if bn > 0:
            AII[c, c] += h * bn; AII[d, c] -= h * bn
        else:
            AII[c, d] += h * bn; AII[d, d] -= h * bn

    for j in range(n):
        for i in range(n):
            c = j * n + i
            if i + 1 < n:
                interior(c, c + 1, (1.0, 0.0))
            if j + 1 < n:
                interior(c, c + n, (0.0, 1.0))
    for f, (c, nv) in enumerate(faces):
        bn = bx * nv[0] + by * nv[1]
        AII[c, c] += 2.0 + (h * bn if bn > 0 else 0.0)
        AIG[c, f] += -2.0 + (h * bn if bn < 0 else 0.0)
        AGI[f, c] = -(2.0 + (h * bn if bn > 0 else 0.0))
        AGG[f, f] = 2.0 - (h * bn if bn < 0 else 0.0)
    return AII, AIG, AGI, AGG


def cosine_ports(n: int, m: int) -> np.ndarray:
    s = (np.arange(n) + 0.5) / n
    side = np.column_stack([np.cos(k * math.pi * s) for k in range(m)])
    side, _ = np.linalg.qr(side)
    return sla.block_diag(side, side, side, side)


def check_A(n: int = 32, m: int = 16) -> dict:
    Q = cosine_ports(n, m)
    out = {"n": n, "modes_per_side": m, "cases": []}
    for pe in (0.0, 20.0, 200.0):
        b = (pe * math.cos(math.pi / 6), pe * math.sin(math.pi / 6))
        AII, AIG, AGI, AGG = adv_diff_piece(n, b)
        M = np.block([[AII, AIG], [AGI, AGG]])
        H = -np.linalg.solve(AII, AIG @ Q)                 # primal constraint modes
        G = -np.linalg.solve(AII.T, AGI.T @ Q)             # adjoint constraint modes
        V, W = np.vstack([H, Q]), np.vstack([G, Q])
        Lam = Q.T @ (AGG - AGI @ np.linalg.solve(AII, AIG)) @ Q
        nl = np.linalg.norm(Lam)
        exact_check = float(np.linalg.norm(W.T @ M @ V - Lam) / nl)
        ev_m = np.linalg.eigvalsh(0.5 * (M + M.T))
        sym_min = float(ev_m.min() / np.abs(ev_m).max())
        ev_l = np.linalg.eigvalsh(0.5 * (Lam + Lam.T))
        lam_sym_min = float(ev_l.min() / np.abs(ev_l).max())
        rows = []
        rng = np.random.default_rng(7)
        for eps in EPS:
            EV = perturb_columns(H, eps, n, rng)
            EW = perturb_columns(G, eps, n, rng)
            Vt, Wt = np.vstack([H + EV, Q]), np.vstack([G + EW, Q])
            Lpg = Wt.T @ M @ Vt
            Lg = Vt.T @ M @ Vt
            ident = float(np.linalg.norm(Lpg - Lam - EW.T @ AII @ EV) / nl)
            sym_g = 0.5 * (Lg + Lg.T)
            rows.append({"eps": eps,
                         "petrov_galerkin_rel_err": float(np.linalg.norm(Lpg - Lam) / nl),
                         "galerkin_rel_err": float(np.linalg.norm(Lg - Lam) / nl),
                         "pg_identity_residual": ident,
                         "galerkin_sym_part_min_eig": float(np.linalg.eigvalsh(sym_g).min())})
        out["cases"].append({
            "peclet": pe, "cell_peclet": pe / n, "exact_identity_residual": exact_check,
            "sym_part_of_M_min_eig_rel": sym_min, "sym_part_of_exact_Lambda_min_eig_rel": lam_sym_min,
            "rows": rows,
            "slope_pg": slope(EPS, [r["petrov_galerkin_rel_err"] for r in rows]),
            "slope_g": slope(EPS, [r["galerkin_rel_err"] for r in rows])})
    return out


# ---------------------------------------------------------------------------
# B. two single-field experts on one piece: linear thermoelasticity
# ---------------------------------------------------------------------------


def thermoelastic_bar(ncell: int = 200):
    """A 1-D bar of steel in uniaxial strain, one backward-Euler step, surface heating at
    x = 0 (which is also clamped), the far end free and insulated. Returns the blocks of
    [[K_u, -G], [-G^T, -N]] [u; theta] = [f; r] and the material numbers."""
    L = 0.1
    E, nu, alpha = 200e9, 0.3, 12e-6
    rho, c, k, T0 = 7850.0, 460.0, 45.0, 293.0
    lam2mu = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    beta = E * alpha / (1 - 2 * nu)
    a = k / (rho * c)
    fo = 0.05
    dt = fo * L * L / a
    h = L / ncell
    # displacement: nodes 1..ncell (node 0 clamped), linear elements
    Ku = np.zeros((ncell, ncell))
    for e in range(ncell):           # element e joins nodes e and e+1; node e+1 -> index e
        idx = [e - 1, e]
        ke = lam2mu / h * np.array([[1.0, -1.0], [-1.0, 1.0]])
        for p in range(2):
            for q in range(2):
                if idx[p] >= 0 and idx[q] >= 0:
                    Ku[idx[p], idx[q]] += ke[p, q]
    # coupling G[node, cell] = integral over the cell of beta * dv_node/dx
    G = np.zeros((ncell, ncell))
    for e in range(ncell):
        G[e, e] += beta                  # node e+1 (index e): slope +1/h on cell e, times h
        if e - 1 >= 0:
            G[e - 1, e] -= beta          # node e (index e-1): slope -1/h on cell e
    # temperature: cells, finite volumes, insulated ends; heat flux q_in enters cell 0
    Ct = rho * c * h * np.eye(ncell)
    Kt = np.zeros((ncell, ncell))
    for e in range(ncell - 1):
        g = k / h
        Kt[e, e] += g; Kt[e + 1, e + 1] += g; Kt[e, e + 1] -= g; Kt[e + 1, e] -= g
    Nt = (Ct + dt * Kt) / T0
    q_in = 1e5
    Qsrc = np.zeros(ncell); Qsrc[0] = q_in
    f = np.zeros(ncell)
    r = -(dt * Qsrc) / T0                      # theta_old = 0, u_old = 0
    delta = beta ** 2 * T0 / (lam2mu * rho * c)
    return {"Ku": Ku, "G": G, "Nt": Nt, "f": f, "r": r, "delta": delta, "dt": dt, "fo": fo,
            "L": L, "beta": beta, "lam2mu": lam2mu}


def check_B(ncell: int = 200) -> dict:
    S = thermoelastic_bar(ncell)
    Ku, G, Nt, f, r = S["Ku"], S["G"], S["Nt"], S["f"], S["r"]
    n = ncell
    Kfull = np.block([[Ku, -G], [-G.T, -Nt]])
    rhs = np.concatenate([f, r])
    exact = np.linalg.solve(Kfull, rhs)
    u_ex, th_ex = exact[:n], exact[n:]
    Lu = np.linalg.cholesky(Ku)
    Ln = np.linalg.cholesky(Nt)
    C = sla.solve_triangular(Ln, G.T, lower=True)               # Ln^{-1} G^T
    C = sla.solve_triangular(Lu, C.T, lower=True).T             # Ln^{-1} G^T Lu^{-T}
    gamma = float(np.linalg.norm(C, 2))
    # staggered (block Gauss-Seidel) sweeps with exact single-field solves
    th = np.zeros(n); errs = []
    for _ in range(8):
        u = np.linalg.solve(Ku, f + G @ th)
        th = np.linalg.solve(Nt, -r - G.T @ u)
        errs.append(math.sqrt((u - u_ex) @ Ku @ (u - u_ex) + (th - th_ex) @ Nt @ (th - th_ex)))
    rates = [errs[i + 1] / errs[i] for i in range(len(errs) - 1) if errs[i] > 1e-300]
    # one-way solution (thermal first, then mechanics): the feedback the coupled answer adds
    th_1w = np.linalg.solve(Nt, -r)
    u_1w = np.linalg.solve(Ku, f + G @ th_1w)

    def bnorm(du, dth):
        return math.sqrt(du @ Ku @ du + dth @ Nt @ dth)

    ref = bnorm(u_ex, th_ex)
    one_way = {"u_rel_err": float(np.linalg.norm(u_1w - u_ex) / np.linalg.norm(u_ex)),
               "theta_rel_err": float(np.linalg.norm(th_1w - th_ex) / np.linalg.norm(th_ex))}

    rng = np.random.default_rng(11)

    def expert(Aop, rhs_vec, eps):
        """The exact single-field solve plus a smooth error of relative size eps in that
        field's own energy norm (what an energy-norm-trained expert would make)."""
        x = np.linalg.solve(Aop, rhs_vec)
        if eps == 0 or not np.any(x):
            return x
        s = (np.arange(n) + 0.5) / n
        p = sum(rng.standard_normal() * math.exp(-kk * kk / 8.0) * np.cos(kk * math.pi * s) for kk in range(8))
        return x + eps * math.sqrt(x @ Aop @ x) * p / math.sqrt(p @ Aop @ p)

    def energy_orthonormal(Bm, Aop):
        """An Aop-orthonormal basis of span(Bm), dropping numerically dependent directions."""
        if Bm.shape[1] == 0:
            return Bm
        Gm = Bm.T @ Aop @ Bm
        w_, V_ = np.linalg.eigh(Gm)
        keep = w_ > 1e-12 * w_.max()
        return Bm @ (V_[:, keep] / np.sqrt(w_[keep]))

    rows = []
    for eps in (0.0, 0.01, 0.1, 1.0):
        for level in (0, 1, 2):
            # round 0: each expert answers its own load; round k adds each expert's response
            # to the coupling term of the other's latest field (thermal load, then heating)
            U = [expert(Ku, f, eps)] if np.any(f) else []
            T = [expert(Nt, -r, eps)]
            for _ in range(level):
                U.append(expert(Ku, G @ T[-1], eps))
                T.append(expert(Nt, -G.T @ U[-1], eps))
            # the host orthonormalises each expert's fields in that field's own energy
            Ub = energy_orthonormal(np.column_stack(U) if U else np.zeros((n, 0)), Ku)
            Tb = energy_orthonormal(np.column_stack(T), Nt)
            nu, nt = Ub.shape[1], Tb.shape[1]
            Z = np.zeros((2 * n, nu + nt))                       # trial space, product of the two
            Z[:n, :nu] = Ub
            Z[n:, nu:] = Tb
            Kr = Z.T @ Kfull @ Z
            ev_r = np.linalg.eigvalsh(0.5 * (Kr + Kr.T))
            yr = np.linalg.solve(Kr, Z.T @ rhs)
            w = Z @ yr
            err = bnorm(w[:n] - u_ex, w[n:] - th_ex)
            # best approximation in the block energy norm: orthogonal projection onto span Z
            Bn = sla.block_diag(Ku, Nt)
            cbest = np.linalg.solve(Z.T @ Bn @ Z, Z.T @ Bn @ exact)
            wb = Z @ cbest
            best = bnorm(wb[:n] - u_ex, wb[n:] - th_ex)
            # quasi-definite: dim(Ub) positive and dim(Tb) negative eigenvalues; with energy-
            # orthonormal bases every |eigenvalue| lies in [1, sqrt(1 + gamma^2)]
            du, dth = w[:n] - u_ex, w[n:] - th_ex
            rows.append({"eps": eps, "enrichment_rounds": level, "trial_dim": int(Z.shape[1]),
                         "inertia_pos_neg": [int((ev_r > 0).sum()), int((ev_r < 0).sum())],
                         "min_abs_eig": float(np.abs(ev_r).min()), "max_abs_eig": float(np.abs(ev_r).max()),
                         "error_rel": err / ref, "best_rel": best / ref,
                         "ratio_to_best": err / best if best > 0 else 1.0,
                         "u_rel_energy_err": float(math.sqrt(du @ Ku @ du) / math.sqrt(u_ex @ Ku @ u_ex)),
                         "theta_rel_energy_err": float(math.sqrt(dth @ Nt @ dth) / math.sqrt(th_ex @ Nt @ th_ex))})
    return {"ncell": n, "steel": True, "fourier": S["fo"], "dt_s": S["dt"], "delta": S["delta"],
            "gamma": gamma, "gamma_sq": gamma ** 2, "sqrt_delta": math.sqrt(S["delta"]),
            "staggered_rates": rates, "one_way": one_way, "rows": rows}


# ---------------------------------------------------------------------------
# C. a nonlinear convex piece: learned subspace correction with a classical safeguard
# ---------------------------------------------------------------------------


def dirichlet_square(n: int):
    """Cell-centred finite volumes for -Lap on the unit square, unit spacing in index units,
    Dirichlet data at half a cell (conductance 2). Returns A and the boundary coupling vector
    for the data g(x, y) = 1 + x + 0.5 cos(pi y) taken at the boundary face centres."""
    N = n * n
    A = sp.lil_matrix((N, N)); b = np.zeros(N)
    s = (np.arange(n) + 0.5) / n

    def g(x, y):
        return 1.0 + x + 0.5 * math.cos(math.pi * y)

    for j in range(n):
        for i in range(n):
            c = j * n + i
            for di, dj in ((1, 0), (0, 1)):
                ii, jj = i + di, j + dj
                if ii < n and jj < n:
                    d = jj * n + ii
                    A[c, c] += 1; A[d, d] += 1; A[c, d] -= 1; A[d, c] -= 1
            if i == 0:
                A[c, c] += 2; b[c] += 2 * g(0.0, s[j])
            if i == n - 1:
                A[c, c] += 2; b[c] += 2 * g(1.0, s[j])
            if j == 0:
                A[c, c] += 2; b[c] += 2 * g(s[i], 0.0)
            if j == n - 1:
                A[c, c] += 2; b[c] += 2 * g(s[i], 1.0)
    return A.tocsr(), b


def check_C(n: int = 32, creact: float = 100.0, max_outer: int = 3000) -> dict:
    A, b = dirichlet_square(n)
    w = creact / n ** 2                      # c u^3 per unit area times the cell area h^2

    def energy(u):
        return 0.5 * u @ A @ u - b @ u + 0.25 * w * np.sum(u ** 4)

    def grad(u):
        return A @ u - b + w * u ** 3

    def hess(u):
        return (A + sp.diags(3 * w * u ** 2)).tocsc()

    def hess_diag(u):
        return A.diagonal() + 3 * w * u ** 2

    def reduced_hess(u, D):
        AD = A @ D
        return D.T @ AD + (D * (3 * w * u ** 2)[:, None]).T @ D

    # the exact discrete solution, Newton with backtracking from the linear solve
    u = spla.spsolve(A.tocsc(), b)
    for _ in range(60):
        d = -spla.spsolve(hess(u), grad(u))
        t, e0, g0 = 1.0, energy(u), grad(u) @ d
        while energy(u + t * d) > e0 + 1e-4 * t * g0:
            t *= 0.5
        u = u + t * d
        if np.linalg.norm(grad(u)) < 1e-12 * np.linalg.norm(b):
            break
    ustar, Estar = u, energy(u)
    unorm = math.sqrt(ustar @ (A @ ustar))
    rng = np.random.default_rng(5)

    gap0 = energy(np.zeros(n * n)) - Estar

    def minimise_span(x, D):
        """The host's exact minimisation of the true energy over x + span(D): Newton with
        backtracking on the small convex problem."""
        c = np.zeros(D.shape[1])
        for _ in range(40):
            xc = x + D @ c
            gr = D.T @ grad(xc)
            Hr = reduced_hess(xc, D)
            step = np.linalg.lstsq(Hr, -gr, rcond=None)[0]
            t, e0 = 1.0, energy(xc)
            while energy(x + D @ (c + t * step)) > e0 + 1e-4 * t * (gr @ step) and t > 1e-12:
                t *= 0.5
            c = c + t * step
            if np.linalg.norm(step) * t < 1e-14 * (1 + np.linalg.norm(c)):
                break
        return x + D @ c

    def run(eps, mode):
        x = np.zeros(n * n)
        gaps, mono = [], True
        Eprev = energy(x)
        egaps = [Eprev - Estar]
        violations, cl_factors = 0, []
        for k in range(max_outer):
            gk = grad(x)
            d_cl = -gk / hess_diag(x)                      # Jacobi-preconditioned gradient
            dirs = [d_cl]
            if mode == "net":
                Hk = hess(x)
                dN = -spla.spsolve(Hk, gk)                 # the direction a good network proposes
                if eps > 0:
                    p = smooth_field(n, rng)
                    sc = math.sqrt(dN @ (Hk @ dN)) / math.sqrt(p @ (Hk @ p))
                    dN = dN + eps * sc * p
                dirs.insert(0, dN)
            elif mode == "garbage":
                dirs.insert(0, smooth_field(n, rng))
            D = np.column_stack(dirs)
            # what one classical step from the same point would reach, for the per-step check
            E_cl = energy(minimise_span(x, d_cl[:, None]))
            gap_k = energy(x) - Estar
            if gap_k > 1e-14 * gap0:
                cl_factors.append((E_cl - Estar) / gap_k)
            x = minimise_span(x, D)
            Ek = energy(x)
            if Ek > E_cl + 1e-10 * abs(E_cl):
                violations += 1
            if Ek > Eprev + 1e-12 * abs(Eprev):
                mono = False
            Eprev = Ek
            egaps.append(Ek - Estar)
            err = math.sqrt((x - ustar) @ (A @ (x - ustar))) / unorm
            gaps.append(err)
            if err < 1e-10:
                break
        ratios = [egaps[i + 1] / egaps[i] for i in range(len(egaps) - 1) if egaps[i] > 1e-14 * gap0]
        return {"outer_iterations": len(gaps), "final_rel_err": gaps[-1],
                "energy_monotone": mono, "energy_above_minimum": bool(energy(x) >= Estar - 1e-12 * abs(Estar)),
                "steps_worse_than_one_classical_step": violations,
                "worst_gap_ratio": float(max(ratios)) if ratios else 0.0,
                "worst_classical_one_step_factor": float(max(cl_factors)) if cl_factors else 0.0,
                "first_errors": gaps[:6]}

    res = {"n": n, "c": creact, "max_outer": max_outer, "runs": []}
    for eps in (0.0, 0.01, 0.1, 0.5, 1.0, 3.0):
        r = run(eps, "net"); r.update({"proposal": f"Newton direction, relative error {eps:g}"})
        res["runs"].append(r)
    r = run(0, "garbage"); r.update({"proposal": "a random smooth field (no information)"})
    res["runs"].append(r)
    r = run(0, "classical"); r.update({"proposal": "none: classical direction only"})
    res["runs"].append(r)
    return res


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def main() -> None:
    A = check_A()
    B = check_B()
    C = check_C()
    L = ["arch_general_form_checks -- sanity checks for the generalized chart operator",
         "=" * 78, "",
         "A. NON-SYMMETRIC PIECE (advection-diffusion, upwind, n = 32, 16 cosine modes per side)",
         "   port-matrix relative error, Petrov-Galerkin (primal + adjoint modes) vs Galerkin (primal only)"]
    for cs in A["cases"]:
        L.append(f"   Peclet {cs['peclet']:g} (cell {cs['cell_peclet']:.3g}); W^T M V = Lambda to "
                 f"{cs['exact_identity_residual']:.1e}; min eig/max |eig| of sym(M) {cs['sym_part_of_M_min_eig_rel']:+.2e},"
                 f" of sym(exact Lambda) {cs['sym_part_of_exact_Lambda_min_eig_rel']:+.2e}")
        L.append("      eps   | PG error   Galerkin error | PG identity residual | min eig sym(Galerkin)")
        for r in cs["rows"]:
            L.append(f"      {r['eps']:<5g} | {r['petrov_galerkin_rel_err']:.2e}   {r['galerkin_rel_err']:.2e}"
                     f"       | {r['pg_identity_residual']:.1e}              | {r['galerkin_sym_part_min_eig']:+.2e}")
        L.append(f"      fitted slopes: PG {cs['slope_pg']:.2f}, Galerkin {cs['slope_g']:.2f}")
    L += ["", "B. TWO SINGLE-FIELD EXPERTS, ONE PIECE (thermoelastic steel bar, 200 cells, one backward-Euler step,",
          f"   Fo = {B['fourier']:g}, dt = {B['dt_s']:.1f} s, surface heating 1e5 W/m^2)",
          f"   coupling: gamma = {B['gamma']:.4f}, gamma^2 = {B['gamma_sq']:.4f}; delta = {B['delta']:.4f} "
          f"(sqrt {B['sqrt_delta']:.4f})",
          "   staggered sweep error ratios (exact single-field solves): "
          + ", ".join(f"{x:.4f}" for x in B["staggered_rates"][:5]),
          f"   one-way answer (no feedback) vs coupled: u {B['one_way']['u_rel_err']:.2e}, "
          f"theta {B['one_way']['theta_rel_err']:.2e}",
          "   host Galerkin over the product of the experts' trial spaces (each orthonormalised in its own energy);",
          f"   errors relative, in the block energy norm and in each field's own energy norm; 1+gamma = {1 + B['gamma']:.3f},"
          f" sqrt(1+gamma^2) = {math.sqrt(1 + B['gamma'] ** 2):.4f}",
          "      eps   rounds dim | inertia(+,-)  |eig| min..max  | error/ref  best/ref   ratio | u error    theta error"]
    for r in B["rows"]:
        L.append(f"      {r['eps']:<5g} {r['enrichment_rounds']}      {r['trial_dim']}   | {str(tuple(r['inertia_pos_neg'])):<12}  "
                 f"{r['min_abs_eig']:.4f}..{r['max_abs_eig']:.4f} | {r['error_rel']:.2e}  {r['best_rel']:.2e}  {r['ratio_to_best']:.3f}"
                 f" | {r['u_rel_energy_err']:.2e}   {r['theta_rel_energy_err']:.2e}")
    L += ["", f"C. NONLINEAR CONVEX PIECE (-Lap u + {C['c']:g} u^3 = 0, n = {C['n']}, Dirichlet data)",
          "   outer iterations (= network calls) to relative energy-norm error 1e-10; the host minimises the",
          f"   true energy over x + span{{proposal, Jacobi-preconditioned gradient}}; capped at {C['max_outer']}",
          "   per step: the energy gap's ratio (worst over the run), the worst one-step factor a classical step",
          "   would have achieved from the same points, and the steps that did worse than that classical step",
          "      proposal                                   | outer its  final error | monotone  E>=E* | worst ratio  classical  worse steps"]
    for r in C["runs"]:
        L.append(f"      {r['proposal']:<42} | {r['outer_iterations']:>9}  {r['final_rel_err']:.2e}  | "
                 f"{str(r['energy_monotone']):<8}  {str(r['energy_above_minimum']):<5} | {r['worst_gap_ratio']:.5f}      "
                 f"{r['worst_classical_one_step_factor']:.5f}    {r['steps_worse_than_one_classical_step']}")
    text = "\n".join(L) + "\n"
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "general_form_checks.json"), "w", encoding="utf-8") as fh:
        json.dump({"probe": "arch_general_form_checks", "A": A, "B": B, "C": C}, fh, indent=1)
    with open(os.path.join(OUT, "general_form_checks.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
