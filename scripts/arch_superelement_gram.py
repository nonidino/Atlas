"""arch_superelement_gram -- two ways a learned expert could supply its port matrix.

For decision 6 of the architecture conversation (the expert's output format).  A
superelement couples through each piece's port matrix Lambda_i, its Dirichlet-to-Neumann
map on its port modes ([[coupling-cost-and-complexity]], Port-m).  A learned expert
could supply it in two ways:

- **direct**: a network head that outputs the matrix itself;
- **energy Gram**: the expert outputs the *constraint modes* H_k -- the piece's response
  field to each unit port mode, with that mode as its exact trace -- and the host computes
  Lambda_jk = v_j^T M v_k, v_k = (H_k, Q e_k), M the piece's own discrete energy
  (stiffness) form.

The Gram route has three properties by construction, whatever the network returns:
symmetric; positive semidefinite (a Gram matrix of an energy); and, because the exact
modes are energy-orthogonal to every trace-free perturbation (the Dirichlet principle),
an error that is the Gram matrix of the field errors E_k,

    Lambda_Gram - Lambda_exact = E^T A E   (exactly, and >= 0 in the Loewner order),

so a field error of relative size eps gives a port-matrix error of order eps^2.  The
coupled answer is then a Ritz-Galerkin solution in the span of the learned modes.

**What is measured.**  The reference-space problem of `arch_coupling_cost.py` (4 x 4
square pieces of 32^2 cells, smooth log-normal coefficient, steady and Fo = 0.1), 16
cosine port modes per edge.  The *exact* constraint modes and port matrices are computed
classically; then both are perturbed by the same relative amount eps -- the modes by
smooth random fields scaled to a relative energy error eps per mode, the matrices by a
symmetric random matrix of relative Frobenius size eps -- standing in for a network of
that accuracy.  Reported for eps = 0.3, 0.1, 0.03, 0.01: the port matrices' relative
error, the identity's check, the smallest eigenvalue of the pieces' port matrices and of
the assembled interface matrix, and the coupled solution's error against the exact
finite-volume solution (the m = 16 truncation floor is reported beside it).

Exact classical solves only; no network.  Headless, not a timing.  Writes
``out/arch/superelement_gram.txt`` and ``.json``.

    set PYTHONIOENCODING=utf-8
    python scripts/arch_superelement_gram.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
OUT = os.path.join(HERE, "out", "arch")

import numpy as np                                                     # noqa: E402
import scipy.sparse as sp                                              # noqa: E402
import scipy.sparse.linalg as spla                                     # noqa: E402

import arch_coupling_cost as C                                         # noqa: E402

M_MODES = 16
EPS = (0.3, 0.1, 0.03, 0.01)
SEED = 7


def local_system(P, sub, p):
    """Piece p's face-Dirichlet matrix A_i, the coupling B_i (cells x faces), the
    half-cell conductances D_i, its cells and its interface faces -- rebuilt exactly as
    `Substructure` builds them, kept as matrices for the energy form."""
    own, _lu, fidx, B, D = sub.pieces[p]
    A = P.A.tocsr()
    F = sub.F
    kap = P.kappa.ravel()
    ins_a = P.pid[F[fidx, 0]] == p
    cell_in = np.where(ins_a, F[fidx, 0], F[fidx, 1])
    cell_out = np.where(ins_a, F[fidx, 1], F[fidx, 0])
    pos = np.full(P.N, -1)
    pos[own] = np.arange(own.size)
    c_f = -np.asarray(A[cell_in, cell_out]).ravel()
    d = np.zeros(own.size)
    np.add.at(d, pos[cell_in], 2.0 * kap[cell_in] - c_f)
    Ai = (A[own][:, own] + sp.diags(d)).tocsc()
    return own, fidx, Ai, B.tocsc(), D


def run(fo, rng) -> dict:
    P = C.Problem(4, 4, 32, fo)
    sub = C.Substructure(P)
    Q = C.port_basis(sub, M_MODES)
    x_ref = spla.splu(sp.csc_matrix(P.A)).solve(P.b)
    nrm = np.linalg.norm(x_ref)
    nq = Q.shape[1]

    pieces = []
    for p in range(P.P):
        own, fidx, Ai, Bi, Di = local_system(P, sub, p)
        Qi = Q[fidx]
        keep = np.flatnonzero(np.any(Qi != 0.0, axis=0))
        Qi = Qi[:, keep]
        lu = spla.splu(Ai)
        BQ = (Bi @ Qi)
        H = lu.solve(np.asarray(BQ))                     # exact constraint modes
        up = lu.solve(P.b[own])                          # exact particular, zero trace
        Lam = Qi.T @ (Di[:, None] * Qi) - BQ.T @ H       # exact port matrix
        Lam = 0.5 * (Lam + Lam.T)
        pieces.append(dict(own=own, fidx=fidx, Ai=Ai, Bi=Bi, Di=Di, Qi=Qi, keep=keep,
                           H=H, up=up, Lam=Lam, BQ=BQ))

    def gram(pc, Hm):
        """Lambda = v^T M v with v = (H, Q), M = [[A, -B], [-B^T, D]]."""
        AH = pc["Ai"] @ Hm
        return (Hm.T @ AH - Hm.T @ pc["BQ"] - pc["BQ"].T @ Hm
                + pc["Qi"].T @ (pc["Di"][:, None] * pc["Qi"]))

    def load(pc, Hm):
        """b_k = H_k^T f - v_k^T M (u_p, 0) = H_k^T f - (H_k^T A u_p - (B Q e_k)^T u_p)."""
        f = P.b[pc["own"]]
        return Hm.T @ f - (Hm.T @ (pc["Ai"] @ pc["up"]) - pc["BQ"].T @ pc["up"])

    def couple(lams, loads, Hs):
        rows, cols, vals = [], [], []
        rhs = np.zeros(nq)
        for pc, L, bvec in zip(pieces, lams, loads):
            r, c = np.meshgrid(pc["keep"], pc["keep"], indexing="ij")
            rows.append(r.ravel()); cols.append(c.ravel()); vals.append(L.ravel())
            rhs[pc["keep"]] += bvec
        S = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(nq, nq)).toarray()
        S = 0.5 * (S + S.T)
        ev_min = float(np.linalg.eigvalsh(S)[0])
        z = np.linalg.solve(S, rhs)
        x = np.zeros(P.N)
        for pc, Hm in zip(pieces, Hs):
            x[pc["own"]] = pc["up"] + Hm @ z[pc["keep"]]
        return x, ev_min

    # the exact superelement: the m-mode truncation floor
    x0, ev0 = couple([pc["Lam"] for pc in pieces], [load(pc, pc["H"]) for pc in pieces],
                     [pc["H"] for pc in pieces])
    floor = float(np.linalg.norm(x0 - x_ref) / nrm)
    out = {"Fo_piece": fo, "pieces": [4, 4], "n": 32, "modes_per_edge": M_MODES,
           "interface_unknowns": int(nq), "truncation_floor_rel_L2": floor,
           "assembled_min_eig_exact": ev0,
           "pieces_min_eig_exact": float(min(np.linalg.eigvalsh(pc["Lam"])[0] for pc in pieces)),
           "rows": []}

    for eps in EPS:
        lam_g, lam_d, loads_g, Hs = [], [], [], []
        err_g, err_d, ident, eig_g, eig_d = [], [], [], [], []
        for pc in pieces:
            n_cells = pc["own"].size
            side = int(round(np.sqrt(n_cells)))
            E = np.empty_like(pc["H"])
            for k in range(pc["H"].shape[1]):
                field = C.smooth_field(side, side, 4.0, rng).ravel()
                e_a = np.sqrt(max(field @ (pc["Ai"] @ field), 1e-300))
                E[:, k] = eps * np.sqrt(max(pc["Lam"][k, k], 1e-300)) / e_a * field
            Hn = pc["H"] + E
            Lg = gram(pc, Hn)
            # the identity Lambda_Gram - Lambda = E^T A E
            ident.append(float(np.linalg.norm(Lg - pc["Lam"] - E.T @ (pc["Ai"] @ E))
                               / np.linalg.norm(pc["Lam"])))
            G = rng.standard_normal(pc["Lam"].shape)
            G = 0.5 * (G + G.T)
            Ld = pc["Lam"] + eps * np.linalg.norm(pc["Lam"]) * G / np.linalg.norm(G)
            err_g.append(float(np.linalg.norm(Lg - pc["Lam"]) / np.linalg.norm(pc["Lam"])))
            err_d.append(float(np.linalg.norm(Ld - pc["Lam"]) / np.linalg.norm(pc["Lam"])))
            eig_g.append(float(np.linalg.eigvalsh(0.5 * (Lg + Lg.T))[0]))
            eig_d.append(float(np.linalg.eigvalsh(Ld)[0]))
            lam_g.append(0.5 * (Lg + Lg.T))
            lam_d.append(Ld)
            loads_g.append(load(pc, Hn))
            Hs.append(Hn)
        xg, evg = couple(lam_g, loads_g, Hs)
        # direct head: the matrix perturbed, the exact modes and loads kept (its best case)
        xd, evd = couple(lam_d, [load(pc, pc["H"]) for pc in pieces], [pc["H"] for pc in pieces])
        lam_scale = max(abs(float(np.linalg.eigvalsh(pc["Lam"])[-1])) for pc in pieces)
        out["rows"].append({
            "eps": eps,
            "gram_port_rel_err_mean": float(np.mean(err_g)),
            "direct_port_rel_err_mean": float(np.mean(err_d)),
            "gram_identity_residual_max": float(np.max(ident)),
            "gram_piece_min_eig_over_max": float(min(eig_g)) / lam_scale,
            "direct_piece_min_eig_over_max": float(min(eig_d)) / lam_scale,
            "direct_pieces_indefinite": int(np.sum(np.array(eig_d) < 0.0)),
            "gram_assembled_min_eig": evg, "direct_assembled_min_eig": evd,
            "gram_solution_rel_L2": float(np.linalg.norm(xg - x_ref) / nrm),
            "direct_solution_rel_L2": float(np.linalg.norm(xd - x_ref) / nrm),
        })
    return out


def text(results) -> str:
    L = []
    w = L.append
    w("arch_superelement_gram -- the port matrix as a direct output or as an energy Gram")
    w("=" * 84)
    for r in results:
        fo = "steady" if r["Fo_piece"] is None else f"Fo = {r['Fo_piece']:g}"
        w("")
        w(f"4 x 4 pieces of 32^2 cells, {fo}; {r['modes_per_edge']} cosine modes per edge,"
          f" {r['interface_unknowns']} interface unknowns")
        w(f"exact superelement: truncation floor {r['truncation_floor_rel_L2']:.2e} (rel L2),"
          f" assembled min eig {r['assembled_min_eig_exact']:.3e},"
          f" pieces' min eig {r['pieces_min_eig_exact']:.3e}")
        w("    eps | port matrix rel. error  | Gram identity | piece min eig / max     |"
          " assembled min eig      | solution rel. L2 error")
        w("        |   Gram       direct     |   residual    |   Gram      direct (#<0) |"
          "   Gram       direct    |   Gram       direct")
        for x in r["rows"]:
            w(f"  {x['eps']:5.2f} | {x['gram_port_rel_err_mean']:9.2e}  {x['direct_port_rel_err_mean']:9.2e}  |"
              f"  {x['gram_identity_residual_max']:9.1e}    | {x['gram_piece_min_eig_over_max']:9.2e} "
              f"{x['direct_piece_min_eig_over_max']:9.2e} ({x['direct_pieces_indefinite']:>2}) |"
              f" {x['gram_assembled_min_eig']:9.2e}  {x['direct_assembled_min_eig']:9.2e} |"
              f" {x['gram_solution_rel_L2']:9.2e}  {x['direct_solution_rel_L2']:9.2e}")
    return "\n".join(L) + "\n"


def main() -> None:
    rng = np.random.default_rng(SEED)
    results = [run(None, rng), run(0.1, rng)]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "superelement_gram.json"), "w", encoding="utf-8") as fh:
        json.dump({"probe": "arch_superelement_gram", "results": results}, fh, indent=1)
    t = text(results)
    with open(os.path.join(OUT, "superelement_gram.txt"), "w", encoding="utf-8") as fh:
        fh.write(t)
    sys.stdout.write(t)


if __name__ == "__main__":
    main()
