"""arch_sni_counterexample -- SNI's Theorem 1 needs a Lipschitz hypothesis on the learned map.

Schwarz Neural Inference (Huang et al., ICLR 2026, arXiv 2504.00510, Theorem 1 and its
proof in Appendix C) assumes

    (a) the classical additive Schwarz-Richardson map F contracts with rate rho < 1,
    (b) the learned local solver differs from the exact one by at most c, uniformly,

and concludes that the learned iteration converges to a fixed point, within
c' = tau t c / (1 - rho) of the classical one.  The distance bound holds.  The
convergence step does not: the proof shows ||u~^m - u~^n|| <= ||u^m - u^n|| + 2c',
which does not make u~^n Cauchy, because 2c' does not vanish.

The smallest case: one subdomain, R = I, t = 1.  Take the exact local solve S(u) = 0,
so F(u) = u + tau (S(u) - u) = (1 - tau) u contracts with rho = 1 - tau.  Take a
learned solve S~(u) = -c tanh(k u), which satisfies |S~(u) - S(u)| <= c for every u.
The learned map F~(u) = (1 - tau) u + tau S~(u) has one fixed point, u = 0, and
F~'(0) = 1 - tau - tau c k.  With tau = 1/2, c = 1, k = 10 that is -4.5: the only
fixed point repels, every orbit stays bounded (|F~(u)| <= |u|/2 + 1/2), so no orbit
from u0 != 0 converges.  Both hypotheses hold; the conclusion fails.

What restores it is a bound on the learned map itself, Lip(F~) < 1 -- the hypothesis of
theorem T1 in the vault's formal-proofs plan.  Pure Python, no dependencies, not a
timing.  Writes out/arch/sni_counterexample.txt.

    set PYTHONIOENCODING=utf-8
    python scripts/arch_sni_counterexample.py
"""

from __future__ import annotations

import math
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "out", "arch", "sni_counterexample.txt")

TAU, C, K = 0.5, 1.0, 10.0


def exact_solve(u: float) -> float:
    return 0.0


def learned_solve(u: float) -> float:
    return -C * math.tanh(K * u)


def schwarz_step(u: float, solve) -> float:
    return u + TAU * (solve(u) - u)


def main() -> None:
    lines = []
    rho = 1.0 - TAU
    worst = max(abs(learned_solve(x / 100.0) - exact_solve(x / 100.0)) for x in range(-500, 501))
    lines.append("arch_sni_counterexample -- SNI Theorem 1, one subdomain, R = I, t = 1")
    lines.append(f"tau = {TAU}, c = {C}, k = {K}")
    lines.append(f"hypothesis (a): classical map contracts, rho = 1 - tau = {rho}")
    lines.append(f"hypothesis (b): max |S~ - S| on [-5, 5] = {worst:.6f} <= c = {C}")
    slope = 1.0 - TAU - TAU * C * K
    lines.append(f"learned map's slope at its only fixed point u = 0: {slope}")
    c_prime = TAU * 1 * C / (1.0 - rho)
    lines.append(f"SNI's distance bound c' = tau t c / (1 - rho) = {c_prime}")
    lines.append("")
    lines.append("   n     classical u^n        learned u~^n     |u~^n - u^n|   <= c'")
    u, v = 1.0, 1.0
    for n in range(0, 41):
        if n <= 10 or n % 10 == 0:
            gap = abs(v - u)
            lines.append(f"{n:4d}  {u: .12e}  {v: .12e}  {gap:.6f}   {gap <= c_prime + 1e-12}")
        u, v = schwarz_step(u, exact_solve), schwarz_step(v, learned_solve)
    tail = []
    for _ in range(1000):
        v = schwarz_step(v, learned_solve)
        tail.append(v)
    lines.append("")
    lines.append(f"after 1040 steps the learned iterates alternate between "
                 f"{min(tail[-2:]):.12f} and {max(tail[-2:]):.12f}")
    lines.append(f"successive difference |u~^(n+1) - u~^n| = {abs(tail[-1] - tail[-2]):.12f} (does not tend to 0)")
    lines.append("")
    lines.append("verdict: both hypotheses hold, the distance bound holds at every n, and the")
    lines.append("learned iteration does not converge.  A Lipschitz bound Lip(F~) < 1 on the learned")
    lines.append("map is the missing hypothesis (T1 of the formal-proofs plan).")
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
