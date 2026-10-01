import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Topology.MetricSpace.Pseudo.Defs

/-!
# A remark beside T1 — accuracy plus classical contraction is not enough

T1 asks that the **learned** map be a contraction. Could one ask less: that the *classical*
map contracts, and that the learned map is uniformly close to it? No.

**Plain statement.** On the real line take the classical map `T u = u / 2`, a contraction
with factor `1/2` and answer `0`, and the learned map

    Tt u = (u - tanh (10 u)) / 2 ,

which is within `1/2` of `T` at every point. `Tt` has a two-cycle: there is an `x` between
`0.3` and `0.34` with `Tt x = -x` and `Tt (-x) = x`. The learned iteration started at `x`
alternates between `x` and `-x` for ever and converges to nothing.

So both hypotheses hold (the classical map contracts, the learned one is uniformly close) and
the conclusion "the learned iteration converges" fails. What is missing is T1's hypothesis: a
Lipschitz bound below one **for the learned map**. Here `Tt` has slope `-4.5` at `0`.

This is the counterexample of the architecture document's Remark 10 to the convergence step
of SNI's Theorem 1 (Huang et al., ICLR 2026), with `τ = 1/2`, `c = 1`, `k = 10`. The point
`x` is a root of `tanh (10 x) = 3 x`, found by the intermediate value theorem; numerically it
is `0.332471`.
-/

namespace Atlas

/-- **The SNI remark.** The classical map `T u = u / 2` contracts with factor `1/2` and has
the fixed point `0`; the learned map `Tt u = (u - tanh (10 u)) / 2` is within `1/2` of it
everywhere; and `Tt` has a two-cycle `x ↦ -x ↦ x` with `0.3 ≤ x ≤ 0.34`, along which its
iterates are `(-1) ^ k * x`. -/
theorem sni_counterexample :
    let T : ℝ → ℝ := fun u => u / 2
    let Tt : ℝ → ℝ := fun u => (u - Real.tanh (10 * u)) / 2
    (∀ u v, dist (T u) (T v) ≤ 1 / 2 * dist u v) ∧ T 0 = 0 ∧
      (∀ u, dist (Tt u) (T u) ≤ 1 / 2) ∧
      ∃ x : ℝ, 0.3 ≤ x ∧ x ≤ 0.34 ∧ Tt x = -x ∧ Tt (-x) = x ∧
        ∀ k : ℕ, Tt^[k] x = (-1) ^ k * x := by
  sorry

end Atlas
