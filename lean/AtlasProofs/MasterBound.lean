import Mathlib.Analysis.Normed.Group.Basic
import Mathlib.Algebra.Field.GeomSum

/-!
# T4 — The master error bound

A simulation marches `u (n+1) = step (u n)`. The true solution, sampled at the same instants,
is `ustar n`. The **error** is `u n - ustar n`.

**Plain statement.**

* (recursion) The new error is the old error carried through one step, plus the **one-step
  defect**: what a single step does wrong when started from the true solution.
* (three-term split) With a coupling, the defect is a sum of three named parts: the experts
  being wrong even with the true interface data (`τ`), the interface problem being posed with
  the wrong operator (`σ`), and the interface solve being stopped early (`γ`).
* (accumulation) If one step magnifies a difference by at most `L`, the error after `N` steps
  is at most `L ^ N` times the initial error, plus the defects, each magnified by the steps
  that follow it.
* (three regimes) With every defect at most `δ`: the error stays below `δ / (1 - L)` for all
  time when `L < 1`; it grows at most linearly, `N δ`, when `L = 1`; and the bound grows
  exponentially when `L > 1`.

A monolithic network obeys the same law. What a decomposition changes is the constants.

The state space is any normed vector space (finite-dimensional or not).
-/

namespace Atlas

open Finset

variable {E : Type*} [SeminormedAddCommGroup E]

/-- **T4a (the exact error recursion).** An identity: the new error is the propagated old
error plus the one-step defect `step (ustar n) - ustar (n + 1)`. -/
theorem error_recursion (step : E → E) (u ustar : ℕ → E)
    (hu : ∀ n, u (n + 1) = step (u n)) (n : ℕ) :
    u (n + 1) - ustar (n + 1) =
      (step (u n) - step (ustar n)) + (step (ustar n) - ustar (n + 1)) := by
  sorry

/-- **T4b (the three-term split).** `Φ lam v` is the composed step forced to use the interface
datum `lam`, and `exact v` is the true evolution. For any three data `lamStar` (the true
solution's trace), `lamDag` (the root of the interface problem actually posed) and `lamK` (what
the solver returned), the defect is `τ + σ + γ`, exactly. -/
theorem defect_split {Λ : Type*} (Φ : Λ → E → E) (exact : E → E)
    (lamStar lamDag lamK : Λ) (v : E) :
    Φ lamK v - exact v =
      (Φ lamStar v - exact v) + (Φ lamDag v - Φ lamStar v) + (Φ lamK v - Φ lamDag v) := by
  sorry

/-- **T4c (accumulation; a discrete Gronwall inequality).** If `e (n+1) ≤ L * e n + d (n+1)`
at every step, with `L ≥ 0`, then `e N ≤ L ^ N * e 0 + ∑ n = 1..N, L ^ (N - n) * d n`.
(The sum is written over `i = n - 1`.) -/
theorem accumulation {L : ℝ} (hL : 0 ≤ L) {e d : ℕ → ℝ}
    (h : ∀ n, e (n + 1) ≤ L * e n + d (n + 1)) (N : ℕ) :
    e N ≤ L ^ N * e 0 + ∑ i ∈ range N, L ^ (N - (i + 1)) * d (i + 1) := by
  sorry

/-- **T4 (the master bound).** If the step magnifies the difference between the computed and
the true trajectory by at most `L` at every instant, then the error after `N` steps is at most
`L ^ N` times the initial error plus the accumulated one-step defects.
The stability hypothesis is needed only along the two trajectories. -/
theorem master_bound {step : E → E} {L : ℝ} (hL : 0 ≤ L) {u ustar : ℕ → E}
    (hu : ∀ n, u (n + 1) = step (u n))
    (hstep : ∀ n, ‖step (u n) - step (ustar n)‖ ≤ L * ‖u n - ustar n‖) (N : ℕ) :
    ‖u N - ustar N‖ ≤ L ^ N * ‖u 0 - ustar 0‖
      + ∑ i ∈ range N, L ^ (N - (i + 1)) * ‖step (ustar i) - ustar (i + 1)‖ := by
  sorry

/-- **T4 with the three terms named** (the boxed bound of the master-error-bound page).
The composed step uses the datum `lamK v` its interface solver returns at state `v`. At step
`n`, `lamStar n` is the true trace and `lamDag n` the root of the posed interface problem. -/
theorem master_bound_three_terms {Λ : Type*} {Φ : Λ → E → E} {lamK : E → Λ} {L : ℝ}
    (hL : 0 ≤ L) {u ustar : ℕ → E} (lamStar lamDag : ℕ → Λ)
    (hu : ∀ n, u (n + 1) = Φ (lamK (u n)) (u n))
    (hstep : ∀ n, ‖Φ (lamK (u n)) (u n) - Φ (lamK (ustar n)) (ustar n)‖ ≤ L * ‖u n - ustar n‖)
    (N : ℕ) :
    ‖u N - ustar N‖ ≤ L ^ N * ‖u 0 - ustar 0‖
      + ∑ i ∈ range N, L ^ (N - (i + 1)) *
          (‖Φ (lamStar i) (ustar i) - ustar (i + 1)‖
            + ‖Φ (lamDag i) (ustar i) - Φ (lamStar i) (ustar i)‖
            + ‖Φ (lamK (ustar i)) (ustar i) - Φ (lamDag i) (ustar i)‖) := by
  sorry

section Regimes

variable {step : E → E} {L δ : ℝ} {u ustar : ℕ → E}

/-- **T4, contractive regime (`L < 1`): bounded for all time.** -/
theorem master_bound_contractive (hL₀ : 0 ≤ L) (hL₁ : L < 1)
    (hu : ∀ n, u (n + 1) = step (u n))
    (hstep : ∀ n, ‖step (u n) - step (ustar n)‖ ≤ L * ‖u n - ustar n‖)
    (hd : ∀ n, ‖step (ustar n) - ustar (n + 1)‖ ≤ δ) (N : ℕ) :
    ‖u N - ustar N‖ ≤ L ^ N * ‖u 0 - ustar 0‖ + δ / (1 - L) := by
  sorry

/-- **T4, non-expansive regime (`L = 1`): at most linear growth.** -/
theorem master_bound_nonexpansive
    (hu : ∀ n, u (n + 1) = step (u n))
    (hstep : ∀ n, ‖step (u n) - step (ustar n)‖ ≤ ‖u n - ustar n‖)
    (hd : ∀ n, ‖step (ustar n) - ustar (n + 1)‖ ≤ δ) (N : ℕ) :
    ‖u N - ustar N‖ ≤ ‖u 0 - ustar 0‖ + N * δ := by
  sorry

/-- **T4, expansive regime (`L > 1`): the bound grows exponentially.** -/
theorem master_bound_expansive (hL : 1 < L)
    (hu : ∀ n, u (n + 1) = step (u n))
    (hstep : ∀ n, ‖step (u n) - step (ustar n)‖ ≤ L * ‖u n - ustar n‖)
    (hd : ∀ n, ‖step (ustar n) - ustar (n + 1)‖ ≤ δ) (N : ℕ) :
    ‖u N - ustar N‖ ≤ L ^ N * ‖u 0 - ustar 0‖ + (L ^ N - 1) / (L - 1) * δ := by
  sorry

end Regimes

/-- **The three regimes are attained**, so they are not artefacts of the proof. For the scalar
step `x ↦ L x` with a defect of exactly `δ` every step and no initial error, the error after
`N` steps is exactly `∑ i < N, L ^ i * δ`: the geometric sum whose three behaviours the regimes
describe. -/
theorem master_bound_attained (L δ : ℝ) (N : ℕ) :
    ∃ u ustar : ℕ → ℝ, (∀ n, u (n + 1) = L * u n) ∧ (∀ n, L * ustar n - ustar (n + 1) = δ) ∧
      u 0 - ustar 0 = 0 ∧ u N - ustar N = ∑ i ∈ range N, L ^ i * δ := by
  sorry

end Atlas
