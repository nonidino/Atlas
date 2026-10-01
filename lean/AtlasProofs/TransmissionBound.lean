import Mathlib.Analysis.Normed.Operator.Basic

/-!
# T4e — The transmission term `σ`, on the substructuring branch

The true interface trace `lamStar` satisfies the exact interface equation `Λ lamStar = χ`.
The trace `lamDag` actually computed satisfies the equation posed with an approximate
interface operator, `Λt lamDag = χ`.

**Plain statement.** If the approximate operator is bounded below by `β > 0` (its smallest
singular value), the two traces differ by at most `‖Λ - Λt‖ * ‖lamStar‖ / β`. If the composed
step is `Cμ`-Lipschitz in its interface datum, the transmission term of the master bound is
at most `Cμ / β * ‖Λ - Λt‖ * ‖lamStar‖`.

Iterating the interface solve does not reduce this term: it is set by how wrong the operator
is, and amplified by `1 / β`.
-/

namespace Atlas

variable {M F : Type*} [NormedAddCommGroup M] [NormedSpace ℝ M]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- **T4e (the trace perturbation).** -/
theorem trace_perturbation (Λ Λt : M →L[ℝ] F) (χ : F) (lamStar lamDag : M) {β : ℝ}
    (hβ : 0 < β) (hΛt : ∀ x, β * ‖x‖ ≤ ‖Λt x‖)
    (hstar : Λ lamStar = χ) (hdag : Λt lamDag = χ) :
    ‖lamDag - lamStar‖ ≤ ‖Λ - Λt‖ * ‖lamStar‖ / β := by
  sorry

/-- **T4e (the transmission term).** `Φ lam v` is the composed step forced to use the
interface datum `lam`; it is `Cμ`-Lipschitz in `lam` at the state `v`. -/
theorem transmission_bound {E : Type*} [SeminormedAddCommGroup E]
    (Λ Λt : M →L[ℝ] F) (χ : F) (lamStar lamDag : M) {β Cμ : ℝ}
    (hβ : 0 < β) (hΛt : ∀ x, β * ‖x‖ ≤ ‖Λt x‖)
    (hstar : Λ lamStar = χ) (hdag : Λt lamDag = χ)
    (Φ : M → E → E) (v : E) (hCμ : 0 ≤ Cμ)
    (hΦ : ∀ lam lam', ‖Φ lam v - Φ lam' v‖ ≤ Cμ * ‖lam - lam'‖) :
    ‖Φ lamDag v - Φ lamStar v‖ ≤ Cμ / β * (‖Λ - Λt‖ * ‖lamStar‖) := by
  sorry

end Atlas
