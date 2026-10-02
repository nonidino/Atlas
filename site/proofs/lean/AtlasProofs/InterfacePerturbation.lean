import Mathlib.Analysis.Normed.Operator.Basic
import Mathlib.Analysis.InnerProductSpace.Basic

/-!
# T24 — Perturbation of the interface solve

The exact interface system is `S c = χ`. The host solves a learned one, `St ct = χt`: both
the matrix and the right-hand side are approximate.

**Plain statement.** If the learned matrix stretches every vector by at least `β > 0`, the
two answers differ by at most

    ( ‖S - St‖ * ‖c‖ + ‖χ - χt‖ ) / β .

For the Gram construction of T25 the constant `β` need not be measured on the learned
matrix: the learned matrix dominates the exact one, so the exact matrix's own coercivity
constant serves. That is the second statement.

An ill-conditioned interface (small `β`) makes the same error in the matrix and in the load
cost more; no network output can make `β` smaller than the classical problem's.

The first statement holds in any real normed spaces; the second in any real inner-product
space.
-/

namespace Atlas

/-- **T24 (perturbation of the interface solve).** `St` is bounded below by `β > 0`. Then
the learned answer `ct` is within `(‖S - St‖ * ‖c‖ + ‖χ - χt‖) / β` of the exact one. -/
theorem interface_perturbation {M F : Type*} [NormedAddCommGroup M] [NormedSpace ℝ M]
    [NormedAddCommGroup F] [NormedSpace ℝ F]
    (S St : M →L[ℝ] F) (χ χt : F) (c ct : M) {β : ℝ}
    (hβ : 0 < β) (hSt : ∀ x, β * ‖x‖ ≤ ‖St x‖)
    (hc : S c = χ) (hct : St ct = χt) :
    ‖ct - c‖ ≤ (‖S - St‖ * ‖c‖ + ‖χ - χt‖) / β := by
  have h1 : St (ct - c) = (S - St) c - (χ - χt) := by
    rw [map_sub, hct, sub_apply, hc]
    abel
  have h2 : β * ‖ct - c‖ ≤ ‖S - St‖ * ‖c‖ + ‖χ - χt‖ :=
    calc β * ‖ct - c‖ ≤ ‖St (ct - c)‖ := hSt _
      _ = ‖(S - St) c - (χ - χt)‖ := by rw [h1]
      _ ≤ ‖(S - St) c‖ + ‖χ - χt‖ := norm_sub_le _ _
      _ ≤ ‖S - St‖ * ‖c‖ + ‖χ - χt‖ := by linarith [(S - St).le_opNorm c]
  rw [le_div_iff₀ hβ]
  linarith

/-- **T24 with T25 (iii): the constant comes from the exact problem.** If the exact matrix is
coercive with constant `β > 0`, `β ‖x‖² ≤ ⟨S x, x⟩`, and the learned matrix dominates it,
`⟨S x, x⟩ ≤ ⟨St x, x⟩` (which T25 (iii) proves for every network output), then the same bound
holds with the exact problem's `β`. -/
theorem interface_perturbation_of_dominates {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E]
    (S St : E →L[ℝ] E) (χ χt c ct : E) {β : ℝ}
    (hβ : 0 < β) (hS : ∀ x, β * ‖x‖ ^ 2 ≤ inner ℝ (S x) x)
    (hdom : ∀ x, inner ℝ (S x) x ≤ inner ℝ (St x) x)
    (hc : S c = χ) (hct : St ct = χt) :
    ‖ct - c‖ ≤ (‖S - St‖ * ‖c‖ + ‖χ - χt‖) / β := by
  refine interface_perturbation S St χ χt c ct hβ (fun x => ?_) hc hct
  by_cases hx : x = 0
  · simp [hx]
  have hxpos : 0 < ‖x‖ := norm_pos_iff.mpr hx
  have h1 : β * ‖x‖ ^ 2 ≤ ‖St x‖ * ‖x‖ :=
    (hS x).trans ((hdom x).trans (real_inner_le_norm _ _))
  have h2 : β * ‖x‖ * ‖x‖ ≤ ‖St x‖ * ‖x‖ := by
    rw [mul_assoc, ← sq]
    exact h1
  exact le_of_mul_le_mul_right h2 hxpos

end Atlas
