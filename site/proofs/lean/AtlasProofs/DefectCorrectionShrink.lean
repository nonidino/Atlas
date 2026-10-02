import Mathlib.Analysis.Calculus.FDeriv.Add
import Mathlib.Analysis.Calculus.FDeriv.Mul
import Mathlib.LinearAlgebra.Eigenspace.Basic
import Mathlib.Analysis.Complex.Basic

/-!
# T5, Corollary 2 — Shrinking the cheap map toward the null element

Defect correction converges at a rate set by `J_Ψ⁻¹ J_Φ`, where `J_Ψ = I - DΨ` is the
derivative of `G_Ψ = I - Ψ` at the settled state. A cheap map that is too close to singular on
some mode (`J_Ψ` nearly zero there) makes the iteration diverge.

**Plain statement.** Replace `Ψ` by `Ψ_α = (1 - α) Ψ`, with `α` between 0 and 1. Then

* `J_{Ψ_α} = α I + (1 - α) J_Ψ`;
* the eigenvalues of `J_{Ψ_α}` are exactly the numbers `α + (1 - α) μ`, with `μ` an eigenvalue
  of `J_Ψ`, on the same eigenvectors;
* so wherever an eigenvalue of `J_Ψ` has non-negative real part, the corresponding eigenvalue
  of `J_{Ψ_α}` has real part at least `α`: the shrunk map is no closer to singular than `α` on
  that mode.

`α = 1` is the constant map of Corollary 1 and `α = 0` is `Ψ` itself.
-/

namespace Atlas

namespace DefectCorrection

/-- **T5, Corollary 2 (the shrunk Jacobian).** If `Ψ` has derivative `D` at `w`, then
`G_{Ψ_α} = I - (1 - α) Ψ` has derivative `α I + (1 - α) (I - D)` there. -/
theorem shrink_hasFDerivAt {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {Ψ : E → E} {D : E →L[ℝ] E} {w : E} (α : ℝ) (hΨ : HasFDerivAt Ψ D w) :
    HasFDerivAt (fun x => x - (1 - α) • Ψ x)
      (α • (1 : E →L[ℝ] E) + (1 - α) • ((1 : E →L[ℝ] E) - D)) w := by
  -- The derivative of `x ↦ x - (1 - α) Ψ x` is `I - (1 - α) D = α I + (1 - α) (I - D)`.
  refine ((hasFDerivAt_id w).sub (hΨ.const_smul (1 - α))).congr_fderiv ?_
  ext x
  simp only [sub_apply, ContinuousLinearMap.id_apply, add_apply, smul_apply, one_apply_eq_self,
    smul_sub, sub_smul, one_smul]
  abel

variable {𝕜 V : Type*} [Field 𝕜] [AddCommGroup V] [Module 𝕜 V]

/-- **T5, Corollary 2 (eigenvalues move along with the shrink).** An eigenvalue `μ` of `J`
gives the eigenvalue `α + (1 - α) μ` of `α I + (1 - α) J`. -/
theorem shrink_hasEigenvalue (J : Module.End 𝕜 V) (α : 𝕜) {μ : 𝕜}
    (h : J.HasEigenvalue μ) :
    (α • (1 : Module.End 𝕜 V) + (1 - α) • J).HasEigenvalue (α + (1 - α) * μ) := by
  -- An eigenvector `v` of `J` for `μ` is one of `α I + (1 - α) J` for `α + (1 - α) μ`.
  obtain ⟨v, hv⟩ := h.exists_hasEigenvector
  refine Module.End.hasEigenvalue_of_hasEigenvector (x := v)
    (Module.End.hasEigenvector_iff.mpr ⟨Module.End.mem_eigenspace_iff.mpr ?_, hv.2⟩)
  rw [LinearMap.add_apply, LinearMap.smul_apply, LinearMap.smul_apply, Module.End.one_apply,
    hv.apply_eq_smul, smul_smul, add_smul]

/-- **T5, Corollary 2 (and those are all of them).** For `α ≠ 1`, every eigenvalue of
`α I + (1 - α) J` is `α + (1 - α) μ` for an eigenvalue `μ` of `J`. -/
theorem shrink_hasEigenvalue_iff (J : Module.End 𝕜 V) {α : 𝕜} (hα : α ≠ 1) (ν : 𝕜) :
    (α • (1 : Module.End 𝕜 V) + (1 - α) • J).HasEigenvalue ν ↔
      ∃ μ, J.HasEigenvalue μ ∧ ν = α + (1 - α) * μ := by
  have h1α : (1 - α) ≠ 0 := sub_ne_zero.mpr (Ne.symm hα)
  constructor
  · -- `(α I + (1 - α) J) v = ν v` gives `J v = ((ν - α) / (1 - α)) v`.
    intro h
    obtain ⟨v, hv⟩ := h.exists_hasEigenvector
    have hJv : J v = ((ν - α) / (1 - α)) • v := by
      have hval := hv.apply_eq_smul
      rw [LinearMap.add_apply, LinearMap.smul_apply, LinearMap.smul_apply,
        Module.End.one_apply] at hval
      have h2 : (1 - α) • J v = (ν - α) • v := by
        rw [sub_smul ν α v, ← hval]
        abel
      calc J v = (1 - α)⁻¹ • ((1 - α) • J v) := by rw [smul_smul, inv_mul_cancel₀ h1α, one_smul]
        _ = ((ν - α) / (1 - α)) • v := by rw [h2, smul_smul, div_eq_inv_mul]
    refine ⟨(ν - α) / (1 - α), Module.End.hasEigenvalue_of_hasEigenvector (x := v)
      (Module.End.hasEigenvector_iff.mpr ⟨Module.End.mem_eigenspace_iff.mpr hJv, hv.2⟩), ?_⟩
    rw [mul_div_cancel₀ _ h1α]
    abel
  · rintro ⟨μ, hμ, rfl⟩
    exact shrink_hasEigenvalue J α hμ

/-- **T5, Corollary 2 (the real part).** For `0 ≤ α ≤ 1`, a complex `μ` with non-negative
real part is moved to a number with real part at least `α`. -/
theorem shrink_re {α : ℝ} (hα₀ : 0 ≤ α) (hα₁ : α ≤ 1) {μ : ℂ} (hμ : 0 ≤ μ.re) :
    α ≤ ((α : ℂ) + (1 - (α : ℂ)) * μ).re := by
  -- `Re(α + (1 - α) μ) = α + (1 - α) Re μ`, and `(1 - α) Re μ ≥ 0`.
  have h : ((α : ℂ) + (1 - (α : ℂ)) * μ).re = α + (1 - α) * μ.re := by
    simp [Complex.add_re, Complex.mul_re]
  rw [h]
  exact le_add_of_nonneg_right (mul_nonneg (sub_nonneg.mpr hα₁) hμ)

end DefectCorrection

end Atlas
