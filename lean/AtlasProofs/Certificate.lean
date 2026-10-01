import Mathlib.Topology.MetricSpace.Contracting
import Mathlib.Analysis.Normed.Operator.Basic

/-!
# T7 — The a-posteriori certificate

A solver returns a state `w`. Its **residual** `dist w (Φ w)` can be computed: apply the
classical step `Φ` once and see how far `w` moves. The certificate turns that computable
number into a bound on the error `dist w wstar`, which cannot be computed because the settled
state `wstar` is unknown.

**Plain statement.** If `Φ` brings `w` closer to the settled state by a factor `L < 1`, then
the error is at most the residual divided by `1 - L`.

**What the constant must be (the lesson of W208).** The factor `L` has to hold *at the state
being certified*. A constant read off one approach to `wstar` says nothing about a state that
approaches from another direction. For a linear step this is exact: the smallest constant
that certifies *every* state is an operator norm, a supremum over all directions, and any
smaller number fails at some state. The last two declarations of this file say so.
-/

namespace Atlas

/-- **T7 (a-posteriori certificate), with exactly the hypothesis it needs.** `wstar` is a
fixed point of `Φ`, and `Φ` contracts the pair `(w, wstar)` by `L < 1`. Then the error of `w`
is at most its residual over `1 - L`. The contraction is required of *this* `w`, not of some
other state. -/
theorem certificate {X : Type*} [MetricSpace X] {Φ : X → X} {L : ℝ} {w wstar : X}
    (hL : L < 1)
    (hstar : Φ wstar = wstar)
    (hw : dist (Φ w) (Φ wstar) ≤ L * dist w wstar) :
    dist w wstar ≤ dist w (Φ w) / (1 - L) := by
  -- `d(w, w⋆) ≤ d(w, Φ w) + d(Φ w, Φ w⋆) ≤ d(w, Φ w) + L d(w, w⋆)`, and `1 - L > 0`.
  have h : dist w wstar ≤ dist w (Φ w) + L * dist w wstar :=
    calc dist w wstar ≤ dist w (Φ w) + dist (Φ w) wstar := dist_triangle _ _ _
      _ = dist w (Φ w) + dist (Φ w) (Φ wstar) := by rw [hstar]
      _ ≤ dist w (Φ w) + L * dist w wstar := add_le_add le_rfl hw
  rw [le_div_iff₀ (sub_pos.mpr hL)]
  linarith

/-- **T7, Banach's form.** If `Φ` contracts *every* pair by `L < 1` on a complete space, it has
a unique fixed point `wstar`, and every state `w` satisfies the certificate. -/
theorem certificate_of_contraction {X : Type*} [MetricSpace X] [CompleteSpace X] [Nonempty X]
    {Φ : X → X} {L : ℝ} (hL₀ : 0 ≤ L) (hL₁ : L < 1)
    (hΦ : ∀ x y, dist (Φ x) (Φ y) ≤ L * dist x y) :
    ∃ wstar : X, Φ wstar = wstar ∧ (∀ v, Φ v = v → v = wstar) ∧
      ∀ w, dist w wstar ≤ dist w (Φ w) / (1 - L) := by
  -- Banach's theorem gives `wstar`; the certificate then applies to every `w`.
  have hKL : ((Real.toNNReal L : NNReal) : ℝ) = L := Real.coe_toNNReal L hL₀
  have hK : ContractingWith (Real.toNNReal L) Φ :=
    ⟨by rw [← NNReal.coe_lt_coe, hKL, NNReal.coe_one]; exact hL₁,
      LipschitzWith.of_dist_le_mul fun x y => by rw [hKL]; exact hΦ x y⟩
  have hstar : Φ (ContractingWith.fixedPoint Φ hK) = ContractingWith.fixedPoint Φ hK :=
    ContractingWith.fixedPoint_isFixedPt hK
  exact ⟨ContractingWith.fixedPoint Φ hK, hstar,
    fun v hv => ContractingWith.fixedPoint_unique hK hv,
    fun w => certificate hL₁ hstar (hΦ _ _)⟩

/-- **T7 for a linear step: the certificate's constant is an operator norm (W208).**
Let `Φ w = A w + b` with `I - A` invertible, inverse `B`, and settled state `wstar`. Then

* every state satisfies `‖w - wstar‖ ≤ ‖B‖ * ‖Φ w - w‖`, and
* `‖B‖` is the least such constant: any `C ≥ 0` that certifies every state is at least `‖B‖`.

So a constant below the operator norm `‖(I - A)⁻¹‖`, such as an error-to-residual ratio read
along one march, under-bounds the error of some state. -/
theorem certificate_constant_is_opNorm {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (A B : E →L[ℝ] E) (b wstar : E)
    (hBA : B * (1 - A) = 1) (hAB : (1 - A) * B = 1)
    (hstar : A wstar + b = wstar) :
    (∀ w, ‖w - wstar‖ ≤ ‖B‖ * ‖(A w + b) - w‖) ∧
      ∀ C : ℝ, 0 ≤ C → (∀ w, ‖w - wstar‖ ≤ C * ‖(A w + b) - w‖) → ‖B‖ ≤ C := by
  -- `B` inverts `I - A` on both sides, pointwise.
  have hBx : ∀ x, B ((1 - A) x) = x := fun x => by
    have h := DFunLike.congr_fun hBA x
    rwa [mul_apply_eq_comp, one_apply_eq_self] at h
  have hxB : ∀ x, (1 - A) (B x) = x := fun x => by
    have h := DFunLike.congr_fun hAB x
    rwa [mul_apply_eq_comp, one_apply_eq_self] at h
  -- The residual is `Φ(w) - w = -(I - A)(w - wstar)`.
  have key : ∀ w, (A w + b) - w = -((1 - A) (w - wstar)) := fun w => by
    have hb : b = wstar - A wstar := eq_sub_of_add_eq' hstar
    rw [hb, sub_apply, one_apply_eq_self, map_sub]
    abel
  refine ⟨fun w => ?_, fun C hC hcert => ?_⟩
  · -- `w - wstar = -B (Φ(w) - w)`
    have hw : w - wstar = -B ((A w + b) - w) := by rw [key w, map_neg, neg_neg, hBx]
    rw [hw, norm_neg]
    exact B.le_opNorm _
  · -- Certify `w = wstar + B x`: its residual is `-x`, so `‖B x‖ ≤ C ‖x‖` for every `x`.
    refine B.opNorm_le_bound hC fun x => ?_
    have h := hcert (wstar + B x)
    rwa [key, add_sub_cancel_left, hxB, norm_neg] at h

/-- **The two-mode example of W208**, the one `tests/test_tier48_defect_correction.py` pins.
For the step `Φ (x, y) = (0.6 x, 0.95 y)`, whose settled state is `0`, the error is `2.5` times
the residual along the fast mode and `20` times along the slow one. A constant of `2.5`, read
off a march that stays in the fast mode, under-bounds the slow mode by a factor of eight. -/
theorem certificate_two_mode_example :
    let Φ : ℝ × ℝ → ℝ × ℝ := fun w => (0.6 * w.1, 0.95 * w.2)
    ‖((1 : ℝ), (0 : ℝ))‖ = 2.5 * ‖Φ (1, 0) - (1, 0)‖ ∧
      ‖((0 : ℝ), (1 : ℝ))‖ = 20 * ‖Φ (0, 1) - (0, 1)‖ := by
  -- The maximum norm: `‖(1, 0)‖ = 1 = 2.5 * 0.4` and `‖(0, 1)‖ = 1 = 20 * 0.05`.
  intro Φ
  simp only [Φ, Prod.mk_sub_mk, Prod.norm_mk, Real.norm_eq_abs]
  norm_num

end Atlas
