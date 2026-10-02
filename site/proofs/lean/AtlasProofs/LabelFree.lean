import AtlasProofs.GramPort

/-!
# T26 — The label-free training objective

To train a network on a piece, the supervised loss compares its fields with the exact ones,
which a classical solver has to compute first:

    L_sup = ∑ k, ‖h k - H k‖²  +  ‖up - u_p‖²        (in the `A_I` energy).

The label-free objective uses only the network's own output and the piece's energy:

    L_en = trace Λt  +  upᵀ A_I up  -  2 f · up .

**Plain statement.** `L_en = L_sup + C`, where `C = trace Λ - u_pᵀ A_I u_p` is a number fixed
by the piece and its source: it does not involve the network's output. So the two objectives
have the same differences between any two outputs, hence the same gradients and the same
minimisers, and the second needs no solved examples.

`H k` are the exact constraint modes (`A_I (H k) = B (q k)`) and `u_p` the exact particular
field (`A_I u_p = f`). The identity is exact in real arithmetic; in floating point the two
objectives differ, which the architecture document registers as a comparison.
-/

namespace Atlas

namespace Piece

open Matrix

variable {U F : Type*} [AddCommGroup U] [Module ℝ U] [AddCommGroup F] [Module ℝ F]
  (P : Piece U F) {ι : Type*} [Fintype ι]

/-- The **label-free objective** for the returned modes `h` and particular field `up`:
`trace Λt + upᵀ A_I up - 2 f · up`. It needs no exact solution. -/
def labelFree (q : ι → F) (f : U →ₗ[ℝ] ℝ) (h : ι → U) (up : U) : ℝ :=
  (P.gram h q).trace + P.a up up - 2 * f up

/-- The **supervised energy-norm loss** against the exact modes `hex` and the exact
particular field `upx`. -/
def supervised (hex : ι → U) (upx : U) (h : ι → U) (up : U) : ℝ :=
  ∑ k, P.a (h k - hex k) (h k - hex k) + P.a (up - upx) (up - upx)

/-- **T26 (the label-free objective).** The label-free objective equals the supervised loss
plus `trace Λ - upxᵀ A_I upx`, which does not involve the network's output `h, up`. -/
theorem labelFree_eq_supervised_add_const (q : ι → F) (f : U →ₗ[ℝ] ℝ) (hex : ι → U) (upx : U)
    (hex_def : ∀ k, P.IsExactMode (hex k) (q k)) (hupx : ∀ v, P.a v upx = f v)
    (h : ι → U) (up : U) :
    P.labelFree q f h up
      = P.supervised hex upx h up + ((P.gram hex q).trace - P.a upx upx) := by
  have htr := P.trace_gram_sub h hex q hex_def
  have hsq : P.a (up - upx) (up - upx) = P.a up up - 2 * f up + P.a upx upx := by
    simp only [map_sub, LinearMap.sub_apply]
    rw [P.a_symm upx up, hupx up]
    ring
  simp only [labelFree, supervised]
  linarith

/-- **T26, as the statement about training.** For any two outputs of the network, the
label-free objective and the supervised loss change by the same amount. -/
theorem labelFree_sub_eq_supervised_sub (q : ι → F) (f : U →ₗ[ℝ] ℝ) (hex : ι → U) (upx : U)
    (hex_def : ∀ k, P.IsExactMode (hex k) (q k)) (hupx : ∀ v, P.a v upx = f v)
    (h₁ h₂ : ι → U) (up₁ up₂ : U) :
    P.labelFree q f h₁ up₁ - P.labelFree q f h₂ up₂
      = P.supervised hex upx h₁ up₁ - P.supervised hex upx h₂ up₂ := by
  rw [P.labelFree_eq_supervised_add_const q f hex upx hex_def hupx h₁ up₁,
    P.labelFree_eq_supervised_add_const q f hex upx hex_def hupx h₂ up₂]
  ring

end Piece

end Atlas
