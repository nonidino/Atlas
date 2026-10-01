import AtlasProofs.GramPort
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Normed.Operator.Basic
import Mathlib.Analysis.Normed.Module.FiniteDimension

/-!
# T8 — Wave variables: a passive piece does not amplify waves

On a side of a piece, `e` is the vector of boundary values (the effort: a temperature) and
`f` the flux into the piece (the flow). For an impedance `Z > 0` the incoming and outgoing
waves are, up to the common factor `1 / (2 √Z)`,

    a = e + Z f,        b = e - Z f.

A source-free linear piece has `f = Λ e`, and maps incoming to outgoing waves by its
**scattering map**, the Cayley transform `S = (I - Z Λ)(I + Z Λ)⁻¹`.

**Plain statement.**

* (the identity) `|b|² = |a|² - 4 Z ⟨f, e⟩`: the outgoing wave is the incoming one minus the
  power the piece absorbs.
* (the bound) If the piece absorbs at least `c |e|²` (`c ≥ 0`) and its flux is at most
  `M |e|`, then `|b|² ≤ (1 - 4 Z c / (1 + Z M)²) |a|²`. This needs no linearity: it holds for
  any pair `e, f`, so also for the *difference* of two states of a nonlinear piece.
* (the scattering map) For a linear `Λ` on a finite-dimensional space with those two
  properties, `I + Z Λ` is invertible and `‖S‖² ≤ 1 - 4 Z c / (1 + Z M)²`.
* (`c = 0`) A piece that only never generates power, `⟨Λ e, e⟩ ≥ 0`, has `‖S‖ ≤ 1`.
* (tier 0 serves at tier 2) The Gram port matrix of T25 is positive semidefinite for every
  network output, so its scattering matrix is non-expansive for every impedance.

The identity and the bound hold in any real inner-product space. Invertibility is stated in
finite dimensions.
-/

namespace Atlas

open Matrix

section Waves

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- **T8 (the wave identity).** `|e - Z f|² = |e + Z f|² - 4 Z ⟨f, e⟩`. -/
theorem wave_identity (e f : E) (Z : ℝ) :
    ‖e - Z • f‖ ^ 2 = ‖e + Z • f‖ ^ 2 - 4 * Z * inner ℝ f e := by
  sorry

/-- **T8 (the bound, for any effort and flow).** If `⟨f, e⟩ ≥ c |e|²` with `c ≥ 0` and
`|f| ≤ M |e|`, then the outgoing wave is smaller than the incoming one by the factor
`1 - 4 Z c / (1 + Z M)²`, in squared norm. -/
theorem wave_bound {e f : E} {Z c M : ℝ} (hZ : 0 < Z) (hc : 0 ≤ c)
    (hacc : c * ‖e‖ ^ 2 ≤ inner ℝ f e) (hf : ‖f‖ ≤ M * ‖e‖) :
    ‖e - Z • f‖ ^ 2 ≤ (1 - 4 * Z * c / (1 + Z * M) ^ 2) * ‖e + Z • f‖ ^ 2 := by
  sorry

/-- **T8 (the Cayley bound).** For a bounded linear `Λ` with `⟨Λ e, e⟩ ≥ c |e|²`, `c ≥ 0`,
and `‖Λ‖ ≤ M`, and any `Z > 0`:
`|(I - Z Λ) e|² ≤ (1 - 4 Z c / (1 + Z M)²) |(I + Z Λ) e|²` for every `e`. -/
theorem cayley_bound (Λ : E →L[ℝ] E) {Z c M : ℝ} (hZ : 0 < Z) (hc : 0 ≤ c)
    (hacc : ∀ e, c * ‖e‖ ^ 2 ≤ inner ℝ (Λ e) e) (hM : ‖Λ‖ ≤ M) (e : E) :
    ‖e - Z • Λ e‖ ^ 2 ≤ (1 - 4 * Z * c / (1 + Z * M) ^ 2) * ‖e + Z • Λ e‖ ^ 2 := by
  sorry

/-- **T8 (the scattering map exists and contracts).** In finite dimensions, under the
hypotheses of `cayley_bound`, `I + Z Λ` has an inverse `B`, and the scattering map
`S = (I - Z Λ) B` satisfies `‖S‖² ≤ 1 - 4 Z c / (1 + Z M)²`.

The space must contain a non-zero vector: on the zero space every `c` satisfies the
hypothesis, and the right-hand side can then be negative. -/
theorem cayley_transform [FiniteDimensional ℝ E] [Nontrivial E] (Λ : E →L[ℝ] E)
    {Z c M : ℝ} (hZ : 0 < Z)
    (hc : 0 ≤ c) (hacc : ∀ e, c * ‖e‖ ^ 2 ≤ inner ℝ (Λ e) e) (hM : ‖Λ‖ ≤ M) :
    ∃ B : E →L[ℝ] E, B * (1 + Z • Λ) = 1 ∧ (1 + Z • Λ) * B = 1 ∧
      ‖(1 - Z • Λ) * B‖ ^ 2 ≤ 1 - 4 * Z * c / (1 + Z * M) ^ 2 := by
  sorry

/-- **T8 with `c = 0`: a passive piece is non-expansive.** If `⟨Λ e, e⟩ ≥ 0` for every `e`,
then in finite dimensions `I + Z Λ` is invertible and the scattering map has norm at most
one, for every impedance `Z > 0`. -/
theorem cayley_nonexpansive [FiniteDimensional ℝ E] (Λ : E →L[ℝ] E) {Z : ℝ} (hZ : 0 < Z)
    (hpos : ∀ e, 0 ≤ inner ℝ (Λ e) e) :
    ∃ B : E →L[ℝ] E, B * (1 + Z • Λ) = 1 ∧ (1 + Z • Λ) * B = 1 ∧
      ‖(1 - Z • Λ) * B‖ ≤ 1 := by
  sorry

end Waves

section Matrices

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **T8 for a positive semidefinite matrix.** If `L` is positive semidefinite and `Z > 0`,
then `1 + Z L` has an inverse `B`, and the scattering matrix `S = (1 - Z L) B` does not
lengthen any vector: `|S a|² ≤ |a|²`. -/
theorem cayley_matrix_nonexpansive {L : Matrix ι ι ℝ} (hL : L.PosSemidef) {Z : ℝ}
    (hZ : 0 < Z) :
    ∃ B : Matrix ι ι ℝ, B * (1 + Z • L) = 1 ∧ (1 + Z • L) * B = 1 ∧
      ∀ a : ι → ℝ, (((1 - Z • L) * B) *ᵥ a) ⬝ᵥ (((1 - Z • L) * B) *ᵥ a) ≤ a ⬝ᵥ a := by
  sorry

/-- **T8 at tier 0: a learned superelement also serves at tier 2.** For *any* fields `h` a
network returned, the scattering matrix formed from the Gram port matrix is non-expansive,
for every impedance `Z > 0`. No certificate of the network is needed. -/
theorem Piece.gram_cayley_nonexpansive {U F : Type*} [AddCommGroup U] [Module ℝ U]
    [AddCommGroup F] [Module ℝ F] (P : Piece U F) (h : ι → U) (q : ι → F) {Z : ℝ}
    (hZ : 0 < Z) :
    ∃ B : Matrix ι ι ℝ, B * (1 + Z • P.gram h q) = 1 ∧ (1 + Z • P.gram h q) * B = 1 ∧
      ∀ a : ι → ℝ, (((1 - Z • P.gram h q) * B) *ᵥ a) ⬝ᵥ (((1 - Z • P.gram h q) * B) *ᵥ a)
        ≤ a ⬝ᵥ a := by
  sorry

end Matrices

end Atlas
