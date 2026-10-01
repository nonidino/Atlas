import Mathlib.Analysis.Normed.Group.Basic
import Mathlib.Topology.Algebra.Group.Basic

/-!
# T5 — Defect correction: a learned map can change the rate, never the answer

`Φ` is the classical step, whose settled states are its fixed points. `Ψ` is any other map
on the same states: a learned operator, a coarse solver, anything cheap. **Defect correction**
(Stetter 1978) builds an iteration from the two. With `G_Φ = I - Φ` and `G_Ψ = I - Ψ`, the next
iterate `w'` is defined by

    G_Ψ w' = G_Ψ w - G_Φ w,        that is,     w' - Ψ w' = Φ w - Ψ w.

The cheap map enters only through a difference, and the classical map supplies the defect.

**Plain statement.**

* (Theorem 1) If the iterates settle anywhere, they settle on a fixed point of the classical
  map. Nothing about the accuracy of `Ψ` enters: only that both maps are continuous at the
  limit. A wrong cheap map can cost classical calls; it cannot change what is returned.
* (Theorem 1, as the code runs it) The same holds when each step solves the equation above
  only approximately, as long as the leftover tends to zero.
* (Corollary 1) If `Ψ` is constant, the iteration **is** the classical march: `w' = Φ w`.

The states form any normed vector space (finite-dimensional or not).
-/

namespace Atlas

namespace DefectCorrection

open Filter Topology

variable {E : Type*} [NormedAddCommGroup E]

/-- `w'` is the exact defect-correction successor of `w`: `G_Ψ w' = G_Ψ w - G_Φ w`. -/
def IsStep (Φ Ψ : E → E) (w w' : E) : Prop :=
  w' - Ψ w' = (w - Ψ w) - (w - Φ w)

/-- **T5, Theorem 1 (consistency, whatever `Ψ` is).** If a sequence of exact
defect-correction steps converges to `wbar`, and `Φ` and `Ψ` are continuous at `wbar`, then
`wbar` is a fixed point of the classical map `Φ`. -/
theorem limit_isFixedPt {Φ Ψ : E → E} {w : ℕ → E} {wbar : E}
    (hstep : ∀ k, IsStep Φ Ψ (w k) (w (k + 1)))
    (hΦ : ContinuousAt Φ wbar) (hΨ : ContinuousAt Ψ wbar)
    (hlim : Tendsto w atTop (𝓝 wbar)) :
    Φ wbar = wbar := by
  sorry

/-- **T5, Theorem 1 with inexact inner solves.** Each step may miss the defect-correction
equation by `ε k`. If `ε k → 0`, the limit is still a fixed point of `Φ`. -/
theorem limit_isFixedPt_of_inexact {Φ Ψ : E → E} {w : ℕ → E} {wbar : E} {ε : ℕ → ℝ}
    (hstep : ∀ k, ‖(w (k + 1) - Ψ (w (k + 1))) - ((w k - Ψ (w k)) - (w k - Φ (w k)))‖ ≤ ε k)
    (hε : Tendsto ε atTop (𝓝 0))
    (hΦ : ContinuousAt Φ wbar) (hΨ : ContinuousAt Ψ wbar)
    (hlim : Tendsto w atTop (𝓝 wbar)) :
    Φ wbar = wbar := by
  sorry

/-- **T5, Corollary 1 (the null element).** If `Ψ` is constant, the defect-correction
successor of `w` is `Φ w`: the iteration is the classical march. -/
theorem step_of_const {Φ Ψ : E → E} (hΨ : ∀ x y, Ψ x = Ψ y) {w w' : E}
    (h : IsStep Φ Ψ w w') : w' = Φ w := by
  sorry

/-- The cheap fixed-point march that solves for the successor: it starts from `Φ w` and
repeats `x ↦ Φ w + (Ψ x - Ψ w)`. One classical call, any number of cheap ones. -/
def innerMarch (Φ Ψ : E → E) (w : E) : ℕ → E
  | 0 => Φ w
  | m + 1 => Φ w + (Ψ (innerMarch Φ Ψ w m) - Ψ w)

/-- A point is a defect-correction successor of `w` exactly when the inner march leaves it
unchanged. -/
theorem isStep_iff {Φ Ψ : E → E} (w x : E) :
    IsStep Φ Ψ w x ↔ x = Φ w + (Ψ x - Ψ w) := by
  sorry

/-- **T5, Corollary 1 for the inner march.** If `Ψ` is constant, the inner march returns
`Φ w` at every stage, so it has converged after one cheap call. -/
theorem innerMarch_of_const {Φ Ψ : E → E} (hΨ : ∀ x y, Ψ x = Ψ y) (w : E) (m : ℕ) :
    innerMarch Φ Ψ w m = Φ w := by
  sorry

end DefectCorrection

end Atlas
