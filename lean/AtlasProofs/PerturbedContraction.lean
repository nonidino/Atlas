import Mathlib.Topology.MetricSpace.Contracting

/-!
# T1 — Perturbed contraction

A coupled iteration built from learned experts is a map `Tl` on the interface data.
The classical decomposition is another map `Tc`, with answer `ac` (`Tc ac = ac`).

**Plain statement.** If the learned map shrinks every distance by a factor `ρ < 1`, and at
the classical answer it differs from the classical map by at most `δ`, then:

* the learned iteration has exactly one answer `al`;
* from any start it reaches `al` geometrically: the distance shrinks by `ρ` every sweep;
* `al` is within `δ / (1 - ρ)` of the classical answer.

Nothing is assumed about how the learned map was trained, and the classical map need not
be a contraction: it only needs to have the answer `ac`.

The space is any complete metric space (finite-dimensional vectors with any norm are one).
-/

namespace Atlas

variable {X : Type*} [MetricSpace X] [CompleteSpace X]

/-- **T1 (perturbed contraction).** `Tl` is a `ρ`-contraction, `ac` is a fixed point of `Tc`,
and the two maps differ by at most `δ` at `ac`. Then `Tl` has a unique fixed point `al`, every
orbit of `Tl` converges to it at rate `ρ`, and `dist al ac ≤ δ / (1 - ρ)`. -/
theorem perturbed_contraction {Tc Tl : X → X} {ρ δ : ℝ} {ac : X}
    (hρ₀ : 0 ≤ ρ) (hρ₁ : ρ < 1)
    (hTl : ∀ x y, dist (Tl x) (Tl y) ≤ ρ * dist x y)
    (hac : Tc ac = ac)
    (hδ : dist (Tl ac) (Tc ac) ≤ δ) :
    ∃ al : X, Tl al = al ∧ (∀ b, Tl b = b → b = al) ∧
      (∀ (x₀ : X) (k : ℕ), dist (Tl^[k] x₀) al ≤ ρ ^ k * dist x₀ al) ∧
      dist al ac ≤ δ / (1 - ρ) := by
  sorry

/-- **T1, with the defect bounded everywhere** (the form the plan states): the same conclusion
when `dist (Tl x) (Tc x) ≤ δ` for every `x`. -/
theorem perturbed_contraction_uniform {Tc Tl : X → X} {ρ δ : ℝ} {ac : X}
    (hρ₀ : 0 ≤ ρ) (hρ₁ : ρ < 1)
    (hTl : ∀ x y, dist (Tl x) (Tl y) ≤ ρ * dist x y)
    (hac : Tc ac = ac)
    (hδ : ∀ x, dist (Tl x) (Tc x) ≤ δ) :
    ∃ al : X, Tl al = al ∧ (∀ b, Tl b = b → b = al) ∧
      (∀ (x₀ : X) (k : ℕ), dist (Tl^[k] x₀) al ≤ ρ ^ k * dist x₀ al) ∧
      dist al ac ≤ δ / (1 - ρ) := by
  sorry

/-- **T1 is sharp.** For every `ρ ∈ [0, 1)` and `δ ≥ 0` there are maps on the real line that
meet T1's hypotheses and whose answers are exactly `δ / (1 - ρ)` apart, so the bound cannot be
improved. (`Tc x = ρ x`, `Tl x = ρ x + δ`.) -/
theorem perturbed_contraction_sharp {ρ δ : ℝ} (hρ₀ : 0 ≤ ρ) (hρ₁ : ρ < 1) (hδ₀ : 0 ≤ δ) :
    ∃ (Tc Tl : ℝ → ℝ) (ac al : ℝ),
      (∀ x y, dist (Tl x) (Tl y) ≤ ρ * dist x y) ∧ Tc ac = ac ∧
      (∀ x, dist (Tl x) (Tc x) ≤ δ) ∧ Tl al = al ∧ dist al ac = δ / (1 - ρ) := by
  sorry

end Atlas
