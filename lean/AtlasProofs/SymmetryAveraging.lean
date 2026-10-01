import Mathlib.Algebra.Module.Defs
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Group.Action.Defs
import Mathlib.Data.Fintype.Card
import Mathlib.Basic.Real.Basic

/-!
# T10 — Symmetry averaging: an exact invariance for any map

A frozen expert `E` does not respect a symmetry. Call it once for every element of the
symmetry group, undo each transformation, and average:

    Ẽ u = (1 / |G|) ∑ g, g⁻¹ • E (g • u).

**Plain statement.**

* For a **finite group** `G`, the averaged map is exactly equivariant, `Ẽ (h • u) = h • Ẽ u`,
  **whatever `E` is**: nonlinear, learned, discontinuous. The proof uses only that multiplying
  every group element by `h` runs through the whole group again.
* If `E` was already equivariant, averaging returns `E` unchanged.
* Averaging over a set that is **not** a group need not be equivariant: for the two mirror
  reflections of the plane and the identity (their product, the point reflection, is missing)
  there is a map whose average is not equivariant.

The group acts on the inputs in any way, and on the outputs additively and commuting with
scalars (a linear action). No dimension is assumed.
-/

namespace Atlas

open Finset

section Average

variable (G 𝕜 : Type*) {V W : Type*} [Group G] [Fintype G] [Field 𝕜]
  [MulAction G V] [AddCommGroup W] [Module 𝕜 W] [DistribMulAction G W] [SMulCommClass G 𝕜 W]

/-- The group average (Reynolds average) of an arbitrary map `E`. -/
noncomputable def average (E : V → W) (u : V) : W :=
  (Fintype.card G : 𝕜)⁻¹ • ∑ g : G, g⁻¹ • E (g • u)

/-- **T10 (symmetry averaging).** The group average of any map is exactly equivariant. -/
theorem average_equivariant (E : V → W) (h : G) (u : V) :
    average G 𝕜 E (h • u) = h • average G 𝕜 E u := by
  sorry

/-- **T10 (averaging changes nothing that was already symmetric).** If `E` is equivariant
and `|G|` is invertible in the scalars, the average of `E` is `E`. -/
theorem average_of_equivariant (E : V → W) (hE : ∀ (g : G) (u : V), E (g • u) = g • E u)
    (hG : (Fintype.card G : 𝕜) ≠ 0) (u : V) :
    average G 𝕜 E u = E u := by
  sorry

end Average

/-- **T10 (a set that is not a group).** `Mx` and `My` are the two mirror reflections of the
plane. The set `{id, Mx, My}` is not a group. Averaging over it, in the same way, gives a map
that is not `Mx`-equivariant, for a suitable `E` and input `u`. -/
theorem average_over_non_group_not_equivariant :
    ∃ (E : ℝ × ℝ → ℝ × ℝ) (u : ℝ × ℝ),
      let Mx : ℝ × ℝ → ℝ × ℝ := fun p => (-p.1, p.2)
      let My : ℝ × ℝ → ℝ × ℝ := fun p => (p.1, -p.2)
      let avg : (ℝ × ℝ → ℝ × ℝ) → ℝ × ℝ → ℝ × ℝ :=
        fun E u => (3 : ℝ)⁻¹ • (E u + Mx (E (Mx u)) + My (E (My u)))
      avg E (Mx u) ≠ Mx (avg E u) := by
  sorry

end Atlas
