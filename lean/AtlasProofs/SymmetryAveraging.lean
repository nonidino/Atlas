import Mathlib.Algebra.Module.Defs
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Group.Action.Defs
import Mathlib.Data.Fintype.Card
import Mathlib.Basic.Real.Basic
import Mathlib.Algebra.BigOperators.GroupWithZero.Action
import Mathlib.Algebra.Module.NatInt
import Mathlib.Tactic.NormNum.Eq

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
  -- Substitute `g' = g h`, which runs over `G` as `g` does; then `g⁻¹ = h g'⁻¹`, and `h` comes
  -- out of the sum because it acts linearly on the outputs.
  have hsum : ∑ g : G, g⁻¹ • E (g • h • u) = h • ∑ g : G, g⁻¹ • E (g • u) := by
    calc ∑ g : G, g⁻¹ • E (g • h • u)
        = ∑ g : G, (g * h⁻¹)⁻¹ • E ((g * h⁻¹) • h • u) :=
          (Equiv.sum_comp (Equiv.mulRight h⁻¹) fun g => g⁻¹ • E (g • h • u)).symm
      _ = ∑ g : G, h • (g⁻¹ • E (g • u)) := by
          refine Finset.sum_congr rfl fun g _ => ?_
          rw [smul_smul (g * h⁻¹) h u, inv_mul_cancel_right, mul_inv_rev, inv_inv, mul_smul]
      _ = h • ∑ g : G, g⁻¹ • E (g • u) := Finset.smul_sum.symm
  unfold average
  rw [hsum]
  exact (smul_comm h _ _).symm

/-- **T10 (averaging changes nothing that was already symmetric).** If `E` is equivariant
and `|G|` is invertible in the scalars, the average of `E` is `E`. -/
theorem average_of_equivariant (E : V → W) (hE : ∀ (g : G) (u : V), E (g • u) = g • E u)
    (hG : (Fintype.card G : 𝕜) ≠ 0) (u : V) :
    average G 𝕜 E u = E u := by
  -- Every term of the sum is `E u`, so the sum is `|G| • E u`.
  have hterm : ∀ g : G, g⁻¹ • E (g • u) = E u := fun g => by rw [hE, inv_smul_smul]
  unfold average
  simp only [hterm, Finset.sum_const, Finset.card_univ]
  rw [← Nat.cast_smul_eq_nsmul 𝕜, smul_smul, inv_mul_cancel₀ hG, one_smul]

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
  -- `E (x, y) = (x + y, 0)` and `u = (0, 1)`: the average at `Mx u = u` is `(-1/3, 0)`, while
  -- `Mx` of the average at `u` is `(1/3, 0)`.
  refine ⟨fun p => (p.1 + p.2, 0), (0, 1), ?_⟩
  intro Mx My avg
  simp only [avg, Mx, My, Prod.smul_mk, Prod.mk_add_mk, ne_eq, Prod.mk.injEq]
  norm_num

end Atlas
