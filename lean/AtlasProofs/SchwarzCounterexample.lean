import AtlasProofs.Schwarz
import Mathlib.Data.Rat.Init
import Mathlib.LinearAlgebra.Matrix.ToLin

/-!
# T3c — A Schwarz fixed point that is not the solution

The plan's statement "every fixed point of restricted additive Schwarz built from exact local
solves, with a partition of unity, solves `A u = b`" is false without a further hypothesis.

Three unknowns, two windows `{1, 2}` and `{2, 3}`, the shared unknown `2` given entirely to
the first window:

        ⎡1 1 0⎤
    A = ⎢1 0 1⎥ ,   b = 0,   u = (1, -1, -1).
        ⎣0 1 1⎦

`A` is invertible, so the only solution of `A u = 0` is `u = 0`. Both windows' own matrices,
`[[1, 1], [1, 0]]` and `[[0, 1], [1, 1]]`, are invertible, so both window solves are exact.
Yet the sweep returns `u` unchanged: the second window's solution disagrees with `u` on the
shared unknown, and the blend discards exactly that value.

What fails is that `A` has a zero on its diagonal at the shared unknown: the problem
restricted to the overlap is singular. A symmetric positive definite `A` cannot do this.
-/

namespace Atlas

/-- The counterexample's sweep. Unknowns `0, 1, 2`; windows `{0, 1}` and `{1, 2}`; the shared
unknown `1` owned wholly by the first window (`Eχ`). Each window's solve is the inverse of its
own matrix: `[[1, 1], [1, 0]]⁻¹ = [[0, 1], [1, -1]]` and `[[0, 1], [1, 1]]⁻¹ = [[-1, 1], [1, 0]]`. -/
private def threeUnknowns : Schwarz ℚ (Fin 3 → ℚ) (fun _ : Fin 2 => Fin 2 → ℚ) where
  A := Matrix.toLin' !![(1 : ℚ), 1, 0; 1, 0, 1; 0, 1, 1]
  R := ![Matrix.toLin' !![(1 : ℚ), 0, 0; 0, 1, 0], Matrix.toLin' !![(0 : ℚ), 1, 0; 0, 0, 1]]
  E := ![Matrix.toLin' !![(1 : ℚ), 0; 0, 1; 0, 0], Matrix.toLin' !![(0 : ℚ), 0; 1, 0; 0, 1]]
  Eχ := ![Matrix.toLin' !![(1 : ℚ), 0; 0, 1; 0, 0], Matrix.toLin' !![(0 : ℚ), 0; 0, 0; 0, 1]]
  solve := ![Matrix.toLin' !![(0 : ℚ), 1; 1, -1], Matrix.toLin' !![(-1 : ℚ), 1; 1, 0]]
  solve_local := by
    intro i w
    fin_cases i <;> ext j <;> fin_cases j <;>
      simp [Matrix.toLin'_apply, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]
  local_solve := by
    intro i w
    fin_cases i <;> ext j <;> fin_cases j <;>
      simp [Matrix.toLin'_apply, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]
  pou := by
    intro v
    ext j
    fin_cases j <;> simp [Matrix.toLin'_apply, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

/-- **T3c (the counterexample).** There is a restricted additive Schwarz sweep, with exact
window solves, a partition of unity and an invertible `A`, that has a fixed point which does
not solve `A u = b`. -/
theorem Schwarz.exists_spurious_fixedPt :
    ∃ (S : Schwarz ℚ (Fin 3 → ℚ) (fun _ : Fin 2 => Fin 2 → ℚ)) (b u : Fin 3 → ℚ),
      Function.Bijective S.A ∧ S.sweep b u = u ∧ S.A u ≠ b := by
  refine ⟨threeUnknowns, 0, ![1, -1, -1], ?_, ?_, ?_⟩
  · -- `A` is invertible: `A⁻¹ = ½ [[1, 1, -1], [1, -1, 1], [-1, 1, 1]]`.
    refine Function.bijective_iff_has_inverse.mpr
      ⟨Matrix.toLin' !![(1 / 2 : ℚ), 1 / 2, -1 / 2; 1 / 2, -1 / 2, 1 / 2; -1 / 2, 1 / 2, 1 / 2],
        fun v => ?_, fun v => ?_⟩ <;>
    · ext j
      fin_cases j <;>
        simp [threeUnknowns, Matrix.toLin'_apply, Matrix.mulVec, dotProduct, Fin.sum_univ_succ] <;>
        ring
  · -- The sweep returns `u`: window 1 returns `(1, -1)` on `{0, 1}`, which is `u` there;
    -- window 2 returns `(1, -1)` on `{1, 2}`, and the blend keeps only its value at `2`.
    ext j
    fin_cases j <;>
      simp [threeUnknowns, Schwarz.sweep, Schwarz.window, Matrix.toLin'_apply, Matrix.mulVec,
        dotProduct, Fin.sum_univ_succ]
  · -- `A u = (0, 0, -2) ≠ 0 = b`
    intro h
    have h2 := congrFun h 2
    simp [threeUnknowns, Matrix.toLin'_apply, Matrix.mulVec, dotProduct, Fin.sum_univ_succ] at h2

end Atlas
