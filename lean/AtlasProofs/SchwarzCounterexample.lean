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

/-- **T3c (the counterexample).** There is a restricted additive Schwarz sweep, with exact
window solves, a partition of unity and an invertible `A`, that has a fixed point which does
not solve `A u = b`. -/
theorem Schwarz.exists_spurious_fixedPt :
    ∃ (S : Schwarz ℚ (Fin 3 → ℚ) (fun _ : Fin 2 => Fin 2 → ℚ)) (b u : Fin 3 → ℚ),
      Function.Bijective S.A ∧ S.sweep b u = u ∧ S.A u ≠ b := by
  sorry

end Atlas
