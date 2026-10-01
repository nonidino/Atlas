import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.Module.LinearMap.Defs

/-!
# T3a — A direct Schur interface solve gives the undivided solution

Cut a linear problem's unknowns into the **interior** unknowns of the pieces (`uI`, all pieces
together) and the **interface** unknowns (`uΓ`). The undivided system is, in blocks,

    AII uI + AIΓ uΓ = fI          (the pieces' own equations)
    AΓI uI + AΓΓ uΓ = fΓ          (the interface's equations)

Substructuring solves it in two stages. Each piece can solve its own equations exactly once
its interface data are given: `solveI` is the inverse of `AII`. Eliminating the interiors
leaves the **Schur system** on the interface alone,

    (AΓΓ - AΓI solveI AIΓ) uΓ = fΓ - AΓI solveI fI.

**Plain statement.** A pair `(uI, uΓ)` solves the undivided system exactly when `uΓ` solves
the Schur system and `uI` is the pieces' own solve with interface data `uΓ`. Nothing is lost
and nothing is added by the cut.

Pure linear algebra over any field; no norm, no dimension count.
-/

namespace Atlas

variable {𝕜 : Type*} [Field 𝕜]
variable {VI VΓ : Type*} [AddCommGroup VI] [Module 𝕜 VI] [AddCommGroup VΓ] [Module 𝕜 VΓ]

/-- **T3a (direct Schur solve).** With an exact interior solve, the undivided block system and
the substructured system (Schur system on the interface, then the pieces' solves) have the
same solutions. -/
theorem schur_iff
    (AII : VI →ₗ[𝕜] VI) (AIΓ : VΓ →ₗ[𝕜] VI) (AΓI : VI →ₗ[𝕜] VΓ) (AΓΓ : VΓ →ₗ[𝕜] VΓ)
    (solveI : VI →ₗ[𝕜] VI)
    (hleft : ∀ x, solveI (AII x) = x) (hright : ∀ x, AII (solveI x) = x)
    (fI uI : VI) (fΓ uΓ : VΓ) :
    (AII uI + AIΓ uΓ = fI ∧ AΓI uI + AΓΓ uΓ = fΓ) ↔
      (uI = solveI (fI - AIΓ uΓ) ∧
        AΓΓ uΓ - AΓI (solveI (AIΓ uΓ)) = fΓ - AΓI (solveI fI)) := by
  sorry

end Atlas
