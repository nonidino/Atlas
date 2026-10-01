import AtlasProofs.Schwarz
import Mathlib.Analysis.Normed.Module.Basic

/-!
# T3c, T3d — A convergent Schwarz sweep converges to the undivided solution

**Plain statement.** Suppose the sweep with zero right-hand side drives every start to zero
(this is what "the Schwarz iteration converges" means; for M-matrices it is Frommer and
Szyld's theorem, *SIAM J. Numer. Anal.* 39, 2001). Then, for any right-hand side that has a
solution:

* the sweep's only fixed point is that solution;
* the sweep's iterates converge to it from every start.

And if the zero-right-hand-side sweep shrinks a norm by `ρ < 1` each time, stopping when the
update is small leaves an error of at most the update divided by `1 - ρ` (T3d). The tolerance
bounds the **update**; the error is larger by `1 / (1 - ρ)`, which is large for a slowly
converging sweep.

Real normed vector spaces (finite-dimensional or not).
-/

namespace Atlas

open Filter Topology

namespace Schwarz

variable {V : Type*} {ι : Type*} {W : ι → Type*}
  [NormedAddCommGroup V] [NormedSpace ℝ V] [Fintype ι]
  [∀ i, AddCommGroup (W i)] [∀ i, Module ℝ (W i)] (S : Schwarz ℝ V W)

/-- **T3c (a convergent sweep has one fixed point: the solution).** -/
theorem fixedPt_eq_solution
    (hconv : ∀ e, Tendsto (fun k => (S.sweep 0)^[k] e) atTop (𝓝 0))
    {b u ustar : V} (hsol : S.A ustar = b) (hfix : S.sweep b u = u) : u = ustar := by
  sorry

/-- **T3c (a convergent sweep converges to the solution).** -/
theorem tendsto_solution
    (hconv : ∀ e, Tendsto (fun k => (S.sweep 0)^[k] e) atTop (𝓝 0))
    {b ustar : V} (hsol : S.A ustar = b) (u₀ : V) :
    Tendsto (fun k => (S.sweep b)^[k] u₀) atTop (𝓝 ustar) := by
  sorry

/-- **T3d (stopping on the update).** If the zero-right-hand-side sweep shrinks the norm by
`ρ < 1`, the error of any iterate is at most its update over `1 - ρ`. -/
theorem error_le_update {ρ : ℝ} (hρ : ρ < 1) (hT : ∀ e, ‖S.sweep 0 e‖ ≤ ρ * ‖e‖)
    {b ustar : V} (hsol : S.A ustar = b) (u : V) :
    ‖u - ustar‖ ≤ ‖S.sweep b u - u‖ / (1 - ρ) := by
  sorry

end Schwarz

end Atlas
