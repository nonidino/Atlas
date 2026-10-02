import AtlasProofs.Schwarz
import AtlasProofs.Certificate
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

/-- The sweep moves an error as the zero-right-hand-side sweep does: if `A ustar = b`, then
`G_b(u) - ustar = G_0(u - ustar)`. -/
private theorem sweep_sub_solution {b ustar : V} (hsol : S.A ustar = b) (u : V) :
    S.sweep b u - ustar = S.sweep 0 (u - ustar) := by
  rw [sweep_eq, sweep_eq, ← hsol, map_sub, zero_sub, neg_sub]
  abel

/-- **T3c (a convergent sweep has one fixed point: the solution).** -/
theorem fixedPt_eq_solution
    (hconv : ∀ e, Tendsto (fun k => (S.sweep 0)^[k] e) atTop (𝓝 0))
    {b u ustar : V} (hsol : S.A ustar = b) (hfix : S.sweep b u = u) : u = ustar := by
  -- `u - ustar` is a fixed point of the zero-right-hand-side sweep, whose orbits tend to `0`.
  have hfix0 : S.sweep 0 (u - ustar) = u - ustar := by
    rw [← sweep_sub_solution S hsol, hfix]
  have h := hconv (u - ustar)
  simp only [Function.iterate_fixed hfix0] at h
  exact sub_eq_zero.mp (tendsto_nhds_unique tendsto_const_nhds h)

/-- **T3c (a convergent sweep converges to the solution).** -/
theorem tendsto_solution
    (hconv : ∀ e, Tendsto (fun k => (S.sweep 0)^[k] e) atTop (𝓝 0))
    {b ustar : V} (hsol : S.A ustar = b) (u₀ : V) :
    Tendsto (fun k => (S.sweep b)^[k] u₀) atTop (𝓝 ustar) := by
  -- The iterates' errors are the zero-right-hand-side sweep's iterates of the first error.
  have hiter : ∀ k, (S.sweep b)^[k] u₀ - ustar = (S.sweep 0)^[k] (u₀ - ustar) := by
    intro k
    induction k with
    | zero => rfl
    | succ k ih =>
      rw [Function.iterate_succ_apply', Function.iterate_succ_apply', sweep_sub_solution S hsol,
        ih]
  have h := (hconv (u₀ - ustar)).const_add ustar
  rw [add_zero] at h
  exact h.congr fun k => by rw [← hiter k, add_sub_cancel]

/-- **T3d (stopping on the update).** If the zero-right-hand-side sweep shrinks the norm by
`ρ < 1`, the error of any iterate is at most its update over `1 - ρ`. -/
theorem error_le_update {ρ : ℝ} (hρ : ρ < 1) (hT : ∀ e, ‖S.sweep 0 e‖ ≤ ρ * ‖e‖)
    {b ustar : V} (hsol : S.A ustar = b) (u : V) :
    ‖u - ustar‖ ≤ ‖S.sweep b u - u‖ / (1 - ρ) := by
  -- `ustar` is the sweep's fixed point and the sweep contracts the pair `(u, ustar)` by `ρ`:
  -- the certificate T7 applies.
  have hstar : S.sweep b ustar = ustar := S.solution_isFixedPt hsol
  have hcon : dist (S.sweep b u) (S.sweep b ustar) ≤ ρ * dist u ustar := by
    rw [dist_eq_norm, dist_eq_norm, hstar, sweep_sub_solution S hsol]
    exact hT _
  have h := certificate hρ hstar hcon
  rwa [dist_eq_norm, dist_eq_norm, norm_sub_rev u (S.sweep b u)] at h

end Schwarz

end Atlas
