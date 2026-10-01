import AtlasProofs.MasterBound
import AtlasProofs.PerturbedContraction
import AtlasProofs.Superelement

/-!
# T22 — The master bound with learned experts, once per tier

The master bound (T4) says: the error after `N` steps is at most `L ^ N` times the initial
error plus every one-step defect, each magnified by the steps after it. T4 splits the defect
of a coupled step into three named parts: `τ` (the experts being wrong even with the right
interface data), `σ` (the interface problem being posed with the wrong operator) and `γ` (the
interface solve being stopped early). This file says what learned experts put into `σ` and
`γ`, for each coupling tier.

**Plain statement, tier 2 (iterated wave exchange).** At a step, the interface datum is the
fixed point of a sweep. The classical sweep `Tc` has the answer `lamStar`. The learned sweep
`Tl` contracts by a certified `ρ < 1` and differs from the classical one by at most `δ` at
`lamStar`. The host runs `k` learned sweeps from a starting guess `lam0`. If the step is
`Cμ`-Lipschitz in its interface datum, then

* `σ ≤ Cμ δ / (1 - ρ)`: the learned interface problem has the wrong answer, by T1's distance;
* `γ ≤ Cμ ρ ^ k dist(lam0, learned answer)`: stopping after `k` sweeps leaves `ρ ^ k` of the
  initial interface error.

Put into T4, over `N` steps the error is at most the master bound with these two terms added
to every step's defect. Every term is measurable: `δ` and `ρ` come from the expert's
certificate, `k` from the host, and the initial interface error is at most a bound `D`.

**Plain statement, tier 0 (the superelement).** There is no iteration, so there is no `γ`.
The defect of a step is the classical step's own defect `τ` plus the **Galerkin error** of
the tier-0 solve, which T25 bounds by the truncation error plus the energy of the network's
field errors. Errors are measured in the energy size.

The two tiers are stated separately because the terms differ. Tier 2 holds in any normed
state space with any complete metric space of interface data. Tier 0 is stated for any
quadratic energy, and then for the tier-0 solve of `Superelement.lean`.
-/

namespace Atlas

open Finset Matrix

section Tier2

variable {Λ : Type*} [MetricSpace Λ] [CompleteSpace Λ] {E : Type*} [SeminormedAddCommGroup E]

/-- **T22, tier 2, one step: the transmission term and the incomplete-solve term.**
`Φ lam v` is the step forced to use the interface datum `lam`, `Cμ`-Lipschitz in `lam` at the
state `v`. `Tc` is the classical interface sweep with answer `lamStar`; `Tl` is the learned
one, a `ρ`-contraction within `δ` of `Tc` at `lamStar`. Then the learned sweep has a fixed
point `lamDag`, and

* `‖Φ lamDag v - Φ lamStar v‖ ≤ Cμ * (δ / (1 - ρ))` (the term `σ`);
* after `k` sweeps from `lam0`,
  `‖Φ (Tl^[k] lam0) v - Φ lamDag v‖ ≤ Cμ * (ρ ^ k * dist lam0 lamDag)` (the term `γ`). -/
theorem tier2_step_defect {Φ : Λ → E → E} {v : E} {Cμ ρ δ : ℝ} (hCμ : 0 ≤ Cμ)
    (hΦ : ∀ lam lam', ‖Φ lam v - Φ lam' v‖ ≤ Cμ * dist lam lam')
    {Tc Tl : Λ → Λ} (hρ₀ : 0 ≤ ρ) (hρ₁ : ρ < 1)
    (hTl : ∀ x y, dist (Tl x) (Tl y) ≤ ρ * dist x y)
    {lamStar : Λ} (hstar : Tc lamStar = lamStar)
    (hδ : dist (Tl lamStar) (Tc lamStar) ≤ δ) (lam0 : Λ) (k : ℕ) :
    ∃ lamDag : Λ, Tl lamDag = lamDag ∧
      ‖Φ lamDag v - Φ lamStar v‖ ≤ Cμ * (δ / (1 - ρ)) ∧
      ‖Φ (Tl^[k] lam0) v - Φ lamDag v‖ ≤ Cμ * (ρ ^ k * dist lam0 lamDag) := by
  obtain ⟨lamDag, hfix, -, hrate, hdist⟩ := perturbed_contraction hρ₀ hρ₁ hTl hstar hδ
  exact ⟨lamDag, hfix, (hΦ _ _).trans (mul_le_mul_of_nonneg_left hdist hCμ),
    (hΦ _ _).trans (mul_le_mul_of_nonneg_left (hrate lam0 k) hCμ)⟩

/-- **T22, tier 2: the master bound with learned experts.** The interface sweeps depend on
the state `v`: `Tc v` is the classical sweep and `Tl v` the learned one, and the step uses
the datum returned by `k` learned sweeps from `lam0 v`. Along the true trajectory `ustar`:
the learned sweep contracts by `ρ < 1`, differs from the classical one by at most `δ` at the
classical answer `lamStar n`, starts within `D` of its own answer, and the step is
`Cμ`-Lipschitz in its datum. If one step magnifies the difference between the computed and
the true trajectory by at most `L`, then after `N` steps the error is at most the master
bound with `Cμ δ / (1 - ρ)` and `Cμ ρ ^ k D` added to every step's defect. -/
theorem learned_master_bound_tier2 {Φ : Λ → E → E} {L Cμ ρ δ D : ℝ} (hL : 0 ≤ L)
    (hCμ : 0 ≤ Cμ) (hρ₀ : 0 ≤ ρ) (hρ₁ : ρ < 1)
    (Tc Tl : E → Λ → Λ) (lam0 : E → Λ) (k : ℕ) {u ustar : ℕ → E} (lamStar : ℕ → Λ)
    (hu : ∀ n, u (n + 1) = Φ ((Tl (u n))^[k] (lam0 (u n))) (u n))
    (hstep : ∀ n, ‖Φ ((Tl (u n))^[k] (lam0 (u n))) (u n)
        - Φ ((Tl (ustar n))^[k] (lam0 (ustar n))) (ustar n)‖ ≤ L * ‖u n - ustar n‖)
    (hΦ : ∀ n lam lam', ‖Φ lam (ustar n) - Φ lam' (ustar n)‖ ≤ Cμ * dist lam lam')
    (hTl : ∀ n x y, dist (Tl (ustar n) x) (Tl (ustar n) y) ≤ ρ * dist x y)
    (hstar : ∀ n, Tc (ustar n) (lamStar n) = lamStar n)
    (hδ : ∀ n, dist (Tl (ustar n) (lamStar n)) (Tc (ustar n) (lamStar n)) ≤ δ)
    (hD : ∀ n lam, Tl (ustar n) lam = lam → dist (lam0 (ustar n)) lam ≤ D)
    (N : ℕ) :
    ‖u N - ustar N‖ ≤ L ^ N * ‖u 0 - ustar 0‖
      + ∑ i ∈ range N, L ^ (N - (i + 1)) *
          (‖Φ (lamStar i) (ustar i) - ustar (i + 1)‖
            + Cμ * (δ / (1 - ρ)) + Cμ * (ρ ^ k * D)) := by
  choose lamDag hfix hσ hγ using fun n =>
    tier2_step_defect (Φ := Φ) (v := ustar n) hCμ (hΦ n) hρ₀ hρ₁ (hTl n) (hstar n) (hδ n)
      (lam0 (ustar n)) k
  have h := master_bound_three_terms (Φ := Φ) (lamK := fun v => (Tl v)^[k] (lam0 v)) hL
    lamStar lamDag hu hstep N
  refine h.trans (add_le_add le_rfl (Finset.sum_le_sum fun i _ =>
    mul_le_mul_of_nonneg_left ?_ (pow_nonneg hL _)))
  have hD' : Cμ * (ρ ^ k * dist (lam0 (ustar i)) (lamDag i)) ≤ Cμ * (ρ ^ k * D) :=
    mul_le_mul_of_nonneg_left
      (mul_le_mul_of_nonneg_left (hD i _ (hfix i)) (pow_nonneg hρ₀ k)) hCμ
  exact add_le_add (add_le_add le_rfl (hσ i)) ((hγ i).trans hD')

end Tier2

section Tier0

variable {X : Type*} [AddCommGroup X] [Module ℝ X] (Q : QuadEnergy X)

/-- **T22, tier 0, for any quadratic energy.** `ustar n` is the true solution at step `n`;
`uex n` is the undivided discrete step from `ustar n` and `ut n` the tier-0 step from
`ustar n`; `u` is the computed trajectory. If the tier-0 step magnifies a difference by at
most `L` in the energy size, the undivided step's defect is at most `τ n`, and the Galerkin
error of the tier-0 solve is at most `g n`, then the error after `N` steps is at most the
master bound with defects `τ n + g n`. There is no incomplete-solve term. -/
theorem learned_master_bound_tier0 {L : ℝ} (hL : 0 ≤ L) {u ustar uex ut : ℕ → X}
    {τ g : ℕ → ℝ}
    (hstab : ∀ n, Q.norm (u (n + 1) - ut n) ≤ L * Q.norm (u n - ustar n))
    (hgal : ∀ n, Q.norm (uex n - ut n) ≤ g n)
    (hτ : ∀ n, Q.norm (uex n - ustar (n + 1)) ≤ τ n) (N : ℕ) :
    Q.norm (u N - ustar N) ≤ L ^ N * Q.norm (u 0 - ustar 0)
      + ∑ i ∈ range N, L ^ (N - (i + 1)) * (τ i + g i) := by
  have hrec : ∀ n, Q.norm (u (n + 1) - ustar (n + 1))
      ≤ L * Q.norm (u n - ustar n) + (τ n + g n) := by
    intro n
    have hsplit : u (n + 1) - ustar (n + 1)
        = (u (n + 1) - ut n) + ((ut n - uex n) + (uex n - ustar (n + 1))) := by abel
    have h1 := Q.norm_add_le (u (n + 1) - ut n) ((ut n - uex n) + (uex n - ustar (n + 1)))
    have h2 := Q.norm_add_le (ut n - uex n) (uex n - ustar (n + 1))
    have h3 : Q.norm (ut n - uex n) = Q.norm (uex n - ut n) := by
      unfold QuadEnergy.norm
      rw [Q.m_sub_comm]
    rw [← hsplit] at h1
    have h4 := hstab n
    have h5 := hgal n
    have h6 := hτ n
    linarith
  have h := accumulation hL (e := fun n => Q.norm (u n - ustar n))
    (d := fun n => τ (n - 1) + g (n - 1)) (fun n => hrec n) N
  exact h

end Tier0

section Tier0Superelement

variable {π ι γ U F : Type*} [Fintype π] [Fintype ι] [Fintype γ]
  [AddCommGroup U] [Module ℝ U] [AddCommGroup F] [Module ℝ F]

/-- **T22, tier 0, with T25 as the estimate.** At step `n` the tier-0 problem posed at the
true state is `S n`, with undivided solution `uex n`, network output `h n, up n` and host
solution `c n`. All steps measure with the same energy size `Q`. Under the two conformity
assumptions, the Galerkin term of the master bound is T25's bound: the error of the field
rebuilt from any reference modes `hex n, upx n` at any coordinates `cstar n`, plus the energy
of the network's field errors. -/
theorem learned_master_bound_tier0_superelement (Q : QuadEnergy (π → U × F))
    (S : ℕ → Superelement π ι γ U F) (hQ : ∀ n, (S n).quad.m = Q.m)
    (hconf : ∀ n, (S n).Conforming) (hbd : ∀ n, (S n).BoundaryDataInPortSpan)
    {L : ℝ} (hL : 0 ≤ L) {u ustar uex : ℕ → π → U × F} {τ : ℕ → ℝ}
    (h : ℕ → π → ι → U) (up : ℕ → π → U) (c : ℕ → γ → ℝ)
    (huex : ∀ n, (S n).IsSolution (uex n))
    (hc : ∀ n, (S n).hostMatrix (h n) *ᵥ c n = (S n).hostRhs (h n) (up n))
    (hstab : ∀ n, Q.norm (u (n + 1) - (S n).field (h n) (up n) (c n))
      ≤ L * Q.norm (u n - ustar n))
    (hτ : ∀ n, Q.norm (uex n - ustar (n + 1)) ≤ τ n)
    (hex : ℕ → π → ι → U) (upx : ℕ → π → U) (cstar : ℕ → γ → ℝ) (N : ℕ) :
    Q.norm (u N - ustar N) ≤ L ^ N * Q.norm (u 0 - ustar 0)
      + ∑ i ∈ range N, L ^ (N - (i + 1)) *
          (τ i + (Q.norm (uex i - (S i).field (hex i) (upx i) (cstar i))
            + Real.sqrt (∑ j, ((S i).piece j).a
                ((S i).fieldError (h i) (up i) (hex i) (upx i) (cstar i) j)
                ((S i).fieldError (h i) (up i) (hex i) (upx i) (cstar i) j)))) := by
  have hnorm : ∀ n w, (S n).quad.norm w = Q.norm w := by
    intro n w
    unfold QuadEnergy.norm
    rw [hQ n]
  refine learned_master_bound_tier0 Q hL
    (ut := fun n => (S n).field (h n) (up n) (c n)) hstab (fun n => ?_) hτ N
  have hb := (S n).energy_error_bound (hconf n) (hbd n) (huex n) (h n) (up n) (hc n)
    (hex n) (upx n) (cstar n)
  rw [hnorm, hnorm] at hb
  exact hb

end Tier0Superelement

end Atlas
