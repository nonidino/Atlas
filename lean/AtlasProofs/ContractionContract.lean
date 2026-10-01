import AtlasProofs.PerturbedContraction
import Mathlib.Analysis.Normed.Lp.PiLp

/-!
# T9a — The contraction contract, one level

At tier 2 the experts are black boxes. Expert `i` maps the incoming wave on its sides to an
outgoing wave, by its scattering map `S i`. One **sweep** of the coupled iteration applies
every expert's map, hands each outgoing wave to the neighbouring side (the **exchange**), and
adds the sources and boundary data `g`:

    a  ↦  Ex (S 1 (a 1), …, S n (a n)) + g .

Distances between global wave vectors are measured in the `ℓ²` product: the square root of
the sum of the squared distances on the pieces (wave power adds up over pieces).

**Plain statement.**

* (the sweep contracts) If every expert's scattering map shrinks distances by `ρ`, and the
  exchange does not lengthen any distance (an isometry in particular), the sweep shrinks
  distances by `ρ` in the `ℓ²` product.
* (the contract) So if `ρ < 1`, and at the classical answer each learned map differs from the
  classical one by at most `δ i`, then by T1 the learned iteration has exactly one answer,
  reaches it geometrically at rate `ρ` from any start, and that answer is within
  `√(∑ δ i²) / (1 - ρ)` of the classical one.

The scattering maps may be nonlinear. Nothing is assumed about how they were trained; `ρ`
must be *certified* for each expert. The defects add up in the `ℓ²` sense: `n` experts each
within `δ` give `√n δ / (1 - ρ)`.
-/

namespace Atlas

variable {ι : Type*} [Fintype ι] {W : ι → Type*} [∀ i, NormedAddCommGroup (W i)]

/-- **One sweep of the wave exchange.** Every expert maps its incoming wave to an outgoing
one; the exchange `Ex` hands the outgoing waves to the neighbours; `g` is added. -/
def waveSweep (S : ∀ i, W i → W i) (Ex : PiLp 2 W → PiLp 2 W) (g : PiLp 2 W)
    (a : PiLp 2 W) : PiLp 2 W :=
  Ex (WithLp.toLp 2 fun i => S i (a i)) + g

/-- **T9a (the sweep contracts).** If every expert's scattering map is `ρ`-Lipschitz and the
exchange is non-expansive, the sweep is `ρ`-Lipschitz in the `ℓ²` product. -/
theorem waveSweep_lipschitz {S : ∀ i, W i → W i} {Ex : PiLp 2 W → PiLp 2 W} {ρ : ℝ}
    (hρ : 0 ≤ ρ) (hS : ∀ i (x y : W i), dist (S i x) (S i y) ≤ ρ * dist x y)
    (hEx : ∀ x y, dist (Ex x) (Ex y) ≤ dist x y) (g a a' : PiLp 2 W) :
    dist (waveSweep S Ex g a) (waveSweep S Ex g a') ≤ ρ * dist a a' := by
  unfold waveSweep
  rw [dist_add_right]
  refine (hEx _ _).trans ?_
  rw [PiLp.dist_eq_of_L2, PiLp.dist_eq_of_L2]
  have h1 : ∑ i, dist (S i (a i)) (S i (a' i)) ^ 2 ≤ ρ ^ 2 * ∑ i, dist (a i) (a' i) ^ 2 := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun i _ => ?_
    have h := hS i (a i) (a' i)
    have h0 : (0 : ℝ) ≤ dist (S i (a i)) (S i (a' i)) := dist_nonneg
    calc dist (S i (a i)) (S i (a' i)) ^ 2 ≤ (ρ * dist (a i) (a' i)) ^ 2 :=
          sq_le_sq' (by linarith) h
      _ = ρ ^ 2 * dist (a i) (a' i) ^ 2 := mul_pow _ _ _
  calc Real.sqrt (∑ i, dist (S i (a i)) (S i (a' i)) ^ 2)
      ≤ Real.sqrt (ρ ^ 2 * ∑ i, dist (a i) (a' i) ^ 2) := Real.sqrt_le_sqrt h1
    _ = ρ * Real.sqrt (∑ i, dist (a i) (a' i) ^ 2) := by
        rw [Real.sqrt_mul (sq_nonneg ρ), Real.sqrt_sq hρ]

/-- **T9a (the contraction contract).** `S i` are the learned scattering maps and `Sc i` the
classical ones, with the same exchange and data. `ac` is the classical answer. If every
learned map is `ρ`-Lipschitz with `ρ < 1`, and at `ac` it differs from the classical map by
at most `δ i`, then the learned sweep has a unique fixed point `al`, every orbit converges to
it at rate `ρ`, and `dist al ac ≤ √(∑ i, δ i ^ 2) / (1 - ρ)`. -/
theorem contraction_contract [∀ i, CompleteSpace (W i)] {S Sc : ∀ i, W i → W i}
    {Ex : PiLp 2 W → PiLp 2 W} {g : PiLp 2 W} {ρ : ℝ} {δ : ι → ℝ}
    (hρ₀ : 0 ≤ ρ) (hρ₁ : ρ < 1)
    (hS : ∀ i (x y : W i), dist (S i x) (S i y) ≤ ρ * dist x y)
    (hEx : ∀ x y, dist (Ex x) (Ex y) ≤ dist x y)
    {ac : PiLp 2 W} (hac : waveSweep Sc Ex g ac = ac)
    (hδ : ∀ i, dist (S i (ac i)) (Sc i (ac i)) ≤ δ i) :
    ∃ al : PiLp 2 W, waveSweep S Ex g al = al ∧
      (∀ b, waveSweep S Ex g b = b → b = al) ∧
      (∀ (a₀ : PiLp 2 W) (k : ℕ),
        dist ((waveSweep S Ex g)^[k] a₀) al ≤ ρ ^ k * dist a₀ al) ∧
      dist al ac ≤ Real.sqrt (∑ i, δ i ^ 2) / (1 - ρ) := by
  have hlip : ∀ a a', dist (waveSweep S Ex g a) (waveSweep S Ex g a') ≤ ρ * dist a a' :=
    fun a a' => waveSweep_lipschitz hρ₀ hS hEx g a a'
  have hdef : dist (waveSweep S Ex g ac) (waveSweep Sc Ex g ac)
      ≤ Real.sqrt (∑ i, δ i ^ 2) := by
    unfold waveSweep
    rw [dist_add_right]
    refine (hEx _ _).trans ?_
    rw [PiLp.dist_eq_of_L2]
    refine Real.sqrt_le_sqrt (Finset.sum_le_sum fun i _ => ?_)
    have h0 : (0 : ℝ) ≤ dist (S i (ac i)) (Sc i (ac i)) := dist_nonneg
    exact sq_le_sq' (by linarith [hδ i]) (hδ i)
  exact perturbed_contraction hρ₀ hρ₁ hlip hac hdef

end Atlas
