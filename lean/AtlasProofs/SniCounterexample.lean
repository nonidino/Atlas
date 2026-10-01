import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Topology.MetricSpace.Pseudo.Defs

/-!
# A remark beside T1 — accuracy plus classical contraction is not enough

T1 asks that the **learned** map be a contraction. Could one ask less: that the *classical*
map contracts, and that the learned map is uniformly close to it? No.

**Plain statement.** On the real line take the classical map `T u = u / 2`, a contraction
with factor `1/2` and answer `0`, and the learned map

    Tt u = (u - tanh (10 u)) / 2 ,

which is within `1/2` of `T` at every point. `Tt` has a two-cycle: there is an `x` between
`0.3` and `0.34` with `Tt x = -x` and `Tt (-x) = x`. The learned iteration started at `x`
alternates between `x` and `-x` for ever and converges to nothing.

So both hypotheses hold (the classical map contracts, the learned one is uniformly close) and
the conclusion "the learned iteration converges" fails. What is missing is T1's hypothesis: a
Lipschitz bound below one **for the learned map**. Here `Tt` has slope `-4.5` at `0`.

This is the counterexample of the architecture document's Remark 10 to the convergence step
of SNI's Theorem 1 (Huang et al., ICLR 2026), with `τ = 1/2`, `c = 1`, `k = 10`. The point
`x` is a root of `tanh (10 x) = 3 x`, found by the intermediate value theorem; numerically it
is `0.332471`.
-/

namespace Atlas

/-- **The SNI remark.** The classical map `T u = u / 2` contracts with factor `1/2` and has
the fixed point `0`; the learned map `Tt u = (u - tanh (10 u)) / 2` is within `1/2` of it
everywhere; and `Tt` has a two-cycle `x ↦ -x ↦ x` with `0.3 ≤ x ≤ 0.34`, along which its
iterates are `(-1) ^ k * x`. -/
theorem sni_counterexample :
    let T : ℝ → ℝ := fun u => u / 2
    let Tt : ℝ → ℝ := fun u => (u - Real.tanh (10 * u)) / 2
    (∀ u v, dist (T u) (T v) ≤ 1 / 2 * dist u v) ∧ T 0 = 0 ∧
      (∀ u, dist (Tt u) (T u) ≤ 1 / 2) ∧
      ∃ x : ℝ, 0.3 ≤ x ∧ x ≤ 0.34 ∧ Tt x = -x ∧ Tt (-x) = x ∧
        ∀ k : ℕ, Tt^[k] x = (-1) ^ k * x := by
  intro T Tt
  have hodd : ∀ u, Tt (-u) = -Tt u := by
    intro u
    simp only [Tt, mul_neg, Real.tanh_neg]
    ring
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro u v
    simp only [T, Real.dist_eq]
    have h : u / 2 - v / 2 = (u - v) / 2 := by ring
    rw [h, abs_div, abs_two]
    linarith
  · simp only [T, zero_div]
  · intro u
    simp only [T, Tt, Real.dist_eq]
    have h : (u - Real.tanh (10 * u)) / 2 - u / 2 = -Real.tanh (10 * u) / 2 := by ring
    rw [h, abs_div, abs_neg, abs_two]
    have h1 := (Real.abs_tanh_lt_one (10 * u)).le
    linarith
  · have htanh : Continuous Real.tanh := by
      have h : Real.tanh = fun x => Real.sinh x / Real.cosh x := by
        funext x
        exact Real.tanh_eq_sinh_div_cosh x
      rw [h]
      exact Real.continuous_sinh.div Real.continuous_cosh fun x => (Real.cosh_pos x).ne'
    have hcont : ContinuousOn (fun x : ℝ => Real.tanh (10 * x) - 3 * x)
        (Set.Icc 0.3 0.34) :=
      ((htanh.comp (continuous_const.mul continuous_id)).sub
        (continuous_const.mul continuous_id)).continuousOn
    have h3 : (19 : ℝ) < Real.exp 3 := by
      have h1 := Real.exp_one_gt_d9
      have h2 : Real.exp 3 = Real.exp 1 ^ 3 := by
        rw [← Real.exp_nat_mul]
        norm_num
      have h27 : (2.7 : ℝ) ≤ Real.exp 1 := by linarith
      have h4 : (2.7 : ℝ) ^ 3 ≤ Real.exp 1 ^ 3 := by gcongr
      rw [h2]
      have h5 : (19 : ℝ) < 2.7 ^ 3 := by norm_num
      linarith
    have hlo : 0 < Real.tanh (10 * 0.3) - 3 * 0.3 := by
      have h10 : (10 : ℝ) * 0.3 = 3 := by norm_num
      rw [h10, Real.tanh_eq]
      have hinv : Real.exp (-3) < 1 / 19 := by
        rw [Real.exp_neg, inv_eq_one_div]
        exact one_div_lt_one_div_of_lt (by norm_num) h3
      have hinvpos := Real.exp_pos (-3)
      have hden : 0 < Real.exp 3 + Real.exp (-3) := by linarith
      rw [sub_pos, lt_div_iff₀ hden]
      linarith
    have hhi : Real.tanh (10 * 0.34) - 3 * 0.34 < 0 := by
      have h := Real.tanh_lt_one (10 * 0.34)
      linarith
    obtain ⟨x, hx, hfx⟩ :=
      intermediate_value_Icc' (by norm_num : (0.3 : ℝ) ≤ 0.34) hcont ⟨hhi.le, hlo.le⟩
    have htx : Real.tanh (10 * x) = 3 * x := by
      have h : Real.tanh (10 * x) - 3 * x = 0 := hfx
      linarith
    have hTx : Tt x = -x := by
      simp only [Tt, htx]
      ring
    refine ⟨x, hx.1, hx.2, hTx, ?_, ?_⟩
    · rw [hodd, hTx, neg_neg]
    · intro k
      induction k with
      | zero => simp
      | succ n ih =>
        rw [Function.iterate_succ_apply', ih]
        rcases neg_one_pow_eq_or ℝ n with h | h
        · rw [h, one_mul, hTx, pow_succ, h]
          ring
        · rw [h, neg_one_mul, hodd, hTx, pow_succ, h]
          ring

end Atlas
