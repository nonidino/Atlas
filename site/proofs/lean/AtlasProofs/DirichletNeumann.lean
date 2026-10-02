import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.Module.LinearMap.Defs
import Mathlib.Tactic.Abel

/-!
# T3b — Every fixed point of Dirichlet–Neumann is the undivided solution

Two pieces meet along an interface and do not overlap. One sweep:

1. piece 1 is solved with the interface **values** `lam` given (a Dirichlet solve);
2. the flux piece 1 then sends through the interface is handed to piece 2 (a Neumann solve),
   which returns its own interface values;
3. `lam` moves toward those values by a relaxation factor `θ`.

**Plain statement.** If a sweep returns the interface values it was given (a fixed point),
then the two pieces' solutions, put side by side, solve the undivided problem. Conversely the
undivided solution is a fixed point. The pieces' solves must be exact; nothing else is assumed,
not even that the iteration converges.

Two forms are checked: the textbook block form, with unknowns on the interface, and the
cell-centred finite-volume form the workbench's `styles.dirichlet_neumann` implements, where
the interface carries face values between cells.

Pure linear algebra over any field.
-/

namespace Atlas

variable {𝕜 : Type*} [Field 𝕜]

/-! ## The block form -/

/-- A linear problem cut into two pieces that share an interface, in block form. The unknowns
are piece 1's interior `u₁`, piece 2's interior `u₂` and the interface values `lam`. Each piece
contributes its own share of the interface block and of the interface right-hand side. -/
structure TwoPieces (𝕜 V₁ V₂ VΓ : Type*) [Field 𝕜]
    [AddCommGroup V₁] [Module 𝕜 V₁] [AddCommGroup V₂] [Module 𝕜 V₂]
    [AddCommGroup VΓ] [Module 𝕜 VΓ] where
  A₁₁ : V₁ →ₗ[𝕜] V₁
  A₁Γ : VΓ →ₗ[𝕜] V₁
  AΓ₁ : V₁ →ₗ[𝕜] VΓ
  A₂₂ : V₂ →ₗ[𝕜] V₂
  A₂Γ : VΓ →ₗ[𝕜] V₂
  AΓ₂ : V₂ →ₗ[𝕜] VΓ
  /-- piece 1's share of the interface block -/
  AΓΓ₁ : VΓ →ₗ[𝕜] VΓ
  /-- piece 2's share of the interface block -/
  AΓΓ₂ : VΓ →ₗ[𝕜] VΓ
  f₁ : V₁
  f₂ : V₂
  fΓ₁ : VΓ
  fΓ₂ : VΓ

namespace TwoPieces

variable {V₁ V₂ VΓ : Type*} [AddCommGroup V₁] [Module 𝕜 V₁] [AddCommGroup V₂] [Module 𝕜 V₂]
  [AddCommGroup VΓ] [Module 𝕜 VΓ] (P : TwoPieces 𝕜 V₁ V₂ VΓ)

/-- `(u₁, u₂, lam)` solves the **undivided** system: both interiors' equations and the
interface's, with the two pieces' shares added. -/
def Solves (u₁ : V₁) (u₂ : V₂) (lam : VΓ) : Prop :=
  P.A₁₁ u₁ + P.A₁Γ lam = P.f₁ ∧
  P.A₂₂ u₂ + P.A₂Γ lam = P.f₂ ∧
  P.AΓ₁ u₁ + P.AΓ₂ u₂ + (P.AΓΓ₁ lam + P.AΓΓ₂ lam) = P.fΓ₁ + P.fΓ₂

/-- What piece 1 sends through the interface: its own interface equations' residual, the
discrete flux. -/
def flux (u₁ : V₁) (lam : VΓ) : VΓ :=
  P.AΓ₁ u₁ + P.AΓΓ₁ lam - P.fΓ₁

/-- Piece 1's solve is exact: given interface values, it returns an interior that satisfies
piece 1's equations. -/
def ExactD (solveD : VΓ → V₁) : Prop :=
  ∀ lam, P.A₁₁ (solveD lam) + P.A₁Γ lam = P.f₁

/-- Piece 2's solve is exact: given the incoming flux `q`, it returns an interior and
interface values that satisfy piece 2's equations with that flux. -/
def ExactN (solveN : VΓ → V₂ × VΓ) : Prop :=
  ∀ q, P.A₂₂ (solveN q).1 + P.A₂Γ (solveN q).2 = P.f₂ ∧
    P.AΓ₂ (solveN q).1 + P.AΓΓ₂ (solveN q).2 = P.fΓ₂ - q

/-- One relaxed Dirichlet–Neumann sweep on the interface values. -/
def dnStep (solveD : VΓ → V₁) (solveN : VΓ → V₂ × VΓ) (θ : 𝕜) (lam : VΓ) : VΓ :=
  lam + θ • ((solveN (P.flux (solveD lam) lam)).2 - lam)

/-- **T3b (Dirichlet–Neumann, block form).** With exact solves and `θ ≠ 0`, a fixed point of
the sweep gives a solution of the undivided system. -/
theorem dn_fixedPoint_solves {solveD : VΓ → V₁} {solveN : VΓ → V₂ × VΓ}
    (hD : P.ExactD solveD) (hN : P.ExactN solveN) {θ : 𝕜} (hθ : θ ≠ 0) {lam : VΓ}
    (hfix : P.dnStep solveD solveN θ lam = lam) :
    P.Solves (solveD lam) (solveN (P.flux (solveD lam) lam)).1 lam := by
  set q := P.flux (solveD lam) lam with hq
  -- `θ ≠ 0`: the Neumann solve returned the interface values it was given.
  have hmu : (solveN q).2 = lam := by
    have h : θ • ((solveN q).2 - lam) = 0 := add_eq_left.mp hfix
    have h' : (solveN q).2 - lam = 0 := by
      rw [← one_smul 𝕜 ((solveN q).2 - lam), ← inv_mul_cancel₀ hθ, mul_smul, h, smul_zero]
    exact sub_eq_zero.mp h'
  obtain ⟨hN₁, hN₂⟩ := hN q
  rw [hmu] at hN₁ hN₂
  refine ⟨hD lam, hN₁, ?_⟩
  -- The Neumann solve's interface equation, with the flux written out, is the undivided one.
  have hq' : q = P.AΓ₁ (solveD lam) + P.AΓΓ₁ lam - P.fΓ₁ := rfl
  calc P.AΓ₁ (solveD lam) + P.AΓ₂ (solveN q).1 + (P.AΓΓ₁ lam + P.AΓΓ₂ lam)
      = (P.AΓ₂ (solveN q).1 + P.AΓΓ₂ lam) + (P.AΓ₁ (solveD lam) + P.AΓΓ₁ lam) := by abel
    _ = (P.fΓ₂ - q) + (P.AΓ₁ (solveD lam) + P.AΓΓ₁ lam) := by rw [hN₂]
    _ = P.fΓ₁ + P.fΓ₂ := by rw [hq']; abel

/-- **T3b, converse.** If each piece's equations determine its solution and the solves return
it, the undivided solution's interface values are a fixed point of the sweep, for every `θ`. -/
theorem dn_solution_isFixedPt {solveD : VΓ → V₁} {solveN : VΓ → V₂ × VΓ}
    (hD : ∀ lam u₁, P.A₁₁ u₁ + P.A₁Γ lam = P.f₁ → solveD lam = u₁)
    (hN : ∀ q u₂ mu, P.A₂₂ u₂ + P.A₂Γ mu = P.f₂ → P.AΓ₂ u₂ + P.AΓΓ₂ mu = P.fΓ₂ - q →
      solveN q = (u₂, mu))
    (θ : 𝕜) {u₁ : V₁} {u₂ : V₂} {lam : VΓ} (h : P.Solves u₁ u₂ lam) :
    P.dnStep solveD solveN θ lam = lam := by
  obtain ⟨h₁, h₂, h₃⟩ := h
  -- The Dirichlet solve returns `u₁`.
  have hd : solveD lam = u₁ := hD lam u₁ h₁
  -- `(u₂, lam)` satisfies piece 2's equations with the flux `q(u₁, lam)`, so the Neumann
  -- solve returns it.
  have hn : solveN (P.flux u₁ lam) = (u₂, lam) := by
    refine hN _ u₂ lam h₂ ?_
    rw [eq_sub_iff_add_eq]
    calc P.AΓ₂ u₂ + P.AΓΓ₂ lam + P.flux u₁ lam
        = P.AΓ₁ u₁ + P.AΓ₂ u₂ + (P.AΓΓ₁ lam + P.AΓΓ₂ lam) - P.fΓ₁ := by unfold flux; abel
      _ = P.fΓ₂ := by rw [h₃]; abel
  -- So the update is zero.
  simp [dnStep, hd, hn]

end TwoPieces

/-! ## The face form the workbench implements -/

/-- Two sets of finite-volume cells, `D` and `N`, that meet along faces. `KD` and `KN` are
each set's operator with the cut faces left out. `PD` and `PN` read, for each cut face, the
value of the cell beside it; `QD` and `QN` put a per-face flow into those cells' equations.
`gD` is the `D` side's half-cell conductance, `rD` and `rN` the two sides' half-cell
resistances, and `h` the undivided grid's face conductance: the two half-cells in series,
which is the harmonic mean. -/
structure FacePair (𝕜 VD VN F : Type*) [Field 𝕜]
    [AddCommGroup VD] [Module 𝕜 VD] [AddCommGroup VN] [Module 𝕜 VN]
    [AddCommGroup F] [Module 𝕜 F] where
  KD : VD →ₗ[𝕜] VD
  KN : VN →ₗ[𝕜] VN
  PD : VD →ₗ[𝕜] F
  PN : VN →ₗ[𝕜] F
  QD : F →ₗ[𝕜] VD
  QN : F →ₗ[𝕜] VN
  gD : F →ₗ[𝕜] F
  rD : F →ₗ[𝕜] F
  rN : F →ₗ[𝕜] F
  h : F →ₗ[𝕜] F
  bD : VD
  bN : VN
  /-- `rD` is the inverse of `gD` -/
  rD_gD : ∀ x, rD (gD x) = x
  /-- `h` is the series combination of the two half-cells -/
  series : ∀ x, h (rD x + rN x) = x

namespace FacePair

variable {VD VN F : Type*} [AddCommGroup VD] [Module 𝕜 VD] [AddCommGroup VN] [Module 𝕜 VN]
  [AddCommGroup F] [Module 𝕜 F] (P : FacePair 𝕜 VD VN F)

/-- **T3b, face form (`styles.dirichlet_neumann`).** The `D` set is solved with the face values
`lam`; the flow it sends is `q = gD (PD uD - lam)`; the `N` set is solved with that flow; its
own face values are `PN uN + rN q`. If those equal `lam`, the flow through every cut face is
the undivided grid's `h (PD uD - PN uN)`, and the pair `(uD, uN)` solves the undivided
system. -/
theorem solves_of_fixedPoint {uD : VD} {uN : VN} {lam q : F}
    (hD : P.KD uD + P.QD (P.gD (P.PD uD)) = P.bD + P.QD (P.gD lam))
    (hq : q = P.gD (P.PD uD - lam))
    (hN : P.KN uN = P.bN + P.QN q)
    (hfix : P.PN uN + P.rN q = lam) :
    q = P.h (P.PD uD - P.PN uN) ∧
      P.KD uD + P.QD (P.h (P.PD uD - P.PN uN)) = P.bD ∧
      P.KN uN - P.QN (P.h (P.PD uD - P.PN uN)) = P.bN := by
  -- `rD q = PD uD - lam` and `rN q = lam - PN uN`; added, `(rD + rN) q = PD uD - PN uN`.
  have hsum : P.rD q + P.rN q = P.PD uD - P.PN uN := by
    rw [hq, P.rD_gD, ← hq, ← hfix]
    abel
  -- So `q = h (PD uD - PN uN)`, the undivided grid's flow through the cut faces.
  have hqh : q = P.h (P.PD uD - P.PN uN) := by rw [← hsum, P.series]
  refine ⟨hqh, ?_, ?_⟩
  · -- the `D` set's equations, `KD uD + QD q = bD`
    rw [← hqh, hq, map_sub, map_sub, ← add_sub_assoc, hD, add_sub_cancel_right]
  · -- the `N` set's equations, `KN uN - QN q = bN`
    rw [← hqh, hN, add_sub_cancel_right]

end FacePair

end Atlas
