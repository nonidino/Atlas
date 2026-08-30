# Phase 0 — Scope and Data Generation

**Type:** Implementation spec — agent task (folder: Atlas 0.1 / Atlas 0.1 implementation)
**Phase of:** [[00-atlas-0.1-implementation-plan]]. **Depends on:** nothing. **Unblocks:** [[impl-atlas-0.1-phase2-experts]], [[impl-atlas-0.1-phase4-validation]].
**Design pages:** [[case-study-rocket-ascent-2d-atlas-0.1]], [[training-and-bootstrap-atlas-0.1]], [[agent-definition-atlas-0.1]].
**Milestones:** M1 (solvers match analytic oracles), M2 (corpus generated).
**Status: BUILT 2026-08-09** — all solvers, oracles, sweep design, HDF5 schema and the coupled generator exist on branch `atlas-0.1`. **M1 passes on 10 of 11 oracles** (Blasius and the nozzle area–Mach criterion are not met, both for reasons established by measurement — §3.7). **M2 is blocked on compute by four orders of magnitude** and needs a decision from §3.8. Full records in [[atlas-0.1-implementation-log]].

---

# 1. Intuition

Atlas 0.1 has no dataset. Nothing in The Well, PDEBench, or PDEgym contains a rocket's internal combustion, its confined nozzle flow, its heated structural shell, and its external plume/atmosphere interaction *as one coupled system with labelled interfaces*. So before any model exists, **we write the classical solvers that will be both the teacher and the judge.**

> ⚠️ **Corrected 2026-08-09 — this paragraph is true only because of its last clause, and it was over-read.** The conjunction "*as one coupled system with labelled interfaces*" carries all of its weight; strip it and the claim fails. Audited in [[physics-simulation-datasets]]: the individual governing families are well covered publicly — PDEgym's 77,840 compressible-Euler trajectories (which Atlas *already* consumes indirectly through the Poseidon checkpoint), and **BLASTNet 2.0's 2.2 TB of reacting compressible turbulent DNS**, which §2's from-scratch justification for the `reacting_flow` expert never considers. What is genuinely unobtainable is (a) **interface flux fields on declared subdomain boundaries** — §3.4's own "Critical" requirement, and structurally absent from every public corpus because no other architecture declares interfaces to label — and (b) **confined nozzle flow through a choking throat**, which no public dataset contains and on which the choked-throat M4 gate therefore depends entirely. This does not remove the need for these solvers; it re-scopes what the generated corpus is *for*, from teaching three governing families to teaching interface behaviour. See [[physics-simulation-datasets]] §5 for the resulting three-tier strategy.

Two solvers cover all seven agents, because the seven agents only span two governing-equation families plus one closed-form ODE:

1. A **2D compressible Navier–Stokes finite-volume solver**, run in two modes — *internal* (walls + a heat-release source term) for agents `a`, `b`, `e`, and *external* (freestream inflow/outflow) for `d`, `f`, `g`.
2. A **2D conduction + plane-stress elasticity solver** for agent `c`.

Plus two small non-PDE modules: a standard atmosphere table and a 3-DOF rigid-body integrator. The rigid-body integrator is written **once here and reused verbatim at runtime** as Atlas's non-learned expert ([[expert-library-atlas-0.1]]) — it is not a "data generator" that gets replaced later.

The key discipline of this phase: **the solvers must be validated against closed-form solutions before they are trusted to teach anything.** A surrogate trained on a subtly wrong solver learns the wrong physics perfectly, and Phase 4 would then be measuring agreement with a bug. M1 exists to prevent that.

A second discipline: the fidelity target is *"the simplest model that still contains the phenomenon we need."* We need a throat that chokes, a nozzle that produces supersonic exit flow, a shock structure in an over/under-expanded plume, a wall that heats and stresses, and a drag/pressure field that varies with Mach and altitude. We do **not** need turbulent combustion chemistry, real gas effects, or LES. Reaching for those would consume the entire project budget in Phase 0.

---

# 2. Theory

## 2.1 Compressible Navier–Stokes, 2D planar, conservative form

State vector and fluxes:

$$
U=\begin{bmatrix}\rho\\ \rho u\\ \rho v\\ E\end{bmatrix},\qquad
F(U)=\begin{bmatrix}\rho u\\ \rho u^2+p\\ \rho uv\\ (E+p)u\end{bmatrix},\qquad
G(U)=\begin{bmatrix}\rho v\\ \rho uv\\ \rho v^2+p\\ (E+p)v\end{bmatrix}
$$

with $u$ the axial ($z$) velocity, $v$ the transverse ($y$) velocity, and the governing equation

$$\partial_t U + \partial_z F(U) + \partial_y G(U) \;=\; \partial_z F_{\text{visc}} + \partial_y G_{\text{visc}} + S$$

Ideal gas closure:

$$E=\frac{p}{\gamma-1}+\tfrac12\rho\!\left(u^2+v^2\right),\qquad p=\rho R T,\qquad c=\sqrt{\gamma p/\rho}$$

Viscous fluxes use the Newtonian stress tensor with Stokes' hypothesis ($\lambda=-\tfrac23\mu$):

$$\tau_{zz}=2\mu\partial_z u+\lambda(\nabla\!\cdot\!\mathbf u),\quad \tau_{yy}=2\mu\partial_y v+\lambda(\nabla\!\cdot\!\mathbf u),\quad \tau_{zy}=\mu(\partial_y u+\partial_z v)$$

and heat flux $q_i=-k\,\partial_i T$ with $k=\mu c_p/\mathrm{Pr}$, $\mathrm{Pr}=0.72$.

**Reaction source term (agent `a` only).** Rather than finite-rate chemistry, use a single progress variable $Y\in[0,1]$ (unburnt → burnt) with a one-step Arrhenius-like law and a prescribed heat of reaction $q_{\text{rxn}}$:

$$\partial_t(\rho Y)+\nabla\!\cdot\!(\rho \mathbf u Y)=\rho\,\omega(Y,T),\qquad
\omega = A\,(1-Y)\exp\!\left(-\frac{T_{\text{act}}}{T}\right)$$
$$S_E = \rho\,\omega\,q_{\text{rxn}}$$

This is a *deliberately simple* combustion model. It produces the two behaviours Atlas needs to learn — a heat-release-driven temperature and pressure rise, and its coupling to the downstream chamber — without any chemical kinetics library.

**Numerical scheme.** Finite volume on a structured (curvilinear where the nozzle contracts) grid:
- **MUSCL reconstruction** with the minmod slope limiter (second-order in smooth regions, TVD across shocks — required, since the plume genuinely contains shocks).
- **HLLC** approximate Riemann solver for the inviscid flux; **Rusanov/local Lax–Friedrichs** as a robust fallback flag (HLLC is sharper; Rusanov is harder to break). Both must be implemented; Rusanov is the default for the first bring-up, HLLC for the production corpus.
- **Central differences** for viscous fluxes.
- **SSP-RK2** time integration.
- CFL condition: $\Delta t_{\text{solver}} \le \sigma \min_{\text{cells}} \dfrac{\min(\Delta z,\Delta y)}{|\mathbf u|+c}$ with $\sigma=0.4$.

**Boundary conditions.**
| Boundary | Treatment |
|---|---|
| Solid wall (chamber/nozzle/airframe) | no-slip for viscous, slip for Euler mode; $\partial_n p=0$; wall temperature from agent `c` coupling, or isothermal in solo mode |
| Injector face ($z=0$) | subsonic mass-flow inlet: prescribe $\dot m''$, $T_{\text{inj}}$, $Y=0$; extrapolate $p$ |
| Freestream inflow (`d`, upstream) | prescribe $(\rho_\infty,u_\infty,v_\infty,p_\infty)$ from altitude + flight Mach |
| Outflow (downstream of `f`, `g`) | supersonic: extrapolate all; subsonic: prescribe $p_\infty$, extrapolate rest |
| Symmetry $y=0$ | mirror ($v\to-v$) — **only** valid at zero angle of attack; disable for $\alpha\neq0$ runs |

## 2.2 Analytic oracles (used for M1 validation)

**Isentropic relations** (validate chamber → throat → exit):
$$\frac{T_0}{T}=1+\frac{\gamma-1}{2}M^2,\qquad \frac{p_0}{p}=\left(1+\frac{\gamma-1}{2}M^2\right)^{\frac{\gamma}{\gamma-1}}$$

**Area–Mach relation** (planar: "area" is the local half-height $\times 2$):
$$\frac{A}{A^*}=\frac1M\left[\frac{2}{\gamma+1}\left(1+\frac{\gamma-1}{2}M^2\right)\right]^{\frac{\gamma+1}{2(\gamma-1)}}$$

For $\varepsilon = A_e/A_t = 3.0$ and $\gamma=1.22$ (hot combustion products), the supersonic root gives the design exit Mach number. **The solver's centreline Mach profile must match this curve to within 2%** in the diverging section when the nozzle is running full and shock-free.

> ⚠️ **Not met on the declared contour, and the reason is geometry.** As built: 2.5% at the exit, 4.6% mean, 27.8% at the throat station, on a fully converged solution (drift $10^{-12}$). Refining $120\times24\to180\times36$ moved the error the *wrong way* (mean 4.6% → 5.2%), so it is resolution-independent; smoothing the throat corner halved it (4.6% → 2.3%). The declared contour has a hard corner at $z_t$ where the wall angle jumps $-31°\to+15°$, which curves the sonic line and throws a Prandtl–Meyer fan — **quasi-1D theory does not describe that flow**, so the oracle is invalid within ~15% of the throat rather than the solver being wrong. Meeting the criterion as literally written needs a rounded throat, which is a change to the declared geometry.

**Rayleigh flow** (constant-area heat addition — validates the reaction zone):
$$\frac{T_0}{T_0^*}=\frac{(\gamma+1)M^2\left[2+(\gamma-1)M^2\right]}{\left(1+\gamma M^2\right)^2}$$
Heat addition drives $M\to1$ from either direction; the reaction zone must show a monotone $T_0$ rise and must **not** thermally choke at the design mass flow (if it does, the prescribed $q_{\text{rxn}}$ is too high for the chosen inlet Mach — a config error, and the generator must detect and reject it).

**Ideal thrust** (validates the $e\!-\!f$ interface integral):
$$F=\dot m\,u_e+(p_e-p_\infty)A_e$$

**Dynamic pressure / max-Q** (validates the trajectory module):
$$q_\infty=\tfrac12\rho_\infty v^2$$

## 2.3 Conduction + plane-stress elasticity (agent `c`)

Transient conduction:
$$\rho_s c_{p,s}\,\partial_t T=\nabla\!\cdot\!\left(k_s\nabla T\right)$$
with Robin (convective + radiative-linearized) conditions on both faces:
$$-k_s\partial_n T = h_{\text{in}}\left(T-T_{\text{gas,in}}\right) \quad\text{(chamber side)},\qquad -k_s\partial_n T = h_{\text{out}}\left(T-T_{\text{gas,out}}\right)+\epsilon\sigma_{\text{SB}}\left(T^4-T_\infty^4\right)$$

Quasi-static plane stress with thermal expansion (the shell is thin; inertia is negligible against the ascent timescale):
$$\nabla\!\cdot\!\boldsymbol\sigma+\mathbf b=0,\qquad \boldsymbol\varepsilon=\tfrac12\!\left(\nabla\mathbf u+\nabla\mathbf u^{\!\top}\right)$$
$$\boldsymbol\sigma=\frac{E_s}{1-\nu^2}\left[(1-\nu)\boldsymbol\varepsilon+\nu\,\mathrm{tr}(\boldsymbol\varepsilon)\mathbf I\right]-\frac{E_s\alpha_s\,\Delta T}{1-\nu}\mathbf I$$

Discretization: Q1 bilinear finite elements on the shell mesh; conduction with backward Euler (unconditionally stable — the shell's thermal timescale is slow and we want big steps); elasticity solved as a linear system each step (the quasi-static assumption makes it a solve, not a march). Both assembled with a sparse CG solver, Jacobi-preconditioned.

**Analytic oracle:** a 1D slab with a step change in surface temperature has the closed-form error-function solution $T(x,t)=T_s+(T_0-T_s)\,\mathrm{erf}\!\left(x/(2\sqrt{\alpha_{\text{th}} t})\right)$; the conduction solver must match this to < 1%. For elasticity, a uniformly heated free plate must produce zero stress (pure thermal expansion) and a fully constrained one must produce $\sigma=-E_s\alpha_s\Delta T/(1-\nu)$ — both are exact tests.

## 2.4 Atmosphere (US Standard 1976, 0–32 km)

Piecewise, with $g_0=9.80665$, $R=287.053$, $T_0=288.15$ K, $p_0=101325$ Pa:

$$
\begin{aligned}
&0\!-\!11\ \text{km}: && L=0.0065,\ T=T_0-Lh, && p=p_0\left(T/T_0\right)^{g_0/(LR)}\\
&11\!-\!20\ \text{km}: && T=216.65, && p=p_{11}\exp\!\left[-g_0(h-11000)/(RT)\right]\\
&20\!-\!32\ \text{km}: && L=-0.001,\ T=216.65-L(h-20000), && p=p_{20}\left(T/216.65\right)^{g_0/(LR)}
\end{aligned}
$$

with $\rho=p/(RT)$, $c=\sqrt{\gamma_{\text{air}}RT}$, $\gamma_{\text{air}}=1.4$. Sutherland's law for viscosity: $\mu=\mu_{\text{ref}}\left(T/T_{\text{ref}}\right)^{3/2}\frac{T_{\text{ref}}+S}{T+S}$, $S=110.4$ K.

## 2.5 Rigid body, 3-DOF planar

State $\mathbf s=(x,y,\theta,v_x,v_y,\omega,m)$ in the **world** frame:

$$\dot m=-\dot m_p,\qquad m\dot{\mathbf v}=\mathbf F_{\text{thrust}}+\mathbf F_{\text{aero}}+m\,\mathbf g(h),\qquad I(m)\,\dot\omega=\tau_{\text{aero}}+\tau_{\text{thrust}}$$

$$\mathbf g(h)=-g_0\left(\frac{R_E}{R_E+h}\right)^{2}\hat{\mathbf y},\qquad R_E=6.371\times10^{6}\ \text{m}$$

$\mathbf F_{\text{thrust}}$ comes from the $e\!-\!f$ interface integral; $\mathbf F_{\text{aero}}$ from the $c\!-\!d$ surface integral of $\left(-p\,\mathbf n+\boldsymbol\tau\!\cdot\!\mathbf n\right)$. Integrated with classical RK4.

> **Note on integrator choice:** a symplectic integrator would be the natural default for a conservative system, but this system is *not* conservative — mass is expelled, drag dissipates, thrust does work. RK4 is the correct choice here; do not "upgrade" it to leapfrog.

---

# 3. Implementation

## 3.1 `solvers/compressible2d.py`

```python
@dataclass
class GasConfig:
    gamma: float            # 1.22 combustion products, 1.4 air
    R: float                # J/(kg K)
    mu_ref: float; T_mu_ref: float; sutherland_S: float
    Pr: float = 0.72
    riemann: str = "hllc"   # "hllc" | "rusanov"
    limiter: str = "minmod"
    cfl: float = 0.4
    inviscid: bool = False

class Compressible2D:
    def __init__(self, grid: CurvilinearGrid, cfg: GasConfig,
                 source: SourceTerm | None = None,
                 bcs: dict[str, BoundaryCondition] = ...): ...
    def step(self, U: np.ndarray, dt: float) -> np.ndarray: ...
    def run_to(self, U0, t_end, save_every) -> Trajectory: ...
    def max_stable_dt(self, U) -> float: ...
```

`U` has shape `[4, Nz, Ny]` (plus a 5th row for `rho*Y` when a reaction source is attached). The grid carries cell centroids, face normals, face areas, and cell volumes so the same class serves both the straight external domains and the contracting nozzle without special-casing.

**Build order (do not reorder — each step is testable alone):**
1. `CurvilinearGrid` + metric terms; unit-test that $\sum_{\text{faces}} \mathbf n\,A = 0$ per cell (closure of the control volume) to machine precision.
2. Rusanov flux + first-order Godunov + SSP-RK2, `inviscid=True`. Test: 1D Sod shock tube along $z$ matches the exact Riemann solution (positions of shock/contact/fan within one cell).
3. MUSCL + minmod. Test: order of accuracy on a smooth isentropic vortex → observed order between 1.7 and 2.0.
4. HLLC. Test: same Sod result, sharper contact; and that HLLC and Rusanov agree to < 1% on smooth flows.
5. Viscous fluxes. Test: laminar flat-plate boundary layer thickness matches Blasius $\delta\approx5.0\,z/\sqrt{\mathrm{Re}_z}$ within 10%.
6. Reaction source term with operator splitting (Strang: half-step source, full-step convection, half-step source). Test: zero-velocity constant-pressure reactor reaches the adiabatic flame temperature implied by $q_{\text{rxn}}$.
7. Wall/inlet/outlet/symmetry BCs. Test: uniform flow is preserved exactly through every BC type (free-stream preservation).

## 3.2 `solvers/thermostruct2d.py`

```python
class ThermoStruct2D:
    def __init__(self, mesh: ShellMesh, mat: SolidMaterial): ...
    def step_thermal(self, T, dt, h_in, T_gas_in, h_out, T_gas_out) -> np.ndarray: ...
    def solve_mechanical(self, T, p_in, p_out) -> tuple[u, sigma]: ...
```

Material default (an aluminium-lithium-like airframe alloy, values are *reference defaults, not a specification*): $\rho_s=2700$, $c_{p,s}=900$, $k_s=120$, $E_s=70$ GPa, $\nu=0.33$, $\alpha_s=23\times10^{-6}$.

Note the mechanical solve takes the **pressure loads from both gas sides** ($p$ from agent `b` inside, from agent `d` outside) — this is exactly the $b\!-\!c$ and $c\!-\!d$ coupling that Atlas will later have to reproduce through its typed edges.

## 3.3 `solvers/trajectory.py` and `solvers/atmosphere.py`

Straightforward given §2.4–2.5. **`trajectory.py` must be written as a pure function of state and applied loads**, with no dependency on any solver internals, because [[impl-atlas-0.1-phase2-experts]] imports it directly as the runtime `rigid_body` expert.

```python
def rhs(s: State, F_thrust: Vec2, F_aero: Vec2, tau: float, mdot: float) -> State: ...
def rk4_step(s: State, dt: float, loads: Loads) -> State: ...
```

## 3.4 Coupled data generation (`data/generate.py`)

A training episode is **one coupled run of all solvers together**, because Atlas must learn interface behaviour, and interface behaviour only exists in a coupled run. Loose (Gauss–Seidel) coupling at the macro-step is sufficient and is what the surrogate will imitate:

```
for macro_step in range(N):
    # 1. gas interior, subcycled at solver CFL
    advance(a, b, e, dt_macro)          # walls use c's current surface temperature
    advance(d, f, g, dt_macro)          # inflow from atmosphere(h), plume inlet from e's exit state
    # 2. interface fluxes
    q_wall_in  = heat_flux(b -> c);  p_wall_in  = pressure(b -> c)
    q_wall_out = heat_flux(d -> c);  p_wall_out = pressure(d -> c)
    # 3. structure
    T_c = thermostruct.step_thermal(...);  u_c, sigma_c = thermostruct.solve_mechanical(...)
    # 4. loads -> trajectory
    F_thrust = integrate_flux(e -> f);  F_aero = integrate_traction(c -> d)
    s = rk4_step(s, dt_macro, loads);   h = s.y
    # 5. snapshot
    save(all agents, all interface fluxes, s, conditioning scalars)
```

**Critical:** the saved record must include the **interface flux fields themselves**, not only the agent interiors. Those fluxes are the supervision signal for the edge behaviour and for the conservation constraint in [[impl-atlas-0.1-phase3-integration]]. A corpus that saves only interiors cannot train or validate the interface layer, and regenerating is expensive — get this right the first time.

## 3.5 Sweep design

| Parameter | Range | Notes |
|---|---|---|
| Chamber pressure $p_c$ | 2, 4, 6, 8 MPa | sets $\dot m$ via choked-throat relation |
| Adiabatic flame temp $T_c$ | 2200, 2800, 3400 K | set via $q_{\text{rxn}}$ |
| Altitude $h$ | 0–25 km | drives $p_\infty$, hence nozzle over/under-expansion |
| Flight Mach $M_\infty$ | 0.1–3.5 | subsonic through supersonic external flow |
| Angle of attack $\alpha$ | 0°, 2°, 5° | breaks the $y$-symmetry; disables the mirror BC |

> ⚠️ **The injector temperature is not a free parameter, and it is not in this table.** Rayleigh flow ties it to the chamber area ratio and the top of the $T_c$ range: at $A_c/A_t=2.5$ the subsonic inlet sits at $M=0.238$, whose choking limit is $T_0^*/T_{0,\text{in}}=4.10$, so $T_c/T_{\text{inject}}$ must stay under ~3.90 with margin. At $T_{\text{inject}}=700$ K that caps $T_c$ at 2730 K and the screening below rejects **246 of 546** candidate configs — all at the hot end, silently truncating the sweep exactly where the physics is most interesting. As built: **$T_{\text{inject}} = 900$ K** (regeneratively preheated propellant), giving 300 + 40 configs with **zero rejections** and a minimum margin of 0.125.

Full factorial is 2304 configs — too many. Use a **Latin hypercube sample of 300 configs for train, plus 40 held-out configs for test**, where the held-out set deliberately includes *corner* cases (highest $p_c$ at highest altitude — the most under-expanded plume; lowest $p_c$ at sea level — the most over-expanded, flow-separating case). Holding out corners rather than random points is the honest test: interpolation inside a swept box is the easy case, and [[transfer-learning-fine-tuning]]'s diversity principle warns that in-distribution success overstates generality.

Each config: run to quasi-steady, then save **200 transient snapshots** at $\Delta t_{\text{macro}}=5\times10^{-2}$ s.

## 3.6 Storage schema (`HDF5`, float32)

```
episode_XXXX.h5
  /meta            p_c, T_c, h0, M_inf, alpha, gamma_gas, seed, solver_versions
  /refs/{agent}    rho_ref, u_ref, T_ref, L_ref, p_ref     # frozen per episode
  /cond/{agent}    [T, n_cond]   Ma, Re, Pr, gamma, p_ratio (gas) | Bi, alphaDT (solid)
  /fields/{agent}  [T, C, Nz, Ny]   nondimensionalized
  /iface/{edge}    [T, C_edge, N_iface]  fluxes + traces on the shared interface
  /rigid           [T, 7]  (x, y, theta, vx, vy, omega, m)
  /loads           [T, 4]  (Fx_thrust, Fy_thrust, Fx_aero, Fy_aero)
```

`data/normalize.py` computes and freezes `/refs` **per episode before the run starts** (from $p_c$, $T_c$, and the throat dimension), never from the realized data — normalizing by statistics of the trajectory itself would leak future information into the model's input scaling.

## 3.7 Acceptance tests (M1 + M2)

**M1 — solver correctness (all must pass before generating any corpus):**
- Grid metric closure to machine precision.
- Sod shock tube vs. exact Riemann solution: wave positions within 1 cell.
- Isentropic vortex: observed order of accuracy ≥ 1.7.
- Free-stream preservation through every BC type: uniform flow unchanged to $10^{-12}$.
- **Nozzle centreline Mach vs. area–Mach relation: < 2% error** through the diverging section, shock-free case.
- Constant-pressure reactor reaches the adiabatic flame temperature within 1%.
- Blasius boundary-layer thickness within 10%.
- Conduction vs. erf-solution slab: < 1%.
- Free thermal expansion produces zero stress; fully constrained produces $-E_s\alpha_s\Delta T/(1-\nu)$ exactly.
- Atmosphere table matches published US Standard 1976 values at 0/5/11/20/30 km to < 0.5%.
- Trajectory: with zero aero and constant thrust, matches the analytic Tsiolkovsky $\Delta v = v_e\ln(m_0/m_f)$ to < 0.1%.

**M2 — corpus validity:**
- ≥ 300 train + 40 test episodes complete without NaN or CFL failure.
- Every episode has all 7 agents, all 7 interfaces, 200 snapshots.
- Global mass budget per episode closes: $\left|\Delta m_{\text{system}} + \int \dot m_{\text{exit}}\,dt\right| / m_0 < 10^{-3}$.
- Held-out corner cases show *visibly different* plume structure from the training mean (a sanity check that the sweep actually spans regimes rather than producing near-identical episodes).

### As-built M1 results (2026-08-09)

| Oracle | Criterion | Measured |
|---|---|---|
| Grid metric closure | machine precision | $4\times10^{-19}$ |
| Sod shock tube | waves within 1 cell | shock 1.0, contact 1.5 cells; $p^*,u^*$ exact to $10^{-3}$ |
| HLLC vs Rusanov | < 1% smooth | 0.4% |
| Isentropic vortex | order ≥ 1.7 | **1.86** |
| Free-stream preservation | $10^{-12}$ | **0.0**, incl. the curvilinear nozzle |
| Constant-p reactor | 1% | 0.02% |
| **Blasius** | **10%** | **15–19% low — NOT MET** |
| Stokes shear layer *(added)* | exact viscous grading | 1.2% |
| Conduction vs erf slab | < 1% | **0.14%** |
| Free / constrained thermal stress | zero / exact | $10^{-4}$ Pa on a $10^{8}$ Pa scale; exact to $10^{-6}$ |
| US1976 at 0/5/11/20/30 km | < 0.5% | < 0.1% |
| Tsiolkovsky | < 0.1% | 0.02% |
| **Nozzle area–Mach** | **< 2%** | **2.5% exit — NOT MET** (see §2.2) |

**Blasius, why it is not met.** Measured $\delta_{99}$ runs 15–19% below the similarity solution, and the shortfall tracks a residual freestream acceleration ($u_e/u_\infty \approx 1.035$) that a *prescribed-primitive* far-field cannot shed — an accelerating outer flow thins the layer and raises the wall shear, which is exactly what was seen ($\delta$ −31%, $\tau_w$ +67% with the plate's leading edge on the inflow plane; −19% and +3.5% after a slip run-up was added). The similarity **scaling** is reproduced exactly: the error is constant across stations, so $\delta \propto \sqrt{z}$ with a low coefficient, not a diffusion bug. A new **Stokes shear-layer** test grades the viscous flux exactly (1.2%) and covers what Blasius was meant to. Closing the remaining gap needs a **characteristic (Riemann-invariant) far-field BC** — an open item; it does not block the corpus, where agent `d` has a genuine freestream at a real flight Mach number.

### 3.8 M2 is blocked on compute — the decision this phase hands upward

Measured, not estimated (`scripts/atlas_generate_corpus.py --estimate` times the real per-substep cost and multiplies by the substep count the CFL condition demands):

$$\underbrace{14\,762\ \text{s}}_{\text{per macro step}}\ \Longrightarrow\ \underbrace{820\ \text{h}}_{\text{per episode}}\ \Longrightarrow\ \underbrace{30.6\ \text{years}}_{\text{327 episodes, 1 core}}$$

Not slow code: $\Delta t_{\text{CFL}}$ is set by the **acoustic** speed while the flow evolves at the convective one, so $\Delta t_{\text{macro}}/\Delta t_{\text{CFL}} \approx 1.5\times10^{5}$ for the nozzle, and 200 snapshots means 10 s of flight resolved at $3\times10^{-7}$ s. Even a 1000× speedup leaves 11 days. The two-timestep-family table in [[00-atlas-0.1-implementation-plan]] already states the ratio; what it never does is multiply it out.

Three routes, none of which the build layer should choose alone:

| | Change | Consequence |
|---|---|---|
| **A** | Quasi-steady gas: re-converge per macro step instead of marching | ~100× cheaper; Atlas then learns a steady *mapping*, not transient dynamics |
| **B** | Keep time-accurate marching, shorten episodes 10 s → ~10 ms | Feasible today; loses the trajectory coupling that motivates the rigid-body expert |
| **C** | Implicit / low-Mach preconditioning to remove the acoustic limit | Physically right; a substantial solver project |

> **Costed, and a fourth route recommended (2026-08-09):** [[impl-atlas-0.1-compute-and-training-budget]] prices all three above and proposes **B′** — decouple the *snapshot cadence* from $\Delta t_{\text{macro}}$ (this section conflates them), giving $T_{\text{ep}}=0.2$ s at $\Delta t_{\text{snap}}=10^{-3}$ s: 4 macro steps, still 200 snapshots, $50\times$ cheaper, with $\Delta t_{\text{model}}$, multi-rate subcycling and the surrogate's speedup ratio all unchanged. Plain option B, read as "10 ms episodes," drags $\Delta t_{\text{model}}$ down to ~3× $\Delta t_{\text{CFL}}$ and destroys the speedup claim.

A reduced-fidelity corpus (11 episodes, physically correct, mass-budget residual $10^{-7}$–$10^{-5}$) is generated and validated as proof the pipeline runs end to end. `EpisodeSpec` exposes `dt_macro`, `n_macro`, `coarsen`, all recorded in `/meta` with a `full_fidelity` flag, so a cheap run can never be mistaken for a production one.

---

# 4. Pitfalls

- **Thermal choking.** If $q_{\text{rxn}}$ is too large for the inlet Mach number, Rayleigh flow drives $M\to1$ inside the reaction zone and the solver will either stall or produce nonsense. The generator must check the Rayleigh limit *before* running and reject the config with a clear error, not discover it 40 minutes in.
- **Symmetry BC with $\alpha\neq0$.** Silently wrong, not loudly wrong. Assert that the mirror BC is disabled whenever $\alpha\neq0$.
- **Normalizing from realized data.** Leaks the future into the input scaling. Freeze `/refs` before the run.
- **Saving interiors only.** Makes the entire interface/conservation layer untrainable and forces corpus regeneration. See §3.4.
- **Over-reaching on combustion fidelity.** A one-step progress variable is the specification, not a placeholder to be upgraded mid-phase.
- **A fixed Arrhenius pre-exponential.** Induction time scales as $\exp(T_{\text{act}}/T_{\text{inject}})$, so one hardcoded $A$ either fails to ignite or burns in the first cell. Size $A$ per episode from agent `a`'s residence time.
- **Feeding geometric altitude to the geopotential layer formulas.** $H = R_E Z/(R_E+Z)$; skipping it is a 2.1% pressure error at 30 km — small enough to look like solver noise, large enough to move a drag calculation.
- **Removing rigid-body modes by pinning nodes.** Any pin that also blocks a component of uniform thermal expansion manufactures stress in a plate that should have none — the free-expansion oracle itself. Use a bordered (Lagrange-multiplier) solve, and a *direct* one: Jacobi-preconditioned CG converges on a square test plate and fails outright on the real shell, whose elements are 20:1 slivers.
- **Bounding-box grids with a masked wall for the nozzle agents.** They give the solver a staircase contour, which cannot meet an area–Mach oracle at any resolution. Body-fitted grids, shared with the tokenizer cell-for-cell.
- **Capping gas substeps to make an episode affordable.** The gas then advances less physical time than the structure and the trajectory, silently desynchronizing the coupling, and still writes a valid-looking file. Shorten the episode or coarsen the grid instead, and record which.

---

## See Also

- [[00-atlas-0.1-implementation-plan]] — global conventions (geometry, agents, edges, timesteps)
- [[impl-atlas-0.1-phase1-scaffold]] — consumes this corpus's schema
- [[impl-atlas-0.1-phase2-experts]] — trains on this corpus
- [[impl-atlas-0.1-phase4-validation]] — uses these solvers as the judge
- [[case-study-rocket-ascent-2d-atlas-0.1]] — design-level statement of this phase
