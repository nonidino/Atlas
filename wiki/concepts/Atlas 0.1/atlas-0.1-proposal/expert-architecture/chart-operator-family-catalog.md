# The chart-operator family catalog — the PDEs, the interactions a decomposition creates, unit-square speedups, and one architecture per family

**Type:** Concept page — **design reference under discussion** (folder: `Atlas 0.1/atlas-0.1-proposal/expert-architecture/`)
**Status:** written 2026-10-10 in answer to the owner's request of that day. The request asked for four things: the PDEs that Atlas's experts will cover; the interactions a full decomposition yields and how each is accounted for (same field across a seam, different fields across a seam, different fields on the same domain); speedups specific to the unit square; and a catalog of chart-operator architectures. The long form is the owner's artifact *Chart Operator Families* (https://claude.ai/artifact/6YFtDeR3cX96ifqn8pbjuk, private to the owner); this page is its vault digest, with the same numbers. **A proposal for discussion, not a decision.** New measurements come from `scripts/arch_family_checks.py` (`out/arch/family_checks.txt`), with classical solves standing in for networks. Everything else is a derivation, a cited result, or **[AI Inference]**.
**Hub:** [[00-proposal-workstreams]] · **Extends:** [[chart-operator-general-form]] · [[chart-operator-architecture]] · **Builds on the record:** [[port-algebra-atlas-0.1]] · [[case-study-ladder-to-f1]] · [[case-study-rocket-ascent-2d-atlas-0.1]] · [[case-study-thermal-strain-atlas-0.1]] · [[coupling-cost-and-complexity]]

---

## 0. The answer in one paragraph

**A chart operator is five choices:**
1. what the chart passes to the network;
2. the trial space the network proposes (its trace space and modes);
3. the ports it exposes and what it conserves exactly;
4. the host contract (guarantee class G0, G0′, G1, G2 or G3);
5. the backbone and its time representation.

The decomposition, the host and the Expert Card are shared by every family. **Families are separated by trace space and guarantee class, not by PDE.** Twelve families cover about forty PDEs, because an expert used on a neighbouring PDE with the same trace space costs accuracy (quadratically, by the Gram identity) and never consistency. **A decomposition produces eight kinds of interaction.** The owner's three are I-1 (same family across a seam), I-2 (different families across a seam) and I-3 (different fields on one piece). The others are non-local exchange, global constraints, changing geometry, time scales and length scales. Each has one host-owned instrument, and all rest on one principle: passive experts joined by power-conserving junctions. **The chart buys the FFT.** Three measurements bear this out: conjugate gradients preconditioned by the plain square's fast sine solve need 30 iterations at every grid from $32^2$ to $256^2$; strip-cropped mode heads save 2.3× (steady) to 5.6× (small time steps) on undistorted pieces; and a bfloat16 correction costs a Gram error of $3.2\times10^{-6}$. **Two-dimensional duality gives a second certificate:** the same diffusion network on $1/\kappa$ bounds each piece's port matrix from below, and the gap to Theorem 5's upper bound was within a factor of 3 of the true error for smooth data.

---

## 1. Anatomy: the five slots

**The idea.** Each piece is the image of the unit square under a certified chart $F$. The network sees the pulled-back problem and proposes a trial space. The host evaluates the true physics on that space and couples the pieces.

### 1.1 Slot 1 — what the chart passes

**Intuition:** a temperature moves with its point, a flux must be re-measured through the deformed surface, and a density is rescaled by the area change. **Rigour:** fields are differential forms, each with its own pull-back (Arnold, Falk & Winther, *Acta Numer.* 15, 2006), with $J=\det DF$:

| kind | examples | pull-back | network input |
|---|---|---|---|
| scalar (0-form) | $T$, $\phi$, $p$, $\psi$, $A_z$ | $\hat u=u\circ F$ | field |
| gradient (1-form) | $\nabla T$, $E$ | $DF^{\top}(E\circ F)$ | computed |
| flux ($(d-1)$-form) | heat flux, current, mass flux, advecting velocity | $J\,DF^{-1}(q\circ F)$ (Piola) | Piola fields; side fluxes are port data |
| density ($d$-form) | $\rho c$, sources | $J\,(f\circ F)$ | $\log J$, $\hat f$ |
| constitutive tensor | $\kappa$, $\sigma$, $\epsilon$, $\nu$ | $\mathsf A=J\,DF^{-1}\kappa\,DF^{-\top}$ | in 2-D: $\tfrac12\log\det\mathsf A$ and a Beltrami coefficient $\mu_{\mathsf A}$ |
| Cartesian vector unknown | displacement, velocity | $DF=RU$; unknowns in the local frame $R$ | stretch $U$, rotation gradient $\hat\nabla\vartheta$ |
| shell midsurface | skins | fundamental forms $a_{\alpha\beta}$, $b_{\alpha\beta}$ | $a$, $b$, thickness; chart $C^2$ |
| moving piece | ALE fluid pieces | mesh velocity by Piola; $\partial_tJ=\hat\nabla\cdot(J\,DF^{-1}\hat w)$ | $\hat w$; geometric conservation law at the host |
| state, step, side types | $T$, $u$, internal variables; Fo or Courant number; port, wall, inflow, outflow | — | channels, scalars, flags |

Three derived consequences:

- **In 2-D, geometry and anisotropy are one thing to a diffusion operator.** The operator sees only $\mathsf A$, and $\det\mathsf A=\det\kappa$ for every chart. The chart changes only $\mathsf A$'s conformal class, which takes two numbers. One envelope therefore covers anisotropic materials on mild charts and isotropic materials on distorted charts, and $\log J$ enters only through capacity, reaction and sources.
- **A material-adapted chart.** **[AI Inference]:** by the measurable Riemann mapping theorem, any bounded 2-D anisotropy can be absorbed into a quasiconformal chart, which would hand the network an isotropic coefficient. The cost is one Beltrami-equation solve per piece.
- **Vector unknowns in the local frame make rotations exact.** With $u=R\tilde u$ and rotation angle $\vartheta$, the strain written in the local frame is $\operatorname{sym}\big(\hat\nabla\tilde u\,U^{-1}+W\tilde u\otimes U^{-1}\hat\nabla\vartheta\big)$, with $W$ the rotation by $90^\circ$. For an isotropic material the elastic energy depends on the chart only through $U$ and $\hat\nabla\vartheta$, so a rigidly rotated piece is the identical problem by construction.

### 1.2 Slot 2 — the trace space fixes the expert's shape

| space | PDEs | primal trace (port) | dual trace | at corners |
|---|---|---|---|---|
| $H^1$ scalar | diffusion, potential flow, $A_z$ | value | normal flux | vertex continuity |
| $H^1$ vector | elasticity, Stokes velocity | displacement or velocity | traction | rigid modes in port space |
| $H(\mathrm{div})$ | mixed Darcy, dual diffusion | normal flux | potential | compatibility |
| $H(\mathrm{curl})$ (3-D) | Maxwell, eddy currents | $n\times E$ | $n\times H$ | gauge null space |
| $H^2$ | Kirchhoff plate, 2-D Stokes stream function | value and slope | effective shear, moment | concentrated corner forces |
| impedance | Helmholtz, time-harmonic Maxwell | incoming $\partial_nu+iku$ | outgoing $\partial_nu-iku$ | none |
| characteristic | Euler, explicit acoustics | incoming amplitudes | outgoing amplitudes | none |

The trial set is $u=u_p+Vc+Zd$ ([[chart-operator-general-form]] §2.1):
- $u_p$, the predicted field;
- $V$, the constraint modes, which have exact traces;
- $Z$, the enrichment modes, which have zero trace (vibration modes, cross-field responses, state tangents).

Every mode keeps version 1's principal part: the plain square's exact modes plus a bubble-weighted correction.

### 1.3 Slots 3 and 4 — ports, conservation, guarantee classes

Ports come from the closed vocabulary of [[port-algebra-atlas-0.1]]: `MECH`, `ROT`, `THERM`, `ELEC` and `ADVEC` with declared passengers.

| class | structure | host | for every network output | families |
|---|---|---|---|---|
| **G0** | linear symmetric energy | Gram, one interface solve | $\tilde\Lambda\succeq0$; $\tilde\Lambda-\Lambda=E^{\top}A_IE\succeq0$; Céa constant 1; machine-checked | CO-D, CO-E, CO-P, CO-S, implicit CO-W |
| **G0′** | linear non-symmetric or complex-symmetric | Petrov–Galerkin with adjoint modes | error $E_W^{\top}A_{II}E_V$; no positivity | CO-T, Oseen steps, Helmholtz |
| **G1** | convex energy | nonlinear Ritz | energy upper bound; at most $\sqrt{L/m}$ times best; never worse than one classical step | CO-DN, CO-H locally, contact |
| **G2** | no energy | LSPG on the true residual, certificate, defect-correction fallback | exact residual; no certified error bound | CO-NS, RANS, friction, radiation coupling |
| **G3** | explicit hyperbolic | face fluxes, convex limiting | exact conservation; positivity; entropy stability only with entropy-stable fluxes | CO-G, high-Péclet CO-T |

What is conserved exactly:
- **G0:** total flux, force and moment per piece, because the constant and rigid modes lie in the port space.
- **G3:** every conserved quantity, through single-valued face fluxes.
- **Incompressible families:** mass, through divergence-free modes.
- **Chemistry:** the elements, by projection.

### 1.4 Slot 5 and the Expert Card

The backbone is version 1's U-shaped convolutional trunk (about $10^6$ parameters, $n=64$), with heads for modes, the predicted field, adjoint modes (G0′), dual modes (§5) and enrichment. Time representations are steady, implicit (Fo as a reaction coefficient), explicit (Courant number) or exponential.

Measured forward cost per piece at $n=64$, batched by 16: 3.8, 7.7 and 25 ms for 0.49, 1.95 and 6.2 M parameters (`arch_backbone_timing.py`).

**[AI Inference]:** the Expert Card gains seven fields:
- `trace_space`;
- `class`;
- `label_map`, so that one network can be several experts;
- `bond_channels` (source and coefficient);
- `heads`;
- `dual`;
- `coupling_numbers`.

---

## 2. The PDE catalog

| group (family · class) | PDEs | Atlas use |
|---|---|---|
| **A. Linear diffusion** (CO-D · G0) | heat $\rho c\,\partial_tT-\nabla\cdot(\kappa\nabla T)=f$; Darcy $-\nabla\cdot(\tfrac{k}{\mu_f}\nabla p)=q$; electrostatics and steady current; 2-D magnetostatics $-\nabla\cdot(\nu\nabla A_z)=J_z$; implicit 2-D eddy currents; potential flow; Prandtl torsion; membranes; Fick diffusion; the implicit Newmark acoustic step $\tfrac{1}{\beta c^2\Delta t^2}p-\nabla^2p=s$ | airframe $c$, nozzle, brakes ([[case-study-brake-thermal-atlas-0.1]]), conductors and motor sections ([[case-study-powertrain-atlas-0.1]]) |
| **B. Nonlinear diffusion and phase change** (CO-DN · G1) | $\kappa(T)$; radiating boundaries (host boundary energy $\oint\varepsilon\sigma_{SB}(\tfrac15T^5-T_\infty^4T)$, convex for $T\ge0$); Stefan problem in enthalpy form; ablation; saturating magnetostatics (convex when $s\mapsto\nu(s)s$ increases); $p$-Laplacian (regularise); phase field by convex splitting (Eyre, *MRS Proc.* 529, 1998); Richards | nozzle, heat shield, hot brakes, motors |
| **C. Transport** (CO-T · G0′, G3 when advection dominates) | advection–diffusion; pure advection and level sets; turbulence-model transport (within CO-NS, G2) | coolant ([[case-study-cooling-loop-atlas-0.1]]), wakes ([[spec-wind-farm-wake-atlas-0.1]]), plume species |
| **D. Solids** (CO-E · G0; CO-H · G1) | linear elasticity; axisymmetric (weight $r$, hoop strain $u_r/r$); nearly incompressible $(u,p)$; implicit elastodynamics; hyperelasticity; incremental plasticity (Ortiz & Stainier, *CMAME* 171, 1999); unilateral contact (host); Coulomb friction (host, G2) | airframe, wing ([[case-study-wing-fsi-atlas-0.1]]), tyre (CS-15), brake pads |
| **E. Thin structures** (CO-P · G0) | Kirchhoff–Love $\nabla\cdot\nabla\cdot(\mathbb D:\nabla\nabla w)=q$; Reissner–Mindlin; Koiter shell | rocket skin, monocoque, wing skins |
| **F. Incompressible flow** (CO-S · G0; CO-NS · G2) | 2-D Stokes $\nu\nabla^4\psi=\hat z\cdot\nabla\times f$; 3-D Stokes with divergence-free modes; Oseen (G0′); Navier–Stokes; RANS; Boussinesq (a bond) | ground effect ([[case-study-ground-effect-atlas-0.1]]), ducts, wind farm |
| **G. Compressible and reacting** (CO-G · G3; CO-R split) | Euler; compressible Navier–Stokes (convective part G3, viscous part G0 per step); reacting flow $\partial_t(\rho Y_k)+\nabla\cdot(\rho uY_k-\rho D_k\nabla Y_k)=\dot\omega_k$ | rocket $a$, $b$, $d$–$g$ |
| **H. Waves** (CO-W) | explicit acoustics (CFL-limited); Helmholtz with impedance ports (G0′, plain transpose); absorbing layers as complex charts | lift-off and exhaust acoustics |
| **I. 3-D electromagnetics** (CO-M) | eddy currents $\nabla\times(\nu\nabla\times A)+\sigma\partial_tA=J_s$ ($H(\mathrm{curl})$, gauge at the host); time-harmonic Maxwell (G0′) | motor end regions |
| **J. Co-located multiphysics: bonds, not families** (I-3) | thermoelasticity; Biot poroelasticity; piezoelectricity; Joule heating with $\sigma(T)$; thermoelectric effects ($\Pi=ST$); porous-electrode batteries | [[case-study-thermal-strain-atlas-0.1]], airframe, batteries |
| **K. Lumped components: not chart operators** | circuits, pipe networks, thermal networks, rigid bodies, motor maps, battery circuits, controllers, as port-Hamiltonian systems $\dot x=(\mathcal J-\mathcal R)\nabla H+Bu$, $y=B^{\top}\nabla H$, passive by construction | everywhere |
| **L. Outside the catalog** | radiative transfer in participating media; rarefied gas; plasma and MHD; turbulence-resolving closures; fracture tips; molecular and quantum models | — |

**Six observations that keep the catalog small:**
1. **Same trace space and class, same network.** Theorem 5's identity holds for any interior field with the right trace, so an expert for PDE $X$ used on PDE $Y$ gives $\tilde\Lambda-\Lambda_Y=E^{\top}A_YE$ with $E=H_X-H_Y$: consistent, and quadratically less accurate.
2. **The Kirchhoff transform moves nonlinearity onto the seams.** In one material, $\Theta=\int_{T_0}^{T}\kappa(s)\,ds$ gives $\kappa(T)\nabla T=\nabla\Theta$, so the steady piece problem is linear in $\Theta$ and uses CO-D's modes. Cutting at material interfaces makes every piece one material. Seam continuity then becomes $K_1^{-1}(\Theta_1)=K_2^{-1}(\Theta_2)$ while flux balance stays linear, and the host runs Newton's method on port coefficients only. The transient capacity term needs frozen-coefficient iteration.
3. **One biharmonic network, two or three uses.** The Kirchhoff plate and the 2-D Stokes stream function are the same equation. **[AI Inference]:** through the Airy stress function, the plate network would also give the dual certificate of plane elasticity (unchecked).
4. **Implicit acoustics and implicit heat** are both $-\nabla\cdot(\mathsf A\nabla u)+cu$ with $c>0$.
5. **An absorbing layer is a complex chart.** The stretch $s_x=1+i\sigma(x)/\omega$ gives $\mathsf A=\operatorname{diag}(s_y/s_x,\,s_x/s_y)$ and $J=s_xs_y$, which is the pull-back formula with a complex Jacobian.
6. **In 2-D the dual of CO-D is CO-D** on $1/\kappa$ (§5).

---

## 3. Interactions and how each is accounted for

| ID | interaction | example | host instrument |
|---|---|---|---|
| I-1 | same family across a seam | two airframe pieces | Gram assembly (G0); Petrov–Galerkin (G0′); energy or residual minimisation (G1, G2); one numerical flux per face (G3); mortar for non-matching sides |
| I-2 | different families across a seam | the air against the rocket frame | typed port intersection; physical frame; primal by $P$, flux by $P^{\top}$; causality; condensed impedance; interface laws; shell–solid and field–lumped constraints; moving seams; multirate |
| I-3 | different fields on one piece | stress and heat in one frame | one-way, staggered or monolithic bond with cross-enrichment; splits through coefficient channels; splitting for chemistry |
| I-4 | non-local | plume radiating to the base | dense or hierarchical host operator; exterior maps; complex charts (G2) |
| I-5 | global constraints, lumped parts | pressure level; vehicle motion | bordered system on constant or rigid modes |
| I-6 | geometry or topology change | ablation; staging | re-chart and re-certify; a new decomposition for topology |
| I-7 | several time scales | braking pulse against cooling | multirate with flux accumulation; waveform relaxation; quasi-static elimination |
| I-8 | several length scales | layups, electrode particles | periodic cell problems (Rayleigh–Ritz bounds), FE$^2$ |

**The principle under all eight.** Passive systems joined by power-conserving junctions form a passive system (van der Schaft & Jeltsema, *Found. Trends Syst. Control* 1, 2014; Willems, *Arch. Rational Mech. Anal.* 45, 1972). Each class realises passivity differently:
- G0 has a positive semidefinite Gram;
- G1 has a convex energy;
- G3 is passive in the entropy sense only with entropy-stable fluxes;
- G0′ is passive only if the port pairing includes the advective flux;
- G2 has no structure, hence its certificate.

**I-1, notes.**
- G0′ is measured quadratic: slopes 1.97, 1.98 and 2.00 at Péclet 0, 20 and 200, against 1.96, 1.12 and 0.99 for Galerkin (`arch_general_form_checks.py`).
- At G3 seams, entropy-stable fluxes (Tadmor, *Acta Numer.* 12, 2003) give the entropy inequality, and open boundaries use characteristic ports (Thompson, *J. Comput. Phys.* 68, 1987; Poinsot & Lele, *J. Comput. Phys.* 101, 1992).
- At corners, the conforming sines-plus-corner basis (C6) removes the paper's Remark 4 problem. $H^2$ families carry concentrated corner forces and stream-function slope compatibility.
- Floating pieces have singular port matrices; a 3-D coarse space for them is open.

**I-2, notes** (for the rocket, see [[chart-operator-general-form]] §3).
1. **Typed intersection.** A seam carries only the port types both sides declare. An empty or partial intersection is a modelling statement the host reports.
2. **Mortar.** If primal values go $u_{\text{air}}=P\,u_{\text{frame}}$, fluxes must return by $P^{\top}$. Then $f_{\text{air}}\cdot u_{\text{air}}=f_{\text{frame}}\cdot u_{\text{frame}}$ identically, so the transfer creates no power ([[interface-transfer-theory]]).
3. **Causality.** The stiffer side imposes the primal value.
   - Thermally (Giles, *IJNMF* 25, 1997; Errera & Chemin, *J. Comput. Phys.* 245, 2013), the solid sets the wall temperature and receives the flux.
   - Mechanically, watch the added-mass instability (Causin, Gerbeau & Nobile, *CMAME* 194, 2005); the cures are Robin exchange or added-mass partitioning.
   - CS-12's structure is about 5300 times its fluid per mode, which is safe.
4. **Condensed impedance.** A G0 neighbour's Gram is its exact Dirichlet-to-Neumann map in port space, and $V^{\top}(K+\tfrac{1}{\beta\Delta t^2}M)V$ includes its inertia. Used as the soft side's Robin condition, it makes the pair monolithic in port space, so no added-mass instability can arise. This is derived, not measured.
5. **Interface laws.** Contact resistance $q_n=h_c(T_1-T_2)$, Beavers–Joseph slip, friction, contact, electrode kinetics and ablation are zero-thickness dissipative two-ports.
6. **Moving seams.** Transpiration for small motion. For large motion, re-chart and satisfy the geometric conservation law discretely (Thomas & Lombard, *AIAA J.* 17, 1979).
7. **Multirate.** The slow side receives the time-integrated flux.

**I-3, the levels** (all measured in `arch_general_form_checks.py`, steel bar, $\gamma^2=\delta=0.0108$).

The scaled backward-Euler step is symmetric quasi-definite:
$$\begin{pmatrix}K_{uu}&-G\\-G^{\top}&-C/T_0\end{pmatrix},$$
with $\gamma^2\le\delta=\beta^2T_0/((\lambda+2\mu)\rho c)$.

| level | what the host does | result |
|---|---|---|
| one-way | heat, then mechanics through the source channel | error of order $\delta$: $9.8\times10^{-3}$ in $u$, $8.7\times10^{-3}$ in $\theta$ |
| staggered | block Gauss–Seidel between the experts | contraction 0.0087 to 0.0107 per sweep, against $\gamma^2=0.0108$ |
| monolithic bond | coupled reduced system over both trial spaces, $B=V_u^{\top}GV_\theta$ on the shared grid | nonsingular for any output; eigenvalue moduli in $[1,1.0050]$ against $\sqrt{1+\gamma^2}=1.0054$; within 1.003 of best against $1+\gamma=1.104$ |

**Cross-enrichment.** With exact experts, the $u$ error falls from 100% to $2.8\times10^{-3}$ to $8.5\times10^{-6}$ over rounds 0, 1 and 2.

**Other co-located cases.**
- **Strong reciprocal coupling.** The fixed-stress split (Kim, Tchelepi & Juanes, *CMAME* 200, 2011; a contraction by Mikelic & Wheeler, *Comput. Geosci.* 17, 2013) and the adiabatic split (Armero & Simo, *IJNME* 35, 1992) both move the coupling's diagonal part into one expert's coefficient channel. **[AI Inference]:** no new expert is needed if the envelopes cover the modified coefficients.
- **Dissipative couplings** (Joule heating) use Newton's method or a coefficient-plus-source fixed point. Power bookkeeping is exact: $\int\sigma\lvert\nabla\phi\rvert^2=c^{\top}\tilde\Lambda_\sigma c$.
- **Chemistry** uses Strang splitting (*SIAM J. Numer. Anal.* 5, 1968) with a pointwise chemistry expert and projection onto $\{EY=EY^0\}$. The splitting error has no slot yet (W116).
- **Coupling envelope.** Below the tolerance, one-way; $\gamma<1$, staggered or monolithic; $\gamma\ge1$, monolithic only.
- **Limit.** Thermal buckling is reported as an instability, not as an error.

**I-4 to I-8, notes.**
- **Radiation exchange** is symmetric in the emissive powers (view-factor reciprocity) and non-symmetric in $T$, so it is G2.
- **Homogenisation bounds.** $e\cdot\kappa^{*}e=\min_\chi\int_Y\kappa\lvert e+\nabla\chi\rvert^2$ means a periodic CO-D expert bounds $\kappa^{*}$ from above for any trial space. In 2-D the dual on $1/\kappa$ bounds it from below (Keller, *J. Math. Phys.* 5, 1964; Mendelson, *J. Appl. Phys.* 46, 1975).
- **FE$^2$** (Feyel & Chaboche, *CMAME* 183, 2000) can use the chart operator as its micro-solver. The classical periodic-cell solver is FFT-based (Moulinec & Suquet, *CMAME* 157, 1998).

**Seam types.**

| sides | ports | instrument |
|---|---|---|
| solid–solid | `MECH`, `THERM` | Gram; contact and resistance laws |
| solid–shell | `MECH` with resultants, `THERM` | kinematic constraint through the thickness |
| solid–fluid | `MECH`, `THERM` | causality, condensed impedance, added-mass check |
| solid–gas | `MECH`, `THERM`; `ADVEC` only if ablating | as above, plus characteristic ports |
| fluid–fluid | `MECH`, `ADVEC`$(h_0,Y_k)$ | I-1; passenger lists matched per face |
| solid–acoustic | `MECH` (pressure, normal velocity) | Gram |
| field–lumped node | constant or rigid modes only | I-5 |

**Co-located pairs, by reciprocity:**
- **Reciprocal:** heat + stress ($\delta\approx0.01$ in metals); pore pressure + stress (Biot ratio $\alpha^2M/K_{dr}$, can exceed 1); field + stress (piezoelectric $k^2$ up to about 0.5); thermoelectric (figure of merit $ZT$).
- **Not reciprocal:** Joule heating; flow + heat (Péclet, Rayleigh); flow + chemistry (Damkoehler); magnetic losses.

**The rocket, mapped onto the decomposition.**
- **$a$–$b$ is a co-located bond, not a seam:** the combustion chemistry and the chamber gas occupy the same pieces, so this edge is I-3 with splitting.
- **$b$–$c$ and $c$–$d$** are I-2.
- **Within $c$** is the thermoelastic bond.
- **The gas–gas edges** are I-1 with Riemann fluxes.
- **The plume radiating to the base** is I-4; the trajectory and gravity are I-5; recession and staging are I-6.

**[AI Inference]:** agent graphs should mark each edge as a seam or a bond. **A question for the rocket case study:** $c$–$d$ declares `ADVEC`$(h_0)$, but at an impermeable skin the port algebra's own rule (§3.2) gives none, unless the skin ablates or the edge also covers an inflow face.

**Port-vocabulary gaps:**
1. an electromagnetic field port (tangential $E$ and $H$, power $\int(E\times H)\cdot n$);
2. species diffusion without bulk flow ($\mu_k$ and $j_k$);
3. the co-located bond, already refused as a port (W32) and as a global field (W117);
4. a dissipative junction.

**[AI Inference]:** these are candidates for the port algebra's §10 amendment procedure. That page is another chat's; the decision is the owner's.

---

## 4. Speedups specific to the unit square

Every piece is the same square on the same grid, which gives three things:
- one tensor for all pieces;
- reference operators diagonalised by fast sine and cosine transforms;
- trivially dyadic hierarchies.

| ID | technique | status |
|---|---|---|
| S1 | batch every piece | measured (v1): 1.6×, 2.2×, 1.2× per piece at $n=64$, batch 16 against 1, for 0.49, 1.95 and 6.2 M parameters |
| S2 | fast reference solve (fast sine transform; fast diagonalisation, Lynch, Rice & Thomas, *Numer. Math.* 6, 1964) | **measured, below** |
| S3, S4 | sine, cosine and Chebyshev spectral layers (SNO, arXiv 2205.10573); factorised weights (TFNO, arXiv 2310.00120); sum factorisation (Orszag, *J. Comput. Phys.* 37, 1980) | cited |
| S5, S6 | MgNet trunk (arXiv 1901.10415, the default's structure); hierarchical-matrix and wavelet layers (MNN-H arXiv 1807.01883; BCR-Net arXiv 1810.08754; WNO arXiv 2205.02191) | cited |
| S7 | strip-cropped mode heads | **measured, below** |
| S8 | coarse trunk with fine side strips | **[AI Inference]:** untested |
| S9 | low-precision network, float64 host | **measured, below** |
| S10 | exponential integrators with the exact reference propagator (Cox & Matthews, *J. Comput. Phys.* 176, 2002; Hochbruck & Ostermann, *Acta Numer.* 19, 2010) | **[AI Inference]**: stable only if the deviation from the reference is not stiff |
| S11 | reuse of modes and Grams across loads, steps and identical pieces | derived |
| S12 | tensor Gauss or Clenshaw–Curtis quadrature (Trefethen, *SIAM Rev.* 50, 2008) | cited |

The square's symmetry group (group convolutions, Cohen & Welling, arXiv 1602.07576) buys exactness, data efficiency and one shared side head, but not speed.

**S2, measured** (`arch_family_checks.py`, bilinear finite elements). The pulled-back $\mathsf A=\kappa JG^{-1}$ has eigenvalues $\kappa K$ and $\kappa/K$, with $K=(1+\lvert\mu\rvert)/(1-\lvert\mu\rvert)$. Hence

$$\frac ba\le\frac{\kappa_{\max}}{\kappa_{\min}}K_{\max}^2,\qquad\text{CG iterations}\le\Big\lceil\tfrac12\sqrt{b/a}\,\ln(2/\mathrm{tol})\Big\rceil,$$

independent of the grid. With $\lvert\mu\rvert\le0.5$ and contrast 10, the bound is $b/a\le90$, giving at most 69 iterations to $10^{-6}$ and 113 to $10^{-10}$:

| grid | to $10^{-6}$ | to $10^{-10}$ | time / sparse direct |
|---|---|---|---|
| $32^2$ | 30 | 52 | 0.53 |
| $64^2$ | 31 | 55 | 0.31 |
| $128^2$ | 31 | 55 | 0.23 |
| $256^2$ | 30 | 54 | 0.09 |

The measured spectral ratio was 23.1 against the bound of 90. Timings vary between runs; the iteration counts do not.

S2 has five uses:
1. the principal part (already the default);
2. a fallback of guaranteed, grid-independent cost;
3. reference-solution generation;
4. **[AI Inference]:** operator-preconditioned outputs $V=V_0+L_0^{-1}N_\theta$, to register as a Stage-0 comparison;
5. constant-velocity advection–diffusion through $c=e^{b\cdot x/(2D)}w$. The factor's range $e^{\mathrm{Pe}/2}$ exhausts double precision near $\mathrm{Pe}\approx70$, so this applies only at moderate Péclet.

**S7, measured.** A mode of wavenumber $k$ decays like $e^{-k\pi y}$. An implicit step at Fourier number Fo raises the rate to $\sqrt{(k\pi)^2+1/\mathrm{Fo}}$, and distortion $K$ can lower it to $k\pi/K$. With 16 modes per side at $n=64$:

| piece | tolerance | area computed | saving | energy lost by a sharp crop |
|---|---|---|---|---|
| steady, undistorted | $10^{-3}$ | 0.438 | 2.28× | $5.5\,\mathrm{tol}^2$ |
| steady, undistorted | $10^{-6}$ | 0.648 | 1.54× | $3.0\,\mathrm{tol}^2$ |
| $\mathrm{Fo}=10^{-2}$ | $10^{-3}$ | 0.342 | 2.93× | $3.9\,\mathrm{tol}^2$ |
| $\mathrm{Fo}=10^{-3}$ | $10^{-3}$ | 0.178 | 5.63× | $0.9\,\mathrm{tol}^2$ |
| $\mathrm{Fo}=10^{-3}$ | $10^{-6}$ | 0.349 | 2.87× | $0.7\,\mathrm{tol}^2$ |
| steady, $K=3$ | $10^{-3}$ | 0.789 | 1.27× | $6.2\,\mathrm{tol}^2$ |
| steady, $K=3$ | $10^{-6}$ | 0.984 | 1.02× | $2.8\,\mathrm{tol}^2$ |

The predicted depths $\min(1,\ln(1/\mathrm{tol})/\mathrm{rate})$ matched the measured ones to within one cell. The saving applies to the mode head and Gram only, collapses under distortion, and is largest for transient pieces.

**S9, measured.** Theorem 5's identity makes rounding an interior error, so rounding enters the port matrix quadratically. This holds provided the host forms the Gram in float64. The principal part keeps the rounded quantity small: the correction is 18% of a mode in energy.

| network output | Gram error, correction rounded | whole mode rounded |
|---|---|---|
| bfloat16 | $3.2\times10^{-6}$ | $7.0\times10^{-5}$ |
| float16 | $5.1\times10^{-8}$ | $1.1\times10^{-6}$ |
| float32 | $7.9\times10^{-16}$ | $1.7\times10^{-14}$ |

For comparison, exact modes with the Gram formed in float32 give $1.5\times10^{-7}$, a linear error. The identity held to $7\times10^{-16}$, and the error stayed positive semidefinite to rounding.

**[AI Inference], as the precision policy:** a bfloat16 network with its correction emitted in bfloat16 or float16, and every Gram in float64. G2 and G3 pay for rounding linearly.

---

## 5. A second certificate from two-dimensional duality

**Intuition.** The value-based expert over-estimates a piece's energy (Theorem 5); a flux-based expert under-estimates it. The gap between them bounds both errors. In 2-D the flux problem is a diffusion problem on the reciprocal conductivity, so the same network computes it.

**Rigour.** Let $\mathcal E(g)=\min\{\int_\Omega\mathsf A\nabla v\cdot\nabla v:\ v|_{\partial\Omega}=g\}$. For every divergence-free $\tau$,
$$2\oint g\,\tau\cdot n\,ds-\int_\Omega\tau\cdot\mathsf A^{-1}\tau\,dx\le\mathcal E(g),$$
with equality at the true flux. This is the two-energy argument of Prager & Synge (*Quart. Appl. Math.* 5, 1947). On a simply connected 2-D piece, $\tau=W\nabla\psi$ with $W$ the rotation by $90^\circ$. Then $\tau\cdot n=-\partial_t\psi$ and $W^{\top}\mathsf A^{-1}W=\mathsf A/\det\mathsf A$, so
$$\mathcal E(g)=\max_\psi\Big[-2\oint g\,\partial_t\psi\,ds-\int_\Omega\nabla\psi\cdot\tfrac{\mathsf A}{\det\mathsf A}\nabla\psi\,dx\Big].$$
For an isotropic pull-back, $\mathsf A/\det\mathsf A=\kappa^{-1}JG^{-1}$: the same Beltrami coefficient with the conductivity inverted. The dual expert is therefore CO-D with $\log\kappa\mapsto-\log\kappa$.

The host forms $\tilde\Lambda_{\text{low}}=Y^{\top}(\tilde\Psi^{\top}M_*\tilde\Psi)^{-1}Y$, with $Y=Q_*^{\top}TQ$ the boundary pairing. For conforming discretisations and every output of both networks,
$$\tilde\Lambda_{\text{low}}\preceq\Lambda_{\text{low}}\preceq\mathcal E\preceq\Lambda\preceq\tilde\Lambda,$$
where the outer inequalities are Theorem 5 applied to each expert.

**Measured** (`arch_family_checks.py` check D; $n=64$ with an $n=128$ reference; $\lvert\mu\rvert\le0.4$; contrast 10; 64 primal ports):

| quantity | value |
|---|---|
| Loewner chain, smallest eigenvalue of each difference, relative | $\ge-6.4\times10^{-16}$ |
| worst-case relative gap: discretisation alone; all-node dual | 4.5%; 13.0% |
| worst-case relative gap with 63, 67 and 127 dual ports | 105%; 32%; 13.3% |
| smooth data, exact experts: 63 corner-aware dual ports; 127 | $7.1\times10^{-4}$; $6.4\times10^{-4}$ |
| smooth data, both experts perturbed by $\varepsilon=0.01$, 0.1, 0.3: true error / computable gap | $2.48\times10^{-3}/5.68\times10^{-3}$; $0.196/0.523$; $0.396/0.863$ |

**What it means.**
1. G0 diffusion pieces in 2-D get a guaranteed error bound for one more batched pass.
2. The dual needs continuous traces, which is a second argument (after Remark 4) for the conforming port basis C6.
3. Worst-case data need about twice the primal's port count; the smooth data a solve produces do not.
4. **[AI Inference]:** on the paper's two-point-flux finite volumes, which form a planar resistor network, planar duality should make the discrete bracket exact, so the gap would measure the networks alone. Unchecked.

---

## 6. The families

| family | trace | chart information beyond CO-D's | ports | class | distinctive device |
|---|---|---|---|---|---|
| **CO-D** linear diffusion | $H^1$ | — | `THERM`, `ELEC`, `MECH` by label | G0 | one network, ten PDEs; dual bracket |
| **CO-DN** nonlinear diffusion | $H^1$ | state, phase fraction | as CO-D | G1 | Kirchhoff and frozen-coefficient routes (Kacanov iteration; Heid & Wihler, *Math. Comp.* 89, 2020) before any new network |
| **CO-T** transport | $H^1$ plus advective flux | Piola velocity (divergence-free preserved), cell Péclet | `THERM` + `ADVEC` | G0′, G3 | adjoint modes from the same network by $\hat b\to-\hat b$ |
| **CO-E** elasticity | vector $H^1$ | stretch, rotation gradient, moduli, $r$ | `MECH` | G0 | local frame; a learned Craig–Bampton superelement (Craig & Bampton, *AIAA J.* 6, 1968) |
| **CO-H** finite strain, history | vector $H^1$ | deformation, internal variables | `MECH` | G1 locally, G2 with friction | a negative Hessian eigenvalue is reported as instability |
| **CO-P** plates, shells | $H^2$ or $H^1\times H^1$ | $a_{\alpha\beta}$, $b_{\alpha\beta}$, thickness | `MECH` (resultants) | G0 | shared with CO-S |
| **CO-S** Stokes | $H^2$ (2-D), $H(\mathrm{div})$ | — | `MECH` | G0 | the plate network relabelled; $(\psi,\omega)$ alternative (Glowinski & Pironneau, *SIAM Rev.* 21, 1979) |
| **CO-NS** Navier–Stokes, RANS | divergence-free vector | state, Reynolds number, eddy viscosity | `MECH` + `ADVEC` | G2 (inner G0′) | LSPG (Carlberg, Barone & Antil, *J. Comput. Phys.* 330, 2017) with fallback |
| **CO-G** compressible | characteristic | Piola fluxes, Courant number, mesh velocity | `MECH` + `THERM` + `ADVEC` | G3 | convex limiting (Guermond, Nazarov, Popov & Tomas, *SISC* 40, 2018; Zalesak 1979) |
| **CO-R** reacting | as CO-G | species | `ADVEC`$(h_0,Y_k)$ | split | pointwise chemistry, element projection |
| **CO-W** waves | impedance | complex coefficients, $k$ | `MECH` | G0′ | scattering matrix; value ports only below $kh=\pi\sqrt2$ |
| **CO-M** 3-D EM | $H(\mathrm{curl})$ | tensors | EM port (gap) | G0, G0′ | gauge at the host; optimised Schwarz (Dolean, Gander & Gerardo-Giorda, *SISC* 31, 2009) |

Card notes, each derived unless labelled:
- **CO-E.** The principal part is precomputed per Poisson ratio, because elasticity is not diagonalised by a sine transform. Fixed-interface vibration modes are the enrichment for dynamics.
- **CO-W.** The Helmholtz piece's scattering matrix is unitary when lossless, by $\oint\lvert\text{in}\rvert^2-\oint\lvert\text{out}\rvert^2=4k\operatorname{Im}\oint\partial_nu\,\bar u$, and a contraction when absorbing. The Gram is the plain transpose, so the error stays quadratic. High frequency needs sweeping or optimised Schwarz on the port graph (Gander & Zhang, *SIAM Rev.* 61, 2019; Ernst & Gander 2012).
- **CO-T.** At Péclet 20 and 200 the exact port matrix's symmetric part had relative smallest eigenvalue $-4.4\times10^{-2}$ and $-1.5\times10^{-1}$ (measured). Value plus diffusive flux is therefore not a passive pairing for transport, so seams must exchange the advective flux.
- **Couplers, which are host objects:** the seam junction, the bond, the interface law, the global field and the non-local operator.

---

## 7. Decisions for the owner

| # | decision | recommendation |
|---|---|---|
| F1 | build order | CO-D; nonlinear conduction through CO-D's coefficient channel; CO-E; the thermoelastic bond on CS-9; CO-T; the biharmonic network; CO-NS; CO-G with CO-R; CO-W; CO-M |
| F2 | port basis | sines plus corner hats (C6) as the default, tested against cosines at Stage 0 |
| F3 | dual bracket | CO-D's 2-D certificate |
| F4 | port vocabulary | the four gaps of §3, through the amendment procedure |
| F5 | moving seams | transpiration below an amplitude threshold on the envelope, re-charting above it |
| F6 | precision | bfloat16 network, float64 Grams |
| F7 | output parametrisation | register $V_0+L_0^{-1}N_\theta$ against the bubble factor |
| F8 | scope | confirm the exclusions of §2, group L |
| F9 | agent graphs | mark edges as seams or bonds; check the rocket's $c$–$d$ `ADVEC` |

**Still open from [[chart-operator-general-form]] §6:**
- whether the general form becomes the paper's core;
- the affine trial set against a learned manifold;
- whether the bond enters the vocabulary (F4);
- Stage 1's first nonlinear family (F1).

---

## 8. For the proofs chat

**The owner relays this.** Three corollaries of T25, all finite-dimensional, Tier A:

1. **The quadratic cost of a mismatched expert.** This is T25(ii) with $H$ the modes of another operator; nothing new beyond restating the hypothesis.
2. **Rounding as an interior perturbation.** Also T25(ii). The statement to formalise is that the identity needs only exact traces and an exactly formed Gram.
3. **The lower half of the dual bracket.** Given a positive semidefinite $M_*$, dual modes with exact traces and any pairing $Y$, the matrix $Y^{\top}(\tilde\Psi^{\top}M_*\tilde\Psi)^{-1}Y$ is Loewner-antitone in the dual Gram. It is therefore below its exact-mode value for every network output, by T25(ii) applied to the dual expert ($\tilde G_*-G_*=E_*^{\top}A_{*,II}E_*\succeq0$) and the fact that inversion reverses the Loewner order on positive definite matrices.

The continuum inequality $\Lambda_{\text{low}}\preceq\mathcal E\preceq\Lambda$ needs conforming discretisations and should be **cited** (Prager & Synge 1947), not formalised.

---

## 9. What this page does not do

1. No network was trained. Every learned output in the checks is an exact solve plus a perturbation.
2. The dual bracket was measured with bilinear finite elements, not with the paper's finite volumes. The exact-discrete version is an inference.
3. G2 and G3 rest on cited results; nothing here checks them.
4. The speed numbers are ratios on one container. The strip-cropping saving is an area count, not a timing.

---

## See Also

- [[chart-operator-general-form]] — the guarantee classes, seams and bonds this catalog organises
- [[chart-operator-architecture]] — version 1, the CO-D card
- [[chart-operator-design-decisions]] — the decisions F1–F9 would extend
- [[chart-operator-training-and-cost]] — the measured cost components
- [[coupling-cost-and-complexity]] — the coupling strategies' cost
- [[port-algebra-atlas-0.1]] — the five port types and the amendment procedure
- [[case-study-ladder-to-f1]] — CS-9 to CS-16
- [[case-study-rocket-ascent-2d-atlas-0.1]] — the agent graph of §3
- [[interface-transfer-theory]] — consistent and conservative transfer as one adjoint pair
- [[formal-proofs-plan]] — where §8's corollaries would go
