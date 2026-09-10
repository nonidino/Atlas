# CS-S2 — a Bounded Donor, Measured: the Patch Is Bounded and the Response Is Not

**Type:** Concept page — **measured result**, negative, off the ladder (folder: `Atlas 0.1/common/`)
**Status:** measured 2026-09-10, Tier 41. Not a rung on [[case-study-ladder-to-f1]]'s climb — the second off-ladder study, hence `S2`; CS-15 is rung 8 and CS-16 is reserved for `vehicle.py`. `scripts/cs_s2_elastic_patch.py`, `scripts/cs_s2_neubernet.py`, `scripts/cs_s2_bounded_donor.py`, `scripts/cs_s2_torsion_control.py`, `scripts/cs_s2_sign_jump.py`, `tests/test_tier41_bounded_donor.py`, `out/cs_s2/cs_s2.json`, `out/cs_s2/torsion_control.json`, `out/cs_s2/sign_jump.json`. Worklist rows **W95**, **W176**, **W179**, **W180**.
**Related:** [[substitution-campaign-checkpoint]] · [[epsilon-halo-measurement]] · [[expert-donor-survey]] · [[case-study-ladder-to-f1]] · [[gap-worklist]] · [[case-study-neural-interface-atlas-0.1]] · [[probed-dtn-coupling]] · [[composition-error-theory]] · [[master-error-bound]] · [[code-and-papers]]

---

# 0. The result, in one paragraph

[[substitution-campaign-checkpoint]] left one live route to a certified learned expert: a foundation model on a **constrained expert class**, starting from the three donors [[expert-donor-survey]] marks *bounded*. This page probes the one of the three with a continuum boundary port — **NeuberNet**, whose input *is* a boundary displacement — through [[epsilon-halo-measurement]]'s instrument unchanged, beside **linear elasticity solved classically on the same disc**, so that *global because of the physics* and *global because of the architecture* can be told apart. **It is not compact, and it is less compact than the physics it approximates.** Along its own ring the elastic physics is global too, as a Dirichlet-to-Neumann operator must be, but it decays into the patch; NeuberNet's response does not decay at all. **W95 binds for it as well**: there is no same-class reference pair. On the way the study found three port defects, a sign network that makes the published operator jump a finite step from zero load (**W180**), and a convention check that could not have seen the worst of the three defects (**W179**).

---

# 1. What was asked, and what would count

[[substitution-campaign-checkpoint]] §6 predicted, marked **[AI Inference]**, that all three bounded donors would read compact. [[epsilon-halo-measurement]] had measured the negative half of the class statement — two scOT checkpoints, no halo — and left the positive half unmeasured. The brief named three outcomes, all of them results:

| outcome | what it would mean |
|---|---|
| compact, with a same-class reference pair | the first route to a real `admit` |
| compact, no reference pair | W95 binds even for bounded donors |
| not compact | branch (a) narrows |

And it named the confound: an elliptic Dirichlet-to-Neumann map is dense along its own boundary **as physics**, so a donor that reads global might only be reading its equation. Hence the classical control on the same patch.

---

# 2. The donor

## 2.1 Provenance, and the licence that is not there

- **Code and weights**: `github.com/grossIt/neubernet` at commit `cce93244` (2025-07-09); paper Grossi, Beghini & Benedetti, *Communications Engineering* 2025.
- **Licence: none.** GitHub reports `license: null`, so the default is all rights reserved. The Zenodo dataset (doi:10.5281/zenodo.14880154) is CC BY 4.0, and the article itself is CC BY-NC-ND 4.0.
- **How it was used.** The user approved a local download on four conditions: `definitions.py` was read in full first; the weights load only through `torch.load(weights_only=True)` under an allowlist of that file's classes; the files live outside the repository in `~/.cache/neubernet`, with sizes and SHA-256 checked on every load and recorded in the artifact; and only derived numbers are committed. `out/cs_s2/operators.npz`, the dense operators, is gitignored.

## 2.2 What the expert is, read from its code

The input row is $116$ numbers: $108$ boundary values, $6$ parameters and a query point $(x,y)$. The boundary values are $u_x$, $u_y$ and ROTY — the rotation about the shaft axis, $u_\theta/\rho$ in magnitude — at $36$ sensors on the circle $\lvert p\rvert=5\,R_n$ around the notch tip, each scaled by $E/\sigma_y$. The parameters are $R/R_n$, the notch angle $\alpha$, $\beta$, $\sigma_y/E$, $E_t/E$ and $\nu$. The output is $14$ fields, among them the six stresses over $\sigma_y$ in the notch frame.

The network is an ensemble of $14$ NOMAD components — a branch MLP on the boundary data feeding a nonlinear manifold decoder at the query point — with $22{,}450{,}322$ parameters, the two auxiliary networks below included. **Production mode wraps it twice.** A sign network, SignNet, sets the signs of the tension and torsion parts of the input to a canonical orientation, and the output stresses are multiplied back by the same calls. A yield network, YieldNet, predicts the elastic von Mises stress; below yield, the input is rescaled up to yield and the output back down. That wrapped operator is what the authors publish, so it is what was measured.

Here: $\alpha=30^\circ$, $R/R_n=50$, $\nu=0.3$, $\sigma_y/E=3\times10^{-3}$, $E_t/E=10^{-2}$, $\beta=0$. Seven of the $36$ sensors fall inside the notch opening, whose half-angle at $5\,R_n$ is $35.7^\circ$, leaving $29$ in material, $0.87\,R_n$ apart. Base states come from a classical far field on a disc of radius $25\,R_n$: tension scaled to an elastic von Mises of $0.7\sigma_y$ (elastic branch) or $1.6\sigma_y$ (plastic), torsion, and tension plus $0.8\times$ torsion.

**One property of the checkpoint cost more than the physics.** It carries $2{,}198{,}346$ subnormal weights — $9.8\%$ of its parameters — and on an x86 CPU every product that touches one takes a microcode assist. One forward of the port's $120$ quadrature points took $61.5$ s unguarded and $0.28$ s with denormals flushed, with the output **bitwise** equal. The adapter now sets the flush, and the third run of the driver, made with it, reproduces the second bit for bit on every in-plane and classical number. With the flush, the whole driver runs in under five minutes on one CPU thread.

---

# 3. The port, and three defects

The seam is the material part of the ring. The trace is the sensor displacements read as a **hat interpolant** $\sum_j \lambda_j\phi_j$ along the arc. The flux is the traction weighted by the same hats — the Galerkin dual of the trace:

$$f_j \;=\; \frac{\displaystyle\int_\Gamma t\,\phi_j\,\rho\,\mathrm ds}{\displaystyle\int_\Gamma \phi_j\,\rho\,\mathrm ds}, \qquad t=\sigma\cdot n/\sigma_y$$

and the seam operator is built the way [[epsilon-halo-measurement]] builds it, one column per sensor:

$$S_{ij} \;=\; \frac{f_i(\lambda_0+a\,e_j)-f_i(\lambda_0)}{a}, \qquad a=10^{-2}$$

The classical control's reactions are exactly $\int t\,N_a\,\rho\,\mathrm ds$, so its flux is the same integral with no pointwise sampling. Both adapters integrate against the same hats; their hat masses agree to $2.0\times10^{-3}$.

## 3.1 A pointwise flux at the knots does not converge

The first run sampled the traction at the sensors themselves. A hat trace has a slope jump at every knot, and a Dirichlet-to-Neumann response to a slope jump is logarithmically singular there — so the classical operator moved $36$–$42\%$ between meshes of $4890$ and $19119$ nodes. The Galerkin flux moves $2.8$–$6.4\%$. The pointwise operators are still measured and reported beside it.

## 3.2 The notch opening has to be filled the way the pipeline fills it

The seven sensors inside the opening carry no boundary data, and the authors' import script fills them by linear interpolation across the gap. A trace clamped at the flanks made the two subjects disagree at the flank-end sensors only. The hats are now continued along the gap line, which is what NeuberNet is shown.

## 3.3 The hoop input was the wrong way round

The first adapter wrote ROTY $=+u_\theta/\rho$ and passed the convention check — at a **tension-only base**, where both subjects' hoop shears are identically zero, so the check carried no information about their sign. The second run's directional stage then read the torsion far field at cosine $-0.969$ and $-0.997$. Two readings fit: SignNet deciding at its own boundary, or a sign convention. `cs_s2_torsion_control.py` separated them at bases **with** torsion on the elastic branch, where SignNet's call is firm and production mode is exactly odd in torsion, stating each convention explicitly:

| base | convention | $s_{yz}$ cosine, error | $s_{xz}$ cosine, error | $t_\theta$ cosine, error | in-plane error |
|---|---|---|---|---|---|
| pure torsion | ROTY $=+u_\theta/\rho$ | $-0.9999$, $198\%$ | $-0.9989$, $202\%$ | $-0.996$, $197\%$ | (the physics has none) |
| pure torsion | ROTY $=-u_\theta/\rho$ | $+0.9999$, $2.3\%$ | $+0.9989$, $5.2\%$ | $+0.996$, $9.6\%$ | (the physics has none) |
| tension + torsion | ROTY $=+u_\theta/\rho$ | $-0.9997$, $200\%$ | $-0.9988$, $195\%$ | $-0.992$, $200\%$ | $5.0\%$ |
| tension + torsion | ROTY $=-u_\theta/\rho$ | $+0.9997$, $2.4\%$ | $+0.9988$, $6.7\%$ | $+0.992$, $13.0\%$ | $5.0\%$ |

The in-plane stresses are bitwise identical under the two conventions, and SignNet's torsion call is equal and opposite under them — **a convention, not a boundary**. The adapter now carries `HOOP_SIGN` $=-1$, and the driver checks torsion on every build. That is **W179**: a convention check vouches only for the components it excites. **[AI Inference]:** the corrected sign is what a right-handed rotation about the shaft axis gives in a frame where $z=x\times y$ — a positive rotation about $y$ moves a point at positive $x$ towards negative $z$, so $u_z=-\rho\,\text{ROTY}$. That is a derivation from the frame, not a reading of the authors' macro or of the solver's documentation, and neither was checked.

---

# 4. The controls

Every reading below stands on these, and each can fail.

| control | what it guards against | reading |
|---|---|---|
| classical patch test, notch-free disc | an FE or recovery error in the control | a uniform axial field returns to $2.9\times10^{-13}$; its Galerkin $x$-flux to $2.6\times10^{-13}$; the torsion flux converges, $7.5\times10^{-4}\to3.1\times10^{-4}$ as $h$ halves |
| Betti reciprocity of the classical operator | an operator that is not a DtN map | $3.3\times10^{-15}$ |
| mesh floor | an operator that is a property of the mesh | Galerkin $2.8$–$6.4\%$; pointwise $36$–$42\%$ (§3.1) |
| convention check, tension, elastic branch | a donor worn wrongly | stress $6.1\%$ overall; per component $s_{xx}$ $12.6\%$, $s_{yy}$ $3.6\%$, $s_{zz}$ $14.2\%$, $s_{xy}$ $5.5\%$; peak $s_{yy}$ $0.466$ against $0.450$; ring flux $t_x$ $25.2\%$ and $t_y$ $3.8\%$ |
| the same, with torsion, corrected hoop | the hoop sign (§3.3) | hoop shears $3.1\%$ and $3.7\%$; $t_\theta$ $9.6\%$ and $13.0\%$ |
| mirrored query points | a check with no teeth | $68\%$ in tension; $40$–$41\%$ on the hoop shears, $11$–$13\times$ the real comparison |
| **positive control**, split-step `WindowNS`, re-run in the same session | an instrument that reads everything as global | **exactly zero from $r=14$**, every row equal to Tier 40's to $10^{-9}$ relative; transverse $d^\star(10^{-2})=4$ again |
| amplitude ladder, donor, $u_y$ port | an operator that is the instrument's float32 floor | disagreement against $a=10^{-2}$: $23.5\%$ at $a=1$, $12.3\%$ at $10^{-1}$, $0.69\%$ at $10^{-3}$, $1.9\%$ at $10^{-4}$, $13.5\%$ at $10^{-5}$ — a V with $a=10^{-2}$ at its bottom, where the cancellation floor is $2\times10^{-4}$ of the operator. The classical column is flat at $4\times10^{-15}$ |

Two things the tension check also shows. Where the physics has no hoop shear, the donor emits some: $s_{yz}$ and $s_{xz}$ at about $1\%$ of $s_{yy}$, and a hoop flux of $6.4\times10^{-3}$ against an in-plane flux of $0.55$. And its field is good at the job it was built for — $s_{yy}$ to $3.6\%$, the peak to $3.5\%$ — while its $t_x$ flux on the ring is $25\%$ off.

---

# 5. Along the seam

$E(r)/\lVert S\rVert_2$, the fraction of the operator's norm left outside a band of half-width $r$ sensors, on the elastic base. The ring is $29$ sensors, so $r=28$ keeps everything:

| port | subject | $r=0$ | $r=1$ | $r=2$ | $r=4$ | $r=8$ | $r=14$ | $r^\star(10^{-1})$ |
|---|---|---|---|---|---|---|---|---|
| $u_x$ | **NeuberNet** | $0.961$ | $0.889$ | $0.811$ | $0.691$ | $0.598$ | $0.457$ | none short of the whole ring |
| $u_x$ | linear elasticity | $0.745$ | $0.414$ | $0.228$ | $0.138$ | $0.106$ | $0.037$ | $9$ |
| $u_y$ | **NeuberNet** | $0.942$ | $0.840$ | $0.760$ | $0.656$ | $0.516$ | $0.298$ | $24$ |
| $u_y$ | linear elasticity | $0.752$ | $0.410$ | $0.210$ | $0.113$ | $0.083$ | $0.046$ | $5$ |

**Read the classical rows first.** A poke at one sensor moves all $29$, and $r^\star(10^{-2})$ is the whole ring on the $u_x$ port: the physics is global along its boundary, as [[expert-donor-survey]]'s fifth finding says an elliptic family must be. What it also does is fall fast in the near field — $0.23$ by $r=2$. **The donor does not**: at every radius from $1$ to $14$, on every port and every base, it keeps more than $1.5\times$ what the physics keeps, and at $r=4$ five times as much.

The whole coupled $87\times87$ operator says the same in three more ways:

| | NeuberNet (three bases) | linear elasticity |
|---|---|---|
| $E(r)/\lVert S\rVert_2$ at $r=0,1,2,4$ | $0.96$, $0.89$, $0.82$, $0.71$ | $0.63$, $0.35$, $0.20$, $0.17$ |
| effective rank at $99\%$ of the energy | $6$, $7$, $6$ | $72$ |
| Betti reciprocity, $\lVert W-W^\top\rVert/\lVert W\rVert$ | $0.99$ | $3\times10^{-15}$ |
| $\lVert S_{\text{donor}}-S_{\text{classical}}\rVert/\lVert S_{\text{classical}}\rVert$ | $1.00$–$1.23$ | — |

Per port, the donor's operator is $1.00$–$1.10$ of the classical norm away from the classical one, with norms of $0.46$, $0.18$ and $0.007$ against $0.81$, $0.79$ and $0.50$ on the $u_x$, $u_y$ and $u_\theta$ ports, and a median diagonal ratio of $0.02$. That is [[substitution-campaign-checkpoint]]'s null-replacement signature — $\lVert\Delta\rVert/\lVert S_i\rVert\approx1$, a swap within reach of deleting the block — measured on Poseidon-T and arriving here on a second architecture family.

---

# 6. Across the seam

Poke one sensor, and profile the largest change in the stress field against depth into the disc, per unit poke — $\sigma/E$ per $u/R_n$ for both subjects. Pokes at $180^\circ$ ($u_x$, $u_y$) and at $90^\circ$ ($u_y$), on both bases.

| reading | NeuberNet (6 profiles) | linear elasticity, fine mesh | linear elasticity, coarse mesh |
|---|---|---|---|
| value at the face | $0.015$–$0.019$ | $1.13$–$1.63$ | $1.02$–$1.25$ |
| $d^\star$ to $10\%$ of the face value | **never, inside the disc** | $1.0$–$1.5\,R_n$ | $1.0$–$2.0\,R_n$ |
| value at $2\,R_n$ over value at $0.5\,R_n$ | $0.95$–$1.39$ | $0.12$–$0.17$ | $0.11$–$0.17$ |
| first depth below $10\%$ of the $0.5\,R_n$ value | never | $2.2$–$2.8\,R_n$ | $2.1$–$2.8\,R_n$ |

**The face value sits on a hat knot**, where the classical response is log-singular — it moves from $1.63$ to $1.02$ between meshes — so $d^\star$ against the face is read against a mesh-limited peak. The two rows referenced to $0.5\,R_n$ do not move with the mesh, and they carry the result: the physics falls by a factor of about six to nine between $0.5$ and $2\,R_n$ on either mesh, and the donor does not fall at all. For the $90^\circ$ poke it *rises* toward the notch tip, from $0.015$ at the face to $0.075$ at $4.9\,R_n$ deep. **[AI Inference]:** the notch's stress concentration amplifying a change that reaches the tip whole, which is what a response with no locality would do.

---

# 7. What the response is: its training manifold

$\mathrm d/\mathrm da$ of the ring flux at $\text{base}+a\,v$, $a=10^{-2}$, against the classical $S\,v$ — norm ratio and cosine:

| base | tension far field | torsion far field | smooth mode $k=3$ on $u_y$ | one sensor on $u_y$ | smooth mode on $u_\theta$ | one sensor on $u_\theta$ |
|---|---|---|---|---|---|---|
| tension, elastic | $1.04$, $0.997$ ($8.3\%$) | **straddles §8's jump**: $8.7$ one way, $0.18$ the other | $1.09$, $0.43$ | $0.063$, $-0.001$ | $0.007$, $0.83$ | $0.0004$, $0.000$ |
| tension, plastic | $1.04$, $0.997$ ($8.6\%$) | $0.99$, $0.996$ ($8.8\%$) | $1.13$, $0.46$ | $0.075$, $0.002$ | $0.036$, $0.77$ | $0.003$, $0.000$ |
| torsion, elastic | **straddles §8's jump**: $6.1$, $0.18$ | $0.97$, $0.996$ ($9.6\%$) | $1.11$, $-0.02$ | $0.18$, $0.002$ | $0.33$, $0.96$ | $0.019$, $0.020$ |
| tension + torsion, elastic | $1.03$, $0.997$ ($8.3\%$) | $1.01$, $0.993$ ($12.1\%$) | $1.53$, $0.31$ | $0.085$, $0.002$ | $0.11$, $0.89$ | $0.003$, $0.012$ |

**Faithful along the two directions it was trained on, where the load is signed; absent everywhere else.** One sensor's worth of boundary data gets $6$–$18\%$ of the physical flux, orthogonal to it; a smooth mode gets a response of about the right size pointing somewhere else. **[AI Inference]:** a branch network that has learned the tangent of a low-dimensional manifold of proportional far-field loads, and nothing transverse to it. The operator's effective rank of $6$–$7$ is the same fact seen from the spectrum.

---

# 8. Where the sign network changes its mind — W180

SignNet's call is not made **at** zero load. On the elastic tension base it holds $+1$ for torsion steps of either sign up to $10^{-3}$ and splits at $10^{-2}$; on the plastic base it holds through $10^{-2}$. Where the call changes, production mode switches between the canonical network and its sign-flipped image, and unless the canonical network's output there is zero, the two branches do not meet. `cs_s2_sign_jump.py` bisects the decision point along the far field and measures the forward across it:

| base | direction | decision point $a^\star$ | flux jump across $a^\star(1\pm10^{-3})$ | linear elasticity across the same interval | jump over the physical response to a unit step |
|---|---|---|---|---|---|
| tension, elastic | torsion | $8.06\times10^{-3}$ | $1.27\times10^{-2}$ | $2.4\times10^{-6}$ | $8.5\%$ |
| tension, plastic | torsion | $1.84\times10^{-2}$ | $1.34\times10^{-2}$ | $5.5\times10^{-6}$ | $9.0\%$ |
| torsion, elastic | tension | $2.52\times10^{-5}$ | $1.73\times10^{-3}$ | $1.4\times10^{-8}$ | $0.6\%$ |
| tension + torsion, elastic | both | never changes | — | — | — |

**The published forward is discontinuous**, three to five orders beyond what the physics moves across the same interval. A difference ladder $J(a)/2a=\lVert F(\text{base}+a v)-F(\text{base}-a v)\rVert/2a$ shows what that does to a probe:

- **straddling** the decision point it reads $4.4\times$ the physics at $a=10^{-2}$ on the elastic base and $31\times$ at $a=10^{-4}$ at zero tension;
- **inside** it, the forward is smooth and reads $0.18\times$ the physics on the elastic base, $0.99\times$ on the plastic one and $0.84\times$ at zero tension — so near zero torsion on the elastic branch the canonical network carries a fifth of the physical response;
- **with both loads signed** the call never changes, and the ladder is flat at $1.00$–$1.04\times$ the physics until the float32 floor.

And at zero load the donor emits flux the physics does not: a hoop flux of $6.4\times10^{-3}$ and $6.7\times10^{-3}$ at the two tension-only bases, where elasticity's is exactly zero.

**Why it matters beyond this donor.** Every instrument here that builds an operator — `probe.py`'s finite-difference columns, `support_reach`, `dense_seam_operator` — differences forward from its base at one step. At a base like these, the operator it returns depends on which side of an unreported decision point the step lands, and the amplitude ladder that found Poseidon-T's float32 floor separates nonlinearity from cancellation, not a jump from a slope. **[AI Inference]:** contact (W165) has a kink by physics and this is a jump by architecture, and a probe that is to certify either would have to difference both ways.

---

# 9. W95 on this donor

**Is there a same-class reference pair, so that $\tau$ and $\sigma$ can exist? No.** NeuberNet is not resolution-fixed — it is queried at arbitrary points, which is the property [[substitution-campaign-checkpoint]] §6 hoped would separate this branch from scOT — and it still has no referent of its own class, because it is defined on **one disc** around one notch tip, inside its training ranges. There is no larger or undecomposed domain to evaluate it on. The obstruction W95 names is the fixed domain, not the resolution.

Two referents of another class exist. For the elastic branch it is the classical patch on this page, and what it measures is the donor's **model error**, not a composition defect. For the plastic branch it is the authors' ANSYS analyses, whose inputs and fields are in the CC BY 4.0 Zenodo dataset; no solver here computes them, and they were not downloaded. And no compiling Atlas graph holds an axisymmetric elastic-plastic agent to put the donor in.

---

# 10. Verdict against the three outcomes

| outcome | measured |
|---|---|
| compact, with a reference pair | **no** — neither half |
| compact, no reference pair | **no** — W95 does bind, and the donor is not compact |
| **not compact** | **this.** Less compact than the elastic physics along the ring at every radius, on every port and base; no decay into the patch where the physics decays by a factor of about six to nine; effective rank $6$–$7$ of $87$ against $72$; faithful only along its training directions, and discontinuous where its sign network changes its call |

> **[AI Inference], and the one this study was set up to test.** The survey's *bounded* described the donor's **domain** — true of every sub-model — and what a halo rule or a certificate consumes is a bounded **response**. The first is not evidence for the second, and on the one donor measured they point opposite ways. Branch (a) narrows to donors whose locality is a property of the **operator** — MACE's $r_{\text{cut}}\times L$, DeepFlame's single point — and both are unmeasured.

**No rule was written.** `L2/R10` is untouched, and nothing on this page is a threshold.

---

# 11. The far field, recorded for W176

[[epsilon-halo-measurement]] §6 found the scOT checkpoints' far field in a handful of coherent modes, and W176 asked whether *banded plus low-rank* reproduces on a second architecture family. The off-band block at radius $r$, on the elastic base:

| port, $r$ | subject | share of the leading mode | share of the top four | effective rank at $99\%$ | leading singular value over $\lVert S\rVert_2$ |
|---|---|---|---|---|---|
| $u_x$, $8$ | NeuberNet | $0.59$ | $0.98$ | $6$ | $0.60$ |
| $u_x$, $8$ | linear elasticity | $0.46$ | $0.92$ | $14$ | $0.11$ |
| $u_x$, $20$ | NeuberNet | $0.48$ | $0.98$ | $6$ | $0.39$ |
| $u_x$, $20$ | linear elasticity | $0.48$ | $0.99$ | $5$ | $0.025$ |
| $u_y$, $8$ | NeuberNet | $0.37$ | $0.90$ | $12$ | $0.52$ |
| $u_y$, $8$ | linear elasticity | $0.45$ | $0.96$ | $12$ | $0.083$ |

The classical corners at $r=20$ come in equal singular-value pairs. **[AI Inference]:** the notch's mirror symmetry pairing the two corners of the operator.

**[AI Inference], and it does not decide W176.** The low-rank far field reproduces. But the donor is low-rank **everywhere** — its whole operator has rank $6$–$7$ — and it has no band to be banded around ($E(0)/\lVert S\rVert=0.96$). The classical control's far field is low-rank too, and it is banded. On this seam, then, a low-rank far field does not separate learned from classical, and *banded plus low-rank* describes the elastic physics better than the donor, which reads *low-rank and unbanded*. A $29$-sensor ring is also not a $128$-cell seam, so ranks do not compare across the two families in size. Recorded, not built.

---

# 12. What this does not claim

- **Not that NeuberNet is a poor model at its own job.** It predicts notch stress fields from proportional far-field loads, and on the elastic branch it does that to a few percent where the physics can check it. What is measured here is the same network used as a **seam operator**, which is not what it was built for.
- **Not that bounded donors in general are not compact.** One donor was measured; MACE and DeepFlame were not, and both bound their response rather than their domain.
- **Not the plastic branch's accuracy.** No referent for it is computed here.
- **Not that the corrected hoop sign is the authors' documented convention.** It is measured against the classical patch's axes, in tension and torsion both.
- **Not a rule.** The measurement was run to decide whether to write one, and it says not to.

---

## See Also

- [[substitution-campaign-checkpoint]] — §6 names branch (a) and predicts this donor compact; the note there records that it is not
- [[epsilon-halo-measurement]] — the instrument, unchanged, and the negative half this page adds a second family to
- [[expert-donor-survey]] — NeuberNet's entry, its licence line and findings 4 and 5, all corrected in place
- [[case-study-ladder-to-f1]] — §15, where this result lands on the schedule
- [[gap-worklist]] — Tier 41: W95, W176, W179, W180
- [[case-study-neural-interface-atlas-0.1]] — CS-S1, the other off-ladder study, which moved a global expert to where its globality is an asset
- [[probed-dtn-coupling]] — the Dirichlet-to-Neumann coupling a DtN-shaped donor would have to join
- [[code-and-papers]] — the NeuberNet row, corrected
