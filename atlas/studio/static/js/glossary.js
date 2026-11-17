// Every shorthand the page shows, explained on hover (the owner's rule: as few
// words as possible on screen, and every shorthand explains itself).  Sources:
// the family catalog's section 1.3 (guarantee classes), the Stage 0 rework (TL0,
// TL, rho), the integrated UI plan (sections 4 to 7).

export const GLOSSARY = {
  G0: 'Guarantee class G0: a linear, symmetric problem with an energy (heat, Darcy, electrostatics, elasticity). Whatever the network outputs, the coupled answer is the best in its space (Céa constant 1); machine-checked.',
  "G0′": 'Guarantee class G0′: linear but not symmetric (drift, Helmholtz). The error is controlled through adjoint modes; no positivity.',
  G1: 'Guarantee class G1: a convex energy (nonlinear diffusion, contact). An energy upper bound, at most √(L/m) times the best, and never worse than one classical step.',
  G2: 'Guarantee class G2: no energy (Navier–Stokes, RANS). The residual is exact and checked; no certified error bound; a classical fallback corrects defects.',
  G3: 'Guarantee class G3: explicit hyperbolic (waves, high-Péclet transport). Exact conservation through face fluxes, and positivity.',
  CO: 'A chart-operator expert exists for this equation.',
  'CO-D': 'The chart operator\'s diffusion family. Heat, Darcy flow and electrostatics share its operator.',
  TL0: 'TL0: the two-level expert with no training. Its coarse space is the floor a learned network must beat.',
  TL: 'TL: the two-level expert with a trained network: a coarse space learned per piece.',
  H2r: 'H2r: response reuse. A square of one material reuses one stored response instead of solving again.',
  'det J': 'det J: the chart\'s Jacobian determinant. Above zero at every cell corner means the chart does not fold.',
  Fo: 'Fourier number Fo = κΔt / (ρc h²): how far heat spreads in one time step, in cells.',
  Pe: 'Péclet number Pe = |u|L / κ: transport by the flow over transport by diffusion.',
  'A/A': 'A/A: an arm timed against itself. Its spread shows how much of a price ratio is noise on this machine.',
  rho: 'ρ = 1 − ε_TL / ε_TL0: the share of TL0\'s error the network removes. 1 is perfect; below 0 is worse than TL0.',
  n: 'n: cells per side of the grid.',
  n_c: 'n_c: the two-level expert\'s coarse grid, in cells per piece side.',
  THERM: 'THERM: a thermal port. Temperature is shared across the seam and heat flux is passed through it.',
  TFI: 'TFI: transfinite interpolation, a Coons patch of the piece\'s four sides. The cheapest chart; it can fold on strongly curved pieces.',
  Laplacian: 'Laplacian: harmonic coordinates, one sparse solve per piece (or the Winslow iteration). Never folds on a convex piece.',
  Conformal: 'Conformal: a map that keeps angles, so the piece\'s equation keeps its form.',
  DUAL: 'DUAL: a second certificate from two-dimensional duality: a guaranteed upper bound on the energy error.',
  kappa: 'κ: thermal conductivity, W/(m·K).',
  rhoc: 'ρc: volumetric heat capacity, J/(m³·K).',
  contrast: 'Contrast: the largest conductivity over the smallest. High contrast is what Stage 0b tests.',
  m: 'Units for every length on screen.',
};

export function tip(key) { return GLOSSARY[key] || null; }
