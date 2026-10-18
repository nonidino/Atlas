"""The chart operator, Stage 0: family CO-D, 2-D linear diffusion on charted pieces.

[[chart-operator-architecture]] (version 1) and the public document's version 2
specify a learned expert that answers one question on the unit square -- given
values on its four sides, what is inside and what leaves each side -- by returning
**constraint modes** with exact traces, so that the host can form each piece's port
matrix as an energy Gram that is positive semidefinite whatever the network returns
(T25, machine-checked).  This package is the first build of it.

The owner's decisions for Stage 0 (2026-10-18), on top of the confirmed design:

  D1  the code lives here, ``atlas/chartop/``;
  D2  the bars of `gates.py`, registered before the first training run;
  D3  the port basis is corner hats plus side sines (cosines are comparison C6);
  D4  a bfloat16 network on the GPU (float32 on the laptop's CPU), every Gram and
      every loss in float64;
  D5  comparison C7 is registered;
  D6  at most $15 of rented GPU time for the whole stage;
  D7  the piece solver is **bilinear finite elements** on the reference grid's
      vertices, not two-point-flux finite volumes, because a two-point flux cannot
      carry the off-diagonal entries that a shearing (Winslow) chart puts into the
      pulled-back tensor;
  D8  the main run trains on the label-free energy objective (T26); labels are
      computed for validation, the held-out test and comparison C2 only.

Modules: `fe` (the piece's discrete energy), `ports` (trace bases), `principal`
(the plain square's exact modes by fast sine transform), `chart` and `certificate`
(geometry, Proposition 3, the envelope), `piece` (one charted piece and its exact
classical solve), `host` (tier-0 coupling), `benchmarks` (the three held-out
layouts), `data`, `model`, `train`, `gates` (registered before training), `card`.
"""
