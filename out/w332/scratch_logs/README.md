# W332 — the experiments as they were first run (2026-09-23)

These are the logs of the experiments that found W332. They were run from scratch scripts, copied here unchanged. `scripts/w332_nozzle_symmetry.py` consolidates them into one reproducible script, with `--old-ghost` to recreate the failure.

Every run is corner case 0 (`underexpanded_max`), `coarsen = 4` and 5 ms macro steps. The engine (`a`, `b`, `e`) is marched alone, with both walls pinned at 288.15 K. Each value is `max|p - p_mirror| / max|p|` for one block, printed five times per macro step. Round-off is ~1e-15.

| log | build repo state | what varies |
|---|---|---|
| `sym_hllc.log`, `sym_inviscid.log`, `sym_rusanov.log` | seam fixes, **before W332** (diff `c26f4fe0d6ca89b4`) | all three blocks' flux and viscosity (Rusanov blew up at 0.26 ms) |
| `flux_hllc.log`, `flux_hlle.log`, `flux_rusanov.log` | before W332 | the nozzle's flux only, inviscid |
| `wall_iso.log`, `wall_adiab.log`, `wall_slip.log` | before W332 | the nozzle's walls only, inviscid, HLLC |
| `w332_inviscid_iso.log` | **with W332** | inviscid, HLLC, isothermal walls (4 of 5 macros; stopped once decisive) |
| `w332_viscous_hllc.log` | **with W332** | viscous, HLLC (3 macros; stopped to free the CPU for the re-march) |

`w332_before_numbers.py` computes the wall's mass leak both ways with the current code, by handing the inviscid flux the isothermal ghost again. It prints the ghost-to-cell density ratio of 125, the 123× mass flux through a wall face, and gate C6 on the coupler's engine blocks: 168, 193 and 119 before, ~1e-15 after.

The finding in one line: only the combination of HLLC and the isothermal ghost breaks symmetry. An adiabatic or slip wall, or an HLLE or Rusanov flux, stays at round-off. The isothermal ghost is 125× denser than the cell beside it, so the wall's Riemann problem is not a reflection.
