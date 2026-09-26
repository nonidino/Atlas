r"""W336 -- the rocket episode's residuals, and the error budget behind "is it right".

Tier 86 left a coupled episode that passes 17 of 17 gates (`out/w321`). Those
gates test symmetry, geometry, the coupler's bookkeeping and the trajectory's
integration. None of them measures whether the coupled run CONSERVES what it
exchanges, or how far its numbers are from the ones a finer run would give.
This script measures both, in three layers.

**1. Residuals of the run itself (stage `audit`).** The episode is re-run under
a ledger that captures, sub-step by sub-step, the face fluxes the solver itself
applied at every block boundary -- the arrays `_inviscid_residual` and
`_viscous_residual` difference, not a recomputation -- weighted exactly as SSP-RK2
weights its two stages. A finite-volume update changes a block's content only
through its boundary faces, so per block and per macro step

    content(after) - content(before) = sum of boundary inflows (+ reaction)

must close to round-off, and the boundary inflows then say where everything went:
through walls (which must pass no mass), through the injector (which should
deliver the declared mdot), through each plane seam (where the two neighbours'
fluxes need not agree, and their difference is mass, momentum or energy the
coupling created), and across the thermal seams (where the gas's heat loss and
the shell's heat gain need not agree either). The shell's own energy balance and
the thrust by two routes (exit-plane momentum, and the forces on the engine's
walls and injector) close the account. The run is compared with the recorded one
step by step, so the audit is about THAT run.

**2. Convergence (stages `dt`, `grid_engine`, `grid_external`).** The coupling
step at 10, 5 and 2.5 ms; the grid at coarsen 4, 2 and 1 for the engine alone
(walls held at 288.15 K) and the external flow alone. Observed orders and
Richardson estimates where the three values are monotone.

**3. Validation (stage `report`).** The engine against ideal quasi-1D nozzle
theory -- the same `normalize.exit_conditions` the build repo uses for its
reference scales -- and the heat-transfer model against its own grid.

**Predictions, registered 2026-09-24 before any arm ran:** see `PREDICTIONS`.

**Tier 88 (2026-09-26).** The four code defects this found (W337-W340) were
fixed in the build repo and the audit re-run on the fixed code; the Tier 87
records are kept in `out/w336_pre_w337`. `PREDICTIONS_T88` were registered
before the fixed run. ``--revert W337,W338,W339,W340`` (or ``all``) puts any
fix's old behaviour back on today's code (`revert`, checked bitwise against
build repo adc470b), and stage ``control_old`` runs the first audited steps
with all four reverted, to be read against the pre-fix audit.

    python scripts/w336_episode_residuals.py --stage audit
    python scripts/w336_episode_residuals.py --stage control_wall
    python scripts/w336_episode_residuals.py --stage dt --dt-macro 1e-2 --t-end 0.06
    python scripts/w336_episode_residuals.py --stage grid_engine --coarsen 2 --t-end 0.03
    python scripts/w336_episode_residuals.py --stage grid_external --coarsen 2 --t-end 0.03
    python scripts/w336_episode_residuals.py --stage report
    python scripts/w336_episode_residuals.py --stage control_old
    python scripts/w336_episode_residuals.py --stage grid_engine --revert W340 --json x.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from atlas.cases import rocket_experts as RE  # noqa: E402
from atlas.cases.thermal_seam import build_repo_identity  # noqa: E402

OUT = os.path.join(ROOT, "out", "w336")
SIDES = ("imin", "imax", "jmin", "jmax")
SIGMA_SB = 5.670374419e-8

#: **Registered 2026-09-24, before any arm ran.** Each is evaluated by
#: `evaluate()` from the stage records; the prose here and the code there are
#: tested against each other (tests/test_tier87_episode_residuals.py).
PREDICTIONS = {
    # gates: these must hold or the instrument (or the code) is wrong
    "A1": "every block's content changes by its boundary inflows to 1e-10 "
          "(mass, both momenta, energy), every macro step",
    "A2": "engine and airframe walls pass mass at under 1e-12 of mdot dt, "
          "face by face, every step",
    # A3 as first written said "to 1e-8 of the step's heat input". A two-step
    # smoke test at dt 2e-4 (not the audited run) read 4.6e-6, and the cause is
    # the instrument's floor, not a leak: `_cg` stops at rtol 1e-10 of a
    # right-hand side dominated by M T / dt, about 5e4 times the step's heat.
    # Re-based on the stored energy before the audit ran, and the step-relative
    # figure is still recorded beside it.
    "A3": "the shell's energy change equals its Robin heat input to 1e-9 of "
          "its stored energy, every step",
    # A4's first evaluator divided each rigid-state component by its own
    # recorded value, so the lateral position and velocity -- 1e-20-level
    # round-off around zero in a symmetric flight -- read O(1) "deviations"
    # while altitude, attitude, vertical velocity and mass matched BITWISE. The
    # prose names the rigid state; the evaluator now measures position,
    # velocity, attitude and mass each against its own magnitude. Changed while
    # the audit ran, after its first five steps were seen; both readings are in
    # the record (`record_dev` is the first).
    "A4": "the instrumented re-run reproduces out/w321's rigid states to 1e-9",
    # measured residuals
    "P1": "the injector delivers the declared mdot to within 2%, averaged over "
          "the last 10 steps",
    "P2": "the a|b and b|e seams create under 1% of mdot dt in mass per step, "
          "averaged over the last 10 steps",
    "P3": "the exit-plane and wall routes to the thrust agree to 2%, averaged "
          "over the last 10 steps",
    "P4": "the loads' cell-centred exit thrust and the solver's own face flux "
          "at the same instant agree to 2%",
    "P5": "the gas's heat loss at the engine walls and the shell's heat gain "
          "agree to 2% per side, averaged over the last 10 steps",
    "P6": "the vehicle's mass loss and the nozzle's mass outflow agree to 2%",
    "P7": "the outer face's radiation, applied against the adjacent gas "
          "temperature instead of the ambient, moves the outer heat input by "
          "under 5%",
    # convergence and validation
    "P8": "seam mass creation scales at least linearly with the coupling step: "
          "10 ms over 5 ms and 5 ms over 2.5 ms both at least 1.5",
    "P9": "v_y at 60 ms moves by under 0.01% between the 5 ms and 2.5 ms "
          "coupling steps",
    "P10": "the engine's throat mdot is within 3% of the ideal choked flow at "
           "coarsen 1",
    "P11": "the engine's face thrust at coarsen 4 is within 5% of coarsen 1",
    "P12": "the engine walls' heat loss at coarsen 1 is more than twice coarsen "
           "4's -- the conduction-limited h is a property of the grid",
    "P13": "the airframe drag at coarsen 4 is within 10% of coarsen 1",
}

#: **Tier 88 -- registered 2026-09-26 11:05 EDT, after the four fixes (W337-W340)
#: were in the build repo and BEFORE the fixed episode was marched or audited.**
#: What had been seen: the build repo's unit tests, and one 30 ms engine-alone
#: smoke run at coarsen 4 (scratch, not a record) whose first four steps read
#: the injector at 1.000000 and every block closing to 1e-13, the engine still
#: in its start-up transient (throat 1.24 of the declared flow at 20 ms). Two
#: 0.1 s engine-alone explorations (fixed, and fixed with W340 reverted) were
#: started before this was written and had not reported a step past 20 ms.
#: Q1-Q9 are the four fixes' definitions of done, made measurable; Q10-Q13
#: are what the fixes should do to the episode; Q14 is the control. The
#: 2026-09-24 predictions above are evaluated on the fixed run as well.
#:
#: **Written AFTER registration, 2026-09-26, and changing no prose or evaluator
#: below.** (1) W340's implementation changed twice after this was registered.
#: At 11:20, before any march, a frozen-instant test (scripts/w340_throat_diag.py)
#: showed the briefed one-layer overlap average leaving the throat's floor
#: where it was (-3.09% against interpolation's -3.04%); the handover became two
#: ghost layers deep (-> 4e-5). At 12:20 the first fixed audit's coupling-step
#: study, read part-way, showed e|f gone from a lag (1e-4) to a 2% floor: two
#: layers are right only where each side's ghost is the other's cells, and e's
#: exit is an outflow boundary. The one-way seams went back to one layer and
#: every coupled stage was re-run (run_t88b.sh; the first build's records are
#: out/w336_t88_twolayer). Q7-Q9 are read on the final build. (2) Q14 says
#: "rigid state bitwise", and on the rented box the lateral components x and
#: v_x -- round-off around zero, 1e-20 and 1e-17 -- differ from the pre-fix
#: audit's box in the last bits, while y, theta, v_y, omega and m are bitwise
#: and every reading agrees to 1.1e-13. It repeats A4's first mistake, above.
#: It is left as registered and reads FAILED; the flight-component reading is
#: recorded beside it by tests/test_tier88_episode_fixes.py.
PREDICTIONS_T88 = {
    # W337
    "Q1": "the injector delivers the declared mdot to 1e-6 at every step of the "
          "audited run",
    "Q2": "the injector delivers the declared mdot to 1e-6 at every step of every "
          "engine-alone grid run that exists (coarsen 4, 2, 1)",
    # W339
    "Q3": "the thermal-seam diagnostic reads no wall work: |work| under 1e-9 of "
          "the conduction, on both sides",
    "Q4": "the engine's thermal seam closes to 0.5% per side (the shell's gain "
          "against the gas's loss), averaged over the last 10 audited steps",
    "Q5": "what remains of the engine's thermal seam is a lag: its mean |rel| over "
          "30-60 ms, both sides, shrinks with the coupling step, 10/5 ms and "
          "5/2.5 ms both at least 1.5",
    # W338
    "Q6": "the radiation booked as applied equals the radiation against the "
          "ambient to 1e-9 of the latter, every audited step, both panels",
    # W340
    "Q7": "the throat seam b|e's mean |mass created| over 30-60 ms shrinks with "
          "the coupling step, 10/5 ms and 5/2.5 ms both at least 1.5",
    "Q8": "the throat seam's mean |mass created| over 30-60 ms is below a|b's at "
          "the episode's 5 ms coupling step",
    "Q9": "the throat seam's mean |mass created| over the last 10 audited steps is "
          "under 0.3% of mdot dt, a tenth of the old floor",
    # what the fixes should do to the episode
    "Q10": "the chamber's volume-mean pressure is within 3% of the declared 8 MPa, "
           "averaged over the last 10 audited steps",
    "Q11": "given its inflow, the nozzle's thrust stays within 2% of ideal planar "
           "theory",
    "Q12": "the vehicle's mass loss and the nozzle's mass outflow agree to 2%, "
           "averaged over the last 10 audited steps",
    "Q13": "the exit-plane and wall routes to the thrust agree to 0.5%, averaged "
           "over the last 10 audited steps (they differed by 1.7%, the throat's "
           "lost momentum)",
    # the control
    "Q14": "with all four fixes reverted on today's code, the first audited steps "
           "reproduce the pre-fix audit: rigid state bitwise, and the injector, "
           "throat seam, engine wall heat and radiation slip to 1e-12",
}


# ---------------------------------------------------------------------------
# the build repo, as the episode runner loads it
# ---------------------------------------------------------------------------


def _mods():
    RE.load_solvers()
    imp = importlib.import_module
    return dict(gen=imp("atlas_build_solvers.data.generate"),
                sweep=imp("atlas_build_solvers.data.sweep"),
                config=imp("atlas_build_solvers.config"),
                normalize=imp("atlas_build_solvers.data.normalize"),
                C2=imp("atlas_build_solvers.solvers.compressible2d"),
                TS=imp("atlas_build_solvers.solvers.thermostruct2d"),
                thermo=imp("atlas_build_solvers.solvers.thermo"))


# ---------------------------------------------------------------------------
# Tier 88: the four fixes, and the old behaviour of each kept as a control
# ---------------------------------------------------------------------------

#: The build-repo fixes of 2026-09-26. `revert` puts any of them back to the
#: code the Tier 86-87 records were made with (build repo adc470b), in this
#: process only, so a stage can show that the old behaviour on today's code
#: reads what the old records read.
FIXES = ("W337", "W338", "W339", "W340")


def _old_inlet_face_flux(self, Fi, Ue):
    """Before W337: the injector face kept the Riemann flux between the inlet
    ghost and the first cell."""
    return None


def _old_outer_robin(self, T, h_out, T_gas_out, radiate, T_inf):
    """Before W338: h_rad added to h_out, the sum against the adjacent gas."""
    a, b, _ = self._face("outer")
    h_o = np.broadcast_to(np.asarray(h_out, dtype=float), a.shape)
    T_g = np.broadcast_to(np.asarray(T_gas_out, dtype=float), a.shape)
    if not radiate:
        return h_o, T_g, np.zeros(a.shape)
    Tw = 0.5 * (T[a] + T[b])
    h_rad = self.mat.emissivity * SIGMA_SB * (Tw ** 2 + T_inf ** 2) * (Tw + T_inf)
    return h_o + h_rad, T_g, h_rad


def _old_wall_energy(NG):
    def old(self, Fi, Fj, T, Qz, Qy, k):
        """Before W339 (W301 alone): the face-averaged conduction replaced by the
        one-sided wall term on isothermal faces, the rest of the average -- the
        work -- kept; adiabatic no-slip faces untouched."""
        blk = self.block
        nz, ny = blk.shape
        for side in ("imin", "imax", "jmin", "jmax"):
            bc = self.bcs.get(side)
            if bc is None or bc.kind != "wall_noslip" or "T_wall" not in bc.params:
                continue
            if side in ("jmin", "jmax"):
                jc, jg, f = (NG, NG - 1, 0) if side == "jmin" else (ny + NG - 1, ny + NG, -1)
                sl_c = (slice(NG, nz + NG), jc)
                sl_g = (slice(NG, nz + NG), jg)
                n, a, vol = blk.n_j[:, f], blk.a_j[:, f], blk.vol[:, 0 if f == 0 else -1]
                target = (slice(None), f)
            else:
                ic, ig, f = (NG, NG - 1, 0) if side == "imin" else (nz + NG - 1, nz + NG, -1)
                sl_c = (ic, slice(NG, ny + NG))
                sl_g = (ig, slice(NG, ny + NG))
                n, a, vol = blk.n_i[f], blk.a_i[f], blk.vol[0 if f == 0 else -1]
                target = (f, slice(None))
            avg = 0.5 * ((Qz[sl_c] + Qz[sl_g]) * n[..., 0] + (Qy[sl_c] + Qy[sl_g]) * n[..., 1]) * a
            T_i = T[sl_c]
            T_w = np.asarray(bc.params["T_wall"], dtype=float)
            T_w = np.broadcast_to(T_w, T_i.shape) if T_w.ndim == 0 else T_w.reshape(T_i.shape)
            dn = 0.5 * vol / np.maximum(a, 1e-30)
            grad = (T_i - T_w) / dn if f == 0 else (T_w - T_i) / dn
            wall = k[sl_c] * grad * a
            fix = wall - avg
            if bc.params.get("isothermal_mask") is not None:
                fix = np.where(np.asarray(bc.params["isothermal_mask"], dtype=bool), fix, 0.0)
            (Fj if side in ("jmin", "jmax") else Fi)[target + (3,)] += fix
    return old


def _old_wiring(gen):
    """Before W340: every plane seam but e-f's core handed over by `_remap`,
    point interpolation at the cell centroids. Build repo adc470b's
    `_wire_engine`, `_wire_external` and `_d_outlet`, verbatim but for the
    module prefix."""
    BC, NG, thermo, atmosphere = gen.BC, gen.NG, gen.thermo, gen.atmosphere

    def d_outlet(self, y, dst):
        out = np.empty((y.size, 4))
        for k, blk in enumerate(self.blocks["d"]):
            sel = (y > 0) if blk.nodes[..., 1].mean() > 0 else (y <= 0)
            if not sel.any():
                continue
            yc = blk.centroid[-1, :, 1]
            order = np.argsort(yc)
            st = gen._remap(self.U["d"][k][-1][order], yc[order], y[sel])
            out[sel] = gen._as_gas(st, self.air_cfg, dst)
        return out

    def wire_engine(self):
        geo = self.geo
        Ua, Ub, Ue = self.U["a"][0], self.U["b"][0], self.U["e"][0]
        ya = self.blocks["a"][0].centroid[-1, :, 1]
        yb0 = self.blocks["b"][0].centroid[0, :, 1]
        yb1 = self.blocks["b"][0].centroid[-1, :, 1]
        ye0 = self.blocks["e"][0].centroid[0, :, 1]
        mdot_flux = self.scales.mdot / (2.0 * geo.chamber_halfheight)
        b_to_a = gen._remap(gen._face_state(Ub, "imin")[0], yb0, ya)
        b_to_a = np.concatenate([b_to_a, gen._face_state(Ua, "imax")[0, :, 4:]], axis=-1)

        def walls(agent):
            return {s: BC("wall_noslip", {"T_wall": self._wall_T(agent, 0, s)})
                    for s in ("jmin", "jmax")}

        self.sol["a"][0].bcs = {
            "imin": BC("inlet_massflow", {"mdot": mdot_flux, "T": gen.T_INJECT}),
            "imax": BC("prescribed", {"state": b_to_a[None]}),
            **walls("a"),
        }
        self.sol["b"][0].bcs = {
            "imin": BC("prescribed", {"state": gen._remap(gen._face_state(Ua, "imax")[0, :, :4], ya, yb0)[None]}),
            "imax": BC("prescribed", {"state": gen._remap(gen._face_state(Ue, "imin")[0], ye0, yb1)[None]}),
            **walls("b"),
        }
        _, p_inf, _, _ = self._atm()
        self.sol["e"][0].bcs = {
            "imin": BC("prescribed", {"state": gen._remap(gen._face_state(Ub, "imax")[0], yb1, ye0)[None]}),
            "imax": BC("outflow", {"p_inf": p_inf}),
            **walls("e"),
        }

    def wire_external(self):
        _, p_inf, _, _ = self._atm()
        free = self._freestream()
        for k, b in enumerate(self.blocks["d"]):
            wall_side = self._d_wall_side(k)
            far_side = "jmax" if wall_side == "jmin" else "jmin"
            self.sol["d"][k].bcs = {
                "imin": BC("freestream", {"prim": free}),
                "imax": BC("outflow", {"p_inf": p_inf}),
                wall_side: self._d_wall_bc(k, wall_side),
                far_side: BC("freestream", {"prim": free}),
            }
        fb, eb = self.blocks["f"][0], self.blocks["e"][0]
        yf = fb.centroid[0, :, 1]
        yf_edges = fb.nodes[0, :, 1]
        ye_edges = eb.nodes[-1, :, 1]
        core = (yf_edges[:-1] >= ye_edges[0] - 1e-12) & (yf_edges[1:] <= ye_edges[-1] + 1e-12)
        inlet = np.empty((yf.size, 4))
        if core.any():
            kk = np.flatnonzero(core)
            edges = np.concatenate([yf_edges[kk], yf_edges[kk[-1] + 1:kk[-1] + 2]])
            avg, cov = gen._cell_average(gen._face_state(self.U["e"][0], "imax")[0], ye_edges, edges)
            if not np.allclose(cov, 1.0):
                raise AssertionError("f's core inlet cells are not covered by e's exit")
            inlet[core] = avg
        if (~core).any():
            inlet[~core] = d_outlet(self, yf[~core], self.gas_cfg)
        Ug, Uf = self.U["g"][0], self.U["f"][0]
        j0, j1 = self.sol["g"][0]._hole
        zg = self.blocks["g"][0].centroid[:, 0, 0]
        zf = fb.centroid[:, 0, 0]
        g_lo = gen._as_gas(gen._remap(Ug[:, j0 - 1], zg, zf), self.air_cfg, self.gas_cfg)
        g_hi = gen._as_gas(gen._remap(Ug[:, j1 + 1], zg, zf), self.air_cfg, self.gas_cfg)
        self.sol["f"][0].bcs = {
            "imin": BC("prescribed", {"state": inlet[None]}),
            "imax": BC("outflow", {"p_inf": p_inf}),
            "jmin": BC("prescribed", {"state": g_lo[:, None]}),
            "jmax": BC("prescribed", {"state": g_hi[:, None]}),
        }
        gb = self.blocks["g"][0]
        yg = gb.centroid[0, :, 1]
        g_in = np.broadcast_to(thermo.prim_to_cons(free, atmosphere.GAMMA_AIR), (yg.size, 4)).copy()
        live = ~gb.blanked[0]
        g_in[live] = d_outlet(self, yg[live], self.air_cfg)
        self.sol["g"][0].bcs = {
            "imin": BC("prescribed", {"state": g_in[None]}),
            "imax": BC("outflow", {"p_inf": p_inf}),
            "jmin": BC("freestream", {"prim": free}),
            "jmax": BC("freestream", {"prim": free}),
        }
        lo = np.stack([gen._remap(Uf[:, d], zf, zg) for d in range(NG)], axis=1)
        hi = np.stack([gen._remap(Uf[:, -1 - d], zf, zg) for d in range(NG)], axis=1)
        self.sol["g"][0].hole_state = (gen._as_gas(lo, self.gas_cfg, self.air_cfg),
                                       gen._as_gas(hi, self.gas_cfg, self.air_cfg))

    return wire_engine, wire_external


def revert(M, which):
    """Put the named fixes' old behaviour back on today's build repo (this
    process only), and say which. Call it BEFORE a `Ledger` is installed: the
    ledger wraps whatever the class holds at that moment."""
    if isinstance(which, str):
        which = which.split(",")
    which = [w.strip() for w in (which or ()) if w and w.strip()]
    if "all" in which:
        which = list(FIXES)
    unknown = [w for w in which if w not in FIXES]
    if unknown:
        raise ValueError("unknown fix %s; the fixes are %s" % (unknown, FIXES))
    C2, TS, gen = M["C2"].Compressible2D, M["TS"].ThermoStruct2D, M["gen"]
    for owner, names in ((C2, ("_inlet_face_flux", "_isothermal_wall_conduction")),
                         (TS, ("_outer_robin",)),
                         (gen.CoupledEpisode, ("_wire_engine", "_wire_external"))):
        for nm in names:
            _SAVED.setdefault((owner, nm), owner.__dict__.get(nm))
    if "W337" in which:
        assert hasattr(C2, "_inlet_face_flux"), "this build repo predates W337"
        C2._inlet_face_flux = _old_inlet_face_flux
    if "W338" in which:
        assert hasattr(TS, "_outer_robin"), "this build repo predates W338"
        TS._outer_robin = _old_outer_robin
    if "W339" in which:
        C2._isothermal_wall_conduction = _old_wall_energy(M["C2"].NG)
    if "W340" in which:
        assert hasattr(gen, "_hand_over"), "this build repo predates W340"
        gen.CoupledEpisode._wire_engine, gen.CoupledEpisode._wire_external = _old_wiring(gen)
    return list(which)


_SAVED = {}


def restore_fixes():
    """Undo every `revert` in this process (tests; a stage never needs it)."""
    for (owner, nm), fn in list(_SAVED.items()):
        if fn is not None:
            setattr(owner, nm, fn)
    _SAVED.clear()


# ---------------------------------------------------------------------------
# the ledger: the solver's own boundary fluxes, time-integrated
# ---------------------------------------------------------------------------


class Ledger:
    """Captures, per `Compressible2D` and per macro step, the INFLOW through each
    boundary side, face by face: the flux arrays the solver's residual
    differences, weighted by SSP-RK2's stage weights (both 0.5 dt, since
    U2 = U + dt (r(U) + r(U1)) / 2). Wrappers call the originals and only read
    their outputs, so the numerics are untouched -- which the audit also checks
    against the recorded run.

    Sign conventions, from the residual's own form: the inviscid residual is
    -div F, so a min face's F enters and a max face's leaves; the viscous one is
    +div F_v, the other way round. A holed block (agent `g`) also has the two
    faces between its live cells and the hole as boundaries."""

    def __init__(self, M):
        self.M = M
        self.C2, self.TS = M["C2"], M["TS"]
        self.active = None
        self.acc = {}
        self.shell = {}

    # -- accumulators --------------------------------------------------------
    def _new(self, s):
        nz, ny = s.block.shape
        nv = s.nv

        def sides():
            d = {"imin": np.zeros((ny, nv)), "imax": np.zeros((ny, nv)),
                 "jmin": np.zeros((nz, nv)), "jmax": np.zeros((nz, nv))}
            if s._hole is not None:
                d["hole_lo"] = np.zeros((nz, nv))
                d["hole_hi"] = np.zeros((nz, nv))
            return d
        return {"inv": sides(), "visc": sides(), "react": np.zeros(nv), "substeps": 0}

    def acc_for(self, s):
        a = self.acc.get(id(s))
        if a is None:
            a = self.acc[id(s)] = self._new(s)
        return a

    def reset(self):
        self.acc = {}
        self.shell = {}

    # -- capture -------------------------------------------------------------
    def _inviscid_faces(self, s, Fi, Fj, w):
        """The face fluxes `_inviscid_face_fluxes` returns (times the face
        length) -- the arrays `_inviscid_residual` differences, the injector's
        prescribed flux (W337) included."""
        nz, ny = s.block.shape
        a = self.acc_for(s)["inv"]
        a["imin"] += w * Fi[0]
        a["imax"] -= w * Fi[nz]
        a["jmin"] += w * Fj[:, 0]
        a["jmax"] -= w * Fj[:, ny]
        if s._hole is not None:
            j0, j1 = s._hole
            a["hole_lo"] -= w * Fj[:, j0]
            a["hole_hi"] += w * Fj[:, j1 + 1]

    def _inviscid(self, s, raw, w):
        blk = s.block
        nz, ny = blk.shape
        a = self.acc_for(s)["inv"]
        if raw.shape[0] == nz + 1 and raw.shape[1] == ny:           # i faces
            F = raw * blk.a_i[..., None]
            a["imin"] += w * F[0]
            a["imax"] -= w * F[nz]
        elif raw.shape[0] == nz and raw.shape[1] == ny + 1:         # j faces
            F = raw * blk.a_j[..., None]
            a["jmin"] += w * F[:, 0]
            a["jmax"] -= w * F[:, ny]
            if s._hole is not None:
                j0, j1 = s._hole
                a["hole_lo"] -= w * F[:, j0]
                a["hole_hi"] += w * F[:, j1 + 1]
        else:
            raise AssertionError("flux array %s fits neither face set of %s"
                                 % (raw.shape, blk.shape))

    def _viscous(self, s, Fi, Fj, w):
        nz, ny = s.block.shape
        a = self.acc_for(s)["visc"]
        a["imin"] -= w * Fi[0]
        a["imax"] += w * Fi[nz]
        a["jmin"] -= w * Fj[:, 0]
        a["jmax"] += w * Fj[:, ny]
        if s._hole is not None:
            j0, j1 = s._hole
            a["hole_lo"] += w * Fj[:, j0]
            a["hole_hi"] -= w * Fj[:, j1 + 1]

    def _shell(self, ts, T, Tn, dt, h_in, T_gas_in, h_out, T_gas_out, radiate, T_inf):
        """The shell's Robin heat, exactly as `step_thermal` assembled it: per
        face segment h L (target - mean of the segment's two nodes at t+dt).

        The outer face's (h, target) come from the solver's own
        `_outer_robin` where it has one (W338, 2026-09-26), so what is booked
        is what was applied, and a monkeypatch that restores the old target is
        booked as the old target. Before W338 the radiation's target was the
        adjacent gas's temperature, which is what the fallback books."""
        a_i, b_i, L_i = ts._face("inner")
        a_o, b_o, L_o = ts._face("outer")
        h_in = np.broadcast_to(np.asarray(h_in, dtype=float), L_i.shape)
        Tg_in = np.broadcast_to(np.asarray(T_gas_in, dtype=float), L_i.shape)
        h_o = np.broadcast_to(np.asarray(h_out, dtype=float), L_o.shape)
        Tg_o = np.broadcast_to(np.asarray(T_gas_out, dtype=float), L_o.shape)
        Tb_i = 0.5 * (Tn[a_i] + Tn[b_i])
        Tb_o = 0.5 * (Tn[a_o] + Tn[b_o])
        if hasattr(ts, "_outer_robin"):
            h_tot, target, h_rad = ts._outer_robin(T, h_out, T_gas_out, radiate, T_inf)
            q_outer = dt * h_tot * L_o * (target - Tb_o)
            q_conv = dt * h_o * L_o * (Tg_o - Tb_o)
            q_rad = q_outer - q_conv                            # as applied
        else:
            if radiate:
                Tw = 0.5 * (T[a_o] + T[b_o])
                h_rad = ts.mat.emissivity * SIGMA_SB * (Tw ** 2 + T_inf ** 2) * (Tw + T_inf)
            else:
                h_rad = np.zeros_like(L_o)
            q_conv = dt * h_o * L_o * (Tg_o - Tb_o)
            q_rad = dt * h_rad * L_o * (Tg_o - Tb_o)            # as applied
        nodes = ts.mesh.nodes.reshape(-1, 2)
        rec = dict(
            dE=float(np.sum(ts.M_th.dot(Tn - T))),
            E_stored=float(np.sum(ts.M_th.dot(Tn))),
            q_in=dt * h_in * L_i * (Tg_in - Tb_i),
            q_conv=q_conv,
            q_rad=q_rad,
            q_rad_sky=dt * h_rad * L_o * (T_inf - Tb_o),        # against the ambient
            z_in=0.5 * (nodes[a_i, 0] + nodes[b_i, 0]),
            T_inner=Tn[a_i], T_outer=Tn[a_o])
        self.shell.setdefault(id(ts), []).append(rec)

    # -- install -------------------------------------------------------------
    def install(self):
        led, C2, TS = self, self.C2, self.TS
        cls = C2.Compressible2D
        if getattr(cls, "_w336_installed", False):
            return
        o_step, o_iwc, o_react = cls.step, cls._isothermal_wall_conduction, cls.react
        o_iff = getattr(cls, "_inviscid_face_fluxes", None)
        self._originals = dict(step=o_step, iwc=o_iwc, react=o_react, fluxes=dict(C2.FLUXES),
                               iff=o_iff, step_thermal=TS.ThermoStruct2D.step_thermal)

        def step(s, U, dt):
            prev = led.active
            led.active = (s, 0.5 * dt)
            try:
                out = o_step(s, U, dt)
            finally:
                led.active = prev
            led.acc_for(s)["substeps"] += 1
            return out

        def iwc(s, Fi, Fj, T, Qz, Qy, k):
            o_iwc(s, Fi, Fj, T, Qz, Qy, k)
            if led.active is not None and led.active[0] is s:
                led._viscous(s, Fi, Fj, led.active[1])

        def react(s, U, dt):
            out = o_react(s, U, dt)
            if out is not U and led.active is not None and led.active[0] is s:
                vol = s.block.vol * (~s.block.blanked)
                led.acc_for(s)["react"] += np.einsum("ij,ijk->k", vol, out - U)
            return out

        def wrap(fn):
            def wrapped(L, R, n, gamma):
                out = fn(L, R, n, gamma)
                if led.active is not None:
                    led._inviscid(led.active[0], out, led.active[1])
                return out
            wrapped.__wrapped__ = fn
            return wrapped

        def iff(s, Ue):
            Fi, Fj = o_iff(s, Ue)
            if led.active is not None and led.active[0] is s:
                led._inviscid_faces(s, Fi, Fj, led.active[1])
            return Fi, Fj

        cls.step, cls._isothermal_wall_conduction, cls.react = step, iwc, react
        if o_iff is not None:
            # the solver hands back the face fluxes it differences (W337 made
            # the injector's a prescribed flux, which no Riemann call returns)
            cls._inviscid_face_fluxes = iff
        else:
            # a solver from before W337: the Riemann calls ARE the face fluxes
            for name, fn in list(C2.FLUXES.items()):
                C2.FLUXES[name] = wrap(fn)
        o_st = TS.ThermoStruct2D.step_thermal

        def step_thermal(ts, T, dt, h_in, T_gas_in, h_out, T_gas_out, radiate=True,
                         T_inf=250.0):
            Tn = o_st(ts, T, dt, h_in, T_gas_in, h_out, T_gas_out, radiate=radiate,
                      T_inf=T_inf)
            led._shell(ts, T, Tn, dt, h_in, T_gas_in, h_out, T_gas_out, radiate, T_inf)
            return Tn

        TS.ThermoStruct2D.step_thermal = step_thermal
        cls._w336_installed = True

    def uninstall(self):
        """Put every wrapped method back (for tests; a stage never needs it)."""
        o = getattr(self, "_originals", None)
        if not o:
            return
        cls = self.C2.Compressible2D
        cls.step, cls._isothermal_wall_conduction, cls.react = o["step"], o["iwc"], o["react"]
        if o.get("iff") is not None:
            cls._inviscid_face_fluxes = o["iff"]
        self.C2.FLUXES.clear()
        self.C2.FLUXES.update(o["fluxes"])
        self.TS.ThermoStruct2D.step_thermal = o["step_thermal"]
        cls._w336_installed = False
        self._originals = None

    def instant(self, s, U):
        """The boundary inflow RATES at one instant: the residual evaluated once
        with weight 1, into a scratch accumulator."""
        keep_acc, keep = self.acc.pop(id(s), None), self.active
        self.active = (s, 1.0)
        try:
            s.residual(U)
            out = self.acc.pop(id(s))
        finally:
            self.active = keep
            if keep_acc is not None:
                self.acc[id(s)] = keep_acc
        return out


# ---------------------------------------------------------------------------
# geometry of the seams: which faces of which block face which neighbour
# ---------------------------------------------------------------------------


class Seams:
    def __init__(self, ep, M):
        gen = M["gen"]
        self.ep, self.gen = ep, gen
        geo = ep.geo
        self.hp = float(geo.plume_halfwidth)
        fb, eb, gb = ep.blocks["f"][0], ep.blocks["e"][0], ep.blocks["g"][0]
        yf_edges = fb.nodes[0, :, 1]
        ye_edges = eb.nodes[-1, :, 1]
        # the same test `_wire_external` uses to hand f's inlet to e or to d
        self.f_core = ((yf_edges[:-1] >= ye_edges[0] - 1e-12)
                       & (yf_edges[1:] <= ye_edges[-1] + 1e-12))
        self.g_live_in = ~gb.blanked[0]
        self.d_frac = {}
        for k, blk in enumerate(ep.blocks["d"]):
            y = blk.nodes[-1, :, 1]
            lo = np.minimum(np.abs(y[:-1]), np.abs(y[1:]))
            hi = np.maximum(np.abs(y[:-1]), np.abs(y[1:]))
            width = np.maximum(hi - lo, 1e-300)
            self.d_frac[k] = dict(
                g=np.clip((hi - np.maximum(lo, self.hp)) / width, 0.0, 1.0),
                f=np.clip((np.minimum(hi, self.hp) - lo) / width, 0.0, 1.0))
        self.d_wall = {k: ep._d_wall_side(k) for k in range(len(ep.blocks["d"]))}
        self.d_on = {k: ep._airframe_frac(ep._wall_edges("d", k, self.d_wall[k])) > 0.0
                     for k in self.d_wall}
        n, L = gen._outward(eb, "imax")
        self.A_exit_z = float(np.sum(n[:, 0] * L))
        self.A_exit_y = float(np.sum(n[:, 1] * L))
        self.z_exit = float(geo.z_exit)
        self.z_inj = float(geo.z_injector)


def _sum_side(acc, side, mask=None, weight=None, kinds=("inv", "visc")):
    tot = 0.0
    for kd in kinds:
        v = acc[kd][side]
        if mask is not None:
            v = v[mask]
        if weight is not None:
            v = v * weight[:, None]
        tot = tot + v.sum(axis=0)
    return np.asarray(tot, dtype=float)


def _totals(s, U):
    vol = s.block.vol * (~s.block.blanked)
    return np.einsum("ij,ijk->k", vol, U)


def block_budget(s, acc, before, after):
    """(residual, scale, per-term sums) of one block's content budget."""
    live = ~s.block.blanked[0]
    inflow = np.zeros(s.nv)
    absum = np.zeros(s.nv)
    for kd in ("inv", "visc"):
        for side, arr in acc[kd].items():
            v = arr[live] if side in ("imin", "imax") else arr
            inflow = inflow + v.sum(axis=0)
            absum = absum + np.abs(v).sum(axis=0)
    inflow = inflow + acc["react"]
    absum = absum + np.abs(acc["react"])
    delta = after - before
    resid = delta - inflow
    scale = np.maximum(np.maximum(np.abs(before), np.abs(after)), absum)
    return resid, scale, delta, inflow


def _rate(s, U, thermo):
    """RMS of dU/dt over live cells, relative to the block's mean content: how
    far from steady the block is, per second."""
    r = s.residual(U)
    live = ~s.block.blanked
    out = {}
    for k, name in ((0, "mass"), (3, "energy")):
        rr = r[..., k][live]
        out[name] = float(np.sqrt(np.mean(rr ** 2)) / max(np.mean(np.abs(U[..., k][live])), 1e-300))
    return out


# ---------------------------------------------------------------------------
# one macro step, accounted for
# ---------------------------------------------------------------------------


def account(ep, led, sm, M, before, after, rigid0, p_inf, dt, parts=("engine", "external", "shell"),
            advanced=None):
    """Everything one macro step moved, and where. `advanced` names the gas
    agents that were stepped (all of them in a coupled step)."""
    gen, thermo = M["gen"], M["thermo"]
    mdot = float(ep.scales.mdot)
    row = dict(t=float(ep.t), p_inf=float(p_inf))
    acc = {(ag, k): led.acc_for(s) for ag in gen.GAS_AGENTS for k, s in enumerate(ep.sol[ag])}
    sol = {(ag, k): s for ag in gen.GAS_AGENTS for k, s in enumerate(ep.sol[ag])}
    run_agents = set(advanced if advanced is not None else gen.GAS_AGENTS)

    # -- A1: every block's budget closes -------------------------------------
    tele, subs = {}, {}
    for (ag, k), s in sol.items():
        if ag not in run_agents:
            continue
        res, scale, delta, inflow = block_budget(s, acc[(ag, k)], before[(ag, k)], after[(ag, k)])
        tele["%s%d" % (ag, k)] = [float(x) for x in np.abs(res[:4]) / np.maximum(scale[:4], 1e-300)]
        subs["%s%d" % (ag, k)] = int(acc[(ag, k)]["substeps"])
    row["telescoping_rel"] = tele
    row["substeps"] = subs

    if "engine" in parts:
        A = {ag: acc[(ag, 0)] for ag in gen.ENGINE_AGENTS}
        # -- A2: walls -------------------------------------------------------
        wall_mass = sum(float(np.abs(A[ag][kd][side][:, 0]).sum())
                        for ag in A for side in ("jmin", "jmax") for kd in ("inv", "visc"))
        wall_E_inv = sum(float(np.abs(A[ag]["inv"][side][:, 3]).sum())
                         for ag in A for side in ("jmin", "jmax"))
        row["engine_wall_mass_rel"] = wall_mass / (mdot * dt)
        row["engine_wall_inviscid_energy_abs"] = wall_E_inv
        # -- the engine's mass account -----------------------------------------
        inj = _sum_side(A["a"], "imin")
        ex = _sum_side(A["e"], "imax")
        c_ab = _sum_side(A["a"], "imax")[:4] + _sum_side(A["b"], "imin")[:4]
        c_be = _sum_side(A["b"], "imax") + _sum_side(A["e"], "imin")
        walls = sum(_sum_side(A[ag], side)[:4] for ag in A for side in ("jmin", "jmax"))
        store = sum((after[(ag, 0)] - before[(ag, 0)])[:4] for ag in gen.ENGINE_AGENTS)
        react_E = float(A["a"]["react"][3])
        row["injector_mdot_over_declared"] = float(inj[0] / dt / mdot)
        row["exit_mdot_over_declared"] = float(-ex[0] / dt / mdot)
        row["throat_mdot_b_over_declared"] = float(-_sum_side(A["b"], "imax")[0] / dt / mdot)
        row["throat_mdot_e_over_declared"] = float(_sum_side(A["e"], "imin")[0] / dt / mdot)
        row["seam_ab_mass_rel"] = float(c_ab[0] / (mdot * dt))
        row["seam_be_mass_rel"] = float(c_be[0] / (mdot * dt))
        row["seam_ab_zmom_rel"] = float(c_ab[1] / max(abs(ex[1]), 1e-300))
        row["seam_be_zmom_rel"] = float(c_be[1] / max(abs(ex[1]), 1e-300))
        row["seam_ab_energy_rel"] = float(c_ab[3] / max(abs(ex[3]), 1e-300))
        row["seam_be_energy_rel"] = float(c_be[3] / max(abs(ex[3]), 1e-300))
        row["engine_mass_storage_rel"] = float(store[0] / (mdot * dt))
        row["engine_mass_closure"] = float((store[0] - (inj[0] + ex[0] + walls[0] + c_ab[0] + c_be[0]))
                                           / (mdot * dt))
        row["reaction_heat"] = react_E
        # -- thrust, by the exit plane and by the walls -------------------------
        I_inj, I_walls, I_exit = inj[1], walls[1], ex[1]
        T_exit = -I_exit / dt - p_inf * sm.A_exit_z
        T_walls = (I_inj + I_walls) / dt - p_inf * sm.A_exit_z
        row["thrust_exit_face_mean"] = float(T_exit)
        row["thrust_walls_mean"] = float(T_walls)
        row["thrust_storage_term"] = float(store[1] / dt)
        row["thrust_seam_term"] = float((c_ab[1] + c_be[1]) / dt)
        row["thrust_routes_rel"] = float((T_walls - T_exit) / T_exit)
        row["thrust_routes_closure"] = float(((T_walls - T_exit) - (store[1] - c_ab[1] - c_be[1]) / dt)
                                             / T_exit)
        se = ep.sol["e"][0]
        inst = led.instant(se, ep.U["e"][0])
        ex_i = _sum_side(inst, "imax")
        T_face_end = -ex_i[1] - p_inf * sm.A_exit_z
        row["thrust_face_end"] = float(T_face_end)
        lb = getattr(ep, "loads_body", None)
        if lb is not None and "shell" in parts:
            row["thrust_cells_end"] = float(-lb[0])
            row["thrust_cells_vs_face_rel"] = float((-lb[0] - T_face_end) / T_face_end)
        row["exit_mdot_end_over_declared"] = float(-ex_i[0] / mdot)
        # -- the engine's walls: heat and force, per side ----------------------
        row["engine_heat_lost"] = {side: float(-sum(_sum_side(A[ag], side)[3] for ag in A))
                                   for side in ("jmin", "jmax")}
        row["engine_wall_force_z"] = {side: float(-sum(_sum_side(A[ag], side)[1] for ag in A) / dt)
                                      for side in ("jmin", "jmax")}
        # -- chamber and exit state --------------------------------------------
        sb = ep.sol["b"][0]
        Wb = thermo.cons_to_prim(ep.U["b"][0], sb.cfg.gamma)
        vol = sb.block.vol
        row["chamber_p_mean"] = float(np.sum(Wb[..., 3] * vol) / np.sum(vol))
        We = thermo.cons_to_prim(ep.U["e"][0][-1], se.cfg.gamma)
        n, L = gen._outward(se.block, "imax")
        rho, u, v, p = We[:, 0], We[:, 1], We[:, 2], We[:, 3]
        un = u * n[:, 0] + v * n[:, 1]
        c = np.sqrt(se.cfg.gamma * p / rho)
        m = rho * un * L
        row["exit_mach_massavg"] = float(np.sum(m * np.sqrt(u ** 2 + v ** 2) / c) / np.sum(m))
        row["exit_p_areaavg"] = float(np.sum(p * L) / np.sum(L))
        row["exit_u_massavg"] = float(np.sum(m * u) / np.sum(m))
        row["rate"] = {ag: _rate(ep.sol[ag][0], ep.U[ag][0], thermo) for ag in gen.ENGINE_AGENTS}

    if "external" in parts:
        # -- the airframe wall -------------------------------------------------
        wm, heat, heat_slot = 0.0, {}, {}
        for k, side in sm.d_wall.items():
            a = acc[("d", k)]
            wm += sum(float(np.abs(a[kd][side][:, 0]).sum()) for kd in ("inv", "visc"))
            on = sm.d_on[k]
            heat[k] = float(-_sum_side(a, side, mask=on)[3])
            heat_slot[k] = float(-_sum_side(a, side, mask=~on)[3]) if (~on).any() else 0.0
        row["airframe_wall_mass_rel"] = wm / (mdot * dt)
        row["airframe_heat_lost"] = heat
        row["slot_heat_lost"] = heat_slot
        row["rate_external"] = {"%s%d" % (ag, k): _rate(s, ep.U[ag][k], thermo)
                                for ag in gen.EXTERNAL_AGENTS if ag in run_agents
                                for k, s in enumerate(ep.sol[ag])}
    if "external" in parts and {"d", "f", "g"} <= run_agents:
        # -- plane seams among the external blocks -------------------------------
        f, g = acc[("f", 0)], acc[("g", 0)]
        d_to = {"f": 0.0, "g": 0.0, "all": 0.0}
        for k in sm.d_frac:
            a = acc[("d", k)]
            d_to["f"] = d_to["f"] + _sum_side(a, "imax", weight=sm.d_frac[k]["f"])
            d_to["g"] = d_to["g"] + _sum_side(a, "imax", weight=sm.d_frac[k]["g"])
            d_to["all"] = d_to["all"] + _sum_side(a, "imax")
        f_core = _sum_side(f, "imin", mask=sm.f_core)
        f_outer = _sum_side(f, "imin", mask=~sm.f_core)
        g_in = _sum_side(g, "imin", mask=sm.g_live_in)
        ref = abs(_sum_side(f, "imax")[0]) + abs(_sum_side(g, "imax", mask=sm.g_live_in)[0])
        if "engine" in parts:
            ex = _sum_side(acc[("e", 0)], "imax")
            row["seam_ef_mass_rel"] = float((ex[0] + f_core[0]) / (mdot * dt))
            row["seam_ef_energy_rel"] = float((ex[3] + f_core[3]) / max(abs(ex[3]), 1e-300))
        row["seam_df_mass_rel"] = float((d_to["f"][0] + f_outer[0]) / max(abs(d_to["f"][0]), 1e-300))
        row["seam_dg_mass_rel"] = float((d_to["g"][0] + g_in[0]) / max(abs(d_to["g"][0]), 1e-300))
        row["seam_d_fg_mass_rel"] = float((d_to["all"][0] + f_outer[0] + g_in[0])
                                          / max(abs(d_to["all"][0]), 1e-300))
        gf = (_sum_side(f, "jmin") + _sum_side(f, "jmax")
              + _sum_side(g, "hole_lo") + _sum_side(g, "hole_hi"))
        row["seam_gf_mass_abs"] = float(gf[0])
        row["seam_gf_mass_over_outflows"] = float(gf[0] / max(ref, 1e-300))

    if "shell" in parts:
        idx = {id(t): k for k, t in enumerate(ep.shell)}
        shell = {}
        for tid, recs in led.shell.items():
            k = idx[tid]
            r = recs[-1]
            eng = r["z_in"] > sm.z_inj
            Q = float(r["q_in"].sum() + r["q_conv"].sum() + r["q_rad"].sum())
            shell[k] = dict(
                energy_residual_rel=float(abs(r["dE"] - Q) / max(abs(r["q_in"]).sum()
                                                                  + abs(r["q_conv"]).sum(), 1e-300)),
                energy_residual_over_stored=float(abs(r["dE"] - Q) / max(r["E_stored"], 1e-300)),
                q_in_engine=float(r["q_in"][eng].sum()),
                q_in_tank=float(r["q_in"][~eng].sum()),
                q_conv_out=float(r["q_conv"].sum()),
                q_rad_applied=float(r["q_rad"].sum()),
                q_rad_ambient=float(r["q_rad_sky"].sum()),
                T_inner_engine_max=float(r["T_inner"][eng].max()) if eng.any() else None,
                T_inner_engine_mean=float(r["T_inner"][eng].mean()) if eng.any() else None)
        row["shell"] = {str(k): v for k, v in shell.items()}
        if "engine" in parts:
            th = {}
            for side in ("jmin", "jmax"):
                k = ep._panel(side)
                gas = row["engine_heat_lost"][side]
                got = shell[k]["q_in_engine"]
                th[side] = dict(panel=k, gas_lost=gas, shell_gained=got,
                                rel=float((got - gas) / max(abs(gas), 1e-300)))
            row["thermal_engine"] = th
        if "external" in parts:
            to = {}
            for k in sm.d_wall:
                gas = row["airframe_heat_lost"][k]
                got = shell[k]["q_conv_out"]
                to[str(k)] = dict(gas_lost=gas, shell_gained_conv=got,
                                  rel=float((got - gas) / max(abs(gas), 1e-300)),
                                  rad_slip_over_outer=float(
                                      (shell[k]["q_rad_applied"] - shell[k]["q_rad_ambient"])
                                      / max(abs(shell[k]["q_conv_out"] + shell[k]["q_rad_applied"]),
                                            1e-300)))
            row["thermal_airframe"] = to
        # -- the vehicle's mass --------------------------------------------------
        if "engine" in parts:
            dm = float(ep.rigid[6] - rigid0[6])
            row["vehicle_mass_loss_over_exit"] = float(-dm / max(-_sum_side(acc[("e", 0)], "imax")[0],
                                                                 1e-300))
            row["rigid"] = [float(x) for x in ep.rigid]
            row["loads"] = [float(x) for x in ep.loads]
            row["loads_body"] = [float(x) for x in getattr(ep, "loads_body", np.zeros(4))]
    return row


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def _persist(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=float)
    for _ in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.25)
    os.replace(tmp, path)


def _machine():
    import platform
    return dict(platform=platform.platform(), python=platform.python_version(),
                processor=platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
                cpus=os.cpu_count(), numpy=np.__version__)


def _episode(M, case, coarsen, dt_macro, n_macro):
    gen, sw, cfgmod = M["gen"], M["sweep"], M["config"]
    cfg = cfgmod.load_config()
    p = sw.corner_cases()[case]
    spec = gen.EpisodeSpec(point=p, n_macro=n_macro, dt_macro=dt_macro, coarsen=coarsen)
    return gen.CoupledEpisode(spec, cfg), p


def _all_totals(ep, M):
    gen = M["gen"]
    return {(ag, k): _totals(s, ep.U[ag][k]) for ag in gen.GAS_AGENTS
            for k, s in enumerate(ep.sol[ag])}


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage_coupled(a, name, old_wall=False):
    """The coupled episode, stepped exactly as `CoupledEpisode.step_macro` steps
    it, with every macro step accounted for."""
    M = _mods()
    reverted = revert(M, getattr(a, "revert", None))
    led = Ledger(M)
    led.install()
    if old_wall:
        C2 = M["C2"]

        def old_residual(self, U):
            Ue = self.ghost(U)
            r = self._inviscid_residual(Ue)
            if not self.cfg.inviscid:
                r = r + self._viscous_residual(Ue)
            if self._hole is not None:
                j0, j1 = self._hole
                r[:, j0:j1 + 1] = 0.0
            return r
        C2.Compressible2D.residual = old_residual
    n_steps = int(round(a.t_end / a.dt_macro))
    ep, p = _episode(M, a.case, a.coarsen, a.dt_macro, n_steps + 1)
    sm = Seams(ep, M)
    rec = None
    if name == "audit" and a.record and os.path.exists(a.record):
        z = np.load(a.record)
        rec = {k: z[k] for k in ("rigid", "loads", "loads_body") if k in z.files}
    out = dict(stage=name, case=a.case, label=p.label, coarsen=a.coarsen, dt_macro=a.dt_macro,
               t_end=a.t_end, old_wall=bool(old_wall), mdot_declared=float(ep.scales.mdot),
               ideal=dict(u_e=float(ep.scales.u_exit), p_e=float(ep.scales.p_exit),
                          T_e=float(ep.scales.T_exit), M_e=float(ep.scales.M_exit),
                          rho_e=float(ep.scales.rho_exit), p_c=float(p.p_c), T_c=float(p.T_c)),
               A_exit_z=sm.A_exit_z, machine=_machine(), steps=[], build_repo=build_repo_identity(),
               reverted=reverted)
    path = os.path.join(OUT, a.json or ("%s.json" % name))
    t0 = time.perf_counter()
    print("W336 %s: case %d (%s), coarsen %d, dt_macro %g, %d steps%s"
          % (name, a.case, p.label, a.coarsen, a.dt_macro, n_steps,
             ", OLD WALL (W332 reverted)" if old_wall else ""), flush=True)
    if reverted:
        print("  REVERTED to the pre-Tier-88 code: %s" % ",".join(reverted), flush=True)
    for n in range(n_steps):
        led.reset()
        before = _all_totals(ep, M)
        rigid0 = ep.rigid.copy()
        p_inf = ep._atm()[1]
        ts = time.perf_counter()
        ep.step_macro()
        wall = time.perf_counter() - ts
        after = _all_totals(ep, M)
        row = account(ep, led, sm, M, before, after, rigid0, p_inf, a.dt_macro)
        row["wall_s"] = wall
        if rec is not None and n + 1 < rec["rigid"].shape[0]:
            dev = {}
            for k, v in (("rigid", ep.rigid), ("loads", ep.loads),
                         ("loads_body", getattr(ep, "loads_body", None))):
                if v is None or k not in rec:
                    continue
                want = rec[k][n + 1]
                dev[k] = float(np.max(np.abs(np.asarray(v) - want)
                                      / np.maximum(np.abs(want), 1e-300)))
                dev[k + "_bitwise"] = bool(np.array_equal(np.asarray(v), want))
            row["record_dev"] = dev
        out["steps"].append(row)
        _persist(path, out)
        tele = max(max(v) for v in row["telescoping_rel"].values())
        print("  step %2d t=%.4f  %.0f s  tele %.1e  wall %.1e  inj %.4f  exit %.4f  "
              "ab %+.1e  be %+.1e  thrust exit %.5g walls %.5g  %s"
              % (n + 1, row["t"], wall, tele, row["engine_wall_mass_rel"],
                 row["injector_mdot_over_declared"], row["exit_mdot_over_declared"],
                 row["seam_ab_mass_rel"], row["seam_be_mass_rel"],
                 row["thrust_exit_face_mean"], row["thrust_walls_mean"],
                 ("dev %.1e" % row["record_dev"]["rigid"]) if "record_dev" in row else ""),
              flush=True)
    out["wall_seconds"] = time.perf_counter() - t0
    out["gas_substeps"] = int(ep.substeps)
    _persist(path, out)
    return out


def stage_grid(a, part):
    """One subsystem alone at one resolution: the engine (walls held at the
    shell's initial 288.15 K) or the external flow (the airframe likewise). The
    rigid state stays at t = 0, so the freestream and back pressure are fixed."""
    M = _mods()
    gen = M["gen"]
    reverted = revert(M, getattr(a, "revert", None))
    led = Ledger(M)
    led.install()
    n_steps = int(round(a.t_end / a.dt_macro))
    ep, p = _episode(M, a.case, a.coarsen, a.dt_macro, n_steps + 1)
    sm = Seams(ep, M)
    name = "grid_%s_c%d" % (part, a.coarsen)
    out = dict(stage=name, case=a.case, label=p.label, coarsen=a.coarsen, dt_macro=a.dt_macro,
               t_end=a.t_end, mdot_declared=float(ep.scales.mdot),
               ideal=dict(u_e=float(ep.scales.u_exit), p_e=float(ep.scales.p_exit),
                          T_e=float(ep.scales.T_exit), M_e=float(ep.scales.M_exit),
                          rho_e=float(ep.scales.rho_exit), p_c=float(p.p_c), T_c=float(p.T_c)),
               A_exit_z=sm.A_exit_z,
               shapes={ag: [list(b.shape) for b in ep.blocks[ag]] for ag in gen.GAS_AGENTS},
               machine=_machine(), steps=[], build_repo=build_repo_identity(), reverted=reverted)
    path = os.path.join(OUT, a.json or ("%s.json" % name))
    agents = gen.ENGINE_AGENTS if part == "engine" else ("d",)
    t0 = time.perf_counter()
    print("W336 %s: case %d, coarsen %d, dt_macro %g, %d steps"
          % (name, a.case, a.coarsen, a.dt_macro, n_steps), flush=True)
    if reverted:
        print("  REVERTED to the pre-Tier-88 code: %s" % ",".join(reverted), flush=True)
    for n in range(n_steps):
        led.reset()
        before = _all_totals(ep, M)
        rigid0 = ep.rigid.copy()
        p_inf = ep._atm()[1]
        ts = time.perf_counter()
        if part == "engine":
            ep._wire_engine()
        else:
            ep._wire_external()
        ep._advance_gas(agents, a.dt_macro)
        ep.t += a.dt_macro
        wall = time.perf_counter() - ts
        after = _all_totals(ep, M)
        if part == "engine":
            row = account(ep, led, sm, M, before, after, rigid0, p_inf, a.dt_macro,
                          parts=("engine",), advanced=agents)
        else:
            row = account(ep, led, sm, M, before, after, rigid0, p_inf, a.dt_macro,
                          parts=("external",), advanced=agents)
            ep._compute_loads()
            lb = ep.loads_body
            row["drag"] = float(lb[2])
            row["side_force"] = float(lb[3])
            row.update(_aero_split(ep, M, p_inf))
        row["wall_s"] = wall
        out["steps"].append(row)
        _persist(path, out)
        if part == "engine":
            print("  step %d t=%.4f  %.0f s  throat %.5f  exit %.5f  thrust %.6g  p_c %.5g  M_e %.4f"
                  "  heat %.5g" % (n + 1, row["t"], wall, row["throat_mdot_e_over_declared"],
                                   row["exit_mdot_over_declared"], row["thrust_exit_face_mean"],
                                   row["chamber_p_mean"], row["exit_mach_massavg"],
                                   sum(row["engine_heat_lost"].values())), flush=True)
        else:
            print("  step %d t=%.4f  %.0f s  drag %.6g (p %.5g, shear %.5g)  heat %.5g  rate %.2e"
                  % (n + 1, row["t"], wall, row["drag"], row["drag_pressure"], row["drag_shear"],
                     sum(row["airframe_heat_lost"].values()),
                     max(v["mass"] for k, v in row["rate_external"].items() if k.startswith("d"))),
                  flush=True)
    out["wall_seconds"] = time.perf_counter() - t0
    out["gas_substeps"] = int(ep.substeps)
    _persist(path, out)
    return out


def _aero_split(ep, M, p_inf):
    """`_compute_loads`' aero part, split into pressure and shear (body z)."""
    gen, thermo = M["gen"], M["thermo"]
    pz, sz = 0.0, 0.0
    for k, blk in enumerate(ep.blocks["d"]):
        sd = ep.sol["d"][k]
        side = ep._d_wall_side(k)
        n_gas, Lw = gen._outward(blk, side)
        Lw = np.asarray(Lw, dtype=float) * ep._airframe_frac(ep._wall_edges("d", k, side))
        n_body = -n_gas
        Wd = thermo.cons_to_prim(gen._face_cells(ep.U["d"][k], side), sd.cfg.gamma)
        pz += float(np.sum(-(Wd[:, 3] - p_inf) * n_body[:, 0] * Lw))
        if not sd.cfg.inviscid:
            t = np.stack([n_body[:, 1], -n_body[:, 0]], axis=-1)
            ut = Wd[:, 1] * t[:, 0] + Wd[:, 2] * t[:, 1]
            T = Wd[:, 3] / (Wd[:, 0] * sd.cfg.R)
            mu = thermo.sutherland(T, sd.cfg.mu_ref, sd.cfg.T_mu_ref, sd.cfg.sutherland_S)
            j = 0 if side == "jmin" else -1
            dn = 0.5 * blk.vol[:, j] / np.maximum(blk.a_j[:, j], 1e-30)
            sz += float(np.sum(mu * ut / dn * Lw * t[:, 0]))
    return dict(drag_pressure=pz, drag_shear=sz)


# ---------------------------------------------------------------------------
# evaluation and the report
# ---------------------------------------------------------------------------


def _load(name, out_dir=None):
    path = os.path.join(out_dir or OUT, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _last(steps, key, n=10, fn=None):
    vals = [fn(s) if fn else s.get(key) for s in steps[-n:]]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _order(q1, q2, q3, r=2.0):
    """Observed order and Richardson estimate from coarse q1, medium q2, fine q3
    at a constant refinement ratio r; None when the differences change sign."""
    d1, d2 = q1 - q2, q2 - q3
    if d1 == 0 or d2 == 0 or np.sign(d1) != np.sign(d2):
        return None, None
    p = float(np.log(abs(d1 / d2)) / np.log(r))
    return p, float(q3 - d2 / (r ** p - 1.0)) if p > 0 else None


def rigid_deviation(steps, record):
    """Per step: position, velocity, attitude and mass, each against its own
    magnitude (lateral round-off does not count as a state), and whether the
    four components that carry the flight -- y, theta, v_y, m -- are bitwise."""
    z = np.load(record)
    R = z["rigid"]
    out = []
    for k, s in enumerate(steps):
        if k + 1 >= R.shape[0] or "rigid" not in s:
            break
        g, w = np.asarray(s["rigid"], dtype=float), R[k + 1]
        dev = max(np.linalg.norm(g[0:2] - w[0:2]) / np.linalg.norm(w[0:2]),
                  np.linalg.norm(g[3:5] - w[3:5]) / np.linalg.norm(w[3:5]),
                  abs(g[2] - w[2]) / abs(w[2]), abs(g[6] - w[6]) / abs(w[6]))
        out.append(dict(dev=float(dev), omega_abs=float(abs(g[5] - w[5])),
                        flight_bitwise=bool(np.array_equal(g[[1, 2, 4, 6]], w[[1, 2, 4, 6]]))))
    return out


RECORD = os.path.join(ROOT, "out", "w321", "episode.npz")


def evaluate(out_dir=None, record=None):
    """Each registered prediction, from the stage records. None = not measured."""
    def _ld(name):
        return _load(name, out_dir)
    record = record or RECORD
    au = _ld("audit.json")
    cw = _ld("control_wall.json")
    res = {}

    def put(name, got, ok):
        res[name] = dict(claim=PREDICTIONS[name], got=got,
                         held=None if ok is None else bool(ok))

    if au and au["steps"]:
        S = au["steps"]
        tele = max(max(max(v) for v in s["telescoping_rel"].values()) for s in S)
        put("A1", tele, tele < 1e-10)
        wm = max(max(s["engine_wall_mass_rel"], s.get("airframe_wall_mass_rel", 0.0)) for s in S)
        put("A2", wm, wm < 1e-12)
        sh = max(max(v["energy_residual_over_stored"] for v in s["shell"].values()) for s in S)
        put("A3", sh, sh < 1e-9)
        rd = rigid_deviation(S, record) if os.path.exists(record) else []
        if rd:
            worst = max(r["dev"] for r in rd)
            om = max(r["omega_abs"] for r in rd)
            put("A4", dict(worst=worst, omega_abs=om, steps=len(rd),
                           flight_bitwise_steps=sum(r["flight_bitwise"] for r in rd)),
                worst < 1e-9 and om < 1e-12)
        else:
            put("A4", None, None)
        inj = _last(S, "injector_mdot_over_declared")
        put("P1", inj, abs(inj - 1.0) < 0.02)
        ab = _last(S, None, fn=lambda s: abs(s["seam_ab_mass_rel"]))
        be = _last(S, None, fn=lambda s: abs(s["seam_be_mass_rel"]))
        put("P2", [ab, be], max(ab, be) < 0.01)
        tr = _last(S, None, fn=lambda s: abs(s["thrust_routes_rel"]))
        put("P3", tr, tr < 0.02)
        cf = max(abs(s["thrust_cells_vs_face_rel"]) for s in S[-10:])
        put("P4", cf, cf < 0.02)
        th = [_last(S, None, fn=lambda s, sd=sd: abs(s["thermal_engine"][sd]["rel"]))
              for sd in ("jmin", "jmax")]
        put("P5", th, max(th) < 0.02)
        vm = _last(S, None, fn=lambda s: abs(s["vehicle_mass_loss_over_exit"] - 1.0))
        put("P6", vm, vm < 0.02)
        rs = max(abs(v["rad_slip_over_outer"]) for s in S for v in s["thermal_airframe"].values())
        put("P7", rs, rs < 0.05)
    if cw and cw["steps"]:
        res["control_wall"] = dict(got=cw["steps"][0]["engine_wall_mass_rel"],
                                   note="must FAIL A2's 1e-12 for the gate to have teeth")
    runs = {dt: _ld(n) for dt, n in ((1e-2, "dt_10ms.json"), (2.5e-3, "dt_2p5ms.json"))}
    if au and all(runs.values()):
        def seam_at(run):
            S = [s for s in run["steps"] if 0.03 < s["t"] <= 0.06 + 1e-9]
            return float(np.mean([abs(s["seam_ab_mass_rel"]) + abs(s["seam_be_mass_rel"]) for s in S]))
        au60 = dict(au, steps=[s for s in au["steps"] if s["t"] <= 0.06 + 1e-9])
        c10, c5, c2 = seam_at(runs[1e-2]), seam_at(au60), seam_at(runs[2.5e-3])
        put("P8", [c10 / c5, c5 / c2], min(c10 / c5, c5 / c2) >= 1.5)

        def vy60(run):
            s = min(run["steps"], key=lambda s: abs(s["t"] - 0.06))
            return s["rigid"][4]
        v5, v2 = vy60(au60), vy60(runs[2.5e-3])
        put("P9", abs(v5 - v2) / abs(v2), abs(v5 - v2) / abs(v2) < 1e-4)
    ge = {c: _ld("grid_engine_c%d.json" % c) for c in (4, 2, 1)}
    if ge[1]:
        g1 = ge[1]["steps"][-1]
        put("P10", g1["throat_mdot_e_over_declared"], abs(g1["throat_mdot_e_over_declared"] - 1) < 0.03)
        if ge[4]:
            g4 = ge[4]["steps"][-1]
            rel = abs(g4["thrust_exit_face_mean"] / g1["thrust_exit_face_mean"] - 1)
            put("P11", rel, rel < 0.05)
            h1, h4 = sum(g1["engine_heat_lost"].values()), sum(g4["engine_heat_lost"].values())
            put("P12", h1 / h4, h1 / h4 > 2.0)
    gx = {c: _ld("grid_external_c%d.json" % c) for c in (4, 2, 1)}
    if gx[1] and gx[4]:
        d1, d4 = gx[1]["steps"][-1]["drag"], gx[4]["steps"][-1]["drag"]
        put("P13", abs(d4 / d1 - 1), abs(d4 / d1 - 1) < 0.10)
    return res


def stage_report(a):
    res = evaluate()
    print("=" * 96)
    print("W336 -- predictions, registered 2026-09-24 before any arm ran")
    print("=" * 96)
    for k in list(PREDICTIONS) + ["control_wall"]:
        if k not in res:
            print("  %-4s not measured" % k)
            continue
        r = res[k]
        held = r.get("held")
        tag = "HELD" if held else ("FAILED" if held is False else "--")
        print("  %-4s %-6s got %s" % (k, tag, json.dumps(r["got"])[:100]))
        if "claim" in r:
            print("        %s" % r["claim"])
    t88 = evaluate_t88()
    print("=" * 96)
    print("Tier 88 -- predictions registered 2026-09-26 11:05 EDT, before the fixed run")
    print("=" * 96)
    for k in PREDICTIONS_T88:
        if k not in t88:
            print("  %-4s not measured" % k)
            continue
        r = t88[k]
        tag = "HELD" if r["held"] else ("FAILED" if r["held"] is False else "--")
        print("  %-4s %-6s got %s" % (k, tag, json.dumps(r["got"])[:100]))
        print("        %s" % r["claim"])
    res["tier88"] = t88
    # convergence tables
    for part, keys in (("engine", ("throat_mdot_e_over_declared", "chamber_p_mean", "exit_mach_massavg",
                                    "exit_p_areaavg", "thrust_exit_face_mean")),
                       ("external", ("drag", "drag_pressure", "drag_shear"))):
        runs = {c: _load("grid_%s_c%d.json" % (part, c)) for c in (4, 2, 1)}
        if not all(runs.values()):
            continue
        print()
        print("grid, %s (last step):" % part)
        for key in keys:
            q = [runs[c]["steps"][-1][key] for c in (4, 2, 1)]
            p_obs, rich = _order(*q)
            print("  %-30s c4 %-13.6g c2 %-13.6g c1 %-13.6g order %s  richardson %s"
                  % (key, q[0], q[1], q[2], "%.2f" % p_obs if p_obs else "--",
                     "%.6g" % rich if rich else "--"))
        heat = [sum(runs[c]["steps"][-1]["%s_heat_lost" % ("engine" if part == "engine" else "airframe")]
                    .values()) for c in (4, 2, 1)]
        print("  %-30s c4 %-13.6g c2 %-13.6g c1 %-13.6g" % ("wall heat lost per step", *heat))
    res["tables"] = tables()
    res["_report_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _persist(os.path.join(OUT, "evaluation.json"), res)


def validation(au):
    """The engine against ideal quasi-1D theory at the flow the nozzle
    actually receives (the last 10 audited steps' mean throat flow), with and
    without the planar nozzle's divergence factor sin(a)/a."""
    S = au["steps"][-10:]

    def m(key):
        return float(np.mean([x[key] for x in S]))
    I = au["ideal"]
    mdot = au["mdot_declared"]
    A_e, p_inf = au["A_exit_z"], au["steps"][-1]["p_inf"]
    r = m("throat_mdot_e_over_declared")
    g = 0.12 / 0.04
    alpha = float(np.arctan((0.12 - 0.04) / (0.70 - 0.40)))
    lam = float(np.sin(alpha) / alpha)
    F1 = r * (mdot * I["u_e"] + I["p_e"] * A_e) - p_inf * A_e
    F2 = r * (lam * mdot * I["u_e"] + I["p_e"] * A_e) - p_inf * A_e
    thrust = m("thrust_exit_face_mean")
    return dict(
        area_ratio=g, half_angle_deg=float(np.degrees(alpha)), planar_divergence=lam,
        flow_ratio=r, ideal_thrust_quasi1d=float(F1), ideal_thrust_planar=float(F2),
        measured=thrust, measured_over_quasi1d=thrust / F1, measured_over_planar=thrust / F2,
        exit_M_over_ideal=m("exit_mach_massavg") / I["M_e"],
        exit_u_over_ideal=m("exit_u_massavg") / I["u_e"],
        exit_p_over_scaled_ideal=m("exit_p_areaavg") / (r * I["p_e"]),
        chamber_p_over_declared=m("chamber_p_mean") / I["p_c"])


PRE_T88 = os.path.join(ROOT, "out", "w336_pre_w337")


def evaluate_t88(out_dir=None, pre_dir=None):
    """Tier 88's registered predictions (`PREDICTIONS_T88`), from the stage
    records. None = not measured. `pre_dir` holds the pre-fix audit Q14 reads
    the control against."""
    def ld(name, d=None):
        return _load(name, d or out_dir)
    pre_dir = pre_dir or PRE_T88
    au = ld("audit.json")
    res = {}

    def put(name, got, ok):
        res[name] = dict(claim=PREDICTIONS_T88[name], got=got,
                         held=None if ok is None else bool(ok))

    def win(run, f, lo=0.03, hi=0.06):
        return float(np.mean([f(x) for x in run["steps"] if lo < x["t"] <= hi + 1e-9]))

    S = au["steps"] if au else []
    if S:
        inj = max(abs(s["injector_mdot_over_declared"] - 1.0) for s in S)
        put("Q1", inj, inj < 1e-6)
        th = [_last(S, None, fn=lambda s, sd=sd: abs(s["thermal_engine"][sd]["rel"]))
              for sd in ("jmin", "jmax")]
        put("Q4", th, max(th) < 5e-3)
        slip = max(abs(v["q_rad_applied"] - v["q_rad_ambient"]) / max(abs(v["q_rad_ambient"]), 1e-300)
                   for s in S for v in s["shell"].values())
        put("Q6", slip, slip < 1e-9)
        be = _last(S, None, fn=lambda s: abs(s["seam_be_mass_rel"]))
        put("Q9", be, be < 3e-3)
        pc = _last(S, "chamber_p_mean") / au["ideal"]["p_c"]
        put("Q10", pc, abs(pc - 1.0) < 0.03)
        v = validation(au)["measured_over_planar"]
        put("Q11", v, abs(v - 1.0) < 0.02)
        vm = _last(S, None, fn=lambda s: abs(s["vehicle_mass_loss_over_exit"] - 1.0))
        put("Q12", vm, vm < 0.02)
        tr = _last(S, None, fn=lambda s: abs(s["thrust_routes_rel"]))
        put("Q13", tr, tr < 5e-3)
    ge = {c: ld("grid_engine_c%d.json" % c) for c in (4, 2, 1)}
    ge = {c: r for c, r in ge.items() if r and r["steps"]}
    if ge:
        worst = {c: max(abs(s["injector_mdot_over_declared"] - 1.0) for s in r["steps"])
                 for c, r in ge.items()}
        put("Q2", dict(("c%d" % c, w) for c, w in worst.items()), max(worst.values()) < 1e-6)
    dg = ld("thermal_seam_diag.json")
    if dg and dg.get("sides"):
        wk = max(abs(s["gas_work"]) / max(abs(s["gas_conduction"]), 1e-300) for s in dg["sides"].values())
        put("Q3", wk, wk < 1e-9)
    runs = {10.0: ld("dt_10ms.json"), 2.5: ld("dt_2p5ms.json")}
    if S and all(runs.values()):
        runs[5.0] = dict(au, steps=[s for s in S if s["t"] <= 0.06 + 1e-9])

        def thermal(s):
            return 0.5 * sum(abs(s["thermal_engine"][sd]["rel"]) for sd in ("jmin", "jmax"))
        q = [win(runs[d], thermal) for d in (10.0, 5.0, 2.5)]
        put("Q5", [q[0] / q[1], q[1] / q[2]], min(q[0] / q[1], q[1] / q[2]) >= 1.5)
        q = [win(runs[d], lambda s: abs(s["seam_be_mass_rel"])) for d in (10.0, 5.0, 2.5)]
        put("Q7", [q[0] / q[1], q[1] / q[2]], min(q[0] / q[1], q[1] / q[2]) >= 1.5)
        ab5 = win(runs[5.0], lambda s: abs(s["seam_ab_mass_rel"]))
        put("Q8", [q[1], ab5], q[1] < ab5)
    co, pre = ld("control_old.json"), ld("audit.json", pre_dir)
    if co and co["steps"] and pre and pre["steps"]:
        n = min(len(co["steps"]), len(pre["steps"]))
        dev, bitwise = 0.0, True

        def rel(a, b):
            return abs(a - b) / max(abs(b), 1e-300)
        for c, p in zip(co["steps"][:n], pre["steps"][:n]):
            bitwise = bitwise and c["rigid"] == p["rigid"]
            vals = [(c["injector_mdot_over_declared"], p["injector_mdot_over_declared"]),
                    (c["seam_be_mass_rel"], p["seam_be_mass_rel"])]
            vals += [(c["engine_heat_lost"][sd], p["engine_heat_lost"][sd]) for sd in ("jmin", "jmax")]
            vals += [(c["shell"][k]["q_rad_applied"] - c["shell"][k]["q_rad_ambient"],
                      p["shell"][k]["q_rad_applied"] - p["shell"][k]["q_rad_ambient"]) for k in p["shell"]]
            dev = max(dev, max(rel(x, y) for x, y in vals))
        put("Q14", dict(steps=n, rigid_bitwise=bitwise, worst_rel=dev), bitwise and dev < 1e-12)
    return res


def tables(out_dir=None, record=None):
    """The numbers the write-up quotes, computed here rather than by hand: the
    audit's late-time account, the coupling-step study at 60 ms, and the engine
    against ideal quasi-1D theory at the flow the nozzle actually receives."""
    def ld(n):
        return _load(n, out_dir)
    out = {}
    au = ld("audit.json")
    if au and au["steps"]:
        S = au["steps"][-10:]

        def m(f):
            return float(np.mean([f(x) for x in S]))
        out["audit_late"] = dict(
            injector=m(lambda x: x["injector_mdot_over_declared"]),
            throat_b=m(lambda x: x["throat_mdot_b_over_declared"]),
            throat_e=m(lambda x: x["throat_mdot_e_over_declared"]),
            exit=m(lambda x: x["exit_mdot_over_declared"]),
            seam=dict((k, m(lambda x, k=k: x[k])) for k in (
                "seam_ab_mass_rel", "seam_be_mass_rel", "seam_ef_mass_rel", "seam_df_mass_rel",
                "seam_dg_mass_rel", "seam_gf_mass_over_outflows", "seam_be_zmom_rel",
                "seam_be_energy_rel")),
            thrust_exit=m(lambda x: x["thrust_exit_face_mean"]),
            thrust_walls=m(lambda x: x["thrust_walls_mean"]),
            thrust_seam_term=m(lambda x: x["thrust_seam_term"]),
            engine_heat_gas=m(lambda x: x["thermal_engine"]["jmax"]["gas_lost"]),
            engine_heat_shell=m(lambda x: x["thermal_engine"]["jmax"]["shell_gained"]),
            rad_slip_abs=m(lambda x: x["shell"]["0"]["q_rad_applied"] - x["shell"]["0"]["q_rad_ambient"]),
            airframe_gas_lost=m(lambda x: x["thermal_airframe"]["0"]["gas_lost"]),
            chamber_p=m(lambda x: x["chamber_p_mean"]),
            exit_M=m(lambda x: x["exit_mach_massavg"]),
            exit_p=m(lambda x: x["exit_p_areaavg"]),
            exit_u=m(lambda x: x["exit_u_massavg"]),
            vehicle_mass_over_exit=m(lambda x: x["vehicle_mass_loss_over_exit"]))
        out["validation"] = validation(au)
    runs = {10.0: ld("dt_10ms.json"), 5.0: au, 2.5: ld("dt_2p5ms.json")}
    if all(runs.values()):
        record = record or RECORD
        rec = np.load(record) if os.path.exists(record) else None
        v0 = float(rec["rigid"][0][4]) if rec is not None else None

        def at(run, t):
            return min(run["steps"], key=lambda x: abs(x["t"] - t))

        def win(run, key, lo=0.04, hi=0.06):
            return float(np.mean([key(x) for x in run["steps"] if lo < x["t"] <= hi + 1e-9]))
        rows = {}
        for name, f in (("dv_y_60ms", lambda r: at(r, 0.06)["rigid"][4] - v0 if v0 is not None else None),
                        ("drag_60ms", lambda r: at(r, 0.06)["loads_body"][2]),
                        ("thrust_40_60_mean", lambda r: win(r, lambda x: x["thrust_exit_face_mean"])),
                        ("shell_Tmax_60ms", lambda r: at(r, 0.06)["shell"]["1"]["T_inner_engine_max"]),
                        ("seam_ab", lambda r: win(r, lambda x: abs(x["seam_ab_mass_rel"]), 0.03)),
                        ("seam_be", lambda r: win(r, lambda x: abs(x["seam_be_mass_rel"]), 0.03)),
                        ("seam_ef", lambda r: win(r, lambda x: abs(x["seam_ef_mass_rel"]), 0.03)),
                        ("seam_df", lambda r: win(r, lambda x: abs(x["seam_df_mass_rel"]), 0.03)),
                        ("seam_gf", lambda r: win(r, lambda x: abs(x["seam_gf_mass_over_outflows"]), 0.03)),
                        ("thermal", lambda r: win(r, lambda x: abs(x["thermal_engine"]["jmax"]["rel"]), 0.03))):
            q = [f(runs[d]) for d in (10.0, 5.0, 2.5)]
            if any(v is None for v in q):                   # no record for v_y(0)
                continue
            p_obs, rich = _order(*q)
            rows[name] = dict(dt10=q[0], dt5=q[1], dt2p5=q[2], order=p_obs, richardson=rich)
        out["coupling_step"] = rows
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=("audit", "control_wall", "control_old", "dt", "grid_engine",
                             "grid_external", "report"))
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--coarsen", type=int, default=4)
    ap.add_argument("--dt-macro", type=float, default=5.0e-3)
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--record", default=os.path.join(ROOT, "out", "w321", "episode.npz"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--revert", default="",
                    help="put fixes' old behaviour back, comma-separated from %s, or 'all'"
                         % ",".join(FIXES))
    a = ap.parse_args(argv)
    if a.stage == "audit":
        a.t_end = a.t_end or 29 * 5.0e-3
        stage_coupled(a, "audit")
    elif a.stage == "control_wall":
        a.t_end = a.t_end or a.dt_macro
        stage_coupled(a, "control_wall", old_wall=True)
    elif a.stage == "control_old":
        # Tier 88's control: every fix put back on today's code, for the first
        # steps of the audited run, to be read against the pre-fix audit
        a.t_end = a.t_end or 2 * a.dt_macro
        a.revert = a.revert or "all"
        a.json = a.json or "control_old.json"
        stage_coupled(a, "control_old")
    elif a.stage == "dt":
        a.t_end = a.t_end or 0.06
        tag = ("%g" % (a.dt_macro * 1e3)).replace(".", "p")
        a.json = a.json or "dt_%sms.json" % tag
        stage_coupled(a, "dt_%sms" % tag)
    elif a.stage in ("grid_engine", "grid_external"):
        a.t_end = a.t_end or 0.03
        stage_grid(a, a.stage.split("_")[1])
    else:
        stage_report(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
