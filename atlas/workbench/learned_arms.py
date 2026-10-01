"""The learned case's arms (demo item 1.5), on the workbench's own farm machinery.

`learned_gate` names the arms; this module builds them:

  * `FarmRun` is the workbench's `WindFarmRun` with two generalizations the case
    needs and nothing else: the freestream band's width in cells scales with the
    grid (it is 0.25 D wide at every resolution: 8 cells of D/32, 16 of D/64, 4 of
    D/16), and each rotor may have its own induction (the training layouts draw
    it).  With neither, it is `WindFarmRun` to the bit (a test checks it).
  * `scaled_spec` gives a case at another cell size: ``T`` is farm-12 at D/64,
    ``Cc`` at D/16; windows, ramp and band scale with it, rotors are physical.
  * `layout_spec` puts a sampled layout's rotors into farm-12's case.
  * `LearnedFarmRun` adds the ``learned`` arm: the decomposed step with every
    window advanced by the network instead of the classical exposed window, and
    the blend, the global projection and the band exactly as the classical arm.

torch is imported only by `LearnedFarmRun`, when a learned arm is built.
"""

from __future__ import annotations

import numpy as np

from . import learned_gate as G
from .families import windfarm as wf
from .spec import CaseSpec, Device, Domain, Window, example_case

BAND_D = wf.BAND * 1.0 / 32.0          # the band is 0.25 D wide


class FarmRun(wf.WindFarmRun):
    """`WindFarmRun` with the band scaled to the grid and per-rotor induction."""

    def __init__(self, spec, arms=wf.ARMS, threads: int = 4, induction=None):
        super().__init__(spec, arms=arms, threads=threads)
        self.band_cells = max(1, int(round(BAND_D / self.dx)))
        #: rotor id -> induction a; a rotor absent from it keeps the disk's default
        self.induction = dict(induction or {})

    def band(self, u, v):
        b, U = self.band_cells, self.u_inf
        u[:, :b] = U
        v[:, :b] = 0.0
        u[:b, :] = U
        v[:b, :] = 0.0
        u[-b:, :] = U
        v[-b:, :] = 0.0
        u[:, -1] = U
        v[:, -1] = 0.0
        if self.fluid is not None:
            u *= self.fluid
            v *= self.fluid
        return u, v

    def forcing(self, u):
        """`WindFarmRun.forcing`, with each rotor's own induction where it has one."""
        dk = self._disk
        fx = np.zeros((self.ny, self.nx))
        rec: dict[str, float] = {}
        for rot in self.rotors:
            rows = np.abs(self.y_c - rot.y_centre) <= 0.5 * rot.diameter + 1e-9
            i_up = int(np.argmin(np.abs(self.x_c - (rot.x_plane - wf.INFLOW_OFFSET))))
            ud = float(np.mean(u[rows, i_up]))
            rec[rot.rotor_id] = ud
            thick = max(ud, 0.05) * self.dt
            if rot.rotor_id in self.induction:
                disk = dk.ActuatorDisk(area=rot.diameter, thickness=thick,
                                       a=float(self.induction[rot.rotor_id]))
            else:
                disk = dk.ActuatorDisk(area=rot.diameter, thickness=thick)
            st = disk(max(ud, 1e-3))
            fx = fx + disk.body_force_field(self.x_c, self.y_c, self.dx, self.dx, st.thrust,
                                            x0=rot.x_plane - 0.5 * thick,
                                            y0=rot.y_centre - 0.5 * rot.diameter)
        return fx, rec

    def power(self, rec):
        """Farm power with each rotor's own induction (`WindFarmRun.power` otherwise)."""
        if not self.induction:
            return super().power(rec)
        dk = self._disk
        by_id = {r.rotor_id: r for r in self.rotors}
        tot = 0.0
        for k, ud in rec.items():
            kw = {"a": float(self.induction[k])} if k in self.induction else {}
            tot += dk.ActuatorDisk(area=by_id[k].diameter, **kw)(max(float(ud), 1e-6)).power
        return float(tot)

    def window_inputs(self, s, fx=None) -> np.ndarray:
        """Every window's (u - U, v, f), window order: ``[n_windows, 3, h, w]``."""
        t = self.tiling
        if fx is None:
            fx, _rec = self.forcing(s.u)
        return np.stack([np.stack([t.cut_one(s.u, k) - self.u_inf, t.cut_one(s.v, k),
                                   t.cut_one(fx, k)]) for k in range(t.n_windows)])


def scaled_spec(spec: CaseSpec, factor: float) -> CaseSpec:
    """The same case at ``factor`` times the cells per length (2: finer, 0.5: coarser).
    Rotors stay where they are physically; windows and the ramp scale with the grid."""
    def sc(n: int) -> int:
        v = n * factor
        if abs(v - round(v)) > 1e-9:
            raise ValueError("%s cells do not scale by %s" % (n, factor))
        return int(round(v))

    s = spec.copy_deep()
    d = s.domain
    s.domain = Domain(nx=sc(d.nx), ny=sc(d.ny), dx=d.dx / factor)
    s.windows = [Window(id=w.id, x0=sc(w.x0), y0=sc(w.y0), nx=sc(w.nx), ny=sc(w.ny))
                 for w in s.windows]
    s.coupling.ramp_cells = max(1, sc(s.coupling.ramp_cells))
    return s


def layout_spec(layout: G.Layout, base: str = G.CASE) -> tuple[CaseSpec, dict[str, float]]:
    """farm-12's case with the layout's rotors, and each rotor's induction."""
    s = example_case(base)
    s.devices = [Device(id="R%d" % (i + 1), x=x, y=y, diameter=1.0)
                 for i, (x, y, _a) in enumerate(layout.rotors)]
    return s, {"R%d" % (i + 1): a for i, (_x, _y, a) in enumerate(layout.rotors)}


class LearnedFarmRun(FarmRun):
    """`FarmRun` with the ``learned`` arm: every window advanced by ``net``."""

    def __init__(self, spec, net, arms=("learned",), threads: int = 4, induction=None):
        import torch
        classical = tuple(a for a in arms if a in wf.ARMS)
        super().__init__(spec, arms=classical, threads=threads, induction=induction)
        self.arms = tuple(arms)
        self.torch = torch
        self.net = net.eval()
        self.torch_threads = int(threads)

    def step(self, arm: str, s):
        if arm == "learned":
            return self._step_learned(s)
        return super().step(arm, s)

    def _step_learned(self, s):
        torch = self.torch
        if torch.get_num_threads() != self.torch_threads:
            torch.set_num_threads(self.torch_threads)
        t = self.tiling
        fx, rec = self.forcing(s.u)
        x = self.window_inputs(s, fx)
        dev = next(self.net.parameters()).device          # the CPU here, a GPU in training
        with torch.inference_mode():
            y = self.net(torch.from_numpy(x.astype(np.float32)).to(dev)).cpu().numpy()
        y = y.astype(np.float64)
        out_u = [t.cut_one(s.u, k) + y[k, 0] for k in range(t.n_windows)]
        out_v = [t.cut_one(s.v, k) + y[k, 1] for k in range(t.n_windows)]
        au, av = t.assemble(out_u), t.assemble(out_v)
        u1, v1 = self.project(au, av) if self.projected else (au, av)
        u1, v1 = self.band(u1, v1)
        return wf.FarmState(u1, v1, rec, 0, locals_=(out_u, out_v))

    def mass_measure(self, arm: str, s) -> float:
        return super().mass_measure("parallel" if arm == "learned" else arm, s)


__all__ = ["FarmRun", "scaled_spec", "layout_spec", "LearnedFarmRun", "BAND_D"]
