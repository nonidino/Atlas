"""The frozen fluid expert and the nondimensional adapter that talks to it.

Phase W0 of the wind-farm implementation guide. Everything about the *outside*
world -- the checkpoint, its normalization, its native timestep, its fixed
viscosity -- is confined to this file, so that `geometry.py`, `disk.py` and
`couple.py` never see a normalization constant.

The expert
----------
`camlab-ethz/Poseidon-T` (scOT, 20.8 M parameters), the first of the two
candidates the guide lists in priority order. Pretrained on NS-Sines (2D
incompressible Navier-Stokes), CE-KH and CE-Gauss on the periodic unit square
with final time 1 and 21 stored snapshots. It is loaded frozen and never
trained; `FrozenFluidExpert` calls `.eval()` and wraps every forward in
`torch.no_grad()` so that a gradient cannot be taken by accident.

Four facts about the checkpoint drive everything below, and three of them are
*not* free parameters of this case study:

1. **Four channels, in the order (rho, u_x, u_y, p)**, each normalized by the
   constants in `scOT/problems/fluids/normalization_constants.py`. For the
   incompressible datasets the density and pressure channels are *placeholders*
   that the loader fills with the exact constants 1 and 0 -- so they are not
   state, and `step()` rewrites them every call rather than letting the model's
   own output for them feed back.
2. **The u and v channels have different standard deviations** (0.391 vs 0.356).
   The expert therefore does not see an isotropic velocity field, and a code
   path that normalized both by the same number would be feeding it a subtly
   sheared world. This is reproduced exactly.
3. **The native lead time is 0.1.** scOT is time-conditioned and was trained
   all2all on lead times `t/20` for `t` in `{0, 2, 4, ..., 14}`, i.e. on
   `{0, 0.1, ..., 0.7}`. Nothing below 0.1 (other than the identity) was ever
   trained, which is exactly the constraint guide section 7.2 says cannot be
   worked around by shrinking the coupling step.
4. **Its viscosity is fixed.** A pretrained operator has whatever viscosity its
   training data had; it is a property of the weights, not a dial. Measured at
   W0 by `scripts/windfarm_w0_bringup.py` from the decay of an exact 2D
   Taylor-Green vortex.

Installing the loader
---------------------
scOT's own `pyproject.toml` pins `torch == 2.0.1` and `transformers == 4.29.2`,
which would drag this project's torch backwards. Install it without its
dependency set:

    pip install "transformers>=4.35,<5"
    pip install --no-deps "git+https://github.com/camlab-ethz/poseidon.git"

The checkpoint (~85 MB) is fetched from HuggingFace on first construction and
cached.

The scaling law, and why it is nearly forced
--------------------------------------------
Write the map between our nondimensional variables (D = U_inf = rho = 1) and
the expert's own as

    x = L x_p,      u = U_s u_p,      t = T_s t_p,      T_s = L / U_s.

Substituting into the incompressible Navier-Stokes equations gives
`nu_ours = nu_p L U_s`, hence

    Re_eff = U_inf D / nu_ours = 1 / (nu_p L U_s) = T_s / (nu_p L^2).

Three requirements then pin all three scales:

* one macro-step must be one *native* expert step, so `T_s = dt_macro / 0.1 = 0.5`;
* the freestream must land inside the expert's velocity distribution
  (`std_u = 0.391`), so `u_p(U_inf) = 1/U_s ~ 0.5`, giving `U_s = 2`;
* hence `L = T_s U_s = 1.0` -- one rotor diameter per 128x128 window.

`Re_eff` is then **not a dial**: it is `0.5 / nu_p`, whatever that turns out to
be. The spec asks for `Re_eff` in `[1e3, 1e4]`; whether the checkpoint delivers
that is a measurement, and `Scaling.check()` states it rather than assuming it.

The consequence worth naming, because it is a real result about composing frozen
experts: `Re_eff = T_s / (nu_p L^2)` means **the window size is not free**.
Doubling the physical size of the region one window covers quarters the effective
Reynolds number. Covering the 24 x 8 domain in fewer, larger windows is not a
performance tuning knob -- it changes the physics being solved.

The Galilean decomposition, and why it is not a workaround
----------------------------------------------------------
W0 experiment E1 measured that the checkpoint **does not preserve a uniform
flow**, which is an exact steady solution of the periodic incompressible
equations at any viscosity: fed `u = (1, 0)`, one step returns a mean of 0.969,
and forty steps leave 0.281. It is pulling the field toward zero mean, which is
what its pretraining distribution (NS-Sines, zero mean flow by construction)
contains. A wind farm is nothing but mean flow, so used naively this checkpoint
cannot do the case study at all.

The fix is a change of frame, not a fudge. For a *constant* mean velocity `U`,
`u(x, t) = U + u'(x - U t, t)` where `u'` solves the same equations with zero
mean -- the transformation is exact, not a first-order splitting, because the
Navier-Stokes equations are Galilean invariant and `U` does not vary in space.
`step()` therefore

  1. subtracts the window-mean velocity,
  2. asks the expert to advance the (zero-mean) fluctuation,
  3. re-zeroes the fluctuation mean and records what the expert tried to do to
     it as `last_mean_drift` -- for periodic unforced flow the mean is exactly
     conserved, so any drift is model error and is reported, not absorbed,
  4. translates the result by `U dt` (exact, spectral), which *is* the mean
     advection term `U . grad u'` that step 2 left out,
  5. adds `U` back, plus any explicit `force` impulse.

Step 4 is where interface data enters once the window stops being periodic: the
translation pulls in material from upstream, which in the coupled system is
supplied by the neighbouring agent. That makes the Galilean split the natural
place for the R1-halo / R2-flux-BC decision of W3 to land, rather than an extra
mechanism bolted next to it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# checkpoint facts -- transcribed from scOT, not guessed
# --------------------------------------------------------------------------

CHECKPOINT = "camlab-ethz/Poseidon-T"

#: `scOT.problems.fluids.normalization_constants.CONSTANTS`, channel order
#: (rho, u_x, u_y, p). The incompressible loaders use all four with rho == 1 and
#: p == 0 held fixed.
EXPERT_MEAN = np.array([0.80, 0.0, 0.0, 0.0], dtype=np.float64)
EXPERT_STD = np.array([0.31, 0.391, 0.356, 0.185], dtype=np.float64)

EXPERT_RES = 128                 # scOT image_size; the checkpoint is not resolution-free
EXPERT_RHO = 1.0                 # placeholder channel value used by the incompressible loaders
EXPERT_P = 0.0

#: Lead times the checkpoint was trained on: `t / 20` for `t` in `{0, 2, ..., 14}`
#: (`BaseTimeDataset.post_init` with `time_step_size=2`, `max_num_time_steps=7`).
EXPERT_TRAINED_LEADS = tuple(round(2 * k / 20.0, 3) for k in range(0, 8))
EXPERT_NATIVE_DT = 0.1           # smallest trained non-identity lead time
EXPERT_MAX_DT = 0.7


# --------------------------------------------------------------------------
# nondimensional scaling
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Scaling:
    """Map between case-study units (D = U_inf = 1) and the expert's own.

    length   rotor diameters D spanned by one expert length unit -- i.e. by one
             128x128 window
    velocity multiples of U_inf per expert velocity unit
    """
    length: float = 1.0
    velocity: float = 2.0

    @property
    def time(self) -> float:
        """Our convective time units per expert time unit."""
        return self.length / self.velocity

    @property
    def dx(self) -> float:
        """Window cell size, in D."""
        return self.length / EXPERT_RES

    def lead(self, dt: float) -> float:
        """Expert lead time for a macro-step `dt` measured in our time units."""
        return dt / self.time

    def re_eff(self, nu_expert: float) -> float:
        """Effective Reynolds number this scaling extracts from the checkpoint."""
        return 1.0 / (nu_expert * self.length * self.velocity)

    def check(self, dt_macro: float, nu_expert: float | None = None) -> dict:
        """Report -- never silently assume -- how this scaling sits against the
        checkpoint's trained envelope and the spec's Re band."""
        lead = self.lead(dt_macro)
        out = {
            "lead": lead,
            "lead_is_native": bool(abs(lead - EXPERT_NATIVE_DT) < 1e-9),
            "lead_below_native": bool(lead < EXPERT_NATIVE_DT - 1e-9),
            "lead_above_max": bool(lead > EXPERT_MAX_DT + 1e-9),
            "u_inf_normalized": float(1.0 / self.velocity / EXPERT_STD[1]),
            "dx_over_D": self.dx,
        }
        if nu_expert is not None:
            re = self.re_eff(nu_expert)
            out["re_eff"] = re
            out["re_in_spec_band"] = bool(1e3 <= re <= 1e4)
        return out


#: The W0 scaling: one macro-step of 0.05 is exactly one native expert step of
#: 0.1, and one 128x128 window covers one rotor diameter.
WINDFARM_SCALING = Scaling(length=1.0, velocity=2.0)


# --------------------------------------------------------------------------
# spectral helpers -- valid on the periodic window the expert was trained on
# --------------------------------------------------------------------------


def _wavenumbers(n: int, length: float, nyquist_zero: bool = True):
    """Fourier wavenumbers on the periodic window.

    `nyquist_zero` is the standard treatment for *differentiation* of a real
    field: the Nyquist coefficient of a real signal is real, multiplying it by
    `i k` makes it imaginary, and `np.real(ifft2(...))` then silently discards
    it -- so the divergence operator and the projection that is supposed to
    cancel it disagree at exactly one mode. Left in, that single mode kept the
    "exact" Leray projection at a divergence of 1.3 instead of 1e-13 on the
    first W0 run. Zero it once, here, so every spectral operator in this file
    uses the same one."""
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    if nyquist_zero and n % 2 == 0:
        k[n // 2] = 0.0
    return k


def spectral_shift(f: np.ndarray, dx: float, dy: float, length: float = 1.0) -> np.ndarray:
    """Translate a periodic field by (dx, dy) exactly, in Fourier space.

    `dx` moves content toward increasing last-axis index. Used for the Galilean
    mean-flow transport of `FrozenFluidExpert.step`; the Nyquist mode is kept
    here (multiplied by `cos(k dx)`, the usual real-signal convention) rather
    than zeroed, because this is an interpolation, not a derivative."""
    n = f.shape[-1]
    kx = _wavenumbers(n, length, nyquist_zero=False)[None, :]
    ky = _wavenumbers(n, length, nyquist_zero=False)[:, None]
    fh = np.fft.fft2(f) * np.exp(-1j * (kx * dx + ky * dy))
    return np.real(np.fft.ifft2(fh))


def divergence(u: np.ndarray, v: np.ndarray, length: float = 1.0) -> np.ndarray:
    """Spectral divergence on the periodic square. `u` is the velocity along the
    LAST array axis, `v` along the second-to-last (see `AXIS_NOTE`)."""
    n = u.shape[-1]
    kx = _wavenumbers(n, length)[None, :]
    ky = _wavenumbers(n, length)[:, None]
    uh = np.fft.fft2(u)
    vh = np.fft.fft2(v)
    return np.real(np.fft.ifft2(1j * kx * uh + 1j * ky * vh))


def leray_project(u: np.ndarray, v: np.ndarray, length: float = 1.0):
    """Exact Leray projection on the periodic square: subtract the gradient part
    in Fourier space. This is the level-4 fallback of guide section 1.2 -- the
    checkpoint decodes velocity directly and has no stream-function head, so mass
    conservation cannot be had at level 5 by construction.

    On a *periodic* window this costs one FFT pair and is exact to floating
    point, which is much cheaper than the "Poisson solve per agent per step" the
    guide budgets for. That discount does not survive to a non-periodic agent
    boundary, and W5 must not assume it does."""
    n = u.shape[-1]
    kx = _wavenumbers(n, length)[None, :]
    ky = _wavenumbers(n, length)[:, None]
    k2 = kx * kx + ky * ky
    # k=0 is the mean flow, which the projection must leave alone; the Nyquist
    # row and column are zeroed by `_wavenumbers`, so they land here too and
    # would divide by zero.
    k2 = np.where(k2 == 0.0, 1.0, k2)
    uh, vh = np.fft.fft2(u), np.fft.fft2(v)
    div = kx * uh + ky * vh
    uh -= kx * div / k2
    vh -= ky * div / k2
    return np.real(np.fft.ifft2(uh)), np.real(np.fft.ifft2(vh))


#: Which array axis the expert's first velocity channel runs along. scOT's
#: `IncompressibleBase` reads `solution[..., 0:2]` and reshapes to (2, 128, 128)
#: with a per-dataset `transpose` flag, so the mapping from "horizontal" to an
#: array axis is a property of the data files rather than something the model
#: card fixes. W0 experiment E2 measures it by advecting a vortex in a uniform
#: stream and watching which way it goes; do not assume it.
AXIS_NOTE = "channel 1 = velocity along the last array axis (x); verified by W0/E2"


# --------------------------------------------------------------------------
# the frozen expert
# --------------------------------------------------------------------------


class FrozenFluidExpert:
    """Poseidon-T, frozen, wrapped so callers work in case-study units.

    Nothing here trains, and nothing here is allowed to: the module holds no
    optimizer, `requires_grad_(False)` is applied to every parameter, and every
    forward runs under `torch.no_grad()`."""

    def __init__(
        self,
        scaling: Scaling = WINDFARM_SCALING,
        checkpoint: str = CHECKPOINT,
        device: str = "cpu",
        threads: int | None = None,
        galilean: bool = True,
    ):
        import torch
        from scOT.model import ScOT

        if threads is not None:
            torch.set_num_threads(threads)
        self._torch = torch
        self.scaling = scaling
        self.galilean = galilean
        self.last_mean_drift = (0.0, 0.0)
        self.checkpoint = checkpoint
        self.device = device
        self.model = ScOT.from_pretrained(checkpoint).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))
        self.n_calls = 0
        if self.model.config.image_size != EXPERT_RES:
            raise AssertionError(
                f"checkpoint image_size {self.model.config.image_size} != {EXPERT_RES}; "
                "the window geometry in geometry.py assumes 128"
            )

    # -- unit conversion ---------------------------------------------------

    def _encode(self, u: np.ndarray, v: np.ndarray):
        """(u, v) in our units -> normalized 4-channel tensor the model expects."""
        s = self.scaling
        fields = np.stack([
            np.full_like(u, EXPERT_RHO),
            u / s.velocity,
            v / s.velocity,
            np.full_like(u, EXPERT_P),
        ])
        norm = (fields - EXPERT_MEAN[:, None, None]) / EXPERT_STD[:, None, None]
        return self._torch.from_numpy(norm[None].astype(np.float32)).to(self.device)

    def _decode(self, out) -> tuple[np.ndarray, np.ndarray]:
        arr = out.detach().to("cpu").numpy()[0].astype(np.float64)
        fields = arr * EXPERT_STD[:, None, None] + EXPERT_MEAN[:, None, None]
        s = self.scaling
        return fields[1] * s.velocity, fields[2] * s.velocity

    # -- the forward -------------------------------------------------------

    def step(self, u: np.ndarray, v: np.ndarray, dt: float, project: bool = False,
             galilean: bool | None = None, force: tuple | None = None):
        """Advance a 128x128 velocity window by `dt` in OUR time units.

        `dt` is converted to the expert's lead time by the scaling; a lead time
        outside the trained envelope raises rather than extrapolating silently,
        because "just take a smaller step" is precisely the remedy that is
        unavailable with a frozen expert."""
        if u.shape != (EXPERT_RES, EXPERT_RES) or v.shape != u.shape:
            raise ValueError(f"expected two {EXPERT_RES}x{EXPERT_RES} fields, got {u.shape}, {v.shape}")
        lead = self.scaling.lead(dt)
        if lead > EXPERT_MAX_DT + 1e-9:
            raise ValueError(
                f"lead time {lead:.4f} exceeds the largest trained lead {EXPERT_MAX_DT}"
            )
        if lead < EXPERT_NATIVE_DT - 1e-9 and lead > 1e-12:
            raise ValueError(
                f"lead time {lead:.4f} is below the expert's native {EXPERT_NATIVE_DT}. "
                "Sub-native steps are out of distribution; see guide section 7.2 -- the "
                "responses are halo exchange, IQN-ILS, or a negative result, not a "
                "smaller step. Pass through `step_unchecked` if measuring this on purpose."
            )
        return self.step_unchecked(u, v, dt, project=project, galilean=galilean, force=force)

    def step_unchecked(self, u: np.ndarray, v: np.ndarray, dt: float, project: bool = False,
                       galilean: bool | None = None, force: tuple | None = None):
        """`step` without the trained-envelope guard, for experiments whose whole
        point is to probe outside it."""
        torch = self._torch
        L = self.scaling.length
        lead = self.scaling.lead(dt)
        gal = self.galilean if galilean is None else galilean

        ubar, vbar = float(np.mean(u)), float(np.mean(v))
        uf, vf = (u - ubar, v - vbar) if gal else (u, v)

        x = self._encode(np.ascontiguousarray(uf), np.ascontiguousarray(vf))
        t = torch.tensor([lead], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(pixel_values=x, time=t).output
        self.n_calls += 1
        u1, v1 = self._decode(out)

        if gal:
            # the expert has advanced the fluctuation in the frame moving with
            # (ubar, vbar); carry it back to the lab frame by translating.
            self.last_mean_drift = (float(np.mean(u1)), float(np.mean(v1)))
            u1 = u1 - self.last_mean_drift[0]
            v1 = v1 - self.last_mean_drift[1]
            u1 = spectral_shift(u1, ubar * dt, vbar * dt, L)
            v1 = spectral_shift(v1, ubar * dt, vbar * dt, L)
            du, dv = (0.0, 0.0) if force is None else (force[0] * dt, force[1] * dt)
            u1 = u1 + ubar + du
            v1 = v1 + vbar + dv
        else:
            self.last_mean_drift = (float(np.mean(u1)) - ubar, float(np.mean(v1)) - vbar)

        if project:
            u1, v1 = leray_project(u1, v1, L)
        return u1, v1


    # -- batched forward ---------------------------------------------------

    def step_many(self, u: np.ndarray, v: np.ndarray, dt: float, project: bool = False,
                  galilean: bool | None = None, force=None, chunk: int = 32,
                  frame: tuple | None = None):
        """Advance a stack of windows at once. `u`, `v` are [B, 128, 128].

        Worth having rather than looping: on CPU the checkpoint costs 0.117 s for
        one window and 0.022 s per window at batch 32, a 5.3x difference that
        comes entirely from threading and cache reuse. The eight-agent system
        needs ~70 windows per macro-step, so this is the difference between a
        seven-minute rollout and an hour one.

        `frame` sets the Galilean frame velocity shared by every window. Pass it
        whenever the windows are tiles of one larger field: the frame has to be
        spatially constant or the transformation is not Galilean. Left as None,
        each window uses its own mean, which is right only for a window that
        stands alone."""
        torch = self._torch
        L = self.scaling.length
        lead = self.scaling.lead(dt)
        gal = self.galilean if galilean is None else galilean
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        if u.ndim != 3 or u.shape[1:] != (EXPERT_RES, EXPERT_RES):
            raise ValueError(f"expected [B, {EXPERT_RES}, {EXPERT_RES}], got {u.shape}")
        B = u.shape[0]

        if not gal:
            ubar = vbar = np.zeros(B)
        elif frame is not None:
            # ONE frame for every window. A Galilean transformation is exact only
            # for a *spatially constant* frame velocity; giving each tile its own
            # window mean makes the frame vary in space, which is not a Galilean
            # transformation at all. In isolation that is invisible -- one window
            # has one mean -- but assembling overlapping tiles that were each
            # shifted by a different amount tears the seams, and the tear grows.
            # Measured: the turbine-free control-volume residual went from 4.8e-4
            # at t = 0.25 to 0.37 by t = 5 before this was made a single frame.
            ubar = np.full(B, float(frame[0]))
            vbar = np.full(B, float(frame[1]))
        else:
            ubar = u.mean(axis=(1, 2))
            vbar = v.mean(axis=(1, 2))
        uf = u - ubar[:, None, None]
        vf = v - vbar[:, None, None]

        s = self.scaling
        fields = np.stack([
            np.full_like(uf, EXPERT_RHO), uf / s.velocity, vf / s.velocity,
            np.full_like(uf, EXPERT_P),
        ], axis=1)                                          # [B, 4, H, W]
        norm = (fields - EXPERT_MEAN[None, :, None, None]) / EXPERT_STD[None, :, None, None]

        outs = []
        for i in range(0, B, chunk):
            x = torch.from_numpy(norm[i:i + chunk].astype(np.float32)).to(self.device)
            t = torch.full((x.shape[0],), lead, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                outs.append(self.model(pixel_values=x, time=t).output.detach().cpu().numpy())
            self.n_calls += x.shape[0]
        arr = np.concatenate(outs, axis=0).astype(np.float64)
        arr = arr * EXPERT_STD[None, :, None, None] + EXPERT_MEAN[None, :, None, None]
        u1 = arr[:, 1] * s.velocity
        v1 = arr[:, 2] * s.velocity

        if gal:
            drift_u = u1.mean(axis=(1, 2))
            drift_v = v1.mean(axis=(1, 2))
            self.last_mean_drift = (float(np.abs(drift_u).max()), float(np.abs(drift_v).max()))
            u1 -= drift_u[:, None, None]
            v1 -= drift_v[:, None, None]
            for b in range(B):
                u1[b] = spectral_shift(u1[b], ubar[b] * dt, vbar[b] * dt, L)
                v1[b] = spectral_shift(v1[b], ubar[b] * dt, vbar[b] * dt, L)
            u1 += ubar[:, None, None]
            v1 += vbar[:, None, None]
            if force is not None:
                u1 += np.asarray(force[0], dtype=np.float64) * dt
                v1 += np.asarray(force[1], dtype=np.float64) * dt

        if project:
            for b in range(B):
                u1[b], v1[b] = leray_project(u1[b], v1[b], L)
        return u1, v1



__all__ = [
    "CHECKPOINT", "EXPERT_MEAN", "EXPERT_STD", "EXPERT_RES", "EXPERT_RHO", "EXPERT_P",
    "EXPERT_TRAINED_LEADS", "EXPERT_NATIVE_DT", "EXPERT_MAX_DT", "AXIS_NOTE",
    "Scaling", "WINDFARM_SCALING", "FrozenFluidExpert",
    "divergence", "leray_project",
]
