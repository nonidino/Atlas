"""Redundant derived channels, computed before tokenization.

A patch embedding is a linear map over an 8x8 stencil. Asking it to *also* learn
a finite-difference operator spends capacity on something we can supply exactly,
so gradients / divergence / vorticity are concatenated as extra channels even
though they are recoverable from the raw field (phase-1 spec 2.1).

Channel budgets are the spec's table 3.1 and are asserted at construction:

    gas   (rho, u, v, p, T)                       5 raw + 7 derived
    a     (rho, u, v, p, T, Y)                    6 raw + 8 derived
    solid (T, ux, uy, s_zz, s_yy, s_zy)           6 raw + 4 derived

plus one structural ACTIVE-MASK channel on every agent. Agents `b` and `e` are
bounding-box grids with a wall inside them (33% of `e`'s cells sit outside the
nozzle), so a token has to be able to tell "this cell is zero" from "this cell is
not part of my domain". The mask is not in the spec's count because it is not a
physical field.

Derivatives use coordinate differences, not index differences: agent `d`'s grid
is geometrically stretched and agents `c`/`d` are two-panel maps whose y jumps
across the vehicle. Panels are differentiated independently -- a central
difference across the seam would straddle the rocket.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..geometry.domains import AgentSpec

GAS_DERIVED = ("div_u", "vort", "grad_rho", "grad_p", "grad_T", "strain", "speed")
REACTING_DERIVED = GAS_DERIVED + ("grad_Y",)
SOLID_DERIVED = ("grad_T", "von_mises", "mean_stress", "grad_disp")


def _diff(x: Tensor, c: Tensor, dim: int) -> Tensor:
    """Non-uniform first derivative along `dim`: central inside, one-sided at the
    two ends. `c` broadcasts against `x` and holds the coordinate."""
    def sl(a, b):
        idx = [slice(None)] * x.ndim
        idx[dim] = slice(a, b)
        return tuple(idx)

    lo = (x[sl(1, 2)] - x[sl(0, 1)]) / (c[sl(1, 2)] - c[sl(0, 1)])
    mid = (x[sl(2, None)] - x[sl(None, -2)]) / (c[sl(2, None)] - c[sl(None, -2)])
    hi = (x[sl(-1, None)] - x[sl(-2, -1)]) / (c[sl(-1, None)] - c[sl(-2, -1)])
    return torch.cat([lo, mid, hi], dim=dim)


class DerivedFeatures(nn.Module):
    """Per-agent, parameter-free. Holds the agent's coordinate buffers.

    One instance per AGENT (geometry differs), unlike the tokenizer, which is one
    instance per tokenizer group (weights are shared)."""

    def __init__(self, spec: AgentSpec):
        super().__init__()
        self.agent = spec.name
        self.fields = spec.fields
        g = spec.grid
        z = torch.from_numpy(g.z_c).float()[None, None, :, None]      # [1,1,Nz,1]
        y = torch.from_numpy(g.y_c).float()[None, None, :, :]         # [1,1,Nz,Ny]
        self.register_buffer("z_c", z, persistent=False)
        self.register_buffer("y_c", y, persistent=False)
        self.register_buffer("mask", torch.from_numpy(g.active).float()[None, None], persistent=False)

        # panel seams: the shell and the body-fitted atmosphere are two disjoint
        # sheets stacked along the y index.
        half = g.n_y // 2
        self.panels = ((0, half), (half, g.n_y)) if spec.two_panel else ((0, g.n_y),)

        self.is_solid = spec.kind == "solid"
        self.derived = (
            SOLID_DERIVED if self.is_solid
            else REACTING_DERIVED if "Y" in spec.fields
            else GAS_DERIVED
        )
        self.n_out = len(spec.fields) + len(self.derived) + 1   # +1 active mask

    # -- derivative helpers -------------------------------------------------
    def _dz(self, x: Tensor) -> Tensor:
        return _diff(x, self.z_c, dim=-2)

    def _dy(self, x: Tensor) -> Tensor:
        return torch.cat(
            [_diff(x[..., s:e], self.y_c[..., s:e], dim=-1) for s, e in self.panels], dim=-1
        )

    def _grad_mag(self, s: Tensor) -> Tensor:
        return torch.sqrt(self._dz(s) ** 2 + self._dy(s) ** 2 + 1e-12)

    # -- forward ------------------------------------------------------------
    def forward(self, fields: Tensor) -> Tensor:
        """fields [B, C_raw, Nz, Ny] -> [B, C_total, Nz, Ny] (raw ‖ derived ‖ mask)."""
        f = {name: fields[:, i:i + 1] for i, name in enumerate(self.fields)}
        if self.is_solid:
            szz, syy, szy = f["s_zz"], f["s_yy"], f["s_zy"]
            vm = torch.sqrt(szz ** 2 - szz * syy + syy ** 2 + 3.0 * szy ** 2 + 1e-12)
            disp = torch.sqrt(
                self._dz(f["ux"]) ** 2 + self._dy(f["ux"]) ** 2
                + self._dz(f["uy"]) ** 2 + self._dy(f["uy"]) ** 2 + 1e-12
            )
            der = [self._grad_mag(f["T"]), vm, 0.5 * (szz + syy), disp]
        else:                                              # any gas agent
            u, v = f["u"], f["v"]
            uz, uy = self._dz(u), self._dy(u)
            vz, vy = self._dz(v), self._dy(v)
            der = [
                uz + vy,                                                   # div u
                vz - uy,                                                   # vorticity
                self._grad_mag(f["rho"]),
                self._grad_mag(f["p"]),
                self._grad_mag(f["T"]),
                torch.sqrt(uz ** 2 + uy ** 2 + vz ** 2 + vy ** 2 + 1e-12),  # |J_u|_F
                torch.sqrt(u ** 2 + v ** 2 + 1e-12),                       # speed
            ]
            if "Y" in f:
                der.append(self._grad_mag(f["Y"]))

        mask = self.mask.expand(fields.shape[0], -1, -1, -1)
        out = torch.cat([fields] + der + [mask], dim=1)
        if out.shape[1] != self.n_out:
            raise AssertionError(f"{self.agent}: {out.shape[1]} channels, expected {self.n_out}")
        return out


__all__ = ["DerivedFeatures", "GAS_DERIVED", "REACTING_DERIVED", "SOLID_DERIVED"]
