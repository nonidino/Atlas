"""The trained network as an expert the host can call (`host.Layout.solve`).

`LearnedExpert.respond` groups the pieces by resolution, runs the network once per
piece -- one forward pass returns all 4m + 1 fields (SE2: one call per piece per
linear solve) -- and hands the host each piece's interior modes and particular field
in float64, the particular field multiplied back by the piece's source scale (the
network sees the source normalised to unit root mean square, `PieceProblem.inputs`).
The envelope is the card's; the host refuses a piece outside it before any call.
``modes_from`` lets a march keep the modes of an earlier call (fixed geometry and
step: the modes are formed once) while each step's call supplies a new particular
field.

The kernel-exact forms (`forms.KernelExpert`) also read each piece's element weights
``J~`` and its source, scaled as the source channel is, and return a level field ``u_1``
with ``u_p = gbar u_1 + ...``.  A march that keeps the first call's modes keeps its level
field too: the new call's ``gbar u_1`` is replaced by ``gbar u_1`` of the kept call, so
that ``H c1 + s0 u_1 = 1`` holds for the pair actually used (for the split form the two
level fields are the same, bit for bit, and nothing changes).
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402

from . import data  # noqa: E402
from .host import Response  # noqa: E402
from .piece import PieceProblem  # noqa: E402


class LearnedExpert:
    name = "learned"

    def __init__(self, net, envelope=data.ENVELOPE, device: str = "cpu",
                 autocast: bool | None = None, batch: int = 16):
        self.net = net.to(device).eval()
        self.kind, self.m = net.kind, net.m
        self.envelope = envelope
        self.device = torch.device(device)
        self.autocast = (self.device.type == "cuda") if autocast is None else autocast
        self.batch = batch
        self.calls = 0
        self.last_modes = None
        self.reads_load = getattr(net, "needs_load_fields", False)
        self.last_levels = None
        self.last_gbars = None

    def forward(self, problems: list[PieceProblem]) -> list[tuple[np.ndarray, np.ndarray]]:
        """(H ((n-1)^2, 4m), u_p ((n-1)^2,)) per problem, float64, u_p in the problem's
        own source units.  For the kernel-exact forms, ``last_levels`` and ``last_gbars``
        then hold each piece's level field and source level (source units)."""
        out = [None] * len(problems)
        levels = [None] * len(problems) if self.reads_load else None
        gbars = [None] * len(problems) if self.reads_load else None
        by_n: dict[int, list[int]] = {}
        for k, p in enumerate(problems):
            by_n.setdefault(p.n, []).append(k)
        with torch.no_grad():
            for n, ks in by_n.items():
                for i0 in range(0, len(ks), self.batch):
                    kk = ks[i0:i0 + self.batch]
                    ins = [problems[k].inputs() for k in kk]
                    x = torch.tensor(np.stack([c for c, _s, _r in ins]), device=self.device)
                    s = torch.tensor(np.stack([sc for _c, sc, _r in ins]), device=self.device)
                    s0 = torch.tensor([problems[k].s0 for k in kk], dtype=torch.float64,
                                      device=self.device)
                    kw = {}
                    if self.reads_load:
                        # the source as the channel scales it (u_p comes back for g / rms)
                        kw["w"] = torch.tensor(np.stack([problems[k].tensors()["w"] for k in kk]),
                                               dtype=torch.float64, device=self.device)
                        kw["g"] = torch.tensor(np.stack([problems[k].g / (ins[j][2] or 1.0)
                                                         for j, k in enumerate(kk)]),
                                               dtype=torch.float64, device=self.device)
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                        enabled=self.autocast):
                        H, up = self.net(x, s, s0, **kw)
                    H = H.double().cpu().numpy()
                    up = up.double().cpu().numpy()
                    if self.reads_load:
                        u1 = self.net.last_level.double().cpu().numpy()
                        gb = self.net.last_gbar.double().cpu().numpy()
                    for j, k in enumerate(kk):
                        rms = ins[j][2]
                        out[k] = (H[j].reshape(H.shape[1], -1).T.copy(),
                                  up[j].reshape(-1) * rms)
                        if self.reads_load:
                            levels[k] = u1[j].reshape(-1).copy()
                            gbars[k] = float(gb[j]) * rms
        self.calls += len(problems)
        self.last_levels, self.last_gbars = levels, gbars
        return out

    def respond(self, problems: list[PieceProblem], pb_of, modes_from=None) -> list[Response]:
        for p in problems:
            if pb_of(p.n).kind != self.kind:
                raise ValueError(f"the expert answers in the {self.kind} basis, the layout "
                                 f"asks for {pb_of(p.n).kind}")
        fw = self.forward(problems)
        levels = self.last_levels or [None] * len(problems)
        if modes_from is None:
            resp = [Response(H, up, lev) for (H, up), lev in zip(fw, levels)]
        else:
            resp = []
            for i, (_H, up) in enumerate(fw):
                kept = modes_from[i]
                lev = levels[i]
                if lev is not None and kept.level is not None:
                    up = up + self.last_gbars[i] * (kept.level - lev)
                    lev = kept.level
                resp.append(Response(kept.H, up, lev))
        self.last_modes = resp
        return resp


def load(path: str, device: str = "cpu", **kw) -> LearnedExpert:
    from .train import load_expert
    return LearnedExpert(load_expert(path, device), device=device, **kw)
