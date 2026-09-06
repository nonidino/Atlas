"""The top-level Atlas module and its state container.

Forward pass (phase-1 spec 3.6):

    tokens   = per-agent tokenizer(fields, cond)          # no cross-agent mixing
    graph    = build_graph(layout, edges)                 # memoized, deterministic
    tokens   = typed message passing over the graph       # n_mp_layers
    tokens   = expert[agent](tokens)                      # identity in Phase 1
    zeta     = hierarchy.pool(tokens) (+ rigid, gravity)
    zeta     = rigid_body expert(zeta)                    # closed form in Phase 3
    tokens   = hierarchy.unpool(tokens, zeta)
    fields   = fields + dt * decoder(tokens)              # INCREMENT

Absent, on purpose: the conservation constraint at the two conservation-typed
edges and multi-rate subcycling. Both are Phase 3
(impl-atlas-0.1-phase3-integration). `dt` here is one uniform step; the per-agent
dt_model table is carried on the AgentSpecs and unused until then.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from ..config import AtlasConfig, load_config
from ..geometry.layout import PatchLayout, build_layout
from .decoder import AgentDecoder
from .edges import Graph, build_graph
from .experts import FIELD_EXPERTS, RIGID_EXPERT
from .features import DerivedFeatures
from .hierarchy import TwoLevelHierarchy
from .message_passing import TypedMessagePassing
from .tokenizer import AgentTokenizer


@dataclass
class AtlasState:
    """One instant of the whole vehicle.

    fields   agent -> [B, C_raw, Nz, Ny]  nondimensionalized (master plan)
    cond     agent -> [B, n_cond]         dimensionless scalars for FiLM
    rigid    [B, n_rigid]                 whole-vehicle rigid-body state
    gravity  [B, 1]                       uniform field; bypasses the edge mechanism
    """
    fields: dict[str, Tensor]
    cond: dict[str, Tensor]
    rigid: Tensor
    gravity: Tensor
    t: float = 0.0

    @property
    def batch(self) -> int:
        return int(self.rigid.shape[0])

    def to(self, device) -> "AtlasState":
        return AtlasState(
            {k: v.to(device) for k, v in self.fields.items()},
            {k: v.to(device) for k, v in self.cond.items()},
            self.rigid.to(device), self.gravity.to(device), self.t,
        )

    def detach(self) -> "AtlasState":
        return AtlasState(
            {k: v.detach() for k, v in self.fields.items()},
            {k: v.detach() for k, v in self.cond.items()},
            self.rigid.detach(), self.gravity.detach(), self.t,
        )

    def replace_fields(self, fields: dict[str, Tensor], t: float) -> "AtlasState":
        return replace(self, fields=fields, t=t)


class Atlas(nn.Module):
    def __init__(self, cfg: AtlasConfig | None = None, layout: PatchLayout | None = None):
        super().__init__()
        self.cfg = cfg = cfg if cfg is not None else load_config()
        self.layout = layout = layout if layout is not None else build_layout(cfg)
        m = cfg.model

        # per-AGENT geometry (parameter-free), per-GROUP weights.
        self.derive = nn.ModuleDict({a.name: DerivedFeatures(a) for a in layout.agents})
        groups: dict[str, tuple] = {}
        for a in layout.agents:
            g = groups.setdefault(a.tokenizer, (self.derive[a.name].n_out, a.patch,
                                                len(a.fields), len(a.cond_names)))
            if g != (self.derive[a.name].n_out, a.patch, len(a.fields), len(a.cond_names)):
                raise ValueError(
                    f"tokenizer group {a.tokenizer!r} is not homogeneous: agent "
                    f"{a.name} disagrees with an earlier member on channels/patch/cond"
                )
        self.tok = nn.ModuleDict({
            name: AgentTokenizer(c_tot, patch, m.d_model, m.n_fourier, n_cond)
            for name, (c_tot, patch, _, n_cond) in groups.items()
        })
        self.dec = nn.ModuleDict({
            name: AgentDecoder(c_raw, patch, m.d_model, m.decoder_out_scale)
            for name, (_, patch, c_raw, _) in groups.items()
        })

        self.mp = TypedMessagePassing(cfg)
        self.experts = nn.ModuleDict({
            fam: ctor() for fam, ctor in FIELD_EXPERTS.items()
        })
        self.rigid_expert = RIGID_EXPERT()
        self.hier = TwoLevelHierarchy(cfg, layout)

        # static, config-derived indices as buffers so `.to(device)` moves them.
        for a in layout.agents:
            self.register_buffer(f"keep_{a.name}", layout.keep_index[a.name], persistent=False)
        self.register_buffer("centroids_norm", layout.centroids_norm, persistent=False)
        self._graph: Graph | None = None

    # -- graph --------------------------------------------------------------
    def graph(self) -> Graph:
        """Memoized on the layout fingerprint; moved to the parameter device once."""
        if self._graph is None or self._graph.edge_index.device != self.centroids_norm.device:
            self._graph = build_graph(self.layout, self.cfg).to(self.centroids_norm.device)
        return self._graph

    # -- pieces -------------------------------------------------------------
    def tokenize(self, state: AtlasState) -> Tensor:
        """[B, P_total, d]. Per-agent and independent -- the first cross-agent
        interaction in the whole model is the edge layer."""
        parts = []
        for a in self.layout.agents:
            x = self.derive[a.name](state.fields[a.name])
            parts.append(self.tok[a.tokenizer](
                x, getattr(self, f"keep_{a.name}"),
                self.centroids_norm[self.layout.agent_slice[a.name]],
                state.cond[a.name],
            ))
        return torch.cat(parts, dim=1)

    def apply_experts(self, tokens: Tensor, state: AtlasState) -> Tensor:
        out = tokens.clone()
        for a in self.layout.agents:
            sl = self.layout.agent_slice[a.name]
            out[:, sl] = self.experts[a.expert](tokens[:, sl], state.cond[a.name], a.name)
        return out

    def decode(self, tokens: Tensor, state: AtlasState, dt: float) -> dict[str, Tensor]:
        fields = {}
        for a in self.layout.agents:
            sl = self.layout.agent_slice[a.name]
            inc = self.dec[a.tokenizer](
                tokens[:, sl], getattr(self, f"keep_{a.name}"), a.n_patch,
                (a.grid.n_z, a.grid.n_y),
            )
            fields[a.name] = state.fields[a.name] + dt * inc
        return fields

    # -- forward ------------------------------------------------------------
    def forward(self, state: AtlasState, dt: float | None = None,
                use_bottleneck: bool | None = None) -> AtlasState:
        dt = self.cfg.dt_macro if dt is None else dt
        use_bn = self.cfg.model.use_bottleneck if use_bottleneck is None else use_bottleneck

        tokens = self.tokenize(state)
        tokens = self.mp(tokens, self.graph())
        tokens = self.apply_experts(tokens, state)
        if use_bn:
            _, pooled = self.hier.pool(tokens, self.layout)
            zeta = self.hier.bottleneck(pooled, state.rigid, state.gravity)
            zeta = self.rigid_expert(zeta, state.rigid, dt)
            tokens = self.hier.unpool(tokens, zeta)
        return state.replace_fields(self.decode(tokens, state, dt), state.t + dt)

    # -- helpers ------------------------------------------------------------
    def zero_decoders_(self) -> None:
        for d in self.dec.values():
            d.zero_output_()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def random_state(model: Atlas, batch: int = 2, seed: int = 0,
                 device: str | torch.device = "cpu") -> AtlasState:
    """Random-but-plausible nondimensional state, for shape and gradient tests.

    Fields are O(1) around a mean of 1 (which is what nondimensionalization by a
    per-episode reference scale is supposed to produce), never raw SI."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    fields, cond = {}, {}
    for a in model.layout.agents:
        fields[a.name] = (1.0 + 0.1 * torch.randn(
            batch, len(a.fields), a.grid.n_z, a.grid.n_y, generator=g)).to(device)
        cond[a.name] = torch.rand(batch, len(a.cond_names), generator=g).to(device)
    return AtlasState(
        fields=fields, cond=cond,
        rigid=torch.randn(batch, model.cfg.model.n_rigid, generator=g).to(device),
        gravity=torch.full((batch, 1), 9.81 / 9.81).to(device),
    )


__all__ = ["Atlas", "AtlasState", "random_state"]
