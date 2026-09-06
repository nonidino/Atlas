"""The 2-level U-Net: token level and a single whole-vehicle bottleneck.

Stage 1 uses two levels, not four (unet-hierarchy-atlas-0.1). Seven agents do not
justify agent-region and per-agent intermediate levels, and each extra level is
failure surface that makes a first end-to-end test hard to diagnose.

    {Z_a}_1..7  --attention pool-->  {Zbar_a}  --aggregate-->  zeta

Pooling is attention-weighted, not mean, so a small high-gradient region (the
throat, a shock) can dominate its agent's summary. Tokens are additionally
biased by their ACTIVE FRACTION: agents `b` and `e` are bounding-box grids with a
wall inside them, and a token covering no domain must not win the pool.

Unpooling broadcasts zeta back by FiLM plus a skip:

    Z_a^i  <-  Z_a^i + MLP( gamma(zeta) * Z_a^i + beta(zeta) )

The skip is what stops the global correction from erasing local detail -- a
whole-vehicle acceleration should shift a boundary-layer profile, not overwrite
it. The residual form also means an untrained hierarchy is the identity.

**The bottleneck is a deliberate all-to-all path.** Gravity and the rigid-body
state enter here (global-fields-and-topology-atlas-0.1: gravity bypasses the edge
mechanism entirely), so every agent reaches every other through it in one step.
Agent isolation is therefore a property of the token-level graph, not of the full
forward pass; `use_bottleneck=False` is what the isolation test runs against.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import AtlasConfig
from ..geometry.layout import PatchLayout
from .utils import near_zero_


class TwoLevelHierarchy(nn.Module):
    def __init__(self, cfg: AtlasConfig, layout: PatchLayout):
        super().__init__()
        d = cfg.model.d_model
        n_agents = len(layout.agents)
        self.slices = [layout.agent_slice[a.name] for a in layout.agents]
        self.token_w = nn.Parameter(torch.zeros(n_agents, d))
        self.agent_w = nn.Parameter(torch.zeros(d))
        self.rigid_proj = nn.Linear(d + cfg.model.n_rigid + 1, d)
        self.film = nn.Linear(d, 2 * d)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        near_zero_(self.film)          # near-identity FiLM at init...
        near_zero_(self.mlp[-1])       # ...and a near-zero unpool correction
        # log(active fraction): a token covering nothing is 1e-3 of a full one in
        # the pool, rather than competing with it on equal terms.
        self.register_buffer(
            "mask_bias", torch.log(layout.active_frac.clamp_min(1e-3)), persistent=False
        )

    def pool(self, tokens: Tensor, layout: PatchLayout) -> tuple[Tensor, Tensor]:
        """[B, P, d] -> (agent_vecs [B, n_agents, d], pooled [B, d])."""
        vecs = []
        for k, sl in enumerate(self.slices):
            z = tokens[:, sl]                                     # [B, P_a, d]
            logit = z @ self.token_w[k] + self.mask_bias[sl][None]
            vecs.append((torch.softmax(logit, dim=1)[..., None] * z).sum(1))
        agent_vecs = torch.stack(vecs, dim=1)                     # [B, n_agents, d]
        w = torch.softmax(agent_vecs @ self.agent_w, dim=1)       # [B, n_agents]
        return agent_vecs, (w[..., None] * agent_vecs).sum(1)

    def bottleneck(self, pooled: Tensor, rigid: Tensor, gravity: Tensor) -> Tensor:
        """Fuse the field summary with the rigid-body state and the uniform field."""
        return self.rigid_proj(torch.cat([pooled, rigid, gravity], dim=-1))

    def unpool(self, tokens: Tensor, zeta: Tensor) -> Tensor:
        gamma, beta = self.film(zeta).chunk(2, dim=-1)
        mod = (1.0 + gamma[:, None, :]) * tokens + beta[:, None, :]
        return tokens + self.mlp(mod)


__all__ = ["TwoLevelHierarchy"]
