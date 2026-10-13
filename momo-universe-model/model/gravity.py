"""MUMGravityModel: encode / predict_acceleration / step / rollout (SPEC 3).

Pure model path: this module (and all of model/, inference/) must never import
teacher integrators or anything under data/.
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .encoders import NodeEncoder, edge_scalars
from .message_passing import MessageLayer
from .pair_force_head import PairForceHead
from .state_update import kick_drift_kick


@dataclass
class RolloutResult:
    pos: torch.Tensor                 # [B, steps_done+1, N, 3]
    vel: torch.Tensor                 # [B, steps_done+1, N, 3]
    t: torch.Tensor                   # [steps_done+1]
    stopped: bool
    stop_reason: Optional[str]
    min_separation: torch.Tensor      # [steps_done+1]
    steps_done: int


class MUMGravityModel(nn.Module):
    def __init__(self, hidden_dim: int = 128, message_layers: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_layers = message_layers
        self.encoder = NodeEncoder(hidden_dim)
        self.layers = nn.ModuleList(
            [MessageLayer(hidden_dim) for _ in range(message_layers)]
        )
        self.force_head = PairForceHead(hidden_dim)

    def encode(self, mu, pos, vel, body_mask):
        """Node scalars -> message passing -> invariant h_i [B,N,H]."""
        edge_feat, _, _ = edge_scalars(mu, pos, vel, None)
        N = pos.shape[1]
        eye = torch.eye(N, dtype=torch.bool, device=pos.device)
        valid = body_mask[:, :, None] & body_mask[:, None, :]
        msg_mask = (valid & ~eye[None]).to(pos.dtype)               # [B,N,N] i receives from j
        h = self.encoder(mu, vel)
        h = h * body_mask[..., None].to(pos.dtype)
        for layer in self.layers:
            h = layer(h, edge_feat, msg_mask)
            h = h * body_mask[..., None].to(pos.dtype)
        return h

    def predict_acceleration(self, mu, pos, mask, vel=None):
        """mu [B,N,1], pos [B,N,3], mask [B,N] -> acc [B,N,3] float32.

        vel is optional: if omitted, zeros are used (pure force head check).
        """
        if vel is None:
            vel = torch.zeros_like(pos)
        h = self.encode(mu, pos, vel, mask)
        return self.force_head(h, mu, pos, mask)

    def step(self, mu, pos, vel, mask, dt):
        """One kick-drift-kick step."""
        acc_fn = lambda p, v: self.predict_acceleration(mu, p, mask, v)
        return kick_drift_kick(acc_fn, pos, vel, mask, dt)

    @torch.no_grad()
    def rollout(self, mu, pos0, vel0, mask, dt, steps, limits=None):
        """Roll out with limits; stops cleanly (no NaN blowup) and records why.

        limits: dict(max_steps, min_separation=0.02, max_radius=10.0).
        """
        limits = limits or {}
        max_steps = int(limits.get("max_steps", steps))
        min_sep_lim = float(limits.get("min_separation", 0.02))
        max_radius = float(limits.get("max_radius", 10.0))

        pos, vel = pos0.clone(), vel0.clone()
        pos_hist, vel_hist, sep_hist = [pos.clone()], [vel.clone()], []
        sep_hist.append(self._min_separation(pos, mask))
        stopped, stop_reason = False, None
        steps_done = 0

        for k in range(min(steps, max_steps)):
            pos, vel = self.step(mu, pos, vel, mask, dt)
            steps_done = k + 1
            if not torch.isfinite(pos).all() or not torch.isfinite(vel).all():
                stopped, stop_reason = True, "non_finite_state"
                steps_done -= 1
                break
            pos_hist.append(pos.clone())
            vel_hist.append(vel.clone())
            sep = self._min_separation(pos, mask)
            sep_hist.append(sep)
            r_max = self._max_radius(pos, mask)
            if float(sep.min()) < min_sep_lim:
                stopped, stop_reason = True, (
                    f"min_separation {float(sep.min()):.5f} < {min_sep_lim}")
                break
            if float(r_max.max()) > max_radius:
                stopped, stop_reason = True, (
                    f"max_radius {float(r_max.max()):.3f} > {max_radius}")
                break
        if steps_done >= max_steps and steps > max_steps:
            stopped, stop_reason = True, f"max_steps {max_steps} reached"

        return RolloutResult(
            pos=torch.stack(pos_hist, dim=1),
            vel=torch.stack(vel_hist, dim=1),
            t=torch.arange(steps_done + 1, dtype=torch.float32) * dt,
            stopped=stopped,
            stop_reason=stop_reason,
            min_separation=torch.stack(sep_hist, dim=0),
            steps_done=steps_done,
        )

    @staticmethod
    def _min_separation(pos, mask):
        B, N, _ = pos.shape
        d = pos[:, :, None, :] - pos[:, None, :, :]
        r = d.norm(dim=-1)
        eye = torch.eye(N, dtype=torch.bool, device=pos.device)
        valid = mask[:, :, None] & mask[:, None, :] & ~eye[None]
        r = torch.where(valid, r, torch.full_like(r, float("inf")))
        return r.view(B, -1).min(dim=-1).values

    @staticmethod
    def _max_radius(pos, mask):
        r = pos.norm(dim=-1)
        r = torch.where(mask, r, torch.zeros_like(r))
        return r.max(dim=-1).values

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
