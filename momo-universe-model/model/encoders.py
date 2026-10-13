"""Node / edge invariant feature construction (SPEC 3).

All features are scalars invariant under rotation, translation and body
permutation, so predicted accelerations transform equivariantly.
"""
import torch
import torch.nn as nn


def node_scalars(mu, vel):
    """mu: [B,N,1], vel: [B,N,3] -> node scalar features [B,N,2]."""
    speed2 = (vel ** 2).sum(dim=-1, keepdim=True)  # |u_i|^2, rotation invariant
    return torch.cat([mu, speed2], dim=-1)


def edge_scalars(mu, pos, vel, pair_mask):
    """Ordered-pair invariant scalars.

    Returns feat [B,N,N,5], pair_mask [B,N,N] (i != j, both valid).
    Features: log r_ij, mu_i+mu_j, mu_i*mu_j, radial rel. velocity,
    tangential rel. speed.
    """
    B, N, _ = pos.shape
    d = pos[:, :, None, :] - pos[:, None, :, :]          # [B,N,N,3] d_ij = x_i - x_j
    r = d.norm(dim=-1, keepdim=True).clamp_min(1e-4)      # [B,N,N,1]
    dv = vel[:, :, None, :] - vel[:, None, :, :]          # u_i - u_j
    d_hat = d / r
    v_rad = (dv * d_hat).sum(dim=-1, keepdim=True)        # radial component
    v_tan = (dv - v_rad * d_hat).norm(dim=-1, keepdim=True)
    mu_i = mu[:, :, None, :].expand(B, N, N, 1)
    mu_j = mu[:, None, :, :].expand(B, N, N, 1)
    feat = torch.cat([
        torch.log(r), mu_i + mu_j, mu_i * mu_j, v_rad, v_tan
    ], dim=-1)
    return feat, d, r


class NodeEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )

    def forward(self, mu, vel):
        return self.mlp(node_scalars(mu, vel))
