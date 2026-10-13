"""Symmetric pair force head (SPEC 3 hard constraint #3).

For each unordered pair i<j: z_ij = [h_i+h_j, |h_i-h_j|, mu_i+mu_j, mu_i*mu_j,
log r_ij]; s_ij = softplus(MLP(z_ij)); F_ij = s_ij * d_ij / r_ij is applied to
body i and the exact same vector is subtracted from body j, so total momentum
and angular momentum are conserved to float precision. No energy conservation
is claimed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PairForceHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        z_dim = 2 * hidden_dim + 3
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h, mu, pos, body_mask, r_clamp=1e-4):
        """Returns acc [B,N,3] (float32), zero on padded bodies."""
        B, N, H = h.shape
        dev = h.device
        # d_ij = x_j - x_i points from i toward j, so s>=0 (softplus) gives
        # an ATTRACTIVE force on i; distance r_ij clamped below.
        d = pos[:, None, :, :] - pos[:, :, None, :]                # [B,N,N,3]
        r = d.norm(dim=-1, keepdim=True).clamp_min(r_clamp)        # [B,N,N,1]
        log_r = torch.log(r)

        h_i = h[:, :, None, :].expand(B, N, N, H)
        h_j = h[:, None, :, :].expand(B, N, N, H)
        mu_i = mu[:, :, None, :].expand(B, N, N, 1)
        mu_j = mu[:, None, :, :].expand(B, N, N, 1)

        # symmetric invariants under i<->j swap
        z = torch.cat([
            h_i + h_j, (h_i - h_j).abs(),
            mu_i + mu_j, mu_i * mu_j, log_r,
        ], dim=-1)
        s = F.softplus(self.mlp(z))                                # [B,N,N,1], s_ij = s_ji

        # upper triangle (i<j) mask, padding excluded
        i_lt_j = torch.triu(torch.ones(N, N, dtype=torch.bool, device=dev), diagonal=1)
        valid = body_mask[:, :, None] & body_mask[:, None, :]
        pair_mask = (i_lt_j[None] & valid).to(h.dtype)             # [B,N,N]

        Fij = s * d / r * pair_mask[..., None]                     # force on i due to j
        # a_i += F_ij / mu_i ; a_j -= F_ij / mu_j  (same vector, exact cancellation)
        mu_safe_i = mu[:, :, None, :].clamp_min(1e-8)
        mu_safe_j = mu[:, None, :, :].clamp_min(1e-8)
        acc = (Fij / mu_safe_i).sum(dim=2) - (Fij / mu_safe_j).sum(dim=1)
        acc = acc * body_mask[..., None].to(h.dtype)
        return acc
