"""Message passing with permutation-invariant (mean) aggregation (SPEC 3)."""
import torch
import torch.nn as nn


class MessageLayer(nn.Module):
    """One round: edge message MLP -> masked mean aggregation -> node update."""

    def __init__(self, hidden_dim, edge_dim=5):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, edge_feat, msg_mask):
        """h: [B,N,H]; edge_feat: [B,N,N,E]; msg_mask: [B,N,N] float (1 = valid i<-j)."""
        B, N, H = h.shape
        h_i = h[:, :, None, :].expand(B, N, N, H)
        h_j = h[:, None, :, :].expand(B, N, N, H)
        m = self.edge_mlp(torch.cat([h_i, h_j, edge_feat], dim=-1))  # message from j to i
        m = m * msg_mask[..., None]
        denom = msg_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        agg = m.sum(dim=2) / denom                                    # mean over senders j
        return h + self.node_mlp(torch.cat([h, agg], dim=-1))         # residual update
