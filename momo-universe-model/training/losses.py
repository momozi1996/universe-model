"""Normalized training losses (SPEC 4).

All losses are normalized by reference scales (force: per-frame RMS
acceleration; state: L0 / V_scale, both 1 in normalized units); absolute
values are returned alongside for logging.
"""
import torch


def force_loss(pred, label, mask, eps=1e-8):
    """pred/label [B,N,3], mask [B,N] -> (normalized loss, absolute MSE)."""
    m = mask[..., None].to(pred.dtype)
    cnt = m.sum(dim=(1, 2)).clamp_min(1.0)
    mse = (((pred - label) ** 2) * m).sum(dim=(1, 2)) / cnt        # per-frame MSE
    rms2 = (((label ** 2) * m).sum(dim=(1, 2)) / cnt)              # per-frame RMS^2
    return (mse / (rms2 + eps)).mean(), mse.mean().detach()


def state_loss(pos_pred, vel_pred, pos_true, vel_true, mask):
    """One-step KDK prediction error, normalized by L0^2 / V_scale^2 (=1 here)."""
    m = mask[..., None].to(pos_pred.dtype)
    cnt = m.sum(dim=(1, 2)).clamp_min(1.0)
    lp = (((pos_pred - pos_true) ** 2) * m).sum(dim=(1, 2)) / cnt
    lv = (((vel_pred - vel_true) ** 2) * m).sum(dim=(1, 2)) / cnt
    return (lp + lv).mean(), (lp.mean().detach(), lv.mean().detach())


def rollout_loss(pos_hist, pos_true, mask):
    """pos_hist [B,K+1,N,3] vs teacher pos_true [B,K+1,N,3]; skip frame 0.
    Normalized by L0^2 (=1)."""
    m = mask[..., None].to(pos_hist.dtype)                        # [B,N,1]
    cnt = m.sum(dim=(1, 2)).clamp_min(1.0)                        # [B]
    err = (((pos_hist[:, 1:] - pos_true[:, 1:]) ** 2) * m[:, None]).sum(dim=(2, 3)) \
        / cnt[:, None]
    return err.mean(), err.mean(dim=0).detach()                   # total, per-step


def energy_total(mu, pos, vel, mask, eps=1e-6):
    """Total Newtonian energy in normalized units (torch, differentiable).
    mu [B,N,1], pos/vel [B,N,3], mask [B,N]."""
    B, N, _ = pos.shape
    m = mask.to(pos.dtype)
    ke = 0.5 * ((mu[..., 0] * m) * (vel ** 2).sum(-1)).sum(dim=1)
    d = pos[:, None, :, :] - pos[:, :, None, :]                     # r_j - r_i
    r = d.norm(dim=-1)
    eye = torch.eye(N, dtype=torch.bool, device=pos.device)
    valid = (mask[:, :, None] & mask[:, None, :] & ~eye[None]).to(pos.dtype)
    pe = -(valid * mu[..., 0][:, :, None] * mu[..., 0][:, None, :]
           / (r + eye[None].to(pos.dtype))).sum(dim=(1, 2)) / 2.0
    return ke + pe


def energy_loss(mu, pos_hist, vel_hist, mask, eps=1e-6):
    """Relative squared energy error at the rollout end: ((E-E0)/|E0|)^2."""
    e0 = energy_total(mu, pos_hist[:, 0], vel_hist[:, 0], mask)
    e1 = energy_total(mu, pos_hist[:, -1], vel_hist[:, -1], mask)
    rel = (e1 - e0) / (e0.abs() + eps)
    return (rel ** 2).mean(), rel.abs().mean().detach()
