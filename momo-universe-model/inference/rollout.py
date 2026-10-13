"""Strict-mode rollout (SPEC 3): range checks, stop-reason recording.

Pure model path: imports only torch + model/. No teacher integrators, no
data/ modules. Model infers from weights + normalization config only.
"""
import numpy as np
import torch

from model.gravity import RolloutResult

DEFAULT_LIMITS = {"max_steps": 10000, "min_separation": 0.02, "max_radius": 10.0}


def _to_tensor(a, dtype=torch.float32):
    if isinstance(a, torch.Tensor):
        return a.to(dtype)
    return torch.as_tensor(np.asarray(a), dtype=dtype)


@torch.no_grad()
def strict_rollout(model, mu, pos0, vel0, mask, dt, steps, limits=None):
    """Run model.rollout with limits; verify finiteness of the returned
    trajectory (never silently returns NaN). Accepts numpy or torch input.

    Returns RolloutResult (torch tensors).
    """
    limits = dict(DEFAULT_LIMITS if limits is None else limits)
    mu_t = _to_tensor(mu)
    pos_t = _to_tensor(pos0)
    vel_t = _to_tensor(vel0)
    if isinstance(mask, torch.Tensor):
        mask_t = mask.bool()
    else:
        mask_t = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
    if mu_t.dim() == 1:                      # [N] -> [N,1]
        mu_t = mu_t[:, None]
    if mu_t.dim() == 2:                      # [N,1] -> [1,N,1]
        mu_t, pos_t, vel_t = mu_t[None], pos_t[None], vel_t[None]
        mask_t = mask_t[None]
    res = model.rollout(mu_t, pos_t, vel_t, mask_t, dt=dt, steps=steps,
                        limits=limits)
    # strict guarantee: everything we return is finite
    assert torch.isfinite(res.pos).all() and torch.isfinite(res.vel).all(), \
        "rollout returned non-finite state without stopping"
    if res.stopped:
        # truncate away the offending frame for separation/radius stops is not
        # required: the recorded frames are all finite by construction
        pass
    return res
