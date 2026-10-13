"""Kick-drift-kick state update using model accelerations (SPEC 3)."""
import torch


def kick_drift_kick(acc_fn, pos, vel, body_mask, dt):
    """acc_fn(pos, vel_for_features) -> acc [B,N,3].

    Returns (pos_next, vel_next). Padding bodies stay at zero.
    Velocities at the half step are used for the second force evaluation.
    """
    m = body_mask[..., None].to(pos.dtype)
    a1 = acc_fn(pos, vel)
    v_half = vel + 0.5 * dt * a1
    pos_next = (pos + dt * v_half) * m
    a2 = acc_fn(pos_next, v_half)
    vel_next = (v_half + 0.5 * dt * a2) * m
    return pos_next, vel_next
