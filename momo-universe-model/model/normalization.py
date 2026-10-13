"""Normalization with fixed reference scales (SPEC 2.1).

One rollout uses one fixed NormState; re-scaling mid-rollout is forbidden.
All computation here is float64 numpy (used at data/load time, not per step).
"""
from dataclasses import dataclass

import numpy as np

G = 1.0


@dataclass
class NormState:
    M0: float
    R0: np.ndarray  # (3,)
    V0: np.ndarray  # (3,)
    L0: float
    T0: float
    V_scale: float


def compute_norm(mass, pos, vel):
    """Build NormState from an initial state (mass[N], pos[N,3], vel[N,3])."""
    mass = np.asarray(mass, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    M0 = float(mass.sum())
    if M0 <= 0:
        raise ValueError("M0 must be positive")
    R0 = (mass[:, None] * pos).sum(axis=0) / M0
    V0 = (mass[:, None] * vel).sum(axis=0) / M0
    rc = pos - R0
    L0 = float(np.sqrt((mass[:, None] * rc ** 2).sum() / M0))
    if L0 <= 0:
        raise ValueError("L0 must be positive")
    T0 = float(np.sqrt(L0 ** 3 / (G * M0)))
    V_scale = L0 / T0
    return NormState(M0=M0, R0=R0, V0=V0, L0=L0, T0=T0, V_scale=V_scale)


def to_norm(norm: NormState, mass, pos, vel, t: float = 0.0):
    """Physical state -> normalized (mu, x, u)."""
    mu = np.asarray(mass, dtype=np.float64) / norm.M0
    x = (np.asarray(pos, dtype=np.float64) - norm.R0 - norm.V0 * t) / norm.L0
    u = (np.asarray(vel, dtype=np.float64) - norm.V0) / norm.V_scale
    return mu, x, u


def from_norm(norm: NormState, mu, x, u, t: float = 0.0):
    """Normalized state -> physical (mass, pos, vel)."""
    mass = np.asarray(mu, dtype=np.float64) * norm.M0
    pos = np.asarray(x, dtype=np.float64) * norm.L0 + norm.R0 + norm.V0 * t
    vel = np.asarray(u, dtype=np.float64) * norm.V_scale + norm.V0
    return mass, pos, vel
