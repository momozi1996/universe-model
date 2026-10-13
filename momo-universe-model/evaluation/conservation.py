"""Conservation drift over model rollouts (SPEC 5.3).

Momentum / angular momentum are conserved by construction (float precision);
energy is NOT claimed to be conserved -- its drift is measured honestly.
All quantities computed in normalized units (G = 1).
"""
import numpy as np
import torch


def _energy(mu, pos, vel):
    """mu [T,N], pos/vel [T,N,3] -> E [T] (float64)."""
    ke = 0.5 * (mu[..., None] * vel ** 2).sum(axis=(-2, -1))
    d = pos[:, None, :, :] - pos[:, :, None, :]
    r = np.linalg.norm(d, axis=-1)
    N = pos.shape[1]
    r[:, np.arange(N), np.arange(N)] = np.inf
    pe = -(mu[:, None, :] * mu[:, :, None] / r).sum(axis=(-2, -1)) / 2.0
    return ke + pe


@torch.no_grad()
def run(model, three_test, rollout_fn, dt=0.01, steps=16, limits=None, n_max=50):
    """Roll out test three-body trajectories; measure drift of P, L, E.
    Returns metrics dict with per-trajectory final drifts and stop stats.
    """
    pos, vel, mass, vl = (three_test["pos"], three_test["vel"],
                          three_test["mass"], three_test["valid_len"])
    idx = np.where(vl > steps)[0][:n_max]
    p_drift, l_drift, e_drift = [], [], []
    e_drift_teacher, stops = [], []
    for i in idx:
        res = rollout_fn(model, mass[i], pos[i, 0], vel[i, 0],
                         np.ones(3, dtype=bool), dt=dt, steps=steps,
                         limits=limits)
        k = res.steps_done
        pm = res.pos[0].numpy().astype(np.float64)
        vm = res.vel[0].numpy().astype(np.float64)
        mu = mass[i].astype(np.float64)
        P = np.linalg.norm((mu[None, :, None] * vm).sum(axis=1), axis=-1)
        L = np.linalg.norm((mu[None, :, None] * np.cross(pm, vm)).sum(axis=1), axis=-1)
        E = _energy(np.broadcast_to(mu, (k + 1, 3)), pm, vm)
        p_drift.append(float(np.abs(P - P[0]).max()))
        l_drift.append(float(np.abs(L - L[0]).max() / max(abs(L[0]), 1e-12)))
        e_drift.append(float(abs(E[-1] - E[0]) / max(abs(E[0]), 1e-12)))
        # teacher energy drift on the same window for scale reference
        Et = _energy(np.broadcast_to(mu, (steps + 1, 3)),
                     pos[i, :steps + 1].astype(np.float64),
                     vel[i, :steps + 1].astype(np.float64))
        e_drift_teacher.append(float(abs(Et[-1] - Et[0]) / max(abs(Et[0]), 1e-12)))
        stops.append(res.stop_reason or "")
    return {
        "n_traj": int(len(idx)),
        "momentum_drift_max": float(np.max(p_drift)) if p_drift else None,
        "angmom_drift_median": float(np.median(l_drift)) if l_drift else None,
        "angmom_drift_max": float(np.max(l_drift)) if l_drift else None,
        "energy_drift_median": float(np.median(e_drift)) if e_drift else None,
        "energy_drift_max": float(np.max(e_drift)) if e_drift else None,
        "energy_drift_teacher_median": float(np.median(e_drift_teacher))
        if e_drift_teacher else None,
        "n_stopped": int(sum(1 for s in stops if s)),
        "stop_reasons": sorted(set(s for s in stops if s)),
    }
