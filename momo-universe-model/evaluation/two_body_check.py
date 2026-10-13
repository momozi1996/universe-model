"""Two-body Kepler comparison (SPEC 5.2).

Model rollout (dt=0.01) vs Kepler analytic solution on two-body test
trajectories, with analytic-Newton + KDK (dt=0.01) as baseline. Teacher code
(data/) is used here only as reference, which is allowed in evaluation/.
"""
import numpy as np
import torch

from data.integrators import newton_acc, kdk_leapfrog, kepler_two_body
from inference.rollout import strict_rollout


def rel_pos_err_curve(pos_model, pos_ref):
    """pos_* [T,N,3] -> relative error per frame (Frobenius over bodies)."""
    num = np.linalg.norm(pos_model - pos_ref, axis=-1)     # [T,N]
    den = np.linalg.norm(pos_ref, axis=-1).max()
    return num.max(axis=-1) / den


@torch.no_grad()
def run(model, two_test, dt=0.01, n_circ=5, n_ell=5, limits=None):
    """two_test: dict from two_body_test.npz. Returns metrics + curves."""
    pos, vel = two_test["pos"], two_test["vel"]
    mass, ecc = two_test["mass"], two_test["ecc"]
    vl = two_test["valid_len"]
    T = pos.shape[1]
    idx_all = np.arange(len(ecc))
    circ = idx_all[np.argsort(ecc)][:n_circ]
    elli = idx_all[np.argsort(-ecc)][:n_ell]
    sel = np.concatenate([circ, elli])

    out = {"per_traj": [], "n_circular": n_circ, "n_elliptical": n_ell}
    curves_model, curves_kdk = {}, {}
    for i in sel:
        n_frames = int(vl[i])
        steps = n_frames - 1
        t_arr = np.arange(n_frames) * dt
        m1, m2 = float(mass[i, 0]), float(mass[i, 1])
        r_rel0 = pos[i, 0, 0] - pos[i, 0, 1]
        v_rel0 = vel[i, 0, 0] - vel[i, 0, 1]
        pos_ref, _ = kepler_two_body(m1, m2, r_rel0, v_rel0, t_arr)   # [2,T,3]
        pos_ref = np.transpose(pos_ref, (1, 0, 2))                    # [T,2,3]

        # model rollout
        res = strict_rollout(model, mass[i], pos[i, 0], vel[i, 0],
                             np.ones(2, dtype=bool), dt=dt, steps=steps,
                             limits=limits)
        pos_m = res.pos[0].numpy()
        k = min(pos_m.shape[0], n_frames)
        err_m = rel_pos_err_curve(pos_m[:k], pos_ref[:k])

        # analytic Newton + KDK baseline, same dt
        _, pos_k, _ = kdk_leapfrog(newton_acc, mass[i], pos[i, 0], vel[i, 0],
                                   dt, steps)
        err_k = rel_pos_err_curve(pos_k[:n_frames], pos_ref)

        out["per_traj"].append({
            "idx": int(i), "ecc": float(ecc[i]),
            "model_stopped": res.stopped, "stop_reason": res.stop_reason,
            "model_frames": int(k),
            "model_err_median": float(np.median(err_m)),
            "model_err_final": float(err_m[-1]),
            "kdk_err_median": float(np.median(err_k)),
            "kdk_err_final": float(err_k[-1]),
        })
        curves_model[int(i)] = err_m
        curves_kdk[int(i)] = err_k

    med = np.median([r["model_err_median"] for r in out["per_traj"]])
    fin = np.median([r["model_err_final"] for r in out["per_traj"]])
    med_k = np.median([r["kdk_err_median"] for r in out["per_traj"]])
    fin_k = np.median([r["kdk_err_final"] for r in out["per_traj"]])
    out.update({"model_err_median_of_medians": float(med),
                "model_err_median_of_finals": float(fin),
                "kdk_err_median_of_medians": float(med_k),
                "kdk_err_median_of_finals": float(fin_k),
                "n_stopped": int(sum(r["model_stopped"] for r in out["per_traj"]))})
    return out, curves_model, curves_kdk
