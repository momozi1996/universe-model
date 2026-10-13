"""Large-dt verdict experiment (SPEC 5.4).

Fixed physical duration T = 0.8 on three-body test trajectories:
  (a) analytic Newton force + KDK, dt = 0.01 (80 steps) and dt = 0.05 (16 steps)
  (b) model rollout,               dt = 0.01 (80 steps) and dt = 0.05 (16 steps)
  (c) reference truth: RK4 dt = 0.001 (800 steps)
Verdict question: is the model at dt=0.05 better than the analytic force at
dt=0.05? (Either answer is acceptable and reported as-is.)
"""
import numpy as np
import torch

from data.integrators import newton_acc, kdk_leapfrog, rk4
from inference.rollout import strict_rollout

T_PHYS = 0.8


def _rel_end_err(pos_a, pos_b):
    num = np.linalg.norm(pos_a - pos_b, axis=-1).max()
    den = np.linalg.norm(pos_b, axis=-1).max()
    return float(num / max(den, 1e-12))


@torch.no_grad()
def run(model, three_test, limits=None, n_max=40):
    vl = three_test["valid_len"]
    pos, vel, mass = three_test["pos"], three_test["vel"], three_test["mass"]
    idx = np.where(vl > int(T_PHYS / 0.01))[0][:n_max]
    rows = []
    for i in idx:
        m = mass[i].astype(np.float64)
        r0, v0 = pos[i, 0].astype(np.float64), vel[i, 0].astype(np.float64)
        # (c) truth
        _, pos_ref, _ = rk4(newton_acc, m, r0, v0, 0.001, int(T_PHYS / 0.001))
        ref_end = pos_ref[-1]
        row = {"idx": int(i)}
        # (a) analytic KDK
        for dt in (0.01, 0.05):
            _, pos_k, _ = kdk_leapfrog(newton_acc, m, r0, v0, dt,
                                       int(round(T_PHYS / dt)))
            row[f"kdk_{dt}"] = _rel_end_err(pos_k[-1], ref_end)
        # (b) model
        for dt in (0.01, 0.05):
            res = strict_rollout(model, m, r0, v0, np.ones(3, bool),
                                 dt=dt, steps=int(round(T_PHYS / dt)),
                                 limits=limits)
            row[f"model_{dt}"] = _rel_end_err(res.pos[0, -1].numpy(), ref_end)
            row[f"model_{dt}_stopped"] = res.stopped
            row[f"model_{dt}_stop_reason"] = res.stop_reason
        rows.append(row)

    def med(key):
        return float(np.median([r[key] for r in rows])) if rows else None

    def p90(key):
        return float(np.percentile([r[key] for r in rows], 90)) if rows else None

    out = {
        "n_traj": len(rows), "T_phys": T_PHYS,
        "kdk_0.01_median": med("kdk_0.01"), "kdk_0.01_p90": p90("kdk_0.01"),
        "kdk_0.05_median": med("kdk_0.05"), "kdk_0.05_p90": p90("kdk_0.05"),
        "model_0.01_median": med("model_0.01"), "model_0.01_p90": p90("model_0.01"),
        "model_0.05_median": med("model_0.05"), "model_0.05_p90": p90("model_0.05"),
        "model_0.05_n_stopped": int(sum(r["model_0.05_stopped"] for r in rows)),
        "model_0.01_n_stopped": int(sum(r["model_0.01_stopped"] for r in rows)),
        "win_fraction_model_0.05_vs_kdk_0.05": float(np.mean(
            [r["model_0.05"] < r["kdk_0.05"] for r in rows])) if rows else None,
        "rows": rows,
    }
    out["verdict_model_better_at_0.05"] = bool(
        out["model_0.05_median"] is not None
        and out["model_0.05_median"] < out["kdk_0.05_median"])
    return out
