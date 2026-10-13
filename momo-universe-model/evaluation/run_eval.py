"""Evaluation entry point (SPEC 5): python -m evaluation.run_eval

Runs the full evaluation suite on the TEST split (used exactly once, after
training/checkpoint selection), writes results/metrics.json + PNG curves.
"""
import json
import os
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation import (independence_check, symmetry_tests, two_body_check,
                        conservation, large_dt)
from inference.load import load_model
from inference.rollout import strict_rollout

LIMITS = {"max_steps": 10000, "min_separation": 0.02, "max_radius": 10.0}
DT = 0.01


def _load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


@torch.no_grad()
def short_window(model, three_test, ks=(4, 8, 16), n_max=100):
    """Test-set k-step rollout relative position error: median + p90."""
    vl = three_test["valid_len"]
    pos, vel, mass = three_test["pos"], three_test["vel"], three_test["mass"]
    out = {"ks": list(ks)}
    curves = []
    for k in ks:
        idx = np.where(vl > k)[0][:n_max]
        errs, stops = [], 0
        for i in idx:
            res = strict_rollout(model, mass[i], pos[i, 0], vel[i, 0],
                                 np.ones(3, bool), dt=DT, steps=k, limits=LIMITS)
            kk = res.steps_done
            pm = res.pos[0, -1].numpy()
            pt = pos[i, kk]
            num = np.linalg.norm(pm - pt)
            den = np.linalg.norm(pt)
            errs.append(float(num / max(den, 1e-12)))
            stops += int(res.stopped)
        out[f"err_{k}_median"] = float(np.median(errs))
        out[f"err_{k}_p90"] = float(np.percentile(errs, 90))
        out[f"err_{k}_max"] = float(np.max(errs))
        out[f"n_{k}"] = len(errs)
        out[f"n_stopped_{k}"] = stops
        curves.append(errs)
    return out, curves


def main():
    t0 = time.time()
    torch.set_num_threads(2)
    os.makedirs("results", exist_ok=True)
    metrics = {}

    # 0. independence (import graph) acceptance
    independence_check.check()
    metrics["independence_check"] = "PASS"
    print("[eval] independence check PASS")

    model, meta = load_model()
    metrics["checkpoint"] = {k: meta[k] for k in ("stage", "val_loss", "ckpt_path")}
    metrics["n_parameters"] = meta.get("n_parameters")
    print(f"[eval] loaded {meta['ckpt_path']} (stage={meta['stage']}, "
          f"val_loss={meta['val_loss']:.5f}), params={meta.get('n_parameters')}")

    two_test = _load("results/data/two_body_test.npz")
    three_test = _load("results/data/three_body_test.npz")

    # 1. symmetry on trained model (three-body test frames batch)
    B = 64
    mu = torch.from_numpy(three_test["mass"][:B]).float().unsqueeze(-1)
    pos = torch.from_numpy(three_test["pos"][:B, 0]).float()
    vel = torch.from_numpy(three_test["vel"][:B, 0]).float()
    mask = torch.ones(B, 3, dtype=torch.bool)
    metrics["symmetry"] = symmetry_tests.run(model, mu, pos, vel, mask)
    print("[eval] symmetry:", json.dumps(metrics["symmetry"], indent=2))

    # 2. two-body Kepler comparison
    tb, curves_m, curves_k = two_body_check.run(model, two_test, dt=DT,
                                                limits=LIMITS)
    metrics["two_body"] = tb
    print(f"[eval] two-body: model median rel err {tb['model_err_median_of_medians']:.3e}, "
          f"final {tb['model_err_median_of_finals']:.3e}; "
          f"KDK baseline {tb['kdk_err_median_of_medians']:.3e}/"
          f"{tb['kdk_err_median_of_finals']:.3e}; stopped={tb['n_stopped']}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, c in curves_m.items():
        ax.plot(np.arange(len(c)) * DT, c, "r-", alpha=0.4)
    for i, c in curves_k.items():
        ax.plot(np.arange(len(c)) * DT, c, "b--", alpha=0.3)
    ax.plot([], [], "r-", label="model rollout dt=0.01")
    ax.plot([], [], "b--", label="analytic Newton + KDK dt=0.01")
    ax.set_yscale("log")
    ax.set_xlabel("t (normalized)")
    ax.set_ylabel("relative position error vs Kepler")
    ax.set_title("Two-body: model vs analytic baseline (Kepler judge)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/two_body_error.png", dpi=120)
    plt.close(fig)

    # 3. three-body short window
    sw, sw_curves = short_window(model, three_test)
    metrics["three_body_short_window"] = sw
    print(f"[eval] three-body short window: "
          + ", ".join(f"k={k}: med={sw[f'err_{k}_median']:.3e} "
                      f"p90={sw[f'err_{k}_p90']:.3e}" for k in sw["ks"]))

    fig, ax = plt.subplots(figsize=(7, 5))
    meds = [sw[f"err_{k}_median"] for k in sw["ks"]]
    p90s = [sw[f"err_{k}_p90"] for k in sw["ks"]]
    ax.plot(sw["ks"], meds, "o-", label="median")
    ax.plot(sw["ks"], p90s, "s--", label="p90")
    ax.set_yscale("log")
    ax.set_xlabel("rollout steps (dt=0.01)")
    ax.set_ylabel("relative position error")
    ax.set_title("Three-body test short-window rollout error")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/three_body_short_window.png", dpi=120)
    plt.close(fig)

    # 4. conservation drift
    cons = conservation.run(model, three_test, strict_rollout, dt=DT, steps=16,
                            limits=LIMITS)
    metrics["conservation"] = cons
    print(f"[eval] conservation: P drift max={cons['momentum_drift_max']:.2e}, "
          f"L drift med={cons['angmom_drift_median']:.2e}, "
          f"E drift med={cons['energy_drift_median']:.3e} "
          f"(teacher {cons['energy_drift_teacher_median']:.2e})")

    # 5. large-dt verdict
    ld = large_dt.run(model, three_test, limits=LIMITS)
    metrics["large_dt"] = {k: v for k, v in ld.items() if k != "rows"}
    metrics["large_dt_rows"] = ld["rows"]
    print(f"[eval] large-dt: KDK(0.05) med={ld['kdk_0.05_median']:.3e} vs "
          f"model(0.05) med={ld['model_0.05_median']:.3e} -> "
          f"model better: {ld['verdict_model_better_at_0.05']}")

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["KDK 0.01", "KDK 0.05", "model 0.01", "model 0.05"]
    keys = ["kdk_0.01", "kdk_0.05", "model_0.01", "model_0.05"]
    data = [[r[k] for r in ld["rows"]] for k in keys]
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel("end relative position error (T=0.8, RK4 dt=0.001 truth)")
    ax.set_title("Large-dt verdict experiment")
    fig.tight_layout()
    fig.savefig("results/large_dt.png", dpi=120)
    plt.close(fig)

    metrics["eval_wall_time_sec"] = time.time() - t0
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] wrote results/metrics.json ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
