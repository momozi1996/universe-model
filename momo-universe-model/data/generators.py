"""Initial-state sampling + teacher trajectory generation (SPEC 2.3).

States are sampled directly in normalized units (G = M0 = L0 = 1): after
sampling, positions/velocities are rescaled so the mass-weighted RMS radius
L0 equals exactly 1 at t = 0, with velocities scaled by s^(-1/2) to preserve
the dynamical family (Kepler similarity).

Usage: python -m data.generators --config configs/pilot.json
"""
import argparse
import json
import os
import time

import numpy as np

from .integrators import newton_acc, rk4, total_energy
from .splits import split_indices

G = 1.0


def _random_rotation(rng):
    """Uniformly random 3D rotation matrix via QR of a Gaussian matrix."""
    a = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(a)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _rescale_to_L0(mass, pos, vel):
    """Rescale so mass-weighted RMS radius L0 = 1 (COM already at origin)."""
    M0 = mass.sum()
    L0 = np.sqrt((mass[:, None] * pos ** 2).sum() / M0)
    s = 1.0 / L0
    return pos * s, vel / np.sqrt(s)


def sample_two_body_ic(rng, ecc_max=0.8, mass_ratio_max=100.0, a_min=0.4, a_max=1.2):
    """One bound two-body IC in normalized units. Returns mass[2], pos[2,3],
    vel[2,3], meta dict."""
    k = 1.0 + rng.random() * (mass_ratio_max - 1.0)         # mass ratio in [1, 100]
    m1 = k / (1.0 + k)
    m2 = 1.0 / (1.0 + k)
    mass = np.array([m1, m2])
    a = a_min + rng.random() * (a_max - a_min)
    e = rng.random() * ecc_max
    mu = G * mass.sum()
    n = np.sqrt(mu / a ** 3)
    E0 = rng.random() * 2.0 * np.pi                          # eccentric anomaly
    xp = a * (np.cos(E0) - e)
    yp = a * np.sqrt(1.0 - e * e) * np.sin(E0)
    den = 1.0 - e * np.cos(E0)
    vxp = -a * n * np.sin(E0) / den
    vyp = a * n * np.sqrt(1.0 - e * e) * np.cos(E0) / den
    rot = _random_rotation(rng)
    r_rel = rot @ np.array([xp, yp, 0.0])
    v_rel = rot @ np.array([vxp, vyp, 0.0])
    M = mass.sum()
    pos = np.stack([(m2 / M) * r_rel, -(m1 / M) * r_rel])
    vel = np.stack([(m2 / M) * v_rel, -(m1 / M) * v_rel])
    pos, vel = _rescale_to_L0(mass, pos, vel)
    return mass, pos, vel, {"e": float(e), "a_rel": float(a), "mass_ratio": float(k)}


def sample_three_body_ic(rng, mass_ratio_max=10.0, sep_lo=0.1, sep_hi=np.inf,
                          max_tries=200):
    """One roughly-bound three-body IC (negative total energy, min initial
    separation in [sep_lo, sep_hi)). Returns mass[3], pos[3,3], vel[3,3], meta.
    """
    close_bucket = np.isfinite(sep_hi)
    for _ in range(max_tries):
        mass = 10.0 ** (rng.random(3) *
                        np.log10(mass_ratio_max))               # ratios up to mass_ratio_max
        mass = mass / mass.sum()                              # M0 = 1
        if close_bucket:
            # construct one close pair at a target separation via fixed-point
            # iteration on the L0 rescale (sep_final = sep_raw / L0_raw)
            target = sep_lo + rng.random() * (sep_hi - sep_lo)
            u = rng.standard_normal(3)
            u /= np.linalg.norm(u)
            w = rng.standard_normal(3)
            w /= np.linalg.norm(w)
            r_far = w * (0.4 + 0.6 * rng.random())
            s = target
            pos = None
            for _ in range(8):
                pos = np.stack([0.5 * s * u, -0.5 * s * u, r_far])
                pos = pos - (mass[:, None] * pos).sum(axis=0)
                L0 = np.sqrt((mass[:, None] * pos ** 2).sum() / mass.sum())
                s_new = target * L0
                if abs(s_new - s) < 1e-6:
                    break
                s = s_new
        else:
            d = rng.standard_normal((3, 3))
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            pos = d * (rng.random(3) ** (1.0 / 3.0))[:, None]  # uniform in unit ball
            pos -= (mass[:, None] * pos).sum(axis=0)           # COM at origin
        vel = rng.standard_normal((3, 3))
        vel -= (mass[:, None] * vel).sum(axis=0)               # zero total momentum
        pos, vel = _rescale_to_L0(mass, pos, vel)
        # scale velocities to be bound: E = -c |PE|, c in [0.3, 0.9]
        c = 0.3 + 0.6 * rng.random()
        ke = 0.5 * (mass[:, None] * vel ** 2).sum()
        pe = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                pe -= G * mass[i] * mass[j] / np.linalg.norm(pos[i] - pos[j])
        s2 = (1.0 - c) * abs(pe) / ke
        vel = vel * np.sqrt(s2)
        # min initial separation bucket
        seps = [np.linalg.norm(pos[i] - pos[j])
                for i in range(3) for j in range(i + 1, 3)]
        sep = min(seps)
        if sep_lo <= sep < sep_hi:
            e_tot = float(total_energy(mass, pos, vel))
            assert e_tot < 0, "must be bound"
            return mass, pos, vel, {"min_init_sep": float(sep), "E0": e_tot}
    raise RuntimeError(f"no IC found with separation in [{sep_lo}, {sep_hi})")


def generate_trajectories(mass, pos0, vel0, dt=0.002, output_dt=0.01, n_frames=128,
                           min_sep_truncate=0.02):
    """Batched teacher integration.

    mass [M,N], pos0/vel0 [M,N,3]. Returns dict of arrays:
      pos/vel/acc [M, n_frames, N, 3] (float64), t [n_frames],
      valid_len [M] int, truncated [M] bool, reason list[str].
    Frames >= valid_len hold the raw integrated values but MUST NOT be used.
    """
    assert abs(output_dt / dt - round(output_dt / dt)) < 1e-9
    sub = int(round(output_dt / dt))
    steps = (n_frames - 1) * sub
    t_all, pos_all, vel_all = rk4(newton_acc, mass, pos0, vel0, dt, steps)
    pos = np.ascontiguousarray(pos_all[::sub].transpose(1, 0, 2, 3))  # [M,T,N,3]
    vel = np.ascontiguousarray(vel_all[::sub].transpose(1, 0, 2, 3))
    t = t_all[::sub]
    M, _, N, _ = pos.shape
    # analytic acceleration labels at every saved frame
    acc = newton_acc(
        np.broadcast_to(mass[:, None, :], (M, n_frames, N)).reshape(M * n_frames, N),
        pos.reshape(M * n_frames, N, 3),
    ).reshape(M, n_frames, N, 3)
    # truncation scan
    d = pos[:, :, :, None, :] - pos[:, :, None, :, :]         # [M,T,N,N,3]
    r = np.linalg.norm(d, axis=-1)
    idx = np.arange(N)
    r[:, :, idx, idx] = np.inf
    min_sep = r.min(axis=(-2, -1))                            # [M,T]
    finite = np.isfinite(pos).all(axis=(-3, -2, -1)) & np.isfinite(vel).all(axis=(-3, -2, -1))
    valid_len = np.full(M, n_frames, dtype=np.int64)
    truncated = np.zeros(M, dtype=bool)
    reasons = []
    for m in range(M):
        bad = np.where((min_sep[m] < min_sep_truncate) | (~finite[m]))[0]
        if len(bad):
            valid_len[m] = bad[0]
            truncated[m] = True
            reasons.append("min_separation<%.2f" % min_sep_truncate
                           if min_sep[m, bad[0]] < min_sep_truncate else "non_finite")
        else:
            reasons.append("")
    return {"pos": pos, "vel": vel, "acc": acc, "t": t, "mass": mass,
            "valid_len": valid_len, "truncated": truncated, "reason": reasons}


def _save_split(out, prefix, idx, traj, extra):
    sel = {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == traj["pos"].shape[:1]
               else v) for k, v in traj.items() if k != "reason"}
    sel["reason"] = np.array([traj["reason"][i] for i in idx])
    for k, v in extra.items():
        sel[k] = v[idx] if isinstance(v, np.ndarray) and len(v) == len(traj["pos"]) else v
    np.savez_compressed(out, **sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.json")
    ap.add_argument("--outdir", default="results/data")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    dcfg = cfg["data"]
    seed = cfg["seed"]
    os.makedirs(args.outdir, exist_ok=True)
    t_start = time.time()
    manifest = {"seed": seed, "version": "mum-gravity-0.1-pilot",
                "dt_teacher": dcfg["dt_teacher"], "output_dt": dcfg["output_dt"],
                "n_frames": dcfg["n_frames"], "G": 1.0,
                "note": "sampled directly in normalized units G=M0=L0=1; "
                        "teacher = self-written RK4 float64 (REBOUND unavailable)",
                "systems": {}}

    # ---------------- two-body ----------------
    rng = np.random.default_rng(seed)
    n2 = dcfg["n_two_body"]
    tb = dcfg["two_body"]
    masses, poss, vels, eccs, ratios = [], [], [], [], []
    for _ in range(n2):
        m, p, v, meta = sample_two_body_ic(rng, **tb)
        masses.append(m); poss.append(p); vels.append(v)
        eccs.append(meta["e"]); ratios.append(meta["mass_ratio"])
    traj2 = generate_trajectories(np.array(masses), np.array(poss), np.array(vels),
                                  dcfg["dt_teacher"], dcfg["output_dt"],
                                  dcfg["n_frames"], dcfg["min_separation_truncate"])
    sp = dcfg["two_body_split"]
    idx2 = split_indices(n2, sp["train"], sp["val"], sp["test"], seed=seed)
    extra2 = {"ecc": np.array(eccs), "mass_ratio": np.array(ratios)}
    for name, idx in idx2.items():
        _save_split(os.path.join(args.outdir, f"two_body_{name}.npz"), "two_body",
                    idx, traj2, extra2)
    manifest["systems"]["two_body"] = {
        "n": n2, "split": {k: len(v) for k, v in idx2.items()},
        "truncated": int(traj2["truncated"].sum()),
        "truncation_rate": float(traj2["truncated"].mean()),
        "ecc_range": [float(min(eccs)), float(max(eccs))],
        "mass_ratio_range": [float(min(ratios)), float(max(ratios))]}

    # ---------------- three-body ----------------
    rng = np.random.default_rng(seed + 1)
    n3 = dcfg["n_three_body"]
    tb3 = dcfg["three_body"]
    n_close = int(round(n3 * tb3["close_bucket_frac"]))
    masses, poss, vels, seps = [], [], [], []
    for i in range(n3):
        if i < n_close:
            lo, hi = tb3["close_bucket_sep"]
        else:
            lo, hi = tb3["min_init_sep"], np.inf
        m, p, v, meta = sample_three_body_ic(rng, tb3["mass_ratio_max"], lo, hi)
        masses.append(m); poss.append(p); vels.append(v); seps.append(meta["min_init_sep"])
    traj3 = generate_trajectories(np.array(masses), np.array(poss), np.array(vels),
                                  dcfg["dt_teacher"], dcfg["output_dt"],
                                  dcfg["n_frames"], dcfg["min_separation_truncate"])
    sp = dcfg["three_body_split"]
    idx3 = split_indices(n3, sp["train"], sp["val"], sp["test"], seed=seed)
    extra3 = {"min_init_sep": np.array(seps)}
    for name, idx in idx3.items():
        _save_split(os.path.join(args.outdir, f"three_body_{name}.npz"), "three_body",
                    idx, traj3, extra3)
    manifest["systems"]["three_body"] = {
        "n": n3, "split": {k: len(v) for k, v in idx3.items()},
        "close_bucket_n": n_close,
        "truncated": int(traj3["truncated"].sum()),
        "truncation_rate": float(traj3["truncated"].mean()),
        "min_init_sep_range": [float(min(seps)), float(max(seps))],
        "mass_ratio_max": tb3["mass_ratio_max"],
        "mass_ratio_note": "pilot narrowed to 10:1 (from 100:1 in tech plan v2.0)"}

    manifest["wall_time_sec"] = time.time() - t_start
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
