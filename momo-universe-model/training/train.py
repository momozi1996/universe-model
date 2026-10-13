"""Staged training CLI (SPEC 4): python -m training.train --config configs/pilot.json

S1 force -> S2 +state -> S3 +rollout(4->8) + 0.01*L_energy. AdamW lr=3e-4,
wd=1e-5, grad clip 1.0, seed=42. Val selects the best checkpoint per stage;
test data is never touched here.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from model.gravity import MUMGravityModel
from training.losses import (force_loss, state_loss, rollout_loss, energy_loss)

TIME_BUDGET_SEC = 40 * 60          # CPU budget guard (spec: 45 min)


# ---------------------------------------------------------------- data utils
def load_split(path, n_max=3):
    z = np.load(path, allow_pickle=False)
    M, T, N, _ = z["pos"].shape
    pos = torch.zeros(M, T, n_max, 3)
    vel = torch.zeros(M, T, n_max, 3)
    acc = torch.zeros(M, T, n_max, 3)
    mu = torch.zeros(M, n_max, 1)
    mask = torch.zeros(M, n_max, dtype=torch.bool)
    pos[:, :, :N] = torch.from_numpy(z["pos"]).float()
    vel[:, :, :N] = torch.from_numpy(z["vel"]).float()
    acc[:, :, :N] = torch.from_numpy(z["acc"]).float()
    mu[:, :N, 0] = torch.from_numpy(z["mass"]).float()
    mask[:, :N] = True
    return {"pos": pos, "vel": vel, "acc": acc, "mu": mu, "mask": mask,
            "valid_len": torch.from_numpy(z["valid_len"]).long(), "T": T}


class FrameBank:
    """Flattened (traj, frame) bank for force/state losses."""

    def __init__(self, splits):
        pos, vel, acc, mu, mask, valid = [], [], [], [], [], []
        pair_ok = []
        for sp in splits:
            M, T = sp["pos"].shape[:2]
            pos.append(sp["pos"].reshape(M * T, *sp["pos"].shape[2:]))
            vel.append(sp["vel"].reshape(M * T, *sp["vel"].shape[2:]))
            acc.append(sp["acc"].reshape(M * T, *sp["acc"].shape[2:]))
            mu.append(sp["mu"][:, None].expand(M, T, *sp["mu"].shape[1:])
                      .reshape(M * T, *sp["mu"].shape[1:]))
            mask.append(sp["mask"][:, None].expand(M, T, sp["mask"].shape[1])
                        .reshape(M * T, sp["mask"].shape[1]))
            vl = sp["valid_len"][:, None].expand(M, T).reshape(M * T)
            t_idx = torch.arange(T)[None, :].expand(M, T).reshape(M * T)
            valid.append(t_idx < vl)
            pair_ok.append(t_idx + 1 < vl)
        self.pos, self.vel, self.acc = torch.cat(pos), torch.cat(vel), torch.cat(acc)
        self.mu, self.mask = torch.cat(mu), torch.cat(mask)
        self.frame_idx = torch.where(torch.cat(valid))[0]
        self.pair_idx = torch.where(torch.cat(pair_ok))[0]

    def frames(self, idx):
        return (self.mu[idx], self.pos[idx], self.vel[idx],
                self.acc[idx], self.mask[idx])


def rollout_train(model, mu, pos0, vel0, mask, dt, steps):
    """Differentiable rollout (training only; inference uses model.rollout)."""
    pos, vel = pos0, vel0
    ph, vh = [pos], [vel]
    for _ in range(steps):
        pos, vel = model.step(mu, pos, vel, mask, dt)
        ph.append(pos)
        vh.append(vel)
    return torch.stack(ph, 1), torch.stack(vh, 1)


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def val_force(model, bank, n_eval=8192, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = bank.frame_idx[torch.randperm(len(bank.frame_idx), generator=g)[:n_eval]]
    mu, pos, vel, acc, mask = bank.frames(idx)
    tot, n = 0.0, 0
    for i in range(0, len(idx), 2048):
        s = slice(i, i + 2048)
        pred = model.predict_acceleration(mu[s], pos[s], mask[s], vel[s])
        l, _ = force_loss(pred, acc[s], mask[s])
        tot += float(l) * (s.stop - s.start)
        n += s.stop - s.start
    return tot / n


@torch.no_grad()
def val_rollout(model, sp, dt=0.01, k=8):
    vl = sp["valid_len"]
    ok = torch.where(vl > k)[0]
    mu, mask = sp["mu"][ok], sp["mask"][ok]
    pos0, vel0 = sp["pos"][ok, 0], sp["vel"][ok, 0]
    pos_true = sp["pos"][ok, :k + 1]
    pos_hist, _ = rollout_train(model, mu, pos0, vel0, mask, dt, k)
    l, _ = rollout_loss(pos_hist, pos_true, mask)
    lf, _ = force_loss(model.predict_acceleration(mu, pos0, mask, vel0),
                       sp["acc"][ok, 0], mask)
    return float(l) + float(lf)


# ------------------------------------------------------------------- stages
def run_stage(model, stage, cfg, bank3, bank_all, sp3_train, sp3_val, bank_val,
              device, t0, history, ckpt_dir):
    tcfg = cfg["training"]
    scfg = tcfg["stages"][stage]
    epochs = scfg["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])
    warmup = int(tcfg.get("lr_warmup_steps", 0))
    gstep = [0]

    def set_lr():
        if warmup:
            f = min(1.0, (gstep[0] + 1) / warmup)
            for pg in opt.param_groups:
                pg["lr"] = tcfg["lr"] * f
    g = torch.Generator().manual_seed(cfg["seed"] + int(stage[1]))
    rng = np.random.default_rng(cfg["seed"] + int(stage[1]))
    best_val, best_state = float("inf"), None
    hist = {"train": [], "val": [], "abs_force_mse": []}
    bs_f = tcfg["batch_frames"]
    bs_t = tcfg["batch_traj"]
    dt = cfg["data"]["output_dt"]
    epochs_done = 0

    for ep in range(epochs):
        model.train()
        ep_loss, ep_abs, nb = 0.0, 0.0, 0
        if stage in ("S1", "S2"):
            src = bank_all.frame_idx
            # random frame subsample per epoch (CPU time budget; recorded)
            n_sub = min(tcfg.get("frames_per_epoch", len(src)), len(src))
            perm = src[torch.randperm(len(src), generator=g)[:n_sub]]
            for i in range(0, len(perm), bs_f):
                idx = perm[i:i + bs_f]
                mu, pos, vel, acc, mask = bank_all.frames(idx)
                pred = model.predict_acceleration(mu, pos, mask, vel)
                lf, abs_mse = force_loss(pred, acc, mask)
                loss = lf
                if stage == "S2":
                    # one-step state loss on the same frames (t -> t+1 pairs
                    # when available; reuse batch frames that have a successor)
                    ok = torch.isin(idx, bank_all.pair_idx)
                    if ok.any():
                        idp = idx[ok]
                        mu2, pos2, vel2, _, mask2 = bank_all.frames(idp)
                        pos_n, vel_n = model.step(mu2, pos2, vel2, mask2, dt)
                        ls, _ = state_loss(pos_n, vel_n, bank_all.pos[idp + 1],
                                           bank_all.vel[idp + 1], mask2)
                        loss = loss + ls
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                set_lr()
                opt.step()
                gstep[0] += 1
                ep_loss += float(loss.detach())
                ep_abs += float(abs_mse)
                nb += 1
        else:  # S3: rollout on three-body train trajectories
            k = scfg["rollout_steps_warm"] if ep < epochs // 2 \
                else scfg["rollout_steps_final"]
            M = sp3_train["pos"].shape[0]
            perm = torch.randperm(M, generator=g)
            for i in range(0, M, bs_t):
                bidx = perm[i:i + bs_t]
                vl = sp3_train["valid_len"][bidx]
                ok = vl > k + 1
                if not ok.any():
                    continue
                bidx = bidx[ok]
                vl = vl[ok]
                # random start frame per trajectory with start+k+1 <= valid_len
                hi = (vl - k - 1).clamp_min(1)
                starts = torch.tensor([rng.integers(0, int(h)) for h in hi])
                mu, mask = sp3_train["mu"][bidx], sp3_train["mask"][bidx]
                rows = torch.arange(len(bidx))
                pos0 = sp3_train["pos"][bidx, starts]
                vel0 = sp3_train["vel"][bidx, starts]
                acc0 = sp3_train["acc"][bidx, starts]
                pos_true = sp3_train["pos"][bidx[:, None], starts[:, None]
                                          + torch.arange(k + 1)[None, :]]
                pos_hist, vel_hist = rollout_train(model, mu, pos0, vel0, mask, dt, k)
                lf, abs_mse = force_loss(model.predict_acceleration(mu, pos0, mask, vel0),
                                         acc0, mask)
                pos_n, vel_n = model.step(mu, pos0, vel0, mask, dt)
                ls, _ = state_loss(pos_n, vel_n, pos_true[:, 1],
                                   sp3_train["vel"][bidx, starts + 1], mask)
                lr_, _ = rollout_loss(pos_hist, pos_true, mask)
                le, _ = energy_loss(mu, pos_hist, vel_hist, mask)
                loss = lf + ls + lr_ + scfg["energy_weight"] * le
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                set_lr()
                opt.step()
                gstep[0] += 1
                ep_loss += float(loss.detach())
                ep_abs += float(abs_mse)
                nb += 1
        epochs_done = ep + 1
        # ---- validation / checkpoint selection ----
        model.eval()
        if stage in ("S1", "S2"):
            v = val_force(model, bank_val)
        else:
            v = val_rollout(model, sp3_val, dt=dt,
                            k=scfg["rollout_steps_final"])
        hist["train"].append(ep_loss / max(nb, 1))
        hist["abs_force_mse"].append(ep_abs / max(nb, 1))
        hist["val"].append(v)
        if v < best_val:
            best_val = v
            best_state = {k2: v2.detach().clone() for k2, v2 in model.state_dict().items()}
        print(f"  [{stage}] epoch {ep + 1:3d}/{epochs} train={hist['train'][-1]:.5f} "
              f"val={v:.5f} abs_force_mse={hist['abs_force_mse'][-1]:.3e}", flush=True)
        if time.time() - t0 > TIME_BUDGET_SEC:
            print(f"  [{stage}] TIME BUDGET hit at epoch {ep + 1}; stopping early")
            break

    history[stage] = hist
    history[stage + "_epochs_done"] = epochs_done
    history[stage + "_best_val"] = best_val
    ckpt = os.path.join(ckpt_dir, f"mum_gravity_pilot_stage{stage[1]}.pt")
    torch.save({"model_state": best_state, "stage": stage, "val_loss": best_val,
                "config": cfg}, ckpt)
    model.load_state_dict(best_state)
    print(f"  [{stage}] done. best val = {best_val:.5f} -> {ckpt}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.json")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.set_num_threads(2)
    device = "cpu"
    t0 = time.time()

    ddir = "results/data"
    two_tr = load_split(os.path.join(ddir, "two_body_train.npz"))
    two_va = load_split(os.path.join(ddir, "two_body_val.npz"))
    thr_tr = load_split(os.path.join(ddir, "three_body_train.npz"))
    thr_va = load_split(os.path.join(ddir, "three_body_val.npz"))
    bank_all = FrameBank([two_tr, thr_tr])
    bank_val = FrameBank([two_va, thr_va])
    print(f"frames: train={len(bank_all.frame_idx)} val={len(bank_val.frame_idx)}")

    model = MUMGravityModel(**cfg["model"]).to(device)
    with torch.no_grad():
        # init so initial s ~ softplus(-2) = 0.127: tames the heavy-tailed
        # per-frame normalized loss at the start of training (see RESULTS.md)
        model.force_head.mlp[-1].bias.fill_(-2.0)
    n_params = model.num_parameters()
    print(f"model parameters: {n_params}")
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/config.json", "w") as f:
        json.dump({"model": cfg["model"], "n_parameters": n_params,
                   "limits": cfg["limits"], "output_dt": cfg["data"]["output_dt"],
                   "seed": cfg["seed"]}, f, indent=2)

    history = {}
    model = run_stage(model, "S1", cfg, None, bank_all, thr_tr, thr_va, bank_val,
                      device, t0, history, "checkpoints")
    model = run_stage(model, "S2", cfg, None, bank_all, thr_tr, thr_va, bank_val,
                      device, t0, history, "checkpoints")
    model = run_stage(model, "S3", cfg, None, bank_all, thr_tr, thr_va, bank_val,
                      device, t0, history, "checkpoints")

    wall = time.time() - t0
    history["wall_time_sec"] = wall
    with open("results/train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # loss curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, s in zip(axes, ["S1", "S2", "S3"]):
        ax.plot(history[s]["train"], label="train")
        ax.plot(history[s]["val"], label="val")
        ax.set_yscale("log")
        ax.set_title(f"{s} (best val {history[s + '_best_val']:.4f})")
        ax.set_xlabel("epoch")
        ax.legend()
    fig.suptitle(f"MUM-Gravity-0.1 staged training (wall {wall / 60:.1f} min)")
    fig.tight_layout()
    fig.savefig("results/loss_curves.png", dpi=120)
    print(f"TOTAL WALL TIME: {wall:.1f} sec ({wall / 60:.2f} min)")


if __name__ == "__main__":
    main()
