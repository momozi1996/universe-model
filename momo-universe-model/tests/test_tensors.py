"""Tensor contract tests: shapes, padding, masks (SPEC 2.4 / 3).

Pure-assert script: python tests/test_tensors.py -> prints PASS.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.gravity import MUMGravityModel


def main():
    torch.manual_seed(0)
    model = MUMGravityModel(hidden_dim=32, message_layers=2)

    B, N = 4, 3
    mu = torch.rand(B, N, 1) * 0.4 + 0.05
    pos = torch.randn(B, N, 3)
    vel = torch.randn(B, N, 3) * 0.5
    mask = torch.ones(B, N, dtype=torch.bool)
    # batch element 1: only 2 bodies (third padded); element 2: 1 body
    mask[1, 2] = False
    mask[2, 1:] = False
    mu = mu * mask[..., None].float()
    pos = pos * mask[..., None].float()
    vel = vel * mask[..., None].float()

    acc = model.predict_acceleration(mu, pos, mask, vel)
    assert acc.shape == (B, N, 3), f"acc shape {acc.shape}"
    assert acc.dtype == torch.float32, f"acc dtype {acc.dtype}"
    assert torch.isfinite(acc).all(), "acc has non-finite values"
    # padded bodies produce exactly zero acceleration
    assert (acc[~mask] == 0).all(), "padding bodies have nonzero acc"
    print("  shapes / dtype / padding contract OK")

    # coincident positions (r=0) must not blow up (clamp 1e-4)
    pos_c = pos.clone()
    pos_c[0, 1] = pos_c[0, 0]
    acc_c = model.predict_acceleration(mu, pos_c, mask, vel)
    assert torch.isfinite(acc_c).all(), "r=0 clamp failed"
    print("  r_ij lower clamp OK (no NaN at coincidence)")

    # single-body system: no pairs -> zero acceleration
    acc_1 = model.predict_acceleration(mu[2:3], pos[2:3], mask[2:3], vel[2:3])
    assert (acc_1[0, 0] == 0).all(), "single body should have zero acc"
    print("  single-body zero-force OK")

    # one-step + rollout shapes and mask handling
    pos_n, vel_n = model.step(mu, pos, vel, mask, dt=0.01)
    assert pos_n.shape == pos.shape and vel_n.shape == vel.shape
    assert (pos_n[~mask] == 0).all() and (vel_n[~mask] == 0).all()
    res = model.rollout(mu, pos, vel, mask, dt=0.01, steps=8,
                        limits={"max_steps": 100, "min_separation": 0.02,
                                "max_radius": 10.0})
    assert res.pos.shape[1] == res.steps_done + 1
    assert res.min_separation.shape[0] == res.steps_done + 1
    assert torch.isfinite(res.pos).all(), "rollout produced NaN"
    print(f"  rollout OK (steps_done={res.steps_done}, stopped={res.stopped}, "
          f"reason={res.stop_reason})")

    # normalization roundtrip < 1e-12 relative
    import numpy as np
    from model.normalization import compute_norm, to_norm, from_norm
    rng = np.random.default_rng(0)
    mass_np = rng.random(3) + 0.1
    pos_np = rng.standard_normal((3, 3))
    vel_np = rng.standard_normal((3, 3))
    norm = compute_norm(mass_np, pos_np, vel_np)
    mu_n, x_n, u_n = to_norm(norm, mass_np, pos_np, vel_np, t=0.37)
    mass_r, pos_r, vel_r = from_norm(norm, mu_n, x_n, u_n, t=0.37)
    assert np.allclose(mass_r, mass_np, rtol=1e-12, atol=1e-15)
    assert np.allclose(pos_r, pos_np, rtol=1e-12, atol=1e-15)
    assert np.allclose(vel_r, vel_np, rtol=1e-12, atol=1e-15)
    print("  NormState roundtrip < 1e-12 OK")

    print("PASS test_tensors")


if __name__ == "__main__":
    main()
