"""Force-head symmetry quick unit tests (SPEC 3 / 5.1).

Pure-assert script: python tests/test_symmetry.py -> prints PASS.
  - permutation / rotation / translation equivariance of predicted acc
  - total force cancellation: |sum_i mu_i a_i| < 1e-6 (momentum conservation)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.gravity import MUMGravityModel


def random_rotation():
    a = torch.randn(3, 3)
    q, r = torch.linalg.qr(a)
    q = q @ torch.diag(torch.sign(torch.diag(r)))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def main():
    torch.manual_seed(7)
    model = MUMGravityModel(hidden_dim=64, message_layers=3).eval()
    torch.set_grad_enabled(False)

    B, N = 2, 3
    mu = (torch.rand(B, N, 1) * 0.5 + 0.05)
    pos = torch.randn(B, N, 3)
    vel = torch.randn(B, N, 3)
    mask = torch.ones(B, N, dtype=torch.bool)

    acc = model.predict_acceleration(mu, pos, mask, vel)

    # ---- permutation: swap bodies 0 and 1 ----
    perm = torch.tensor([1, 0, 2])
    acc_p = model.predict_acceleration(mu[:, perm], pos[:, perm],
                                       mask[:, perm], vel[:, perm])
    err = (acc_p - acc[:, perm]).abs().max() / acc.abs().max()
    assert err < 1e-5, f"permutation error {err:.2e}"
    print(f"  permutation equivariance rel err = {float(err):.2e}")

    # ---- rotation: acc must rotate with the system ----
    Q = random_rotation()
    pos_r = pos @ Q.T
    vel_r = vel @ Q.T
    acc_r = model.predict_acceleration(mu, pos_r, mask, vel_r)
    err = ((acc_r - acc @ Q.T).abs().max() / acc.abs().max())
    assert err < 1e-5, f"rotation error {err:.2e}"
    print(f"  rotation equivariance rel err = {float(err):.2e}")

    # ---- translation: acc must be invariant ----
    shift = torch.randn(1, 1, 3) * 3.0
    acc_t = model.predict_acceleration(mu, pos + shift, mask, vel)
    err = ((acc_t - acc).abs().max() / acc.abs().max())
    assert err < 1e-5, f"translation error {err:.2e}"
    print(f"  translation invariance rel err = {float(err):.2e}")

    # ---- momentum conservation: sum_i mu_i a_i == 0 (float precision) ----
    tot = (mu * acc.double()).sum(dim=1)
    assert tot.abs().max() < 1e-6, f"|sum mu a| = {tot.abs().max():.2e}"
    print(f"  momentum conservation |sum mu_i a_i| = {float(tot.abs().max()):.2e}")

    # ---- angular momentum: sum_i mu_i x_i x a_i == 0 ----
    L = (mu * torch.linalg.cross(pos.double(), acc.double(), dim=-1)).sum(dim=1)
    Lmag = L.abs().max()
    assert Lmag < 1e-5, f"|sum mu x cross a| = {Lmag:.2e}"
    print(f"  angular momentum conservation |sum mu x x a| = {float(Lmag):.2e}")

    print("PASS test_symmetry")


if __name__ == "__main__":
    main()
