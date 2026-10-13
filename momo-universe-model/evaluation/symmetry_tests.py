"""Symmetry evaluation on the trained model (SPEC 5.1).

Permutation / random rotation / translation consistency of predicted
accelerations (< 1e-5 relative) and total-force cancellation |sum F| < 1e-6.
"""
import torch


def _random_rotation(g):
    a = torch.randn(3, 3, generator=g)
    q, r = torch.linalg.qr(a)
    q = q @ torch.diag(torch.sign(torch.diag(r)))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


@torch.no_grad()
def run(model, mu, pos, vel, mask, seed=123):
    """mu [B,N,1], pos/vel [B,N,3], mask [B,N] -> metrics dict."""
    g = torch.Generator().manual_seed(seed)
    acc = model.predict_acceleration(mu, pos, mask, vel)
    scale = acc.abs().max().clamp_min(1e-12)

    perm = torch.randperm(pos.shape[1], generator=g)
    acc_p = model.predict_acceleration(mu[:, perm], pos[:, perm], mask[:, perm],
                                       vel[:, perm])
    e_perm = float((acc_p - acc[:, perm]).abs().max() / scale)

    Q = _random_rotation(g)
    acc_r = model.predict_acceleration(mu, pos @ Q.T, mask, vel @ Q.T)
    e_rot = float((acc_r - acc @ Q.T).abs().max() / scale)

    shift = torch.randn(1, 1, 3, generator=g) * 5.0
    acc_t = model.predict_acceleration(mu, pos + shift, mask, vel)
    e_trans = float((acc_t - acc).abs().max() / scale)

    # total force cancellation; accumulated in float64 to expose the exact
    # add/subtract symmetry of the force head. Absolute residuals are bounded
    # by float32 divide/multiply roundtrip mu*(F/mu) ~ eps*|F|, so the
    # physically meaningful criterion is the relative one; the absolute number
    # is reported as-is.
    F_tot_per_frame = (mu.double() * acc.double()).sum(dim=1).abs().max(dim=1).values
    F_scale = (mu.double() * acc.double()).abs().sum(dim=1).max(dim=1).values
    F_tot = float(F_tot_per_frame.max())
    F_rel = float((F_tot_per_frame / F_scale.clamp_min(1e-30)).max())

    return {"perm_rel_err": e_perm, "rot_rel_err": e_rot,
            "trans_rel_err": e_trans,
            "total_force_max_abs": F_tot,
            "total_force_max_rel": F_rel,
            "perm_ok": e_perm < 1e-5, "rot_ok": e_rot < 1e-5,
            "trans_ok": e_trans < 1e-5,
            "momentum_ok_rel": F_rel < 1e-6,
            "momentum_ok_abs": F_tot < 1e-6}
