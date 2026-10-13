"""Teacher quality acceptance (SPEC 2.2).

Pure-assert script: python tests/test_integrators.py -> prints PASS.
  - two-body KDK / RK4 vs Kepler analytic solution
  - RK4 dt=0.001, 256 steps: position error < 1e-6
  - RK4 energy / angular-momentum drift < 1e-8
  - near-circular degenerate branch of the Kepler solver
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.integrators import (G, newton_acc, kdk_leapfrog, rk4, kepler_two_body,
                              total_energy, total_angular_momentum)


def rel_pos_err(pos_num, pos_ref):
    """pos [T,N,3] vs ref [N,T,3] -> max relative position error."""
    ref = np.transpose(pos_ref, (1, 0, 2))
    num = pos_num
    err = np.linalg.norm(num - ref, axis=-1).max()
    scale = np.linalg.norm(ref, axis=-1).max()
    return err / scale, err


def main():
    # ---- elliptical two-body, e ~ 0.5, unequal masses ----
    m1, m2 = 0.75, 0.25
    mass = np.array([m1, m2])
    M = m1 + m2
    a, e = 1.0, 0.5
    # relative IC at pericenter along x, velocity along y
    rp = a * (1.0 - e)
    vp = np.sqrt(G * M * (1.0 + e) / (a * (1.0 - e)))
    r0_rel = np.array([rp, 0.0, 0.0])
    v0_rel = np.array([0.0, vp, 0.0])
    r0 = np.stack([(m2 / M) * r0_rel, -(m1 / M) * r0_rel])
    v0 = np.stack([(m2 / M) * v0_rel, -(m1 / M) * v0_rel])

    dt, steps = 0.001, 256
    t, pos_rk4, vel_rk4 = rk4(newton_acc, mass, r0, v0, dt, steps)
    pos_kep, vel_kep = kepler_two_body(m1, m2, r0_rel, v0_rel, t)

    rel_err, abs_err = rel_pos_err(pos_rk4, pos_kep)
    assert abs_err < 1e-6, f"RK4 vs Kepler abs position error {abs_err:.3e} >= 1e-6"
    print(f"  RK4(dt=0.001,256) vs Kepler: abs pos err = {abs_err:.3e} (< 1e-6)")

    # energy / angular momentum drift of RK4 teacher
    E = total_energy(np.broadcast_to(mass, (steps + 1, 2)),
                     pos_rk4.reshape(-1, 2, 3), vel_rk4.reshape(-1, 2, 3))
    L = total_angular_momentum(np.broadcast_to(mass, (steps + 1, 2)),
                               pos_rk4.reshape(-1, 2, 3), vel_rk4.reshape(-1, 2, 3))
    e_drift = abs(E.max() - E.min()) / abs(E[0])
    l_drift = np.linalg.norm(L - L[0], axis=-1).max() / np.linalg.norm(L[0])
    assert e_drift < 1e-8, f"RK4 energy drift {e_drift:.3e} >= 1e-8"
    assert l_drift < 1e-8, f"RK4 angular momentum drift {l_drift:.3e} >= 1e-8"
    print(f"  RK4 energy drift = {e_drift:.3e}, L drift = {l_drift:.3e} (< 1e-8)")

    # ---- KDK cross-check vs Kepler (looser, symplectic 2nd order) ----
    _, pos_kdk, _ = kdk_leapfrog(newton_acc, mass, r0, v0, dt, steps)
    _, abs_err_kdk = rel_pos_err(pos_kdk, pos_kep)
    assert abs_err_kdk < 1e-4, f"KDK vs Kepler abs err {abs_err_kdk:.3e}"
    print(f"  KDK(dt=0.001,256) vs Kepler: abs pos err = {abs_err_kdk:.3e} (< 1e-4)")

    # ---- near-circular degenerate branch ----
    e_circ = 1e-10
    rc = 1.0
    vc = np.sqrt(G * M / rc)
    r0c_rel = np.array([rc, 0.0, 0.0])
    v0c_rel = np.array([0.0, vc * (1 + e_circ), 0.0])     # essentially circular
    r0c = np.stack([(m2 / M) * r0c_rel, -(m1 / M) * r0c_rel])
    v0c = np.stack([(m2 / M) * v0c_rel, -(m1 / M) * v0c_rel])
    t_c, pos_rk4c, _ = rk4(newton_acc, mass, r0c, v0c, dt, steps)
    pos_kepc, _ = kepler_two_body(m1, m2, r0c_rel, v0c_rel, t_c)
    _, abs_errc = rel_pos_err(pos_rk4c, pos_kepc)
    assert abs_errc < 1e-5, f"circular branch abs err {abs_errc:.3e}"
    print(f"  near-circular Kepler branch vs RK4: abs pos err = {abs_errc:.3e} (< 1e-5)")

    # ---- Kepler energy consistency along the analytic solution ----
    Ek = total_energy(np.broadcast_to(mass, (steps + 1, 2)),
                      np.transpose(pos_kep, (1, 0, 2)),
                      np.transpose(vel_kep, (1, 0, 2)))
    assert abs(Ek.max() - Ek.min()) / abs(Ek[0]) < 1e-12
    print(f"  Kepler analytic energy constancy: {(abs(Ek.max()-Ek.min())/abs(Ek[0])):.3e}")

    # ---- newton_acc sanity: two bodies on x axis ----
    acc = newton_acc(np.array([1.0, 2.0]), np.array([[0.0, 0, 0], [1.0, 0, 0]]))
    assert np.allclose(acc[0], [2.0, 0, 0]) and np.allclose(acc[1], [-1.0, 0, 0])
    print("  newton_acc two-body sanity OK")

    print("PASS test_integrators")


if __name__ == "__main__":
    main()
