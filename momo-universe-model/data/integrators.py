"""Self-written high-precision teacher integrators (SPEC 2.2).

Approved deviation from the original plan: REBOUND is unavailable offline, so
the teacher is implemented here in numpy float64:
  - newton_acc      : O(N^2) analytic Newtonian acceleration, G=1
  - kdk_leapfrog    : kick-drift-kick leapfrog (2nd order, symplectic)
  - rk4             : classic 4th-order Runge-Kutta (teacher workhorse)
  - kepler_two_body : analytic two-body solution (absolute judge for 2-body)

All functions are float64 and support batched initial states where noted:
mass [..., N], r0/v0 [..., N, 3].
"""
import numpy as np

G = 1.0


def newton_acc(mass, pos):
    """Analytic Newtonian acceleration. mass [..., N], pos [..., N, 3] ->
    acc [..., N, 3]. acc_i = G * sum_{j != i} m_j (r_j - r_i) / |r_j - r_i|^3.
    """
    mass = np.asarray(mass, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    d = pos[..., None, :, :] - pos[..., :, None, :]          # d_ij = r_j - r_i
    r2 = (d ** 2).sum(axis=-1)
    N = pos.shape[-2]
    idx = np.arange(N)
    r2[..., idx, idx] = np.inf                                # mask diagonal
    inv_r3 = r2 ** -1.5
    return G * (inv_r3[..., None] * d * mass[..., None, :, None]).sum(axis=-2)


def kdk_leapfrog(acc_fn, mass, r0, v0, dt, steps):
    """Kick-drift-kick leapfrog. Returns (t[steps+1], pos[steps+1,...,N,3], vel)."""
    r = np.array(r0, dtype=np.float64, copy=True)
    v = np.array(v0, dtype=np.float64, copy=True)
    pos_out = np.empty((steps + 1,) + r.shape, dtype=np.float64)
    vel_out = np.empty((steps + 1,) + v.shape, dtype=np.float64)
    pos_out[0], vel_out[0] = r, v
    a = acc_fn(mass, r)
    for k in range(steps):
        v = v + 0.5 * dt * a
        r = r + dt * v
        a = acc_fn(mass, r)
        v = v + 0.5 * dt * a
        pos_out[k + 1], vel_out[k + 1] = r, v
    t = np.arange(steps + 1, dtype=np.float64) * dt
    return t, pos_out, vel_out


def rk4(acc_fn, mass, r0, v0, dt, steps):
    """Classic RK4 on the first-order system (r' = v, v' = acc(r)).
    Returns (t[steps+1], pos, vel). Teacher workhorse; use small dt.
    """
    r = np.array(r0, dtype=np.float64, copy=True)
    v = np.array(v0, dtype=np.float64, copy=True)
    pos_out = np.empty((steps + 1,) + r.shape, dtype=np.float64)
    vel_out = np.empty((steps + 1,) + v.shape, dtype=np.float64)
    pos_out[0], vel_out[0] = r, v
    for k in range(steps):
        a1 = acc_fn(mass, r)
        r2 = r + 0.5 * dt * v
        v2 = v + 0.5 * dt * a1
        a2 = acc_fn(mass, r2)
        r3 = r + 0.5 * dt * v2
        v3 = v + 0.5 * dt * a2
        a3 = acc_fn(mass, r3)
        r4 = r + dt * v3
        v4 = v + dt * a3
        a4 = acc_fn(mass, r4)
        r = r + dt * (v + 2.0 * v2 + 2.0 * v3 + v4) / 6.0
        v = v + dt * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0
        pos_out[k + 1], vel_out[k + 1] = r, v
    t = np.arange(steps + 1, dtype=np.float64) * dt
    return t, pos_out, vel_out


def _solve_kepler_elliptic(M, e):
    """Solve M = E - e sin E (e < 1) by Newton iteration. M, E in radians."""
    M = np.mod(M, 2.0 * np.pi)
    E = M + e * np.sin(M)                                   # good seed for e < 1
    for _ in range(60):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = f / fp
        E = E - dE
        if abs(dE) < 1e-15:
            break
    return E


def _solve_kepler_hyperbolic(M, e):
    """Solve M = e sinh H - H (e > 1) by Newton iteration."""
    H = np.arcsinh(M / e) if abs(M) > 1e-12 else 0.0        # asymptotic seed
    for _ in range(80):
        f = e * np.sinh(H) - H - M
        fp = e * np.cosh(H) - 1.0
        dH = f / fp
        H = H - dH
        if abs(dH) < 1e-14:
            break
    return H


def kepler_two_body(m1, m2, r0, v0, t_array):
    """Analytic two-body solution (absolute judge).

    m1, m2  : body masses (G = 1)
    r0, v0  : RELATIVE position / velocity (body1 - body2) at t = 0, shape (3,)
    t_array : (T,) times
    Returns pos [2, T, 3], vel [2, T, 3] for (body1, body2) in the COM frame
    with zero COM velocity.

    Elliptic (E_tot < 0, e < 1) and hyperbolic (E_tot > 0, e > 1) branches are
    implemented; near-circular orbits (e < 1e-8) use the exact circular
    degenerate handling. Parabolic (|E| < tiny) is not supported.
    """
    r0 = np.asarray(r0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    t_array = np.asarray(t_array, dtype=np.float64)
    M = m1 + m2
    mu = G * M

    r0n = np.linalg.norm(r0)
    v02 = float(v0 @ v0)
    h_vec = np.cross(r0, v0)
    h = np.linalg.norm(h_vec)
    if h < 1e-14:
        raise ValueError("degenerate: zero angular momentum (radial orbit)")
    h_hat = h_vec / h
    eps = 0.5 * v02 - mu / r0n                              # specific energy
    if abs(eps) < 1e-12:
        raise ValueError("parabolic orbit not supported")
    e_vec = np.cross(v0, h_vec) / mu - r0 / r0n
    e = float(np.linalg.norm(e_vec))

    ECIRC = 1e-8
    if eps < 0:                                             # ---- elliptic ----
        a = -mu / (2.0 * eps)
        n = np.sqrt(mu / a ** 3)
        if e < ECIRC:                                       # near-circular degenerate
            # exact circular orbit at radius a, phase 0 along r0
            P = r0 / r0n
            Q = np.cross(h_hat, P)
            pos_rel = np.empty((len(t_array), 3))
            vel_rel = np.empty((len(t_array), 3))
            for k, t in enumerate(t_array):
                th = n * t
                pos_rel[k] = a * (np.cos(th) * P + np.sin(th) * Q)
                vel_rel[k] = a * n * (-np.sin(th) * P + np.cos(th) * Q)
        else:
            P = e_vec / e
            Q = np.cross(h_hat, P)
            # initial eccentric anomaly from r0, v0
            cosE0 = (1.0 - r0n / a) / e
            sinE0 = (r0 @ v0) / (e * np.sqrt(mu * a))
            E0 = np.arctan2(sinE0, cosE0)
            M0 = E0 - e * np.sin(E0)
            pos_rel = np.empty((len(t_array), 3))
            vel_rel = np.empty((len(t_array), 3))
            for k, t in enumerate(t_array):
                E = _solve_kepler_elliptic(M0 + n * t, e)
                den = 1.0 - e * np.cos(E)
                xp = a * (np.cos(E) - e)
                yp = a * np.sqrt(1.0 - e * e) * np.sin(E)
                vxp = -a * n * np.sin(E) / den
                vyp = a * n * np.sqrt(1.0 - e * e) * np.cos(E) / den
                pos_rel[k] = xp * P + yp * Q
                vel_rel[k] = vxp * P + vyp * Q
    else:                                                   # ---- hyperbolic ----
        a = mu / (2.0 * eps)                                # > 0 by convention
        n = np.sqrt(mu / a ** 3)
        if e <= 1.0:
            raise ValueError("inconsistent hyperbolic state (e <= 1, eps > 0)")
        P = e_vec / e
        Q = np.cross(h_hat, P)
        coshH0 = (1.0 + r0n / a) / e
        sinhH0 = (r0 @ v0) / (e * np.sqrt(mu * a))
        H0 = np.arcsinh(sinhH0) if sinhH0 != 0 else 0.0
        if coshH0 < np.cosh(H0):                            # sign consistency guard
            H0 = -H0
        M0 = e * np.sinh(H0) - H0
        pos_rel = np.empty((len(t_array), 3))
        vel_rel = np.empty((len(t_array), 3))
        for k, t in enumerate(t_array):
            H = _solve_kepler_hyperbolic(M0 + n * t, e)
            den = e * np.cosh(H) - 1.0
            xp = a * (e - np.cosh(H))
            yp = a * np.sqrt(e * e - 1.0) * np.sinh(H)
            vxp = -a * n * np.sinh(H) / den
            vyp = a * n * np.sqrt(e * e - 1.0) * np.cosh(H) / den
            pos_rel[k] = xp * P + yp * Q
            vel_rel[k] = vxp * P + vyp * Q

    # split relative motion into the two bodies (COM at origin, COM vel = 0)
    pos = np.empty((2, len(t_array), 3))
    vel = np.empty((2, len(t_array), 3))
    pos[0] = (m2 / M) * pos_rel
    pos[1] = -(m1 / M) * pos_rel
    vel[0] = (m2 / M) * vel_rel
    vel[1] = -(m1 / M) * vel_rel
    return pos, vel


def total_energy(mass, pos, vel):
    """Total Newtonian energy (batched). mass [...,N], pos/vel [...,N,3]."""
    ke = 0.5 * (mass[..., None] * vel ** 2).sum(axis=(-2, -1))
    d = pos[..., None, :, :] - pos[..., :, None, :]
    r = np.linalg.norm(d, axis=-1)
    N = pos.shape[-2]
    idx = np.arange(N)
    r[..., idx, idx] = np.inf
    pe = -G * (mass[..., None, :] * mass[..., :, None] / r).sum(axis=(-2, -1)) / 2.0
    return ke + pe


def total_angular_momentum(mass, pos, vel):
    """Total angular momentum vector (batched) -> [..., 3]."""
    return (mass[..., None] * np.cross(pos, vel)).sum(axis=-2)


def total_momentum(mass, vel):
    return (mass[..., None] * vel).sum(axis=-2)
