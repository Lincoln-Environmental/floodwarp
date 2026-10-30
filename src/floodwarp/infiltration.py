"""Small, transparent infiltration formulations used by Overlander."""

import numpy as np


def exponential_infiltration_capacity(
    cumulative_infiltration_m,
    infiltration_f0_m_per_s,
    infiltration_fd_m,
    infiltration_k_m_per_s,
):
    """Return fc(F) = k + (f0-k) exp(-F/fd) using NumPy broadcasting."""
    cumulative = np.maximum(np.asarray(cumulative_infiltration_m, dtype=np.float64), 0.0)
    f0 = np.asarray(infiltration_f0_m_per_s, dtype=np.float64)
    fd = np.asarray(infiltration_fd_m, dtype=np.float64)
    k = np.asarray(infiltration_k_m_per_s, dtype=np.float64)
    if np.any(fd <= 0.0):
        raise ValueError("infiltration_fd_m must be positive")
    if np.any(f0 < 0.0) or np.any(k < 0.0) or np.any(f0 < k):
        raise ValueError("infiltration rates must satisfy 0 <= k <= f0")
    return k + (f0 - k) * np.exp(-cumulative / fd)


def green_ampt_infiltration_increment_m(
    cumulative_infiltration_m,
    ks_m_per_s,
    wetting_front_suction_m,
    delta_theta,
    dt_s,
):
    """Return the finite-step ponded Green-Ampt infiltration increment (m).

    Solves the integrated Green-Ampt equation

    ``Ks*dt = dF - A*ln((F0+dF+A)/(F0+A))``, where
    ``A = psi_f*delta_theta``. Inputs broadcast as NumPy arrays. The 80-step
    vectorized bisection is a transparent CPU reference; GPU execution uses a
    fixed-count safeguarded Newton iteration. ``A=0`` gives ``Ks*dt`` exactly,
    and ``Ks=0`` or ``dt=0`` gives zero.
    """
    f0, ks, psi, deficit, dt = np.broadcast_arrays(
        np.asarray(cumulative_infiltration_m, dtype=np.float64),
        np.asarray(ks_m_per_s, dtype=np.float64),
        np.asarray(wetting_front_suction_m, dtype=np.float64),
        np.asarray(delta_theta, dtype=np.float64),
        np.asarray(dt_s, dtype=np.float64),
    )
    if np.any(~np.isfinite(f0)) or np.any(f0 < 0.0):
        raise ValueError("cumulative infiltration must be finite and non-negative")
    if np.any(~np.isfinite(ks)) or np.any(ks < 0.0):
        raise ValueError("Ks must be finite and non-negative")
    if np.any(~np.isfinite(psi)) or np.any(psi <= 0.0):
        raise ValueError("wetting-front suction must be finite and positive")
    if np.any(~np.isfinite(deficit)) or np.any((deficit < 0.0) | (deficit > 1.0)):
        raise ValueError("delta_theta must be finite and in [0, 1]")
    if np.any(~np.isfinite(dt)) or np.any(dt < 0.0):
        raise ValueError("dt_s must be finite and non-negative")

    target = np.asarray(ks * dt, dtype=np.float64).copy()
    a = psi * deficit
    increment = target.copy()
    solve = (a > 0.0) & (target > 0.0)
    if np.any(solve):
        f = f0[solve]
        aa = a[solve]
        rhs = target[solve]
        lower = np.zeros_like(rhs)
        upper = rhs + 2.0 * np.sqrt(aa * rhs) + 2.0 * aa
        for _ in range(80):
            middle = lower + 0.5 * (upper - lower)
            residual = middle - aa * np.log1p(middle / (f + aa))
            below = residual < rhs
            lower = np.where(below, middle, lower)
            upper = np.where(below, upper, middle)
        increment[solve] = lower + 0.5 * (upper - lower)
    return increment


def green_ampt_rainfall_increment_m(
    cumulative_infiltration_m,
    ks_m_per_s,
    wetting_front_suction_m,
    delta_theta,
    rainfall_rate_m_per_s,
    dt_s,
    ponded_depth_m=0.0,
):
    """Return actual Green-Ampt infiltration under constant rain (m).

    A dry surface infiltrates all rain until the wetting front reaches
    ``Fp = Ks*psi_f*delta_theta/(rain_rate-Ks)``. The remaining part of the
    step uses the integrated ponded equation. Existing surface storage makes
    the whole step ponded. This reference uses the same transition as the
    Warp source kernels; surface-water availability caps the result.
    """
    f0, ks, psi, deficit, rain, dt, ponded = np.broadcast_arrays(
        np.asarray(cumulative_infiltration_m, dtype=np.float64),
        np.asarray(ks_m_per_s, dtype=np.float64),
        np.asarray(wetting_front_suction_m, dtype=np.float64),
        np.asarray(delta_theta, dtype=np.float64),
        np.asarray(rainfall_rate_m_per_s, dtype=np.float64),
        np.asarray(dt_s, dtype=np.float64),
        np.asarray(ponded_depth_m, dtype=np.float64),
    )
    if np.any(~np.isfinite(rain)) or np.any(rain < 0.0):
        raise ValueError("rainfall_rate_m_per_s must be finite and non-negative")
    if np.any(~np.isfinite(ponded)) or np.any(ponded < 0.0):
        raise ValueError("ponded_depth_m must be finite and non-negative")
    potential = green_ampt_infiltration_increment_m(f0, ks, psi, deficit, dt)
    actual = np.minimum(potential, ponded + rain * dt)
    dry = (ponded == 0.0) & (rain > 0.0) & (dt > 0.0)
    unponded = dry & (rain <= ks)
    actual = np.where(unponded, rain * dt, actual)
    transition = dry & (rain > ks) & (ks > 0.0) & (deficit > 0.0)
    if np.any(transition):
        ff = f0[transition]
        kk = ks[transition]
        aa = psi[transition] * deficit[transition]
        rr = rain[transition]
        step = dt[transition]
        ponding_f = kk * aa / (rr - kk)
        pre_time = np.minimum(step, np.maximum(0.0, (ponding_f - ff) / rr))
        pre_depth = rr * pre_time
        post_depth = green_ampt_infiltration_increment_m(
            ff + pre_depth, kk, psi[transition], deficit[transition],
            step - pre_time,
        )
        actual[transition] = np.minimum(rr * step, pre_depth + post_depth)
    return actual
