"""Fidelity of accepted interval-mean transport on identical time bounds."""

import numpy as np
from scipy.signal import correlate, correlation_lags, find_peaks


def cdf(*, t, mass):
    return np.r_[0.0, np.cumsum(mass)] / mass.sum()


def quantile(*, edges, cumulative, fraction):
    i = min(np.searchsorted(cumulative, fraction, side="left") - 1, len(edges) - 2)
    return float(
        edges[i]
        + (edges[i + 1] - edges[i])
        * (fraction - cumulative[i])
        / (cumulative[i + 1] - cumulative[i])
    )


def profile(*, t, q, cfg):
    dt = np.diff(t)
    mass = dt * q
    cumulative = cdf(t=t, mass=mass)
    times = {
        p: quantile(edges=t, cumulative=cumulative, fraction=p / 100)
        for p in (1, 10, 25, 50, 75, 90, 99)
    }
    mid = t[:-1] + dt / 2
    centroid = float(np.average(mid, weights=mass))
    peak = q.max()
    top = np.flatnonzero(q >= cfg["peak_plateau"] * peak)
    peaks = find_peaks(np.r_[0, q, 0], prominence=0.1 * peak, height=0.25 * peak)[0]
    one = len(peaks) == 1
    above = np.flatnonzero(q >= peak / 2)
    simple = one and np.all(np.diff(above) == 1)
    return dict(
        cumulative=cumulative,
        times=times,
        centre=centroid,
        peak=peak,
        peak_start=t[top[0]],
        peak_end=t[top[-1] + 1],
        modes=len(peaks),
        width80=times[90] - times[10],
        width50=times[75] - times[25],
        sd=float(np.sqrt(np.average((mid - centroid) ** 2 + dt**2 / 12, weights=mass))),
        fwhm=t[above[-1] + 1] - t[above[0]] if simple else None,
        sharp=one and len(top) <= 2 and top[0] > 0 and top[-1] < len(q) - 1,
        peak_time=(t[top[0]] + t[top[-1] + 1]) / 2,
    )


def transport_distance(*, difference, dt):
    """Exact integral of the absolute piecewise-linear CDF difference."""
    a, b = difference[:-1], difference[1:]
    y = (abs(a) + abs(b)) / 2
    cross = a * b < 0
    y[cross] = (a[cross] ** 2 + b[cross] ** 2) / (2 * (abs(a[cross]) + abs(b[cross])))
    return float(y @ dt)


def aligned_distance(*, edges, model_density, reference_density, shift):
    knots = np.union1d(edges, edges - shift)
    mid = (knots[:-1] + knots[1:]) / 2

    def density(*, coordinates, values):
        idx = np.searchsorted(edges, coordinates, side="right") - 1
        return np.where(
            (idx >= 0) & (idx < len(values)),
            values[np.clip(idx, 0, len(values) - 1)],
            0.0,
        )

    a = density(coordinates=mid + shift, values=model_density)
    b = density(coordinates=mid, values=reference_density)
    return float(50 * abs(a - b) @ np.diff(knots))


def evaluate(*, edges, model, reference, config, background=0.0):
    """Timing is candidate minus reference. Volumes use the full supplied window."""
    t = np.asarray(edges, dtype=float)
    a = np.asarray(model, dtype=float) - background
    b = np.asarray(reference, dtype=float) - background
    dt = np.diff(t)
    if (
        a.ndim != 1
        or b.shape != a.shape
        or dt.shape != a.shape
        or len(a) < 2
        or not (
            np.all(np.isfinite(t))
            and np.all(np.isfinite(a))
            and np.all(np.isfinite(b))
            and np.all(dt > 0)
        )
    ):
        raise ValueError(
            "Expected two finite interval-mean series and increasing common bounds"
        )
    peak = b.max()
    net = float(b @ dt)
    err = a - b
    out = dict(
        reference_peak_m3s=peak,
        reference_net_m3=net,
        model_net_m3=float(a @ dt),
        rmse_m3s=float(np.sqrt(np.average(err**2, weights=dt))),
        interval_max_s=dt.max(),
        duration_s=t[-1] - t[0],
        background_m3s=background,
        reference_status="usable",
    )
    if peak < config["minimum_peak_m3s"] or net < config["minimum_volume_m3"]:
        out["reference_status"] = "negligible_or_inward_reference"
        return out
    out.update(
        peak_bias_pct=100 * (a.max() / peak - 1),
        volume_bias_pct=100 * (err @ dt) / net,
        nrmse_peak=out["rmse_m3s"] / peak,
        integrated_abs_pct=100 * (abs(err) @ dt) / net,
    )
    variability = np.sqrt(np.average((b - net / dt.sum()) ** 2, weights=dt))
    if variability > max(config["minimum_peak_m3s"], 0.01 * peak):
        out["nrmse_sd"] = out["rmse_m3s"] / variability
    for q, label in ((a, "model"), (b, "reference")):
        pos = np.maximum(q, 0) @ dt
        neg = np.maximum(-q, 0) @ dt
        out[label + "_reverse_m3"] = neg
        out[label + "_truncated"] = bool(
            q[-1] > config["tail_fraction"] * max(q.max(), 0.0)
        )
        out[label + "_active_at_start"] = bool(
            q[0] > config["tail_fraction"] * max(q.max(), 0.0)
        )
        if pos < config["minimum_volume_m3"] or neg > pos * config["reversal_fraction"]:
            out["reference_status"] = label + "_insufficient_outward_or_reversing"
    if out["reference_status"] != "usable":
        return out
    pa, pb = np.maximum(a, 0), np.maximum(b, 0)
    m = profile(t=t, q=pa, cfg=config)
    r = profile(t=t, q=pb, cfg=config)
    span = r["width80"]
    out.update(
        reference_width80_s=span,
        model_width80_s=m["width80"],
        reference_modes=r["modes"],
        model_modes=m["modes"],
        reference_peak_to_mean=r["peak"] / (pb @ dt / dt.sum()),
        model_peak_to_mean=m["peak"] / (pa @ dt / dt.sum()),
        reference_width50_s=r["width50"],
        model_width50_s=m["width50"],
        width_resolved=bool(span >= config["width_bins"] * dt.max()),
    )
    lower = np.maximum(t[:-1], r["times"][1])
    upper = np.minimum(t[1:], r["times"][99])
    active = np.maximum(0.0, upper - lower)
    out["active_nrmse_peak"] = np.sqrt(np.average(err**2, weights=active)) / peak
    times = {f"t{p}_delay": m["times"][p] - r["times"][p] for p in (10, 25, 50, 75, 90)}
    times["centroid_delay"] = m["centre"] - r["centre"]
    times["wasserstein"] = transport_distance(
        difference=m["cumulative"] - r["cumulative"], dt=dt
    )
    out["peak_delay_lower_s"] = m["peak_start"] - r["peak_end"]
    out["peak_delay_upper_s"] = m["peak_end"] - r["peak_start"]
    if r["sharp"] and m["sharp"] and out["width_resolved"]:
        times["peak_delay"] = m["peak_time"] - r["peak_time"]
        for who, prof in (("model", m), ("reference", r)):
            out[who + "_rise_s"] = prof["peak_time"] - prof["times"][10]
            out[who + "_recession_s"] = prof["times"][90] - prof["peak_time"]
    for threshold in (0.1, 0.5):
        ir = np.flatnonzero(pb >= threshold * peak)
        im = np.flatnonzero(pa >= threshold * peak)
        if im.size and im[0] and ir[0]:
            times[f"arrival{round(100 * threshold)}_delay"] = t[im[0]] - t[ir[0]]
    for label in ("width80", "width50", "sd", "fwhm"):
        if m[label] is not None and r[label] is not None and r[label] > 0:
            out[label + "_bias_pct"] = 100 * (m[label] / r[label] - 1)
            out["model_" + label + "_s"] = m[label]
            out["reference_" + label + "_s"] = r[label]
    md, rd = pa / (pa @ dt), pb / (pb @ dt)
    out["shape_tv_pct"] = 50 * abs(md - rd) @ dt
    out["aligned_tv_pct"] = aligned_distance(
        edges=t, model_density=md, reference_density=rd, shift=times["centroid_delay"]
    )
    if (
        out["width_resolved"]
        and np.allclose(dt, dt[0])
        and r["modes"] == m["modes"] == 1
        and not any(
            out[k]
            for k in (
                "model_truncated",
                "reference_truncated",
                "model_active_at_start",
                "reference_active_at_start",
            )
        )
    ):
        corr = correlate(md, rd)
        lags = correlation_lags(len(md), len(rd))
        winners = np.flatnonzero(np.isclose(corr, corr.max(), rtol=1.0e-8, atol=0))
        if len(winners) == 1:
            times["xcorr_lag"] = lags[winners[0]] * dt[0]
    for label, value in times.items():
        out[label + "_s"] = float(value)
        out[label + "_pct"] = float(100 * value / span)
    return out
