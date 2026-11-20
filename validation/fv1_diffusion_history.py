"""Hydrograph shape and late shock histories, isolated from steady-state GPU timings."""

import json
from pathlib import Path
import numpy as np
from validation.fv1_diffusion_domains import design
from validation.fv1_diffusion_matrix import forecast
from validation.fv2_regime_jump_history import measure
from validation.fv1_riemann import report_json


def interpolated_width(*, series, times):
    y = np.array(series)
    peak = y.argmax()
    if peak in (0, len(y) - 1):
        return None
    above = y >= 0.5 * y[peak]
    left = np.flatnonzero(~above[:peak])
    right = np.flatnonzero(~above[peak + 1 :])
    if not left.size or not right.size:
        return None
    i = left[-1]
    j = peak + 1 + right[0]
    level = 0.5 * y[peak]
    a = np.interp(level, y[i : i + 2], times[i : i + 2])
    b = np.interp(level, y[j - 1 : j + 1][::-1], times[j - 1 : j + 1][::-1])
    return float(b - a)


def histories(*, directory, settings, spacings, methods, shock_methods, shock_spacings):
    rows = []
    for case in design():
        if case.family not in ("pulse", "gravity_pulse"):
            continue
        for dx in spacings:
            for method in methods:
                r = forecast(
                    case=case, dx=dx, method=method, policy="common", multiplier=1.0, settings=settings
                )
                t = np.array(r["interval_end_s"]) - case.duration / (2 * settings["samples"])
                for key in ("exact_m2_s", "model_m2_s"):
                    q = np.array(r[key])
                    # Fit peak time to a parabola through the three nearest samples.
                    peaks = []
                    for v in q.T:
                        i = v.argmax()
                        peak = t[i]
                        if 0 < i < len(t) - 1 and (2 * v[i] - v[i + 1] - v[i - 1]) > 0:
                            peak += (
                                (t[1] - t[0]) * (v[i + 1] - v[i - 1]) / (2 * (2 * v[i] - v[i + 1] - v[i - 1]))
                            )
                        peaks.append(float(peak))
                    transit = peaks[-1] - peaks[0]
                    r[key + "_diagnostics"] = dict(
                        peak_times_s=peaks,
                        width_s=[interpolated_width(series=v, times=t) for v in q.T],
                        speed_m_s=float((case.gauges[-1] - case.gauges[0]) / transit)
                        if transit > 0
                        else None,
                    )
                rows.append(r)
        report_json(path=directory / "pulse_histories.json", content=rows)
    rows = []
    for method in shock_methods:
        for dx in shock_spacings:
            rows.append(
                measure(
                    dx=dx,
                    method=method,
                    duration=80.0,
                    interval=0.5,
                    device=settings["device"],
                    grade=0.4,
                    position=22.5,
                    length=1920.0,
                    width=20.0,
                    gravity=9.81,
                )
            )
        report_json(path=directory / "shock_histories.json", content=rows)
        print("shock", method, flush=True)


if __name__ == "__main__":
    directory = Path("validation/fv1_diffusion_review")
    settings = json.loads((directory / "matrix/configuration.json").read_text())["settings"]
    settings["samples"] = 60
    methods = ("fv1", "low_froude_hll", "euler_mc", "hancock_mc", "hancock_low_froude", "mc", "li")
    shock_methods = (
        "fv1",
        "euler_mc",
        "hancock_mc",
        "hancock_minmod",
        "hancock_adaptive",
        "hancock_low_froude",
        "mc",
        "adaptive",
    )
    histories(
        directory=directory,
        settings=settings,
        spacings=(1.0, 5.0, 10.0),
        methods=methods,
        shock_methods=shock_methods,
        shock_spacings=(5.0, 2.5, 1.25),
    )
