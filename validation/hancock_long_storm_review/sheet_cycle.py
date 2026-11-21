"""Uniform SWE sheet with initial discharge, six hours of rain and exact frictional recession."""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from floodwarp.model import create_model, advance
from floodwarp.inputs import terrain_from_array


def analytical(*, t, nodes, rain, h0, q0, manning, gravity):
    cumulative = np.r_[0.0, np.cumsum(0.5 * (rain[:-1] + rain[1:]) * np.diff(nodes))]

    def depth(t):
        j = min(np.searchsorted(nodes, t, side="right") - 1, len(nodes) - 2)
        dt = t - nodes[j]
        return (
            h0
            + cumulative[j]
            + rain[j] * dt
            + 0.5 * (rain[j + 1] - rain[j]) * dt**2 / (nodes[j + 1] - nodes[j])
        )

    def inverse_depth_power(t):
        return depth(t) ** (-7 / 3)

    def discharge(t):
        interior = nodes[(nodes > 0) & (nodes < t)]
        integral = quad(inverse_depth_power, 0.0, t, points=interior, epsrel=1.0e-12)[0]
        return 1.0 / (1.0 / q0 + gravity * manning**2 * integral)

    discharge_volume = quad(
        discharge, 0.0, t, points=nodes[(nodes > 0) & (nodes < t)], epsrel=1.0e-10
    )[0]
    return depth(t), discharge_volume


def check_cycle(
    *,
    directory,
    methods,
    nodes,
    rain,
    dx,
    length,
    width,
    h0,
    q0,
    manning,
    gravity,
    interval,
    device,
):
    span = nodes[-1]
    # Twice the maximum gravity speed includes the endpoint dry-bed Riemann bound.
    max_h = h0 + np.trapezoid(rain, nodes)
    if length / 2 <= span * (q0 / h0 + 2 * np.sqrt(gravity * max_h)):
        raise ValueError("Central section enters the boundary signal cone")
    times = np.arange(interval, span + 1.0, interval)
    truth = np.array(
        [
            analytical(
                t=t,
                nodes=nodes,
                rain=rain,
                h0=h0,
                q0=q0,
                manning=manning,
                gravity=gravity,
            )
            for t in times
        ]
    )
    terrain = terrain_from_array(
        elevation=np.zeros((round(width / dx), round(length / dx))),
        dx=dx,
        dy=dx,
        row_order="south_to_north",
        vertical_datum="local",
    )
    rows = []
    for method, mode in methods:
        opt = (
            dict(execution_mode=mode, gravity=gravity)
            if method != "li"
            else dict(
                track_cumulative_face_volumes=True, track_absolute_face_volumes=False
            )
        )
        if method == "fv2":
            opt["limiter"] = "adaptive"
        m = create_model(
            terrain=terrain,
            solver_method=method,
            initial_depth=h0,
            manning_n=manning,
            rainfall_series=(nodes, rain),
            device=device,
            **opt,
        )
        s = m.solver
        if method == "li":
            qx = np.full((s.ny, s.nx + 1), q0)
            qx[:, [0, -1]] = 0.0
            s.set_face_fluxes(qx=qx, qy=np.zeros((s.ny + 1, s.nx)))
        else:
            s.set_momentum(hu=q0, hv=0.0)
        samples = []
        for i, t in enumerate(times):
            stats = advance(model=m, end_time=t)
            volume = s.get_cumulative_face_volumes()[
                "cumulative_net_qx_volume_m3" if method == "li" else "qx"
            ]
            samples.append(
                dict(
                    t_s=t,
                    depth_m=float(s.get_water_depth()[:, s.nx // 2].mean()),
                    volume_m2=volume[:, s.nx // 2].sum() / width,
                    exact_depth_m=truth[i, 0],
                    exact_volume_m2=truth[i, 1],
                    steps=stats["steps"],
                    loop_s=stats["loop_seconds"],
                )
            )
        change = np.array([v["volume_m2"] for v in samples]) - truth[:, 1]
        rows.append(
            dict(
                method=method,
                mode=mode,
                section_discharge_rmse_m2s=float(
                    np.mean((np.diff(np.r_[0.0, change]) / interval) ** 2) ** 0.5
                ),
                final_volume_error_m2=change[-1],
                depth_rmse_m=float(
                    np.mean(
                        (np.array([v["depth_m"] for v in samples]) - truth[:, 0]) ** 2
                    )
                    ** 0.5
                ),
                samples=samples,
            )
        )
    (directory / "sheet_cycle.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    directory = Path(__file__).parent
    config = json.loads((directory / "cases.json").read_text())
    nodes = np.r_[config["hours"], 8.0] * 3600.0
    rain = np.r_[config["storms_mm_h"]["moderate"], 0.0] / 3.6e6
    methods = (
        ("fv1", "sparse"),
        ("hancock", "tiles"),
        ("fv2", "dense"),
        ("li", "native"),
    )
    check_cycle(
        directory=directory,
        methods=methods,
        nodes=nodes,
        rain=rain,
        dx=10.0,
        length=144000.0,
        width=20.0,
        h0=0.03,
        q0=0.006,
        manning=0.035,
        gravity=9.80665,
        interval=900.0,
        device="cuda:0",
    )
