"""Two isolated basins joined by a sharp-crested weir during a six-hour storm.

The low receiver remains below crest. The reference integrates the continuous
weir rating and rainfall independently of the surface/structure stepper.
"""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from floodwarp.inputs import terrain_from_array
from floodwarp.model import create_model, advance


def basin_event(
    *,
    directory,
    nodes,
    rates,
    times,
    caps,
    modes,
    dx,
    crest,
    coefficient,
    length,
    gravity,
    device,
):
    a = dx**2
    factor = coefficient * length * (2 / 3) * np.sqrt(2 * gravity)

    def rhs(t, state):
        q = factor * max(0.0, state[0] - crest) ** 1.5
        return [np.interp(t, nodes, rates) - q / a, q / a]

    trace = solve_ivp(
        rhs,
        (0.0, times[-1]),
        [0.0, 0.0],
        rtol=2.0e-12,
        atol=1.0e-15,
        max_step=30.0,
        dense_output=True,
    ).sol(times)
    z = np.zeros((1, 130))
    z[0, -1] = -0.25
    active = np.zeros_like(z, dtype=bool)
    active[0, [0, -1]] = True
    terrain = terrain_from_array(
        elevation=z,
        active=active,
        dx=dx,
        dy=dx,
        row_order="south_to_north",
        vertical_datum="local",
    )
    weir = dict(
        id="spillway",
        type="weir",
        weir_type="sharp_crested",
        inlet_row=0,
        inlet_col=0,
        outlet_row=0,
        outlet_col=z.shape[1] - 1,
        crest_elevation_m=crest,
        crest_length_m=length,
        discharge_coefficient=coefficient,
    )
    multiplier = np.zeros_like(z)
    multiplier[0, 0] = 1.0
    rows = []
    for cap in caps:
        for mode in modes:
            m = create_model(
                terrain=terrain,
                structures=[weir],
                rainfall_series=(nodes, rates),
                gravity=gravity,
                explicit_maximum_dt=cap,
                execution_mode=mode,
                device=device,
            )
            s = m.solver
            s.set_rainfall_spatial_multiplier(multiplier=multiplier)
            history = []
            for j, t in enumerate(times):
                stats = advance(model=m, end_time=t)
                depths = s.get_water_depth()[0, [0, -1]]
                data = s.get_structure_diagnostics()[0]
                assert depths[1] - 0.25 < crest
                np.testing.assert_allclose(
                    data["net_volume_m3"], a * depths[1], rtol=4.0e-13, atol=1.0e-12
                )
                history.append(
                    dict(
                        t_s=t,
                        depths_m=depths.tolist(),
                        reference_depth_m=trace[:, j].tolist(),
                        steps=stats["steps"],
                        loop_s=stats["loop_seconds"],
                        **data,
                    )
                )
            rows.append(
                dict(
                    dx=dx,
                    mode=mode,
                    cap_s=cap,
                    rmse_m=float(
                        np.mean(
                            (np.array([v["depths_m"] for v in history]).T - trace) ** 2
                        )
                        ** 0.5
                    ),
                    ledger=s.cumulative_accounting(),
                    history=history,
                )
            )
    (directory / ("structure_event_" + device.split(":")[0] + ".json")).write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    review = Path(__file__).parent
    config = json.loads((review / "cases.json").read_text())
    nodes = np.r_[config["hours"], 8.0] * 3600.0
    rates = np.r_[config["storms_mm_h"]["moderate"], 0.0] / 3.6e6
    times = np.arange(900.0, 28801.0, 900.0)
    caps = (60.0, 15.0, 5.0)
    modes = ("dense", "tiles")
    import os

    device = os.environ.get("FLOODWARP_STRUCTURE_DEVICE", "cuda:0")
    basin_event(
        directory=review,
        nodes=nodes,
        rates=rates,
        times=times,
        caps=caps,
        modes=modes,
        dx=100.0,
        crest=0.025,
        coefficient=0.62,
        length=2.0,
        gravity=9.81,
        device=device,
    )
