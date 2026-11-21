"""Exact full-opening exchange between two equal-area reservoirs under a common storm."""

import json
from pathlib import Path

import numpy as np

from floodwarp.inputs import terrain_from_array
from floodwarp.model import advance, create_model


def exact_depths(*, time, nodes, rates, initial, reservoir_area, coefficient):
    head = initial[0] - initial[1]
    root = max(0.0, np.sqrt(abs(head)) - coefficient * time / reservoir_area)
    knots = np.r_[nodes[nodes < time], time]
    rain = np.trapezoid(np.interp(knots, nodes, rates), knots)
    mean = np.mean(initial) + rain
    difference = np.sign(head) * root**2 / 2
    return np.array([mean + difference, mean - difference])


def simulate(
    *,
    geometry,
    opening_area,
    initial,
    nodes,
    rates,
    times,
    shape,
    spacing,
    cap,
    gravity,
    losses,
    device,
):
    z = np.zeros(shape)
    mask = np.zeros(shape, dtype=bool)
    mask[0, [0, -1]] = True
    h = z.copy()
    h[0, [0, -1]] = initial
    terrain = terrain_from_array(
        elevation=z,
        active=mask,
        dx=spacing,
        dy=spacing,
        row_order="south_to_north",
        vertical_datum="local",
    )
    incident = dict(
        inlet_row=0,
        inlet_col=0,
        outlet_row=0,
        outlet_col=shape[1] - 1,
        entrance_loss_coefficient=losses[0],
        exit_loss_coefficient=losses[1],
        **geometry,
    )
    model = create_model(
        terrain=terrain,
        structures=[incident],
        initial_depth=h,
        gravity=gravity,
        execution_mode="tiles",
        explicit_maximum_dt=cap,
        rainfall_series=(nodes, rates),
        device=device,
    )
    coefficient = opening_area * np.sqrt(2 * gravity / (1 + sum(losses)))
    samples = []
    for t in times:
        advance(model=model, end_time=t)
        actual = model.solver.get_water_depth()[0, [0, -1]]
        expected = exact_depths(
            time=t,
            nodes=nodes,
            rates=rates,
            initial=initial,
            reservoir_area=spacing**2,
            coefficient=coefficient,
        )
        ledger = model.solver.get_structure_diagnostics()[0]
        exact_transfer = (
            spacing**2 * ((initial[0] - initial[1]) - (expected[0] - expected[1])) / 2
        )
        np.testing.assert_allclose(
            ledger["net_volume_m3"],
            spacing**2 * ((initial[0] - initial[1]) - (actual[0] - actual[1])) / 2,
            rtol=3e-12,
            atol=5e-8,
        )
        samples.append(
            dict(
                t_s=t,
                model_m=actual.tolist(),
                reference_m=expected.tolist(),
                signed_model_m3=ledger["net_volume_m3"],
                signed_reference_m3=exact_transfer,
            )
        )
    e = np.asarray([s["model_m"] for s in samples]) - np.asarray(
        [s["reference_m"] for s in samples]
    )
    return dict(
        type=geometry["id"],
        cap_s=cap,
        device=device,
        initial_m=initial,
        rmse_m=float(np.sqrt(np.mean(e**2))),
        samples=samples,
        accounting=model.solver.cumulative_accounting(),
    )


def evaluate(
    *,
    review,
    cases,
    caps,
    nodes,
    rates,
    times,
    shape,
    spacing,
    gravity,
    losses,
    device,
):
    records = []
    for geometry, area, initial in cases:
        previous = None
        for cap in caps:
            row = simulate(
                geometry=geometry,
                opening_area=area,
                initial=initial,
                nodes=nodes,
                rates=rates,
                times=times,
                shape=shape,
                spacing=spacing,
                cap=cap,
                gravity=gravity,
                losses=losses,
                device=device,
            )
            if previous is not None and row["rmse_m"] >= previous:
                raise ArithmeticError(
                    "Submerged-opening timestep refinement did not improve agreement"
                )
            previous = row["rmse_m"]
            records.append(row)
    (review / "opening_event_cpu.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    review = Path(__file__).parent
    config = json.loads((review / "cases.json").read_text())
    nodes = np.r_[config["hours"], 8.0] * 3600
    rates = np.r_[config["storms_mm_h"]["moderate"], 0.0] / 3.6e6
    times = np.arange(900.0, 28801.0, 900.0)
    cases = (
        (
            dict(
                id="circular",
                type="culvert",
                shape="circular",
                diameter_m=0.8,
                inlet_invert_m=0.0,
                outlet_invert_m=0.0,
            ),
            np.pi * 0.8**2 / 4,
            [1.25, 1.0],
        ),
        (
            dict(
                id="box",
                type="culvert",
                shape="box",
                width_m=0.8,
                height_m=0.7,
                inlet_invert_m=0.0,
                outlet_invert_m=0.0,
            ),
            0.8 * 0.7,
            [1.0, 1.25],
        ),
        (
            dict(
                id="bridge",
                type="bridge",
                opening_width_m=1.2,
                opening_height_m=0.7,
                opening_invert_m=0.0,
            ),
            1.2 * 0.7,
            [1.25, 1.0],
        ),
    )
    caps = (15.0, 5.0)
    shape = (1, 130)
    spacing = 300.0
    gravity = 9.81
    losses = (0.5, 1.0)
    device = "cpu"
    evaluate(
        review=review,
        cases=cases,
        caps=caps,
        nodes=nodes,
        rates=rates,
        times=times,
        shape=shape,
        spacing=spacing,
        gravity=gravity,
        losses=losses,
        device=device,
    )
