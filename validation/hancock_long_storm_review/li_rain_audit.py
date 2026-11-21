"""Reconcile LI's native rainfall ledger with compensated quadrature of the device forcing."""

import inspect
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from floodwarp import local_inertial_kernels as li
from floodwarp.model import advance
from validation.hancock_long_storm_review.landforms import storm_model


@wp.kernel
def sum_rain(
    ctrl: wp.array(dtype=wp.float64),
    applied: wp.array2d(dtype=wp.float64),
    area: wp.float64,
    output_accumulates: int,
    totals: wp.array(dtype=wp.float64),
):
    if ctrl[li.DL_DONE] == wp.float64(0.0):
        depth = ctrl[li.DL_RAIN_RATE] * ctrl[li.DL_DT]
        v = depth * area - totals[1]
        updated = totals[0] + v
        totals[1] = (updated - totals[0]) - v
        totals[0] = updated
        totals[2] += wp.float64(1.0)
        if output_accumulates == 0:
            totals[3] = wp.max(totals[3], wp.abs(applied[0, 0] - depth))


def audit(*, review, name, device):
    row = json.loads((review / "events" / (name + ".json")).read_text())
    config = json.loads((review / "cases.json").read_text())
    model, expected = storm_model(
        config=config,
        landform=row["landform"],
        storm=row["storm"],
        method=row["method"],
        mode=row["mode"],
        dx=row["dx"],
        terrain_dx=row["terrain_dx"],
        dimensions=row["dimensions_m"],
        duration=row["hours"] * 3600,
        device=device,
    )
    buf = wp.zeros(4, dtype=wp.float64, device=device)
    wp.load_module(module=sum_rain.module, device=device)
    original = wp.launch
    parameters = list(
        inspect.signature(li.device_loop_apply_sources_kernel.func).parameters
    )
    area = float(np.prod(row["dimensions_m"]))
    inputs_seen = set()

    def trace(kernel, *args, **kwargs):
        result = original(kernel, *args, **kwargs)
        if kernel is li.device_loop_apply_sources_kernel:
            inputs = dict(zip(parameters, kwargs["inputs"], strict=True))
            inputs_seen.add(
                (
                    int(inputs["accumulate_accounting"]),
                    int(inputs["accumulate_output"]),
                    int(inputs["use_uniform_rate"]),
                )
            )
            original(
                kernel=sum_rain,
                dim=1,
                device=device,
                inputs=[
                    inputs["control"],
                    inputs["applied_rain"],
                    area,
                    inputs["accumulate_output"],
                    buf,
                ],
            )
        return result

    with patch.object(wp, "launch", trace):
        for s in row["samples"]:
            stats = advance(model=model, end_time=s["t_s"])
            if stats["steps"] != s["steps"]:
                raise ArithmeticError(
                    "LI instrumentation changed the accepted trajectory"
                )
    actual = buf.numpy()
    ledger = model.solver.cumulative_accounting()
    if int(actual[2]) != row["steps"] or actual[3] != 0:
        raise ArithmeticError(
            "Rain depth or step count differs from the device forcing"
        )
    np.testing.assert_allclose(actual[0], expected, rtol=1e-12, atol=1e-10)
    data = dict(
        run=name,
        steps=row["steps"],
        native_m3=ledger["rain"],
        archived_native_m3=row["rain_m3"],
        expected_m3=expected,
        compensated_m3=actual[0],
        maximum_applied_depth_difference_m=actual[3],
        source_accounting_modes=[list(v) for v in sorted(inputs_seen)],
    )
    (review / "li_rain_audit.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    review = Path(__file__).parent
    name = "refinement_240x160_plain_moderate_li_native_1m_10dem_24h"
    device = "cuda:0"
    audit(review=review, name=name, device=device)
