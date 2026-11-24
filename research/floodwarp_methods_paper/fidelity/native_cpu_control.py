"""Exercise the full six-hour forcing and native storm restart on a small CPU control."""

import json
from pathlib import Path

import numpy as np
from floodwarp.model import advance, diagnostics, load_checkpoint, save_checkpoint
from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.fidelity import mean_storms, native_matrix
from research.floodwarp_methods_paper.terrain.generators import native_inputs


def run(*, directory, design):
    study = directory.parent
    case = json.loads((directory / "mean_settings.json").read_text())
    base, inputs = mean_storms.base_inputs(directory=directory, design=case)
    spec = dict(
        base,
        case=design["case"],
        dimensions=design["dimensions_m"],
        dx=design["dx_m"],
        dem=design["dx_m"],
        method="hancock_dense",
        requested_method="hancock_auto",
        initial_grid="native_1m_block_mean",
        terrain_aggregation="native_1m_block_mean",
        hours=design["end_h"],
        factor=1.0,
    )
    cache = study.parents[1] / ".fidelity-restarts"
    folder, receipt = native_inputs.generate(
        spec=spec,
        inputs=inputs,
        cache_root=cache / "native_inputs",
        strip_rows=design["strip_rows"],
        include_native=True,
        channel_core_m=case["channel_core_min_depth_m"],
    )
    definition = dict(
        spec=spec,
        physical_sha=runner.digest(receipt),
        design=design,
        numerical_sources=runner.source_identity(
            root=study.parents[1], study=study, stage="long-storms"
        ),
        sources={
            p.name: runner.checksum(p)
            for p in (
                Path(__file__),
                Path(native_matrix.__file__),
                Path(native_matrix.native_event.__file__),
            )
        },
    )
    ident = runner.digest(definition)[:24]
    target = directory / "native_cpu_control" / ident
    target.mkdir(parents=True, exist_ok=True)
    if (target / "result.json").is_file():
        return
    kwargs = dict(
        spec=spec,
        bundle=folder,
        receipt=receipt,
        config=inputs["cfg"],
        template=inputs["template"],
        device="cpu",
    )
    direct, rain, forcing = native_matrix.prepare(**kwargs)
    restart, second_rain, _ = native_matrix.prepare(**kwargs)
    if second_rain != rain:
        raise ValueError("Fresh runs produced different rainfall inputs")
    end = design["end_h"] * 3600
    cutoff = design["checkpoint_h"] * 3600
    checkpoint = cache / "native_cpu" / ident / "state.npz"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in range(design["interval_s"], end + 1, design["interval_s"]):
        first = advance(model=direct, end_time=t, use_cuda_graph=False)
        second = advance(model=restart, end_time=t, use_cuda_graph=False)
        if first["steps"] != second["steps"]:
            raise ArithmeticError("Restart changed accepted timestep count")
        np.testing.assert_array_equal(direct.solver.q.numpy(), restart.solver.q.numpy())
        if t == cutoff:
            save_checkpoint(model=restart, path=checkpoint)
            restart, _, _ = native_matrix.prepare(**kwargs)
            load_checkpoint(model=restart, path=checkpoint)
        account = diagnostics(model=direct)
        rows.append(
            dict(
                time_s=t,
                accepted=first["steps"],
                rain_m3=account["rain"],
                infiltration_m3=account["infiltration"],
                storage_m3=account["storage_m3"],
                outflow_m3=account["boundary_outflow"],
                residual_m3=account["residual_m3"],
            )
        )
    if direct.solver.time != end or cutoff >= end:
        raise ArithmeticError("Incomplete storm and restart control")
    np.testing.assert_allclose(rows[-1]["rain_m3"], rain, rtol=1e-9, atol=1e-11)
    for k in ("maximum", "volume_x", "volume_y", "rain_sum", "soil"):
        np.testing.assert_array_equal(
            direct.solver._fields[k].numpy(), restart.solver._fields[k].numpy()
        )
    residual = max(abs(r["residual_m3"]) for r in rows)
    if residual > 1e-8 * max(1, rain + direct.solver.initial_volume):
        raise ArithmeticError("Invalid storm water balance")
    runner.write_json(
        path=target / "result.json",
        data=dict(
            inputs=definition,
            device="cpu",
            expected_rain_m3=rain,
            forcing=forcing,
            rows=rows,
            max_abs_residual_m3=residual,
            checkpoint_h=design["checkpoint_h"],
            end_h=design["end_h"],
            accepted=sum(r["accepted"] for r in rows),
            native_sha256=receipt["native_sha256"],
        ),
    )
    print(
        "Native CPU restart control",
        design["end_h"],
        "hours",
        sum(r["accepted"] for r in rows),
        "accepted steps",
        flush=True,
    )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    design = json.loads((directory / "native_cpu_settings.json").read_text())
    run(directory=directory, design=design)
