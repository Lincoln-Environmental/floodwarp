"""Measured error under wall-clock budgets for lowland controls and steep hydraulic problems."""

from pathlib import Path
import json
import numpy as np
from validation.fv1_riemann import report_json
from validation.fv1_riemann_cases import analytical_domain, coordinates, cell_averages
from validation.fv2_accuracy import configured
from validation.fv2_cost import timed, reset, benchmark
from validation.fv2_lowland import geometry, raster, simulate
from validation.fv2_terrain import terrain, open_boundaries


def floodplain(*, directory, methods, grids, cases, repeats, device):
    selections = json.loads((directory / "lowland.json").read_text())
    rows = []
    for case in cases:
        ref = next(r for r in selections if r["case"] == case and not r["fixed_terrain"])
        native = geometry(case=case, dx=0.625, length=160.0, width=80.0)
        for dx in grids:
            fields = raster(native=native, ratio=round(dx / 0.625), subdivide=1)
            for m in methods:
                control = next(r for r in ref["runs"] if r["dx"] == dx and r["method"] == m)
                rep = []
                for i in range(repeats + 1):
                    row = simulate(
                        fields=fields,
                        case=case,
                        dx=dx,
                        method=m,
                        cap=dx / 8,
                        interval=10.0,
                        count=12,
                        device=device,
                        length=160.0,
                        sections=(-40.0, 0.0, 40.0),
                        gravity=9.80665,
                    )
                    np.testing.assert_allclose(row["hydro"], control["flux_m3_s"], rtol=1e-11, atol=1e-11)
                    if i:
                        rep.append(
                            dict(
                                total_s=row["hydraulic"],
                                capture_s=row["capture"],
                                preload_s=row["compilation"],
                                steady_s=row["hydraulic"] - row["capture"] - row["compilation"],
                                steps=row["steps"],
                            )
                        )
                e = control["errors"]
                rows.append(
                    dict(
                        case=case,
                        method=m,
                        dx=dx,
                        depth_rmse=e["depth_rmse_m"],
                        section_rmse=float(np.mean(e["hydro_rmse"])),
                        replicates=rep,
                        median_s=float(np.median([r["steady_s"] for r in rep])),
                    )
                )
        report_json(path=directory / "lowland_cost.json", content=rows)


def synthetic(*, directory, cases, grids, methods, repeats, device):
    refs = json.loads((directory / "terrain.json").read_text())
    rows = []
    for case in cases:
        group = next(r for r in refs if r["case"] == case and not r["terrain_fixed_5m"])
        for dx in grids:
            d = terrain(name=case, dx=dx, length=320.0, width=160.0)
            for m in methods:
                control = next(r for r in group["candidates"] if r["dx"] == dx and r["method"] == m)
                s = configured(domain=d, method=m, cap=dx / 32, device=device)
                open_boundaries(solver=s, active=s._active, north_only=False)
                start = s.capture_state()
                warm = timed(solver=s, duration=d.duration, samples=4)
                reps = []
                for _ in range(repeats):
                    reset(solver=s, state=start)
                    reps.append(timed(solver=s, duration=d.duration, samples=4))
                np.testing.assert_allclose(
                    s.get_water_depth().max(), control["against_mc"]["depth_max_m"], atol=1e-12, rtol=1e-12
                )
                rows.append(
                    dict(
                        case=case,
                        dx=dx,
                        method=m,
                        warm=warm,
                        replicates=reps,
                        depth_rmse=control["against_mc"]["depth_rmse_m"],
                        median_s=float(np.median([r["full_hydraulic_s"] for r in reps])),
                    )
                )
        report_json(path=directory / "synthetic_cost.json", content=rows)


def exact_cost(*, directory, methods, grids, cases, repeats, device):
    rows = []
    for case in cases:
        for dx in grids:
            d = analytical_domain(name=case, dx=dx, length=640.0, width=320.0, gravity=9.81)
            x, _ = coordinates(length=640.0, width=320.0, dx=dx)
            exact = cell_averages(name=case, x=x, dx=dx, time=d.duration, gravity=9.81)
            for method in methods:
                row, _ = benchmark(
                    domain=d, method=method, mode="dense", device=device, repeats=repeats, exact=exact
                )
                row.update(case=case)
                rows.append(row)
        report_json(path=directory / "exact_cost.json", content=rows)


if __name__ == "__main__":
    directory = Path("validation/fv2_review/regimes")
    device = "cuda:0"
    repeats = 5
    floodplain(
        directory=directory,
        methods=("contact_hll", "minmod", "adaptive", "li"),
        grids=(10.0, 5.0, 2.5),
        cases=("wide_wave", "overbank", "embankment", "spatial_rain"),
        repeats=repeats,
        device=device,
    )
    synthetic(
        directory=directory,
        methods=("fv1", "minmod", "adaptive"),
        grids=(10.0, 5.0, 2.5),
        cases=("plane", "steep", "irregular", "barrier", "oblique_channel"),
        repeats=repeats,
        device=device,
    )
    exact_cost(
        directory=directory,
        methods=("fv1", "minmod", "adaptive"),
        grids=(20.0, 10.0, 5.0, 2.5, 1.25),
        cases=("stoker", "ritter", "shear"),
        repeats=repeats,
        device=device,
    )
