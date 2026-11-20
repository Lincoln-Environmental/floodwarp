"""Requalify the adaptive reconstruction using the archived exact-solution operators."""

from pathlib import Path
from validation.fv1_riemann import report_json
from validation.fv2_accuracy import analytical, convergence
from validation.fv2_additional import fixed_time_space
from validation.fv2_equilibria import controls
from validation.fv2_macdonald_manning import run
from validation.fv2_reference_check import reference_check


if __name__ == "__main__":
    directory = Path("validation/fv2_review/regimes")
    solver = "adaptive"
    spacings = (20.0, 10.0, 5.0, 2.5)
    device = "cuda:0"
    mms = dict(length=8.0, duration=0.5, device=device, bed_amplitude=0.12, limiter=solver)
    report_json(
        path=directory / "adaptive_mms.json",
        content=fixed_time_space(grids=(16, 32, 64, 128), dt=1 / 1024, **mms),
    )
    report_json(
        path=directory / "adaptive_time.json",
        content=convergence(grids=(), temporal_grid=16, timesteps=(1 / 64, 1 / 128, 1 / 256, 1 / 512), **mms),
    )
    report_json(
        path=directory / "controls.json",
        content=controls(spacings=spacings, device=device, methods=("fv1", "minmod", "mc", solver)),
    )
    analytical(
        output=directory / "analytical.json",
        names=("lake", "ritter", "stoker", "thacker", "shear", "jump", "jump_offset"),
        methods=("fv1", "minmod", solver),
        spacings=spacings,
        device=device,
        length=640.0,
        width=40.0,
    )
    run(
        output=directory / "macdonald.json",
        device=device,
        methods=("fv1", "minmod", "mc", solver),
        spacings=spacings,
        gauges=(-80.0, 0.0, 80.0),
        length=640.0,
        width=40.0,
        duration=20.0,
        interval=5.0,
        depth_at_zero=0.5,
        decay=0.001,
        discharge=0.5,
        manning=0.025,
        gravity=9.81,
    )
    reference_check(
        output=directory / "terrain.json",
        cases=("plane", "steep", "irregular", "expansion", "rainfall", "barrier", "oblique_channel"),
        spacings=(10.0, 5.0, 2.5),
        base_dx=5.0,
        fine_grids=(1.25, 0.625),
        length=320.0,
        width=160.0,
        device=device,
        samples=4,
        wet=0.01,
        candidates=("fv1", "minmod", solver),
    )
