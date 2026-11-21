"""Replay the retained analytical drivers with a scheduling-only override."""

import os
from pathlib import Path
from unittest.mock import patch

from floodwarp.fv2_solver import HancockSolver
from validation.fv1_diffusion_truth import run
from validation.fv1_riemann import report_json
from validation.fv2_accuracy import convergence
from validation.fv2_regime_jump_history import measure
from validation.hancock_default_qualification import verify_qualification


def replay(
    *,
    output,
    archive,
    spacings,
    shock_spacings,
    device,
    grids,
    timesteps,
    waves,
    controls,
):
    configure = HancockSolver.configure_execution

    def configure_tiles(self, **options):
        configure(self, **dict(options, mode="tiles"))

    with patch.object(HancockSolver, "configure_execution", configure_tiles):
        run(
            output=output,
            methods=("default",),
            spacings=spacings,
            grids=grids,
            device=device,
            integrations=(),
            time_steps=timesteps,
            truth_cases=controls,
            wave_cases=waves,
        )
        result = convergence(
            grids=grids,
            temporal_grid=16,
            timesteps=timesteps,
            length=8.0,
            duration=0.5,
            device=device,
            bed_amplitude=0.04,
            limiter="adaptive",
            integrator="hancock",
        )
        report_json(path=output / "manufactured.json", content=result)
        histories = [
            measure(
                dx=dx,
                method="default",
                duration=80.0,
                interval=0.5,
                device=device,
                grade=0.4,
                position=22.5,
                length=1920.0,
                width=20.0,
                gravity=9.81,
            )
            for dx in shock_spacings
        ]
        report_json(path=output / "shock_histories.json", content=histories)
    names = (
        "controls.json",
        "analytical.json",
        "manning.json",
        "rain.json",
        "radial.json",
        "shock_histories.json",
    )
    counts = verify_qualification(output=output, archive=archive, names=names)
    report_json(path=output / "verified_counts.json", content=counts)


if __name__ == "__main__":
    output = Path(
        os.environ.get(
            "FLOODWARP_REGRESSION_RESULTS",
            "validation/hancock_long_storm_review/regression",
        )
    )
    archive = Path("validation/fv1_diffusion_review")
    controls = (
        ("sloping_ritter", 0.05),
        ("subcritical", 0.0),
        ("transcritical", 0.0),
        ("transcritical_jump", 0.0),
    )
    waves = ("lake", "ritter", "stoker", "shear", "thacker", "jump", "jump_offset")
    device = "cuda:0"
    spacings = (10.0, 5.0, 2.5)
    shock_spacings = (5.0, 2.5, 1.25)
    grids = (16, 32, 64, 128)
    timesteps = (1 / 64, 1 / 128, 1 / 256, 1 / 512)
    replay(
        output=output,
        archive=archive,
        spacings=spacings,
        shock_spacings=shock_spacings,
        device=device,
        grids=grids,
        timesteps=timesteps,
        controls=controls,
        waves=waves,
    )
