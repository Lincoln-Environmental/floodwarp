"""Verify optional directional CFL and both tile policies against independent solutions."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from floodwarp.fv2_solver import HancockSolver
from validation.fv1_diffusion_domains import design
from validation.fv1_diffusion_matrix import forecast
from validation.fv1_diffusion_truth import run
from validation.fv1_riemann import report_json
from validation.fv2_accuracy import convergence
from validation.fv2_regime_jump_history import measure


@contextmanager
def numerical_options(*, policy):
    original = HancockSolver.__init__

    def initialise(self, **options):
        original(self, **dict(options, execution_mode="tiles", cfl_policy=policy))

    with patch.object(HancockSolver, "__init__", initialise):
        yield


def run_controls(
    *, review, policies, settings, spacings, grids, timesteps, waves, controls
):
    for policy in policies:
        output = review / ("analytical_" + policy)
        with numerical_options(policy=policy):
            run(
                output=output,
                methods=("default",),
                spacings=spacings,
                grids=grids,
                device=settings["device"],
                integrations=(),
                time_steps=timesteps,
                truth_cases=controls,
                wave_cases=waves,
            )
            report_json(
                path=output / "manufactured.json",
                content=convergence(
                    grids=grids,
                    temporal_grid=16,
                    timesteps=timesteps,
                    length=8.0,
                    duration=0.5,
                    device=settings["device"],
                    bed_amplitude=0.04,
                    limiter="adaptive",
                    integrator="hancock",
                ),
            )
            report_json(
                path=output / "shock_history.json",
                content=[
                    measure(
                        dx=dx,
                        method="default",
                        duration=80.0,
                        interval=0.5,
                        device=settings["device"],
                        grade=0.4,
                        position=22.5,
                        length=1920.0,
                        width=20.0,
                        gravity=9.81,
                    )
                    for dx in spacings
                ],
            )
            rows = [
                forecast(
                    case=case,
                    dx=dx,
                    method="hancock_adaptive",
                    policy="production",
                    multiplier=1.0,
                    settings=settings,
                    keep_series=False,
                    repeats=0,
                )
                for case in design()
                for dx in spacings
            ]
            report_json(path=output / "normal_policy.json", content=rows)


if __name__ == "__main__":
    review = Path(__file__).parent
    policies = ("legacy", "directional")
    settings = dict(
        device="cuda:0",
        gravity=9.80665,
        samples=12,
        order=16,
        max_dt=5.0,
        wet=(0.001, 0.01, 0.05),
        li_floor=1.0e-8,
    )
    spacings = (10.0, 5.0, 1.0)
    grids = (16, 32, 64, 128)
    timesteps = (1 / 64, 1 / 128, 1 / 256, 1 / 512)
    waves = ("lake", "ritter", "stoker", "shear", "thacker", "jump", "jump_offset")
    controls = (
        ("sloping_ritter", 0.05),
        ("subcritical", 0.0),
        ("transcritical", 0.0),
        ("transcritical_jump", 0.0),
    )
    run_controls(
        review=review,
        policies=policies,
        settings=settings,
        spacings=spacings,
        grids=grids,
        timesteps=timesteps,
        waves=waves,
        controls=controls,
    )
