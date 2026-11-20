"""Independent full-SWE verification of the one-flux experimental schemes."""

from pathlib import Path
from validation import fv2_flow_truth, fv2_macdonald_manning, fv2_radial_flow, fv2_rain_flow
from validation.fv1_riemann import report_json
from validation.fv2_accuracy import analytical, convergence


def run(*, output, methods, spacings, grids, device, integrations, time_steps, truth_cases, wave_cases):
    output.mkdir(exist_ok=True, parents=True)
    rows = {}
    for integration in integrations:
        rows[integration] = convergence(
            grids=grids,
            temporal_grid=16,
            timesteps=time_steps,
            length=8.0,
            duration=0.5,
            device=device,
            bed_amplitude=0.04,
            integrator=integration,
        )
        report_json(path=output / "manufactured.json", content=rows)
    common = dict(spacings=spacings, methods=methods, device=device)
    analytical(output=output / "analytical.json", names=wave_cases, length=640.0, width=40.0, **common)
    fv2_flow_truth.run(
        output=output / "controls.json",
        cases=truth_cases,
        gauges=(0.0, 40.0, 80.0),
        duration=20.0,
        interval=1.0,
        length=640.0,
        width=40.0,
        gravity=9.81,
        head=2.0,
        height=0.4,
        half_width=60.0,
        discharge=1.5,
        jump=22.5,
        quadrature=16,
        timestep_ratio=1 / 64,
        window=(-140.0, 160.0),
        wet=0.01,
        **common,
    )
    fv2_macdonald_manning.run(
        output=output / "manning.json",
        gauges=(-80.0, 0.0, 80.0),
        length=480.0,
        width=40.0,
        duration=20.0,
        interval=2.0,
        depth_at_zero=0.5,
        decay=0.001,
        discharge=0.5,
        manning=0.025,
        gravity=9.81,
        **common,
    )
    fv2_rain_flow.run(
        output=output / "rain.json",
        cases=(("thin_transcritical", 0.03, 0.004), ("deep_subcritical", 0.15, 0.03)),
        gauges=(-60.0, 0.0, 60.0),
        duration=20.0,
        interval=2.0,
        length=480.0,
        width=40.0,
        mean_rain=2e-5,
        manning=0.025,
        roughness_change=0.015,
        gravity=9.81,
        window=100.0,
        cap_ratio=1 / 32,
        wet=0.01,
        **common,
    )
    fv2_radial_flow.run(
        output=output / "radial.json",
        gauges=(40.0, 80.0, 120.0),
        gravity=9.81,
        curvature=0.0001,
        first_radius=80.0,
        second_radius=160.0,
        length=480.0,
        cycles=1,
        samples=48,
        quadrature=16,
        wet=0.01,
        cap_ratio=1 / 32,
        **common,
    )


if __name__ == "__main__":
    directory = Path("validation/fv1_diffusion_review")
    spacings = (10.0, 5.0, 2.5)
    methods = (
        "fv1",
        "euler_mc",
        "hancock_mc",
        "hancock_low_froude",
        "hancock_adaptive",
        "low_froude_hll",
        "minmod",
        "mc",
        "adaptive",
        "hancock_minmod",
        "euler_minmod",
    )
    grids = (16, 32, 64, 128)
    integrations = ("hancock", "euler", "fv2")
    time_steps = (1 / 64, 1 / 128, 1 / 256, 1 / 512)
    wave_cases = ("lake", "ritter", "stoker", "shear", "thacker", "jump", "jump_offset")
    truth_cases = (
        ("sloping_ritter", 0.05),
        ("subcritical", 0.0),
        ("transcritical", 0.0),
        ("transcritical_jump", 0.0),
    )
    device = "cuda:0"
    run(
        output=directory,
        methods=methods,
        spacings=spacings,
        grids=grids,
        device=device,
        integrations=integrations,
        time_steps=time_steps,
        truth_cases=truth_cases,
        wave_cases=wave_cases,
    )
