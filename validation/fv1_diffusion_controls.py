"""Additional independent reduced-equation and full-SWE controls at matrix spacings."""

from pathlib import Path
from validation import (
    fv2_flow_truth,
    fv2_equilibria,
    fv2_local_inertia,
    fv2_rain_flow,
    fv2_radial_flow,
    fv2_macdonald_manning,
)
from validation.fv1_riemann import report_json


def controls(*, directory, device, methods, spacings, li_cases):
    fv2_local_inertia.experiment(
        directory=directory,
        cases=li_cases,
        spacings=spacings,
        methods=(*methods, "li"),
        n_values=(1e-8,),
        gravity=9.80665,
        duration=8.0,
        length=400.0,
        width=200.0,
        cap_ratio=1 / 32,
        device=device,
        repeats=3,
        wet=0.01,
    )
    rows = fv2_equilibria.controls(spacings=spacings, device=device, methods=methods)
    report_json(path=directory / "oblique_steady_shear.json", content=rows)
    common = dict(spacings=(min(spacings),), methods=methods, device=device)
    fv2_flow_truth.run(
        output=directory / "controls_1m.json",
        cases=(
            ("subcritical", 0.0),
            ("supercritical", 0.0),
            ("transcritical", 0.0),
            ("transcritical_jump", 0.0),
        ),
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
    fv2_rain_flow.run(
        output=directory / "rain_1m.json",
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
    fv2_macdonald_manning.run(
        output=directory / "manning_1m.json",
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
    fv2_radial_flow.run(
        output=directory / "radial_1m.json",
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
    methods = (
        "fv1",
        "low_froude_hll",
        "euler_mc",
        "euler_minmod",
        "hancock_mc",
        "hancock_minmod",
        "mc",
        "minmod",
        "adaptive",
    )
    li_cases = (
        "weak_dam",
        "stoker",
        "ritter",
        "damped_contact",
        "deep_rain",
        "thin_rain",
        "accelerating_plane",
        "lake",
    )
    spacings = (10.0, 5.0, 1.0)
    controls(directory=directory, device="cuda:0", methods=methods, spacings=spacings, li_cases=li_cases)
