"""Re-run the independent SWE cases through the omitted-method public factory."""
import json
from pathlib import Path

from validation.fv1_diffusion_truth import run
from validation.fv1_riemann import report_json
from validation.fv2_accuracy import convergence
from validation.fv2_regime_jump_history import measure


def verify_qualification(*, output, archive, names):
    """Compare all hydraulic metrics and trajectories, including accepted timestep controls."""
    counts = {}
    for name in names:
        candidate = json.loads((output / name).read_text())
        previous = json.loads((archive / name).read_text())
        if name == "manning.json":
            candidate, previous = candidate["rows"], previous["rows"]
        previous = [row for row in previous if row["method"] == "hancock_adaptive"]
        if not previous:
            raise ValueError(f"Missing historical adaptive results in {name}")
        for actual, expected in zip(candidate, previous, strict=True):
            if actual.pop("method") != "default":
                raise ValueError(f"Unexpected solver route in {name}")
            del expected["method"]
            for row in (actual, expected):
                for stats in row.get("stats", []):
                    for timing in ("compilation_seconds", "graph_capture_seconds", "loop_seconds", "wall_seconds"):
                        del stats[timing]
            if actual != expected:
                raise AssertionError(f"Hydraulic results differ in {name} at {actual.get('dx', actual.get('dx_m'))} m")
        counts[name] = len(previous)
    return counts


if __name__ == "__main__":
    output = Path("validation/hancock_default_review")
    archive = Path("validation/fv1_diffusion_review")
    comparisons = ("analytical.json", "controls.json", "rain.json", "manning.json", "radial.json", "shock_histories.json")
    device = "cuda:0"
    spacings = (10., 5., 2.5)
    grids = (16, 32, 64, 128)
    timesteps = (1 / 64, 1 / 128, 1 / 256, 1 / 512)
    waves = ("lake", "ritter", "stoker", "shear", "thacker", "jump", "jump_offset")
    controls = (("sloping_ritter", .05), ("subcritical", 0.), ("transcritical", 0.),
                ("transcritical_jump", 0.))
    run(output=output, methods=("default",), spacings=spacings, grids=grids, device=device,
        integrations=(), time_steps=timesteps, truth_cases=controls, wave_cases=waves)
    manufactured = convergence(grids=grids, temporal_grid=16, timesteps=timesteps,
        length=8., duration=.5, device=device, bed_amplitude=.04, limiter="adaptive", integrator="hancock")
    report_json(path=output / "manufactured.json", content=manufactured)
    shock_spacings = (5., 2.5, 1.25)
    histories = [measure(dx=dx, method="default", duration=80., interval=.5, device=device,
                         grade=.4, position=22.5, length=1920., width=20., gravity=9.81)
                 for dx in shock_spacings]
    report_json(path=output / "shock_histories.json", content=histories)
    counts = verify_qualification(output=output, archive=archive, names=comparisons)
    report_json(path=output / "verified_counts.json", content=counts)
