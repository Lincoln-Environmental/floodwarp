"""Recheck boundary-independent spatial support using identical FP64 trajectories."""

import json
from pathlib import Path
from validation.fv1_diffusion_domains import design
from validation.fv1_diffusion_matrix import forecast
from validation.fv1_riemann import report_json


def refresh(*, directory, settings, affected):
    for case in design():
        path = directory / (case.name + ".json")
        rows = json.loads(path.read_text())
        for r in rows:
            if case.name in affected:
                updated = forecast(
                    case=case,
                    dx=r["dx"],
                    method=r["method"],
                    policy=r["policy"],
                    multiplier=1.0,
                    settings=settings,
                )
                if abs(updated["hydro_rmse"] - r["hydro_rmse"]) > 1e-12 or updated["steps"] != r["steps"]:
                    raise ArithmeticError("Replay changed accepted transport or timestep count")
                # Preserve graph capture, throughput and warm timing from the isolated timing matrix.
                updated.pop("graph_capture_s")
                r.update(updated)
            else:
                r["interior_radius_m"] = case.length / 6
        report_json(path=path, content=rows)


if __name__ == "__main__":
    directory = Path("validation/fv1_diffusion_review/matrix")
    settings = json.loads((directory / "configuration.json").read_text())["settings"]
    affected = ("front_flat", "front_plain", "front_gentle")
    refresh(directory=directory, settings=settings, affected=affected)
