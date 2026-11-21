"""Test accepted predictor growth limits with the unchanged face CFL and retry gate."""

from pathlib import Path
import json
from time import perf_counter
from unittest.mock import patch

import numpy as np
import warp as wp

from floodwarp import fv2_kernels
from floodwarp.fv2_solver import HancockSolver
from floodwarp.model import advance, diagnostics
from validation.hancock_long_storm_review.landforms import storm_model


@wp.kernel
def save_predictor(
    previous: wp.array(dtype=wp.float64), old: wp.array(dtype=wp.float64)
):
    old[0] = previous[0]


@wp.kernel
def limit_growth(
    old: wp.array(dtype=wp.float64),
    previous: wp.array(dtype=wp.float64),
    ctrl: wp.array(dtype=wp.float64),
    growth: wp.float64,
):
    if old[0] > wp.float64(0.0) and ctrl[2] <= ctrl[17] and ctrl[7] == wp.float64(0.0):
        previous[0] = wp.min(previous[0], old[0] * growth)


class GrowthHancock(HancockSolver):
    def _load_extra_step_modules(self):
        super()._load_extra_step_modules()
        wp.load_module(module=limit_growth.module, device=self.device)

    def _launch(self, kernel, count, inputs):
        if kernel is not fv2_kernels.finish:
            return super()._launch(kernel=kernel, count=count, inputs=inputs)
        previous = self._fields["fv2_cfl_predictor"]
        super()._launch(
            kernel=save_predictor, count=1, inputs=[previous, self.old_predictor]
        )
        super()._launch(kernel=kernel, count=count, inputs=inputs)
        super()._launch(
            kernel=limit_growth,
            count=1,
            inputs=[self.old_predictor, previous, self.ctrl, self.growth],
        )


def run(*, review, config, device, pairs, factors):
    output = []
    for land, storm in pairs:
        base = json.loads(
            next(
                (review / "events").glob(
                    f"primary_*_{land}_{storm}_hancock_tiles_*.json"
                )
            ).read_text()
        )
        with np.load(
            Path("outputs/hancock_long_storm_review/fields") / (base["name"] + ".npz")
        ) as f:
            base_depth = f["depth"]
        for factor in factors:
            constructor = HancockSolver if factor is None else GrowthHancock
            with patch("floodwarp.fv2_solver.HancockSolver", constructor):
                m, _ = storm_model(
                    config=config,
                    landform=land,
                    storm=storm,
                    method="hancock",
                    mode="tiles",
                    dx=base["dx"],
                    terrain_dx=base["terrain_dx"],
                    dimensions=base["dimensions_m"],
                    duration=base["hours"] * 3600.0,
                    device=device,
                )
            m.solver.growth = factor
            m.solver.old_predictor = wp.zeros(1, dtype=wp.float64, device=device)
            steps = rejects = 0
            elapsed = 0.0
            for s in base["samples"]:
                tick = perf_counter()
                stats = advance(model=m, end_time=s["t_s"])
                wp.synchronize_device(device=device)
                elapsed += (
                    perf_counter()
                    - tick
                    - stats["compilation_seconds"]
                    - stats["graph_capture_seconds"]
                )
                steps += stats["steps"]
                rejects += stats["rejected_cfl_trials"]
            state = m.solver.get_water_depth()
            diag = diagnostics(model=m)
            output.append(
                dict(
                    landform=land,
                    storm=storm,
                    growth=factor,
                    steps=steps,
                    rejected=rejects,
                    hydraulic_s=elapsed,
                    depth_change_rmse_m=float(
                        np.sqrt(np.mean((state - base_depth) ** 2))
                    ),
                    outlet_volume_m3=diag["boundary_outflow"],
                    residual_m3=diag["residual_m3"],
                )
            )
    (review / "retry_growth.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    review = Path(__file__).parent
    config = json.loads((review / "cases.json").read_text())
    pairs = (
        ("mixed", "spatial"),
        ("fan", "back"),
        ("incised", "moderate"),
        ("coastal", "moderate"),
    )
    factors = (None, 1.05, 1.2, 1.5)
    device = "cuda:0"
    run(review=review, config=config, device=device, pairs=pairs, factors=factors)
