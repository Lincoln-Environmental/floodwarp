"""Attribute complete synthetic events in a separate uncaptured CUDA replay."""

import json
from pathlib import Path

import numpy as np
import warp as wp

from floodwarp.model import advance
from validation.hancock_long_storm_review.landforms import storm_model


def attribute(*, review, config, pairs, methods, device):
    archive = review / "storm_kernel_profiles.json"
    records = json.loads(archive.read_text()) if archive.exists() else []
    completed = {r["run"] for r in records}
    for land, storm in pairs:
        for method, mode in methods:
            primary = next(
                (review / "events").glob(
                    f"primary_*_{land}_{storm}_{method}_{mode}_*.json"
                )
            )
            run = json.loads(primary.read_text())
            if run["name"] in completed:
                continue
            model, _ = storm_model(
                config=config,
                landform=land,
                storm=storm,
                method=method,
                mode=mode,
                dx=run["dx"],
                terrain_dx=run["terrain_dx"],
                dimensions=run["dimensions_m"],
                duration=run["hours"] * 3600,
                device=device,
            )
            print(land, storm, method, mode, flush=True)
            samples = []
            for sample in run["samples"]:
                with wp.ScopedTimer(
                    name="uncaptured storm", cuda_filter=wp.TIMING_KERNEL, print=False
                ) as timer:
                    stats = advance(
                        model=model, end_time=sample["t_s"], use_cuda_graph=False
                    )
                groups = {}
                for event in timer.timing_results:
                    key = (
                        event.name.split("kernel ")[-1]
                        .split("_cuda")[0]
                        .rsplit("_", 1)[0]
                    )
                    groups[key] = groups.get(key, 0.0) + event.elapsed / 1000.0
                if stats["steps"] != sample["steps"]:
                    raise AssertionError(
                        "Uncaptured replay changed the accepted timestep count"
                    )
                samples.append(
                    dict(
                        t_s=sample["t_s"],
                        kernel_s=groups,
                        steps=stats["steps"],
                        uncaptured_loop_s=stats["loop_seconds"],
                    )
                )
            checks = model.solver.cumulative_accounting()
            np.testing.assert_allclose(
                checks["storage"], run["final_m3"], rtol=1.0e-13, atol=1.0e-12
            )
            records.append(dict(run=run["name"], samples=samples))
            temp = archive.with_suffix(".tmp")
            temp.write_text(json.dumps(records, separators=(",", ":")) + "\n")
            temp.replace(archive)


if __name__ == "__main__":
    review = Path(__file__).parent
    config = json.loads((review / "cases.json").read_text())
    pairs = (
        ("plain", "moderate"),
        ("hill", "intense"),
        ("coastal", "front"),
        ("fan", "back"),
        ("mixed", "spatial"),
        ("floodplain", "soil"),
    )
    methods = (("hancock", "tiles"),)
    device = "cuda:0"
    attribute(review=review, config=config, pairs=pairs, methods=methods, device=device)
