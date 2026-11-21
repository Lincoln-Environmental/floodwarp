"""Instrument complete CUDA-graph replay in a separate run using the device global timer."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from floodwarp.model import advance, diagnostics
from validation.hancock_long_storm_review.landforms import storm_model


@wp.func_native("""
#ifdef __CUDA_ARCH__
    unsigned long long ticks;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ticks));
    return static_cast<wp::int64>(ticks);
#else
    return 0;
#endif
""")
def global_ns() -> wp.int64:
    """CUDA's nanosecond device-wide timer."""


@wp.kernel
def stamp(
    start: wp.array(dtype=wp.int64),
    total: wp.array(dtype=wp.int64),
    calls: wp.array(dtype=wp.int64),
    slot: int,
):
    now = global_ns()
    if slot < 0:
        start[0] = now
    else:
        total[slot] += now - start[0]
        calls[slot] += wp.int64(1)


class GraphTimer:
    def __init__(self, *, device, limit):
        self.real_launch = wp.launch
        self.device = device
        self.start = wp.zeros(1, dtype=wp.int64, device=device)
        self.total = wp.zeros(limit, dtype=wp.int64, device=device)
        self.calls = wp.zeros_like(self.total)
        self.names = {}
        wp.load_module(module=stamp.module, device=device)

    def launch(self, kernel, *args, **kwargs):
        if kwargs.get("record_cmd"):
            raise RuntimeError(
                "Explicit recorded commands require a separate timing adapter"
            )
        name = kernel.func.__name__
        slot = self.names.setdefault(name, len(self.names))
        if slot >= self.total.size:
            raise ValueError("Profiling table exhausted")
        stream = kwargs.get("stream")
        self.real_launch(
            kernel=stamp,
            dim=1,
            device=self.device,
            stream=stream,
            inputs=[self.start, self.total, self.calls, -1],
        )
        result = self.real_launch(kernel, *args, **kwargs)
        self.real_launch(
            kernel=stamp,
            dim=1,
            device=self.device,
            stream=stream,
            inputs=[self.start, self.total, self.calls, slot],
        )
        return result

    def summary(self):
        values = self.total.numpy()
        calls = self.calls.numpy()
        return {
            name: dict(seconds=float(values[i]) * 1e-9, launches=int(calls[i]))
            for name, i in self.names.items()
        }


def category(*, name):
    if any(
        term in name for term in ("rain", "source", "infiltrat", "friction", "manning")
    ):
        return "source and source activity"
    if any(
        term in name
        for term in ("mark_trial", "compact", "tile", "released", "mark_active")
    ):
        return "scheduling"
    if (
        any(term in name for term in ("slope", "compressive", "inverse_depth"))
        or name == "predict"
    ):
        return "reconstruction and prediction"
    if any(term in name for term in ("accept_faces", "volume", "account")):
        return "accepted transport and accounting"
    if any(
        term in name
        for term in ("face", "flux", "drain", "limit", "euler", "update", "discharge")
    ):
        return "flux and update"
    return "control and other"


def attribute(*, review, prefix, config, limit, device):
    folder = review / "graph_profiles"
    folder.mkdir(exist_ok=True)
    for f in sorted((review / "events").glob(prefix)):
        row = json.loads(f.read_text())
        output = folder / f.name
        if output.exists():
            continue
        print(row["name"], flush=True)
        m, _ = storm_model(
            config=config,
            landform=row["landform"],
            storm=row["storm"],
            method=row["method"],
            mode=row["mode"],
            dx=row["dx"],
            terrain_dx=row["terrain_dx"],
            dimensions=row["dimensions_m"],
            duration=row["hours"] * 3600.0,
            device=device,
        )
        probe = GraphTimer(device=device, limit=limit)
        history = []
        for sample in row["samples"]:
            with patch.object(wp, "launch", probe.launch):
                stats = advance(model=m, end_time=sample["t_s"])
            if stats["steps"] != sample["steps"]:
                raise ValueError("Instrumentation changed accepted timestep count")
            totals = {}
            for name, k in probe.summary().items():
                group = category(name=name)
                totals[group] = totals.get(group, 0.0) + k["seconds"]
            history.append(dict(t_s=sample["t_s"], cumulative_device_s=totals))
        np.testing.assert_allclose(
            diagnostics(model=m)["storage_m3"], row["final_m3"], rtol=1e-12, atol=1e-10
        )
        data = dict(run=row["name"], kernels=probe.summary(), samples=history)
        if not any(k["launches"] >= row["steps"] for k in data["kernels"].values()):
            raise ValueError("Missing replayed graph nodes")
        tmp = output.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")) + "\n")
        tmp.replace(output)


if __name__ == "__main__":
    review = Path(__file__).parent
    prefix = "primary_*.json"
    config = json.loads((review / "cases.json").read_text())
    limit = 160
    device = "cuda:0"
    attribute(review=review, prefix=prefix, config=config, limit=limit, device=device)
