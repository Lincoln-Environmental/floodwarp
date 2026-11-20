"""Read the archived FloodWarp LI Orton ledger and completed FV storms. No solver execution."""

import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from validation.fv1_riemann import report_json
from validation.fv2_storm_evidence import on_footprint, overlap, read_snapshot
from validation.sparse_fv1_benchmark import select_checkpoints


def liquid_fields(*, directory, hours):
    result = {}
    for path in directory.glob("checkpoint-*.npz"):
        with np.load(path) as a:
            m = json.loads(str(a["metadata"]))
            t = m["host"]["simulation_time"]
            if t not in [h * 3600 for h in hours]:
                continue
            result[int(t / 3600)] = (a["water_depth"], a["max_water_depth_m"], a["active_mask"].astype(bool))
    return result


def archived_comparison(*, directory, compressed, checkpoints, hours, duration, interval, threshold):
    with gzip.open(compressed / "purau_event_72h-li_0-record.json.gz", "rt") as f:
        li = json.load(f)
    with gzip.open(compressed / "purau_event_72h-fv1_0-record.json.gz", "rt") as f:
        fv1 = json.load(f)
    if li["signature"] != fv1["signature"]:
        raise ValueError("Archived LI and FV1 inputs differ")
    hydro = {}
    sources = {}
    for name, record in (("5 m LI", li), ("5 m FV1", fv1)):
        rows = [r for r in record["ledger"] if r["seconds"] <= duration]
        np.testing.assert_array_equal(
            [r["seconds"] for r in rows], np.arange(interval, duration + 1, interval)
        )
        hydro[name] = np.diff([0, *[r["outflow_m3"] for r in rows]]) / interval
    for name, file in (("20 m FV1", "20m_contact_hll"), ("20 m FV2", "20m_fv2_mc")):
        r = json.loads((directory / (file + ".json")).read_text())
        if (
            r["time_s"] != duration
            or r["forcing"]["rate_nodes_sha256"] != li["signature"]["fields"]["rates"]
            or r["forcing"]["node_times_sha256"] != li["signature"]["fields"]["times"]
        ):
            raise ValueError("Different scenario")
        hydro[name] = np.array([v["outflow_m3_s"] for v in r["hydrograph"]])
        sources[name] = Path(r["origin"]).parent
    if not np.array_equal(
        hydro["5 m FV1"],
        [
            v["outflow_m3_s"]
            for v in json.loads((directory / "5m_contact_hll.json").read_text())["hydrograph"]
        ],
    ):
        raise ValueError("5 m control differs")
    records = []
    for name, series in hydro.items():
        records.append(
            dict(
                name=name,
                peak_m3_s=float(series.max()),
                peak_end_s=int((series.argmax() + 1) * interval),
                export_m3=float(series.sum() * interval),
                vs_fv2_hydrograph_rmse_m3_s=float(np.sqrt(np.mean((series - hydro["20 m FV2"]) ** 2))),
                vs_fv2_correlation=float(np.corrcoef(series, hydro["20 m FV2"])[0, 1]),
            )
        )
    fields = []
    if (checkpoints / "li_0").is_dir() and all(p.is_dir() for p in sources.values()):
        snaps = liquid_fields(directory=checkpoints / "li_0", hours=hours)
        fvpaths = select_checkpoints(directory=checkpoints / "fv1_0", hours=hours)
        for h, path in zip(hours, fvpaths, strict=True):
            base, peak, mask = snaps[h]
            vectors = {"5 m LI": (base, peak)}
            state, maximum, active = read_snapshot(path=path)
            np.testing.assert_array_equal(active, mask)
            vectors["5 m FV1"] = state[..., 0], maximum
            for name, src in sources.items():
                q, m, _ = read_snapshot(path=src / f"hour-{h:02d}.npz")
                vectors[name] = tuple(
                    on_footprint(field=a, dx=20, base=5, shape=mask.shape) for a in (q[..., 0], m)
                )
            candidate, env = vectors["20 m FV2"]
            for name, (z, m) in vectors.items():
                fields.append(
                    dict(
                        hour=h,
                        name=name,
                        vs_fv2_depth_rmse_m=float(np.sqrt(np.mean((candidate[mask] - z[mask]) ** 2))),
                        vs_fv2_envelope_rmse_m=float(np.sqrt(np.mean((env[mask] - m[mask]) ** 2))),
                        vs_fv2_wet_iou=overlap(a=candidate > threshold, b=z > threshold, mask=mask),
                    )
                )
    if fields:
        report_json(
            path=directory / "archived_li_fields.json",
            content=dict(hours=hours, threshold_m=threshold, comparisons=fields),
        )
    report_json(
        path=directory / "archived_li.json",
        content=dict(
            shared_5m_input_fields=li["signature"]["fields"],
            profile="physical_candidate",
            hydrographs=records,
            spatial_evidence="archived_li_fields.json",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for name, series in hydro.items():
        t = (np.arange(len(series)) + 0.5) * interval / 3600
        for ax in axes:
            ax.plot(t, series, lw=1.5, label=name)
    axes[0].set_xlim(0, duration / 3600)
    axes[1].set_xlim(22, 28)
    for ax in axes:
        ax.set(xlabel="Storm time (h)", ylabel="Mean outlet discharge (m³/s)")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.savefig(directory / "archived_li.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    directory = Path("validation/fv2_review/hirds48")
    records = Path("validation/repair_review/completed_runs")
    checkpoints = Path("outputs/qualification_optimized/purau_event_72h")
    archived_comparison(
        directory=directory,
        compressed=records,
        checkpoints=checkpoints,
        hours=(24, 48),
        duration=172800,
        interval=60,
        threshold=0.01,
    )
