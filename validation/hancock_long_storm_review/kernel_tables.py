"""Reduce the graph timer records to complete-event and interval timing tables."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from validation.hancock_long_storm_review.graph_profile import category


def records(*, review, groups):
    events, intervals = [], []
    for f in sorted((review / "graph_profiles").glob("*.json")):
        raw = json.loads(f.read_text())
        original = json.loads((review / "events" / f.name).read_text())
        previous = dict.fromkeys(groups, 0.0)
        case = {
            key: original[key]
            for key in ("name", "landform", "storm", "method", "mode")
        }
        for snap, state in zip(raw["samples"], original["samples"], strict=True):
            cumulative = snap["cumulative_device_s"]
            missing = set(cumulative) - set(groups)
            if missing:
                raise KeyError(missing)
            current = {group: cumulative.get(group, 0.0) for group in groups}
            intervals.append(
                dict(
                    **case,
                    t_s=snap["t_s"],
                    steps=state["steps"],
                    **{g: current[g] - previous[g] for g in groups},
                )
            )
            previous = current
        totals = dict.fromkeys(groups, 0.0)
        for name, values in raw["kernels"].items():
            group = category(name=name)
            if name == "donor_budget" or "continuity" in name:
                group = "flux and update"
            elif "accumulate_max" in name:
                group = "accepted transport and accounting"
            totals[group] += values["seconds"]
        events.append(
            dict(
                **case,
                steps=original["steps"],
                uninstrumented_hydraulic_s=original["hydraulic_s"],
                **totals,
            )
        )
    return events, intervals


def write(*, file, rows):
    with file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def render(*, file, rows, groups, methods):
    labels, values = [], []
    for method, mode in methods:
        matching = [r for r in rows if r["method"] == method and r["mode"] == mode]
        total = sum(r["steps"] for r in matching)
        if total:
            labels.append(method.upper() + " " + mode)
            values.append([1e6 * sum(r[g] for r in matching) / total for g in groups])
    data = np.asarray(values)
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    start = np.zeros(len(data))
    for i, group in enumerate(groups):
        ax.barh(labels, data[:, i], left=start, label=group)
        start += data[:, i]
    ax.invert_yaxis()
    ax.set_xlabel("Instrumented time per accepted step (µs)")
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(file, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    review = Path(__file__).parent
    groups = (
        "source and source activity",
        "reconstruction and prediction",
        "flux and update",
        "accepted transport and accounting",
        "scheduling",
        "control and other",
    )
    methods = (
        ("fv1", "sparse"),
        ("hancock", "dense"),
        ("hancock", "sparse"),
        ("hancock", "tiles"),
        ("fv2", "dense"),
        ("li", "native"),
    )
    expected_events = 226
    events, intervals = records(review=review, groups=groups)
    if len(events) != expected_events:
        raise ValueError(
            f"Expected {expected_events} complete graph profiles. Found {len(events)}."
        )
    write(file=review / "graph_kernel_totals.csv", rows=events)
    write(file=review / "graph_kernel_intervals.csv", rows=intervals)
    render(
        file=review / "graph_kernel_cost.png",
        rows=events,
        groups=groups,
        methods=methods,
    )
