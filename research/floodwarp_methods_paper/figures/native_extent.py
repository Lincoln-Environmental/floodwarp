"""Pair outlet peaks and event footprints with surface water remaining after recession."""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from research.floodwarp_methods_paper import metrics


def observations(*, windows, differences, thresholds):
    ends = {r["source_id"]: r for r in windows if r["horizon"] == "complete-native-metre" and float(r["factor"]) == 1}
    rows = {}
    for pair in differences:
        ids = pair["state_source"], pair["comparator_state_source"]
        if any(i not in ends for i in ids):
            continue
        case, dx, end = pair["case"], float(pair["dx_m"]), float(pair["end_h"])
        if any(float(ends[i]["end_h"]) != end for i in ids):
            continue
        for i in ids:
            r = ends[i]
            if (r["case"] != case or float(r["dx_m"]) != dx or pair["native_fields"] != r["native_sha"]
                    or float(pair["cadence_s"]) != float(r["interval_s"])):
                raise ValueError("Native extent and discharge must use identical terrain, grid and timing")
        if ends[ids[0]]["method"] != pair["method"] or ends[ids[1]]["method"] != "adaptive":
            raise ValueError("Native extent comparison has different numerical methods from the outlet record")
        for candidate, ident in ((True, ids[0]), (False, ids[1])):
            r = ends[ident]
            area = {}
            for cm in thresholds:
                base = float(pair[f"maximum_reference_wet_area_{cm}cm_m2"])
                value = base + float(pair[f"maximum_wet_area_difference_{cm}cm_m2"]) if candidate else base
                final = float(pair[f"reference_wet_area_{cm}cm_m2"])
                if candidate:
                    final += float(pair[f"wet_area_difference_{cm}cm_m2"])
                if not 0 <= final <= value <= int(r["cells"]) * dx**2:
                    raise ValueError("Final or cumulative wet area falls outside its observed footprint")
                measured = r.get(f"model_grid_wet_area_{cm}cm_m2")
                if measured not in ("", None) and float(measured) != final:
                    raise ValueError("Endpoint wet area disagrees with the storm water account")
                area[f"area_{cm}cm_m2"] = value
                area[f"final_area_{cm}cm_m2"] = final
            record = dict(case=case, dx_m=dx, end_h=end, cadence_s=float(r["interval_s"]),
                          method=r["method"], source_id=ident, outlet_peak_m3s=float(r["outlet_peak_m3s"]),
                          storage_m3=float(r["storage_m3"]), **area)
            key = case, end, dx, r["method"]
            if key in rows and rows[key] != record:
                raise ValueError("Conflicting observations of the same peak or cumulative wet area")
            rows[key] = record
    return [rows[key] for key in sorted(rows)]


def generate(*, study, thresholds, methods, save):
    folder = study / "fidelity/native_results"
    water = folder / "measured_windows.csv"
    maps = folder / "native_grid_field_differences.csv"
    if not water.is_file() or not maps.is_file():
        return
    rows = observations(windows=metrics.csv_read(file=water), differences=metrics.csv_read(file=maps), thresholds=thresholds)
    groups = defaultdict(list)
    for row in rows:
        groups[row["case"], row["end_h"], row["cadence_s"]].append(row)
    target = study / "supplementary/native_grid_extent"
    metrics.csv_write(file=target / "observations.csv", rows=rows)
    for (case, end, cadence), group in groups.items():
        fields = [("outlet_peak_m3s", "Outlet peak (m³/s)", 1)]
        fields.extend((f"area_{cm}cm_m2", f"Area ever ≥{cm} cm (ha)", 10000) for cm in thresholds)
        fields.append(("storage_m3", "Final surface storage (10³ m³)", 1000))
        fields.extend((f"final_area_{cm}cm_m2", f"Final area ≥{cm} cm (ha)", 10000) for cm in thresholds)
        fig, axes = plt.subplots(2, len(thresholds) + 1, figsize=(7.2, 4.7), layout="constrained", squeeze=False)
        grids = sorted({row["dx_m"] for row in group})
        for ax, (field, ylabel, divisor) in zip(axes.flat, fields, strict=True):
            for (name, (label, colour, marker)), offset in zip(methods.items(), np.linspace(-0.16, 0.16, len(methods)), strict=True):
                points = [r for r in group if r["method"] == name]
                ax.scatter([grids.index(r["dx_m"]) + offset for r in points], [r[field] / divisor for r in points],
                           label=label, marker=marker, color=colour, s=24, clip_on=False)
            ax.set(xticks=range(len(grids)), xticklabels=[f"{dx:g}" for dx in grids],
                   xlabel="Hydraulic grid (m)", ylabel=ylabel)
            ax.set_xlim(-0.45, len(grids)-0.55)
            largest = max(r[field] for r in group) / divisor
            ax.set_ylim(0, largest * 1.08 if largest else 1)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle(f"{case} · {end:g} h · {cadence:g} s discharge intervals", fontsize=10)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=len(methods))
        save(fig=fig, folder=target, name=f"{case}_{end:g}h")
