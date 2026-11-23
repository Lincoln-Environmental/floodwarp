"""Compare depth and extent after the matched recession and retain independent reference checks."""

from functools import partial
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.fidelity import assessment, endpoints
from validation.hancock_long_storm_review.accuracy import block_mean


def field(*, item, directory, support):
    spec, data = item["s"], item["d"]
    end = round(spec["hours"] * 3600)
    path = endpoints.endpoint_path(directory=directory, ident=item["id"], end_s=end)
    if path.is_file():
        with np.load(path) as f:
            desc = json.loads(f["receipt"].item())
            if desc["id"] != item["id"] or desc["end_s"] != end:
                raise ValueError("Wrong endpoint field identity")
            source_dx = desc["support"]
    elif "samples" in data:
        source_dx = spec["dx"]
        path = item["path"] / (data["name"] + ".npz")
    else:
        policy = directory / "policy_runs" / item["id"]
        path = policy / "spatial.npz"
        if not path.is_file():
            if (policy / "result.json").is_file():
                receipt = json.loads((policy / "result.json").read_text())
                if not receipt["reused"]:
                    raise ValueError("Missing recalculated policy endpoint field")
            elif item.get("parent_id", item["id"]) != item["id"]:
                raise ValueError("Missing extended endpoint field")
            path = item["path"] / "spatial.npz"
        source_dx = spec["support"]
    k = round(support / source_dx)
    if not k or not np.isclose(k * source_dx, support):
        raise ValueError(f"Non-nested endpoint support {source_dx} -> {support}")
    with np.load(path) as f:
        return tuple(block_mean(values=f[n], factor=k) for n in ("depth", "maximum"))


def difference(*, candidate, reference, support, thresholds):
    h, maximum = candidate
    hr, mr = reference
    if h.shape != hr.shape or maximum.shape != h.shape or mr.shape != hr.shape:
        raise ValueError("Hydraulic maps have different shape")
    diff = h - hr
    wet = (h >= thresholds[0]) | (hr >= thresholds[0])
    out = dict(
        depth_rmse_m=float(np.sqrt(np.mean(diff**2))),
        wet_depth_rmse_m=float(np.sqrt(np.mean(diff[wet] ** 2))) if wet.any() else None,
        maximum_depth_rmse_m=float(np.sqrt(np.mean((maximum - mr) ** 2))),
        maximum_depth_bias_m=float(np.mean(maximum - mr)),
    )
    for t in thresholds:
        cm = round(100 * t)
        wa, wb = h >= t, hr >= t
        union = np.count_nonzero(wa | wb)
        overlap = np.count_nonzero(wa & wb)
        out[f"iou_{cm}cm"] = overlap / union if union else None
        out[f"extent_error_{cm}cm"] = 1 - overlap / union if union else None
        out[f"wet_area_difference_{cm}cm_m2"] = support**2 * (
            int(wa.sum()) - int(wb.sum())
        )
        out[f"reference_wet_area_{cm}cm_m2"] = support**2 * int(wb.sum())
        reach = []
        for mask in (wa, wb):
            components, count = label(mask)
            outlet = np.unique(components[:, -1])
            wet_at_outlet = np.isin(components, outlet[outlet > 0])
            reach.append((count, int(wet_at_outlet.sum())))
        out[f"component_difference_{cm}cm"] = reach[0][0] - reach[1][0]
        out[f"outlet_connected_area_difference_{cm}cm_m2"] = support**2 * (
            reach[0][1] - reach[1][1]
        )
    return out


def process(*, groups, directory, target, settings):
    result = []
    targets = settings["thresholds_m"]
    support_map = json.loads((directory / "spatial_settings.json").read_text())[
        "support_m"
    ]
    for (cohort, case, bed), items in sorted(groups.items()):
        if not cohort.startswith("complete-"):
            continue
        ref, controls = assessment.references(items=items)
        if ref is None:
            result.extend(
                dict(
                    cohort=cohort,
                    case=case,
                    representation=bed,
                    method=i["s"]["method"],
                    dx=i["s"]["dx"],
                    id=i["id"],
                    reference_status="missing_reference",
                )
                for i in items
            )
            continue
        support = max(
            support_map.get(case, 0),
            *(i["s"].get("support", settings["small_support_m"]) for i in items),
        )
        read = partial(field, directory=directory, support=support)
        truth = read(item=ref)
        deltas = [
            difference(
                candidate=read(item=c),
                reference=truth,
                support=support,
                thresholds=targets,
            )
            for c in controls.values()
            if c is not None
        ]
        scale = float(np.sqrt(np.mean(truth[0] ** 2)))
        depth_limit = max(settings["depth_floor_m"], settings["depth_relative"] * scale)
        limits = dict(
            depth_rmse_m=depth_limit,
            wet_depth_rmse_m=depth_limit,
            maximum_depth_rmse_m=settings["maximum_depth_m"],
        )
        for t in targets:
            limits[f"extent_error_{round(100 * t)}cm"] = settings["extent_error"]
        for item in items:
            if item["factor"] != 1:
                continue
            d = difference(
                candidate=read(item=item),
                reference=truth,
                support=support,
                thresholds=targets,
            )
            record = dict(
                cohort=cohort,
                case=case,
                representation=bed,
                method=item["s"]["method"],
                dx=item["s"]["dx"],
                id=item["id"],
                end_h=item["s"]["hours"],
                support_m=support,
                reference_id=ref["id"],
                ref_dx=ref["s"]["dx"],
                reference_status="three_refinements"
                if len(deltas) == 3
                else "incomplete_refinements",
                **d,
            )
            for kind, control in controls.items():
                record[kind + "_id"] = control["id"] if control else None
            for k, limit in limits.items():
                vs = [obs[k] for obs in deltas if obs[k] is not None]
                spread = max(vs) if len(vs) == 3 else None
                eligible = (
                    spread is not None
                    and spread <= limit
                    and d[k] is not None
                    and item["s"]["dx"] > ref["s"]["dx"]
                )
                if k.startswith("extent_error"):
                    eligible = (
                        eligible
                        and d[k.replace("extent_error", "reference_wet_area") + "_m2"]
                        > 0
                    )
                record.update(
                    {
                        k + "_sensitivity": spread,
                        k + "_limit": limit,
                        "qualified_" + k: eligible,
                    }
                )
            result.append(record)
    metrics.csv_write(file=target / "recession_depth.csv", rows=result)
    from research.floodwarp_methods_paper.fidelity.statistics import frontiers

    cases = {
        item["id"]: item
        for key, members in groups.items()
        if key[0].startswith("complete-")
        for item in members
    }
    measured = []
    for row in result:
        item = cases[row["id"]]
        measured.append(
            dict(
                row,
                station="domain",
                interval_max_s=item["s"].get("interval", 900),
                experiment_id=item["id"],
                runtime_s=item["d"]["hydraulic_s"],
                **assessment.numerical_identity(method=item["s"]["method"]),
            )
        )
    objectives = (
        "depth_rmse_m",
        "wet_depth_rmse_m",
        "maximum_depth_rmse_m",
        *(f"extent_error_{round(100 * t)}cm" for t in targets),
    )
    frontier = frontiers(rows=measured, objectives=objectives)
    metrics.csv_write(file=target / "recession_depth_pareto.csv", rows=frontier)
    return result
