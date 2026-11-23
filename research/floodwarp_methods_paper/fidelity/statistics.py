"""Matched signs, independent terrain resampling, cadence loss and measured cost frontiers."""

from collections import defaultdict

import numpy as np
from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.fidelity.assessment import (
    curve,
    numerical_identity,
)
from research.floodwarp_methods_paper.fidelity.hydrograph import evaluate


def stratify(*, rows, objectives):
    strata = defaultdict(list)
    for r in rows:
        for k in objectives:
            if not r.get("qualified_" + k):
                continue
            axes = dict(
                overall="all", landform=r.get("landform", r["case"]), grid_m=r["dx"]
            )
            for var, scale in (
                ("gradient", 100),
                ("reference_peak_m3s", 1),
                ("distance_m", 1),
            ):
                if r.get(var, 0) > 0:
                    value = r[var] * scale
                    axes[var + "_decade"] = f"1e{int(np.floor(np.log10(value)))}"
            if r.get("wet_fraction") is not None:
                axes["wet_fraction_quintile"] = min(4, int(5 * r["wet_fraction"]))
            if "hours" in r:
                axes["duration_h"] = r["hours"]
                axes["initial"] = r["input_initial"]
                axes["regime"] = r.get("flow_regime", "unmeasured")
            for axis, level in axes.items():
                strata[
                    r["cohort"],
                    r["station"],
                    r.get("representation"),
                    r["method"],
                    k,
                    axis,
                    level,
                ].append(r[k])
    out = []
    for (cohort, station, rep, method, metric, axis, level), v in strata.items():
        a = np.asarray(v)
        out.append(
            dict(
                cohort=cohort,
                station=station,
                representation=rep,
                method=method,
                metric=metric,
                axis=axis,
                level=level,
                n=len(v),
                negative=sum(a < 0),
                positive=sum(a > 0),
                q25=np.quantile(a, 0.25),
                median=np.median(a),
                q75=np.quantile(a, 0.75),
                absolute_median=np.median(abs(a)),
                absolute_p90=np.quantile(abs(a), 0.9),
            )
        )
    return out


def hydraulic_time(row):
    return row["runtime_s"]


def matched(*, rows, objectives):
    groups = defaultdict(dict)
    for r in rows:
        key = (
            r["cohort"],
            r["case"],
            r["representation"],
            r["dx"],
            r["station"],
            r.get("cap_courant", 0),
        )
        groups[key][r["method"]] = r
    pairs = []
    for key, g in groups.items():
        policies = [
            r
            for name, r in g.items()
            if name in ("hancock_dense", "hancock_adaptive", "hancock_auto")
        ]
        if policies:
            g["hancock_fastest"] = min(policies, key=hydraulic_time)
        for first, second in (
            ("fv1", "hancock_dense"),
            ("hancock_dense", "adaptive"),
            ("fv1", "hancock_fastest"),
            ("hancock_fastest", "adaptive"),
            ("fv1", "roe"),
            ("fv1", "hllc"),
        ):
            if second == "hancock_dense" and second not in g:
                second = "hancock_adaptive"
            if first == "hancock_dense" and first not in g:
                first = "hancock_adaptive"
            if first not in g or second not in g:
                continue
            a, b = g[first], g[second]
            for k in objectives:
                if not (a.get("qualified_" + k) and b.get("qualified_" + k)):
                    continue
                gain = abs(a[k]) - abs(b[k])
                u = a.get("uncertainty_" + k, 0) + b.get("uncertainty_" + k, 0)
                pairs.append(
                    dict(
                        cohort=key[0],
                        case=key[1],
                        representation=key[2],
                        dx=key[3],
                        station=key[4],
                        cap=key[5],
                        first=first,
                        second=second,
                        metric=k,
                        measured_first=a["method"],
                        measured_second=b["method"],
                        first_id=a["experiment_id"],
                        second_id=b["experiment_id"],
                        terrain_id=f"{a.get('landform', key[1])}-{a.get('seed', '')}",
                        signed_first=a[k],
                        signed_second=b[k],
                        absolute_error_reduction=gain,
                        reference_envelope_sum=u,
                        exceeds_reference=abs(gain) > u,
                        runtime_ratio=b["runtime_s"] / a["runtime_s"],
                        gradient=a.get("gradient"),
                    )
                )
    return pairs


def bootstrap(*, pairs, draws, seed):
    rng = np.random.default_rng(seed)
    groups = defaultdict(dict)
    for r in pairs:
        if (
            r["cohort"] != "complete-legacy"
            or r["station"] != "outlet"
            or r["dx"] != 10
        ):
            continue
        k = r["first"], r["second"], r["metric"]
        groups[k].setdefault(r["terrain_id"], []).append(r["absolute_error_reduction"])
    out = []
    for (first, second, metric), g in groups.items():
        v = [np.asarray(i) for i in g.values()]
        if len(v) < 2:
            continue
        values = [
            np.median(np.concatenate([v[j] for j in draw]))
            for draw in rng.integers(len(v), size=(draws, len(v)))
        ]
        out.append(
            dict(
                first=first,
                second=second,
                metric=metric,
                terrain_groups=len(v),
                pairs=sum(map(len, v)),
                median_gain=np.median(np.concatenate(v)),
                grouped_low=np.quantile(values, 0.025),
                grouped_high=np.quantile(values, 0.975),
            )
        )
    return out


def frontiers(*, rows, objectives):
    groups = defaultdict(list)
    for r in rows:
        for k in objectives:
            key = (
                r["cohort"],
                r["case"],
                r["representation"],
                r["station"],
                r.get("cap_courant", 0),
                r.get("interval_max_s"),
                k,
            )
            groups[key].append(r)
    result = []
    for (cohort, case, rep, station, cap, dt, k), g in groups.items():
        valid = [r for r in g if r.get("qualified_" + k)]
        for r in g:
            ok = r.get("qualified_" + k, False)
            err = abs(r[k]) if k in r else None
            dominated = (
                any(
                    abs(i[k]) <= err
                    and i["runtime_s"] <= r["runtime_s"]
                    and (abs(i[k]) < err or i["runtime_s"] < r["runtime_s"])
                    for i in valid
                )
                if ok
                else None
            )
            result.append(
                dict(
                    cohort=cohort,
                    case=case,
                    representation=rep,
                    station=station,
                    cap=cap,
                    interval_s=dt,
                    objective=k,
                    method=r["method"],
                    **{
                        name: r[name]
                        for name in ("formulation", "limiter_flux", "execution_mode")
                        if name in r
                    },
                    dx=r["dx"],
                    experiment_id=r["experiment_id"],
                    reference_id=r.get("reference_id", "analytical"),
                    absolute_error=err,
                    runtime_s=r["runtime_s"],
                    eligible=ok,
                    nondominated=not dominated if ok else None,
                )
            )
    return result


def cadence(*, groups, cfg):
    rows = []
    for (cohort, case, rep), items in groups.items():
        for i in items:
            for station in ("outlet", "section"):
                t, q = curve(it=i, station=station)
                dt = np.diff(t)
                if dt.max() >= 900 or t[-1] % 900 or np.any(900 % dt):
                    continue
                bins = ((t[1:] - 1) // 900).astype(int)
                coarse = np.bincount(bins, weights=q * dt) / 900
                coarse_same = np.repeat(coarse, np.bincount(bins))
                delta = evaluate(edges=t, model=coarse_same, reference=q, config=cfg)
                rows.append(
                    dict(
                        cohort=cohort,
                        case=case,
                        rep=rep,
                        method=i["s"]["method"],
                        dx=i["s"]["dx"],
                        station=station,
                        id=i["id"],
                        native_interval=dt.max(),
                        peak_bias_pct=delta.get("peak_bias_pct"),
                        shape_tv_pct=delta.get("shape_tv_pct"),
                        width80_bias_pct=delta.get("width80_bias_pct"),
                        volume_error_m3=(coarse_same - q) @ dt,
                    )
                )
    return rows


def orient(*, directory, cfg):
    path = directory / "oblique_runs"
    source = path / (path / "current.txt").read_text().strip()
    rows = []
    for f in sorted(source.glob("*.npz")):
        orient, dx, method = f.stem.split("_", 2)
        with np.load(f) as d:
            for j, x in enumerate(d["stations"]):
                row = evaluate(
                    edges=d["edges"],
                    model=d["flow"][:, j],
                    reference=d["truth"][:, j],
                    config=cfg,
                )
                rows.append(
                    dict(
                        row,
                        case=orient,
                        method=method,
                        **numerical_identity(method=method, exact=True),
                        dx=int(dx),
                        station=int(x),
                        distance_m=d["distance"][j],
                        runtime_s=float(d["runtime_s"]),
                        experiment_id=f.stem,
                    )
                )
    return rows


def run(*, directory, rows, groups, pulses, cfg):
    target = directory / "results"
    objectives = cfg["reference_limits"]
    all_rows = rows + pulses
    pairs = matched(rows=all_rows, objectives=objectives)
    fronts = frontiers(rows=all_rows, objectives=objectives)
    from research.floodwarp_methods_paper.fidelity.budgets import feasible_cost
    import json

    budgets = json.loads((directory / "application_settings.json").read_text())
    feasible_cost(
        rows=fronts,
        target=target,
        tolerances=budgets["tolerances_pct"],
        spacings=budgets["fv1_budget_grids_m"],
    )
    for suffix in ("processed/pareto/storms.csv", "processed/million/pareto.csv"):
        for r in metrics.csv_read(directory.parent / suffix):
            if r["objective"] in ("inundation", "memory"):
                fronts.append(
                    dict(r, cohort=suffix, station="domain", absolute_error=r["error"])
                )
    for name, items in (
        ("strata", stratify(rows=all_rows, objectives=objectives)),
        ("pairs", pairs),
        ("pareto", fronts),
        ("paired_intervals", bootstrap(pairs=pairs, draws=1000, seed=24317)),
        ("cadence", cadence(groups=groups, cfg=cfg)),
        ("orientation", orient(directory=directory, cfg=cfg)),
    ):
        metrics.csv_write(file=target / (name + ".csv"), rows=items)
