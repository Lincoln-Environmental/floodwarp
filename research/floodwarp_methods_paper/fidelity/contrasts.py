"""Retain practical solver differences even where independent accuracy is unresolved."""

from collections import defaultdict
from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.fidelity.assessment import compare


def compute(*, groups, cfg, directory):
    output = []
    for (cohort, case, rep), items in groups.items():
        if not cohort.startswith("complete-"):
            continue
        grids = defaultdict(dict)
        for it in items:
            if it["factor"] == 1:
                grids[it["s"]["dx"]][it["s"]["method"]] = it
        for dx, g in grids.items():
            second = g.get("hancock_dense", g.get("hancock_adaptive"))
            if not second:
                continue
            for method in ("fv1", "adaptive", "li", "hllc", "roe"):
                first = g.get(method)
                if not first:
                    continue
                for loc in ("outlet", "section"):
                    row = compare(it=first, ref=second, station=loc, cfg=cfg)
                    a, b = first["d"], second["d"]
                    output.append(
                        dict(
                            row,
                            cohort=cohort,
                            case=case,
                            representation=rep,
                            dx=dx,
                            station=loc,
                            candidate=method,
                            comparator=second["s"]["method"],
                            candidate_id=first["id"],
                            comparator_id=second["id"],
                            hours=first["s"]["hours"],
                            runtime_ratio=a["hydraulic_s"] / b["hydraulic_s"],
                            infiltration_difference_m3=a["infiltration_m3"]
                            - b["infiltration_m3"],
                            storage_difference_m3=a.get(
                                "final_storage_m3", a.get("final_m3")
                            )
                            - b.get("final_storage_m3", b.get("final_m3")),
                        )
                    )
    metrics.csv_write(file=directory / "results/intermodel.csv", rows=output)
