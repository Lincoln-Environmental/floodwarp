"""Regression guards for publication rankings and grouping of repeated timings."""

from research.floodwarp_methods_paper.catalog import scientific_ids
from research.floodwarp_methods_paper.metrics import pareto


def test_pareto_requires_strict_domination_and_independent_reference():
    common = dict(
        case="a", method="fv1", dx=5, qualified_depth=True, reference_self=False
    )
    rows = [
        dict(common, experiment_id="fast", depth=2.0, seconds=1.0),
        dict(common, experiment_id="tie", depth=2.0, seconds=1.0),
        dict(common, experiment_id="accurate", depth=0.5, seconds=2.0),
        dict(common, experiment_id="dominated", depth=2.0, seconds=2.0),
        dict(common, experiment_id="self", reference_self=True, depth=0.0, seconds=0.1),
        dict(
            common,
            experiment_id="unresolved",
            qualified_depth=False,
            depth=0.2,
            seconds=0.2,
        ),
        dict(common, case="b", experiment_id="other storm", depth=0.1, seconds=0.1),
    ]
    results = pareto(
        rows=rows, keys=("case",), objectives={"depth": "depth"}, cost="seconds"
    )
    decisions = {r["experiment_id"]: r["nondominated"] for r in results}
    assert decisions == {
        "fast": True,
        "tie": True,
        "accurate": True,
        "dominated": False,
        "other storm": True,
    }


def test_timing_repeats_share_hydraulic_identity_across_stages():
    forcing = dict(
        terrain_sha256="bed",
        initial_depth_sha256="depth",
        initial_hu_sha256="momentum",
        rain_mask_sha256="mask",
        rain_rates_ms=[0.001, 0],
        rain_times_s=[0, 21600],
        infiltration_sha256="soil",
        boundary_x_sha256="left",
        boundary_y_sha256="right",
        stage_m=None,
        shape=[8, 12],
        roughness=0.035,
        numerical={"solver_method": "fv1"},
    )
    row = dict(forcing=forcing, mode="sparse")
    spec = dict(case="abc", dx=10, dem=10, hours=8, multiplier=1)
    record = dict(context={"source_sha256": "original"}, spec=spec, stage="long-storms")
    ids = scientific_ids(record=record, row=row)
    repeat = dict(record, stage="performance", spec=dict(spec, repeat=2))
    assert scientific_ids(record=repeat, row=row) == ids
    changed = scientific_ids(
        record=dict(record, context={"source_sha256": "revised"}), row=row
    )
    assert changed[0] == ids[0] and changed[1] != ids[1]
    different = scientific_ids(
        record=record, row=dict(row, forcing=dict(forcing, terrain_sha256="alternate"))
    )
    assert different[0] != ids[0] and different[1] != ids[1]


def test_fan_control_preserves_bed_and_replaces_only_initial_state():
    import json
    from pathlib import Path

    import numpy as np

    from research.floodwarp_methods_paper.plan import read_config
    from research.floodwarp_methods_paper.terrain_controls import fixed_initial
    from research.floodwarp_methods_paper.terrain.generators.generate import fields

    root = Path(__file__).resolve().parents[1]
    config = read_config(
        path=root / "research/floodwarp_methods_paper/configs/study.json"
    )
    template = json.loads(
        (root / "validation/hancock_long_storm_review/cases.json").read_text()
    )
    options = dict(
        design=config["synthetic"],
        template=template,
        family="fan",
        seed=18217,
        initial="channel",
        dx=1,
    )
    controlled, _ = fixed_initial(**options, dem=10)
    original, _ = fields(**options, dem=10)
    sampled, _ = fields(**options, dem=1)
    assert not np.array_equal(original[1], sampled[1])
    for index in (0, 3, 4):
        np.testing.assert_array_equal(controlled[index], original[index])
    for index in (1, 2):
        np.testing.assert_array_equal(controlled[index], sampled[index])
