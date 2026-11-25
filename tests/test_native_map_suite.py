from research.floodwarp_methods_paper.fidelity.native_map_suite import inventory


def test_reference_maps_respect_terrain_split_and_wait_for_requested_ordinary_grid_peers():
    specs = []
    for case, family in (("hill-a", "hill"), ("fan-b", "fan")):
        for dx in (1, 5):
            for name, factor in (("fv1", 1), ("hancock_dense", 1), ("adaptive", 1), ("adaptive", 0.5)):
                if dx != 1 and factor != 1:
                    continue
                specs.append(dict(case=case, landform=family, split="development", requested_method=name,
                                  factor=factor, dx=dx, experiment_class="independent_reference"))
    observed = [
        dict(case="hill-a", dx_m=5, method=method, factor=1, horizon="original_window", timing_repeat="")
        for method in ("fv1", "adaptive")
    ]
    observed.append(dict(observed[0], method="hancock_dense", timing_repeat=1))
    options = dict(specs=specs, windows=observed, heldout_family="fan", case="*", dx=5)
    assert list(inventory(**options, partition="fit", method=None)) == [(("hill-a", 5), {("hancock_dense", 1)})]
    assert list(inventory(**options, partition="fit", method="fv1")) == [(("hill-a", 5), set())]
    assert list(inventory(**options, partition="test", method="fv1")) == [(("fan-b", 5), {("fv1", 1), ("adaptive", 1)})]


def test_cost_independent_extent_uses_original_cells_and_actual_complete_event():
    import pytest
    from research.floodwarp_methods_paper.figures.native_extent import observations

    common = dict(case="hill", native_sha="terrain", factor=1, end_h=12, dx_m=5,
                  cells=100, interval_s=60, outlet_peak_m3s=2, horizon="complete-native-metre")
    windows = [dict(common, source_id="f", method="fv1"), dict(common, source_id="r", method="adaptive")]
    pair = dict(case="hill", dx_m=5, end_h=12, cadence_s=60, native_fields="terrain", method="fv1",
                state_source="f", comparator_state_source="r",
                maximum_reference_wet_area_5cm_m2=50, maximum_wet_area_difference_5cm_m2=200,
                maximum_reference_wet_area_10cm_m2=0, maximum_wet_area_difference_10cm_m2=25)
    fv, rk = sorted(observations(windows=windows, differences=[pair], thresholds=(5, 10)), key=run_id)
    assert (fv["area_5cm_m2"], fv["area_10cm_m2"], rk["area_10cm_m2"]) == (250, 25, 0)
    assert not observations(windows=windows, differences=[dict(pair, end_h=8)], thresholds=(5, 10))
    with pytest.raises(ValueError, match="identical terrain"):
        observations(windows=windows, differences=[dict(pair, native_fields="other")], thresholds=(5, 10))


def run_id(row):
    return row["source_id"]
