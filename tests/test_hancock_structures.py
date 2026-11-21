"""Independent discharge checks and transactional Hancock link integration."""

import numpy as np
import pytest
import warp as wp

from floodwarp.inputs import terrain_from_array
from floodwarp.model import (
    advance,
    create_model,
    diagnostics,
    load_checkpoint,
    save_checkpoint,
)


def link(*, kind, end=96, **extra):
    base = dict(id=kind, inlet_row=0, inlet_col=0, outlet_row=0, outlet_col=end)
    if kind in ("sharp_crested", "broad_crested"):
        base.update(
            type="weir",
            weir_type=kind,
            crest_length_m=1.0,
            crest_elevation_m=0.5,
            discharge_coefficient=0.62,
        )
    elif kind == "bridge":
        base.update(
            type=kind, opening_invert_m=0.0, opening_width_m=1.0, opening_height_m=0.8
        )
    else:
        base.update(
            type="culvert",
            shape=kind,
            inlet_invert_m=0.0,
            outlet_invert_m=0.0,
            width_m=1.0,
            height_m=0.8,
            diameter_m=0.8,
        )
    base.update(extra)
    return base


def model(*, links, device="cpu", mode="tiles", cap=0.01, a=1.5, b=0.0, rain=0.0):
    z = np.zeros((1, 160))
    h = np.zeros_like(z)
    h[0, 0], h[0, 96] = a, b
    mask = np.zeros_like(z, dtype=bool)
    mask[0, [0, 96, 159]] = True
    terrain = terrain_from_array(
        elevation=z,
        active=mask,
        dx=1.0,
        dy=1.0,
        row_order="south_to_north",
        vertical_datum="local",
    )
    return create_model(
        terrain=terrain,
        structures=links,
        device=device,
        explicit_maximum_dt=cap,
        rainfall=rain,
        initial_depth=h,
        execution_mode=mode,
    )


@pytest.mark.parametrize(
    "kind", ("circular", "box", "bridge", "sharp_crested", "broad_crested")
)
@pytest.mark.parametrize("reverse", (False, True))
def test_accepted_capacity(kind, reverse):
    g = 9.81
    a, b = (0.0, 1.5) if reverse else (1.5, 0.0)
    m = model(links=[link(kind=kind)], a=a, b=b)
    advance(model=m, end_time=0.001)
    expected = {
        "circular": np.pi * 0.4**2 * np.sqrt(2 * g * 1.5),
        "box": 0.8 * np.sqrt(2 * g * 1.5),
        "bridge": 0.8 * np.sqrt(2 * g * 1.5),
        "sharp_crested": 0.62 * (2 / 3) * np.sqrt(2 * g),
        "broad_crested": 0.62 * np.sqrt(8 * g / 27),
    }[kind]
    # Opening law uses upstream head above crest, equal to 1 m in both weir controls.
    d = m.solver.get_structure_diagnostics()[0]
    assert diagnostics(model=m)["structure_transfers"] == [d]
    assert d["discharge_m3_s"] == pytest.approx(
        -expected if reverse else expected, rel=1.0e-13
    )
    assert abs(m.solver.cumulative_accounting()["residual"]) < 1.0e-14
    assert all(
        np.all(v == 0.0) for v in m.solver.get_cumulative_face_volumes().values()
    )


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_structure_retry_shared_donor_and_restart(device, tmp_path):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    links = [
        link(kind="box", inlet_invert_m=0.4, width_m=80.0, height_m=5.0),
        link(kind="sharp_crested", end=159, crest_elevation_m=0.8, crest_length_m=80.0),
    ]
    dense = model(links=links, mode="dense", device=device, cap=3.0, rain=1.0e-6)
    tiled = model(links=links, mode="tiles", device=device, cap=3.0, rain=1.0e-6)
    # A single excessive trial uses no link volume and changes no physical ledger.
    tiled.solver.ctrl.assign(
        np.r_[
            0.0,
            3.0,
            0.0,
            0.0,
            np.inf,
            0.0,
            0.0,
            0.0,
            3.0,
            0.0,
            0.0,
            0.0,
            0.0,
            10.0,
            3.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    tiled.solver._enqueue_step()
    assert tiled.solver.ctrl.numpy()[16] == 1.0
    assert all(
        v["forward_volume_m3"] == 0.0 for v in tiled.solver.get_structure_diagnostics()
    )
    assert tiled.solver.cumulative_accounting()["rain"] == 0.0
    # Restore the identical trial predictor before comparing accepted histories.
    tiled.solver.restore_state(state=dense.solver.capture_state())
    for m in (dense, tiled):
        stats = advance(model=m, end_time=3.0)
        assert stats["rejected_cfl_trials"] > 0
        assert abs(m.solver.cumulative_accounting()["residual"]) < 1.0e-12
        assert (m.solver.get_water_depth() >= 0.0).all()
    np.testing.assert_array_equal(
        tiled.solver.get_conservative_state(), dense.solver.get_conservative_state()
    )
    assert (
        tiled.solver.get_structure_diagnostics()
        == dense.solver.get_structure_diagnostics()
    )
    ledger = tiled.solver.get_structure_diagnostics()
    assert all(d["forward_volume_m3"] > 0 for d in ledger)
    assert ledger[0]["limited_steps"] > 0
    assert tiled.solver.get_water_depth()[0, 0] >= 0.4
    file = tmp_path / "links.npz"
    save_checkpoint(model=tiled, path=file)
    restarted = model(links=links, mode="dense", device=device, cap=3.0, rain=1.0e-6)
    load_checkpoint(model=restarted, path=file)
    for m in (tiled, restarted):
        advance(model=m, end_time=4.0)
    for name, arr in tiled.solver.capture_state()["arrays"].items():
        np.testing.assert_array_equal(
            arr, restarted.solver._fields[name].numpy(), err_msg=name
        )
    with pytest.raises(ValueError, match="match"):
        load_checkpoint(
            model=model(links=[links[0]], device=device, cap=3.0, rain=1.0e-6),
            path=file,
        )


def test_no_reverse_flow_below_sill_or_through_check_valve():
    for record in (
        link(kind="box", one_way=True),
        link(kind="box", outlet_invert_m=2.0),
    ):
        m = model(links=[record], a=0.0, b=1.0)
        advance(model=m, end_time=0.1)
        assert m.solver.get_structure_diagnostics()[0]["net_volume_m3"] == 0.0


def test_shared_receiver_cannot_overtake_both_donors():
    links = [
        link(kind="box", end=159, width_m=1.0e5),
        link(kind="box", end=159, inlet_col=96, id="second", width_m=1.0e5),
    ]
    m = model(links=links, a=1.0, b=1.0)
    advance(model=m, end_time=0.001)
    a, b, c = m.solver.get_water_depth()[0, [0, 96, 159]]
    assert 0.0 < c <= min(a, b)
    assert a + b + c == pytest.approx(2.0, abs=1.0e-14)
    assert all(row["limited_steps"] for row in m.solver.get_structure_diagnostics())


def test_remote_structure_wets_connected_surface_on_following_step():
    bed = np.zeros((1, 160))
    active = np.zeros_like(bed, dtype=bool)
    active[0, [0, 96, 97]] = True
    terrain = terrain_from_array(
        elevation=bed,
        active=active,
        dx=1.0,
        dy=1.0,
        row_order="south_to_north",
        vertical_datum="local",
    )
    initial = np.zeros_like(bed)
    initial[0, 0] = 1.5
    final = []
    for mode in ("dense", "tiles"):
        simulation = create_model(
            terrain=terrain,
            structures=[link(kind="box")],
            execution_mode=mode,
            initial_depth=initial,
            device="cpu",
            explicit_maximum_dt=0.001,
        )
        advance(model=simulation, end_time=0.001)
        first = simulation.solver.get_water_depth().copy()
        assert first[0, 96] > 0.0 and first[0, 97] == 0.0
        assert simulation.solver.get_max_water_depth()[0, 96] == first[0, 96]
        advance(model=simulation, end_time=0.002)
        assert simulation.solver.get_water_depth()[0, 97] > 0.0
        assert simulation.solver.get_cumulative_face_volumes()["qx"][0, 97] > 0.0
        final.append(simulation.solver.capture_state()["arrays"])
    for name in final[0]:
        np.testing.assert_array_equal(final[0][name], final[1][name], err_msg=name)
