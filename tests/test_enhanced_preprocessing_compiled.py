"""Strict offline implementation equivalence; no CUDA execution required."""

import numpy as np
import pytest

from floodwarp.enhanced_geometry import (valley_ownership, _valley_ownership_reference,
    coefficient_chords, _coefficient_chords_reference, scalar_nodes, _scalar_nodes_reference,
    encode_port_coefficients, _encode_port_coefficients_reference, component_stage_tables)
from floodwarp.enhanced_reduction import wet_components, _wet_components_reference, reduce_cell, boundary_nodes


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (2, 3), (5, 5), (8, 9)])
def test_compiled_labels_preserve_root_numbering_and_plateau_ownership(shape):
    rng = np.random.default_rng(seed=731)
    for _ in range(30):
        bed = rng.integers(low=0, high=6, size=shape).astype(float)
        np.testing.assert_array_equal(valley_ownership(bed=bed), _valley_ownership_reference(bed=bed))
        for stage in (-1., 0., 1., 3., 5., 6., 10.):
            np.testing.assert_array_equal(wet_components(bed=bed, stage_m=stage), _wet_components_reference(bed=bed, stage_m=stage))


@pytest.mark.parametrize("columns", [4, 6])
@pytest.mark.parametrize("merge_jumps", [False, True])
def test_compiled_chords_and_nodes_retain_certificates_bitwise(columns, merge_jumps):
    rng = np.random.default_rng(seed=731)
    for _ in range(50):
        levels = np.cumsum(rng.uniform(.001, 1., size=40))
        table = np.empty((39, columns))
        table[:, :2] = np.column_stack((levels[:-1], levels[1:]))
        for column in range(2, columns, 2):
            values = np.exp(rng.uniform(-5., 5., size=len(levels)))
            table[:, column:column + 2] = np.column_stack((values[:-1], values[1:]))
        table[12:, 0:2] += 1.
        table[20, 2] *= 1.00001
        arguments = dict(table=table, relative_tolerance=.005, critical_levels={levels[4]}, merge_jumps=merge_jumps)
        result, error = coefficient_chords(**arguments)
        reference, expected_error = _coefficient_chords_reference(**arguments)
        np.testing.assert_array_equal(result, reference)
        assert error == expected_error
        for column in range(2, columns, 2):
            selected = result[:, [0, 1, column, column + 1]]
            for actual, expected in zip(scalar_nodes(table=selected), _scalar_nodes_reference(table=selected), strict=True):
                np.testing.assert_array_equal(actual, expected)


def test_reusing_wet_labels_preserves_all_native_reduction_fields():
    bed = np.array([[0., 1., 0.], [.2, .5, .1], [0., 1., 0.]])
    for stage in (.15, .4, 2.):
        arguments = dict(bed_m=bed, stage_m=stage, resolution_m=1., manning_n=.04, compute_conductance=False)
        original = reduce_cell(**arguments)
        cached = reduce_cell(**arguments, _wet_labels=wet_components(bed=bed, stage_m=stage))
        for name, value in vars(original).items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, getattr(cached, name))


@pytest.mark.parametrize("tolerance", [None, .005])
def test_compiled_scalar_encoder_preserves_upward_rounding_and_certificates(tolerance):
    rng = np.random.default_rng(seed=731)
    for _ in range(100):
        levels = np.cumsum(rng.uniform(.001, 1., 50))
        rows = np.empty((49, 6))
        rows[:, :2] = np.column_stack((levels[:-1], levels[1:]))
        for column in (2, 4):
            values = np.exp(rng.uniform(-15., 15., len(levels)))
            rows[:, column:column + 2] = np.column_stack((values[:-1], values[1:]))
        rows[:5, 2:4] = .5
        arguments = (rows, {levels[0]}, tolerance)
        actual = encode_port_coefficients(arguments)
        expected = _encode_port_coefficients_reference(arguments)
        for a, b in zip(actual[0], expected[0], strict=True):
            for x, y in zip(a, b, strict=True):
                np.testing.assert_array_equal(x, y)
        assert actual[1:] == expected[1:]


@pytest.mark.parametrize("spatial", [False, True])
@pytest.mark.parametrize("formulation", ["matrix_lumped", "transport_star"])
def test_local_compiled_matrices_match_original_stage_and_tail_operators(spatial, formulation):
    rng = np.random.default_rng(seed=816)
    for _ in range(4):
        bed = rng.integers(low=0, high=8, size=(5, 5)).astype(float) / 10.
        roughness = rng.uniform(.02, .08, bed.shape) if spatial else .04
        owners = valley_ownership(bed=bed)
        for owner in np.unique(owners):
            mask = owners == owner
            entries, contacts, contact_n = [], [], []
            for side, boundary in enumerate(boundary_nodes(shape=bed.shape)):
                selected = np.flatnonzero(mask.ravel()[boundary])
                cuts = np.r_[0, np.flatnonzero(np.diff(selected) != 1) + 1, len(selected)]
                for start, stop in zip(cuts[:-1], cuts[1:]):
                    nodes = boundary[selected[start:stop]]
                    if not len(nodes):
                        continue
                    entries.append((len(entries), side, nodes))
                    contacts.append(bed.ravel()[nodes])
                    if spatial:
                        contact_n.append(roughness.ravel()[nodes])
            if not entries:
                continue
            parameters = dict(patch=bed, owner_mask=mask, maximum_stage_m=2., entries=entries,
                contacts=tuple(contacts), port_sills=np.array([c.min() for c in contacts]),
                contact_roughness=tuple(contact_n) if spatial else None, manning_n=roughness,
                resolution_m=1., factor=5, formulation=formulation, opening_depth_m=1.e-5,
                interpolation_tolerance=.02, maximum_rows_per_port=32768, unbounded_stage=True,
                tail_reference=float(bed.min()))
            actual = component_stage_tables(parameters)
            reference = component_stage_tables(dict(parameters, _compiled_native=False))
            for candidate, expected in zip(actual[0], reference[0], strict=True):
                np.testing.assert_array_equal(candidate[:, :2], expected[:, :2])
                np.testing.assert_allclose(candidate[:, 2:], expected[:, 2:], rtol=1.e-11, atol=0.)
            for candidate, expected in zip(actual[1], reference[1], strict=True):
                np.testing.assert_array_equal(candidate, expected)
            assert actual[2] <= parameters["interpolation_tolerance"]
            assert actual[3] == reference[3]
