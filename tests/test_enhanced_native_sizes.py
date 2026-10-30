"""FP64 local operators, physical orientation and backend selection, CPU only."""

import numpy as np
import pytest

from floodwarp.enhanced_native_compiled import prepare_component, evaluate_component, native_values
from floodwarp.enhanced_native_sparse import DENSE_SYSTEM_NODES
from floodwarp.enhanced_geometry import (
    build_component_geometry, build_dynamic_geometry, scaled_tail_coefficients, energy_normalized_diagonal,
)
from floodwarp.enhanced_reduction import (
    PortGroup, boundary_nodes, wet_components, reduce_cell, port_coefficients, transport_star,
)


def local_arguments(*, bed, owner, roughness, formulation):
    entries, contacts, contact_n = [], [], []
    for side, boundary in enumerate(boundary_nodes(shape=bed.shape)):
        selected = np.flatnonzero(owner.ravel()[boundary])
        cuts = np.r_[0, np.flatnonzero(np.diff(selected) != 1) + 1, len(selected)]
        for start, stop in zip(cuts[:-1], cuts[1:], strict=True):
            nodes = boundary[selected[start:stop]]
            if len(nodes):
                entries.append((len(entries), side, nodes))
                contacts.append(bed.ravel()[nodes])
                if np.ndim(roughness):
                    contact_n.append(roughness.ravel()[nodes])
    return dict(patch=bed, owner_mask=owner, entries=entries, contacts=tuple(contacts),
        port_sills=np.array([c.min() for c in contacts]), manning_n=roughness,
        contact_roughness=tuple(contact_n) if np.ndim(roughness) else None,
        resolution=1., factor=bed.shape[0], formulation=formulation)


def oracle_values(*, arguments, stage):
    patch = np.where(arguments['owner_mask'], arguments['patch'], stage + 1.)
    active = [e for e in arguments['entries'] if stage > arguments['port_sills'][e[0]]]
    if not active:
        return {}
    contacts = arguments['contacts']
    nodes = tuple(e[2][contacts[e[0]] < stage] for e in active)
    labels = wet_components(bed=patch, stage_m=stage).ravel()
    groups = tuple(PortGroup(side=e[1], component=int(labels[n[0]]), segments=np.arange(len(n)))
        for e, n in zip(active, nodes, strict=True))
    cs = tuple(contacts[e[0]][contacts[e[0]] < stage] for e in active)
    cn = arguments['contact_roughness']
    contact_n = tuple(cn[e[0]][contacts[e[0]] < stage] for e in active) if cn is not None else None
    reduction = reduce_cell(bed_m=patch, stage_m=stage, resolution_m=arguments['resolution'],
        manning_n=arguments['manning_n'], port_groups=groups, contact_sills_m=cs,
        contact_nodes=nodes, contact_manning_n=contact_n, compute_conductance=False,
        momentum_source_weights=np.maximum(stage-patch, 0.)**(5./3.))
    diagonal, _ = port_coefficients(reduction=reduction, formulation='matrix_lumped')
    star_applied = False
    if arguments['formulation'] == 'transport_star':
        diagonal, information = transport_star(reduction=reduction)
        star_applied = information['applied']
    if arguments.get('inertia_weighting') == 'energy_normalized' and not star_applied:
        diagonal = energy_normalized_diagonal(mass=reduction.inertance)
    capacities = np.array([arguments['resolution'] * np.sum((stage-c)**(5./3.)/n)
        for c, n in zip(cs, contact_n, strict=True)]) if cn is not None else np.array([
            arguments['resolution'] * np.sum((stage-c)**(5./3.))/arguments['manning_n'] for c in cs])
    magnitude = np.abs(reduction.flux_basis)
    damping = magnitude.T @ (reduction.edge_resistance * (magnitude @ capacities)**2) / capacities**2
    return {e[0]: np.array([d*(stage-arguments['port_sills'][e[0]]),
        r*(stage-arguments['port_sills'][e[0]])**(10./3.)])
        for e, d, r in zip(active, diagonal, damping, strict=True)}


@pytest.mark.parametrize('shape', [(1, 100), (100, 1), (10, 10), (20, 20)])
@pytest.mark.parametrize('formulation', ['matrix_lumped', 'transport_star'])
@pytest.mark.parametrize('spatial', [False, True])
def test_large_stage_and_tail_coefficients_match_original(shape, formulation, spatial):
    y, x = np.indices(shape)
    bed = (x + y).astype(float) * .01
    roughness = .03 + (x + 2*y) * .0001 if spatial else .04
    args = local_arguments(bed=bed, owner=np.ones(shape, bool), roughness=roughness, formulation=formulation)
    prepared = prepare_component(**args)
    diagnostics = {}
    for stage in (.015, .095, .195, float(bed.max()+.1), float(bed.max()+2.)):
        actual = evaluate_component(prepared=prepared, stage=stage, diagnostics=diagnostics)
        expected = oracle_values(arguments=args, stage=stage)
        assert actual is not None
        assert actual.keys() == expected.keys()
        for port in actual:
            np.testing.assert_allclose(actual[port], expected[port], rtol=1.e-11, atol=0.)
    for reciprocal in (0., .01, .2):
        actual = evaluate_component(prepared=prepared, stage=1., reciprocal=reciprocal,
            reference=0., tail=True, diagnostics=diagnostics)
        expected = scaled_tail_coefficients(reciprocal_head=reciprocal, reference=0.,
            patch=bed, owner_mask=args['owner_mask'], entries=args['entries'], contacts=args['contacts'],
            port_sills=args['port_sills'], contact_roughness=args['contact_roughness'],
            manning_n=roughness, resolution_m=1., factor=shape[0], formulation=formulation)
        if actual is None:
            # The certified encoder intentionally uses its independent oracle
            # near an FP32 rounding boundary. Also check the raw accelerated
            # FP64 result here, rather than dropping that comparison.
            assert set(diagnostics['fallback_reasons']) == {'FP32_rounding_boundary'}
            values, status, _ = native_values(*prepared, 1., reciprocal, 0., True)
            assert status == 1
            actual = {p: value for p, value in enumerate(values) if value[0] > 0.}
        for port in actual:
            np.testing.assert_allclose(actual[port], expected[port], rtol=1.e-11, atol=0.)
    assert diagnostics.get('sparse_lu' if np.prod(shape) > DENSE_SYSTEM_NODES else 'dense_cholesky', 0) > 0
    assert set(diagnostics.get('fallback_reasons', {})) <= {'FP32_rounding_boundary'}
    if np.prod(shape) > DENSE_SYSTEM_NODES:
        assert diagnostics['topology_reuses'] > 0


def test_backend_uses_wet_component_size_and_does_not_join_masked_nodes():
    bed = np.zeros((20, 20))
    owner = np.zeros(bed.shape, bool)
    owner[::3] = True
    args = local_arguments(bed=bed, owner=owner, roughness=.04, formulation='matrix_lumped')
    prepared = prepare_component(**args)
    diagnostics = {}
    actual = evaluate_component(prepared=prepared, stage=.5, diagnostics=diagnostics)
    expected = oracle_values(arguments=args, stage=.5)
    for port in actual:
        np.testing.assert_allclose(actual[port], expected[port], rtol=1.e-11, atol=0.)
    assert diagnostics['maximum_system_nodes'] == 20
    assert diagnostics['maximum_system_nodes'] < DENSE_SYSTEM_NODES
    assert diagnostics['dense_cholesky'] == 7
    assert diagnostics.get('sparse_lu', 0) == 0


def test_dense_dispatch_respects_owner_mask_without_a_caller_bed_sentinel():
    bed = np.zeros((20,20))
    owner = np.zeros_like(bed,bool)
    owner[0,:] = True
    args = local_arguments(bed=bed,owner=owner,roughness=.04,formulation='matrix_lumped')
    diagnostics = {}
    prepared = prepare_component(**args)
    actual = evaluate_component(prepared=prepared,stage=.37,diagnostics=diagnostics)
    expected = oracle_values(arguments=args,stage=.37)
    if actual is None:
        values,status,_ = native_values(*prepared,.37,1.,0.,False)
        assert status == 1
        assert diagnostics['fallback_reasons'] == {'FP32_rounding_boundary':1}
        actual = {p:value for p,value in enumerate(values) if value[0] > 0.}
    for port in actual:
        np.testing.assert_allclose(actual[port],expected[port],rtol=1.e-11,atol=0.)
    assert diagnostics['maximum_system_nodes'] == 20


@pytest.mark.parametrize('axis', [0, 1])
def test_single_row_column_contacts_have_physical_orientation(axis):
    bed = np.arange(100, dtype=float).reshape((1, 100) if axis == 0 else (100, 1)) * .001
    args = local_arguments(bed=bed, owner=np.ones_like(bed, bool), roughness=.04, formulation='matrix_lumped')
    west, east, south, north = boundary_nodes(shape=bed.shape)
    expected_length = 1 if axis == 0 else 100
    assert len(west) == len(east) == expected_length
    assert len(south) == len(north) == 101 - expected_length
    assert np.all(west % bed.shape[1] == 0)
    assert np.all(east % bed.shape[1] == bed.shape[1]-1)
    assert np.all(south // bed.shape[1] == 0)
    assert np.all(north // bed.shape[1] == bed.shape[0]-1)
    # Rotating the physical strip exchanges W/E with S/N. Coefficients follow
    # those faces, rather than treating both strips as row-major x edges.
    a = evaluate_component(prepared=prepare_component(**args), stage=.3)
    other = local_arguments(bed=bed.T, owner=np.ones_like(bed.T, bool), roughness=.04, formulation='matrix_lumped')
    b = evaluate_component(prepared=prepare_component(**other), stage=.3)
    for first, second in ((0, 2), (1, 3), (2, 0), (3, 1)):
        np.testing.assert_allclose(a[first], b[second], rtol=1.e-11, atol=0.)


def test_connected_builder_uses_acceleration_and_exact_flat_closure():
    y, x = np.indices((10, 20))
    for bed in (np.zeros((10, 20)), .01*(x+y)):
        geometry = build_component_geometry(bed_m=bed, factor=10, resolution_m=1., manning_n=.04,
            maximum_stage_m=2., open_sides=(0,1,2,3), formulation='transport_star',
            interpolation_tolerance=.02, maximum_ports_per_cell=512, maximum_rows_per_port=32768,
            unbounded_stage=True)
        assert geometry.diagnostics['connected_domain_identity']
        diagnostics = geometry.diagnostics['reduction_backends']
        if np.any(bed):
            assert diagnostics['dense_cholesky'] > 0
            assert diagnostics['stage_evaluations'] > 0
        else:
            assert diagnostics['analytical_flat_stages'] > 0
            np.testing.assert_allclose(geometry.rows[:, 2:4], .5, rtol=2.e-16, atol=0.)


def test_numeric_failure_is_diagnosed_and_retains_oracle(monkeypatch):
    import floodwarp.enhanced_native_sparse as module
    bed = np.indices((20,20)).sum(axis=0) * .01
    args = local_arguments(bed=bed, owner=np.ones_like(bed, bool), roughness=.04, formulation='matrix_lumped')
    def fail(**arguments):
        raise np.linalg.LinAlgError('deliberate factor failure')
    monkeypatch.setattr(module, 'factor_system', fail)
    diagnostics = {}
    assert evaluate_component(prepared=prepare_component(**args), stage=1., diagnostics=diagnostics) is None
    assert diagnostics['fallbacks'] == 1
    assert diagnostics['fallback_reasons'] == {'LinAlgError: deliberate factor failure': 1}


@pytest.mark.parametrize('formulation', ['matrix_lumped', 'transport_star'])
def test_plateau_activation_one_sided_and_partial_mask(formulation):
    y, x = np.indices((15, 15))
    bed = ((x+y)//5).astype(float) * .1
    owner = np.ones_like(bed, bool)
    owner[4:7, 4:7] = False
    roughness = .03 + .001*x
    args = local_arguments(bed=bed, owner=owner, roughness=roughness, formulation=formulation)
    prepared = prepare_component(**args)
    for activation in np.unique(bed):
        for offset in (-1.e-7, 0., 1.e-7):
            stage = float(activation + offset)
            actual = evaluate_component(prepared=prepared, stage=stage)
            expected = oracle_values(arguments=args, stage=stage)
            assert actual is not None
            assert actual.keys() == expected.keys()
            for port in actual:
                np.testing.assert_allclose(actual[port], expected[port], rtol=1.e-11, atol=0.)


@pytest.mark.parametrize('size', [10,20])
@pytest.mark.parametrize('formulation', ['matrix_lumped','transport_star'])
def test_energy_normalized_weighting_uses_same_dense_and_sparse_systems(size,formulation):
    y,x = np.indices((size,size))
    bed = .01*(x+y)
    args = local_arguments(bed=bed,owner=np.ones_like(bed,bool),roughness=.03+.0003*(x+2*y),formulation=formulation)
    args['inertia_weighting'] = 'energy_normalized'
    prepared = prepare_component(**args)
    diagnostics = {}
    for stage in (.075,.155,float(bed.max()+.17)):
        actual = evaluate_component(prepared=prepared,stage=stage,diagnostics=diagnostics)
        expected = oracle_values(arguments=args,stage=stage)
        assert actual is not None
        for port in actual:
            np.testing.assert_allclose(actual[port],expected[port],rtol=1.e-11,atol=0.)
    assert diagnostics.get('sparse_lu' if size*size > DENSE_SYSTEM_NODES else 'dense_cholesky',0) > 0


def test_connected_energy_normalized_builder_is_accelerated_and_matches_oracle(monkeypatch):
    import floodwarp.enhanced_native_compiled as module
    y,x = np.indices((10,20))
    arguments = dict(bed_m=.01*(x+y),factor=10,resolution_m=1.,manning_n=.04,
        maximum_stage_m=2.,open_sides=(0,1,2,3),formulation='matrix_lumped',
        inertia_weighting='energy_normalized',interpolation_tolerance=.02,
        maximum_ports_per_cell=512,maximum_rows_per_port=32768)
    accelerated = build_dynamic_geometry(**arguments)
    assert accelerated.diagnostics['reduction_backends']['dense_cholesky'] > 0
    def use_oracle(**options):
        return None
    monkeypatch.setattr(module,'prepare_component',use_oracle)
    original = build_dynamic_geometry(**arguments)
    np.testing.assert_array_equal(accelerated.rows[:,:2],original.rows[:,:2])
    np.testing.assert_allclose(accelerated.rows[:,2:],original.rows[:,2:],rtol=1.e-11,atol=0.)


@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('partitioned',[False,True])
def test_single_hydraulic_row_column_have_physical_face_orientation(axis,partitioned):
    shape = (10,30) if axis == 0 else (30,10)
    y,x = np.indices(shape)
    bed = .001*(x+y)
    options = dict(native_partition=np.zeros(shape,np.int32)) if partitioned else {}
    geometry = build_component_geometry(bed_m=bed,factor=10,resolution_m=1.,manning_n=.04,
        maximum_stage_m=1.,open_sides=(0,1,2,3),formulation='matrix_lumped',
        interpolation_tolerance=.02,maximum_ports_per_cell=512,maximum_rows_per_port=32768,**options)
    internal = geometry.endpoints[:,1] >= 0
    assert np.count_nonzero(internal) == 2
    keys = geometry.connection_keys[internal]
    assert np.all(keys[:,4] == (1 if axis == 0 else 3))
    assert np.all(keys[:,1]-keys[:,0] == (1 if axis == 0 else shape[1]))
    for connection in np.flatnonzero(~internal):
        first,_,last,_,side = geometry.connection_keys[connection]
        if side == 0:
            assert first % shape[1] == last % shape[1] == 0
        elif side == 1:
            assert first % shape[1] == last % shape[1] == shape[1]-1
        elif side == 2:
            assert first // shape[1] == last // shape[1] == 0
        else:
            assert first // shape[1] == last // shape[1] == shape[0]-1


def test_port_batches_bound_basis_memory_and_refactor_when_stage_weights_change(monkeypatch):
    import floodwarp.enhanced_native_sparse as module
    y,x = np.indices((20,20))
    bed = .001*(x+y)
    args = local_arguments(bed=bed,owner=np.ones_like(bed,bool),roughness=.04,formulation='matrix_lumped')
    # Every physical boundary pixel is a separate port. Lower the workspace
    # budget to exercise disk-backed coefficient batches on a bounded fixture.
    entries,contacts = [],[]
    for side,boundary in enumerate(boundary_nodes(shape=bed.shape)):
        for node in boundary:
            entries.append((len(entries),side,np.array([node],dtype=np.int64)))
            contacts.append(bed.ravel()[np.array([node])])
    args.update(entries=entries,contacts=tuple(contacts),port_sills=np.array([c[0] for c in contacts]))
    monkeypatch.setattr(module,'WORKSPACE_BYTES',2048)
    monkeypatch.setattr(module,'RHS_BATCH_PORTS',3)
    prepared = prepare_component(**args)
    diagnostics = {}
    for stage in (.137,.237):
        actual = evaluate_component(prepared=prepared,stage=stage,diagnostics=diagnostics)
        expected = oracle_values(arguments=args,stage=stage)
        for port in actual:
            np.testing.assert_allclose(actual[port],expected[port],rtol=1.e-11,atol=0.)
    assert diagnostics['disk_backed_basis'] == 2
    assert diagnostics['factorizations'] == 2
    assert diagnostics['sparse_lu'] == 2
    assert diagnostics['topology_builds'] == 1
    assert diagnostics['topology_reuses'] == 1
    assert diagnostics['solve_batches'] > diagnostics['factorizations']
