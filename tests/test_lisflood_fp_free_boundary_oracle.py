"""Validation-only reference arithmetic; no production code import of oracle."""

import numpy as np
import pytest

from validation.lisflood_fp_free_boundary import free_boundary_unit_discharge


def test_float32_free_operator_matches_reference_intermediate_rounding() -> None:
    result = free_boundary_unit_discharge(
        q_old=0.13, edge_depth=0.3456789, edge_bed=1000.125,
        inward_depth=0.3123456, inward_bed=1000.0, manning_n=0.035,
        spacing=10.0, dt=0.25, arithmetic="float32",
    )
    assert isinstance(result, np.float32)
    assert float(result) == pytest.approx(0.14275598526000977, abs=1e-9)


def test_float64_operator_resolves_float32_rounding_difference() -> None:
    controls = dict(
        q_old=0.13, edge_depth=0.3456789, edge_bed=1000.125,
        inward_depth=0.3123456, inward_bed=1000.0, manning_n=0.035,
        spacing=10.0, dt=0.25,
    )
    fp32 = free_boundary_unit_discharge(**controls, arithmetic="float32")
    fp64 = free_boundary_unit_discharge(**controls, arithmetic="float64")
    assert isinstance(fp64, np.float64)
    assert float(fp64) == pytest.approx(0.1427539602965614, abs=1e-12)
    assert abs(float(fp32) - float(fp64)) > 1e-6


@pytest.mark.parametrize("arithmetic", ["float32", "float64", "float_surface_only"])
def test_face_orientation(arithmetic: str) -> None:
    controls = dict(
        q_old=-0.03, edge_depth=0.6, edge_bed=15.2,
        inward_depth=0.5, inward_bed=15.25, manning_n=0.04,
        spacing=5.0, dt=0.5, arithmetic=arithmetic,
    )
    positive = free_boundary_unit_discharge(**controls, direction=1.0)
    negative = free_boundary_unit_discharge(**controls, direction=-1.0)
    assert float(positive) == pytest.approx(-float(negative), abs=1e-14)
    assert positive > 0


def test_reference_arithmetic_not_exposed_by_production() -> None:
    import inspect
    from floodwarp.local_inertial_solver import LocalInertialSolver
    assert "free_outflow_arithmetic_mode" not in inspect.signature(LocalInertialSolver.__init__).parameters
