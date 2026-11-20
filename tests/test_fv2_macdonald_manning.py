"""Mathematical checks for the varying-depth Manning benchmark."""
import numpy as np

from validation.fv2_macdonald_manning import exact_fields, averaged_fields


def test_macdonald_steady_momentum_balance():
    x = np.linspace(-250.0, 250.0, 2001)
    q = 0.5
    n = 0.025
    g = 9.81
    h0 = 0.5
    decay = 0.001
    z, h, _ = exact_fields(
        x=x, depth_at_zero=h0, decay=decay, discharge=q, manning=n, gravity=g
    )
    lhs = np.gradient(q * q / h + 0.5 * g * h * h, x, edge_order=2)
    rhs = -g * h * np.gradient(z, x, edge_order=2) - g * n * n * q * abs(q) / h ** (7.0 / 3.0)
    np.testing.assert_allclose(lhs[5:-5], rhs[5:-5], rtol=0.002, atol=2.e-7)


def test_macdonald_cell_averages_against_quadrature():
    q = 0.5
    n = 0.025
    g = 9.81
    h0 = 0.5
    decay = 0.001
    dx = 10.0
    centre = np.array([-60.0, -10.0, 30.0, 80.0])
    bed, state = averaged_fields(
        x=centre, dx=dx, depth_at_zero=h0, decay=decay, discharge=q, manning=n, gravity=g
    )
    nodes, weights = np.polynomial.legendre.leggauss(32)
    points = centre[:, None] + 0.5 * dx * nodes[None, :]
    continuous_bed, continuous_h, _ = exact_fields(
        x=points, depth_at_zero=h0, decay=decay, discharge=q, manning=n, gravity=g
    )
    np.testing.assert_allclose(bed, continuous_bed @ (weights / 2.0), rtol=1.e-13, atol=1.e-13)
    np.testing.assert_allclose(state[:, 0], continuous_h @ (weights / 2.0), rtol=1.e-13, atol=1.e-13)
    np.testing.assert_allclose(state[:, 1], q, rtol=0.0, atol=0.0)


def test_macdonald_short_solver_replay_cpu(tmp_path):
    import json

    from validation.fv2_macdonald_manning import run

    output = tmp_path / "macdonald.json"
    run(
        output=output, device="cpu", methods=("fv1", "mc"), spacings=(20.0,),
        gauges=(-80.0, 0.0, 80.0), length=640.0, width=40.0,
        duration=2.0, interval=1.0, depth_at_zero=0.5,
        decay=0.001, discharge=0.5, manning=0.025, gravity=9.81,
    )
    rows = json.loads(output.read_text())["rows"]
    assert [row["method"] for row in rows] == ["fv1", "mc"]
    for row in rows:
        assert row["steps"] > 0
        assert row["depth_min_m"] >= 0.0
        assert np.isfinite(row["discharge_rmse_m2_s"])
        assert abs(row["ledger"]["residual"]) < 1.e-7
