import numpy as np
import pytest
import warp as wp

from floodwarp.full_swe_reference import hll_flux
from floodwarp.fv2_solver import HancockSolver, MUSCLEulerSolver
from validation.fv1_diffusion_domains import design, front_depth, front_mean, simple_wave


def test_frictional_front_satisfies_momentum_and_cell_average():
    for case in design():
        if case.family != "front":
            continue
        x = np.array([0.01, 0.1, 1.0, 5.0, 15.0])
        dh = 1.0e-5
        h = front_depth(distance=x, case=case)
        deriv = (front_depth(distance=x + dh, case=case) - front_depth(distance=x - dh, case=case)) / (2 * dh)
        np.testing.assert_allclose(
            deriv, (case.n * case.speed) ** 2 / h ** (4 / 3) - case.slope, rtol=4e-7, atol=2e-10
        )
        low = np.array([-2.0, 0.0, 0.1, 25.0])
        for hi in (low + 2.5, low + 10):
            mean = front_mean(low=low, high=hi, case=case, order=16)
            other = front_mean(low=low, high=hi, case=case, order=48)
            np.testing.assert_allclose(mean, other, atol=2e-9, rtol=1e-8)


def test_simple_wave_satisfies_conservation_laws():
    case = next(c for c in design() if c.family == "gravity_pulse")
    g = 9.80665
    x = np.linspace(-35, 35, 51)
    delta = 1e-4
    h1, m1 = simple_wave(case=case, x=x, t=15.0 - delta, gravity=g)
    h2, m2 = simple_wave(case=case, x=x, t=15.0 + delta, gravity=g)
    hl, ml = simple_wave(case=case, x=x - delta, t=15.0, gravity=g)
    hr, mr = simple_wave(case=case, x=x + delta, t=15.0, gravity=g)
    np.testing.assert_allclose((h2 - h1 + mr - ml) / (2 * delta), 0.0, atol=1e-11)
    np.testing.assert_allclose(
        (m2 - m1 + mr**2 / hr - ml**2 / hl + g / 2 * (hr**2 - hl**2)) / (2 * delta), 0.0, atol=1e-11
    )


def test_low_froude_preserves_transport_and_one_sided_branches():
    rng = np.random.default_rng(seed=1160)
    h = rng.uniform(1e-3, 1.0, (2, 4096))
    velocity = rng.uniform(-5.0, 5.0, (2, 4096, 2))
    q = np.concatenate((h[..., None], h[..., None] * velocity), axis=-1)
    q[0, :10] = 0.0
    kwargs = dict(left=q[0], right=q[1], gravity=9.81, dry_depth=0.0001)
    baseline = hll_flux(flux_scheme="contact_hll", **kwargs)
    corrected = hll_flux(flux_scheme="low_froude_hll", **kwargs)
    np.testing.assert_array_equal(corrected[:, [0, 2]], baseline[:, [0, 2]])
    coupled = (
        np.maximum(
            abs(velocity[0, :, 0]) / np.sqrt(9.81 * h[0]), abs(velocity[1, :, 0]) / np.sqrt(9.81 * h[1])
        )
        >= 1
    )
    coupled[:10] = True
    np.testing.assert_array_equal(corrected[coupled], baseline[coupled])
    normal = np.array([[1.0, -0.01, 0.0], [1.0, 0.01, 0.0]])
    test = hll_flux(
        left=normal[:1], right=normal[1:], gravity=9.81, dry_depth=0.0001, flux_scheme="low_froude_hll"
    )
    a = np.sqrt(9.81) + 0.01
    expected = 9.81 / 2 + 0.01**2 - a * 0.01 * (0.01 / np.sqrt(9.81))
    np.testing.assert_allclose(test[0, 1], expected, atol=1e-14)


@pytest.mark.parametrize("factory", (HancockSolver, MUSCLEulerSolver))
@pytest.mark.parametrize("device", ("cuda:0", "cpu"))
def test_large_datum_dry_lake(factory, device):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    y, x = np.indices((18, 32))
    z = 2.0e6 + 0.3 * np.sin(x / 3) + 0.25 * np.cos(y / 3)
    z[:, 16] = 2.0e6 + 5
    h = np.where(x < 16, np.maximum(0.0, 2.0e6 + 0.1 - z), 0.0)
    s = factory(
        ny=18,
        nx=32,
        dx=1.0,
        dy=2.0,
        bed_elevation=z,
        manning_n=0.02,
        device=device,
        active_mask=np.ones_like(h, bool),
    )
    s.set_water_depth(depth=h)
    s.run_device_loop(duration=0.5)
    conserved = s.get_conservative_state()
    np.testing.assert_allclose(conserved[..., 0], h, atol=3e-15, rtol=0)
    np.testing.assert_allclose(conserved[..., 1:], 0.0, atol=3e-15)
    assert abs(s.cumulative_accounting()["residual"]) < 1e-12
