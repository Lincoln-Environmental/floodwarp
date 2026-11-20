"""Exact steady-profile residual decomposition before evolving any time steps."""

from pathlib import Path
import numpy as np
from floodwarp.fv2_reference import directional, traces
from floodwarp.full_swe_reference import hll_flux
from validation.fv1_riemann import report_json
from validation.fv1_riemann_cases import coordinates
from validation.fv2_flow_truth import bump_state, case_data


def residuals(*, dx, limiter, regime, jump, gravity, discharge):
    x, _ = coordinates(length=640.0, width=dx, dx=dx)
    d, _ = case_data(
        case=(regime, 0),
        x=x,
        dx=dx,
        duration=0.0,
        gravity=gravity,
        head=2.0,
        height=0.4,
        half_width=60.0,
        discharge=discharge,
        jump=jump,
        quadrature=24,
    )
    q = d.initial
    f, pl, pr, se = directional(q=q, z=d.bed, g=gravity, dry=1e-4, axis=1, limiter=limiter)
    source = (pl[:, 1:] - pr[:, :-1] - gravity * q[..., 0] * se) / dx
    edges = np.concatenate((x[0] - dx / 2, [x[0, -1] + dx / 2]))
    _, h = bump_state(
        x=edges, regime=regime, height=0.4, half_width=60.0, discharge=discharge, gravity=gravity, jump=jump
    )
    # Bernoulli flow satisfies dF_normal/dx = -g*h*dz/dx, including the
    # conjugate-depth jump (continuous momentum flux). This integrates the true source.
    exact_flux = discharge**2 / h + gravity * h**2 / 2
    exact_source = np.diff(exact_flux) / dx
    minus, plus, _, dz = traces(q=q, z=d.bed, axis=1, limiter=limiter, dry=1e-4, gravity=gravity)
    # Hold depth, hydrostatic cut and source fixed. Impose the known constant
    # cell momentum on each uncut trace, then apply the same velocity-preserving cut.
    l = plus[:, :-1].copy()
    r = minus[:, 1:].copy()
    rise = np.diff(d.bed, axis=1) - (dz[:, :-1] + dz[:, 1:]) / 2
    for side, step in ((l, rise), (r, -rise)):
        orig = side[..., 0].copy()
        side[..., 0] = np.maximum(0, orig - np.maximum(0, step))
        side[..., 1] = discharge * (side[..., 0] / orig)
    forcing_control = f.copy()
    forcing_control[:, 1:-1] = hll_flux(
        left=l, right=r, gravity=gravity, dry_depth=1e-4, flux_scheme="contact_hll"
    )
    rows = []
    for region, mask in (
        ("just_shock", abs(x - jump) < 2.1 * dx),
        ("smooth", (abs(x) < 55) & (abs(x - jump) > 3 * dx)),
    ):
        for name, flux, bed_source in (
            ("full", f, source),
            ("exact_bed_source", f, exact_source[None, :]),
            ("constant_momentum_trace", forcing_control, source),
        ):
            dh = -np.diff(flux[..., 0], axis=1) / dx
            dm = -np.diff(flux[..., 1], axis=1) / dx + bed_source
            rows.append(
                dict(
                    regime=regime,
                    dx=dx,
                    limiter=limiter,
                    region=region,
                    operator=name,
                    dh_dt_rmse=float(np.mean(dh[mask] ** 2) ** 0.5),
                    dm_dt_rmse=float(np.mean(dm[mask] ** 2) ** 0.5),
                )
            )
    return rows


if __name__ == "__main__":
    output = Path("validation/fv2_review/regimes/residual.json")
    rows = []
    for regime in ("transcritical_jump", "transcritical"):
        for dx in (10.0, 5.0, 2.5, 1.25):
            for limiter in ("mc", "minmod", "adaptive"):
                rows.extend(
                    residuals(dx=dx, limiter=limiter, regime=regime, jump=22.5, gravity=9.81, discharge=1.5)
                )
    report_json(path=output, content=rows)
