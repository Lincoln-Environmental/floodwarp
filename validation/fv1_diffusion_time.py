"""Analytical timestep isolation where reconstructed spatial gradients are exact."""

from dataclasses import asdict
from pathlib import Path
import numpy as np
from validation.fv1_diffusion_domains import Case, prepare, sheet
from validation.fv2_local_inertia import build
from validation.fv1_riemann import report_json


def temporal(*, case, dx, timesteps, methods, gravity, device, output):
    d, _, _, _, _ = prepare(case=case, dx=dx, gravity=gravity, samples=1, order=32)
    h, q = sheet(case=case, t=case.duration, gravity=gravity)
    terminal = h ** (2 / 3) * np.sqrt(case.slope) / case.n
    rate = gravity * case.slope / terminal
    exact_vol = (
        h
        * terminal
        / rate
        * np.log(np.cosh(rate * case.duration) + case.speed / terminal * np.sinh(rate * case.duration))
    )
    rows = []
    for method in methods:
        for cap in timesteps:
            s = build(d=d, method=method, dx=dx, cap=cap, device=device, gravity=gravity)
            stats = s.run_device_loop(duration=case.duration)
            tran = s.get_cumulative_face_volumes()["qx"][:, s.nx // 2].sum() / case.width
            cell = s.get_conservative_state()[s.ny // 2, s.nx // 2, 1]
            rows.append(
                dict(
                    method=method,
                    cap=cap,
                    dt=stats["dt_mean_s"],
                    steps=stats["steps"],
                    momentum_error=float(abs(cell - q)),
                    transport_error=float(abs(tran - exact_vol)),
                )
            )
    report_json(path=output, content=dict(case=asdict(case), exact_q=q, exact_transport=exact_vol, rows=rows))


if __name__ == "__main__":
    case = Case("accelerating", "sheet", 0.05, 0.08, 0.12, 0.3, 0.0, 0.0, 400.0, 40.0, 4.0, 0.0, (0.0,))
    methods = ("fv1", "euler_mc", "hancock_mc", "mc")
    timesteps = (1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64)
    temporal(
        case=case,
        dx=5.0,
        timesteps=timesteps,
        methods=methods,
        gravity=9.80665,
        device="cuda:0",
        output=Path("validation/fv1_diffusion_review/accelerating_time.json"),
    )
