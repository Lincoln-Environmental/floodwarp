"""Resolve the numerical shock in space and time after the initial projection transient."""

from pathlib import Path
import os
import numpy as np
from validation.fv1_riemann import report_json
from validation.fv1_riemann_cases import coordinates
from validation.fv2_accuracy import configured
from validation.fv2_flow_truth import case_data


def measure(*, dx, method, duration, interval, device, grade, position, length, width, gravity):
    x, _ = coordinates(length=length, width=width, dx=dx)
    d, _ = case_data(
        case=("transcritical_jump", 0),
        x=x,
        dx=dx,
        duration=duration,
        gravity=gravity,
        head=2.0,
        height=grade,
        half_width=60.0,
        discharge=1.5,
        jump=position,
        quadrature=16,
    )
    s = configured(domain=d, method=method, cap=dx / 160, device=device)
    all_x = np.arange(s.nx + 1) * dx - length / 2
    keep = abs(all_x - position) < 25.0
    past = np.zeros(keep.sum())
    series = []
    for _ in range(round(duration / interval)):
        s.run_device_loop(duration=interval)
        vol = s.get_cumulative_face_volumes()["qx"][:, keep].sum(axis=0) / width
        series.append((vol - past) / interval)
        past = vol
    tail = np.asarray(series)[len(series) // 2 :]
    return dict(
        dx=dx,
        method=method,
        section_x=all_x[keep].tolist(),
        interval=interval,
        flux=np.asarray(series).tolist(),
        final_half_max_departure=float(np.max(abs(tail - 1.5))),
        final_half_max_section_std=float(tail.std(axis=0).max()),
        late_spatial_q_rmse=float(np.mean((tail - 1.5) ** 2) ** 0.5),
        final_half_largest_section_range=float(np.ptp(tail, axis=0).max()),
        balance=s.cumulative_accounting(),
    )


if __name__ == "__main__":
    revision = os.environ.get("FV2_STUDY_REVISION", "halo")
    output = Path("validation/fv2_review/regimes") / ("jump_history_" + revision + ".json")
    rows = []
    for method in os.environ.get("FV2_JUMP_METHODS", "fv1,minmod,mc,adaptive").split(","):
        for dx in (5.0, 2.5, 1.25):
            rows.append(
                measure(
                    dx=dx,
                    method=method,
                    duration=80.0,
                    interval=0.5,
                    device="cuda:0",
                    grade=0.4,
                    position=22.5,
                    length=1920.0,
                    width=20.0,
                    gravity=9.81,
                )
            )
        report_json(path=output, content=rows)
        print(
            method,
            [(r["dx"], r["final_half_max_section_std"]) for r in rows if r["method"] == method],
            flush=True,
        )
