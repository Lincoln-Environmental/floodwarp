"""Compare exact per-stage dynamic face scheduling to dense and fixed-mask launches."""

from pathlib import Path
import json
import numpy as np
from validation.fv1_riemann import report_json
from validation.fv1_riemann_catchment import checkpoint_solver
from validation.fv2_regime_profile import record
from validation.fv2_storm_cost import record as orton


def compare(*, output, checkpoint, device, modes, repeats):
    rows = []
    for case in ("dry_patch", "wet_large"):
        fingerprints = []
        for mode in modes:
            row = record(
                method="adaptive",
                case=case,
                mode=mode,
                device=device,
                dx=5.0,
                length=1920.0,
                width=1280.0,
                seconds=8.0,
                cap=5 / 64,
                repeats=repeats,
            )
            rows.append(row)
            fingerprints.append(row["field_sha256"])
        if len(set(fingerprints)) != 1:
            raise ArithmeticError("Different dynamic/dense/sparse hydraulic results")
    state = checkpoint_solver(scheme="contact_hll", device="cpu", path=checkpoint).capture_state()
    for factor in (4, 1):
        expected = None
        for mode in modes:
            row, q = orton(
                state=state,
                factor=factor,
                method="fv2_adaptive",
                device=device,
                duration=12.0,
                repeats=repeats,
                kernel_interval=1.0,
                mode=mode,
            )
            if expected is None:
                expected = q.copy()
            np.testing.assert_array_equal(q, expected)
            rows.append(row)
            print("Orton", factor * 5, mode, row["median_wall_s"], flush=True)
    report_json(path=output, content=rows)


if __name__ == "__main__":
    compare(
        output=Path("validation/fv2_review/regimes/sparse.json"),
        checkpoint=Path(
            json.loads(Path("validation/fv1_riemann_review/corrected/catchment.json").read_text())[
                "checkpoint"
            ]
        ),
        device="cuda:0",
        modes=("dense", "sparse", "dynamic"),
        repeats=5,
    )
