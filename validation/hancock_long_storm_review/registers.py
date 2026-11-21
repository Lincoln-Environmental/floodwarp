"""Compiled forward-kernel CUDA properties on the qualification device."""

import json
from pathlib import Path

import warp as wp

from floodwarp import fv1_plus_kernels, fv2_kernels
from validation.fv2_regime_profile import record as run_control


def record(*, kernels, device, output):
    data = []
    for kernel in kernels:
        for layout, concrete in (("dense", kernel), ("tiles", kernel.tile_kernel)):
            suggested, grid = wp.get_suggested_block_size(
                kernel=concrete, device=device
            )
            data.append(
                dict(
                    kernel=kernel.key,
                    layout=layout,
                    suggested_block_size=suggested,
                    min_grid_size=grid,
                    **wp.get_cuda_kernel_properties(kernel=concrete, device=device),
                )
            )
    output.write_text(json.dumps(data, indent=2) + "\n")


def measure_fusion(*, output, modes, cases, options):
    rows = []
    for case in cases:
        for mode in modes:
            expected = None
            for fused in (False, True):
                row = run_control(
                    mode=mode,
                    case=case,
                    method="hancock_adaptive",
                    fuse_prediction=fused,
                    **options,
                )
                if expected is not None and row["field_sha256"] != expected:
                    raise ArithmeticError("Fusion changed the final hydraulic solution")
                expected = row["field_sha256"]
                row["fused_prediction"] = fused
                rows.append(row)
    output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    kernels = (
        fv2_kernels.split_sources,
        fv2_kernels.slopes,
        fv1_plus_kernels.predict,
        fv1_plus_kernels.slopes_predict,
        fv2_kernels.faces,
        fv2_kernels.euler,
    )
    output = Path(__file__).with_name("registers.json")
    device = "cuda:0"
    record(kernels=kernels, device=device, output=output)
    options = dict(
        device=device,
        dx=1.0,
        length=1024.0,
        width=640.0,
        seconds=2.0,
        cap=1 / 32,
        repeats=5,
    )
    modes = ("dense", "tiles")
    cases = ("stoker", "dry_patch")
    measure_fusion(
        output=output.with_name("fusion.json"),
        modes=modes,
        cases=cases,
        options=options,
    )
