"""Reusable Warp functions for subgrid storage hypsometry."""
import warp as wp

from floodwarp.config import BED_FLOAT, WP_FLOAT


@wp.func
def subgrid_stage_from_storage(
    storage: WP_FLOAT,
    cell_z: wp.array3d(dtype=BED_FLOAT),
    cell_storage_breakpoints: wp.array3d(dtype=BED_FLOAT),
    j: int,
    i: int,
    subgrid_cell_count: int,
) -> BED_FLOAT:
    """Invert one coarse cell's native-resolution storage curve."""
    if storage <= WP_FLOAT(0.0):
        return cell_z[j, i, 0]
    eta = cell_z[j, i, subgrid_cell_count - 1] + BED_FLOAT(
        BED_FLOAT(storage) - cell_storage_breakpoints[j, i, subgrid_cell_count - 1]
    )
    # Breakpoints can contain last-bit nonmonotonicity on equal-height
    # plateaus. Preserve the original first-match arithmetic exactly.
    m = int(0)
    found = int(0)
    while m < subgrid_cell_count - 1 and found == 0:
        if BED_FLOAT(storage) <= cell_storage_breakpoints[j, i, m + 1]:
            eta = cell_z[j, i, m] + BED_FLOAT(
                (BED_FLOAT(storage) - cell_storage_breakpoints[j, i, m])
                * BED_FLOAT(subgrid_cell_count)
                / BED_FLOAT(m + 1)
            )
            found = 1
        m += 1
    return eta


@wp.func
def subgrid_wet_fraction_from_storage(
    storage_depth: WP_FLOAT,
    cell_z: wp.array3d(dtype=BED_FLOAT),
    cell_storage_breakpoints: wp.array3d(dtype=BED_FLOAT),
    j: int,
    i: int,
    subgrid_cell_count: int,
) -> WP_FLOAT:
    """Return the fraction of native subcells with positive water depth."""
    eta = subgrid_stage_from_storage(
        storage_depth,
        cell_z,
        cell_storage_breakpoints,
        j,
        i,
        subgrid_cell_count,
    )
    wet_count = int(0)
    upper = subgrid_cell_count
    while wet_count < upper:
        middle = (wet_count + upper) // 2
        if eta > cell_z[j, i, middle]:
            wet_count = middle + 1
        else:
            upper = middle
    return WP_FLOAT(wet_count) / WP_FLOAT(subgrid_cell_count)
