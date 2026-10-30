BOUNDARY_INTERNAL = 0
BOUNDARY_CLOSED = 1
BOUNDARY_FREE_OUTFLOW = 2
# Prescribed water level with hydrostatic wet/dry reconstruction. At a dry
# exterior the pressure head stops at the local face sill, not below its bed.
# Where stage exceeds that sill this remains a two-way stage boundary.
BOUNDARY_SPECIFIED_STAGE = 3
# A specified-stage face on an irregular mask boundary. Valid only on
# an interior qy face whose j-1 cell is active and j cell is inactive;
# this is the north-facing polygon edge in south-to-north layout.
BOUNDARY_MASKED_NORTH_STAGE = 4
# A specified-stage face on an irregular mask boundary. Valid only on
# an interior qx face whose i-1 cell is inactive and i cell is active;
# this is a west-facing raster step along the polygon edge.
BOUNDARY_MASKED_WEST_STAGE = 5
# One-way free outflow across an irregular mask boundary (an open
# hydraulic edge against nodata; ghost cell at the active bed with zero
# depth, outward flux only). Valid only on an interior qx face whose
# i-1 cell is inactive and i cell is active; outward is q < 0 (westward).
BOUNDARY_MASKED_WEST_FREE = 6
# As above for an interior qx face whose i-1 cell is active and i cell
# is inactive; outward is q > 0 (eastward).
BOUNDARY_MASKED_EAST_FREE = 7
# As above for an interior qy face whose j-1 cell is inactive and j cell
# is active; outward is q < 0 (southward).
BOUNDARY_MASKED_SOUTH_FREE = 8
# As above for an interior qy face whose j-1 cell is active and j cell
# is inactive; outward is q > 0 (northward).
BOUNDARY_MASKED_NORTH_FREE = 9
# Fixed-stage faces on the opposite two orientations of an irregular mask.
# EAST: active west cell / inactive east cell. SOUTH: inactive south cell /
# active north cell.
BOUNDARY_MASKED_EAST_STAGE = 10
BOUNDARY_MASKED_SOUTH_STAGE = 11


def get_boundary_name(btype: int) -> str:
    if btype == BOUNDARY_INTERNAL:
        return "INTERNAL"
    elif btype == BOUNDARY_CLOSED:
        return "CLOSED"
    elif btype == BOUNDARY_FREE_OUTFLOW:
        return "FREE_OUTFLOW"
    elif btype == BOUNDARY_SPECIFIED_STAGE:
        return "SPECIFIED_STAGE"
    elif btype == BOUNDARY_MASKED_NORTH_STAGE:
        return "MASKED_NORTH_STAGE"
    elif btype == BOUNDARY_MASKED_EAST_STAGE:
        return "MASKED_EAST_STAGE"
    elif btype == BOUNDARY_MASKED_SOUTH_STAGE:
        return "MASKED_SOUTH_STAGE"
    elif btype == BOUNDARY_MASKED_WEST_STAGE:
        return "MASKED_WEST_STAGE"
    elif btype == BOUNDARY_MASKED_WEST_FREE:
        return "MASKED_WEST_FREE"
    elif btype == BOUNDARY_MASKED_EAST_FREE:
        return "MASKED_EAST_FREE"
    elif btype == BOUNDARY_MASKED_SOUTH_FREE:
        return "MASKED_SOUTH_FREE"
    elif btype == BOUNDARY_MASKED_NORTH_FREE:
        return "MASKED_NORTH_FREE"
    return "UNKNOWN"
