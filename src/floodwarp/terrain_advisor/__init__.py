"""Metric terrain diagnostics and raster export, independent of hydraulic execution."""

from floodwarp.terrain_advisor.diagnostics import analyse_array


def analyse_dem(**options):
    """Read a georeferenced raster using the optional raster dependencies."""
    from floodwarp.terrain_advisor.raster import analyse_dem as read_dem

    return read_dem(**options)


__all__ = ["analyse_array", "analyse_dem"]
