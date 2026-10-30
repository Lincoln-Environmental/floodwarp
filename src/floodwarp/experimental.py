"""Visible, noninteractive notices for unqualified hydraulic formulations."""
import warnings


class ExperimentalSGSWarning(UserWarning):
    """SGS results require independent verification."""


def warn_experimental_sgs(*, formulation):
    warnings.warn(
        f"Experimental SGS formulation selected: {formulation}. "
        "This method has not been fully validated across hydraulic conditions "
        "and may produce inaccurate water depths, flow velocities or inundation "
        "extents. Results require independent verification.",
        ExperimentalSGSWarning,
        stacklevel=3,
    )
