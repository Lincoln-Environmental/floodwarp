"""FloodWarp surface hydraulics. Copyright Lincoln Agritech Ltd.
Developed by Patrick Durney.

SPDX-License-Identifier: AGPL-3.0-only
"""
from floodwarp.local_inertial_solver import LocalInertialSolver
from floodwarp.full_swe_solver import FullSWESolver
from floodwarp.solver_factory import create_overland_solver
from floodwarp.experimental import ExperimentalSGSWarning

__version__ = "0.1.3"
__all__ = ["LocalInertialSolver", "FullSWESolver", "create_overland_solver", "ExperimentalSGSWarning"]
