from pathlib import Path

import numpy as np

from floodwarp.inputs import terrain_from_array, boundary_faces
from floodwarp.model import create_model
from examples.common import run_example
from floodwarp.sampling import sample_targets


def synthetic_terrain(*, shape, dx, dy, slope):
    elevation = np.broadcast_to(slope*dx*np.arange(shape[1])[::-1], shape).copy()
    return terrain_from_array(elevation=elevation, dx=dx, dy=dy,
                              row_order='south_to_north', vertical_datum='local')


if __name__ == "__main__":
    shape = (12, 16)
    dx = 5.
    dy = 5.
    slope = .01
    methods = ('fv1', 'li')
    manning_n = .035
    rainfall_m_s = 20./3.6e6
    duration_s = 10.
    sample_interval_s = 2.
    maximum_dt_s = .5
    device = 'cpu'
    make_plots = False
    use_cuda_graph = False
    output_root = Path('outputs/synthetic')

    terrain = synthetic_terrain(shape=shape, dx=dx, dy=dy, slope=slope)
    qx, qy = boundary_faces(terrain=terrain)
    qx[:, -1] = 2
    for method in methods:
        model = create_model(terrain=terrain, solver_method=method, manning_n=manning_n,
            rainfall=rainfall_m_s, device=device, boundary_qx=qx, boundary_qy=qy,
            explicit_maximum_dt=maximum_dt_s)
        run_example(model=model, output=output_root/method,
            sample_times=sample_targets(duration=duration_s, interval=sample_interval_s),
            make_plots=make_plots, use_cuda_graph=use_cuda_graph)
