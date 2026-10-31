"""Orton Bradley Park 200 m installation crop; historical Purau file names."""
import json
from pathlib import Path

import numpy as np

from floodwarp.inputs import terrain_from_array
from floodwarp.model import create_model
from examples.common import run_example


def load_inputs(*, directory: Path):
    metadata = json.loads((directory/'provenance.json').read_text())
    terrain = terrain_from_array(elevation=np.load(directory/'terrain.npy', allow_pickle=False),
        active=np.load(directory/'active.npy', allow_pickle=False), dx=metadata['dx'], dy=metadata['dy'],
        row_order=metadata['row_order'], vertical_datum=metadata['vertical_datum'], crs=metadata['crs'])
    terrain.metadata.update(transform=metadata['transform'], provenance=metadata)
    with np.load(directory/'boundaries.npz', allow_pickle=False) as data:
        qx, qy = data['qx'].copy(), data['qy'].copy()
    rain = np.loadtxt(directory/'rainfall.csv', delimiter=',', skiprows=1)
    return terrain, qx, qy, (rain[:, 0], rain[:, 1])


if __name__ == "__main__":
    input_directory = Path('examples/purau')
    output_root = Path('outputs/purau')
    methods = ('li', 'fv1')
    manning_n = .035
    infiltration_m_s = 0.
    duration_s = 120.
    sample_interval_s = 10.
    maximum_dt_s = .5
    device = 'cpu'
    make_plots = True
    use_cuda_graph = False

    terrain, qx, qy, rainfall_series = load_inputs(directory=input_directory)
    for method in methods:
        model = create_model(terrain=terrain, solver_method=method, manning_n=manning_n,
            infiltration=infiltration_m_s, rainfall_series=rainfall_series, device=device,
            boundary_qx=qx, boundary_qy=qy, explicit_maximum_dt=maximum_dt_s)
        run_example(model=model, output=output_root/method,
            sample_times=np.arange(sample_interval_s, duration_s+sample_interval_s/2, sample_interval_s),
            make_plots=make_plots, use_cuda_graph=use_cuda_graph)
