"""Optional full-domain FloodWarp replay from the published ordinary inputs.

This is a substantial 72-hour simulation, not an installation smoke test.
Figures and benchmark tables can instead be regenerated without hydraulics
using reproduce.py and the retained completed results.
"""

from pathlib import Path
import json

import numpy as np

from floodwarp.inputs import read_terrain, boundary_faces
from floodwarp.model import create_model, advance, diagnostics, save_checkpoint


def load_forcing(*, path):
    with path.open('rb') as stream:
        ny, nx, count = np.fromfile(file=stream, dtype='<i4', count=3)
        times = np.fromfile(file=stream, dtype='<f8', count=int(count))
        rates = np.fromfile(file=stream, dtype='<f8', count=int(count))
        values = np.fromfile(file=stream, dtype='<f8').reshape(6, int(ny), int(nx))[:, ::-1].copy()
    return times, rates, values


def sample_targets(*, duration, interval):
    """Include the exact requested final time, without duplicates or overshoot."""
    if not np.isfinite([duration, interval]).all() or min(duration, interval) <= 0.:
        raise ValueError('Positive finite duration and sample interval required')
    interior = np.arange(1, int(np.floor(duration / interval)) + 1, dtype=np.float64) * interval
    return np.r_[interior[interior < duration], float(duration)]


def run(*, directory, output, method, device, duration, sample_interval, maximum_dt):
    targets = sample_targets(duration=duration, interval=sample_interval)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Replay starts fresh; use an empty output directory. Checkpoint restoration is explicit.')
    terrain = read_terrain(path=directory / 'outputs/orton_bradley_dem_5m.tif', vertical_datum='NZVD2016')
    times, rates, fields = load_forcing(path=directory / 'inputs/inputs_f64.bin')
    np.testing.assert_array_equal(actual=fields[0].astype(bool), desired=terrain.active)
    roughness = np.loadtxt(fname=directory / 'inputs/roughness.asc', skiprows=6)[::-1].copy()
    # Packet flags lie on the ghost row north of each physical south donor.
    ports = np.r_[fields[5, 1:], np.zeros_like(fields[5, :1])].astype(bool)
    qx, qy = boundary_faces(terrain=terrain)
    rows, columns = np.nonzero(ports)
    qy[rows + 1, columns] = 3
    options = dict(explicit_minimum_dt=1.e-6, explicit_maximum_dt=maximum_dt)
    if method == 'fv1':
        options.update(gravity=9.80665, dry_depth=.0001, cfl=.4, stage_boundary_momentum='discharge_extrapolation')
    else:
        options.update(infiltration_method='exponential', infiltration_f0_m_per_s=fields[2],
            infiltration_fd_m=fields[3], infiltration_k_m_per_s=fields[4], adaptive_dt_growth_factor=None,
            track_max_water_depth=True)
    model = create_model(terrain=terrain, solver_method=method, device=device, manning_n=roughness,
        rainfall_series=(times, rates), initial_depth=0., boundary_qx=qx, boundary_qy=qy,
        stage_qy=0., stage_datum='NZVD2016', **options)
    if method == 'fv1':
        model.solver.set_exponential_infiltration(f0=fields[2], fd=fields[3], conductivity=fields[4])
    else:
        model.solver.gravity = type(model.solver.gravity)(9.80665)
        model.solver.dry_depth = type(model.solver.dry_depth)(.0001)
        model.solver.cfl_number = type(model.solver.cfl_number)(.4)
    model.solver.set_rainfall_spatial_multiplier(multiplier=fields[1])
    output.mkdir(parents=True, exist_ok=True)
    ledger = []
    for target in targets:
        advance(model=model, end_time=float(target))
        ledger.append(diagnostics(model=model))
        if target % 3600. == 0. or target >= duration:
            save_checkpoint(model=model, path=output / 'restart.npz')
            (output / 'ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    np.savez_compressed(file=output / 'fields.npz', depth=model.solver.get_water_depth(), maximum=model.solver.get_max_water_depth())
    return model


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    output = Path('outputs/orton_bradley_replay_fv1')
    method = 'fv1'
    device = 'cuda:0'
    duration = 259200.
    sample_interval = 60.
    maximum_dt = 1.
    run(directory=directory, output=output, method=method, device=device, duration=duration,
        sample_interval=sample_interval, maximum_dt=maximum_dt)
