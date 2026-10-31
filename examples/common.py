import csv
import json
from pathlib import Path

import numpy as np

from floodwarp.model import advance, diagnostics, save_checkpoint


def run_example(*, model, output: Path, sample_times, make_plots, use_cuda_graph):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    depths = []
    previous_time = 0.
    previous_volume = 0.
    for target in sample_times:
        advance(model=model, end_time=float(target), use_cuda_graph=use_cuda_graph)
        row = diagnostics(model=model)
        row['mean_outflow_m3_s'] = (row['boundary_outflow']-previous_volume)/(target-previous_time)
        previous_time, previous_volume = float(target), row['boundary_outflow']
        rows.append(row)
        depths.append(model.solver.get_water_depth().copy())
    np.savez_compressed(file=output/'depths.npz', times_s=np.asarray(sample_times), depth=np.asarray(depths), active=model.terrain.active)
    with (output/'hydrograph.csv').open('w', newline='') as stream:
        names = ['time_s', 'mean_outflow_m3_s', 'boundary_outflow', 'boundary_inflow', 'rain', 'infiltration', 'storage_m3', 'residual_m3']
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    (output/'diagnostics.json').write_text(json.dumps(dict(terrain=model.terrain.metadata, samples=rows), indent=2)+'\n')
    save_checkpoint(model=model, path=output/'checkpoint.npz')
    if make_plots:
        plot_results(model=model, rows=rows, depth=depths[-1], output=output)
    print(output, 'final residual m3:', rows[-1]['residual_m3'])


def plot_results(*, model, rows, depth, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    image = axes[0].imshow(np.where(model.terrain.active, depth, np.nan), origin='lower',
        extent=(0, depth.shape[1]*model.terrain.dx, 0, depth.shape[0]*model.terrain.dy))
    figure.colorbar(image, ax=axes[0], label='Water depth (m)')
    axes[0].set(xlabel='East from crop origin (m)', ylabel='North from crop origin (m)', title='Final depth')
    axes[1].plot([row['time_s'] for row in rows], [row['mean_outflow_m3_s'] for row in rows])
    axes[1].set(xlabel='Time (s)', ylabel='Interval mean outlet discharge (m³/s)')
    provenance = model.terrain.metadata.get('provenance', {})
    if provenance.get('name'):
        figure.suptitle(provenance['name'] + ' — 200 m installation demonstration')
    figure.tight_layout()
    figure.savefig(output/'depth_and_hydrograph.png', dpi=150)
    plt.close(figure)
