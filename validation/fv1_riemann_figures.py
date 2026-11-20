"""Plot measured cost and grid refinement from the retained FV1 investigation."""

from pathlib import Path
import json

import matplotlib
import numpy as np

from validation.fv1_riemann import report_json


def load(*, path):
    return json.loads(path.read_text())


def draw_series(*, axis, rows, x_key, styles):
    for scheme, (colour, marker, line) in styles.items():
        selected = [item for item in rows if item['scheme'] == scheme]
        axis.plot([item[x_key] for item in selected], [item['error'] for item in selected],
                  label=scheme, color=colour, marker=marker, linestyle=line, linewidth=1.45, markersize=5)
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.tick_params(axis='x', which='minor', labelbottom=False)
    axis.grid(visible=True, axis='both', alpha=.25)


def gpu_figure(*, plot, rows, styles, directory):
    figure, panels = plot.subplots(nrows=2, ncols=2, figsize=(10.5, 7), layout='constrained')
    for (left, right), policy in zip(panels, ('common', 'adaptive'), strict=True):
        selected = [dict(scheme=row['scheme'], dx=row['dx_m'], error=row['accuracy']['depth_rmse_m'],
                         wall=row['median_wall_seconds'] * 1.e3) for row in rows if row['policy'] == policy]
        draw_series(axis=left, rows=selected, x_key='dx', styles=styles)
        draw_series(axis=right, rows=selected, x_key='wall', styles=styles)
        left.set_xticks([2.5, 5., 10., 20.], ['2.5', '5', '10', '20'])
        ticks = (10, 20, 40, 80, 160, 250) if policy == 'common' else (4, 8, 16, 32, 64)
        right.set_xticks(ticks, [str(value) for value in ticks])
        left.set_xlabel('Cell spacing (m)')
        right.set_xlabel('Median hydraulic wall time, warmed graph (ms)')
        left.set_ylabel('Depth RMSE vs analytical Stoker (m)')
        left.set_title('Common timestep' if policy == 'common' else 'Scheme-specific CFL')
    panels[0, 1].legend(fontsize=8)
    figure.savefig(directory / 'stoker_accuracy_cost.png', dpi=180)
    plot.close(figure)


def grid_figure(*, plot, data, styles, directory):
    figure, panels = plot.subplots(nrows=2, ncols=3, figsize=(12, 7), layout='constrained')
    names = ('plane', 'irregular', 'steep', 'expansion', 'rainfall')
    rates = []
    for name, axis in zip(names, panels.flat, strict=False):
        selected = [row for row in data['rows'] if row['case'] == name and row['refinement'] == 2 and 'deep_reference' in row]
        plotted = []
        for row in selected:
            errors = [value['depth_rmse_m'] for value in row['deep_reference'].values()]
            plotted.append(dict(dx=row['dx_m'], scheme=row['scheme'], error=float(np.mean(errors))))
            axis.vlines(x=row['dx_m'], ymin=min(errors), ymax=max(errors), colors=styles[row['scheme']][0], alpha=.5)
        draw_series(axis=axis, rows=plotted, x_key='dx', styles=styles)
        axis.set_xticks([2.5, 5., 10., 20.], ['2.5', '5', '10', '20'])
        axis.set_title(name.capitalize())
        axis.set_xlabel('Cell spacing (m)')
        axis.set_ylabel('Depth RMSE to fine-grid solutions (m)')
        for scheme in styles:
            matching = sorted([row for row in plotted if row['scheme'] == scheme], key=spacing, reverse=True)
            orders = [float(np.log(a['error'] / b['error']) / np.log(a['dx'] / b['dx']))
                      for a, b in zip(matching[:-1], matching[1:], strict=True)]
            rates.append(dict(case=name, scheme=scheme, spacings_m=[row['dx'] for row in matching],
                depth_rmse_means_m=[row['error'] for row in matching], pairwise_order=orders))
    handles, labels = panels[0, 0].get_legend_handles_labels()
    panels[1, 2].axis('off')
    panels[1, 2].legend(handles=handles, labels=labels, loc='center', frameon=False)
    figure.savefig(directory / 'synthetic_grid_refinement.png', dpi=180)
    plot.close(figure)
    report_json(path=directory / 'grid_rates.json', content=dict(order_definition='log2 of adjacent errors against the mean of the two fine-solution RMSEs',
                rows=rates))


def spacing(row):
    return row['dx']


def run(*, directory, styles):
    matplotlib.use(backend='Agg')
    import matplotlib.pyplot as plot

    gpu_figure(plot=plot, rows=load(path=directory / 'gpu.json')['rows'], styles=styles, directory=directory)
    grid_figure(plot=plot, data=load(path=directory / 'grid.json'), styles=styles, directory=directory)


if __name__ == "__main__":
    directory = Path('validation/fv1_riemann_review/corrected')
    styles = dict(legacy_hll=('#787878', 'x', ':'), contact_hll=('#2363a1', 'o', '-'),
                  star_contact_hll=('#ad7821', '^', '--'), roe=('#a03053', 's', '-.'), hllc=('#187e6b', 'd', '--'))
    run(directory=directory, styles=styles)
