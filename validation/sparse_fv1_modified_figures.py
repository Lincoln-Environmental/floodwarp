"""Regenerate comparison figures from small exported tables, without hydraulics."""

from pathlib import Path
import json

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def runtime_figure(*, summary, table, output, colours, size, dpi, bar_width):
    figure = Figure(figsize=size, layout='constrained')
    FigureCanvasAgg(figure=figure)
    axes = figure.subplots(nrows=2, ncols=1)
    labels = dict(dense='Dense FV1', released='Released sparse FV1', modified='Modified sparse FV1')
    for name in ('dense', 'released', 'modified'):
        axes[0].plot(np.r_[0., table['time_h']], np.r_[0., table[name + '_hydraulic_seconds_cumulative']],
            label=labels[name], color=colours[name])
    axes[0].set_xlabel(xlabel='Simulation time [h]')
    axes[0].set_ylabel(ylabel='Cumulative hydraulic runtime [s]')
    axes[0].set_xlim(left=0., right=summary['duration_s'] / 3600.)
    axes[0].legend()
    phases = summary['phases']
    positions = np.arange(len(phases))
    keys = dict(dense='dense_hydraulic_seconds', released='released_sparse_hydraulic_seconds', modified='modified_hydraulic_seconds')
    for offset, name in zip((-1, 0, 1), ('dense', 'released', 'modified')):
        axes[1].bar(x=positions + offset * bar_width, height=[row[keys[name]] for row in phases],
            width=bar_width, label=labels[name], color=colours[name])
    axes[1].set_xticks(ticks=positions, labels=[f'{row["start_h"]:g}–{row["end_h"]:g} h' for row in phases])
    axes[1].set_ylabel(ylabel='Hydraulic runtime per phase [s]')
    axes[1].set_xlabel(xlabel='Matched event phase')
    axes[1].legend()
    for axis in axes:
        axis.grid(visible=True, axis='y', alpha=.2)
    figure.suptitle(t='Orton Bradley Park — full 72-hour FV1 execution runtime')
    figure.supxlabel(t='One new modified run against retained same-device controls; warm-up/validation excluded from hydraulic timers.', fontsize=9)
    figure.savefig(fname=output, dpi=dpi)


def hydrograph_figure(*, table, output, colours, size, dpi, peak_window_h, minimum_difference_range, notice):
    figure = Figure(figsize=size, layout='constrained')
    FigureCanvasAgg(figure=figure)
    axes = figure.subplots(nrows=3, ncols=1)
    hours = table['interval_mid_h']
    columns = dict(modified='floodwarp_modified_sparse_fv1', released='floodwarp_released_sparse_fv1',
        dense='floodwarp_dense_fv1', reference='lisflood_fv1')
    labels = dict(modified='Modified sparse FV1', released='Released sparse FV1', dense='Dense FV1', reference='External LISFLOOD-FP FV1')
    styles = dict(modified='-', released='--', dense=':', reference='-.')
    for name in ('reference', 'dense', 'released', 'modified'):
        discharge = table[columns[name] + '_mean_net_outlet_m3_s']
        for axis in axes[:2]:
            axis.plot(hours, discharge, label=labels[name], color=colours[name], linestyle=styles[name])
    current = table[columns['modified'] + '_mean_net_outlet_m3_s']
    maximum_difference = 0.
    for name in ('dense', 'released'):
        difference = current - table[columns[name] + '_mean_net_outlet_m3_s']
        maximum_difference = max(maximum_difference, float(np.max(np.abs(difference))))
        axes[2].plot(hours, difference, label='Modified − ' + labels[name], color=colours[name])
    limit = max(minimum_difference_range, maximum_difference)
    axes[2].set_ylim(bottom=-limit, top=limit)
    axes[2].set_title(label=f'Execution differences · actual max |ΔQ| = {maximum_difference:.6g} m³/s')
    axes[2].set_ylabel(ylabel='Δ interval-mean discharge [m³/s]')
    axes[0].set_title(label='Full event · actual 60-second interval-mean net discharge')
    axes[1].set_title(label='Peak detail · samples placed at interval midpoints')
    peak_hour = float(hours[np.argmax(current)])
    axes[1].set_xlim(left=peak_hour - peak_window_h, right=peak_hour + peak_window_h)
    for axis in (axes[0], axes[2]):
        axis.set_xlim(left=0., right=float(table['interval_end_s'][-1] / 3600.))
    for axis in axes[:2]:
        axis.set_ylabel(ylabel='Discharge [m³/s]')
    for axis in axes:
        axis.set_xlabel(xlabel='Time from simulation start [h]')
        axis.grid(visible=True, alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle(t='Orton Bradley Park — modified and retained FV1 hydrographs')
    figure.supxlabel(t=notice, fontsize=8)
    figure.savefig(fname=output, dpi=dpi)


def make_figures(*, directory, colours, runtime_size, hydrograph_size, dpi,
                 bar_width, peak_window_h, minimum_difference_range, notice):
    summary = json.loads((directory / 'summary.json').read_text())
    runtime = np.genfromtxt(fname=directory / 'runtime_72h.csv', delimiter=',', names=True)
    hydrograph = np.genfromtxt(fname=directory / 'hydrographs_72h.csv', delimiter=',', names=True)
    runtime_figure(summary=summary, table=runtime, output=directory / 'runtime_comparison.png',
        colours=colours, size=runtime_size, dpi=dpi, bar_width=bar_width)
    hydrograph_figure(table=hydrograph, output=directory / 'hydrograph_comparison.png',
        colours=colours, size=hydrograph_size, dpi=dpi, peak_window_h=peak_window_h,
        minimum_difference_range=minimum_difference_range, notice=notice)


if __name__ == "__main__":
    directory = Path('validation/sparse_modified_event_review')
    colours = dict(modified='#116f97', released='#756094', dense='#666666', reference='#d0662f')
    runtime_size = (12., 9.)
    hydrograph_size = (12., 12.)
    dpi = 180
    bar_width = .25
    peak_window_h = 2.
    minimum_difference_range = 1.e-6
    notice = 'Research demonstration only — not an official flood-hazard assessment. Independent site-specific validation is required before decision-making.'
    make_figures(directory=directory, colours=colours, runtime_size=runtime_size,
        hydrograph_size=hydrograph_size, dpi=dpi, bar_width=bar_width,
        peak_window_h=peak_window_h, minimum_difference_range=minimum_difference_range, notice=notice)
