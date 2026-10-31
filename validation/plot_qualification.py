"""Export scientific maps and scalar hydrographs for a completed matched case."""
from pathlib import Path
import json

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm,SymLogNorm

from validation.qualification_report import load_records
from validation.benchmark_restart import digest_file,publish_json


def plot_case(*,directory,output,minimum_depth_m,error_linear_m,dpi):
    records=load_records(case_directory=directory)
    fields={}
    for method in records:
        with np.load(file=directory/(method+'_0')/'fields.npz',allow_pickle=False) as saved:
            fields[method]={name:saved[name].copy() for name in ('depth','maximum')}
    spacing=records['reference'][0]['signature']['spacing_m']
    ny,nx=fields['reference']['depth'].shape
    extent=(0.,nx*spacing,0.,ny*spacing)
    maximum=max(float(value['maximum'].max()) for value in fields.values())
    norm=LogNorm(vmin=minimum_depth_m,vmax=max(maximum,minimum_depth_m*10.))
    figure=Figure(figsize=(15.,16.),layout='constrained');FigureCanvasAgg(figure=figure)
    grid=figure.add_gridspec(nrows=4,ncols=3)
    labels=dict(reference='LISFLOOD-FP FV1 GPU',fv1='FloodWarp FV1',li='FloodWarp LI')
    for column,method in enumerate(('reference','fv1','li')):
        for row,name in ((0,'maximum'),(2,'depth')):
            axis=figure.add_subplot(grid[row,column])
            shown=np.ma.masked_where(fields[method][name]<=0.,fields[method][name])
            image=axis.imshow(X=shown,origin='lower',extent=extent,norm=norm,cmap='viridis',interpolation='nearest')
            axis.set_title(label=labels[method]+('\nEvent maximum' if name=='maximum' else '\nFinal depth'))
            axis.set_xlabel(xlabel='Local easting [m]');axis.set_ylabel(ylabel='Local northing [m]')
            figure.colorbar(mappable=image,ax=axis,label='Depth [m]',shrink=.8)
    for column,method in enumerate(('fv1','li')):
        axis=figure.add_subplot(grid[1,column])
        error=fields[method]['maximum']-fields['reference']['maximum']
        limit=max(error_linear_m,float(np.max(np.abs(error))))
        image=axis.imshow(X=error,origin='lower',extent=extent,
            norm=SymLogNorm(linthresh=error_linear_m,vmin=-limit,vmax=limit),
            cmap='RdBu_r',interpolation='nearest')
        axis.set_title(label=labels[method]+' maximum minus reference')
        axis.set_xlabel(xlabel='Local easting [m]');axis.set_ylabel(ylabel='Local northing [m]')
        figure.colorbar(mappable=image,ax=axis,label='Depth difference [m]',shrink=.8)
    text_axis=figure.add_subplot(grid[1,2]);text_axis.set_axis_off()
    summary=json.loads((directory/'summary.json').read_text())
    lines=['Matched terrain, rain, soil and coast','Different numerical formulations','']
    for method in ('fv1','li'):
        metrics=summary['results'][method]['physical']
        lines.extend([labels[method],f'Final RMSE: {metrics["depth_rmse_m"]:.6g} m',
            f'Maximum-map RMSE: {metrics["maximum_envelope_rmse_m"]:.6g} m',
            f'IoU at 0.10 m: {metrics["inundation"]["0.1"]["iou"]:.5f}',''])
    text_axis.text(x=0.,y=1.,s='\n'.join(lines),va='top',transform=text_axis.transAxes)
    hydro=figure.add_subplot(grid[3,:])
    for method in ('reference','fv1','li'):
        rows=records[method][0]['ledger']
        times=np.asarray([row['seconds'] for row in rows])
        volume=np.asarray([row['outflow_m3']-row['inflow_m3'] for row in rows])
        discharge=np.diff(np.r_[0.,volume])/np.diff(np.r_[0.,times])
        hydro.plot(times/3600.,discharge,label=labels[method])
    hydro.set_xlabel(xlabel='Simulated time [h]');hydro.set_ylabel(ylabel='Interval-mean net outlet discharge [m³/s]')
    hydro.legend();hydro.grid(visible=True,alpha=.25)
    figure.suptitle(t=directory.name+' — independently checked physical comparison')
    output.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(fname=output,dpi=dpi)
    publish_json(path=output.with_suffix('.json'),value=dict(case=directory.name,
        figure_sha256=digest_file(path=output),driver_sha256=digest_file(path=Path(__file__)),
        fields_sha256={method:digest_file(path=directory/(method+'_0')/'fields.npz') for method in records},
        scalar_interval_s=records['reference'][0]['signature']['sample_interval_s'],
        note='Event maxima are pointwise envelopes, not simultaneous water storage. Colour limits span actual errors.'))


if __name__ == "__main__":
    directory=Path('outputs/qualification_optimized/purau_coast_crop_grid_control')
    output=Path('validation/repair_review/qualified_measurements/coastal_comparison.png')
    minimum_depth_m=.001
    error_linear_m=.001
    dpi=150
    plot_case(directory=directory,output=output,minimum_depth_m=minimum_depth_m,error_linear_m=error_linear_m,dpi=dpi)
