"""Reproducible same-device FloodWarp versus independent LISFLOOD-FP FV1.

Only ordinary arrays/files cross the process boundary. No research package is
imported. The separately installed FP64 reference executable retains upstream
FV1 operators and its documented rainfall/soil/stage/diagnostic adapters.
"""
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
import ctypes
import hashlib
import json
import os
import subprocess
import uuid

import numpy as np
import warp as wp

from floodwarp.config import WP_FLOAT
from floodwarp.inputs import terrain_from_array, boundary_faces
from floodwarp.model import create_model, diagnostics, save_checkpoint
from floodwarp.rainfall import Hyetograph


@dataclass
class Case:
    name: str
    bed: np.ndarray
    active: np.ndarray
    roughness: np.ndarray
    multiplier: np.ndarray
    f0: np.ndarray
    fd: np.ndarray
    conductivity: np.ndarray
    ports: np.ndarray
    initial: np.ndarray
    times: np.ndarray
    rates: np.ndarray
    spacing: float
    duration: float
    sample_interval: float
    provenance: dict


class ReferenceHeader(ctypes.Structure):
    _fields_=[('magic',ctypes.c_char*16),('identity',ctypes.c_char*65),
        ('count',ctypes.c_uint64),('steps',ctypes.c_uint64),('trace_count',ctypes.c_uint64),
        ('nx',ctypes.c_int),('ny',ctypes.c_int),('pitch',ctypes.c_int),
        ('time',ctypes.c_double),('dt',ctypes.c_double),('minimum_dt',ctypes.c_double),
        ('next_save',ctypes.c_double),('initial_storage',ctypes.c_double),
        ('previous_volume',ctypes.c_double),('cumulative_time',ctypes.c_double),
        ('instantaneous',ctypes.c_double*2),('cumulative',ctypes.c_double*2)]


def file_hash(*, path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write_json(*, path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(target=path)


def prepare_directory(*, directory):
    if directory.exists():
        directory.rename(target=directory.with_name(directory.name+'.incomplete-'+uuid.uuid4().hex))
    directory.mkdir(parents=True,exist_ok=False)


def load_event(*, directory, duration, sample_interval):
    """Read retained SI arrays directly; never load a prepared model."""
    with np.load(file=directory/'comparison_inputs.npz',allow_pickle=False) as packet:
        values={name:packet[name][::-1].copy() if packet[name].ndim==2 else packet[name].copy() for name in packet.files}
    with (directory/'inputs_f64.bin').open('rb') as stream:
        ny,nx,count=np.fromfile(file=stream,dtype='<i4',count=3)
        times=np.fromfile(file=stream,dtype='<f8',count=int(count))
        rates=np.fromfile(file=stream,dtype='<f8',count=int(count))
        fields=np.fromfile(file=stream,dtype='<f8').reshape(6,int(ny),int(nx))[:,::-1].copy()
    np.testing.assert_array_equal(actual=fields[0].astype(bool),desired=values['active'])
    roughness=np.loadtxt(fname=directory/'roughness.asc',skiprows=6)[::-1].copy()
    return Case(name='purau_event',bed=values['bed'],active=values['active'].astype(bool),
        roughness=roughness,multiplier=fields[1],f0=fields[2],fd=fields[3],conductivity=fields[4],
        ports=np.r_[values['ports'][1:],np.zeros_like(values['ports'][:1])].astype(bool),initial=np.zeros_like(values['bed']),times=times,rates=rates,
        spacing=5.,duration=duration,sample_interval=sample_interval,
        provenance=dict(source=str(directory),files={name:file_hash(path=directory/name) for name in
            ('comparison_inputs.npz','inputs_f64.bin','roughness.asc','manifest.json')},
            scope='Established full Purau 5 m terrain, rainfall, exponential soil and coastal contacts'))


def crop_event(*, event, size, duration, sample_interval):
    rows,columns=np.nonzero(event.ports)
    if not len(rows):raise ValueError('Missing retained coastal outlets')
    index=len(rows)//2
    row=max(0,min(int(rows[index])-size+2,event.bed.shape[0]-size))
    column=max(0,min(int(columns[index])-size//2,event.bed.shape[1]-size))
    selection=(slice(row,row+size),slice(column,column+size))
    fields={name:getattr(event,name)[selection].copy() for name in
        ('bed','active','roughness','multiplier','f0','fd','conductivity','ports','initial')}
    return replace(event,name='purau_coast_crop',duration=duration,sample_interval=sample_interval,
        provenance=dict(event.provenance,crop=[row,column,size],scope='Coastal crop with artificial closed cut edges'),**fields)


def refine_case(*, case, factor):
    fields={name:getattr(case,name).repeat(factor,axis=0).repeat(factor,axis=1) for name in
        ('bed','active','roughness','multiplier','f0','fd','conductivity','initial')}
    # Only the north face of the last refined row belongs to the original port.
    ports=np.zeros(fields['bed'].shape,dtype=bool)
    ports[factor-1::factor,:]=case.ports.repeat(factor,axis=1)
    return replace(case,name=case.name+'_refined',spacing=case.spacing/factor,ports=ports,**fields)


def analytical_case(*, name, shape, spacing, duration, sample_interval, roughness, depth, rain, infiltration):
    zeros=np.zeros(shape)
    initial=np.full(shape,depth)
    if name=='dam_break':initial[:,shape[1]//2:]=depth/2.
    return Case(name=name,bed=zeros,active=np.ones(shape,bool),roughness=np.full(shape,roughness),
        multiplier=np.ones(shape),f0=np.full(shape,infiltration),fd=np.ones(shape),
        conductivity=np.full(shape,infiltration),ports=np.zeros(shape,bool),initial=initial,
        times=np.array([0.,duration/2.,duration]),rates=np.full(3,rain),spacing=spacing,
        duration=duration,sample_interval=sample_interval,provenance=dict(scope='Flat closed analytical control'))


def case_signature(*, case, numerical, binary):
    fields={name:hashlib.sha256(np.ascontiguousarray(getattr(case,name)).tobytes()).hexdigest()
        for name in ('bed','active','roughness','multiplier','f0','fd','conductivity','ports','initial','times','rates')}
    sources={path.name:file_hash(path=path) for path in Path('src/floodwarp').glob('*.py')}
    return dict(name=case.name,shape=list(case.bed.shape),duration_s=case.duration,
        sample_interval_s=case.sample_interval,spacing_m=case.spacing,fields=fields,numerical=numerical,
        external_binary_sha256=file_hash(path=binary),sources=sources,provenance=case.provenance)


def write_ascii(*, path, values, spacing):
    with path.open('w') as stream:
        stream.write(f'ncols {values.shape[1]}\nnrows {values.shape[0]}\nxllcorner 0\nyllcorner 0\ncellsize {spacing:.17g}\nNODATA_value -9999\n')
        np.savetxt(fname=stream,X=values[::-1],fmt='%.17g')


def prepare_reference(*, case, directory, numerical):
    prepare_directory(directory=directory)
    active=np.pad(case.active,1)
    bed=np.pad(np.where(case.active,case.bed,numerical['wall_elevation_m']),1,
        constant_values=numerical['wall_elevation_m'])
    fields=[active.astype(float),np.pad(case.multiplier,1),np.pad(case.f0,1),
        np.pad(case.fd,1,constant_values=1.),np.pad(case.conductivity,1)]
    ports=np.pad(case.ports,1)[::-1]
    flags=np.zeros_like(ports,dtype=float);flags[:-1]=ports[1:]
    if np.any(flags.astype(bool)&active[::-1]):raise ValueError('Port must adjoin inactive north cell')
    write_ascii(path=directory/'terrain.asc',values=bed,spacing=case.spacing)
    write_ascii(path=directory/'roughness.asc',values=np.pad(case.roughness,1),spacing=case.spacing)
    write_ascii(path=directory/'initial.asc',values=np.pad(case.initial,1),spacing=case.spacing)
    with (directory/'inputs_f64.bin').open('wb') as stream:
        np.asarray([*active.shape,len(case.times)],dtype='<i4').tofile(file=stream)
        case.times.astype('<f8').tofile(file=stream);case.rates.astype('<f8').tofile(file=stream)
        for field in fields:field[::-1].astype('<f8').tofile(file=stream)
        flags.astype('<f8').tofile(file=stream)
    lines=['cuda','fv1','DEMfile terrain.asc','manningfile roughness.asc','startfile initial.asc',
        'resroot purau','dirroot results',f'sim_time {case.duration:.17g}',
        f'initial_tstep {numerical["maximum_dt_s"]:.17g}',f'massint {case.sample_interval:.17g}',
        f'saveint {case.sample_interval:.17g}',f'cfl {numerical["cfl"]:.17g}',
        f'gravity {numerical["gravity_m_s2"]:.17g}',f'depththresh {numerical["dry_depth_m"]:.17g}',
        'output_precision 17']
    (directory/'purau.par').write_text('\n'.join(lines)+'\n')


def read_reference_checkpoint(*, path):
    with path.open('rb') as stream:
        header=ReferenceHeader.from_buffer_copy(stream.read(ctypes.sizeof(ReferenceHeader)))
        if header.magic!=b'PURAU-FV1-R1':raise ValueError('Reference checkpoint schema mismatch')
        arrays=np.fromfile(file=stream,dtype=np.float64,count=14*header.count).reshape(14,header.ny+2,header.pitch)
        trace=np.fromfile(file=stream,dtype=np.float64,count=header.trace_count)
        footer=np.fromfile(file=stream,dtype=np.uint64,count=1)
        if len(footer)!=1 or footer[0]!=header.steps or stream.read(1):raise ValueError('Incomplete reference checkpoint')
    return header,arrays,trace


def run_reference(*, case, directory, numerical, binary, signature, timeout):
    began=perf_counter()
    prepare_reference(case=case,directory=directory,numerical=numerical)
    prepared=perf_counter()-began
    identity=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PURAU_INPUT=str((directory/'inputs_f64.bin').resolve()),
        PURAU_CHECKPOINT_ID=identity,PURAU_CHECKPOINT_WRITE=str((directory/'checkpoint.bin').resolve()),
        PURAU_STOP_AT=str(case.duration),MATCHED_SOURCE_WET_DEPTH='-1')
    environment.pop('PURAU_CHECKPOINT_READ',None)
    started=perf_counter()
    with (directory/'run.log').open('w') as stream:
        subprocess.run(args=[str(binary.resolve()),'purau.par'],cwd=directory,env=environment,
            stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    process=perf_counter()-started
    profile=json.loads((directory/'capacity_profile.json').read_text())
    header,arrays,trace=read_reference_checkpoint(path=directory/'checkpoint.bin')
    if not np.isclose(header.time,case.duration,atol=1e-9,rtol=0.):raise ValueError('Incomplete reference run')
    rows=np.atleast_1d(np.genfromtxt(fname=directory/'purau_ledger.csv',delimiter=',',names=True))
    unique={float(row['seconds']):row for row in rows if row['seconds']>0.}
    ledger=[{name:float(row[name]) for name in rows.dtype.names} for row in unique.values()]
    maximum=arrays[9,1:header.ny+1,1:header.nx+1][::-1][1:-1,1:-1].copy()
    encoded=np.fromfile(file=directory/f'state_{int(case.duration):06d}.bin',dtype=np.float64)
    depth=encoded.reshape(3,header.ny,header.nx)[0,::-1][1:-1,1:-1].copy()
    np.savez_compressed(file=directory/'fields.npz',depth=depth,maximum=maximum)
    record=dict(solver='LISFLOOD-FP FV1 GPU',steps=int(header.steps),time_s=header.time,
        preparation_seconds=prepared,process_seconds=process,end_to_end_seconds=prepared+process,
        hydraulic_seconds=profile['simulation_seconds_excluding_diagnostics'],profile=profile,
        peak_explicit_device_bytes=profile['peak_explicit_cuda_allocation_bytes'],
        dt_min_s=float(trace.min()),dt_mean_s=float(trace.mean()),dt_max_s=float(trace.max()),
        ledger=ledger,signature=signature)
    write_json(path=directory/'record.json',value=record)
    return record


def build_floodwarp(*, case, method, device, numerical):
    terrain=terrain_from_array(elevation=case.bed,active=case.active,dx=case.spacing,dy=case.spacing,
        row_order='south_to_north',vertical_datum='local')
    bx,by=boundary_faces(terrain=terrain)
    # Native ports belong to their active south donor, including raster steps.
    rows,cols=np.nonzero(case.ports);by[rows+1,cols]=3
    options=dict(explicit_minimum_dt=numerical['minimum_dt_s'],explicit_maximum_dt=numerical['maximum_dt_s'])
    if method=='fv1':options.update(gravity=numerical['gravity_m_s2'],dry_depth=numerical['dry_depth_m'],
        cfl=numerical['cfl'],stage_boundary_momentum=numerical['fv1_stage_momentum'])
    else:options.update(track_max_water_depth=True,infiltration_method='exponential',infiltration_f0_m_per_s=case.f0,
        infiltration_fd_m=case.fd,infiltration_k_m_per_s=case.conductivity,adaptive_dt_growth_factor=None)
    model=create_model(terrain=terrain,solver_method=method,device=device,manning_n=case.roughness,
        initial_depth=case.initial,rainfall_series=(case.times,case.rates),boundary_qx=bx,boundary_qy=by,
        stage_qy=0.,stage_datum='local',**options)
    solver=model.solver
    if method=='li':
        solver.gravity=WP_FLOAT(numerical['gravity_m_s2']);solver.dry_depth=WP_FLOAT(numerical['dry_depth_m'])
        solver.cfl_number=WP_FLOAT(numerical['cfl'])
    else:solver.set_exponential_infiltration(f0=case.f0,fd=case.fd,conductivity=case.conductivity)
    solver.set_rainfall_spatial_multiplier(multiplier=case.multiplier)
    return model


def device_array_bytes(*, solver):
    arrays=[value for value in vars(solver).values() if isinstance(value,wp.array)]
    arrays.extend(getattr(solver,'_fields',{}).values())
    unique={array.ptr:array for array in arrays}
    return sum(array.capacity for array in unique.values())


def run_floodwarp(*, case, method, directory, device, numerical, signature, graph):
    prepare_directory(directory=directory)
    wp.synchronize_device(device=device)
    began=perf_counter()
    model=build_floodwarp(case=case,method=method,device=device,numerical=numerical)
    solver=model.solver
    wp.synchronize_device(device=device)
    prepared=perf_counter()-began
    samples=np.arange(case.sample_interval,case.duration+case.sample_interval/2.,case.sample_interval)
    ledger=[];statistics=[];hydraulic=0.;started=perf_counter()
    for target in samples:
        current=solver.time if method=='fv1' else solver.simulation_time
        if method=='li':
            stats=solver.run_device_loop(duration=float(target-current),rainfall_hyetograph=model.rainfall,
                use_cuda_graph=graph)
        else:stats=solver.run_device_loop(duration=float(target-current),use_cuda_graph=graph)
        wp.synchronize_device(device=device)
        hydraulic+=stats['loop_seconds'];statistics.append(stats)
        balance=diagnostics(model=model)
        ledger.append(dict(seconds=float(target),storage_m3=balance['storage_m3'],rain_m3=balance['rain'],
            infiltration_m3=balance['infiltration'],outflow_m3=balance['boundary_outflow'],
            inflow_m3=balance['boundary_inflow'],residual_m3=balance['residual_m3']))
        print(f'{case.name} {method} {target/3600:.3f} h; solver {hydraulic:.3f} s',flush=True)
        if len(samples)>1:save_checkpoint(model=model,path=directory/'restart.npz')
    wp.synchronize_device(device=device)
    process=perf_counter()-started
    depth=solver.get_water_depth().copy()
    maximum=solver.get_max_water_depth().copy() if hasattr(solver,'get_max_water_depth') and getattr(solver,'track_max_water_depth',True) else depth.copy()
    np.savez_compressed(file=directory/'fields.npz',depth=depth,maximum=maximum)
    steps=sum(int(row['steps']) for row in statistics)
    record=dict(solver='FloodWarp '+method.upper(),steps=steps,time_s=case.duration,
        preparation_seconds=prepared,process_seconds=process,end_to_end_seconds=prepared+process,
        hydraulic_seconds=hydraulic,peak_explicit_device_bytes=device_array_bytes(solver=solver),
        dt_min_s=min(row['dt_min_s'] for row in statistics),dt_mean_s=case.duration/steps,
        dt_max_s=max(row['dt_max_s'] for row in statistics),ledger=ledger,signature=signature,
        graph=graph,statistics=statistics)
    write_json(path=directory/'record.json',value=record)
    return record


def physical_metrics(*, candidate, reference, active, thresholds, indifference_m, candidate_rows, reference_rows):
    error=candidate[active]-reference[active]
    extent={}
    for threshold in thresholds:
        selected=active & ~((np.abs(candidate-threshold)<=indifference_m) & (np.abs(reference-threshold)<=indifference_m))
        wet_a=(candidate>=threshold)&selected;wet_b=(reference>=threshold)&selected
        union=np.count_nonzero(wet_a|wet_b)
        extent[str(threshold)]=dict(iou=float(np.count_nonzero(wet_a&wet_b)/union) if union else 1.,
            indifferent_cells=int(np.count_nonzero(active&~selected)))
    t=np.array([row['seconds'] for row in candidate_rows])
    tr=np.array([row['seconds'] for row in reference_rows])
    va=np.array([row['outflow_m3']-row['inflow_m3'] for row in candidate_rows])
    vb=np.array([row['outflow_m3']-row['inflow_m3'] for row in reference_rows])
    vb=np.interp(x=t,xp=tr,fp=vb)
    qa=np.diff(np.r_[0.,va])/np.diff(np.r_[0.,t]);qb=np.diff(np.r_[0.,vb])/np.diff(np.r_[0.,t])
    return dict(depth_rmse_m=float(np.sqrt(np.mean(error**2))),maximum_depth_error_m=float(np.max(np.abs(error))),
        inundation=extent,hydrograph_rmse_m3_s=float(np.sqrt(np.mean((qa-qb)**2))),
        peak_discharge_m3_s=float(qa.max()),reference_peak_discharge_m3_s=float(qb.max()),
        peak_interval_end_s=float(t[int(np.argmax(qa))]),reference_peak_interval_end_s=float(t[int(np.argmax(qb))]),
        cumulative_discharge_error_m3=float(va[-1]-vb[-1]),
        normalised_balance_residual=max(abs(row['residual_m3'])/max(1.,row['rain_m3']+row['inflow_m3']) for row in candidate_rows))


def run_benchmarks(*, cases, numerical, binary, output, device, repeats, timeout, thresholds, indifference_m):
    wp.init()
    output.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for case in cases:
        signature=case_signature(case=case,numerical=numerical,binary=binary)
        # A short independent warm-up is retained separately from measured repeats.
        for method in ('reference','li','fv1'):
            warm=replace(case,duration=min(case.duration,case.sample_interval),sample_interval=min(case.duration,case.sample_interval))
            warm_path=output/case.name/(method+'_warmup')
            if not (warm_path/'record.json').exists():
                if method=='reference':run_reference(case=warm,directory=warm_path,numerical=numerical,binary=binary,signature=signature,timeout=timeout)
                else:run_floodwarp(case=warm,method=method,directory=warm_path,device=device,numerical=numerical,signature=signature,graph=None)
        records={}
        for method in ('reference','li','fv1'):
            rows=[]
            for repeat in range(repeats):
                directory=output/case.name/f'{method}_{repeat}'
                path=directory/'record.json'
                if path.exists():
                    record=json.loads(path.read_text())
                    if record['signature']!=signature:raise ValueError('Completed benchmark source/configuration drift: '+str(path))
                elif method=='reference':record=run_reference(case=case,directory=directory,numerical=numerical,binary=binary,signature=signature,timeout=timeout)
                else:record=run_floodwarp(case=case,method=method,directory=directory,device=device,numerical=numerical,signature=signature,graph=None)
                rows.append(record)
                print(case.name,method,repeat,record['hydraulic_seconds'],flush=True)
            records[method]=dict(hydraulic_median_s=float(np.median([row['hydraulic_seconds'] for row in rows])),
                hydraulic_range_s=[min(row['hydraulic_seconds'] for row in rows),max(row['hydraulic_seconds'] for row in rows)],
                end_to_end_median_s=float(np.median([row['end_to_end_seconds'] for row in rows])),
                device_bytes=max(row['peak_explicit_device_bytes'] for row in rows),steps=rows[0]['steps'],
                dt_min_s=rows[0]['dt_min_s'],dt_mean_s=rows[0]['dt_mean_s'])
        reference=json.loads((output/case.name/'reference_0/record.json').read_text())
        with np.load(file=output/case.name/'reference_0/fields.npz') as arrays:
            depth=arrays['depth'].copy();maximum=arrays['maximum'].copy()
        for method in ('li','fv1'):
            record=json.loads((output/case.name/f'{method}_0/record.json').read_text())
            with np.load(file=output/case.name/f'{method}_0/fields.npz') as arrays:
                candidate=arrays['depth'].copy();candidate_maximum=arrays['maximum'].copy()
            records[method]['physical']=physical_metrics(candidate=candidate,reference=depth,active=case.active,
                thresholds=thresholds,indifference_m=indifference_m,candidate_rows=record['ledger'],reference_rows=reference['ledger'])
            maximum_error=(candidate_maximum-maximum)[case.active]
            records[method]['physical']['maximum_envelope_rmse_m']=float(np.sqrt(np.mean(maximum_error**2)))
            records[method]['physical']['maximum_envelope_max_error_m']=float(np.max(np.abs(maximum_error)))
            records[method]['runtime_ratio_to_lisflood_fv1']=records[method]['hydraulic_median_s']/records['reference']['hydraulic_median_s']
            records[method]['simulated_seconds_per_wall_second']=case.duration/records[method]['hydraulic_median_s']
        summaries.append(dict(case=case.name,duration_s=case.duration,spacing_m=case.spacing,
            shape=list(case.bed.shape),numerical=numerical,results=records))
        write_json(path=output/case.name/'summary.json',value=summaries[-1])
        write_json(path=output/'summary.json',value=summaries)
    return summaries


if __name__ == "__main__":
    reference_root=Path('/home/patrickdurney/PycharmProjects/Overlander/.external')
    binary=reference_root/'synthetic_catchment_nn_matched_20261028_v1/fv1_build/build/lisflood'
    event_directory=reference_root/'purau_lisflood_fp64_20261023/event_5m_fp64_v3'
    output=Path('outputs/repair_benchmark')
    device='cuda:0'
    scope=os.environ.get('FLOODWARP_BENCHMARK_SCOPE','small')
    repeats=1 if scope in ('long','full72') else 3
    timeout=14400.
    numerical=dict(cfl=.4,maximum_dt_s=1.,minimum_dt_s=.000001,gravity_m_s2=9.80665,
        dry_depth_m=.0001,wall_elevation_m=1000000.,fv1_stage_momentum='discharge_extrapolation')
    thresholds=(.01,.10)
    indifference_m=.00001
    event=load_event(directory=event_directory,duration=3600.,sample_interval=600.)
    crop=crop_event(event=event,size=40,duration=3600.,sample_interval=600.)
    cases=[analytical_case(name='lake',shape=(8,40),spacing=5.,duration=60.,sample_interval=60.,
        roughness=.035,depth=.2,rain=0.,infiltration=0.),
        analytical_case(name='dam_break',shape=(4,80),spacing=5.,duration=60.,sample_interval=60.,
        roughness=.035,depth=.2,rain=0.,infiltration=0.),
        analytical_case(name='rain_infiltration',shape=(16,16),spacing=5.,duration=60.,sample_interval=60.,
        roughness=.035,depth=0.,rain=.00002,infiltration=.00004),crop]
    if scope=='full':cases=[event]
    if scope=='long':cases=[replace(crop,name='purau_coast_crop_72h',duration=259200.,sample_interval=3600.)]
    if scope=='full72':cases=[replace(event,name='purau_event_72h',duration=259200.,sample_interval=3600.)]
    if scope=='refined':cases=[refine_case(case=crop,factor=2)]
    if scope=='cfl':
        numerical=dict(numerical,cfl=.2)
        cases=[replace(crop,name='purau_coast_crop_cfl02')]
    summaries=run_benchmarks(cases=cases,numerical=numerical,binary=binary,output=output,device=device,
        repeats=repeats,timeout=timeout,thresholds=thresholds,indifference_m=indifference_m)
