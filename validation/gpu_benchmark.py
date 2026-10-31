"""Reproducible same-device FloodWarp versus independent LISFLOOD-FP FV1.

Only ordinary arrays/files cross the process boundary. No research package is
imported. The separately installed FP64 reference executable retains upstream
FV1 operators and its documented rainfall/soil/stage/diagnostic adapters.
"""
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
import ctypes
import gc
import hashlib
import json
import os
import subprocess

import numpy as np
import warp as wp
from scipy.ndimage import binary_erosion

from floodwarp.config import WP_FLOAT
from floodwarp.inputs import terrain_from_array, boundary_faces
from floodwarp.model import create_model, diagnostics, save_checkpoint, load_checkpoint
from floodwarp.rainfall import Hyetograph
from validation.benchmark_restart import (publish_json, sample_targets, read_progress,
    publish_progress, read_completed,publish_record)


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
    checkpoint_interval: float | None = None
    field_interval: float | None = None


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
    publish_json(path=path,value=value)


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
    ports=np.zeros(shape,bool)
    if name=='boundary_outflow':ports[-1,:]=True
    return Case(name=name,bed=zeros,active=np.ones(shape,bool),roughness=np.full(shape,roughness),
        multiplier=np.ones(shape),f0=np.full(shape,infiltration),fd=np.ones(shape),
        conductivity=np.full(shape,infiltration),ports=ports,initial=initial,
        times=np.array([0.,duration/2.,duration]),rates=np.full(3,rain),spacing=spacing,
        duration=duration,sample_interval=sample_interval,provenance=dict(scope='Flat closed analytical control'))


def case_signature(*, case, numerical, binary):
    fields={name:hashlib.sha256(np.ascontiguousarray(getattr(case,name)).tobytes()).hexdigest()
        for name in ('bed','active','roughness','multiplier','f0','fd','conductivity','ports','initial','times','rates')}
    sources={path.name:file_hash(path=path) for path in Path('src/floodwarp').glob('*.py')}
    drivers={path.name:file_hash(path=path) for path in
        (Path(__file__),Path(__file__).with_name('benchmark_restart.py'))}
    reference_report=binary.parent.parent/'build_report.json'
    return dict(name=case.name,shape=list(case.bed.shape),duration_s=case.duration,
        sample_interval_s=case.sample_interval,spacing_m=case.spacing,fields=fields,numerical=numerical,
        checkpoint_interval_s=case.checkpoint_interval,
        field_interval_s=case.field_interval,
        external_binary_sha256=file_hash(path=binary),sources=sources,benchmark_sources=drivers,
        external_build_report_sha256=file_hash(path=reference_report) if reference_report.exists() else None,
        provenance=case.provenance)


def preserve_sources(*, directory, signature):
    """Retain the actual implementation, including runs begun before a commit."""
    directory.mkdir(parents=True,exist_ok=True)
    for name,digest in signature['sources'].items():
        target=directory/name
        if target.exists() and file_hash(path=target)==digest:continue
        if target.exists():raise ValueError('Preserved source snapshot differs: '+str(target))
        current=Path('src/floodwarp')/name
        if file_hash(path=current)!=digest:
            raise ValueError('Recorded numerical source changed without a preserved snapshot: '+name)
        target.write_bytes(current.read_bytes())
    for name,digest in signature.get('benchmark_sources',{}).items():
        current=Path(__file__).with_name(name);target=directory/('benchmark-'+name)
        if target.exists() and file_hash(path=target)==digest:continue
        if target.exists() or file_hash(path=current)!=digest:
            raise ValueError('Benchmark driver snapshot differs: '+str(target))
        target.write_bytes(current.read_bytes())


def write_ascii(*, path, values, spacing):
    with path.open('w') as stream:
        stream.write(f'ncols {values.shape[1]}\nnrows {values.shape[0]}\nxllcorner 0\nyllcorner 0\ncellsize {spacing:.17g}\nNODATA_value -9999\n')
        np.savetxt(fname=stream,X=values[::-1],fmt='%.17g')


def prepare_reference(*, case, directory, numerical):
    directory.mkdir(parents=True,exist_ok=True)
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
        raw=stream.read(ctypes.sizeof(ReferenceHeader))
        if len(raw)!=ctypes.sizeof(ReferenceHeader):raise ValueError('Incomplete reference checkpoint header')
        header=ReferenceHeader.from_buffer_copy(raw)
        if header.magic!=b'PURAU-FV1-R1':raise ValueError('Reference checkpoint schema mismatch')
        if (header.nx<=0 or header.ny<=0 or header.pitch<header.nx+2 or
                header.count!=(header.ny+2)*header.pitch or header.steps!=header.trace_count or
                not np.isfinite([header.time,header.dt,header.minimum_dt]).all()):
            raise ValueError('Invalid reference checkpoint geometry/time/step trace')
        expected_size=ctypes.sizeof(ReferenceHeader)+8*(14*header.count+header.trace_count+1)
        if path.stat().st_size!=expected_size:raise ValueError('Incomplete reference checkpoint size')
        arrays=np.fromfile(file=stream,dtype=np.float64,count=14*header.count).reshape(14,header.ny+2,header.pitch)
        trace=np.fromfile(file=stream,dtype=np.float64,count=header.trace_count)
        footer=np.fromfile(file=stream,dtype=np.uint64,count=1)
        if len(footer)!=1 or footer[0]!=header.steps or stream.read(1):raise ValueError('Incomplete reference checkpoint')
    return header,arrays,trace


def validate_completed_simulation(*,case,depth,maximum,ledger,time_s):
    """Completion requires valid physical fields and normalised water balance."""
    if time_s!=case.duration or not ledger or ledger[-1]['seconds']!=case.duration:
        raise ValueError('Simulation final time or scalar coverage is incomplete')
    for field in (depth,maximum):
        if field.shape!=case.bed.shape or not np.isfinite(field).all() or np.any(field<0.):
            raise ValueError('Simulation has invalid/nonfinite/negative depth fields')
        if np.any(field[~case.active]!=0.):raise ValueError('Simulation has water on inactive cells')
    if np.any(maximum+1e-10<depth):raise ValueError('Maximum envelope is below final depth')
    initial=float(np.sum(case.initial[case.active])*case.spacing**2)
    for row in ledger:
        values=[row[name] for name in ('storage_m3','rain_m3','infiltration_m3',
            'outflow_m3','inflow_m3','residual_m3')]
        if not np.isfinite(values).all():raise ValueError('Simulation has nonfinite water accounting')
        if abs(row['residual_m3'])>1e-8*max(1.,initial+row['rain_m3']+row['inflow_m3']):
            raise ValueError('Simulation failed normalised conservation criterion')
    final=ledger[-1]
    stored=float(np.sum(depth[case.active])*case.spacing**2)
    balance=initial+final['rain_m3']+final['inflow_m3']-final['infiltration_m3']-final['outflow_m3']-stored
    tolerance=1e-8*max(1.,initial+final['rain_m3']+final['inflow_m3'])
    if abs(stored-final['storage_m3'])>tolerance or abs(balance)>tolerance:
        raise ValueError('Simulation final fields disagree with conservative water ledger')


def run_reference(*, case, directory, numerical, binary, signature, timeout, after_checkpoint=None):
    complete=read_completed(directory=directory,signature=signature,duration=case.duration)
    if complete is not None:return complete
    progress=read_progress(directory=directory,signature=signature)
    if progress is None:
        progress=dict(schema=1,signature=signature,time_s=0.,checkpoint=None,
            checkpoint_sha256=None,ledger=[],statistics=[],timings=dict(
                preparation_seconds=0.,process_seconds=0.,hydraulic_seconds=0.,restarts=-1))
        write_json(path=directory/'progress.json',value=progress)
    began=perf_counter()
    prepare_reference(case=case,directory=directory,numerical=numerical)
    prepared=perf_counter()-began
    identity=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PURAU_INPUT=str((directory/'inputs_f64.bin').resolve()),
        PURAU_CHECKPOINT_ID=identity,MATCHED_SOURCE_WET_DEPTH='-1',
        PURAU_FIELD_INTERVAL=str(case.field_interval or case.sample_interval))
    timings=progress['timings'];timings['preparation_seconds']+=prepared;timings['restarts']+=1
    ledger=progress['ledger'];statistics=progress['statistics']
    interval=case.checkpoint_interval or case.sample_interval
    # Existing native restart endpoints must coincide with mass diagnostics,
    # except the final time which the executable caps independently.
    targets=sample_targets(duration=case.duration,interval=interval)
    if np.any(np.abs(targets[:-1]/case.sample_interval-np.rint(targets[:-1]/case.sample_interval))>1e-10):
        raise ValueError('Reference checkpoint interval must align with scalar samples')
    for target in targets:
        if target<=progress['time_s']:continue
        previous_time=progress['time_s']
        if progress['checkpoint'] is None:environment.pop('PURAU_CHECKPOINT_READ',None)
        else:environment['PURAU_CHECKPOINT_READ']=str((directory/progress['checkpoint']).resolve())
        environment['PURAU_STOP_AT']=str(float(target))
        segment_profile=None
        segment_rows=None

        def save_state(*, path):
            nonlocal segment_profile,segment_rows
            environment['PURAU_CHECKPOINT_WRITE']=str(path.resolve())
            started=perf_counter()
            with (directory/f'run-{int(previous_time):06d}.log').open('w') as stream:
                subprocess.run(args=[str(binary.resolve()),'purau.par'],cwd=directory,env=environment,
                    stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=timeout)
            timings['process_seconds']+=perf_counter()-started
            segment_profile=json.loads((directory/'capacity_profile.json').read_text())
            header,_,_=read_reference_checkpoint(path=path)
            if header.identity.decode()!=identity or header.time!=target:
                raise ValueError('Reference checkpoint identity/final solver time mismatch')
            segment_rows=np.atleast_1d(np.genfromtxt(fname=directory/'purau_ledger.csv',delimiter=',',names=True))
            unique={float(row['seconds']):row for row in segment_rows if previous_time<row['seconds']<=target}
            ledger.extend({name:float(row[name]) for name in segment_rows.dtype.names} for row in unique.values())
            if not ledger or ledger[-1]['seconds']!=target:
                raise ValueError('Reference final scalar ledger time is incomplete')
            # Progress uses one statistics entry per scalar sample. Segment
            # runtime is assigned once; other entries contain no added cost.
            statistics.extend(dict(loop_seconds=0.) for _ in unique)
            statistics[-1]['loop_seconds']=segment_profile['simulation_seconds_excluding_diagnostics']
            statistics[-1]['profile']=segment_profile
            timings['hydraulic_seconds']+=segment_profile['simulation_seconds_excluding_diagnostics']

        progress=publish_progress(directory=directory,signature=signature,time_s=target,ledger=ledger,
            statistics=statistics,timings=timings,save_state=save_state,suffix='.bin')
        print(f'{case.name} reference {target/3600:.3f} h; solver {timings["hydraulic_seconds"]:.3f} s',flush=True)
        if after_checkpoint is not None:after_checkpoint(time_s=float(target))
    header,arrays,trace=read_reference_checkpoint(path=directory/progress['checkpoint'])
    if not np.isclose(header.time,case.duration,atol=1e-9,rtol=0.):raise ValueError('Incomplete reference run')
    profile=[row['profile'] for row in statistics if 'profile' in row][-1]
    maximum=np.maximum(case.initial,arrays[9,1:header.ny+1,1:header.nx+1][::-1][1:-1,1:-1])
    # U1 is the current FV1 state after every accepted update, including a
    # non-multiple final time without a regular full-field output.
    depth=arrays[0,1:header.ny+1,1:header.nx+1][::-1][1:-1,1:-1].copy()
    validate_completed_simulation(case=case,depth=depth,maximum=maximum,ledger=ledger,time_s=header.time)
    np.savez_compressed(file=directory/'fields.npz',depth=depth,maximum=maximum)
    record=dict(solver='LISFLOOD-FP FV1 GPU',steps=int(header.steps),time_s=header.time,
        preparation_seconds=timings['preparation_seconds'],process_seconds=timings['process_seconds'],
        end_to_end_seconds=timings['preparation_seconds']+timings['process_seconds'],
        hydraulic_seconds=timings['hydraulic_seconds'],profile=profile,restarts=timings['restarts'],
        peak_explicit_device_bytes=max(row['profile']['peak_explicit_cuda_allocation_bytes']
            for row in statistics if 'profile' in row),
        persistent_solver_array_bytes=profile.get('persistent_explicit_cuda_allocation_bytes'),
        dt_min_s=float(trace.min()),dt_mean_s=float(trace.mean()),dt_max_s=float(trace.max()),
        ledger=ledger,signature=signature,fields_sha256=file_hash(path=directory/'fields.npz'),
        physical_completion_verified=True)
    publish_record(directory=directory,record=record)
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


def reset_device_allocation_peak(*,device):
    """Reset the default-pool high watermark used by Warp's array allocator.

    CUDA attribute 8 is CU_MEMPOOL_ATTR_USED_MEM_HIGH, resettable with zero.
    Allocated bytes exclude reserved/context memory, as does the native FV1
    explicit-allocation counter; persistent capacity is reported separately.
    """
    resolved=wp.get_device(device)
    if not resolved.is_cuda:return 0
    gc.collect();wp.synchronize_device(device=resolved)
    driver=ctypes.CDLL('libcuda.so.1')
    driver.cuDeviceGetDefaultMemPool.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_int]
    driver.cuMemPoolSetAttribute.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
    pool=ctypes.c_void_p();zero=ctypes.c_uint64(0)
    with wp.ScopedDevice(device=resolved):
        error=driver.cuDeviceGetDefaultMemPool(ctypes.byref(pool),resolved.ordinal)
        if error:raise RuntimeError(f'Cannot query CUDA memory pool: {error}')
        error=driver.cuMemPoolSetAttribute(pool,8,ctypes.byref(zero))
        if error:raise RuntimeError(f'Cannot reset CUDA allocation high watermark: {error}')
    return wp.get_mempool_used_mem_current(device=resolved)


def run_floodwarp(*, case, method, directory, device, numerical, signature, graph,
                  after_checkpoint=None):
    complete=read_completed(directory=directory,signature=signature,duration=case.duration)
    if complete is not None:return complete
    progress=read_progress(directory=directory,signature=signature)
    directory.mkdir(parents=True,exist_ok=True)
    if progress is None:
        progress=dict(schema=1,signature=signature,time_s=0.,checkpoint=None,
            checkpoint_sha256=None,ledger=[],statistics=[],timings=dict(
                preparation_seconds=0.,process_seconds=0.,hydraulic_seconds=0.,restarts=-1))
        write_json(path=directory/'progress.json',value=progress)
    baseline_allocation=reset_device_allocation_peak(device=device)
    wp.synchronize_device(device=device)
    compilation_started=perf_counter()
    wp.load_module(module='floodwarp',recursive=True,device=device)
    wp.synchronize_device(device=device)
    compilation_seconds=perf_counter()-compilation_started
    began=perf_counter()
    model=build_floodwarp(case=case,method=method,device=device,numerical=numerical)
    solver=model.solver
    if progress is not None and progress['checkpoint'] is not None:
        load_checkpoint(model=model,path=directory/progress['checkpoint'])
        restored=solver.time if method=='fv1' else solver.simulation_time
        if restored!=progress['time_s']:
            raise ValueError('Restored benchmark solver time differs from progress manifest')
    wp.synchronize_device(device=device)
    prepared=perf_counter()-began
    samples=sample_targets(duration=case.duration,interval=case.sample_interval)
    ledger=[] if progress is None else progress['ledger']
    statistics=[] if progress is None else progress['statistics']
    timings=dict(preparation_seconds=0.,process_seconds=0.,hydraulic_seconds=0.,restarts=-1) if progress is None else progress['timings']
    timings['preparation_seconds']+=prepared
    timings['compilation_seconds']=timings.get('compilation_seconds',0.)+compilation_seconds
    timings['restarts']+=1
    hydraulic=timings['hydraulic_seconds'];started=perf_counter()

    def save_state(*, path):
        checkpoint_started=perf_counter()
        save_checkpoint(model=model,path=path)
        timings['process_seconds']+=perf_counter()-checkpoint_started
        resolved=wp.get_device(device)
        if resolved.is_cuda:
            timings['peak_explicit_device_bytes']=max(timings.get('peak_explicit_device_bytes',0),
                wp.get_mempool_used_mem_high(device=resolved)-baseline_allocation)

    for target in samples:
        current=solver.time if method=='fv1' else solver.simulation_time
        if target<=current:continue
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
        timings['hydraulic_seconds']=hydraulic
        timings['process_seconds']+=perf_counter()-started
        interval=case.checkpoint_interval or case.sample_interval
        next_target=min(case.duration,(np.floor((target+1e-10)/interval)+1.)*interval)
        checkpoint_due=target==case.duration or next_target-target>=interval-case.sample_interval*1e-9
        if checkpoint_due:
            publish_progress(directory=directory,signature=signature,time_s=target,ledger=ledger,
                statistics=statistics,timings=timings,save_state=save_state,suffix='.npz')
            if after_checkpoint is not None:after_checkpoint(time_s=float(target))
        started=perf_counter()
    wp.synchronize_device(device=device)
    process=timings['process_seconds']+perf_counter()-started
    output_started=perf_counter()
    final_time=solver.time if method=='fv1' else solver.simulation_time
    if final_time!=case.duration:
        raise ValueError('Benchmark solver did not reach requested final time')
    depth=solver.get_water_depth().copy()
    maximum=solver.get_max_water_depth().copy() if hasattr(solver,'get_max_water_depth') and getattr(solver,'track_max_water_depth',True) else depth.copy()
    maximum=np.maximum(case.initial,maximum)
    validate_completed_simulation(case=case,depth=depth,maximum=maximum,ledger=ledger,time_s=final_time)
    np.savez_compressed(file=directory/'fields.npz',depth=depth,maximum=maximum)
    process+=perf_counter()-output_started
    steps=sum(int(row['steps']) for row in statistics)
    resolved=wp.get_device(device)
    peak_allocation=(wp.get_mempool_used_mem_high(device=resolved)-baseline_allocation) if resolved.is_cuda else None
    if peak_allocation is not None:peak_allocation=max(peak_allocation,timings.get('peak_explicit_device_bytes',0))
    record=dict(solver='FloodWarp '+method.upper(),steps=steps,time_s=case.duration,
        preparation_seconds=timings['preparation_seconds'],process_seconds=process,
        end_to_end_seconds=timings['preparation_seconds']+process+timings['compilation_seconds'],
        compilation_seconds=timings['compilation_seconds'],
        graph_capture_seconds=sum(row['graph_capture_seconds'] for row in statistics),
        diagnostics_and_checkpoint_seconds=process-sum(row['loop_seconds']+
            row['graph_capture_seconds']+row.get('compilation_seconds',0.) for row in statistics),
        hydraulic_seconds=hydraulic,persistent_solver_array_bytes=device_array_bytes(solver=solver),
        peak_explicit_device_bytes=peak_allocation,restarts=timings['restarts'],
        allocation_measurement='CUDA pool allocation high watermark above pre-run baseline, including checkpoint temporaries' if resolved.is_cuda else 'CPU arrays',
        dt_min_s=min(row['dt_min_s'] for row in statistics),dt_mean_s=case.duration/steps,
        dt_max_s=max(row['dt_max_s'] for row in statistics),ledger=ledger,signature=signature,
        graph=graph,statistics=statistics,fields_sha256=file_hash(path=directory/'fields.npz'),
        physical_completion_verified=True)
    publish_record(directory=directory,record=record)
    return record


def physical_metrics(*, candidate, reference, active, thresholds, indifference_m, candidate_rows, reference_rows):
    error=candidate[active]-reference[active]
    reference_norm=float(np.linalg.norm(reference[active]))
    wet=active&(np.maximum(candidate,reference)>=min(thresholds))
    wet_error=(candidate-reference)[wet]
    interior=binary_erosion(input=active,iterations=3,border_value=0)
    boundary=active & ~interior
    field_error=candidate-reference
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
        depth_bias_m=float(np.mean(error)),
        depth_relative_l2=float(np.linalg.norm(error)/reference_norm) if reference_norm>0. else None,
        absolute_depth_error_percentiles_m=dict(zip(('50','90','95','99'),
            map(float,np.percentile(np.abs(error),[50.,90.,95.,99.])))),
        wet_union_cells=int(np.count_nonzero(wet)),
        wet_union_depth_rmse_m=float(np.sqrt(np.mean(wet_error**2))) if wet_error.size else None,
        wet_union_absolute_depth_error_percentiles_m=dict(zip(('50','90','95','99'),
            map(float,np.percentile(np.abs(wet_error),[50.,90.,95.,99.])))) if wet_error.size else None,
        meaningful_depth_coverage=bool(np.any(np.maximum(candidate,reference)[active]>=min(thresholds))),
        interior_rmse_m=float(np.sqrt(np.mean(field_error[interior]**2))) if np.any(interior) else None,
        boundary_band_rmse_m=float(np.sqrt(np.mean(field_error[boundary]**2))) if np.any(boundary) else None,
        inundation=extent,hydrograph_rmse_m3_s=float(np.sqrt(np.mean((qa-qb)**2))),
        hydrograph_coverage=bool(np.max(np.abs(qb))>0.),
        hydrograph_nrmse=float(np.sqrt(np.mean((qa-qb)**2))/np.max(np.abs(qb))) if np.max(np.abs(qb))>0. else None,
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
        preserve_sources(directory=output/case.name/'source_snapshot',signature=signature)
        # A short independent warm-up is retained separately from measured repeats.
        for method in ('reference','li','fv1'):
            warm=replace(case,duration=min(case.duration,case.sample_interval),sample_interval=min(case.duration,case.sample_interval))
            warm_signature=case_signature(case=warm,numerical=numerical,binary=binary)
            warm_path=output/case.name/(method+'_warmup')
            if method=='reference':run_reference(case=warm,directory=warm_path,numerical=numerical,binary=binary,signature=warm_signature,timeout=timeout)
            else:run_floodwarp(case=warm,method=method,directory=warm_path,device=device,numerical=numerical,signature=warm_signature,graph=None)
        records={}
        for method in ('reference','li','fv1'):
            rows=[]
            for repeat in range(repeats):
                directory=output/case.name/f'{method}_{repeat}'
                path=directory/'record.json'
                if method=='reference':record=run_reference(case=case,directory=directory,numerical=numerical,binary=binary,signature=signature,timeout=timeout)
                else:record=run_floodwarp(case=case,method=method,directory=directory,device=device,numerical=numerical,signature=signature,graph=None)
                rows.append(record)
                print(case.name,method,repeat,record['hydraulic_seconds'],flush=True)
            records[method]=dict(hydraulic_median_s=float(np.median([row['hydraulic_seconds'] for row in rows])),
                hydraulic_range_s=[min(row['hydraulic_seconds'] for row in rows),max(row['hydraulic_seconds'] for row in rows)],
                end_to_end_median_s=float(np.median([row['end_to_end_seconds'] for row in rows])),
                peak_explicit_device_bytes=max((row['peak_explicit_device_bytes'] for row in rows
                    if row['peak_explicit_device_bytes'] is not None),default=None),
                persistent_solver_array_bytes=max((row.get('persistent_solver_array_bytes') or 0 for row in rows),default=0),
                steps=rows[0]['steps'],
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
    binary=Path(os.environ.get('FLOODWARP_REFERENCE_BINARY','outputs/qualified_reference/build/lisflood'))
    event_directory=reference_root/'purau_lisflood_fp64_20261023/event_5m_fp64_v3'
    output=Path(os.environ.get('FLOODWARP_BENCHMARK_OUTPUT','outputs/qualification_benchmark'))
    device='cuda:0'
    scope=os.environ.get('FLOODWARP_BENCHMARK_SCOPE','small')
    repeats=1 if scope in ('long','full72') else 3
    timeout=14400.
    numerical=dict(cfl=.4,maximum_dt_s=1.,minimum_dt_s=.000001,gravity_m_s2=9.80665,
        dry_depth_m=.0001,wall_elevation_m=1000000.,fv1_stage_momentum='discharge_extrapolation')
    thresholds=(.01,.10)
    indifference_m=.00001
    event=load_event(directory=event_directory,duration=3600.,sample_interval=60.)
    crop=crop_event(event=event,size=40,duration=3600.,sample_interval=60.)
    cases=[analytical_case(name='lake',shape=(8,40),spacing=5.,duration=60.,sample_interval=1.,
        roughness=.035,depth=.2,rain=0.,infiltration=0.),
        analytical_case(name='dam_break',shape=(4,80),spacing=5.,duration=60.,sample_interval=1.,
        roughness=.035,depth=.2,rain=0.,infiltration=0.),
        analytical_case(name='rain_infiltration',shape=(16,16),spacing=5.,duration=60.,sample_interval=1.,
        roughness=.035,depth=0.,rain=.00002,infiltration=.00004),
        analytical_case(name='boundary_outflow',shape=(16,16),spacing=5.,duration=60.,sample_interval=1.,
        roughness=.035,depth=.2,rain=0.,infiltration=0.),crop]
    if scope=='full':cases=[event]
    if scope=='long':cases=[replace(crop,name='purau_coast_crop_72h',duration=259200.)]
    if scope=='full72':cases=[replace(event,name='purau_event_72h',duration=259200.)]
    if scope=='refined':cases=[refine_case(case=replace(crop,duration=129600.),factor=2)]
    if scope=='grid_control':cases=[replace(crop,name='purau_coast_crop_grid_control',duration=129600.)]
    if scope=='cfl':
        numerical=dict(numerical,cfl=.2)
        cases=[replace(crop,name='purau_coast_crop_cfl02',duration=129600.)]
    cases=[replace(case,checkpoint_interval=min(3600.,case.duration),field_interval=min(3600.,case.duration)) for case in cases]
    summaries=run_benchmarks(cases=cases,numerical=numerical,binary=binary,output=output,device=device,
        repeats=repeats,timeout=timeout,thresholds=thresholds,indifference_m=indifference_m)
