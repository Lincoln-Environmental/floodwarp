from floodwarp.experimental import warn_experimental_sgs
"""Component-aware production SGS with conservative coarse-grid interfaces.

The independent component state is authoritative. Coarse arrays are conservative
output/forcing interfaces, never a route for mixing disconnected storage.
"""
import numpy as np
import warp as wp
import inspect
import time
from pathlib import Path

from floodwarp.config import NP_FLOAT, WP_FLOAT, PRECISION_MODE
from floodwarp.local_inertial_solver import LocalInertialSolver
from floodwarp.local_inertial_kernels import (
    DL_SIZE, device_loop_timestep_cap_kernel, device_loop_stage_fill_kernel,
    device_loop_accumulate_face_volume_qx_kernel, device_loop_accumulate_face_volume_qy_kernel,
)
from floodwarp.enhanced_geometry import (
    build_component_geometry, pack_coefficients, component_volumes,
    StageRangeError, NumericalFailure, storage_stage, native_input_identity,
    hydraulic_face_masks,face_topology_identity,pack_geometry,
)
from floodwarp.enhanced_state import (
    coarse_storage_to_components, coarse_component_diagnostics, coarse_face_connections,
    structure_component_endpoints,
)
from floodwarp.structure_kernels import calculate_structure_requests
from floodwarp import enhanced_kernels as kernels


class EnhancedSGSSolver(LocalInertialSolver):
    """Production SGS with independent component volumes and local momenta.

    A supplied geometry is reusable. An explicit maximum_stage_m retains the
    bounded validation reference; otherwise a fully wet reciprocal-head tail
    covers all higher stages without runtime native reductions.
    LocalInertialSolver selects this class for enhanced and legacy subgrid modes.
    """

    # The component storage-change bound scales as 1/source rate, whereas
    # conventional source-predicted wave CFL has a square-root dependence.
    # Closing the SAME interval-mean-rate fixed point to the shared 1e-12
    # tolerance can therefore need more iterations. This is a bounded host
    # forcing negotiation, not a changed timestep or a momentum correction.
    source_forcing_max_attempts=64
    # Component continuity commits every base tick, so both native and coarse
    # maxima can be accumulated without reading delayed LTS volume registers.
    _lts_tracks_max_water_depth=True

    def __init__(self,*args,**options):
        """Accept the existing production constructor, including positional inputs.

        Binding uses the production signature itself, so forcing/infiltration
        options retain one definition. Additional hydraulic table controls are
        those explicitly declared by _initialize; unknown options still fail.
        """
        base_signature=inspect.signature(LocalInertialSolver.__init__)
        enhanced_names=set(inspect.signature(self._initialize).parameters)-set(base_signature.parameters)
        enhanced_names.discard('production_options')
        enhanced={key:options.pop(key) for key in tuple(options) if key in enhanced_names}
        bound=base_signature.bind(self,*args,**options);bound.apply_defaults()
        parameters=dict(bound.arguments);parameters.pop('self')
        mode=parameters.pop('hydraulic_geometry_mode')
        if mode not in ('coarse','subgrid','enhanced'):
            raise ValueError('EnhancedSGSSolver cannot initialize a different hydraulic method')
        if parameters['subgrid_bed_elevation'] is None or parameters['subgrid_resolution_m'] is None:
            raise ValueError('Enhanced SGS requires subgrid_bed_elevation and subgrid_resolution_m')
        warn_experimental_sgs(formulation="enhanced component-aware")
        self._initialize(**parameters,**enhanced)

    def _initialize(self, *, ny, nx, dx, dy, bed_elevation, manning_n, active_mask,
                 subgrid_bed_elevation, subgrid_resolution_m, device='cpu',
                 subgrid_manning_n=None, enhanced_geometry=None, maximum_stage_m=None,
                 coefficient_tolerance=.005, coefficient_encoding='scalar',
                 momentum_formulation='transport_star', momentum_transport='discharge',momentum_integration='legacy',
                 coefficient_fraction=.2, interpolation_tolerance=.02, component_timestep='storage',
                 opening_depth_m=1e-5, maximum_ports_per_cell=None,
                 maximum_rows_per_port=None, preprocessing_workers=1,
                 enhanced_geometry_cache_dir=None, native_active_mask=None,
                 native_north_outlets=None, native_partition=None, **production_options):
        if PRECISION_MODE=='mixed':
            raise ValueError('Enhanced and traditional SGS require fp64 or fp32 precision')
        if production_options.get('free_outflow_closure') == 'local_or_prescribed_slope':
            raise ValueError('Enhanced SGS does not implement local_or_prescribed_slope FREE boundaries')
        if momentum_transport not in ('discharge','kinetic'):
            raise ValueError('momentum_transport must be discharge or kinetic')
        if momentum_integration not in ('legacy','analytic'):
            raise ValueError('momentum_integration must be legacy or analytic')
        self._exact_momentum=int(momentum_integration=='analytic')
        if not np.isfinite(coefficient_fraction) or not 0<coefficient_fraction<1:
            raise ValueError('coefficient_fraction must lie strictly between zero and one')
        if component_timestep not in ('storage','wave','provisional'):
            raise ValueError('component_timestep must be storage, wave or provisional')
        if component_timestep!='storage' and momentum_transport!='discharge':
            raise ValueError('Experimental wave controller requires discharge continuation')
        self._component_timestep=component_timestep
        # Use the established native input contract before allocating state.
        from floodwarp.subgrid_geometry import build_subgrid_geometry
        contract=build_subgrid_geometry(ny=ny,nx=nx,dx=dx,dy=dy,
            subgrid_bed_elevation=subgrid_bed_elevation,subgrid_resolution_m=subgrid_resolution_m)
        native=np.asarray(subgrid_bed_elevation,dtype=np.float64).copy()
        factor=contract.factor
        # Every port owns at least one native contact. This combinatorial
        # bound admits all native components instead of imposing a fixed
        # two-/four-/512-port physical support restriction. The row default
        # is solely the signed-32-bit GPU indexing capacity, not a stage cap.
        if maximum_ports_per_cell is None:maximum_ports_per_cell=4*factor*factor
        if maximum_rows_per_port is None:maximum_rows_per_port=int(np.iinfo(np.int32).max)
        unbounded_stage=maximum_stage_m is None
        if unbounded_stage:maximum_stage_m=float(native.max()+max(dx,dy))
        if production_options.get('validation_profile') not in (None,'default'):
            raise ValueError('Traditional and enhanced SGS support only the default validation profile')
        production_options['validation_profile'] = 'default'
        self._enhanced_ready=False
        super().__init__(ny=ny,nx=nx,dx=dx,dy=dy,bed_elevation=bed_elevation,
            manning_n=manning_n,active_mask=active_mask,device=device,
            hydraulic_geometry_mode='coarse',**production_options)
        self.hydraulic_geometry_mode='enhanced'
        self._native_bed=native
        self._native_resolution_m=float(subgrid_resolution_m)
        self._subgrid_factor=factor
        native_n=(np.asarray(manning_n).repeat(factor,0).repeat(factor,1)
                  if subgrid_manning_n is None else np.asarray(subgrid_manning_n))
        if subgrid_manning_n is None:
            native_n=np.where(np.asarray(active_mask,dtype=bool).repeat(factor,0).repeat(factor,1),native_n,1.)
        self._native_manning=native_n.copy()
        self._enhanced_native_options=dict(native_active_mask=native_active_mask,
            native_north_outlets=native_north_outlets,native_partition=native_partition)
        self._enhanced_build_parameters=dict(maximum_stage_m=maximum_stage_m,
            formulation=momentum_formulation,interpolation_tolerance=interpolation_tolerance,
            maximum_ports_per_cell=maximum_ports_per_cell,maximum_rows_per_port=maximum_rows_per_port,
            opening_depth_m=opening_depth_m,contact_grouping='connected_runs',inertia_weighting='uniform',
            unbounded_stage=unbounded_stage)
        self._enhanced_encoding=coefficient_encoding
        self._enhanced_coefficient_tolerance=coefficient_tolerance
        self._enhanced_preprocessing_workers=preprocessing_workers
        self._enhanced_geometry_cache_dir=enhanced_geometry_cache_dir
        self._enhanced_preparation=dict(kind='supplied_geometry',cache_hit=False,cache_load_s=0.,cache_save_s=0.,
            native_preprocessing_s=0.,compression_s=0.)
        face_x=self.boundary_type_qx.numpy()!=1
        face_y=self.boundary_type_qy.numpy()!=1
        if enhanced_geometry is None:
            geometry=self._build_boundary_geometry(face_open_x=face_x,face_open_y=face_y)
        else:
            geometry=enhanced_geometry
            identity=native_input_identity(bed_m=native,factor=factor,resolution_m=subgrid_resolution_m,
                manning_n=native_n,active_mask=active_mask,**self._enhanced_native_options)
            if geometry.diagnostics.get('native_input_identity')!=identity:
                raise ValueError('Reused enhanced geometry is not bound to this terrain, mask and Manning field')
            if geometry.shape!=(ny,nx) or geometry.factor!=factor or geometry.resolution!=subgrid_resolution_m:
                raise ValueError('reused enhanced geometry does not match model dimensions')
            expected=(np.asarray(active_mask,dtype=bool).repeat(factor,0).repeat(factor,1)
                if native_active_mask is None else np.asarray(native_active_mask,dtype=bool))
            if not np.array_equal(geometry.pixel_owner>=0,expected):
                raise ValueError('reused enhanced geometry does not match active mask')
            self._enhanced_build_parameters=dict(geometry.diagnostics.get('build_parameters',
                self._enhanced_build_parameters))
            self._enhanced_build_parameters['maximum_stage_m']=geometry.tail_start if geometry.tail_rows is not None else geometry.maximum_stage
            self._enhanced_build_parameters['formulation']=geometry.formulation
            if geometry.coefficient_ranges is not None:
                self._enhanced_encoding='scalar'
                self._enhanced_coefficient_tolerance=geometry.diagnostics['coefficient_compression_tolerance']
            elif geometry.coefficient_values is not None:self._enhanced_encoding='packed_intervals'
            else:self._enhanced_encoding='uncompressed'
            expected=face_topology_identity(face_open_x=face_x,face_open_y=face_y)
            if geometry.diagnostics.get('face_topology_identity')!=expected:
                geometry=self._build_boundary_geometry(face_open_x=face_x,face_open_y=face_y)

        self._enhanced_geometry=geometry
        self._coefficient_fraction=float(coefficient_fraction)
        self._kinetic_transport=int(momentum_transport=='kinetic')
        self._component_count=len(geometry.state_cell)
        self._connection_count=len(geometry.endpoints)
        # Direct attributes are covered by complete production snapshots.
        self._upload_component_geometry(geometry=geometry)
        for name in ('volume','soil','cap'):
            setattr(self,'_ec_'+name,wp.zeros(self._component_count,dtype=WP_FLOAT,device=self.device))
        for name in ('eta','area','maximum_eta','alpha','alpha_free','local_dt'):
            setattr(self,'_ec_'+name,wp.zeros(self._component_count,dtype=wp.float64,device=self.device))
        self._ec_control=wp.zeros(DL_SIZE,dtype=wp.float64,device=self.device)
        self._ec_status=wp.zeros(1,dtype=int,device=self.device)
        self._ec_boundary_closure_loss=wp.zeros(1,dtype=wp.float64,device=self.device)
        self._ec_applied=wp.zeros((self._component_count,3),dtype=WP_FLOAT,device=self.device)
        self._defer_component_sources=False
        self._defer_component_outputs=False
        self._ec_source_ledger=wp.zeros((self._component_count if self.use_reduced_mass_accounting else 1,4),
            dtype=wp.float64,device=self.device)
        self._ec_source_sums=wp.zeros((1,4),dtype=wp.float64,device=self.device)
        self._track_timestep_limits=False
        self._ec_timestep_limits=wp.zeros((1,6),dtype=wp.float64,device=self.device)
        self._install_connection_layout(geometry=geometry)
        self._init_component_structures()
        self._enhanced_ready=True
        self.set_water_depth(depth=np.zeros((ny,nx)))
        if self._component_timestep=='provisional':self._ensure_provisional_buffers()

    def _build_boundary_geometry(self, *, face_open_x, face_open_y):
        cache_directory=self._enhanced_geometry_cache_dir
        if cache_directory is not None:
            from floodwarp.enhanced_cache import geometry_cache_key,load_geometry_cache,save_geometry_cache
            key=geometry_cache_key(native_identity=native_input_identity(bed_m=self._native_bed,
                    factor=self._subgrid_factor,resolution_m=self._native_resolution_m,
                    manning_n=self._native_manning,active_mask=self._active_mask_host,**self._enhanced_native_options),
                face_identity=face_topology_identity(face_open_x=face_open_x,face_open_y=face_open_y),
                parameters=self._enhanced_build_parameters,encoding=self._enhanced_encoding,
                tolerance=self._enhanced_coefficient_tolerance)
            cached=load_geometry_cache(directory=Path(cache_directory),key=key)
            if cached is not None:
                geometry,load_s=cached
                self._enhanced_preparation.update(kind='validated_cache_load',cache_hit=True,
                    cache_load_s=load_s,native_preprocessing_s=0.,compression_s=0.)
                return geometry
        started=time.perf_counter()
        geometry=build_component_geometry(bed_m=self._native_bed,factor=self._subgrid_factor,
            resolution_m=self._native_resolution_m,manning_n=self._native_manning,
            active_mask=self._active_mask_host,open_sides=(0,1,2,3),face_open_x=face_open_x,
            face_open_y=face_open_y,preprocessing_workers=self._enhanced_preprocessing_workers,
            **self._enhanced_build_parameters,**self._enhanced_native_options)
        preprocessing_s=time.perf_counter()-started
        started=time.perf_counter()
        if self._enhanced_encoding=='scalar':
            geometry=pack_coefficients(geometry=geometry,relative_tolerance=self._enhanced_coefficient_tolerance,
                preprocessing_workers=self._enhanced_preprocessing_workers)
        elif self._enhanced_encoding=='packed_intervals':geometry=pack_geometry(geometry=geometry)
        elif self._enhanced_encoding!='uncompressed':
            raise ValueError('coefficient_encoding must be scalar or uncompressed')
        self._enhanced_preparation.update(kind='cold_build',cache_hit=False,native_preprocessing_s=preprocessing_s,
            compression_s=time.perf_counter()-started)
        if cache_directory is not None:
            started=time.perf_counter()
            save_geometry_cache(geometry=geometry,directory=Path(cache_directory),key=key)
            self._enhanced_preparation['cache_save_s']=time.perf_counter()-started
        return geometry

    def _init_component_structures(self):
        if not self.structures:return
        endpoints=structure_component_endpoints(geometry=self._enhanced_geometry,
            native_bed=self._native_bed,records=self.structures)
        components=np.unique(endpoints);slots=np.searchsorted(components,endpoints)
        members=[[] for _ in components]
        for s,(a,b) in enumerate(slots):
            members[a].append(s+1);members[b].append(-s-1)
        arrays=dict(components=components,
            offsets=np.r_[0,np.cumsum([len(m) for m in members])].astype(np.int32),
            adjacency=np.asarray([s for m in members for s in m],dtype=np.int32))
        for name,value in arrays.items():
            setattr(self,'_ec_structure_'+name,wp.array(value,dtype=int,device=self.device))
        self._structure_slot_a=wp.array(slots[:,0],dtype=int,device=self.device)
        self._structure_slot_b=wp.array(slots[:,1],dtype=int,device=self.device)
        for name in ('stage','storage'):
            setattr(self,'_structure_endpoint_'+name,wp.zeros(len(components),dtype=WP_FLOAT,device=self.device))
        self._structure_donor_requested=wp.zeros(len(components),dtype=WP_FLOAT,device=self.device)
        self._ec_structure_alpha=wp.zeros(len(components),dtype=wp.float64,device=self.device)

    def _apply_structure_transfers(self, *, dt):
        if not self.structures:return
        self._component_stages()
        components=self._ec_structure_components
        wp.launch(kernel=kernels.structure_endpoints,dim=len(components),
            inputs=[components,self._ec_volume,self._ec_eta,self._structure_endpoint_stage,
                self._structure_endpoint_storage,self._structure_donor_requested],device=self.device)
        params=self._structure_params
        wp.launch(kernel=calculate_structure_requests,dim=self._structure_count,
            inputs=[self._structure_endpoint_stage,self._structure_endpoint_storage,
                components,self._structure_slot_a,self._structure_slot_b,self._structure_kind,
                self._structure_one_way,params['width'],params['height'],params['invert_a'],params['invert_b'],
                params['length'],params['manning'],params['entrance'],params['exit'],params['crest'],
                params['weir_length'],params['weir_factor'],params['max_q'],WP_FLOAT(dt),
                self._structure_donor_requested,self._structure_request,self._structure_q,self._structure_head],device=self.device)
        safety=1.000000000000002 if NP_FLOAT==np.float64 else 1.000001
        wp.launch(kernel=kernels.structure_demand,dim=len(components),
            inputs=[components,self._ec_volume,self._ec_structure_offsets,self._ec_structure_adjacency,
                self._structure_request,safety,self._ec_structure_alpha],device=self.device)
        wp.launch(kernel=kernels.structure_accept,dim=self._structure_count,
            inputs=[self._structure_slot_a,self._structure_slot_b,self._structure_request,self._ec_structure_alpha,
                self._structure_positive,self._structure_negative,self._structure_net,self._structure_max_q,
                self._structure_q,self._structure_limiter_count,self._structure_actual],device=self.device)
        wp.launch(kernel=kernels.structure_continuity,dim=len(components),
            inputs=[components,self._ec_volume,self._ec_structure_offsets,self._ec_structure_adjacency,
                self._structure_actual,self._ec_status],device=self.device)
        self._component_stages(track=self.track_max_water_depth);self._aggregate_components()

    def _upload_component_geometry(self, *, geometry):
        if hasattr(self,'_embedded_coefficients_valid'):self._embedded_coefficients_valid.zero_()
        for name in ('coefficient_values','coefficient_ranges','coefficient_stage_index','tail_rows','tail_offsets','tail_halves'):
            if getattr(geometry,name) is None and hasattr(self,'_ec_'+name):delattr(self,'_ec_'+name)
        for name,value in vars(geometry).items():
            if not isinstance(value,np.ndarray) or name in ('pixel_owner','port_sill','connection_keys'):continue
            dtype=wp.float64
            if value.dtype.kind in 'iu':dtype=wp.uint16 if value.dtype==np.uint16 else wp.int32
            elif name=='coefficient_values' and value.dtype==np.float32:dtype=wp.float32
            setattr(self,'_ec_'+name,wp.array(value,dtype=dtype,device=self.device))

        # Four validated interval hints per connection: left/right D and R.
        # Constant and non-scalar encodings need no per-connection cache.
        ranges=geometry.coefficient_ranges
        variable=ranges is not None and (np.any(ranges[:,1]-ranges[:,0]>1) or np.any(ranges[:,3]-ranges[:,2]>1))
        count=max(1,len(geometry.endpoints)) if variable else 1
        self._ec_scalar_interval=wp.full((count,4),-1,dtype=int,device=self.device)
        self._ec_rain_bound=wp.zeros(1,dtype=wp.float64,device=self.device)

    def _install_connection_layout(self, *, geometry):
        self._connection_count=len(geometry.endpoints)
        for name in ('q','proposed'):
            setattr(self,'_ec_'+name,wp.zeros(self._connection_count,dtype=WP_FLOAT,device=self.device))
        for name in ('d','r','forcing','previous_d'):
            setattr(self,'_ec_'+name,wp.zeros(self._connection_count,dtype=wp.float64,device=self.device))
        self._ec_heads=wp.array(geometry.sill,dtype=wp.float64,device=self.device)
        self._ec_kind=wp.zeros(self._connection_count,dtype=int,device=self.device)
        self._ec_has_boundary_override=0
        self._ec_boundary_override=wp.zeros(1,dtype=int,device=self.device)
        self._ec_boundary_head_values=wp.zeros(1,dtype=wp.float64,device=self.device)
        face,sign,offsets,members=coarse_face_connections(geometry=geometry)
        for name,value in (('face',face),('sign',sign),('face_offsets',offsets),('face_members',members)):
            setattr(self,'_ec_'+name,wp.array(value,dtype=int,device=self.device))

    def _replace_boundary_geometry(self, *, geometry):
        """Configuration-time port changes preserve component storage and live Q.

        Keys identify the original native face runs exactly. New contacts start
        at Q=0; closing contacts cease exchange and their kinetic metric is
        recorded. No native graph work is performed during hydraulic steps or
        time-varying stage updates.
        """
        old=self._enhanced_geometry
        for name in ('pixel_owner','state_cell','source_area','storage_offsets','storage_rows'):
            if not np.array_equal(getattr(old,name),getattr(geometry,name)):
                raise RuntimeError('Boundary change altered independent storage; no state changed')
        if old.connection_keys is None or geometry.connection_keys is None:
            raise ValueError('Boundary topology changes require native connection identities')
        old_keys={tuple(key):i for i,key in enumerate(old.connection_keys)}
        if len(old_keys)!=len(old.endpoints) or len(set(map(tuple,geometry.connection_keys)))!=len(geometry.endpoints):
            raise RuntimeError('Nonunique native connection identity; no state changed')
        old_q=self._ec_q.numpy();old_d=self._ec_previous_d.numpy()
        old_heads=self.get_component_boundary_stages()
        new_heads=np.full(len(geometry.endpoints),np.nan)
        discharge=np.zeros(len(geometry.endpoints),dtype=NP_FLOAT);previous=np.zeros(len(discharge))
        retained=np.zeros(len(old_q),dtype=bool)
        for f,key in enumerate(geometry.connection_keys):
            previous_f=old_keys.get(tuple(key))
            if previous_f is not None:
                discharge[f]=old_q[previous_f];previous[f]=old_d[previous_f];retained[previous_f]=True
                new_heads[f]=old_heads[previous_f]
        removed=float(.5*np.sum(old_d[~retained]*old_q[~retained].astype(np.float64)**2))
        self._upload_component_geometry(geometry=geometry)
        self._install_connection_layout(geometry=geometry)
        self._enhanced_geometry=geometry
        wp.copy(self._ec_q,wp.array(discharge,dtype=WP_FLOAT,device=self.device))
        wp.copy(self._ec_previous_d,wp.array(previous,dtype=wp.float64,device=self.device))
        self._assign_component_boundary_stages(stage_m=new_heads)
        ledger=self._ec_boundary_closure_loss.numpy();ledger[0]+=removed
        wp.copy(self._ec_boundary_closure_loss,wp.array(ledger,dtype=wp.float64,device=self.device))
        self._aggregate_components()

    def _prepare_boundary_change(self, *, type_x, type_y):
        face_x,face_y=hydraulic_face_masks(shape=(self.ny,self.nx),open_sides=(0,1,2,3),
            face_open_x=type_x!=1,face_open_y=type_y!=1)
        identity=face_topology_identity(face_open_x=face_x,face_open_y=face_y)
        if identity==self._enhanced_geometry.diagnostics.get('face_topology_identity'):return None
        return self._build_boundary_geometry(face_open_x=face_x,face_open_y=face_y)

    def set_boundary_type_qx(self, boundary_type):
        self._validate_boundary_type_qx(boundary_type=boundary_type)
        geometry=self._prepare_boundary_change(type_x=boundary_type,type_y=self.boundary_type_qy.numpy())
        if geometry is not None:self._replace_boundary_geometry(geometry=geometry)
        super().set_boundary_type_qx(boundary_type=boundary_type)

    def set_boundary_type_qy(self, boundary_type):
        self._validate_boundary_type_qy(boundary_type=boundary_type)
        geometry=self._prepare_boundary_change(type_x=self.boundary_type_qx.numpy(),type_y=boundary_type)
        if geometry is not None:self._replace_boundary_geometry(geometry=geometry)
        super().set_boundary_type_qy(boundary_type=boundary_type)

    def _assign_component_boundary_stages(self, *, stage_m, native_contacts=False):
        mask=np.isfinite(stage_m)
        self._ec_has_boundary_override=int(np.any(mask))
        if self._ec_has_boundary_override:
            self._ec_boundary_override=wp.array(mask.astype(np.int32)*(2 if native_contacts else 1),dtype=int,device=self.device)
            self._ec_boundary_head_values=wp.array(np.where(mask,stage_m,0.),dtype=wp.float64,device=self.device)
        else:
            self._ec_boundary_override=wp.zeros(1,dtype=int,device=self.device)
            self._ec_boundary_head_values=wp.zeros(1,dtype=wp.float64,device=self.device)
        self._lts_subcycle_graph=None

    def set_component_boundary_stages(self, *, stage_m, native_contacts=False):
        """Optional independent stage per existing native external contact.

        NaN entries inherit the ordinary coarse face stage. None clears all
        overrides. Values apply only to specified-stage boundaries, and cannot
        open a closed face, create a native contact or force an internal link.
        The two extra vectors are allocated only when this interface is used.
        """
        values=np.full(self._connection_count,np.nan) if stage_m is None else np.asarray(stage_m,dtype=float)
        if values.shape!=(self._connection_count,) or np.any(np.isinf(values)):
            raise ValueError('Component boundary stages need one finite/NaN value per connection')
        finite=np.isfinite(values);g=self._enhanced_geometry
        if np.any(finite & (g.endpoints[:,1]>=0)):
            raise ValueError('Component boundary stage overrides must target external contacts')
        faces=self._ec_face.numpy();types=np.r_[self.boundary_type_qx.numpy().ravel(),self.boundary_type_qy.numpy().ravel()]
        if native_contacts and not self._enhanced_geometry.diagnostics.get('native_boundary_contacts',0):
            raise ValueError('Independent native stages require explicitly prescribed native coastal contacts')
        if not native_contacts and np.any(~np.isin(types[faces[finite]],(3,4,5))):
            raise ValueError('Component boundary stages require specified-stage boundary types')
        if np.any(values[finite]>g.maximum_stage):raise ValueError('Component stage exceeds the declared validation range')
        self._assign_component_boundary_stages(stage_m=values,native_contacts=native_contacts)

    def get_component_boundary_stages(self):
        if not self._ec_has_boundary_override:return np.full(self._connection_count,np.nan)
        return np.where(self._ec_boundary_override.numpy()!=0,self._ec_boundary_head_values.numpy(),np.nan)

    def capture_state(self, *, device_buffers=None, excluded_device_fields=()):
        state=super().capture_state(device_buffers=device_buffers,excluded_device_fields=excluded_device_fields)
        state['host']['enhanced_geometry']=self._enhanced_geometry
        state['host']['enhanced_build_parameters']=dict(self._enhanced_build_parameters)
        state['host']['enhanced_encoding']=self._enhanced_encoding
        state['host']['enhanced_coefficient_tolerance']=self._enhanced_coefficient_tolerance
        state['host']['enhanced_momentum_transport']=self._kinetic_transport
        state['host']['enhanced_exact_momentum']=self._exact_momentum
        state['host']['enhanced_coefficient_fraction']=self._coefficient_fraction
        state['host']['enhanced_component_timestep']=self._component_timestep
        state['host']['enhanced_has_boundary_override']=self._ec_has_boundary_override
        state['host']['enhanced_track_timestep_limits']=self._track_timestep_limits
        state['host']['enhanced_accuracy_options']=getattr(self,'_accuracy_options',None)
        state['host']['enhanced_accuracy_execution']=getattr(self,'_accuracy_execution','reference')
        return state

    def restore_state(self, state):
        geometry=state['host']['enhanced_geometry']
        if geometry.diagnostics['native_input_identity']!=self._enhanced_geometry.diagnostics['native_input_identity']:
            raise ValueError('Checkpoint terrain, mask and roughness differ from this model')
        if geometry is not self._enhanced_geometry:self._upload_component_geometry(geometry=geometry)
        for name,snapshot in state['device'].items():
            if snapshot is None:continue
            current=getattr(self,name,None)
            if current is None or current.shape!=snapshot.shape or current.dtype!=snapshot.dtype:
                setattr(self,name,wp.empty_like(snapshot))
        for name in ('_ec_coefficient_values','_ec_coefficient_ranges','_ec_coefficient_stage_index',
                     '_ec_tail_rows','_ec_tail_offsets','_ec_tail_halves'):
            if name not in state['device'] and hasattr(self,name):delattr(self,name)
        self._enhanced_geometry=geometry;self._connection_count=len(geometry.endpoints)
        self._enhanced_build_parameters=dict(state['host']['enhanced_build_parameters'])
        self._enhanced_encoding=state['host']['enhanced_encoding']
        self._enhanced_coefficient_tolerance=state['host']['enhanced_coefficient_tolerance']
        self._kinetic_transport=state['host']['enhanced_momentum_transport']
        self._exact_momentum=state['host'].get('enhanced_exact_momentum',0)
        self._coefficient_fraction=state['host']['enhanced_coefficient_fraction']
        self._component_timestep=state['host'].get('enhanced_component_timestep','storage')
        self._ec_has_boundary_override=state['host'].get('enhanced_has_boundary_override',0)
        self._track_timestep_limits=state['host'].get('enhanced_track_timestep_limits',False)
        if state['host'].get('enhanced_accuracy_options') is not None:
            self._accuracy_options=dict(state['host']['enhanced_accuracy_options'])
        self._accuracy_execution=state['host'].get('enhanced_accuracy_execution','reference')
        super().restore_state(state=state)
        if self._component_timestep=='adaptive':
            from floodwarp import enhanced_accuracy
            enhanced_accuracy.rebind(solver=self)

    def _component_stages(self, *, track=False, control=None,volume=None,eta=None,area=None):
        if control is None:control=self._ec_control
        if volume is None:volume=self._ec_volume
        if eta is None:eta=self._ec_eta
        if area is None:area=self._ec_area
        if self._component_count:
            wp.launch(kernel=kernels.stages,dim=self._component_count,inputs=[volume,
                self._ec_storage_offsets,self._ec_storage_rows,eta,area,
                self._ec_maximum_eta,self._enhanced_geometry.maximum_stage,int(track),self._ec_status,control],device=self.device)

    def _component_coefficients(self, *, control=None,eta=None,inertia_only=False,status=None,use_stage_qx=0,use_stage_qy=0):
        if control is None:control=self._ec_control
        if eta is None:eta=self._ec_eta
        if status is None:status=self._ec_status
        if not self._connection_count:return
        g=self._enhanced_geometry
        wp.launch(kernel=kernels.coefficients,block_dim=128,dim=self._connection_count,inputs=[self._ec_endpoints,
            self._ec_halves,self._ec_sill,self._ec_offsets,self._ec_rows,eta,self._ec_heads,
            self._ec_kind,wp.float64(float(self.dry_depth)),self._ec_d,self._ec_r,self._ec_forcing,
            wp.float64(g.maximum_stage),status,getattr(self,'_ec_coefficient_values',self._ec_rows),
            -1 if g.coefficient_ranges is not None else (0 if g.coefficient_values is not None else 2),
            getattr(self,'_ec_coefficient_stage_index',self._ec_adjacency),
            getattr(self,'_ec_coefficient_ranges',self._ec_halves),control,wp.float64(g.tail_start),wp.float64(g.tail_reference),
            getattr(self,'_ec_tail_halves',self._ec_halves),getattr(self,'_ec_tail_offsets',self._ec_offsets),
            getattr(self,'_ec_tail_rows',self._ec_storage_rows),self._ec_scalar_interval,int(inertia_only),
            self._ec_face,self.nx,self.ny*(self.nx+1),self.boundary_type_qx,self.boundary_type_qy,
            self.boundary_stage_qx,self.boundary_stage_qy,self._ec_boundary_override,
            self._ec_boundary_head_values,self._ec_has_boundary_override,int(use_stage_qx),int(use_stage_qy)],device=self.device)

    def _check_component_status(self):
        status=int(self._ec_status.numpy()[0])
        if status==1:raise StageRangeError('Enhanced state exceeds preprocessed coefficient stage range')
        if status==6:
            if self._component_timestep=='adaptive':
                raise NumericalFailure('Adaptive hydraulic trial exceeded its error/wave retry or minimum-timestep limit; trial state rolled back')
            raise NumericalFailure('Provisional source-CFL did not converge to a finite positive timestep')
        if status==5:raise NumericalFailure('Nested hook requested flow through an inactive contact or increased/reversed a post-limiter flux')
        if status:raise NumericalFailure(f'Enhanced hydraulic failure, status {status}')

    def _aggregate_components(self, *, full_fields=True, faces=True, accumulate_maxima=False):
        wp.launch(kernel=kernels.coarse_state,dim=(self.ny,self.nx),inputs=[self._ec_cell_component_offsets,
            self._ec_volume,self._ec_source_area,self._ec_soil,self._ec_cap,self._ec_applied,
            float(self.dx)*float(self.dy),self.water_depth,self.cumulative_infiltration,
            self.infiltration_cap_remaining_m,self.applied_rain,self.applied_other,self.applied_infil,
            int(full_fields),self.active_mask,
            self.max_water_depth_m if self.track_max_water_depth else self.water_depth,
            self._diagnostic_max_depth_m if self.track_cumulative_face_volumes and self.track_diagnostic_max_depth else self.water_depth,
            int(accumulate_maxima and self.track_max_water_depth),
            int(accumulate_maxima and self.track_cumulative_face_volumes and self.track_diagnostic_max_depth)],device=self.device)
        if not faces:return
        wp.launch(kernel=kernels.coarse_flux,dim=self.ny*(self.nx+1)+(self.ny+1)*self.nx,
            inputs=[self._ec_face_offsets,self._ec_face_members,self._ec_sign,self._ec_q,float(self.dx),
                self.nx,self.ny*(self.nx+1),self.qx,self.qy],device=self.device)
        if full_fields:
            wp.copy(self.qx_next,self.qx);wp.copy(self.qy_next,self.qy)

    def set_water_depth(self, depth):
        if not self._enhanced_ready:return super().set_water_depth(depth=depth)
        values=coarse_storage_to_components(geometry=self._enhanced_geometry,
            native_bed=self._native_bed,storage_depth=depth)
        super().set_water_depth(depth=depth)
        wp.copy(self._ec_volume,wp.array(values,dtype=WP_FLOAT,device=self.device))
        for field in (self._ec_q,self._ec_proposed,self._ec_previous_d,self._ec_soil,
                      self._ec_applied,self._ec_status,self._ec_control,self._ec_boundary_closure_loss,
                      self._ec_source_ledger,self._ec_source_sums):field.zero_()
        self._ec_maximum_eta.fill_(-np.inf)
        self._component_stages(track=True)
        self._aggregate_components()
        self.initial_volume=float(np.sum(self._ec_volume.numpy(),dtype=np.float64))
        if hasattr(self,'_accuracy_state'):
            self._accuracy_state.zero_()
            self._accuracy_condition.zero_()
            self._accuracy_defect_counts.zero_()
        if hasattr(self,'_embedded_coefficients_valid'):self._embedded_coefficients_valid.zero_()

    def set_component_state(self, *, volume, discharge=None):
        """Explicit initialization/restart from independent storage [m3] and Q [m3/s].

        This resets event clocks/soil/accounting like set_water_depth. Use the
        complete capture_state/restore_state pair for exact continued events.
        """
        # CPU NumPy views may alias live Warp arrays, which reset clears.
        values=np.array(volume,dtype=float,copy=True)
        if values.shape!=(self._component_count,) or np.any(values<0) or not np.all(np.isfinite(values)):
            raise ValueError('finite nonnegative component volumes required')
        flux=np.zeros(self._connection_count) if discharge is None else np.array(discharge,dtype=float,copy=True)
        if flux.shape!=(self._connection_count,) or not np.all(np.isfinite(flux)):
            raise ValueError('finite connection discharges required')
        self.set_water_depth(depth=np.zeros((self.ny,self.nx)))
        wp.copy(self._ec_volume,wp.array(values,dtype=WP_FLOAT,device=self.device))
        wp.copy(self._ec_q,wp.array(flux,dtype=WP_FLOAT,device=self.device))
        self._ec_maximum_eta.fill_(-np.inf)
        self._component_stages(track=True);self._check_component_status();self._aggregate_components()
        self.initial_volume=float(np.sum(self._ec_volume.numpy(),dtype=np.float64))

    def get_component_state(self):
        self._component_stages()
        return dict(volume_m3=self._ec_volume.numpy(),stage_m=self._ec_eta.numpy(),
                    discharge_m3_per_s=self._ec_q.numpy(),maximum_stage_m=self._ec_maximum_eta.numpy(),
                    cumulative_infiltration_m=self._ec_soil.numpy(),remaining_infiltration_cap_m=self._ec_cap.numpy())

    def get_native_water_depth(self, *, maximum=False):
        if maximum and not self.track_max_water_depth:
            raise RuntimeError('Maximum native depth requires track_max_water_depth=True')
        if not maximum:self._component_stages()
        stages=(self._ec_maximum_eta if maximum else self._ec_eta).numpy()
        owner=self._enhanced_geometry.pixel_owner
        result=np.zeros_like(self._native_bed)
        mask=owner>=0;result[mask]=np.maximum(stages[owner[mask]]-self._native_bed[mask],0.)
        return result

    def get_max_local_water_depth(self):
        field=self.get_native_water_depth(maximum=True);f=self._subgrid_factor
        return field.reshape(self.ny,f,self.nx,f).max(axis=(1,3))

    def is_subgrid(self):
        return self._enhanced_ready

    def get_subgrid_state(self):
        return coarse_component_diagnostics(geometry=self._enhanced_geometry,
            native_bed=self._native_bed,volume=self._ec_volume.numpy())

    def get_water_surface_elevation(self):
        """Storage-equivalent coarse stage; independent physical stages use get_component_state."""
        return self.get_subgrid_state()['water_surface_elevation']

    def set_water_surface_elevation(self, stage):
        values=np.asarray(stage,dtype=float)
        if values.shape!=(self.ny,self.nx) or not np.all(np.isfinite(values)):
            raise ValueError('finite coarse stage field required')
        f=self._subgrid_factor
        native_depth=np.maximum(values.repeat(f,0).repeat(f,1)-self._native_bed,0.)
        native_depth[self._enhanced_geometry.pixel_owner<0]=0.
        self.set_component_state(volume=component_volumes(geometry=self._enhanced_geometry,native_depth=native_depth))

    def _configure_infiltration(self, **kwargs):
        super()._configure_infiltration(**kwargs)
        if getattr(self,'_enhanced_ready',False):self._ec_soil.zero_()

    def set_infiltration_storage_limit_m(self, limit_m):
        super().set_infiltration_storage_limit_m(limit_m=limit_m)
        if self._enhanced_ready:
            values=self.infiltration_cap_remaining_m.numpy().ravel()[self._enhanced_geometry.state_cell]
            wp.copy(self._ec_cap,wp.array(values,dtype=WP_FLOAT,device=self.device))

    def clear_infiltration_limits(self):
        super().clear_infiltration_limits()
        if self._enhanced_ready:self._ec_cap.zero_()

    def restore_infiltration_constraints(self, *, state):
        super().restore_infiltration_constraints(state=state)
        wp.copy(self._ec_cap,state['device']['_ec_cap'])

    def enable_timestep_diagnostics(self):
        """Record exact local controller bounds; no extra kernels or reductions.

        Optional profiling memory is48bytes/component. Read it only at an
        observation boundary; ordinary hydraulic timesteps perform no readback.
        """
        self._ec_timestep_limits=wp.zeros((max(1,self._component_count),6),dtype=wp.float64,device=self.device)
        self._track_timestep_limits=True

    def get_timestep_limit_diagnostics(self, *, count=10):
        """Describe the last controller evaluation, including tied minima."""
        if not self._track_timestep_limits:raise ValueError('Enable timestep diagnostics before stepping')
        if not isinstance(count,int) or count<1:raise ValueError('count must be a positive integer')
        local=self._ec_local_dt.numpy();bounds=self._ec_timestep_limits.numpy()
        if self._component_count and not np.all(bounds[:self._component_count,0]>0.):
            raise ValueError('No successful timestep evaluation recorded')
        order=np.argsort(local,kind='stable')[:count];records=[]
        for component in order:
            wave,change,demand,source,stage,area=bounds[component]
            kind='user_cap'
            if self._component_timestep=='storage' and change<=wave and change<=self.maximum_dt:kind='storage_change'
            elif wave<=self.maximum_dt:kind='wave_cfl'
            cell=int(self._enhanced_geometry.state_cell[component]);j,i=divmod(cell,self.nx)
            records.append(dict(component=int(component),cell=[j,i],bound=kind,local_dt_s=float(local[component]),
                wave_dt_s=float(wave),storage_change_dt_s=float(change),absolute_connection_flow_m3_s=float(demand),
                positive_source_m3_s=float(source),stage_m=float(stage),wetted_area_m2=float(area),
                source_area_m2=float(self._enhanced_geometry.source_area[component])))
        minimum=float(local.min()) if len(local) else float(self.maximum_dt)
        return dict(time_s=self.simulation_time,last_accepted_dt_s=float(self.current_dt),
            raw_minimum_dt_s=minimum,tied_minimum_components=int(np.count_nonzero(local==minimum)),records=records)

    def calculate_cfl_dt(self, water_depth=None):
        if self._component_timestep=='adaptive':
            raise ValueError('Adaptive temporal-error selection requires a hydraulic trial in run_device_loop')
        if self._component_timestep=='provisional':
            if water_depth is not None:
                raise ValueError('Provisional source-CFL operates on authoritative component state, not a coarse predictor')
            return self._calculate_provisional_timestep(remaining_time=None)
        eta=self._ec_eta;area=self._ec_area
        if water_depth is not None:
            if not isinstance(water_depth,wp.array) or water_depth.shape!=(self.ny,self.nx) or water_depth.dtype!=WP_FLOAT:
                raise ValueError('CFL coarse predictor must be a solver-precision Warp depth array of model shape')
            if not hasattr(self,'_ec_predictor_volume'):
                self._ec_predictor_volume=wp.zeros_like(self._ec_volume)
                self._ec_predictor_eta=wp.zeros_like(self._ec_eta);self._ec_predictor_area=wp.zeros_like(self._ec_area)
                self._ec_predictor_status=wp.zeros(1,dtype=int,device=self.device)
            self._check_component_status();self._ec_predictor_status.zero_()
            wp.launch(kernel=kernels.coarse_predictor_volume,dim=(self.ny,self.nx),
                inputs=[self._ec_cell_component_offsets,self._ec_source_area,self._ec_volume,water_depth,
                    float(self.dx)*float(self.dy),self._ec_predictor_volume,self._ec_predictor_status],device=self.device)
            if self._ec_predictor_status.numpy()[0]:
                raise ValueError('CFL predictor depths must be finite and nonnegative on active cells')
            eta=self._ec_predictor_eta;area=self._ec_predictor_area
            self._component_stages(volume=self._ec_predictor_volume,eta=eta,area=area)
        else:self._component_stages()
        self._component_coefficients(eta=eta)
        cap=self.maximum_dt
        if self.adaptive_dt_growth_factor is not None and self._adaptive_dt_reference is not None and self._adaptive_dt_reference>0.:
            cap=min(cap,self._adaptive_dt_reference*self.adaptive_dt_growth_factor)
        wp.launch(kernel=kernels.set_control,dim=1,inputs=[self._ec_control,cap],device=self.device)
        if self._component_count:
            wp.launch(kernel=kernels.timestep,block_dim=128,dim=self._component_count,inputs=[self._ec_adjacency_offsets,
                self._ec_adjacency,self._ec_storage_offsets,self._ec_storage_rows,self._ec_d,self._ec_q,
                eta,area,self._ec_state_cell,self.nx,self._ec_source_area,
                self.rainfall_rate,self.other_source_rate,self.additional_source_rate,
                float(self.gravity),float(self.cfl_number),self._coefficient_fraction,float(self.dry_depth),
                self._ec_control,self._ec_status,self.rainfall_spatial_multiplier,0,self._ec_local_dt,int(self._component_timestep=='wave'),
                int(self._track_timestep_limits),self._ec_timestep_limits],device=self.device)
        self._check_component_status()
        return float(self._ec_control.numpy()[1])

    def calculate_stable_timestep(self, remaining_time=None):
        if self._component_timestep=='provisional':return self._calculate_provisional_timestep(remaining_time=remaining_time)
        candidate=self.calculate_cfl_dt()
        if self.adaptive_dt_growth_factor is not None and self._adaptive_dt_reference>0:
            candidate=min(candidate,self._adaptive_dt_reference*self.adaptive_dt_growth_factor)
        return candidate if remaining_time is None else min(candidate,remaining_time)

    def _component_sources(self, *, control=None, use_hyetograph=0,committed=False,refresh_stage=False,provisional=0):
        if control is None:control=self._ec_control
        if not self._component_count:return
        status=self._ec_trial_status if provisional else self._ec_status
        eta=self._ec_trial_eta if provisional else self._ec_eta
        area=self._ec_trial_area if provisional else self._ec_area
        wp.launch(kernel=kernels.sources,block_dim=128,dim=self._component_count,inputs=[self._ec_volume,
            self._ec_state_cell,self.nx,self._ec_source_area,self._ec_storage_offsets,self._ec_storage_rows,
            self.rainfall_rate,self.other_source_rate,self.additional_source_rate,self.infiltration_capacity,
            self.infiltration_f0,self.infiltration_fd,self.infiltration_k,self.green_ampt_ks,self.green_ampt_psi,
            self.green_ampt_delta_theta,self._ec_soil,self._ec_cap,self.infiltration_exclusion_mask,
            self.infiltration_method,self.source_mode,self.infiltration_cap_active,self.dry_depth,
            control,self.cumulative_totals,self._ec_applied,status,
            self.rainfall_spatial_multiplier,use_hyetograph,self._ec_adjacency_offsets,self._ec_adjacency,
            self._ec_q,int(committed),int(self._defer_component_sources),self._ec_source_ledger,
            int(refresh_stage),eta,area,self._ec_maximum_eta,
            wp.float64(self._enhanced_geometry.maximum_stage),int(provisional),
            self._ec_trial_volume if provisional else self._ec_volume],device=self.device)













    def _enqueue_component_hydraulics(self, *, control, use_hyetograph=0,collect_free_outflow_diagnostics=False,hook_dt=None, defer_outputs=False, require_faces=True,coefficients_ready=False):
        """Shared GPU/CPU kernel sequence; no readback or native graph work."""
        if not self.apply_sources_after_momentum:
            self._component_sources(control=control,use_hyetograph=use_hyetograph,refresh_stage=defer_outputs)
        # Global stages are valid from prior continuity; accepted sources
        # refresh them in their own component pass. Host/LTS retain their path.
        if not defer_outputs:self._component_stages(control=control)
        if not coefficients_ready:self._component_coefficients(control=control)
        if self._connection_count:
            wp.launch(kernel=kernels.momentum,dim=self._connection_count,inputs=[self._ec_d,self._ec_r,
                self._ec_forcing,self._ec_q,self._ec_previous_d,self._ec_proposed,control,
                float(self.gravity),self._ec_status,self._kinetic_transport,self._ec_endpoints,self._ec_sill,
                self._ec_eta,self._ec_heads,self._ec_kind,float(self.dry_depth),self._exact_momentum],device=self.device)
        if self._component_count:
            safety=1.000000000000002 if NP_FLOAT==np.float64 else 1.000001
            wp.launch(kernel=kernels.demands,dim=self._component_count,inputs=[self._ec_volume,
                self._ec_adjacency_offsets,self._ec_adjacency,self._ec_proposed,self._ec_alpha,
                control,safety,self._ec_status,self._ec_kind,self._ec_alpha_free,self.free_outflow_limiter_mode],device=self.device)
        if self._connection_count:
            wp.launch(kernel=kernels.limit,dim=self._connection_count,inputs=[self._ec_endpoints,
                self._ec_proposed,self._ec_alpha,self._ec_q,control,self.cumulative_totals,self._ec_status,
                self._ec_kind,self._ec_alpha_free],device=self.device)
        if collect_free_outflow_diagnostics:
            wp.launch(kernel=kernels.free_outflow_diagnostics,dim=self.ny*self.nx,
                inputs=[self._ec_cell_component_offsets,self._ec_adjacency_offsets,self._ec_adjacency,
                    self._ec_kind,self._ec_proposed,self._ec_q,control,self._ec_status,self.free_outflow_diagnostics],device=self.device)
        if self.apply_sources_after_momentum:
            self._component_sources(control=control,use_hyetograph=use_hyetograph,committed=True)
        if self._component_count:
            wp.launch(kernel=kernels.continuity,dim=self._component_count,inputs=[self._ec_volume,
                self._ec_adjacency_offsets,self._ec_adjacency,self._ec_q,control,self._ec_status,
                int(defer_outputs)*(2 if self.apply_sources_after_momentum else 1),
                self._ec_storage_offsets,self._ec_storage_rows,self._ec_eta,self._ec_area,
                self._ec_maximum_eta,wp.float64(self._enhanced_geometry.maximum_stage),int(self.track_max_water_depth)],device=self.device)
        if not defer_outputs:self._component_stages(track=self.track_max_water_depth,control=control)
        self._aggregate_components(full_fields=not defer_outputs,faces=require_faces,
            accumulate_maxima=defer_outputs)

    def step(self, remaining_time=None, fixed_dt=None):
        if self._component_timestep=='adaptive':
            raise ValueError('Error-controlled step doubling uses run_device_loop, on CPU or CUDA')
        dt=self.calculate_stable_timestep(remaining_time=remaining_time) if fixed_dt is None else float(fixed_dt)
        if not np.isfinite(dt) or dt<=0 or (remaining_time is not None and dt>remaining_time+1e-12):
            raise ValueError('finite positive dt no greater than remaining_time required')
        if dt<self.minimum_dt and (remaining_time is None or remaining_time>=self.minimum_dt):
            raise RuntimeError(f'Enhanced stable dt {dt} is below minimum_dt {self.minimum_dt}')
        wp.launch(kernel=kernels.set_control,dim=1,inputs=[self._ec_control,dt],device=self.device)
        self._enqueue_component_hydraulics(control=self._ec_control,hook_dt=dt,
            coefficients_ready=self._component_timestep=='provisional' and fixed_dt is None)
        if self.structures:self._apply_structure_transfers(dt=dt)
        self._check_component_status()
        self._accumulate_face_volume_diagnostics(qx=self.qx,qy=self.qy,dt_wp=WP_FLOAT(dt))
        self._accumulate_depth_diagnostic();self._accumulate_max_water_depth()
        boundary_limited=remaining_time is not None and dt>=remaining_time-1e-12 and remaining_time<self.maximum_dt-1e-12
        if not boundary_limited or self._adaptive_dt_reference<=0:self._adaptive_dt_reference=dt
        self.current_dt=dt;self.simulation_time+=dt
        return dt

    def _flush_pending_mass_accounting(self):
        if not getattr(self,'_enhanced_ready',False):return super()._flush_pending_mass_accounting()
        if self._defer_component_sources and self._component_count:
            wp.utils.array_sum(self._ec_source_ledger,out=self._ec_source_sums,axis=0)
            wp.launch(kernel=kernels.book_source_totals,dim=4,
                inputs=[self._ec_source_sums,self.cumulative_totals],device=self.device)
            self._ec_source_ledger.zero_()
        if self._defer_component_outputs:
            self._aggregate_components()

    def run_device_loop(self, *args, **kwargs):
        call=None
        if self._component_timestep=='adaptive':
            from floodwarp import enhanced_accuracy
            enhanced_accuracy.validate(solver=self)
            call=inspect.signature(LocalInertialSolver.run_device_loop).bind(self,*args,**kwargs)
            self._accuracy_collect_free_outflow_diagnostics=bool(call.arguments.get('collect_free_outflow_diagnostics',False))
            if self._accuracy_options['estimator']=='embedded' and self._accuracy_execution=='optimized':
                observe_sync=call.arguments.get('on_sync')
                observe_progress=call.arguments.get('on_progress')
                if callable(observe_sync):
                    def synchronized_sync(steps,simulation_time):
                        self._aggregate_components()
                        return observe_sync(steps,simulation_time)
                    call.arguments['on_sync']=synchronized_sync
                if callable(observe_progress):
                    def synchronized_progress(steps,simulation_time,execution):
                        self._aggregate_components()
                        return observe_progress(steps,simulation_time,execution)
                    call.arguments['on_progress']=synchronized_progress
        # Generic table signatures must exist before CUDA capture can load this
        # module. Adding an overload during capture may unload an active module.
        values=getattr(self,'_ec_coefficient_values',self._ec_rows)
        indices=getattr(self,'_ec_coefficient_stage_index',self._ec_adjacency)
        wp.overload(kernels.coefficients,{'values':wp.array2d(dtype=values.dtype),
            'indices':wp.array(dtype=indices.dtype)})
        wp.load_module(module=kernels.coefficients.module,device=self.device)
        self._defer_component_sources=bool(self.use_reduced_mass_accounting)
        self._defer_component_outputs=True
        try:
            if call is not None:return LocalInertialSolver.run_device_loop(*call.args,**call.kwargs)
            return super().run_device_loop(*args,**kwargs)
        finally:
            self._flush_pending_mass_accounting()
            self._defer_component_sources=False
            self._defer_component_outputs=False

    def configure_temporal_accuracy(self, *, stage_atol_m, discharge_atol_m3_s,
            relative_tolerance, coefficient_relative_tolerance=.05, safety=.8, max_attempts=64,
            estimator='doubling',defect_evaluation='analytic',wave_proposal_fraction=.8):
        """Opt in to conservative, error-tested step doubling; defaults stay unchanged."""
        from floodwarp import enhanced_accuracy
        enhanced_accuracy.configure(solver=self,stage_atol_m=stage_atol_m,
            discharge_atol_m3_s=discharge_atol_m3_s,relative_tolerance=relative_tolerance,
            coefficient_relative_tolerance=coefficient_relative_tolerance,safety=safety,max_attempts=max_attempts,
            estimator=estimator,defect_evaluation=defect_evaluation,wave_proposal_fraction=wave_proposal_fraction)

    def temporal_accuracy_statistics(self):
        from floodwarp import enhanced_accuracy
        return enhanced_accuracy.statistics(solver=self)

    def _ensure_provisional_buffers(self):
        from floodwarp import enhanced_timestep as trial
        if getattr(self,'_ec_trial_volume',None) is not None and len(self._ec_trial_volume)==self._component_count:return
        self._ec_trial_volume=wp.zeros_like(self._ec_volume)
        self._ec_trial_eta=wp.zeros_like(self._ec_eta)
        self._ec_trial_area=wp.zeros_like(self._ec_area)
        self._ec_trial_empty=wp.zeros(1,dtype=wp.float64,device=self.device)
        self._ec_trial_control=wp.zeros_like(self._ec_control)
        self._ec_trial_status=wp.zeros(1,dtype=int,device=self.device)
        self._ec_trial_condition=wp.zeros(1,dtype=int,device=self.device)
        self._ec_trial_statistics=wp.zeros(7,dtype=wp.float64,device=self.device)
        wp.load_module(module=trial.initialize.module,device=self.device)

    def _prepare_provisional_graph(self, *, control,end_time,minimum_dt,
            hyeto_times,hyeto_rates,use_hyetograph,storm_duration,
            stage_times_qx,stage_values_qx,use_stage_qx,stage_times_qy,stage_values_qy,use_stage_qy,
            checkpoint_times,checkpoint_count):
        self._ensure_provisional_buffers()
        self._ec_trial_options=dict(end_time=end_time,minimum_dt=minimum_dt,hyeto_times=hyeto_times,
            hyeto_rates=hyeto_rates,use_hyetograph=use_hyetograph,storm_duration=storm_duration,
            stage_times_qx=stage_times_qx,stage_values_qx=stage_values_qx,use_stage_qx=use_stage_qx,
            stage_times_qy=stage_times_qy,stage_values_qy=stage_values_qy,use_stage_qy=use_stage_qy,
            checkpoint_times=checkpoint_times,checkpoint_count=checkpoint_count)
        if self._component_timestep=='adaptive':
            from floodwarp import enhanced_accuracy
            enhanced_accuracy.prepare_graph(solver=self,control=control,options=self._ec_trial_options)
            return
        if self.device.is_cuda:
            with wp.ScopedDevice(self.device):
                if not wp.is_conditional_graph_supported():
                    raise ValueError('Provisional source-CFL requires CUDA conditional-graph support')
            with wp.ScopedCapture(device=self.device,force_module_load=True) as body:
                self._enqueue_provisional_attempt(control=control)
            with wp.ScopedCapture(device=self.device) as loop:
                wp.capture_while(self._ec_trial_condition,while_body=body.graph)
            self._ec_trial_graph=loop.graph
            self._ec_trial_body=body.graph

    def _enqueue_provisional_attempt(self, *, control):
        from floodwarp import enhanced_timestep as trial
        options=self._ec_trial_options
        wp.launch(kernel=trial.prepare,dim=1,inputs=[control,self._ec_trial_control,
            self._ec_trial_status,self._ec_trial_statistics],device=self.device)
        wp.launch(kernel=kernels.device_finalize,dim=1,inputs=[self._ec_trial_control,
            wp.float64(options['end_time']),wp.float64(0.),wp.float64(self.maximum_dt),
            wp.float64(self.adaptive_dt_growth_factor or 0.),0,self.cfl_number,self.dx,self.gravity,self.dry_depth,
            options['hyeto_times'],options['hyeto_rates'],options['use_hyetograph'],wp.float64(options['storm_duration']),
            int(not self.apply_sources_after_momentum),options['stage_times_qx'],options['stage_values_qx'],options['use_stage_qx'],
            options['stage_times_qy'],options['stage_values_qy'],options['use_stage_qy'],
            options['checkpoint_times'],options['checkpoint_count']],device=self.device)
        wp.launch(kernel=trial.remember,dim=1,inputs=[self._ec_trial_control,self._ec_trial_statistics],device=self.device)
        self._component_sources(control=self._ec_trial_control,use_hyetograph=options['use_hyetograph'],
            provisional=2 if self.apply_sources_after_momentum else 1)
        self._component_coefficients(control=self._ec_trial_control,eta=self._ec_trial_eta,
            status=self._ec_trial_status,use_stage_qx=options['use_stage_qx'],use_stage_qy=options['use_stage_qy'])
        if self._component_count:
            wp.launch(kernel=trial.wave_bound,block_dim=128,dim=self._component_count,inputs=[self._ec_adjacency_offsets,
                self._ec_adjacency,self._ec_d,self._ec_trial_area,self._ec_trial_eta,float(self.gravity),float(self.cfl_number),
                self._ec_trial_statistics,self._ec_trial_status,self._ec_local_dt,int(self._track_timestep_limits),
                self._ec_timestep_limits],device=self.device)
        wp.launch(kernel=trial.decide,dim=1,inputs=[control,self._ec_trial_control,self._ec_status,self._ec_trial_status,
            self._ec_trial_statistics,self._ec_trial_condition,wp.float64(options['minimum_dt']),wp.float64(self.maximum_dt),
            wp.float64(self.adaptive_dt_growth_factor or 0.),wp.float64(options['end_time']),float(self.cfl_number)],device=self.device)

    def _enqueue_provisional_selection(self, *, control):
        from floodwarp import enhanced_timestep as trial
        wp.launch(kernel=trial.initialize,dim=1,inputs=[control,self._ec_status,self._ec_trial_statistics,
            self._ec_trial_condition,float(self.maximum_dt)],device=self.device)
        if self.device.is_cuda:
            if self.device.is_capturing:wp.capture_while(self._ec_trial_condition,while_body=self._ec_trial_body)
            else:wp.capture_launch(self._ec_trial_graph)
        else:
            while self._ec_trial_condition.numpy()[0]:self._enqueue_provisional_attempt(control=control)

    def _calculate_provisional_timestep(self, *, remaining_time):
        from floodwarp import enhanced_timestep as trial
        self._ensure_provisional_buffers()
        wp.launch(kernel=trial.host_clock,dim=1,inputs=[self._ec_control,float(self.simulation_time),
            float(self._adaptive_dt_reference or 0.)],device=self.device)
        if not hasattr(self,'_ec_trial_empty'):
            self._ec_trial_empty=wp.zeros(1,dtype=wp.float64,device=self.device)
        empty=self._ec_trial_empty
        end=self.simulation_time+(self.maximum_dt if remaining_time is None else remaining_time)
        self._prepare_provisional_graph(control=self._ec_control,end_time=end,minimum_dt=0.,hyeto_times=empty,
            hyeto_rates=empty,use_hyetograph=0,storm_duration=0.,stage_times_qx=empty,stage_values_qx=empty,
            use_stage_qx=0,stage_times_qy=empty,stage_values_qy=empty,use_stage_qy=0,
            checkpoint_times=empty,checkpoint_count=0)
        self._enqueue_provisional_selection(control=self._ec_control)
        self._check_component_status()
        return float(self._ec_control.numpy()[1])

    def provisional_timestep_statistics(self):
        """Read only at an observation boundary; all trial work is device-side."""
        if not hasattr(self,'_ec_trial_statistics'):raise ValueError('No provisional timestep has been evaluated')
        values=self._ec_trial_statistics.numpy()
        return dict(checks=int(values[4]),rejections=int(values[5]),last_attempts=int(values[3]),
            maximum_accepted_courant=float(values[6]),last_candidate_s=float(values[1]),last_wave_bound_s=float(values[2]))

    def _enqueue_enhanced_device_step(self, *, control,dt_min_scratch,end_time,
        hyeto_times,hyeto_rates,use_hyetograph,storm_duration,
        stage_times_qx,stage_values_qx,use_stage_qx,stage_times_qy,stage_values_qy,use_stage_qy,
        checkpoint_times,checkpoint_count,dt_history,history_capacity,record_history,
        checkpoint_outlet_rows,checkpoint_field_indices,checkpoint_times_out,checkpoint_outlet_out,
        checkpoint_volume_out,checkpoint_depth_out,collect_free_outflow_diagnostics):
        if self._component_timestep=='adaptive':
            from floodwarp import enhanced_accuracy
            enhanced_accuracy.enqueue(solver=self,control=control)
        elif self._component_timestep=='provisional':
            self._enqueue_provisional_selection(control=control)
        else:
            wp.launch(kernel=device_loop_timestep_cap_kernel,dim=1,
                inputs=[control,dt_min_scratch,wp.float64(end_time),wp.float64(self.maximum_dt),
                    wp.float64(self.adaptive_dt_growth_factor or 0.),hyeto_times,use_hyetograph,
                    wp.float64(storm_duration),checkpoint_times,checkpoint_count],device=self.device)
            wp.launch(kernel=kernels.prepare_device_timestep,dim=1,
                inputs=[control,wp.float64(self.maximum_dt),hyeto_times,hyeto_rates,use_hyetograph,
                    self._ec_rain_bound,wp.float64(self.adaptive_dt_growth_factor or 0.),
                    wp.float64(storm_duration)],device=self.device)
            # Initialization, every accepted continuity update, and all public
            # volume exchanges already refresh eta/area. Sources have not changed
            # volume here; momentum still refreshes them AFTER accepted sources.
            self._component_coefficients(control=control,inertia_only=True)
            if self._component_count:
                wp.launch(kernel=kernels.timestep,block_dim=128,dim=self._component_count,
                    inputs=[self._ec_adjacency_offsets,self._ec_adjacency,self._ec_storage_offsets,
                        self._ec_storage_rows,self._ec_d,self._ec_q,self._ec_eta,self._ec_area,
                        self._ec_state_cell,self.nx,self._ec_source_area,self.rainfall_rate,self.other_source_rate,
                        self.additional_source_rate,float(self.gravity),float(self.cfl_number),
                        self._coefficient_fraction,float(self.dry_depth),control,self._ec_status,
                        self.rainfall_spatial_multiplier,use_hyetograph,self._ec_local_dt,int(self._component_timestep=='wave'),
                        int(self._track_timestep_limits),self._ec_timestep_limits],device=self.device)
            wp.launch(kernel=kernels.device_finalize,dim=1,
                inputs=[control,wp.float64(end_time),wp.float64(self.minimum_dt),wp.float64(self.maximum_dt),
                    wp.float64(self.adaptive_dt_growth_factor or 0.),0,self.cfl_number,self.dx,self.gravity,self.dry_depth,
                    hyeto_times,hyeto_rates,use_hyetograph,wp.float64(storm_duration),int(not self.apply_sources_after_momentum),
                    stage_times_qx,stage_values_qx,use_stage_qx,stage_times_qy,stage_values_qy,use_stage_qy,
                    checkpoint_times,checkpoint_count],device=self.device)
        if (use_stage_qx or use_stage_qy) and self._component_timestep!='adaptive':
            wp.launch(kernel=device_loop_stage_fill_kernel,dim=(self.ny+1,self.nx+1),
                inputs=[self.boundary_type_qx,self.boundary_stage_qx,self.boundary_type_qy,self.boundary_stage_qy,
                    control,use_stage_qx,use_stage_qy],device=self.device)
        if hasattr(self,'_ec_audit_stats'):
            from floodwarp import enhanced_execution_audit as execution_audit
            execution_audit.enqueue_before(solver=self,control=control)
        if self._component_timestep!='adaptive':
            self._enqueue_component_hydraulics(control=control,use_hyetograph=use_hyetograph,
                collect_free_outflow_diagnostics=collect_free_outflow_diagnostics,
                defer_outputs=True,require_faces=self.track_cumulative_face_volumes or checkpoint_count>0,
                coefficients_ready=self._component_timestep=='provisional')
        if hasattr(self,'_ec_audit_stats'):
            execution_audit.enqueue_after(solver=self,control=control,end_time=end_time)
        if self.track_cumulative_face_volumes and self._component_timestep!='adaptive':
            wp.launch(kernel=device_loop_accumulate_face_volume_qx_kernel,dim=(self.ny,self.nx+1),
                inputs=[self.qx,self.cumulative_net_qx_volume_m3,
                    self.cumulative_abs_qx_volume_m3 if self.track_absolute_face_volumes else self._diagnostic_absolute_placeholder,
                    self._diagnostic_peak_qx_rows,self.dy,control,int(self.track_absolute_face_volumes)],device=self.device)
            wp.launch(kernel=device_loop_accumulate_face_volume_qy_kernel,dim=(self.ny+1,self.nx),
                inputs=[self.qy,self.cumulative_net_qy_volume_m3,
                    self.cumulative_abs_qy_volume_m3 if self.track_absolute_face_volumes else self._diagnostic_absolute_placeholder,
                    self._diagnostic_peak_qy_rows,self.dx,control,int(self.track_absolute_face_volumes)],device=self.device)
        # The accepted coarse reconstruction updates both independent max
        # arrays in the same pass; component maxima are tracked in stages.
        if checkpoint_count:
            wp.launch(kernel=kernels.checkpoint_fields,dim=(self.ny,self.nx),
                inputs=[self.water_depth,self.qx,self._ec_cell_component_offsets,self._ec_volume,control,self._ec_status,
                    checkpoint_outlet_rows,checkpoint_field_indices,checkpoint_outlet_out,checkpoint_volume_out,
                    checkpoint_depth_out,checkpoint_count],device=self.device)
        if hasattr(self,'_ec_run_summary'):
            from floodwarp import enhanced_run_diagnostics as audit
            wp.launch(kernel=audit.component_transport,dim=self._component_count,
                inputs=[control,self._ec_status,self._ec_adjacency_offsets,self._ec_adjacency,
                    self._ec_proposed,self._ec_q,self._ec_alpha,self._ec_run_components],device=self.device)
            wp.launch(kernel=audit.accepted_step,dim=1,
                inputs=[control,self._ec_status,self.cumulative_totals,self._ec_run_summary,
                    self._ec_run_histogram,self._ec_run_samples,wp.float64(self._ec_run_sample_s),wp.float64(self._ec_run_origin_s)],device=self.device)
        wp.launch(kernel=kernels.finish_device_step,dim=1,
            inputs=[control,self._ec_status,dt_history,history_capacity,record_history,
                checkpoint_times_out,checkpoint_count],device=self.device)

    def run_spatially_adaptive_loop(self, *args, **kwargs):
        if self._component_timestep!='storage':
            raise ValueError('Experimental wave controller is global-only; spatial LTS retains its validated storage controller')
        if not hasattr(self,'_ec_lts_applied'):
            self._ec_lts_applied=wp.zeros_like(self._ec_applied)
            self._ec_lts_cell_dt=wp.zeros((self.ny,self.nx),dtype=WP_FLOAT,device=self.device)
            self._ec_lts_levels=wp.zeros(self._connection_count,dtype=int,device=self.device)
            self._ec_lts_updates=wp.zeros(1,dtype=int,device=self.device)
        if len(self._ec_lts_levels)!=self._connection_count:
            self._ec_lts_levels=wp.zeros(self._connection_count,dtype=int,device=self.device)
        self._ec_lts_updates.zero_()
        result=super().run_spatially_adaptive_loop(*args,**kwargs)
        self._check_component_status()
        result['component_momentum_updates']=int(self._ec_lts_updates.numpy()[0])
        result['component_volume_updates']=self._component_count*result['base_steps']
        # V is the effective component state, committed every base tick. Unlike
        # traditional pending registers this has no delayed coarse storage copy.
        # Do not report scheduled cell commits as saved continuity work.
        result['scheduled_cell_commits']=result['cell_updates']
        result['cell_updates']=result['equivalent_global_cell_updates']
        result['cell_work_fraction']=1.
        result['ideal_cell_work_speedup']=1.
        result['source_plus_continuity_work_fraction']=1.
        result['ideal_source_plus_continuity_speedup']=1.
        result['component_continuity_base_clock']=True
        return result

    def _enhanced_lts_face_timesteps(self, *, cap_dt,rain_max,use_hyetograph):
        self._check_component_status();self._ec_control.zero_();self._ec_lts_applied.zero_()
        wp.launch(kernel=kernels.lts_forcing,dim=1,
            inputs=[self._ec_control,cap_dt,rain_max],device=self.device)
        wp.launch(kernel=kernels.set_control,dim=1,inputs=[self._ec_control,cap_dt],device=self.device)
        self._component_stages();self._component_coefficients()
        if self._component_count:
            wp.launch(kernel=kernels.timestep,block_dim=128,dim=self._component_count,
                inputs=[self._ec_adjacency_offsets,self._ec_adjacency,self._ec_storage_offsets,
                    self._ec_storage_rows,self._ec_d,self._ec_q,self._ec_eta,self._ec_area,
                    self._ec_state_cell,self.nx,self._ec_source_area,self.rainfall_rate,self.other_source_rate,
                    self.additional_source_rate,float(self.gravity),float(self.cfl_number),
                    self._coefficient_fraction,float(self.dry_depth),self._ec_control,self._ec_status,
                    self.rainfall_spatial_multiplier,int(use_hyetograph),self._ec_local_dt,0,
                    int(self._track_timestep_limits),self._ec_timestep_limits],device=self.device)
        wp.launch(kernel=kernels.lts_cell_dt,dim=(self.ny,self.nx),
            inputs=[self._ec_cell_component_offsets,self._ec_local_dt,self._ec_lts_cell_dt,
                self.dt_min_arr,cap_dt],device=self.device)
        wp.launch(kernel=kernels.lts_face_dt,dim=(self.ny+1,self.nx+1),
            inputs=[self._ec_lts_cell_dt,self.boundary_type_qx,self.boundary_type_qy,
                self._lts_face_dt_qx,self._lts_face_dt_qy,WP_FLOAT(cap_dt)],device=self.device)

    def _prepare_lts_schedule(self, **kwargs):
        result=super()._prepare_lts_schedule(**kwargs)
        if self._connection_count:
            wp.launch(kernel=kernels.lts_connection_levels,dim=self._connection_count,
                inputs=[self._ec_endpoints,self._ec_state_cell,self._lts_cell_levels,self._ec_lts_levels],device=self.device)
        return result

    def _enhanced_lts_finalize(self, *, cap_dt,max_levels):
        wp.launch(kernel=kernels.lts_finalize,dim=1,
            inputs=[self._ec_control,cap_dt,float(self.minimum_dt),max_levels,self._lts_schedule_control],device=self.device)

    def _lts_launch_sources_all(self, *, schedule,interval_start,interval_end,rainfall_hyetograph):
        if self.apply_sources_after_momentum:
            rain=self._lts_hyetograph_rate(hyetograph=rainfall_hyetograph,time_s=interval_start)
        else:
            rain=self._lts_hyetograph_mean_rate(hyetograph=rainfall_hyetograph,
                start_s=interval_start,end_s=interval_end)
        wp.launch(kernel=kernels.lts_forcing,dim=1,
            inputs=[self._ec_control,float(schedule['base_dt']),rain],device=self.device)
        self._enqueue_lts_sources(use_hyetograph=rainfall_hyetograph is not None)

    def _enqueue_lts_sources(self, *, use_hyetograph):
        self._component_sources(use_hyetograph=int(use_hyetograph),
            committed=self.apply_sources_after_momentum)
        if self._component_count:
            wp.launch(kernel=kernels.lts_accumulate_sources,dim=(self._component_count,3),
                inputs=[self._ec_applied,self._ec_lts_applied],device=self.device)
        if self.apply_sources_after_momentum:self._transport_component_interval()

    def _lts_refresh_stage_from_effective_depth(self):
        self._component_stages();self._aggregate_components()

    def _lts_launch_momentum(self, schedule, level, dt):
        if level!=0:return
        wp.launch(kernel=kernels.set_control,dim=1,
            inputs=[self._ec_control,float(schedule['base_dt'])],device=self.device)
        self._enqueue_lts_momentum()

    def _enqueue_lts_momentum(self):
        self._component_stages();self._component_coefficients()
        if self._connection_count:
            wp.launch(kernel=kernels.lts_momentum,dim=self._connection_count,
                inputs=[self._ec_d,self._ec_r,self._ec_forcing,self._ec_q,self._ec_previous_d,
                    self._ec_lts_levels,self._ec_control,float(self.gravity),self._ec_status,self._kinetic_transport,
                    self._ec_endpoints,self._ec_sill,self._ec_eta,self._ec_heads,self._ec_kind,
                    float(self.dry_depth),self._ec_lts_updates,self._exact_momentum],device=self.device)

    def _lts_accept_level_momentum(self, schedule, level):
        # Each reduced connection reads only its own previous Q. The due kernel
        # can therefore publish it in place without a lagged transverse Q field.
        return

    def _lts_limit_held_outflow_all(self, schedule):
        if self._connection_count:wp.copy(self._ec_proposed,self._ec_q)
        safety=1.000000000000002 if NP_FLOAT==np.float64 else 1.000001
        if self._component_count:
            wp.launch(kernel=kernels.demands,dim=self._component_count,
                inputs=[self._ec_volume,self._ec_adjacency_offsets,self._ec_adjacency,self._ec_proposed,
                    self._ec_alpha,self._ec_control,safety,self._ec_status,self._ec_kind,
                    self._ec_alpha_free,self.free_outflow_limiter_mode],device=self.device)
        if self._connection_count:
            wp.launch(kernel=kernels.limit,dim=self._connection_count,
                inputs=[self._ec_endpoints,self._ec_proposed,self._ec_alpha,self._ec_q,self._ec_control,
                    self.cumulative_totals,self._ec_status,self._ec_kind,self._ec_alpha_free],device=self.device)

    def _lts_transport_held_faces_all(self, schedule):
        if not self.apply_sources_after_momentum:self._transport_component_interval()

    def _transport_component_interval(self):
        if self._component_count:
            wp.launch(kernel=kernels.continuity,dim=self._component_count,
                inputs=[self._ec_volume,self._ec_adjacency_offsets,self._ec_adjacency,self._ec_q,
                    self._ec_control,self._ec_status,0,self._ec_storage_offsets,self._ec_storage_rows,
                    self._ec_eta,self._ec_area,self._ec_maximum_eta,
                    wp.float64(self._enhanced_geometry.maximum_stage),int(self.track_max_water_depth)],device=self.device)
        wp.launch(kernel=kernels.lts_next_tick,dim=1,inputs=[self._ec_control],device=self.device)
        self._component_stages(track=self.track_max_water_depth);self._aggregate_components()
        self._accumulate_max_water_depth()

    def _lts_launch_continuity(self, schedule, level):
        # Effective component volumes already include every base-clock flux.
        return

    def _finish_lts_source_accounting(self):
        # Accepted component sources are already in the external ledger once.
        wp.copy(self._ec_applied,self._ec_lts_applied);self._aggregate_components()

    def _lts_build_subcycle_graph(self, schedule, use_uniform_rain):
        if not wp.get_device(self.device).is_cuda:return False
        with wp.ScopedDevice(self.device):
            if not wp.is_conditional_graph_supported():return False
        if not self._component_count:return False
        values=getattr(self,'_ec_coefficient_values',self._ec_rows)
        indices=getattr(self,'_ec_coefficient_stage_index',self._ec_adjacency)
        wp.overload(kernels.coefficients,{'values':wp.array2d(dtype=values.dtype),
            'indices':wp.array(dtype=indices.dtype)})
        wp.load_module(module=kernels.coefficients.module,device=self.device)
        shape=(id(self._ec_q),id(self._ec_rows),id(values),id(self._lts_schedule_control),
            id(self._ec_boundary_override),id(self._ec_boundary_head_values),
            bool(self._ec_has_boundary_override),bool(self.apply_sources_after_momentum),bool(use_uniform_rain))
        if self._lts_subcycle_graph is not None and self._lts_subcycle_graph_shape==shape:return True
        with wp.ScopedCapture(device=self.device,force_module_load=True) as body:
            wp.launch(kernel=kernels.lts_graph_prepare,dim=1,
                inputs=[self._ec_control,self._lts_schedule_control,self._lts_graph_forcing],device=self.device)
            if not self.apply_sources_after_momentum:self._enqueue_lts_sources(use_hyetograph=use_uniform_rain)
            self._enqueue_lts_momentum()
            self._lts_limit_held_outflow_all(schedule=schedule)
            if self.apply_sources_after_momentum:self._enqueue_lts_sources(use_hyetograph=use_uniform_rain)
            else:self._transport_component_interval()
            wp.launch(kernel=kernels.lts_graph_finish,dim=1,
                inputs=[self._ec_control,self._lts_graph_tick,self._lts_graph_condition,self._ec_status],device=self.device)
        with wp.ScopedCapture(device=self.device) as loop:
            wp.capture_while(self._lts_graph_condition,while_body=body.graph)
        self._lts_subcycle_graph=loop.graph;self._lts_subcycle_graph_shape=shape
        return True

    def set_face_fluxes(self, qx, qy):
        """Initialize underdetermined group Q by minimum kinetic energy per face.

        With positive d, Q_k = Q_total (1/d_k)/sum(1/d) minimizes
        sum(d_k Q_k^2)/2 subject to the requested signed face discharge.
        A wholly dry face uses geometric contact widths as its initial split;
        the ordinary dry momentum update then removes incompatible discharge.
        Exact restart uses capture_state, which preserves every group Q.
        """
        if np.shape(qx)!=(self.ny,self.nx+1) or np.shape(qy)!=(self.ny+1,self.nx):
            raise ValueError('Coarse staggered flux shapes required')
        if not np.all(np.isfinite(qx)) or not np.all(np.isfinite(qy)):
            raise ValueError('Finite face fluxes required')
        self._component_stages();self._component_coefficients();self._check_component_status()
        coarse=np.r_[np.asarray(qx).ravel(),np.asarray(qy).ravel()]*float(self.dx)
        g=self._enhanced_geometry;d=self._ec_d.numpy()
        faces=self._ec_face.numpy();sign=self._ec_sign.numpy();q=self._ec_q.numpy()
        for face in range(len(coarse)):
            links=np.flatnonzero(faces==face)
            if not len(links):
                if coarse[face]!=0.:raise ValueError('Nonzero coarse flux has no active native face contacts')
                continue
            active=d[links]>0.;weights=np.zeros(len(links))
            if np.any(active):weights[active]=1./d[links[active]]
            else:weights=g.width[links].copy()
            q[links]=sign[links]*coarse[face]*weights/weights.sum()
        super().set_face_fluxes(qx=qx,qy=qy)
        wp.copy(self._ec_q,wp.array(q,dtype=WP_FLOAT,device=self.device));self._aggregate_components()

    @property
    def current_infiltration_capacity_m_per_s(self):
        if self.infiltration_method not in (1,3):return self.infiltration_capacity.numpy()
        g=self._enhanced_geometry;cells=g.state_cell
        F=np.maximum(self._ec_soil.numpy(),0.)
        if self.infiltration_method==1:
            f0=self.infiltration_f0.numpy().ravel()[cells]
            fd=np.maximum(self.infiltration_fd.numpy().ravel()[cells],1e-30)
            k=self.infiltration_k.numpy().ravel()[cells]
            value=k+(f0-k)*np.exp(-F/fd)
        else:
            ks=self.green_ampt_ks.numpy().ravel()[cells]
            suction=self.green_ampt_psi.numpy().ravel()[cells]*self.green_ampt_delta_theta.numpy().ravel()[cells]
            value=ks.copy();wet=(suction>0)&(F>0)
            value[wet]*=1.+suction[wet]/F[wet]
            value[(suction>0)&(F==0)&(ks>0)]=np.inf
        return (np.bincount(cells,weights=value*g.source_area,minlength=self.ny*self.nx)/
                (float(self.dx)*float(self.dy))).reshape(self.ny,self.nx)

    def _coarse_volume_exchange(self, *, requested, allow_sink, ledger):
        """Apply a caller's coarse actuator without pooling hydraulic storage.

        An unlocated positive source covers owned plan area; an unlocated sink
        shares demand in proportion to each component's available volume. No
        component supplies another's deficit. Explicit component forcing can
        instead use the component state API. Coupling calls synchronize just
        as the established production actuators do.
        """
        change=np.asarray(requested,dtype=np.float64)
        if change.shape!=(self.ny,self.nx) or not np.all(np.isfinite(change)):
            raise ValueError('Finite coarse volume field required')
        if not allow_sink and np.any(change<0):raise ValueError('Positive volumes required')
        if np.any(change[self._active_mask_host!=1]!=0):raise ValueError('Volume exchange targets inactive cells')
        g=self._enhanced_geometry;before=self._ec_volume.numpy();after=before.astype(np.float64).copy()
        for cell,amount in enumerate(change.ravel()):
            lo,hi=g.cell_component_offsets[cell:cell+2]
            if lo==hi or amount==0.:continue
            if amount>0:
                weights=g.source_area[lo:hi]/g.source_area[lo:hi].sum()
                after[lo:hi]+=amount*weights
            else:
                total=after[lo:hi].sum()
                if total>0.:after[lo:hi]*=1.-min(-amount/total,1.)
        represented=after.astype(NP_FLOAT)
        for state,v in enumerate(represented):
            curve=g.storage_rows[g.storage_offsets[state]:g.storage_offsets[state+1]]
            if storage_stage(volume=float(v),curve=curve)[0]>g.maximum_stage:
                raise StageRangeError('External volume exceeds preprocessed stage range; no state modified')
        actual=np.bincount(g.state_cell,weights=represented.astype(np.float64)-before.astype(np.float64),
            minlength=self.ny*self.nx).reshape(self.ny,self.nx)
        wp.copy(self._ec_volume,wp.array(represented,dtype=WP_FLOAT,device=self.device))
        if ledger is not None:
            totals=self.cumulative_totals.numpy();totals[ledger]+=float(actual.sum())*(1. if ledger==2 else -1.)
            wp.copy(self.cumulative_totals,wp.array(totals,dtype=wp.float64,device=self.device))
        self._component_stages(track=self.track_max_water_depth)
        self._check_component_status();self._aggregate_components()
        if np.any(change>0):self._accumulate_max_water_depth()
        self._lisflood_cached_maximum_depth_m=None;self.last_timestep_max_depth_m=None
        return actual






















    def subgrid_memory_report(self):
        arrays=[value for name,value in vars(self).items() if name.startswith('_ec_') and isinstance(value,wp.array)]
        return dict(enabled=True,factor=self._subgrid_factor,
                    device_bytes=sum(a.size*wp.types.type_size_in_bytes(a.dtype) for a in arrays))
