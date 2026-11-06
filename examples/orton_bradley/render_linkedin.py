"""A 45-second, full-frame social film from the validated instantaneous event.

Only saved h(t) is animated. No hydraulic solver is constructed or advanced.
"""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import os
import struct
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator
from scipy.spatial import ConvexHull

from examples.orton_bradley.export_visualisation import atomic_json
from examples.orton_bradley.reproduce import file_hash
from examples.orton_bradley.remote_rendering import create_render_state, render_image
from examples.orton_bradley.render_visualisation import encode_video, save_image, water_colormap


def audit_sources(*, dataset, aerial, output, wet_threshold):
    paths = dict(dataset=dataset/'metadata/dataset.json', terrain=dataset/'metadata/terrain.npz',
        validation=dataset/'diagnostics/validation.json', frame_manifest=dataset/'diagnostics/frame_manifest.json',
        diagnostics=dataset/'diagnostics/ledger.json', aerial=aerial/'data/aerial.json',
        aerial_npz=aerial/'data/aerial.npz', interactive_manifest=aerial/'data/manifest.json',
        conference=dataset/'renders/conference/orton_bradley_fv1_3d.mp4',
        execution=dataset/'metadata/last_execution.json')
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing existing {name} asset: {path}')
    hashes = {name: file_hash(path=path) for name,path in paths.items()}
    identity=json.loads(paths['dataset'].read_text())
    validation=json.loads(paths['validation'].read_text())
    records=json.loads(paths['frame_manifest'].read_text())
    ledger=json.loads(paths['diagnostics'].read_text())
    imagery=json.loads(paths['aerial'].read_text())
    interactive=json.loads(paths['interactive_manifest'].read_text())
    if validation['status']!='passed' or validation['final_time_s']!=259200.:
        raise ValueError('The local native event has not passed full 72-hour validation')
    expected=np.arange(0.,259201.,300.)
    np.testing.assert_array_equal(actual=[row['time_s'] for row in records],desired=expected)
    np.testing.assert_array_equal(actual=[row['time_s'] for row in ledger],desired=np.arange(60.,259201.,60.))
    if imagery['grid']!=identity['grid'] or imagery['array_row_order']!=identity['array_row_order']:
        raise ValueError('Aerial and terrain geographic alignment/orientation differ')
    if interactive['configuration']['wet_threshold_m']!=wet_threshold:
        raise ValueError('Native display threshold differs')
    with np.load(file=paths['terrain'],allow_pickle=False) as saved:
        active,dem=saved['active'],saved['dem']
    with np.load(file=paths['aerial_npz'],allow_pickle=False) as saved:
        rgb,covered=saved['rgb'],saved['covered']
    if rgb.shape!=(*dem.shape,3) or covered.shape!=dem.shape or rgb.dtype!=np.uint8:
        raise ValueError('Prepared aerial arrays do not match the native DEM')
    if hashlib.sha256(rgb.tobytes()+covered.astype(np.uint8).tobytes()).hexdigest()!=imagery['resource']['raw_sha256']:
        raise ValueError('Prepared NPZ aerial colours differ from their tested resource identity')
    if np.any(active & ~covered):
        raise ValueError('Prepared imagery does not cover the complete active domain')
    # Scan native instantaneous fields, never the maximum-depth envelope.
    wet_counts=[]
    for index,row in enumerate(records):
        path=dataset/row['path']
        if not path.is_file():
            raise FileNotFoundError('Missing native instantaneous snapshot: '+str(path))
        if file_hash(path=path)!=row['sha256']:
            raise ValueError('Native instantaneous frame checksum differs: '+str(path))
        with np.load(file=path,allow_pickle=False) as saved:
            depth=saved['depth']
            if (float(saved['time_s'])!=row['time_s'] or depth.shape!=dem.shape or depth.dtype!=np.float64
                    or not np.isfinite(depth).all() or np.any(depth<0.) or np.any(depth[~active]!=0.)):
                raise ValueError('Invalid native instantaneous field: '+str(path))
            wet_counts.append(int(np.count_nonzero(active & (depth>=wet_threshold))))
        if interactive['frames'][index]['source_sha256']!=row['sha256']:
            raise ValueError('Tested display manifest belongs to another event')
    peak_row=ledger[int(np.argmax([row['discharge_m3_s'] for row in ledger]))]
    peak_flow_time=float(expected[int(np.argmin(np.abs(expected-peak_row['time_s'])))])
    peak_extent_time=float(expected[int(np.argmax(wet_counts))])
    report=dict(status='passed', paths={name:str(path.resolve()) for name,path in paths.items()},sha256=hashes,
        validation=validation, native_dataset_identity=identity,
        hydraulic_source_revision=json.loads((dataset/'metadata/last_execution.json').read_text())['source_commit'],
        frame_count=len(records),frame_manifest=records, wet_threshold_m=wet_threshold,
        peak_flow_diagnostic_time_s=peak_row['time_s'],peak_flow_snapshot_time_s=peak_flow_time,
        peak_flow_m3_s=peak_row['discharge_m3_s'],peak_sampled_extent_time_s=peak_extent_time,
        peak_sampled_extent_m2=max(wet_counts)*identity['grid']['dx']*identity['grid']['dy'],
        aerial_metadata=imagery, aerial_fallback=False,source_files_unchanged=True)
    atomic_json(path=output/'source_audit.json',value=report)
    return report


def schedule_frames(*, saved_times, duration, fps, control_video, control_simulation, mandatory_times):
    count=int(round(duration*fps))
    if count!=duration*fps or count<2:
        raise ValueError('Duration must give an integer number of video frames')
    video_times=np.arange(count,dtype=np.float64)/fps
    production_times=video_times*(control_video[-1]/duration)
    ideal=np.interp(production_times,control_video,control_simulation)
    indices=np.searchsorted(saved_times,ideal,side='right')-1
    indices=np.clip(indices,0,len(saved_times)-1)
    for target in np.unique(mandatory_times):
        source=int(np.searchsorted(saved_times,target))
        if source>=len(saved_times) or saved_times[source]!=target:
            raise ValueError('Mandatory native snapshot is missing')
        if not np.any(indices==source):
            position=int(np.argmin(np.abs(ideal-target)))
            indices[position]=source
    if np.any(np.diff(indices)<0):
        raise ValueError('Timeline would reverse the event')
    return [dict(index=i,video_time_s=float(video_times[i]),production_time_s=float(production_times[i]),
        source_index=int(indices[i]),time_s=float(saved_times[indices[i]])) for i in range(count)]


def camera_pose(*, production_time, camera):
    values={name:float(PchipInterpolator(x=camera['time_s'],y=data)(production_time))
            for name,data in camera.items() if name!='time_s'}
    return values


def fonts_for_size(*, width, regular_path, bold_path):
    scale=width/1920.
    return {name:ImageFont.truetype(font=str(path),size=max(8,int(round(size*scale))))
        for name,size,path in [('title',54,bold_path),('subtitle',27,regular_path),('clock',38,bold_path),
            ('label',18,regular_path),('qualification',21,regular_path),('credit',17,regular_path)]}


def overlay_frame(*, image, row, profile, fonts):
    width,height=image.size
    scale=width/1920.
    overlay=Image.new(mode='RGBA',size=image.size,color=(0,0,0,0))
    draw=ImageDraw.Draw(im=overlay)
    for y in range(int(240*scale)):
        alpha=int(150*(1-y/(240*scale)))
        draw.line(xy=[(0,y),(width,y)],fill=(5,12,17,alpha))
    for y in range(int(210*scale)):
        alpha=int(190*y/(210*scale))
        draw.line(xy=[(0,height-int(210*scale)+y),(width,height-int(210*scale)+y)],fill=(5,12,17,alpha))
    margin=int(54*scale)
    opening=max(0.,min(1.,(profile.get('opening_fade_end_s',8.)-row['production_time_s'])
        /profile.get('opening_fade_duration_s',2.)))
    ending=max(0.,min(1.,(row['production_time_s']-profile.get('closing_start_s',42.))
        /profile.get('closing_fade_duration_s',1.)))
    if opening>0.:
        draw.text(xy=(margin,int(42*scale)),text=profile['opening'],font=fonts['title'],fill=(242,248,250,int(255*opening)))
        draw.text(xy=(margin,int(112*scale)),text=profile['subtitle'],font=fonts['subtitle'],fill=(222,237,242,int(255*opening)))
    if ending>0.:
        draw.text(xy=(margin,int(42*scale)),text=profile['closing'],font=fonts['title'],fill=(242,248,250,int(255*ending)))
        draw.text(xy=(margin,int(112*scale)),text='Orton Bradley · runoff to recession',font=fonts['subtitle'],fill=(222,237,242,int(255*ending)))
    seconds=row['time_s']
    clock=f'{int(seconds//3600):02d}:{int(seconds%3600//60):02d}'
    draw.text(xy=(width-margin,int(42*scale)),text='ELAPSED SIMULATION TIME',font=fonts['label'],fill='#e7f0f4',anchor='ra')
    draw.text(xy=(width-margin,int(70*scale)),text=clock,font=fonts['clock'],fill='#f4d28e',anchor='ra')
    draw.text(xy=(width-margin,int(125*scale)),text='Variable playback speed · 5-minute snapshots',font=fonts['label'],fill='#e1ebef',anchor='ra')
    draw.text(xy=(width-margin,int(154*scale)),text='Vertical scale 1:1',font=fonts['label'],fill='#b7cbd5',anchor='ra')
    legend_width=int(350*scale)
    legend_y=height-int(226*scale)
    draw.rounded_rectangle(xy=(margin-int(12*scale),legend_y-int(36*scale),margin+legend_width+int(16*scale),legend_y+int(59*scale)),
        radius=int(7*scale),fill=(7,20,28,205))
    draw.text(xy=(margin,legend_y-int(29*scale)),text='Water depth (m, logarithmic)',font=fonts['label'],fill='#f0f5f7')
    rgba=(water_colormap()(np.linspace(0.,1.,legend_width))[:,:3]*255).astype(np.uint8)
    bar=Image.fromarray(obj=np.repeat(rgba[np.newaxis],max(1,int(13*scale)),axis=0))
    overlay.paste(im=bar,box=(margin,legend_y))
    minimum,maximum=profile['wet_threshold_m'],profile['depth_maximum_m']
    for value in (minimum,.1,1.,maximum):
        x=margin+legend_width*np.log(value/minimum)/np.log(maximum/minimum)
        draw.text(xy=(x,legend_y+int(22*scale)),text=f'{value:g}',font=fonts['label'],fill='#d7e7ed',anchor='ma')
    draw.text(xy=(margin,height-int(140*scale)),text=profile['qualification'],font=fonts['qualification'],fill='#f0f5f7')
    for line,y in zip(profile['credits'],(91,63,35)):
        draw.text(xy=(margin,height-int(y*scale)),text=line,font=fonts['credit'],fill='#c9d9e1')
    return Image.alpha_composite(im1=image.convert('RGBA'),im2=overlay).convert('RGB')


def render_scene(*, state, dataset, row, profile):
    pose=camera_pose(production_time=row['production_time_s'],camera=profile['camera'])
    state['centre'][2]=pose['target_z']
    view=dict(frame=row['source_index'],width=profile['width'],height=profile['height'],
        azimuth=pose['azimuth'],elevation=pose['elevation'],distance=pose['distance'],
        east=pose['target_x']-state['centre'][0],north=pose['target_y']-state['centre'][1],
        opacity=profile['water_opacity'],aerial=1)
    result=render_image(state=state,dataset=dataset,view=view)
    if result['time_s']!=row['time_s']:
        raise ValueError('Rendered field timestamp differs from the schedule')
    return result


def render_one(*, state, dataset, row, profile, fonts):
    result=render_scene(state=state,dataset=dataset,row=row,profile=profile)
    image=Image.open(fp=BytesIO(result['image'])).convert('RGB')
    return overlay_frame(image=image,row=row,profile=profile,fonts=fonts)


def verify_scene_updates(*, state, dataset, schedule, profile, peak_time):
    peak=next(row.copy() for row in schedule if row['time_s']==peak_time)
    dry=schedule[0].copy()
    dry['production_time_s']=peak['production_time_s']
    moved=peak.copy()
    moved['production_time_s']=profile['timeline_video_s'][-1]
    results=[render_scene(state=state,dataset=dataset,row=row,profile=profile) for row in (dry,peak,moved)]
    arrays=[np.asarray(Image.open(fp=BytesIO(result['image']))) for result in results]
    hydraulic_change=float(np.mean(np.any(arrays[0]!=arrays[1],axis=2)))
    camera_change=float(np.mean(np.any(arrays[1]!=arrays[2],axis=2)))
    if results[0]['wet_triangles']!=0 or results[1]['wet_triangles']<=0 or hydraulic_change==0. or camera_change==0.:
        raise ArithmeticError('Water or independent camera update failed')
    return dict(dry_triangles=results[0]['wet_triangles'],peak_triangles=results[1]['wet_triangles'],
        hydraulic_changed_pixel_fraction=hydraulic_change,camera_changed_pixel_fraction=camera_change,
        method='Raw scene, fixed camera with dry/peak fields, then fixed peak field with moved camera')


def verify_camera_path(*, state, schedule, profile):
    # Perspective extrema occur on the convex hull when all points are in front.
    # Check every scheduled camera, without projecting a million points per frame.
    points=state['points'][np.unique(state['triangles'])]
    points=points[ConvexHull(points=points).vertices]
    bounds=np.array([np.inf,-np.inf,np.inf,-np.inf])
    for row in schedule:
        pose=camera_pose(production_time=row['production_time_s'],camera=profile['camera'])
        azimuth,elevation=np.deg2rad([pose['azimuth'],pose['elevation']])
        offset=np.array([np.cos(azimuth)*np.cos(elevation),np.sin(azimuth)*np.cos(elevation),np.sin(elevation)])
        forward=-offset
        right=np.cross(forward,[0.,0.,1.])
        right/=np.linalg.norm(right)
        up=np.cross(right,forward)
        eye=np.array([pose['target_x'],pose['target_y'],pose['target_z']])+pose['distance']*offset
        delta=points-eye
        depth=delta@forward
        if np.any(depth<=0.) or eye[2]<=points[:,2].max()+profile['depth_maximum_m']:
            raise ValueError('Camera could enter or pass behind the model')
        half_height=np.tan(np.deg2rad(state['plotter'].camera.view_angle/2.))
        x=(delta@right)/(depth*half_height*profile['width']/profile['height'])
        y=(delta@up)/(depth*half_height)
        bounds=np.array([min(bounds[0],x.min()),max(bounds[1],x.max()),
            min(bounds[2],y.min()),max(bounds[3],y.max())])
    if np.any(np.abs(bounds)>=1.):
        raise ValueError('Camera path clips the native terrain: '+str(bounds))
    return dict(all_scheduled_cameras_checked=len(schedule),terrain_fully_in_frame=True,
        camera_above_model=True,normalised_viewport_bounds=bounds.tolist())


def verify_mp4(*, path, expected_duration, fps):
    import imageio_ffmpeg
    reader=imageio_ffmpeg.read_frames(path=str(path))
    metadata=next(reader)
    reader.close()
    if abs(metadata['duration']-expected_duration)>1.e-3 or metadata['codec']!='h264' or not metadata['pix_fmt'].startswith('yuv420p'):
        raise ValueError('MP4 duration, codec or pixel format differs')
    header=subprocess.run(args=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-i',str(path)],capture_output=True,text=True)
    if 'Audio:' in header.stderr:
        raise ValueError('Unexpected audio stream')
    boxes={}
    with path.open('rb') as stream:
        while stream.tell()<path.stat().st_size:
            offset=stream.tell()
            raw=stream.read(8)
            if len(raw)!=8:
                break
            size,kind=struct.unpack('>I4s',raw)
            if size==1:
                size=struct.unpack('>Q',stream.read(8))[0]
            if size==0:
                size=path.stat().st_size-offset
            if size<8:
                raise ValueError('Invalid MP4 box')
            boxes[kind.decode('ascii',errors='replace')]=offset
            stream.seek(offset+size)
    if not boxes.get('moov',float('inf'))<boxes.get('mdat',0):
        raise ValueError('MP4 is not faststart')
    return dict(container_duration_s=metadata['duration'],codec=metadata['codec'],pixel_format=metadata['pix_fmt'],
        constant_frame_rate=fps,faststart=True,audio_streams=0)


def contact_sheet(*, root, schedule, selections, destination):
    thumbnails=[]
    indices={}
    for label,target in selections.items():
        index=int(np.argmin([abs(row['time_s']-target) for row in schedule]))
        indices[label]=index
        with Image.open(fp=root/'images'/f'{index:06d}.png') as image:
            thumb=image.resize(size=(640,360),resample=Image.Resampling.LANCZOS)
        tile=Image.new(mode='RGB',size=(640,402),color='#0b1720')
        tile.paste(im=thumb,box=(0,0))
        draw=ImageDraw.Draw(im=tile)
        row=schedule[index]
        draw.text(xy=(12,373),text=f'{label} | video {row["video_time_s"]:.2f}s | simulation {row["time_s"]/3600.:.3f}h',fill='white')
        thumbnails.append(tile)
    sheet=Image.new(mode='RGB',size=(1920,402*int(np.ceil(len(thumbnails)/3))),color='#0b1720')
    for index,tile in enumerate(thumbnails):
        sheet.paste(im=tile,box=((index%3)*640,(index//3)*402))
    save_image(image=sheet,path=destination)
    return indices


def run_render(*, dataset, aerial, output, audit, profile, movie_path, regular_font, bold_font, review_only):
    saved_times=np.array([row['time_s'] for row in audit['frame_manifest']])
    required=[0.,72000.,audit['peak_flow_snapshot_time_s'],audit['peak_sampled_extent_time_s'],172800.,259200.]
    schedule=schedule_frames(saved_times=saved_times,duration=profile['duration_s'],fps=profile['fps'],
        control_video=profile['timeline_video_s'],control_simulation=profile['timeline_simulation_s'],mandatory_times=required)
    root=output/profile['name']
    root.mkdir(parents=True,exist_ok=True)
    source_root=Path(__file__).resolve().parents[2]
    renderer_files=[Path(__file__),source_root/'examples/orton_bradley/render_visualisation.py',
        source_root/'examples/orton_bradley/remote_rendering.py']
    identity=dict(profile=profile,sources=audit['sha256'],
        frame_hashes={str(row['time_s']):audit['frame_manifest'][row['source_index']]['sha256'] for row in schedule},
        renderer_hashes={str(path.relative_to(source_root)):file_hash(path=path) for path in renderer_files},
        fonts={str(regular_font):file_hash(path=regular_font),str(bold_font):file_hash(path=bold_font)},
        schedule=schedule, vertical_scale=1., spatial_interpolation='Native cell-centre triangles, all three vertices wet',
        hydraulic_time_interpolation='None; saved snapshot held',water_geometry='z=DEM+h',
        source_revision=subprocess.run(args=['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip())
    identity_path=root/'render_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise ValueError('Render identity changed; preserve completed images and choose a new profile output name')
    atomic_json(path=identity_path,value=identity)
    selected=dict(start=0.,rising=72000.,peak_extent=audit['peak_sampled_extent_time_s'],
                  peak_flow=audit['peak_flow_snapshot_time_s'],recession=172800.,final=259200.)
    if review_only:
        chosen=sorted(set(int(np.argmin([abs(row['time_s']-target) for row in schedule])) for target in selected.values()))
    else:
        chosen=range(len(schedule))
    fonts=fonts_for_size(width=profile['width'],regular_path=regular_font,bold_path=bold_font)
    started=time.perf_counter()
    state=create_render_state(dataset=dataset,directory=aerial,width=profile['width'],height=profile['height'])
    try:
        scene_check=verify_scene_updates(state=state,dataset=dataset,schedule=schedule,profile=profile,
            peak_time=audit['peak_sampled_extent_time_s'])
        if (state['manifest']['configuration']['depth_maximum_m']!=profile['depth_maximum_m']
                or state['manifest']['configuration']['wet_threshold_m']!=profile['wet_threshold_m']):
            raise ValueError('Legend limits differ from the established water renderer')
        camera_check=verify_camera_path(state=state,schedule=schedule,profile=profile)
        for index in chosen:
            path=root/'images'/f'{index:06d}.png'
            if path.exists():
                with Image.open(fp=path) as image:
                    if image.size!=(profile['width'],profile['height']):
                        raise ValueError('Retained frame dimensions differ')
                    image.verify()
                continue
            image=render_one(state=state,dataset=dataset,row=schedule[index],profile=profile,fonts=fonts)
            save_image(image=image,path=path)
            if review_only or index%60==0:
                print(f'{profile["name"]}: {index+1}/{len(schedule)} frames; {time.perf_counter()-started:.1f}s',flush=True)
    finally:
        state['plotter'].close()
    if review_only:
        print('Representative 3D views ready for visual review',flush=True)
        return dict(status='review',path=str(root))
    if len(list((root/'images').glob('*.png')))!=len(schedule):
        raise ValueError('Missing or duplicated video frames')
    media=encode_video(image_directory=root/'images',path=movie_path,width=profile['width'],height=profile['height'],
        fps=profile['fps'],frame_count=len(schedule),crf=profile['crf'])
    media.update(verify_mp4(path=movie_path,expected_duration=profile['duration_s'],fps=profile['fps']))
    media['render_seconds']=time.perf_counter()-started
    indices=contact_sheet(root=root,schedule=schedule,selections=selected,destination=root/'contact_sheet.png')
    report=dict(status='complete',media=media,profile=profile,identity_sha256=file_hash(path=identity_path),
        schedule=schedule,representative_frame_indices=indices,contact_sheet=str(root/'contact_sheet.png'),
        scene_update_check=scene_check,
        camera_path_check=camera_check,
        exact_wet_classification=True,unchanged_vertical_scale=1.,peak_flow_snapshot_s=audit['peak_flow_snapshot_time_s'],
        peak_extent_snapshot_s=audit['peak_sampled_extent_time_s'],timeline_adjustment=None,
        initial_peak_flow_peak_extent_final_present=all(any(row['time_s']==value for row in schedule) for value in required),
        camera_motion_independent_of_hydraulics=True,representation_limitations=[
            'Five-minute source snapshots held without temporal interpolation.',
            'All three vertices of each water triangle must be wet; narrow strips can be omitted.',
            'Opaque depth colouring improves shallow-water legibility; water height is not exaggerated.',
            'Aerial imagery is 2023 colour on the conditioned native bare-earth DEM; no building/tree geometry is added.'])
    atomic_json(path=root/'render_report.json',value=report)
    return report


def finalise_outputs(*, output, audit):
    # Recheck the original inputs after rendering, including every native snapshot.
    for name,path in audit['paths'].items():
        if file_hash(path=Path(path))!=audit['sha256'][name]:
            raise ValueError('Original source changed during rendering: '+path)
    dataset=Path(audit['paths']['dataset']).parents[1]
    for row in audit['frame_manifest']:
        if file_hash(path=dataset/row['path'])!=row['sha256']:
            raise ValueError('Original hydraulic snapshot changed during rendering')
    reports={}
    for name in ('preview','production'):
        path=output/name/'render_report.json'
        if path.exists():
            reports[name]=json.loads(path.read_text())
    cover=None
    if 'production' in reports:
        index=reports['production']['representative_frame_indices']['peak_extent']
        cover_path=output/'orton_bradley_floodwarp_linkedin_cover.jpg'
        with Image.open(fp=output/'production/images'/f'{index:06d}.png') as image:
            image.convert('RGB').save(fp=cover_path,quality=95,subsampling=0)
        cover=dict(path=str(cover_path),bytes=cover_path.stat().st_size,sha256=file_hash(path=cover_path),
            source_video_frame=index,simulation_time_s=audit['peak_sampled_extent_time_s'])
    report=dict(status='complete' if set(reports)=={'preview','production'} else 'preview_complete',
        source_audit=str(output/'source_audit.json'),source_audit_sha256=file_hash(path=output/'source_audit.json'),
        source_revision=audit['hydraulic_source_revision'],source_paths=audit['paths'],source_sha256=audit['sha256'],
        native_validation=audit['validation'],source_files_unchanged=True,reports=reports,cover=cover,
        simulation_rerun=False,new_imagery_downloaded=False,software_offscreen_rendering=True)
    atomic_json(path=output/'render_report.json',value=report)


if __name__ == "__main__":
    project_root=Path(__file__).resolve().parents[2]
    dataset=project_root/'outputs/orton_bradley_visualisation'
    aerial=project_root/'outputs/orton_bradley_interactive'
    output=dataset/'renders/linkedin'
    action=os.environ.get('FLOODWARP_LINKEDIN_ACTION','preview')  # review, preview, full
    regular_font=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    bold_font=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
    software_rendering=True
    wet_threshold=.01
    depth_maximum=7.
    preview_duration=8.
    fps=30
    crf=19
    timeline_video=[0.,6.,31.,42.,45.]
    timeline_simulation=[0.,64800.,108000.,259200.,259200.]
    camera=dict(time_s=[0.,6.,19.,31.,42.,45.],
        azimuth=[116.,118.,125.,132.,139.,140.], elevation=[36.,37.,40.,40.,38.,38.],
        distance=[8600.,8400.,7800.,8000.,9000.,9100.],
        target_x=[2000.,1950.,1800.,1800.,2000.,2000.],
        target_y=[3100.,3200.,3500.,3500.,3100.,3100.],
        target_z=[180.,170.,150.,150.,180.,180.])
    profile=dict(name='preview' if action in ('review','preview') else 'production',
        width=960 if action in ('review','preview') else 1920,
        height=540 if action in ('review','preview') else 1080,
        duration_s=preview_duration if action in ('review','preview') else 45.,fps=fps,crf=crf,
        camera=camera,timeline_video_s=timeline_video,timeline_simulation_s=timeline_simulation,
        wet_threshold_m=wet_threshold,depth_maximum_m=depth_maximum,water_opacity=1.,
        opening='FloodWarp | Orton Bradley',subtitle='A simulated storm, from runoff to recession.',
        closing='FloodWarp | Lincoln Agritech',
        qualification='Uncalibrated 100-year design rainfall simulation.',
        credits=['Patrick Durney / Lincoln Agritech Ltd · FloodWarp · AGPL-3.0-only',
            'Terrain: LINZ · Environment Canterbury / Christchurch City Council · Aerial Surveys · Landpro · AAM NZ · CC BY 4.0',
            'Aerial imagery: Environment Canterbury / Landpro · LINZ Data Service · Canterbury 2023 · CC BY 4.0'])
    if software_rendering:
        os.environ['LIBGL_ALWAYS_SOFTWARE']='1'
        os.environ['VTK_DEFAULT_OPENGL_WINDOW']='vtkEGLRenderWindow'
        os.environ['ALLOW_PLOTTING']='true'
        os.environ['MPLBACKEND']='Agg'
    output.mkdir(parents=True,exist_ok=True)
    audit=audit_sources(dataset=dataset,aerial=aerial,output=output,wet_threshold=wet_threshold)
    movie=output/('orton_bradley_floodwarp_linkedin_preview.mp4' if action in ('review','preview') else 'orton_bradley_floodwarp_linkedin_45s_1080p.mp4')
    if action not in ('review','preview','full'):
        raise ValueError('Action must be review, preview or full')
    result=run_render(dataset=dataset,aerial=aerial,output=output,audit=audit,profile=profile,
        movie_path=movie,regular_font=regular_font,bold_font=bold_font,review_only=action=='review')
    if action!='review':
        finalise_outputs(output=output,audit=audit)
    print(json.dumps({key:value for key,value in result.items() if key not in ('schedule','profile')},indent=2))
