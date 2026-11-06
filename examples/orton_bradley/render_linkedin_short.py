"""Twenty-second edit, rendered from native event snapshots rather than the old movie."""
from pathlib import Path
import json
import os

from examples.orton_bradley.reproduce import file_hash
from examples.orton_bradley.render_linkedin import audit_sources, finalise_outputs, run_render


def render_short_edit(*, dataset, aerial, output, action, profile, regular_font, bold_font,
                      movie_name, software_rendering):
    if action not in ('review','preview','full'):
        raise ValueError('Action must be review, preview or full')
    if software_rendering:
        os.environ['LIBGL_ALWAYS_SOFTWARE']='1'
        os.environ['VTK_DEFAULT_OPENGL_WINDOW']='vtkEGLRenderWindow'
        os.environ['ALLOW_PLOTTING']='true'
        os.environ['MPLBACKEND']='Agg'
    output.mkdir(parents=True,exist_ok=True)
    audit=audit_sources(dataset=dataset,aerial=aerial,output=output,
        wet_threshold=profile['wet_threshold_m'])
    result=run_render(dataset=dataset,aerial=aerial,output=output,audit=audit,profile=profile,
        movie_path=output/movie_name,regular_font=regular_font,bold_font=bold_font,
        review_only=action=='review')
    if action!='review':
        finalise_outputs(output=output,audit=audit)
    return result


if __name__ == "__main__":
    project_root=Path(__file__).resolve().parents[2]
    dataset=project_root/'outputs/orton_bradley_visualisation'
    aerial=project_root/'outputs/orton_bradley_interactive'
    output=dataset/'renders/linkedin_20s'
    action=os.environ.get('FLOODWARP_LINKEDIN_ACTION','preview')
    regular_font=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    bold_font=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
    software_rendering=True
    duration=20.
    preview_duration=6.
    fps=30
    crf=19
    timeline_video=[0.,1.5,4.,10.5,13.5,18.,20.]
    timeline_simulation=[0.,64800.,82800.,90000.,108000.,259200.,259200.]
    camera=dict(time_s=[0.,1.5,7.8,13.5,18.,20.],
        azimuth=[116.,118.,125.,132.,139.,140.],elevation=[36.,37.,40.,40.,38.,38.],
        distance=[8600.,8400.,7800.,8000.,9000.,9100.],
        target_x=[2000.,1950.,1800.,1800.,2000.,2000.],
        target_y=[3100.,3200.,3500.,3500.,3100.,3100.],
        target_z=[180.,170.,150.,150.,180.,180.])
    preview=action in ('review','preview')
    movie_name=('orton_bradley_floodwarp_linkedin_20s_preview.mp4' if preview
        else 'orton_bradley_floodwarp_linkedin_20s_1080p.mp4')
    profile=dict(name='preview' if preview else 'production',width=960 if preview else 1920,
        height=540 if preview else 1080,duration_s=preview_duration if preview else duration,
        fps=fps,crf=crf,camera=camera,timeline_video_s=timeline_video,
        timeline_simulation_s=timeline_simulation,wet_threshold_m=.01,depth_maximum_m=7.,water_opacity=1.,
        opening='FloodWarp | Orton Bradley',subtitle='A simulated storm, from runoff to recession.',
        opening_fade_end_s=2.,opening_fade_duration_s=.5,
        closing='FloodWarp | Lincoln Agritech',closing_start_s=18.,closing_fade_duration_s=.6,
        qualification='Uncalibrated 100-year design rainfall simulation.',
        credits=['Patrick Durney / Lincoln Agritech Ltd · FloodWarp · AGPL-3.0-only',
            'Terrain: LINZ · Environment Canterbury / Christchurch City Council · Aerial Surveys · Landpro · AAM NZ · CC BY 4.0',
            'Aerial imagery: Environment Canterbury / Landpro · LINZ Data Service · Canterbury 2023 · CC BY 4.0'],
        profile_source_sha256=file_hash(path=Path(__file__)))
    result=render_short_edit(dataset=dataset,aerial=aerial,output=output,action=action,profile=profile,
        regular_font=regular_font,bold_font=bold_font,movie_name=movie_name,
        software_rendering=software_rendering)
    print(json.dumps({key:value for key,value in result.items() if key not in ('schedule','profile')},indent=2))
