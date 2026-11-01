"""Edit paths and parameters below for your own projected metre GeoTIFF."""
from pathlib import Path

from floodwarp.inputs import read_terrain, read_aligned_field, boundary_faces
from floodwarp.model import create_model
from examples.common import run_example
from floodwarp.sampling import sample_targets


if __name__ == "__main__":
    terrain_path = Path('examples/purau/terrain.tif')
    roughness_path = None  # Or Path('my_manning.tif'), exactly aligned with terrain.
    vertical_datum = 'NZVD2016'  # Replace with your DEM's verified datum.
    output_path = Path('outputs/direct')
    solver_method = 'fv1'
    manning_n = .035
    rainfall_m_s = 20./3.6e6
    infiltration_m_s = 0.
    initial_depth_m = 0.
    duration_s = 20.
    sample_interval_s = 5.
    maximum_dt_s = .5
    device = 'cpu'
    make_plots = True
    use_cuda_graph = False

    terrain = read_terrain(path=terrain_path, vertical_datum=vertical_datum)
    roughness = manning_n if roughness_path is None else read_aligned_field(path=roughness_path, terrain=terrain)
    qx, qy = boundary_faces(terrain=terrain)
    qy[-1, terrain.active[-1]] = 2
    model = create_model(terrain=terrain, solver_method=solver_method, manning_n=roughness,
        rainfall=rainfall_m_s, infiltration=infiltration_m_s, initial_depth=initial_depth_m,
        boundary_qx=qx, boundary_qy=qy, device=device, explicit_maximum_dt=maximum_dt_s)
    run_example(model=model, output=output_path,
        sample_times=sample_targets(duration=duration_s, interval=sample_interval_s),
        make_plots=make_plots, use_cuda_graph=use_cuda_graph)
