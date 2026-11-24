"""Cross-check the installed advisor against the archived pre-run assessment of the same DEM."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from affine import Affine
import numpy as np
import rasterio

from floodwarp.solver_advisor import load_model, recommend
from research.floodwarp_methods_paper import metrics, runner


def run(*, directory, terrains, grids, objective):
    folder = directory/"native_features"
    observations = metrics.csv_read(file=folder/"observations.csv")
    settings = json.loads((directory/"native_feature_settings.json").read_text())
    model = load_model()
    outputs = []
    with TemporaryDirectory() as temporary:
        work = Path(temporary)
        for name, expected_loss in terrains.items():
            dem = work/(name+".tif")
            for dx in grids:
                (row,) = [r for r in observations if r["terrain_id"] == name and int(r["dx_m"]) == dx]
                path = folder/row["case"]/f"{dx}m"/row["identity"]/"pre_run.json"
                obs = json.loads(path.read_text())
                land = directory/"native_terrain"/row["case"]/obs["identity"]["physical"]/"inputs.json"
                if land.exists():
                    receipt = json.loads(land.read_text())
                else:
                    receipt = next(json.loads(p.read_text()) for p in
                                   (directory.parents[2]/".fidelity-restarts/native_inputs").glob("*/manifest.json")
                                   if runner.digest(json.loads(p.read_text())) == obs["identity"]["physical"])
                store = directory.parents[2]/".fidelity-restarts/native_inputs"/runner.digest(receipt["inputs"])
                if not dem.exists():
                    z = np.load(store/"bed_1m.npy")
                    with rasterio.open(dem, "w", driver="GTiff", width=z.shape[1], height=z.shape[0], count=1,
                                       dtype="float64", transform=Affine.translation(0,z.shape[0])*Affine.scale(1,-1),
                                       crs=settings["local_crs"], nodata=-9999) as dataset:
                        dataset.write(z[::-1],1)
                        dataset.set_band_unit(1,"m")
                ctl = work/"control.json"
                runner.write_json(path=ctl, data=dict(window_m=obs["control_window_m"], spill_head_m=settings["spill_head_m"],
                                                       axis="west_east"))
                hyd = work/"forcing.json"
                runner.write_json(path=hyd,data=obs["forcing"])
                dest = work/(name+str(dx))
                subprocess.run([sys.executable, "-I", "-m", "floodwarp.solver_advisor", "--dem", str(dem),
                                "--resolution", str(dx), "--compare-resolutions", str(dx), "--objective", objective,
                                "--output", str(dest), "--control-json", str(ctl), "--hydraulic-json", str(hyd)],
                               cwd=work, check=True,stdout=subprocess.PIPE,text=True)
                rec = json.loads((dest/"recommendation.json").read_text())["selected"]
                offline = recommend(model=model,features=obs["features"],objective=objective)
                control = rec["terrain_diagnostics"]
                loss = control["native_connected"] and not control["represented_connected"]
                if loss != expected_loss or loss != obs["features"]["critical_connection_lost"]:
                    raise AssertionError(f"Incorrect connectivity at {name} {dx} m")
                for k in ("native_spill_m","represented_spill_m","max_local_barrier_lowering_m"):
                    np.testing.assert_allclose(control[k], obs["control"][k], atol=1e-12,rtol=0)
                if (rec["status"],rec["method"]) != (offline["status"],offline["method"]):
                    raise AssertionError(f"Public and offline decisions differ at {name} {dx} m")
                outputs.append(dict(terrain_id=name, dx_m=dx, lost=loss, status=rec["status"], method=rec["method"],
                                    max_lowering_m=control["max_local_barrier_lowering_m"], native_spill_m=control["native_spill_m"],
                                    represented_spill_m=control["represented_spill_m"], model=runner.digest(model),
                                    pre_run_sha256=runner.checksum(path), native_dem_sha256=runner.checksum(dem)))
                print(name,dx,rec["status"],"connection lost",loss,flush=True)
    metrics.csv_write(file=directory/"advisor_design/installed_cli.csv",rows=outputs)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    terrains = {"floodplain-8103": True, "rolling-8105": False}
    grids = (5,10)
    objective = "peak_discharge"
    run(directory=directory,terrains=terrains,grids=grids,objective=objective)
