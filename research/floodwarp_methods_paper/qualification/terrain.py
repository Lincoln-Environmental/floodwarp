"""Control-window and full-domain terrain diagnostics on the represented hydraulic DEM."""
import numpy as np
from affine import Affine
from rasterio.crs import CRS
from scipy import ndimage

from floodwarp.terrain_advisor.diagnostics import analyse_array,terrain
from floodwarp.terrain_advisor.raster import write_maps
from research.floodwarp_methods_paper.qualification.hydraulics import represented
from research.floodwarp_methods_paper.terrain.generators.generate import array_sha


def window_bed(*, xb, yb, length, width, seed, grade, incision, landform, options):
    u, v = xb / length, yb / width
    phase = np.random.default_rng(seed=seed).uniform(low=-0.3, high=0.3)
    centre = 0.47 + 0.10 * np.sin(2 * np.pi * u + phase)
    base = grade * (length - xb)
    valley = np.exp(-(((v - centre) / 0.13) ** 2))
    if landform == "mixed":
        base = 0.001 * (length - xb) + grade * np.maximum(0.48 * length - xb, 0.0)
    z = base - incision * valley
    if landform == "plain":
        z -= 0.035 * np.exp(-(((u - 0.65) / 0.12) ** 2) - ((v - 0.65) / 0.18) ** 2)
    elif landform == "floodplain":
        # Overtoppable embankment interrupted by a low saddle.
        z += (
            0.10
            * np.exp(-(((u - 0.7) / 0.045) ** 2))
            * (1.0 - 0.93 * np.exp(-(((v - 0.55) / 0.11) ** 2)))
        )
    elif landform == "coastal":
        z -= 0.055 * np.exp(-(((u - 0.75) / 0.19) ** 2) - ((v - 0.35) / 0.19) ** 2)
    elif landform == "rolling":
        z += 0.17 * np.sin(3 * np.pi * u) * np.cos(2 * np.pi * v + phase)
    elif landform == "fan":
        z += 0.10 * np.cos(3 * np.pi * (v - 0.5) / (u + 0.5)) * np.sin(np.pi * u)
        z -= 0.07 * np.exp(-(((v - 0.45 - 0.3 * u) / 0.07) ** 2))
    elif landform in ("hill", "incised", "mixed"):
        z += (0.014 if landform == "hill" else 0.032) * width * (v - centre) ** 2
        z += (
            0.15
            * np.exp(-(((u - 0.61) / 0.05) ** 2))
            * (1.0 - np.exp(-(((v - centre) / 0.045) ** 2)))
        )
    shoulder=options["family_control_halfwidth_m"]
    height=options["control_bank_height_m"]
    gap=np.exp(-((yb-width*centre)/shoulder)**2)
    bank=np.exp(-((xb-0.875*length)/shoulder)**2)
    return z+height*bank*(1-gap)-height*gap*np.exp(-((u-0.875)/0.06)**2)


def connection(*, elevation, level):
    """Four-neighbour wet cells across opposing faces, consistent with face fluxes."""
    mask=elevation<=level
    labels,_=ndimage.label(mask)
    west=np.unique(labels[:,0])
    return bool(np.intersect1d(west[west>0],labels[:,-1]).size)


def measure(*,spec,config,template,options,directory):
    length,width=spec["dimensions"]
    dx=spec["dx"]
    (z,_,_,_,_),params=represented(options=options,settings=spec,design=config["synthetic"],template=template,
        family=spec["landform"],seed=spec["seed"],initial=spec["initial"],dx=dx,dem=dx)
    native_hash=array_sha(z)
    domain,_=terrain(z=z[::-1],dx=dx,areas=np.full_like(z,dx**2),
        channel_area=max(100.,0.02*length*width),window_m=30.)
    phase=np.random.default_rng(params["phase_seed"]).uniform(-0.3,0.3)
    y0=width*(0.47+0.10*np.sin(2*np.pi*0.875+phase))
    # Common 80 m square aligned to 10 m edges around the known opening.
    side=80
    west=int((0.875*length-side/2)//10)*10
    south=int((y0-side/2)//10)*10
    rows,cols=np.indices((side,side))
    xx=west+cols+.5
    yy=south+side-rows-.5
    native=window_bed(xb=xx,yb=yy,length=length,width=width,seed=params["phase_seed"],
                      grade=params["grade"],incision=params["incision_m"],landform=spec["landform"],options=options)
    crs=CRS.from_wkt('LOCAL_CS["FloodWarp synthetic metre grid",LOCAL_DATUM["Local",0],UNIT["metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]')
    transform=Affine.translation(west,south+side) @ Affine.scale(1,-1)
    result={}
    for grid in (1,5,10):
        jj,ii=np.indices((side//grid,side//grid))
        actual=window_bed(xb=west+(ii+.5)*grid,yb=south+side-(jj+.5)*grid,length=length,width=width,
            seed=params["phase_seed"],grade=params["grade"],incision=params["incision_m"],landform=spec["landform"],options=options)
        if grid==dx:
            j=south//grid
            i=west//grid
            np.testing.assert_allclose(actual,z[j:j+side//grid,i:i+side//grid][::-1],rtol=1e-13,atol=1e-12)
        summary,maps,coarse=analyse_array(elevation=native,native_resolution=1.,resolution=grid,represented_elevation=actual)
        # Lowest stage that spans the native window and an independently fixed 1 cm head margin.
        low,high=float(native.min()),float(native.max())
        for iteration in range(48):
            mid=(low+high)*.5
            if connection(elevation=native,level=mid):
                high=mid
            else:
                low=mid
        stage=high+.01
        summary["comparison"].update(probe_stage_m=stage,native_connected=connection(elevation=native,level=stage),
            represented_connected=connection(elevation=actual,level=stage),opening_width_cells=2*options["family_control_halfwidth_m"]/grid,
            max_local_barrier_lowering_m=float(np.max(maps["barrier_lowering"])))
        result[str(grid)]=summary
        folder=directory/f"control_{grid}m"
        write_maps(maps={k:maps[k] for k in ("lost_channel","barrier_lowering","saddle_change","outlet_displacement","elevation_change")},folder=folder,transform=transform,crs=crs)
        write_maps(maps={k:coarse[k] for k in ("elevation","contributing_area","channel","fill_depth")},folder=folder/"represented",
                   transform=transform @ Affine.scale(grid),crs=crs)
    return dict(terrain_sha256=native_hash,domain=domain,control_window_m=[west,south,side,side],window=result)
