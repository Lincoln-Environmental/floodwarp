"""Generate the boundary repair evidence from accepted transport and immutable storm receipts."""

import json
from pathlib import Path

from research.floodwarp_methods_paper import metrics, runner


def run(*, directory):
    source=directory/"outlet_results"
    rows=metrics.csv_read(file=source/"events.csv")
    history=metrics.csv_read(file=directory/"mean_storms/complete_events.csv")
    before={(r["method"],r["dx_m"]):r for r in history}
    plane=directory/"free_plane"
    sha=runner.source_identity(root=directory.parents[2],study=directory.parent,stage="long-storms")
    (current,)=[p.parent for p in plane.glob("*/identity.json") if json.loads(p.read_text())["solver_sha256"]==sha]
    sheet=metrics.csv_read(file=current/"discharge.csv")
    old=metrics.csv_read(file=plane/"discharge.csv")
    text=["# FREE outlet qualification", "",
          "The exact Manning reference is steady uniform flow on a 7.329% plane. "
          "Discharge below uses the accepted face volume between 30 and 60 s at the same section, 400 m from the upstream boundary.",
          "", "| Method | Grid (m) | Cap (s) | Original face bias (%) | Repaired face bias (%) | Buffered section bias (%) |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in sheet:
        if r["start_s"] not in ("30.0","30") or r["face"]!="east" or int(r["downstream_extension_m"]):
            continue
        terms=("method","dx_m","maximum_dt_s","face","time_s")
        matches=[s for s in old if all(s[k]==r[k] for k in terms) and int(s["downstream_extension_m"])==0]
        buffer=[s for s in sheet if all(s[k]==r[k] for k in terms) and int(s["downstream_extension_m"])==400]
        if len(matches)!=1 or len(buffer)!=1:
            raise ValueError("Inconsistent analytical gauge support")
        text.append(f"| {r['method']} | {r['dx_m']} | {r['maximum_dt_s']} | {float(matches[0]['discharge_bias_pct']):+.4f} | "
                    f"{float(r['discharge_bias_pct']):+.4f} | {float(buffer[0]['discharge_bias_pct']):+.4f} |")
    text += ["","The terminal slope previously vanished when the reconstruction stencil reached the FREE face. "
             "This left a half-cell bed discontinuity at the last internal face and suppressed the surface-gradient momentum source. "
             "Hancock and RK2 now extrapolate the bed and surface increments from the wet interior at an explicitly open face, "
             "with zero normal velocity gradient and the existing dry and positivity checks. The residual uniform-sheet error "
             "decreases by a factor of four when the timestep cap halves, consistent with the time discretisation of the source balance. "
             "FV1 reconstructs from piecewise-constant cell beds. At this grade, the bed drops 0.07329 m per metre, exceeding the 0.03 m sheet depth even on the finest tested grid. Its hydrostatic cut to the higher cell bed clips the downstream state. At the 0.1 s cap the buffered FV1 deficit drops from 46.84% on 10 m cells to 36.82% on 1 m cells. The extra face deficit falls from 25.66 percentage points to below the displayed precision.",
             "", "## Full incised storm", "",
             "The gauge remains at the eastern edge of the original 104.8576 km² catchment. The extension continues the 1 m valley, "
             "initial channel state and soil. Its additional rainfall is zero. The source fields on the original footprint retain their historical values.",
             "", "| Method | Grid (m) | Extension (m) | CFL factor | Pre-fix direct peak (m³/s) | Peak (m³/s) | Minute ending (h) | "
             "Fixed 15 min peak (m³/s) | Moving 15 min peak (m³/s) | Export at gauge (m³) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows,key=sort_case):
        prev=before[r["method"],r["dx_m"]]
        historical=prev["outlet_peak_m3s"] if float(r["cfl_factor"])==1 and float(r["extension_m"])==0 else None
        value=f"{float(historical):.3f}" if historical else "—"
        text.append(f"| {r['method']} | {r['dx_m']} | {r['extension_m']} | {r['cfl_factor']} | {value} | "
                    f"{float(r['minute_peak_m3s']):.3f} | {float(r['minute_end_s'])/3600:.3f} | "
                    f"{float(r['fixed_15min_peak_m3s']):.3f} | {float(r['moving_15min_peak_m3s']):.3f} | {float(r['gauge_m3']):,.1f} |")
    text += ["", f"{len(rows)} of 14 trajectories have completed the historical 16-hour comparison. "
             "Each hydrograph is an interval average of accepted transport. On these irregular terrains a solver difference alone does not establish accuracy.",
             "", "[Water accounts, infiltration inside and outside the rainfall footprint](events.csv) · "
             "[Crest and whole-recession last 40 m budgets](last_40m.csv) · [Spatial extent](extent.csv) · "
             "[Same-grid wet-union depth differences](depth_disagreement.csv)"]
    (source/"README.md").write_text("\n".join(text)+"\n")


def sort_case(row):
    return -int(row["dx_m"]),int(row["extension_m"]),row["method"],float(row["cfl_factor"])


if __name__ == "__main__":
    directory=Path(__file__).resolve().parent
    run(directory=directory)
