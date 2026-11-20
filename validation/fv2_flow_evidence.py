"""Publish discharge waveforms, physical references and four-grid errors for the analytical tests."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from validation.fv1_riemann import report_json
from validation.fv2_figures import rows_table
from validation.fv2_radial_flow import radius


def physical_checks(*, cases, radial):
    """Check the storage-derived reference flux against independent pointwise SWE flux integration."""
    errors = []
    for r in cases:
        if r["case"] != "sloping_ritter" or r["method"] != "fv1" or r["dx"] != 5:
            continue
        gravity = r["setup"]["gravity"]
        head = r["setup"]["upstream_depth_m"]
        a = gravity * r["slope"]
        c = np.sqrt(head * gravity)
        for record in r["gauges"]:
            x = record["x_m"]

            def instantaneous(t):
                xi = (x - 0.5 * a * t * t) / t
                if xi <= -c:
                    return head * a * t
                if xi >= 2 * c:
                    return 0.0
                return (2 * c - xi) ** 2 / (9 * gravity) * (a * t + 2 * (c + xi) / 3)

            cuts = []
            for speed in (-c, 2 * c):
                for t in np.roots([0.5 * a, speed, -x]):
                    if np.isreal(t) and 0 < t.real < r["duration_s"]:
                        cuts.append(float(t.real))
            value = quad(instantaneous, 0, r["duration_s"], points=cuts, epsabs=1e-11, epsrel=1e-12)[0]
            errors.append(abs(value - record["exact_transport_m2"]))
    pulses = []
    basin = radial[0]
    omega = np.pi / basin["period_s"]
    first = basin["setup"]["first_radius_m"]
    second = basin["setup"]["second_radius_m"]
    for section in basin["gauges"]:
        x = section["x_m"]

        def section_flux(t):
            rad = radius(time=t, omega=omega, first_radius=first, second_radius=second)
            head = basin["setup"]["initial_centre_depth_m"] * (first / rad) ** 2
            dilation = (second**2 - first**2) * omega * np.sin(omega * t) * np.cos(omega * t) / (rad * rad)
            return 4 * head * dilation * x * max(0, rad * rad - x * x) ** 1.5 / (3 * rad * rad)

        dt = basin["interval_s"]
        for i, q in enumerate(section["exact_m3_s"]):
            integral = quad(section_flux, i * dt, (i + 1) * dt, epsabs=1e-9, epsrel=1e-11)[0]
            pulses.append(abs(integral / dt - q))
    if max(errors) > 1e-9 or max(pulses) > 1e-8:
        raise AssertionError(
            "Independent integration of h times normal velocity disagrees with the reference"
        )
    return dict(
        ritter_max_absolute_volume_error_m2=max(errors),
        basin_max_interval_mean_discharge_error_m3_s=max(pulses),
    )


def label_case(*, row):
    if row["case"] == "sloping_ritter":
        return f"Ritter {100 * row['slope']:+g}%"
    return row["case"].replace("_", " ")


def flow_summary(*, row, key):
    return float(np.sqrt(np.mean([g[key] ** 2 for g in row["gauges"]])))


def plot(*, rows, radial, dx, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    panels = (
        ("sloping_ritter", 0, 40),
        ("sloping_ritter", 0.02, 40),
        ("sloping_ritter", 0.08, 40),
        ("transcritical_jump", 0, 40),
        ("radial", 0, 80),
        ("radial", 0, 120),
    )
    for ax, (case, slope, gauge) in zip(axes.flat, panels, strict=True):
        data = radial if case == "radial" else [r for r in rows if r["case"] == case and r["slope"] == slope]
        for method, color in (("fv1", "#2166ac"), ("roe", "#b67016"), ("mc", "#198754")):
            r = next(v for v in data if v["dx"] == dx and v["method"] == method)
            g = next(v for v in r["gauges"] if v["x_m"] == gauge)
            unit = "m³/s" if case == "radial" else "m²/s"
            exact, model = (
                (g["exact_m3_s"], g["model_m3_s"])
                if case == "radial"
                else (g["exact_interval_mean_m2_s"], g["model_interval_mean_m2_s"])
            )
            t = (0.5 + np.arange(len(exact))) * r["interval_s"]
            if method == "fv1":
                ax.plot(t, exact, "k--", lw=2, label="Exact", zorder=5)
            ax.plot(
                t,
                model,
                color=color,
                label={"fv1": "FV1 contact-HLL", "roe": "FV1 Roe", "mc": "FV2 MC"}[method],
                lw=1.3,
            )
        name = "Radial basin" if case == "radial" else label_case(row=r)
        ax.set(
            title=f"{name.capitalize()}, x = {gauge} m",
            xlabel="Time (s)",
            ylabel=f"Section discharge ({unit})",
        )
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def publish(*, directory, methods, dx):
    rows = json.loads((directory / "flow_truth.json").read_text())
    rain = json.loads((directory / "rain_flow.json").read_text())
    radial = json.loads((directory / "radial_flow.json").read_text())
    if (len(rows), len(rain), len(radial)) != (160, 40, 20):
        raise ValueError("Incomplete flow matrix")
    checks = physical_checks(cases=rows, radial=radial)
    report_json(path=directory / "flow_reference_checks.json", content=checks)
    plot(rows=rows, radial=radial, dx=dx, path=directory / "analytical_hydrographs.png")
    by_case = {}
    for r in rows + rain:
        by_case.setdefault(label_case(row=r), {})[r["dx"], r["method"]] = r
    measured = []
    convergence = []
    for name, records in by_case.items():
        measured.append(
            [name, *[flow_summary(row=records[dx, method], key="rmse_m2_s") for method in methods]]
        )
        for method in ("fv1", "mc"):
            spacing = sorted([d for d, m in records if m == method], reverse=True)
            e = [flow_summary(row=records[d, method], key="rmse_m2_s") for d in spacing]
            convergence.append([name, method, *e, *np.log2(np.array(e[:-1]) / e[1:])])
    table = rows_table(
        title=f"Unit-width discharge errors at {dx:g} m",
        columns=["Analytical case", "FV1 contact", "FV1 Roe", "FV1 HLLC", "FV2 minmod", "FV2 MC"],
        rows=measured,
        note="RMSE in m²/s, weighted equally across three fixed sections and all sampling intervals. Secant fluxes are differences of accepted, signed face-volume ledgers divided by section width and sample duration. All five runs at each resolution use the same timestep cap.",
    )
    refinement = rows_table(
        title="Does discharge converge under grid refinement?",
        columns=["Case", "Method", "20 m", "10 m", "5 m", "2.5 m", "p₁", "p₂", "p₃"],
        rows=convergence,
        note="Three-section discharge RMSE in m²/s and consecutive observed orders. Moving fronts and shocks can give nonmonotone errors. The JSON includes each gauge separately.",
    )
    peaks = []
    for r in radial:
        if r["dx"] == dx:
            for g in r["gauges"]:
                peaks.append(
                    [
                        r["method"],
                        g["x_m"],
                        g["cycle2_peak_exact_m3_s"],
                        g["cycle2_peak_model_m3_s"],
                        g["cycle2_peak_time_error_s"]
                        if g["cycle2_peak_time_error_s"] is not None
                        else "No outward peak",
                        g["hydrograph_rmse_m3_s"],
                        g["gross_positive_volume_error_m3"],
                    ]
                )
    waves = rows_table(
        title="Full 2D oscillating-basin discharge",
        columns=[
            "Method",
            "Section x (m)",
            "Exact second-cycle peak (m³/s)",
            "Model peak (m³/s)",
            "Peak time error (s)",
            "Hydrograph RMSE (m³/s)",
            "Positive-transport volume error (m³)",
        ],
        rows=peaks,
        note=f"Full-width discharges at {dx:g} m. Timing uses interval means over 64 intervals per cycle. The timing error is undefined when the model has no positive discharge in the second cycle. Peak attenuation records loss of wave amplitude. Two cycles are used for volume and waveform errors.",
    )
    intro = """# Independent analytical flow comparisons

These 220 GPU runs span eleven analytical flow regimes, four grid spacings and five matched schemes. They measure discharge through physical cross-sections, including peak attenuation, flow reversal and wetting. Depth, velocity, momentum, minimum depth and mass residuals are retained for every run. The benchmark equations and physical parameters are independent of the solver.

**Accelerating dam breaks.** Set the bed to z = −Sx and let a = gS. Transform the flat-bed Ritter solution with x′ = x − at²/2 and u = u′ + at. Mass and inertial acceleration then satisfy the sloping-bed equations exactly. The three sections at 0, 40 and 80 m measure cumulative downstream storage and its exact one-second differences. The reservoir starts at 2 m. The 20 s solutions include horizontal, 2% and 8% downhill, and 2% uphill beds. Upstream and downstream model walls are 800 m from the initial front. The comparison window and sections are outside their physical domains of influence. An independent time integral of pointwise hu at each section checks the exact transport.

**Flow over a crest.** Four steady solutions conserve unit-width discharge q = 1.5 m²/s and total head z + h + q²/(2gh²) along each smooth branch. A raised cosine-squared bump is 0.4 m high and 120 m wide. The subcritical and supercritical cases use opposite Bernoulli roots. The transcritical solution changes branch at the critical crest. The fourth case includes a stationary hydraulic jump at x = 22.5 m on the downstream slope. Its conjugate depths carry equal mass and normal momentum flux and dissipate energy. Quadrature splits cells at the crest, bump limits and jump. These are exact-state preservation tests over 20 s in the interior, with remote walls outside the sampled characteristics. They do not measure the time to establish a steady flow from rest. The SWASHES collection describes this solution family. [Delestre et al.](https://doi.org/10.1002/fld.3741)

**Rain-fed slopes with Manning friction.** The exact profiles prescribe constant depth, spatial rainfall R(x), roughness n(x) and q′ = R. With vertical rainfall carrying zero horizontal momentum, the required bed gradient is z′ = −2Rq/(gh²) − n²q²/h^(10/3). All input cell averages are polynomial integrals. Rain increases from 36 to 108 mm/h and n from 0.025 to 0.040 across the domain. The 30 mm film crosses critical flow. The 150 mm case stays subcritical. The initial state is the exact moving equilibrium. The sampled 40 s interval tests transport and friction under sustained rainfall. This forcing differs from the Orton design event.

**Unsteady full 2D discharge.** A frictionless paraboloid of water expands and contracts over bed z = αr². Its radius obeys R²(t) = A²cos²(ωt) + B²sin²(ωt), with ω² = 2gα, A = 80 m and B = 160 m. Within the moving shoreline h = αB²(A/R)²(1 − r²/R²) and (u,v) = (R′/R)(x,y). Substitution gives R″ + 2gαR = 2gαA²B²/R³. Integrating h over the half-plane beyond each section gives the exact interval transport. Independently integrating hu across the section and over time checks the reference. A second initial quadrature at twice the degree checks cell-averaging error. Both model axes reconstruct and transport water in this case.

![Analytical and simulated section hydrographs](analytical_hydrographs.png)

"""
    (directory / "FLOW_TRUTH.md").write_text(
        intro
        + table
        + "\n"
        + waves
        + "\n"
        + refinement
        + "\nThe catchment receives rainfall, exchanges water with soil and drains through survey-dependent openings. These controlled discharges test mechanisms. The Orton peak needs its own convergence or observational evidence.\n"
    )


if __name__ == "__main__":
    directory = Path("validation/fv2_review")
    methods = ("fv1", "roe", "hllc", "minmod", "mc")
    publish(directory=directory, methods=methods, dx=5.0)
