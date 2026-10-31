"""Summarise measured static FV1 controls and explicit stage-gate decisions."""

from pathlib import Path
import hashlib
import json
import re
import shutil


def write_report(*, directory, budget_path, test_log):
    benchmark = json.loads((directory / 'benchmark.json').read_text())
    memory = json.loads((directory / 'memory.json').read_text())
    assessment = json.loads((directory / 'coarsenability.json').read_text())
    snapshot = directory / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    paths = [Path('src/floodwarp') / name for name in
        ('static_fv1.py', 'static_fv1_kernels.py', 'static_fv1_mesh.py', 'full_swe_solver.py', 'full_swe_kernels.py')]
    paths.extend(Path('validation') / name for name in
        ('static_fv1_benchmark.py', 'static_fv1_profile.py', 'static_fv1_memory.py', 'static_fv1_assess.py', 'static_fv1_report.py'))
    hashes = {}
    for path in paths:
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(src=path, dst=snapshot / path.name)
    passed, skipped = re.findall(r'(\d+) passed, (\d+) skipped', test_log.read_text())[-1]
    budget_hash = hashlib.sha256(budget_path.read_bytes()).hexdigest()
    fastest = max(row['measured_speedup'] for case in benchmark['cases'] for row in case['configurations'])
    slowest = min(row['measured_speedup'] for case in benchmark['cases'] for row in case['configurations'])
    lines = ['# Conservative static block FV1 — experimental stage result', '',
        '**Numerical foundation implemented; production runtime gate failed. Dynamic adaptation is deferred.** '
        'This is a real leaf-volume FV1 evolution with conservative shared subfaces, not a compressed depth map.', '',
        'The forest has regular blocks, two spatial levels (2:1), fixed topology and one global timestep. '
        'Each fine physical cell belongs to exactly one evolved leaf. Each represented fine subface is '
        'solved once with FloodWarp’s existing HLL operator. Its accepted mass flux enters the two '
        'neighbouring leaves with opposite signs. A coarse donor shares one volume budget across all '
        'of its subfaces; hydrostatic pressure corrections remain side-specific.', '',
        'Finest geometry is immutable. Coarse bed is the area mean used only to define the parent '
        'common free surface; fine subface beds are retained. Restriction preserves area-integrated '
        'water and momentum. Prolongation preserves those totals and equilibrium where representable. '
        'Mixed wet/dry, non-affine/connectivity-changing geometry, mask/boundary interfaces, protected '
        'features and source/roughness heterogeneity veto coarsening. Unrepresentable transfers stay fine. '
        'A later unrepresentable sloping coarse wet/dry state fails the experimental run; it is not '
        'silently clipped into an accepted solution.', '',
        'Sources and soil/history arrays stay at fine resolution. Surface availability beneath a wet '
        'parent comes from its common free surface and velocity; accepted fine source increments are '
        'restricted to that parent. That availability/reconstruction is a documented spatial '
        'approximation, not parameter averaging. Source setters cannot bypass heterogeneity vetoes. '
        'The existing finest source/boundary-aware global CFL scan is retained.', '',
        '## Accuracy and runtime', '',
        'Pinned FP64 environment, RTX 4070 Ti SUPER. Each matched 6 s control uses warm-up and '
        'three graph-timed repetitions on a 64×64 physical grid. Median hydraulic seconds are below. '
        'Topology setup, end-to-end loop time, timesteps, per-step time, per-level leaf counts and '
        'subface interface counts are in benchmark.json. Setup/JIT are outside hydraulic timing.', '',
        '| Control | Dense [s] | Static B=4 [s] | Static B=8 [s] | B=4 leaves | B=4 fine subfaces | B=4 depth RMSE [m] | B=4 relative L2 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    gates = []
    for case in benchmark['cases']:
        first, second = case['configurations']
        diag, metric = first['diagnostics'], first['accuracy']
        relative = metric['depth_relative_l2']
        lines.append(f'| {case["name"]} | {case["dense_median_seconds"]:.6f} | {first["median_seconds"]:.6f} | '
            f'{second["median_seconds"]:.6f} | {diag["hydraulic_leaves"]} | '
            f'{diag["evaluated_faces_x"] + diag["evaluated_faces_y"]} | {metric["depth_rmse_m"]:.7g} | '
            + (f'{relative:.7g}' if relative is not None else 'dry') + ' |')
        for row in case['configurations']:
            m = row['accuracy']
            extent = all(entry['iou'] >= .95 or entry['mismatch_fraction'] <= .0005
                         for entry in m['maximum_event_inundation'].values())
            integrity = m['minimum_depth_m'] >= 0. and abs(m['mass_residual_m3']) <= m['mass_residual_limit_m3']
            hydro = None if m['hydrograph_nrmse'] is None else (m['hydrograph_nrmse'] <= .05 and
                abs(m['peak_discharge_relative_error']) <= .02 and abs(m['peak_timing_error_s']) <= benchmark['sample_interval_s'])
            gates.append(dict(case=case['name'], block_size=row['block_size'],
                depth_and_maximum_budget=bool(row['depth_accuracy_budget_pass']), extent_budget=bool(extent),
                positivity_and_conservation=bool(integrity), hydrograph_budget=hydro,
                runtime_benefit=row['measured_speedup'] > 1.,
                source_and_export_difference_small=abs(m['infiltration_difference_m3']) <= 5. and abs(m['cumulative_export_difference_m3']) <= 5.))
    lines.extend(['', f'Measured dense/static ratios range from {slowest:.3f} to {fastest:.3f}: '
        f'**{100. * (1. / fastest - 1.):.1f}–{100. * (1. / slowest - 1.):.1f}% slower**. '
        'The B=8 lake/front/channel controls have zero eligible coarse roots; they measure the '
        'all-fine bookkeeping cost and must not be presented as multiresolution accuracy successes. '
        'B=4 does evolve coarse wet leaves in lake, ponding and moving-front controls.', '',
        'The coarsened moving front has approximately 3.89 mm domain depth RMSE and 1.51% relative '
        'L2 error, within the predeclared relative budget. The runtime rejection is substantive, '
        'not a rejection of insignificant floating-point differences. Lake, ponding, infiltration '
        'and protected-channel controls reproduce the uniform result to FP64 tolerance. '
        'Closed controls have no applicable outlet hydrograph; the channel outlet comparison '
        'records NRMSE, peak magnitude and timing separately.', '',
        'Separate instrumented profiles identify restriction, two fine projections, raw-outflow '
        'aggregation, subface acceptance/scatter, clearing and additional launches as overhead. '
        'Fine sources/CFL are not removed. Bounded one-step R(A_f(U)) versus A_c(R(U)) diagnostics '
        'are retained; there is no production fine-grid shadow evolution.', '',
        '## Memory', '',
        '| Control | Mode/block | Persistent bytes | Peak additional live pool bytes |', '|---|---|---:|---:|'])
    for row in memory['rows']:
        label = 'dense' if row['mode'] == 'dense' else 'static/' + str(row['block_size'])
        lines.append(f'| {row["case"]} | {label} | {row["persistent_bytes"]} | {row["peak_additional_live_pool_bytes"]} |')
    lines.extend(['', 'Static construction retains a resident uniform source model. The separate memory probe '
        'reports its bytes and whole-process peak as well as the additional static allocation. '
        'CUDA context/code/reserved memory is excluded. Dense fine projection/face/history arrays '
        'remain allocated and leaf fields add memory; there is no sparse-storage saving.', '',
        '## Allocation and adaptation stage decisions', '',
        'The sampled ideal high-water state-plus-face pool bounds for Purau save at most '
        f'{assessment["ideal_persistent_pool_bounds"][0]["ideal_total_savings_bytes"]:,} bytes '
        f'({100. * assessment["ideal_persistent_pool_bounds"][0]["ideal_total_savings_fraction"]:.2f}% of '
        '246,266,984 bytes), before pool growth slack, maps and padding. A conservative-state-only '
        'pool saves only about 3.3% of total allocation. The current model has no device-capacity '
        'bottleneck on a 16 GiB GPU. Block-sparse storage is therefore deferred at its occupancy/'
        'memory gate; no pool exhaustion/deallocation/restart implementation is claimed.', '',
        'Coarsenability is measured inside active regions independently of exact sparsity. At '
        '25 h there are 139,576 fully wet 2×2 physical parents, 19 pass the simple hydraulic '
        'indicator, and none pass the initial affine-geometry safeguard. Across all six sampled '
        'phases, 4/8/16-cell block choices admit no coarse Purau roots. These are limitations of '
        'this conservative initial representation and thresholds, not proof that all AMR is '
        'unprofitable. Completed-event fields serve diagnostics only.', '',
        'Static numerical controls pass, but the measured runtime gate fails and no admissible '
        'full Purau candidate exists under the initial safeguards. Dynamic refinement/coarsening, '
        'Haar indicator comparisons, sparse-storage integration and combined speedups remain '
        'unimplemented. No static/adaptive 72-hour Purau simulation is launched. A lower-overhead '
        'leaf/source/geometry representation needs evidence before advancing that stage.', '',
        '## Verification and scope', '',
        f'Full suite: **{passed} passed, {skipped} skipped**. The focused static/sparse log contains '
        '**80 passing controls**: stepped/affine lake equilibrium, positivity, common safe steps, '
        'ordinary CFL, moving and oblique fronts, physical crossings in all four orientations, '
        'rain ponding, fully infiltrating rain, heterogeneous sources, disconnected regions, '
        'protected narrow channel/barriers, source-change vetoes, shared donor budgets, masked '
        'boundaries, 32 repeated conservative transfers, CPU/CUDA disk restart and exact sparse '
        'history/activation controls. Tiny positive water at a high datum is retained by stable '
        'bed-difference arithmetic. Repeated transfer testing is not a claim of dynamic adaptation.', '',
        'Both research methods remain opt-in on dev. The published v0.1.0 tag/main remain fixed '
        'at 45a374c and contain neither research implementation. Dense FP64 FV1 remains the '
        'reference/fallback. No Riemann law, source order, boundary closure, precision, finest '
        'CFL restriction, reference data, external solver implementation or Overlander checkout '
        'was changed by this stage.', ''])
    (directory / 'REPORT.md').write_text('\n'.join(lines))
    (directory / 'gates.json').write_text(json.dumps(dict(budget_sha256=budget_hash,
        sources=hashes, controls=gates, static_runtime_gate_pass=False,
        dynamic_adaptation='deferred after failed static runtime gate', sparse_storage='deferred at memory/occupancy gate'), indent=2) + '\n')


if __name__ == "__main__":
    directory = Path('validation/static_review')
    budget_path = Path('docs/STATIC_FV1_BUDGETS.md')
    test_log = directory / 'tests_full_cpu_cuda.txt'
    write_report(directory=directory, budget_path=budget_path, test_log=test_log)
