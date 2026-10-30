"""Versioned LI parity and engineering assessment of immutable replay arrays."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def write_json(*, path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(*, path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def depth_metrics(*, actual, reference, percentiles):
    error = np.abs(actual - reference)
    norm = float(np.linalg.norm(reference))
    return dict(rmse_m=float(np.sqrt(np.mean(error**2))), mae_m=float(np.mean(error)),
                max_m=float(np.max(error)), relative_l2=float(np.linalg.norm(error) / norm) if norm else None,
                **{f'p{p}_m': float(np.percentile(error, q=p)) for p in percentiles})


def wet_metrics(*, actual, reference, threshold, area, bands):
    wet_a, wet_r = actual > threshold, reference > threshold
    tp = int(np.count_nonzero(wet_a & wet_r))
    fp = int(np.count_nonzero(wet_a & ~wet_r))
    fn = int(np.count_nonzero(~wet_a & wet_r))
    mismatch = wet_a != wet_r
    distance = np.maximum(np.abs(actual - threshold), np.abs(reference - threshold))
    differences = np.abs(actual - reference)[mismatch]
    return dict(threshold_m=threshold, iou=tp / (tp + fp + fn) if tp + fp + fn else 1.,
        precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn) if tp + fn else None,
        empty_union=not bool(tp + fp + fn), true_wet=tp, false_wet=fp, false_dry=fn,
        disagreement_cells=fp + fn, disagreement_m2=(fp + fn) * area,
        false_wet_m2=fp * area, false_dry_m2=fn * area,
        mismatch_max_depth_error_m=float(differences.max()) if differences.size else 0.,
        mismatch_median_depth_error_m=float(np.median(differences)) if differences.size else 0.,
        **{f'both_within_{band:g}m_cells': int(np.count_nonzero(mismatch & (distance <= band))) for band in bands})


def compare_parity(*, source, target, output, source_commit, target_commit, report_name):
    left = json.loads((source / 'manifest.json').read_text())
    right = json.loads((target / 'manifest.json').read_text())
    assert [row['label'] for row in left] == [row['label'] for row in right]
    rows = []
    for a, b in zip(left, right, strict=True):
        differences = {}
        with np.load(file=source / (a['label'] + '.npz')) as x, np.load(file=target / (b['label'] + '.npz')) as y:
            assert x.files == y.files
            for key in x.files:
                bitwise = x[key].dtype == y[key].dtype and x[key].shape == y[key].shape and x[key].tobytes() == y[key].tobytes()
                if not bitwise:
                    delta = np.abs(x[key] - y[key])
                    differences[key] = dict(max_absolute=float(delta.max()), first_index=np.argwhere(delta != 0)[0].tolist())
        config_equal = a['numerical'] == b['numerical'] and all(a[key] == b[key] for key in ('gravity', 'cfl', 'cap', 'minimum_dt', 'device', 'dtype'))
        rows.append(dict(label=a['label'], bitwise=not differences, configuration_equal=config_equal,
                         steps=a['steps'], differences=differences, hashes=a['hashes']))
    report = dict(source_commit=source_commit, starting_target_head=target_commit, cases=len(rows),
        all_bitwise=all(row['bitwise'] and row['configuration_equal'] for row in rows), rows=rows,
        source_channel_rename='groundwater_sources -> additional_sources; corresponding ledger values compared by index',
        scope='CPU FP64; every accepted dt and sampled depth/stage/face/source/ledger/residual. No GPU extraction-parity claim.',
        first_divergence=None if all(row['bitwise'] for row in rows) else 'See differences; operation-level tracing required before qualification.')
    write_json(path=output / report_name, value=report)
    if not report['all_bitwise']:
        raise AssertionError('Extraction parity failed; do not qualify engineering agreement')


def assess_runs(*, roots, fixtures, output, thresholds, bands, percentiles, engineering):
    manifest = json.loads((fixtures / 'manifest.json').read_text())
    for case in manifest['cases']:
        for name, expected in case['files'].items():
            if hashlib.sha256((fixtures / case['id'] / name).read_bytes()).hexdigest() != expected:
                raise ValueError('Reference checksum mismatch: ' + case['id'] + '/' + name)
    summaries, wet_rows, depth_rows, hydro_rows = [], [], [], []
    for group, root in roots.items():
        manifest = json.loads((root / 'manifest.json').read_text())
        for row in manifest:
            if group == 'baseline' and (row['profile'] is None or (row['profile'] == 'lisflood_fp_acc' and row['closure'] is None)):
                continue
            case = row['case']
            directory = fixtures / case
            scenario = json.loads((directory / 'scenario.json').read_text())
            area = scenario['spacing']**2
            with np.load(file=root / (row['label'] + '.npz')) as data:
                data = {key: data[key].copy() for key in data.files}
            with np.load(file=directory / 'depth_series.npz') as ref:
                reference = ref['depth'].copy()
                assert np.array_equal(data['times'], ref['times_s'])
            with np.load(file=directory / 'arrays.npz') as ref:
                ref_q = np.interp(data['times'], ref['times'], ref['qout'])
                ref_export = np.interp(data['times'], ref['times'], ref['cumulative_out'])
            budget = json.loads((directory / 'report.json').read_text())['budget']
            supplied = budget['rain'] + budget['boundary_inflow'] + scenario['initial'] * area * reference.shape[1] * reference.shape[2]
            failed = set()
            final_wet, final_depth = [], None
            for index, (actual, expected, time) in enumerate(zip(data['depth'], reference, data['times'], strict=True)):
                identity = dict(group=group, label=row['label'], time_s=float(time))
                metrics = depth_metrics(actual=actual, reference=expected, percentiles=percentiles)
                depth_rows.append(dict(**identity, **metrics))
                if not (metrics['rmse_m'] <= engineering['rmse_m'] or (metrics['relative_l2'] is not None and metrics['relative_l2'] <= engineering['relative_l2'])):
                    failed.add('depth_RMSE_or_relative_L2')
                if metrics['p95_m'] > engineering['p95_m']:
                    failed.add('depth_P95')
                if metrics['max_m'] > engineering['max_m']:
                    failed.add('depth_max')
                current_wet = []
                for threshold in thresholds:
                    wet = wet_metrics(actual=actual, reference=expected, threshold=threshold, area=area, bands=bands)
                    wet_rows.append(dict(**identity, **wet))
                    current_wet.append(wet)
                    if threshold in engineering['extent_thresholds'] and not (wet['iou'] >= engineering['iou'] or wet['disagreement_cells'] == wet[f"both_within_{engineering['band_m']:g}m_cells"]):
                        failed.add(f'extent_{threshold:g}m')
                final_wet, final_depth = current_wet, metrics
            spacing = scenario['spacing']
            outward = np.concatenate((-data['qx'][:, :, 0], data['qx'][:, :, -1], -data['qy'][:, 0, :], data['qy'][:, -1, :]), axis=1) * spacing
            q = np.maximum(outward, 0.).sum(axis=1)
            ledger = dict(zip(row['ledger_keys'], data['ledger'][-1].tolist(), strict=True))
            export = data['ledger'][:, row['ledger_keys'].index('boundary_outflow')]
            q_rmse = float(np.sqrt(np.mean((q - ref_q)**2)))
            peak = float(np.max(np.abs(ref_q)))
            q_nrmse = q_rmse / peak if peak else None
            if (peak and q_nrmse > engineering['q_nrmse']) or (not peak and q_rmse):
                failed.add('outlet_NRMSE')
            volume_difference = float(export[-1] - ref_export[-1])
            storage_difference = float((data['depth'][-1] - reference[-1]).sum() * area)
            if abs(volume_difference) > engineering['volume_fraction'] * supplied:
                failed.add('export_fraction_supplied')
            if abs(storage_difference) > engineering['volume_fraction'] * supplied:
                failed.add('storage_fraction_supplied')
            for time, actual_q, expected_q, actual_v, expected_v in zip(data['times'], q, ref_q, export, ref_export, strict=True):
                hydro_rows.append(dict(group=group, label=row['label'], time_s=float(time), actual_q_m3_s=float(actual_q), reference_q_m3_s=float(expected_q),
                    difference_q_m3_s=float(actual_q - expected_q), actual_export_m3=float(actual_v), reference_export_m3=float(expected_v), difference_export_m3=float(actual_v - expected_v)))
            integrity = row['finite'] and row['minimum_depth'] >= 0 and row['max_mass_residual_m3'] <= 1.e-8 * max(1., supplied)
            summaries.append(dict(group=group, label=row['label'], numerical=row['numerical'], cfl=row['cfl'], maximum_dt=row['cap'], minimum_dt=row['minimum_dt'],
                integrity=integrity, engineering_screen=not failed and integrity, failed_gates=sorted(failed), final_depth=final_depth, final_wet=final_wet,
                budget=ledger, reference_budget=budget, supplied_m3=supplied, export_difference_m3=volume_difference, storage_difference_m3=storage_difference,
                export_difference_fraction_supplied=volume_difference / supplied if supplied else None,
                infiltration_difference_m3=ledger['infiltration'] - budget['infiltration'],
                outlet_nrmse=q_nrmse, outlet_rmse_m3_s=q_rmse, max_mass_residual_m3=row['max_mass_residual_m3']))
    write_json(path=output / 'engineering_results.json', value=dict(criteria=engineering, rows=summaries))
    write_csv(path=output / 'wet_thresholds.csv', rows=wet_rows)
    write_csv(path=output / 'depth_errors.csv', rows=depth_rows)
    write_csv(path=output / 'hydrographs.csv', rows=hydro_rows)


if __name__ == "__main__":
    scratch = Path('outputs/li_review_v1/baselines')
    fixtures = Path('validation/fixtures')
    output = Path('validation/li_review_v1')
    source_commit = 'ba503b8bde51d3f212199e81efc7038d9fffecc8'
    target_commit = '77c794a95db7aad8d79fa0574420a13e79273cb8'
    roots = dict(baseline=scratch / 'target_results', attribution=Path('outputs/li_review_v1/attribution_v2'), shipped_defaults=Path('outputs/li_review_v1/shipped_defaults_v2'))
    thresholds = (.0001, .001, .005, .01, .05, .1)
    bands = (.000001, .00001, .0001, .001)
    percentiles = (50, 90, 95, 99)
    engineering = dict(rmse_m=.001, relative_l2=.02, p95_m=.005, max_m=.01,
        extent_thresholds=(.01, .05, .1), iou=.95, band_m=.001, q_nrmse=.05, volume_fraction=.02)
    compare_parity(source=scratch / 'source_results', target=scratch / 'target_results', output=output,
                   source_commit=source_commit, target_commit=target_commit, report_name='parity.json')
    compare_parity(source=scratch / 'source_defaults', target=scratch / 'target_defaults', output=output,
                   source_commit=source_commit, target_commit=target_commit, report_name='default_parity.json')
    assess_runs(roots=roots, fixtures=fixtures, output=output, thresholds=thresholds, bands=bands,
                percentiles=percentiles, engineering=engineering)
