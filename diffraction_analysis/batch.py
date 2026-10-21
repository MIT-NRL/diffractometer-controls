"""Batch fits and parameter trends independent of Qt and the instrument."""

from dataclasses import asdict
import numpy as np

from .curves import data_keys, numeric, select_curve
from .fitting import fit_curve

METRICS = {'Peak position': 'center', 'Peak height': 'height', 'Peak area': 'amplitude',
           'FWHM': 'fwhm', 'Sigma': 'sigma'}


def coordinate_fields(snapshot):
    rows = snapshot.get('rows', [])
    motors = snapshot.get('start', {}).get('motors', [])
    fields = []
    descriptions = data_keys(snapshot)
    for key in dict.fromkeys([*motors, *(rows[0] if rows else [])]):
        if key == 'time' or key.startswith('ts_') or key.endswith('_user_setpoint'):
            continue
        if descriptions.get(key, {}).get('shape'):
            continue
        if any(numeric(row.get(key)) is not None and numeric(row.get(key)).ndim == 0 for row in rows):
            fields.append(key)
    return fields


def fit_record(snapshot, field, point, x_field, config, *, coordinates_for=None):
    curve = select_curve(snapshot, field, point=point, x_field=x_field)
    row = snapshot['rows'][point]
    sequences = snapshot.get('seq_nums', [])
    coordinates = {}
    for key in coordinate_fields(snapshot) if coordinates_for is None else coordinates_for:
        value = numeric(row.get(key))
        if value is not None and value.ndim == 0 and np.isfinite(value):
            coordinates[key] = float(value)
    record = {'point': curve['point'], 'sequence': sequences[point] if point < len(sequences) else point + 1,
              'coordinates': coordinates, 'error': '', 'fit': None,
              'units': {'coordinate': curve['x_units'], 'intensity': curve['y_units']}}
    try:
        record['fit'] = fit_curve(curve['x'], curve['y'], config)
    except (ValueError, RuntimeError, TypeError) as exc:
        record['error'] = str(exc)
    return record


def fit_snapshot(snapshot, field, x_field, config, *, progress=None, cancelled=None):
    """Fit all frames of one signal with one configuration; retain failed rows.

    Scalar data is a single trace and therefore produces one fit. Cancellation
    is checked between frames; already completed records remain available.
    """
    rows = snapshot.get('rows', [])
    if not rows:
        raise ValueError('The selected run has no acquisition rows.')
    first = numeric(rows[0].get(field))
    if first is None:
        raise ValueError('The selected signal is not numeric.')
    points = range(len(rows)) if first.ndim == 1 else range(1)
    records = []
    stopped = False
    coordinates_for = coordinate_fields(snapshot)
    for index, point in enumerate(points):
        if cancelled is not None and cancelled():
            stopped = True
            break
        try:
            record = fit_record(snapshot, field, point, x_field, config, coordinates_for=coordinates_for)
        except (ValueError, RuntimeError, TypeError) as exc:
            sequences = snapshot.get('seq_nums', [])
            record = {'point': point, 'sequence': sequences[point] if point < len(sequences) else point + 1,
                      'coordinates': {}, 'fit': None, 'error': str(exc)}
        records.append(record)
        if progress is not None:
            progress(index + 1, len(points))
    return {'uid': snapshot.get('uid', ''), 'field': field, 'x_field': x_field,
            'config': asdict(config), 'records': records, 'cancelled': stopped,
            'total': len(points)}


def parameter_trend(records, metric='Peak position', peak=1, coordinate=''):
    """Return ordered x/y/error arrays, with NaNs for failed or missing values."""
    suffix = METRICS[metric]
    x, y, stderr = [], [], []
    for record in records:
        x.append(record['coordinates'].get(coordinate, np.nan) if coordinate else record['sequence'])
        fit = record.get('fit')
        parameter = fit['parameters'].get(f'p{peak}_{suffix}') if fit is not None else None
        y.append(parameter['value'] if parameter else np.nan)
        error = parameter.get('stderr') if parameter else None
        stderr.append(float(error) if error is not None else np.nan)
    return np.asarray(x, float), np.asarray(y, float), np.asarray(stderr, float)


def export_batch(batch):
    """JSON-ready fit results, retaining selection, configuration and failures."""
    result = {key: value for key, value in batch.items() if key != 'records'}
    result['records'] = []
    for record in batch['records']:
        item = {key: value for key, value in record.items() if key != 'fit'}
        fit = record['fit']
        item['fit'] = None if fit is None else {key: value for key, value in fit.items()
            if key not in {'x', 'y', 'fit', 'residual', 'components'}}
        result['records'].append(item)
    return result
