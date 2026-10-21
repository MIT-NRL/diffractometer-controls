"""Select numeric scalar traces and 1D frames from normalized run snapshots."""

import numpy as np


def data_keys(snapshot):
    keys = {}
    for descriptor in snapshot.get('descriptor_docs', []):
        keys.update(descriptor.get('data_keys', {}))
    return keys


def numeric(value):
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
        return array if array.ndim <= 1 else None
    except (TypeError, ValueError):
        return None


def signal_fields(snapshot):
    rows = snapshot.get('rows', [])
    if not rows:
        return []
    start = snapshot.get('start', {})
    declarations = start.get('live_plot_fields', {})
    declared = {key for key, spec in declarations.items() if spec.get('role', 'signal') == 'signal'}
    excluded = set(start.get('motors', []))
    candidates = []
    for key in rows[0]:
        if key in excluded or key == 'time' or key.startswith('ts_') or key.endswith('_user_setpoint'):
            continue
        if key.endswith('_position_x') or declarations.get(key, {}).get('role') in {'axis', 'coordinate', 'elapsed_time'}:
            continue
        value = next((numeric(row[key]) for row in rows if key in row), None)
        if value is not None and (not declared or key in declared):
            candidates.append(key)
    return sorted(candidates, key=lambda key: (not key.endswith('_counts'), key))


def signal_label(snapshot, field):
    declarations = snapshot.get('start', {}).get('live_plot_fields', {})
    label = declarations.get(field, {}).get('label', field)
    duplicates = sum(declarations.get(key, {}).get('label', key) == label for key in signal_fields(snapshot))
    return f'{label} [{field}]' if duplicates > 1 else label


def axis_fields(snapshot, field, point=0):
    rows = snapshot.get('rows', [])
    if not rows:
        return []
    value = numeric(rows[min(point, len(rows) - 1)].get(field))
    if value is None:
        return []
    if value.ndim == 1:
        return [key for key, item in rows[min(point, len(rows) - 1)].items()
                if key != field and numeric(item) is not None and numeric(item).shape == value.shape]
    return [key for key in rows[0] if key != field and numeric(rows[0][key]) is not None and numeric(rows[0][key]).ndim == 0]


def default_axis(snapshot, field, point=0):
    axes = axis_fields(snapshot, field, point)
    conventional = field.removesuffix('_counts') + '_position_x'
    if conventional in axes:
        return conventional
    for key, spec in snapshot.get('start', {}).get('live_plot_fields', {}).items():
        if key in axes and spec.get('role') in {'coordinate', 'axis'}:
            return key
    for motor in snapshot.get('start', {}).get('motors', []):
        if motor in axes:
            return motor
    return ''


def select_curve(snapshot, field, *, point=0, x_field=''):
    rows = snapshot.get('rows', [])
    if not rows or point < 0 or point >= len(rows):
        raise ValueError('Select an available acquisition point.')
    value = numeric(rows[point].get(field))
    if value is None:
        raise ValueError('The selected signal is not numeric scalar or 1D data.')
    keys = data_keys(snapshot)
    spec = snapshot.get('start', {}).get('live_plot_fields', {}).get(field, {})
    y_label = spec.get('label', field)
    y_units = spec.get('units', keys.get(field, {}).get('units', ''))
    if value.ndim == 1:
        y = value
        x = np.arange(len(y), dtype=float) if not x_field else numeric(rows[point].get(x_field))
        point_label = snapshot.get('seq_nums', [])[point] if point < len(snapshot.get('seq_nums', [])) else point + 1
        label = f'{signal_label(snapshot, field)} · point {point_label}'
        x_label = x_field or 'Bin'
    else:
        pairs = []
        sequences = snapshot.get('seq_nums', [])
        for index, row in enumerate(rows):
            signal = numeric(row.get(field))
            axis = numeric(row.get(x_field)) if x_field else np.asarray(sequences[index] if index < len(sequences) else index + 1, dtype=float)
            if signal is not None and signal.ndim == 0 and axis is not None and axis.ndim == 0:
                pairs.append((float(axis), float(signal)))
        if not pairs:
            raise ValueError('No numeric scalar values and coordinates were recorded.')
        x, y = np.asarray(pairs, dtype=float).T
        label, x_label = signal_label(snapshot, field), x_field or 'Point'
    if x is None or np.shape(x) != np.shape(y):
        raise ValueError('The chosen coordinate does not match the selected signal shape.')
    return {'x': np.asarray(x, dtype=float), 'y': np.asarray(y, dtype=float),
            'label': label, 'field': field, 'point': point if value.ndim == 1 else None,
            'x_field': x_field, 'x_label': x_label, 'y_label': y_label,
            'x_units': keys.get(x_field, {}).get('units', ''), 'y_units': y_units,
            'uid': snapshot.get('uid', '')}
