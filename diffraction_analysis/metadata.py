"""Read-only outcome and overview presentation for Bluesky run metadata."""

import datetime as dt
import math
from collections.abc import Mapping


def _count(value):
    try:
        number = float(value)
        return int(number) if math.isfinite(number) and number >= 0 and number.is_integer() else None
    except (ValueError, TypeError, OverflowError):
        return None


def run_outcome(start, stop):
    """Report evidence from Stop; never label a missing Stop as a live run.

    RunEngine.stop() also emits success. An undersized successful run is only
    labelled Partial: the documents cannot prove an operator requested a stop.
    """
    start = start or {}
    planned = _count(start.get('num_points'))
    if planned is None:
        planned = _count((start.get('plan_pattern_args') or {}).get('num_steps'))
    events = (stop or {}).get('num_events') or {}
    acquired = _count(events.get('primary'))
    raw = str((stop or {}).get('exit_status') or '').strip().lower()
    reason = str((stop or {}).get('reason') or '').strip()
    note = ''
    if stop is None:
        label = 'Unfinished'
        note = 'No Stop document; this does not establish whether acquisition is still active.'
    elif raw == 'success':
        if acquired is not None and planned is not None and acquired < planned:
            label = 'Partial'
            note = 'Successful exit with fewer primary events than planned; may have been stopped early.'
        else:
            label = 'Complete'
            note = 'Recorded success. An operator stop may also be recorded as success by Bluesky.'
    else:
        label = {'fail': 'Failed', 'failed': 'Failed', 'abort': 'Aborted',
                 'aborted': 'Aborted', 'stop': 'Stopped', 'stopped': 'Stopped'}.get(raw, raw.capitalize() or 'Unknown')
    return {'status': label, 'exit_status': raw, 'reason': reason, 'note': note,
            'acquired': acquired, 'planned': planned}


def _timestamp(value):
    try:
        return dt.datetime.fromtimestamp(float(value), dt.timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
    except (ValueError, TypeError, OverflowError, OSError):
        return 'Not recorded'


def _quantity(value, units=''):
    if isinstance(value, (int, float)):
        text = f'{value:.7g}'
    else:
        text = str(value)
    return f'{text} {units}'.strip()


def read_metadata_details(run, metadata):
    """Read small stored baseline streams, never PVs or diffraction arrays.

    Older runs have no catalog baseline summary. Keep any recovered readings
    explicitly under viewer_details, without rewriting acquisition metadata.
    An unavailable baseline is ancillary and must not prevent plotting a run.
    """
    from .history import _run_streams, _stream_descriptor_docs, _to_plain_value

    details = {'baseline_readings': dict(metadata.get('baseline_readings') or {}),
               'baseline_source': 'Catalog baseline summary', 'field_units': {},
               'monitor_streams': [], 'notes': []}
    start = metadata.get('start') or {}
    try:
        streams = _run_streams(run)
        details['monitor_streams'] = [str(key) for key in streams if str(key).endswith('_monitor')]
        for name in ('primary', 'baseline'):
            if name not in streams:
                continue
            stream = streams[name]
            descriptors = _stream_descriptor_docs(start.get('uid', ''), name, stream, start)
            keys = {key: spec for descriptor in descriptors for key, spec in descriptor.get('data_keys', {}).items()}
            details['field_units'].update({key: spec.get('units', '') for key, spec in keys.items()})
            if name != 'baseline' or details['baseline_readings']:
                continue
            dataset = stream.read()
            for key, variable in dataset.data_vars.items():
                spec = keys.get(key, {})
                values = getattr(variable, 'values', variable)
                # Baseline readbacks are scalars per event. Never turn a
                # detector array or timestamps into an environment reading.
                if key == 'time' or spec.get('shape') or getattr(values, 'ndim', 0) != 1 or not len(values):
                    continue
                details['baseline_readings'][key] = {
                    'start': _to_plain_value(values[0]), 'end': _to_plain_value(values[-1]) if len(values) > 1 else None,
                    'units': spec.get('units', ''),
                }
            details['baseline_source'] = 'Stored baseline stream (first / last readings)'
    except Exception as exc:
        details['notes'].append(f'Additional stored metadata unavailable: {exc}')
    return {**metadata, 'viewer_details': details}


def acquisition_details(metadata):
    """Present recorded scan geometry and timing, preserving units and zeros."""
    start = metadata.get('start') or {}
    args = start.get('plan_args') or {}
    pattern = start.get('plan_pattern_args') or {}
    params = {**args, **pattern}
    config = start.get('det_config') or {}
    units = dict((metadata.get('viewer_details') or {}).get('field_units') or {})
    for descriptor in metadata.get('stream_descriptors') or []:
        units.update({key: spec.get('units', '') for key, spec in descriptor.get('data_keys', {}).items()})
    timing = {}
    exposure = next((value for value in (args.get('acquire_time'), config.get('acquire_time'),
                                        (start.get('acquisition_monitor') or {}).get('duration')) if value is not None), None)
    if exposure is not None:
        timing['Exposure / point (set)'] = _quantity(exposure, 's')
    if config.get('software_dwell_time') is not None:
        timing['Software dwell / point'] = _quantity(config['software_dwell_time'], 's')
    if args.get('delay') is not None:
        delay = args['delay']
        timing['Delay between points'] = _quantity(delay, 's')
    if start.get('estimated_total_time_s') is not None:
        timing['Estimated run duration'] = _quantity(start['estimated_total_time_s'], 's')
    if config.get('acquire_times'):
        timing['Recorded acquisition presets'] = {key: _quantity(value, 's') for key, value in config['acquire_times'].items()}

    motors = start.get('motors') or []
    geometry = {}
    axes = [('motor', ''), ('motor1', '1'), ('motor2', '2'), ('motor_outer', '_outer'), ('motor_inner', '_inner')]
    for name, suffix in axes:
        motor = params.get(name)
        if not motor and name == 'motor' and len(motors) == 1:
            motor = motors[0]
        if not motor:
            continue
        motor = str(motor)
        row = {}
        for label, key in [('Start (planned)', 'start_pos'), ('Stop (planned)', 'stop_pos'), ('Step size', 'step_size')]:
            value = params.get(key + suffix)
            if value is not None:
                row[label] = _quantity(value, units.get(motor, ''))
        count = params.get('num_steps' + suffix, params.get('num_steps'))
        if count is not None:
            row['Number of points'] = count
        if name == 'motor' and params.get('position_list') is not None:
            row['Positions (planned)'] = params['position_list']
        if row:
            geometry[motor] = row
    # Standard Bluesky scan/grid_scan metadata uses triples of motor, start,
    # stop (or quadruples with a point count for an outer-product scan).
    if not geometry and motors and isinstance(pattern.get('args'), (list, tuple)):
        values = pattern['args']
        stride = 4 if start.get('plan_pattern') == 'outer_product' else 3
        for index, motor in enumerate(motors):
            offset = index * stride
            if len(values) < offset + stride:
                break
            row = {'Start (planned)': _quantity(values[offset + 1], units.get(motor, '')),
                   'Stop (planned)': _quantity(values[offset + 2], units.get(motor, ''))}
            count = values[offset + 3] if stride == 4 else pattern.get('num')
            if count is not None:
                row['Number of points'] = count
            geometry[motor] = row
    result = {'Timing': timing, 'Scan geometry': geometry or {'Details': 'No recorded scan geometry (for example, a count run).'}}
    if start.get('plan_pattern'):
        result['Scan pattern'] = start['plan_pattern']
    if start.get('shape'):
        result['Scan shape'] = start['shape']
    detectors = start.get('detector_configurations') or {}
    settings = {}
    for name, values in detectors.items():
        row = {}
        for label, key in [('Exposure / point', 'acquire_time'), ('Bins', 'nbins'), ('Lower threshold', 'soft_lld')]:
            if values.get(key) is not None:
                row[label] = _quantity(values[key], 's') if key == 'acquire_time' else values[key]
        if row:
            settings[name] = row
    if not settings:
        settings = {label: config[key] for label, key in [('Bins', 'nbins'), ('Lower threshold', 'soft_lld')] if config.get(key) is not None}
    if settings:
        result['Detector settings'] = settings
    if start.get('file_type'):
        result['Selected file format'] = start['file_type']
    return result


def environment_readings(metadata):
    details = metadata.get('viewer_details') or {}
    baseline = details.get('baseline_readings', metadata.get('baseline_readings')) or {}
    streams = details.get('monitor_streams') or []
    rows = []
    for name, reading in sorted(baseline.items(), key=lambda pair: ('reactor_power' not in pair[0].lower(), pair[0])):
        if name.endswith('_setpoint') or not isinstance(reading, Mapping):
            continue
        monitor = f'{name}_monitor'
        rows.append({'signal': name, 'start': reading.get('start'), 'end': reading.get('end'),
                     'units': reading.get('units', ''), 'monitor': monitor if monitor in streams else '',
                     'source': details.get('baseline_source', 'Catalog baseline summary')})
    return rows


def run_overview(metadata, *, point_count=None):
    """Ordered sections of useful fields; originals remain in All metadata."""
    start = metadata.get('start') or {}
    stop = metadata.get('stop')
    outcome = run_outcome(start, stop)
    run = {'Status': outcome['status'], 'Exit status': outcome['exit_status'] or 'Not recorded'}
    if outcome['reason']:
        run['Reason'] = outcome['reason']
    if outcome['note'] and outcome['status'] != 'Complete':
        run['Status detail'] = outcome['note']
    for label, key in [('Title', 'title'), ('Scan ID', 'scan_id'), ('Plan', 'plan_name'), ('Run UID', 'uid')]:
        if start.get(key) is not None:
            run[label] = start[key]
    run['Started'] = _timestamp(start.get('time'))
    run['Ended'] = _timestamp((stop or {}).get('time'))
    try:
        duration = float(stop['time']) - float(start['time'])
        if math.isfinite(duration) and duration >= 0:
            run['Duration'] = f'{duration:.3g} s'
    except (TypeError, KeyError, ValueError, OverflowError):
        pass
    acquisition = {}
    for label, value in [('Points recorded', outcome['acquired']),
                         ('Points planned', outcome['planned']), ('Points loaded', point_count),
                         ('Data type', start.get('data_type')), ('Detectors', start.get('detectors')),
                         ('Motors', start.get('motors'))]:
        if value is not None:
            acquisition[label] = value
    detail = acquisition_details(metadata)
    acquisition.update(detail['Timing'])
    experiment = {label: start[key] for label, key in [
        ('Sample', 'sample'), ('Experiment', 'experiment'), ('Proposal', 'proposal_id'),
        ('User', 'user'), ('Operator', 'operator'), ('Gauge volume', 'gauge_volume'),
        ('Facility', 'facility'), ('Beamline', 'beamline_id'), ('Mode', 'bluesky_mode'), ('Notes', 'notes')]
        if start.get(key) not in (None, '')}
    export = metadata.get('file_export') or {}
    storage = {'Export status': export.get('status') or 'Not recorded (older run)'}
    for label, key in [('File', 'file_uri'), ('Format', 'format'), ('SHA-256', 'sha256'),
                       ('Export error', 'error'), ('Recovery journal', 'journal')]:
        if export.get(key):
            storage[label] = export[key]
    sections = {'Run': run, 'Acquisition': acquisition}
    sections['Acquisition details'] = detail
    if experiment:
        sections['Sample / experiment'] = experiment
    sections['Storage'] = storage
    baseline = (metadata.get('viewer_details') or {}).get('baseline_readings', metadata.get('baseline_readings'))
    if isinstance(baseline, Mapping) and baseline:
        sections['Baseline / environment'] = baseline
    if metadata.get('recovered_plotting_metadata'):
        sections['Plotting metadata recovered for this view'] = metadata['recovered_plotting_metadata']
    return sections
