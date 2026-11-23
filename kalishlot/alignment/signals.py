"""Signals the aligner can measure, and the values they can be sent to.

A signal is one number measured by an open kalishlot device, with an id
'<device_id>/<quantity>', e.g. 'basler_camera:24916690/com_x'. Each kind of
signal comes from a provider:

    provider.available(devices)  -> [{'id', 'label', 'unit', 'device_id'}, ...]
    provider.measure(ids, since) -> {id: Metric}   (alignment/metrics.Metric)

`devices` is kalishlot's {device_id: adapter}. A provider measures all the
ids it is asked for in one go (one capture per camera, not one per signal),
using only frames taken after host time `since`.

Providers so far:
    ComSignals    camera centre of mass: com_x, com_y (uncropped sensor px,
                  the markers' units) and com_counts (summed weights)
Later: the live measurement's metrics (intensity, ellipticity, splitting,
via adapters/live_measure.py) and spectrum signals. Add a provider class and
list it in PROVIDERS; the aligner box offers its signals by itself.

Target values (resolve_value) are resolved once, when a run starts:
    {'kind': 'number', 'value': 12.5}
    {'kind': 'marker', 'device_id': ..., 'marker': <id or label>, 'coord': 'x' | 'y'}
    {'kind': 'hold'}      the signal's value at the start of the run
"""

import statistics
import time

from alignment.metrics import CaptureMetrics, Metric
from alignment.targets import find_marker


def robust_std(values):
    if len(values) < 2:
        return float('nan')
    med = statistics.median(values)
    return 1.4826 * statistics.median(abs(v - med) for v in values)


COM_QUANTITIES = {'com_x': ('sensor_x', 'px'), 'com_y': ('sensor_y', 'px'),
                  'com_counts': ('counts', 'counts')}


def reduce_com(samples, quantities, min_valid_fraction=0.7, prefix=''):
    """Median of CoM results per quantity -> {prefix + quantity: Metric}.

    Only results without a spot (invalid) are dropped. A spot touching the frame
    edge (clipped) still counts: its centre is pulled inwards, but it points the
    right way, and a spot steered onto a marker ends up inside the frame anyway
    (lab, 2026-11-23: a large spot entering from the edge was refused before).
    Saturated ones count too (only the weights clip). Fewer than
    min_valid_fraction usable results make every quantity invalid, with the
    commonest reason."""
    good = [s for s in samples if s.get('valid') and s.get('sensor_x') is not None]
    if len(good) < min_valid_fraction * max(len(samples), 1):
        reasons = [s.get('reason') or 'invalid' for s in samples if s not in good]
        reason = (f'{len(good)}/{len(samples)} valid centre-of-mass results'
                  + (f' ({statistics.mode(reasons)})' if reasons else ''))
        return {prefix + q: Metric(prefix + q, n=len(good), reason=reason) for q in quantities}
    out = {}
    for q in quantities:
        vals = [float(s[COM_QUANTITIES[q][0]]) for s in good]
        out[prefix + q] = Metric(prefix + q, value=statistics.median(vals),
                                 spread=robust_std(vals), n=len(vals), valid=True)
    return out


def split_id(signal_id):
    device_id, _, quantity = signal_id.rpartition('/')
    if not device_id:
        raise ValueError(f'signal id must be <device_id>/<quantity>, got {signal_id!r}')
    return device_id, quantity


class ComSignals:
    """Centre of mass of every open camera (an adapter with com_samples)."""

    name = 'com'

    def __init__(self, samples=10, min_valid_fraction=0.7, timeout_s=5.0):
        self.samples = int(samples)
        self.min_valid_fraction = float(min_valid_fraction)
        self.timeout_s = float(timeout_s)

    @staticmethod
    def _is_camera(adapter):
        return hasattr(adapter, 'com_samples') and hasattr(adapter, 'com_command')

    def available(self, devices):
        out = []
        for device_id, adapter in devices.items():
            if not self._is_camera(adapter):
                continue
            short = device_id.split(':')[0].replace('_camera', '')
            for q, (_, unit) in COM_QUANTITIES.items():
                axis = q.split('_')[1]
                out.append({'id': f'{device_id}/{q}', 'device_id': device_id, 'unit': unit,
                            'label': f'{short} CoM {axis} ({device_id.split(":", 1)[-1]})'})
        return out

    def owns(self, signal_id, devices):
        device_id, quantity = split_id(signal_id)
        return quantity in COM_QUANTITIES and self._is_camera(devices.get(device_id))

    def prepare(self, ids, devices):
        """Turn the centre of mass on where it is off (shown in the camera box)."""
        for device_id in {split_id(i)[0] for i in ids}:
            adapter = devices.get(device_id)
            if adapter is None:
                raise RuntimeError(f'camera {device_id} is not open')
            if not adapter.com_describe().get('com'):
                adapter.command('com_on', {})

    def measure(self, ids, devices, since):
        by_camera = {}
        for i in ids:
            device_id, quantity = split_id(i)
            by_camera.setdefault(device_id, []).append(quantity)
        out = {}
        for device_id, quantities in by_camera.items():
            adapter = devices.get(device_id)
            if adapter is None:
                for q in quantities:
                    out[f'{device_id}/{q}'] = Metric(f'{device_id}/{q}', reason='camera closed')
                continue
            wait = max(since - time.time(), 0.0) + self.timeout_s
            try:
                samples = adapter.com_samples(self.samples, since=since, timeout=wait)
            except (RuntimeError, TimeoutError) as error:
                for q in quantities:
                    out[f'{device_id}/{q}'] = Metric(f'{device_id}/{q}', reason=str(error))
                continue
            out.update(reduce_com(samples, quantities, self.min_valid_fraction,
                                  prefix=f'{device_id}/'))
        return out


PROVIDERS = (ComSignals,)


class SignalSet:
    """The providers, and measure() over a chosen list of signal ids."""

    def __init__(self, devices_fn, providers=None):
        self.devices_fn = devices_fn          # () -> {device_id: adapter}
        self.providers = list(providers) if providers is not None else [p() for p in PROVIDERS]

    def available(self):
        devices = self.devices_fn()
        return [dict(s, provider=p.name) for p in self.providers for s in p.available(devices)]

    def _provider(self, signal_id, devices):
        for p in self.providers:
            if p.owns(signal_id, devices):
                return p
        raise ValueError(f'no open device measures {signal_id!r}')

    def group(self, ids):
        devices = self.devices_fn()
        groups = {}
        for i in ids:
            groups.setdefault(self._provider(i, devices), []).append(i)
        return groups, devices

    def prepare(self, ids):
        groups, devices = self.group(ids)
        for p, sub in groups.items():
            p.prepare(sub, devices)

    def measure(self, ids, delay_s=0.0):
        """CaptureMetrics of `ids`, from data taken at least delay_s after the call."""
        since = time.time() + float(delay_s)
        groups, devices = self.group(ids)
        metrics = {}
        for p, sub in groups.items():
            metrics.update(p.measure(sub, devices, since))
        return CaptureMetrics(metrics=metrics)


def resolve_value(value, signal_id, devices, start_metrics=None):
    """A target value spec -> float (see the module docstring)."""
    kind = (value or {}).get('kind', 'number')
    if kind == 'number':
        return float(value['value'])
    if kind == 'marker':
        adapter = devices.get(value['device_id'])
        if adapter is None:
            raise ValueError(f'camera {value["device_id"]} is not open')
        markers = adapter.describe().get('markers') or []
        m = find_marker(markers, value['marker'])
        return float(m[value.get('coord', 'x')])
    if kind == 'hold':
        metric = (start_metrics or {}).get(signal_id)
        if metric is None or not metric.valid:
            raise ValueError(f'cannot hold {signal_id}: no valid value at the start '
                             f'({metric.reason if metric else "not measured"})')
        return float(metric.value)
    raise ValueError(f'unknown target value kind {kind!r}')
