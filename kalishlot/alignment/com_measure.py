"""measure() for the alignment loop from kalishlot cameras' centre of mass.

The marker loop's original measurement is the Gaussian fit of the 0th-order
mode, captured during a scope scan (alignment/live_capture.py, metrics.py).
When only the spot's position matters, the centre of mass is enough: it needs
no scan and no model, takes milliseconds per frame, and works on any spot, on
any number of cameras at once.

    measure = ComMeasure({'basler': 'basler_camera:24916690',
                          'ximea': 'ximea_camera:QXNAA2506000'})
    target = measure.marker_target({'basler': 'Mirror center', 'ximea': 'Aperture'},
                                   tolerance_px=3)
    AlignmentLoop(axes, measure, target).run()

The cameras stay in the running kalishlot server; this asks it over HTTP for
fresh results (the camera command com_samples). The axes are opened by the
caller, in its own process. The kalishlot aligner box measures the same way
in-process (alignment/signals.py: ComSignals, the same reduce_com).

One evaluation = per camera, the median sensor_x/_y of `samples` CoM results
of frames taken after the call (plus `delay_s`, for the vibrations of the
move to die out). Metrics: <name>_com_x, <name>_com_y in uncropped sensor
pixels (the markers' units), and <name>_com_counts (the summed weights, a
brightness). A camera with fewer than `min_valid_fraction` valid results
(no spot) makes its metrics invalid, which the loop counts as "worse",
never as a position. A spot touching the frame edge or saturated still
counts (alignment/signals.py: reduce_com).

The target is a VectorTarget over every camera's (x, y): the distance is the
root of the summed squares over all cameras, so tolerance_px bounds the
combined miss (3 px over two cameras = e.g. 2.1 px on each).
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request

from alignment.metrics import CaptureMetrics
from alignment.signals import reduce_com
from alignment.targets import VectorTarget, find_marker

KALISHLOT_URL = 'http://localhost:8090'
HTTP_TIMEOUT_S = 15


def http_json(url, payload=None, timeout=HTTP_TIMEOUT_S):
    """GET (payload None) or POST JSON; the decoded reply. Errors carry the server's detail."""
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read().decode('utf-8')).get('detail')
        except Exception:
            detail = error.reason
        raise RuntimeError(f'kalishlot: {detail} ({url})') from None


class ComMeasure:
    """Callable measure() over the centre of mass of one or more kalishlot cameras.

    cameras : {short name: kalishlot device id}; the names prefix the metrics
    samples : CoM results per camera per evaluation (median)
    delay_s : only frames taken this long after the call count (vibrations)
    """

    def __init__(self, cameras, samples=10, delay_s=0.3, min_valid_fraction=0.7,
                 timeout_s=5.0, url=KALISHLOT_URL, request=http_json, now=time.time):
        if not cameras:
            raise ValueError('no cameras given')
        self.cameras = dict(cameras)
        self.samples = int(samples)
        self.delay_s = float(delay_s)
        self.min_valid_fraction = float(min_valid_fraction)
        self.timeout_s = float(timeout_s)
        self.url = url.rstrip('/')
        self._request = request
        self._now = now

    def _command(self, device_id, name, args=None):
        quoted = urllib.parse.quote(device_id, safe='')
        return self._request(f'{self.url}/api/devices/{quoted}/command',
                             {'name': name, 'args': args or {}})

    def devices(self):
        """kalishlot's open devices, {device id: describe()}."""
        return {d['device_id']: d for d in self._request(f'{self.url}/api/devices')}

    def ensure_on(self):
        """Check every camera is open in kalishlot, and turn its centre of mass on."""
        open_devices = self.devices()
        for name, device_id in self.cameras.items():
            if device_id not in open_devices:
                raise RuntimeError(f'camera {name} ({device_id}) is not open in kalishlot; '
                                   f'open: {sorted(open_devices)}')
            if not open_devices[device_id].get('com'):
                self._command(device_id, 'com_on')

    def marker_target(self, markers, tolerance_px=3.0, **kw):
        """VectorTarget putting each camera's CoM on a marker of that camera.

        markers : {camera short name: marker label or id}, read live from kalishlot."""
        open_devices = self.devices()
        metrics, goal, labels = [], [], []
        for name, marker in markers.items():
            device_id = self.cameras[name]
            if device_id not in open_devices:
                raise RuntimeError(f'camera {name} ({device_id}) is not open in kalishlot')
            m = find_marker(open_devices[device_id].get('markers') or [], marker)
            metrics += [f'{name}_com_x', f'{name}_com_y']
            goal += [float(m['x']), float(m['y'])]
            labels.append(f'{name}:{m.get("label") or m.get("id")}')
        return VectorTarget(tuple(metrics), tuple(goal), tolerance=float(tolerance_px),
                            label=' + '.join(labels), unit='px', **kw)

    def __call__(self):
        since = self._now() + self.delay_s
        metrics = {}
        for name, device_id in self.cameras.items():
            reply = self._command(device_id, 'com_samples',
                                  {'n': self.samples, 'since': since,
                                   'timeout': self.delay_s + self.timeout_s})
            metrics.update(self._reduce(name, reply['samples']))
        return CaptureMetrics(metrics=metrics)

    def _reduce(self, name, samples):
        return reduce_com(samples, ('com_x', 'com_y', 'com_counts'), self.min_valid_fraction,
                          prefix=f'{name}_')
