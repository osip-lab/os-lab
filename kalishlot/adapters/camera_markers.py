"""Labelled marker circles shared by camera adapters.

Mixin adding the markers vocabulary on top of DeviceAdapter: the command
set_markers, the 'markers' event, and a 'markers' entry in describe() and in
the persisted settings. A marker is an annotation - "the mode sat here with
the long arm at 45 cm" - so the list is kept with the camera rather than in a
browser: it survives reloads and server restarts (device_state.json) and every
viewer sees the same set, live.

Coordinates are UNCROPPED sensor pixels, unlike the fit and the guess, which
are in pixels of the current frame: a marker records a place on the sensor,
and has to stay on it when the ROI changes - that is the whole point of
comparing where the mode sits across configurations. The box converts.

A marker made from a fit result also carries `ellipse` ({a, b, angle}: the 2 std
semi-axes and rotation); it is drawn as that ellipse instead of the circle, and
renamed, hidden and deleted like any other. Its `r` is the mean semi-axis.

The browser owns the editing: it sends the whole list after every change
(add, rename, show/hide, delete) and draws from the 'markers' event. One
command for all of it keeps the server a store with validation, not a second
copy of the GUI's logic.
"""

import math

MAX_MARKERS = 50
MAX_LABEL_CHARS = 40
MAX_TOKEN_CHARS = 40   # id and colour


def _clean_marker(raw):
    """One validated marker dict, or ValueError naming what is wrong."""
    if not isinstance(raw, dict):
        raise ValueError(f'a marker must be an object, not {raw!r}')
    marker = {}
    for key in ('x', 'y', 'r'):
        try:
            value = float(raw[key])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f'marker needs a number {key!r}') from None
        if not math.isfinite(value) or (key == 'r' and value < 0):
            raise ValueError(f'marker {key!r} = {value} is out of range')
        marker[key] = round(value, 2)
    for key, limit in (('id', MAX_TOKEN_CHARS), ('label', MAX_LABEL_CHARS),
                       ('color', MAX_TOKEN_CHARS)):
        value = str(raw.get(key, '')).strip()
        if len(value) > limit:
            raise ValueError(f'marker {key!r} is longer than {limit} characters')
        marker[key] = value
    if not marker['id']:
        raise ValueError('marker needs an id')
    marker['visible'] = bool(raw.get('visible', True))
    ellipse = raw.get('ellipse')
    if ellipse is not None:
        marker['ellipse'] = _clean_ellipse(ellipse)
    return marker


def _clean_ellipse(raw):
    """{'a', 'b', 'angle'}: semi-axes in sensor px and the rotation in radians."""
    if not isinstance(raw, dict):
        raise ValueError(f'marker ellipse must be an object, not {raw!r}')
    ellipse = {}
    for key in ('a', 'b', 'angle'):
        try:
            value = float(raw[key])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f'marker ellipse needs a number {key!r}') from None
        if not math.isfinite(value) or (key != 'angle' and value < 0):
            raise ValueError(f'marker ellipse {key!r} = {value} is out of range')
        ellipse[key] = round(value, 4)
    return ellipse


def clean_markers(raw_list):
    if not isinstance(raw_list, list):
        raise ValueError('markers must be a list')
    if len(raw_list) > MAX_MARKERS:
        raise ValueError(f'at most {MAX_MARKERS} markers')
    markers = [_clean_marker(raw) for raw in raw_list]
    ids = [marker['id'] for marker in markers]
    if len(set(ids)) != len(ids):
        raise ValueError('marker ids must be unique')
    return markers


class CameraMarkersMixin:
    def _init_markers(self):
        self._markers = []

    def markers_describe(self):
        return {'markers': [dict(marker) for marker in self._markers]}

    def markers_command(self, name, args):
        """The command's result, or None when it is not a marker command."""
        if name != 'set_markers':
            return None
        self._markers = clean_markers(args.get('markers'))
        self.emit({'type': 'markers', 'markers': self._markers})
        return {'ok': True, 'markers': self._markers}

    def restore_markers(self, saved):
        """Markers from a settings snapshot; a malformed list is dropped
        rather than costing the rest of the restore."""
        try:
            self._markers = clean_markers(saved or [])
        except ValueError:
            self._markers = []
