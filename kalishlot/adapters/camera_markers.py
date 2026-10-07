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

The list can also be saved to a file and loaded back (commands save_markers and
load_markers), through Windows' own file windows on the lab PC - see
file_dialogs.py. A file holds the whole list plus a little provenance; since
coordinates are uncropped sensor pixels, it means the same thing under any ROI.
Loading either replaces the markers on the camera or is added on top of them.
"""

import json
import math
import uuid
from datetime import datetime
from pathlib import Path

from . import file_dialogs

MAX_MARKERS = 50
MAX_LABEL_CHARS = 40
MAX_TOKEN_CHARS = 40   # id and colour
FILE_FORMAT = 'kalishlot-markers'
FILE_VERSION = 1
FILE_TYPES = (('Marker files', '*.json'), ('All files', '*.*'))
LOAD_MODES = ('replace', 'add')


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


def read_marker_file(path):
    """The validated marker list in a file `save_markers` wrote (a bare list of
    markers is accepted too), or ValueError saying what is wrong with it."""
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError(f'cannot read {Path(path).name}: {error}') from None
    if isinstance(data, dict):
        if data.get('format') != FILE_FORMAT:
            raise ValueError(f'{Path(path).name} is not a marker file')
        data = data.get('markers')
    return clean_markers(data)


def merge_markers(existing, loaded):
    """`loaded` after `existing`, a loaded marker whose id is taken getting a
    new one. ValueError when the two together are more than MAX_MARKERS."""
    if len(existing) + len(loaded) > MAX_MARKERS:
        raise ValueError(f'{len(existing)} + {len(loaded)} markers are more '
                         f'than the {MAX_MARKERS} allowed')
    taken = {marker['id'] for marker in existing}
    merged = [dict(marker) for marker in existing]
    for marker in loaded:
        marker = dict(marker)
        while marker['id'] in taken:
            marker['id'] = uuid.uuid4().hex[:8]
        taken.add(marker['id'])
        merged.append(marker)
    return merged


class CameraMarkersMixin:
    # the file windows; replaced by the tests, which have no screen
    pick_open_file = staticmethod(file_dialogs.ask_open_file)
    pick_save_file = staticmethod(file_dialogs.ask_save_file)
    _markers_dir = None         # where the last file was, for the next window

    def _init_markers(self):
        self._markers = []

    def markers_describe(self):
        return {'markers': [dict(marker) for marker in self._markers]}

    def markers_command(self, name, args):
        """The command's result, or None when it is not a marker command."""
        if name == 'save_markers':
            return self._save_markers()
        if name == 'load_markers':
            return self._load_markers(args.get('mode'))
        if name != 'set_markers':
            return None
        self._set_markers(clean_markers(args.get('markers')))
        return {'ok': True, 'markers': self._markers}

    def _set_markers(self, markers):
        self._markers = markers
        self.emit({'type': 'markers', 'markers': self._markers})

    def _save_markers(self):
        """Ask where, then write the list. `path` is None when cancelled."""
        if not self._markers:
            raise ValueError('there are no markers to save')
        stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        path = self.pick_save_file(
            title='Save the markers as', initial_dir=self._markers_dir,
            filetypes=FILE_TYPES, default_extension='.json',
            initial_name=f'markers_{stamp}.json')
        if path is None:
            return {'ok': True, 'path': None}
        record = {'format': FILE_FORMAT, 'version': FILE_VERSION,
                  'saved': datetime.now().isoformat(timespec='seconds'),
                  'camera': f'{self.type_name}:{self.address}',
                  'markers': self._markers}
        Path(path).write_text(json.dumps(record, indent=1), encoding='utf-8')
        self._markers_dir = str(Path(path).parent)
        return {'ok': True, 'path': path, 'count': len(self._markers)}

    def _load_markers(self, mode):
        """Ask which file, then put its markers on the camera: `mode`
        'replace' drops the present ones, 'add' keeps them. `path` is None
        (and nothing changes) when the window is cancelled."""
        if mode not in LOAD_MODES:
            raise ValueError(f'load_markers needs a mode of {LOAD_MODES}, '
                             f'not {mode!r}')
        path = self.pick_open_file(
            title='Open a marker file', initial_dir=self._markers_dir,
            filetypes=FILE_TYPES)
        if path is None:
            return {'ok': True, 'path': None}
        loaded = read_marker_file(path)
        self._markers_dir = str(Path(path).parent)
        self._set_markers(loaded if mode == 'replace'
                          else merge_markers(self._markers, loaded))
        return {'ok': True, 'path': path, 'count': len(loaded),
                'markers': self._markers}

    def restore_markers(self, saved):
        """Markers from a settings snapshot; a malformed list is dropped
        rather than costing the rest of the restore."""
        try:
            self._markers = clean_markers(saved or [])
        except ValueError:
            self._markers = []
