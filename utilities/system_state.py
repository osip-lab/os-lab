"""A record of the lab's measurable parameters at one moment, saved to / read from JSON.

    state = SystemState(note='before walking the lens')
    state.set('large_mirror_piezo.x_v', 37.5)                        # one value, by dotted path
    state.update('ximea', exposure_us=2000, frame_rate_hz=14.28,      # several values of one device
                 roi={'offset_x': 366, 'offset_y': 544, 'width': 432, 'height': 400})
    state['picoscope'] = {'sample_rate_hz': 1e6, 'channels': {'A': {'range_v': 2.0}}}   # a whole section
    state.set('manual.long_arm_ruler_mm', 451.5)
    state.save('states/2026-10-24_before.json')

    old = SystemState.load('states/2026-10-24_before.json')
    old.get('ximea.exposure_us')          # 2000
    old.get('ximea.gain_db')              # None: not recorded in this file (camera was off, say)
    'ximea.gain_db' in old                # False
    old.diff(state)                       # what changed between two records

What it holds. Nested sections of plain JSON values, nothing fixed in advance: the
top level is usually one section per device ('picoscope', 'ximea', 'rigol_dg',
'large_mirror_piezo', ...) plus 'manual' for values typed in by hand, and any section
can hold nested dicts. A new parameter or a new device needs no change here, and a
record holds only what was set: a partial record is the normal case, and a missing
value means "not recorded", never zero.

Conventions (not enforced):
- put the unit in the name: x_v, exposure_us, sample_rate_hz, position_steps, ruler_mm;
- one section per device, named as kalishlot names the device, holding what that
  device's adapter returns from settings_snapshot() - so the function that will reset
  devices from a record can hand each section to that adapter's restore_settings();
- readings that are not settings (an encoder's reported position, a motor's steps
  from home) go in the same device section, under a name that says so, e.g.
  'encoder_position_um', 'steps_from_home'.

The file also stores when it was taken (timestamp: local time with UTC offset, set
at creation, renewed by stamp()) and an optional note. load() also reads a plain
JSON dict without that wrapper ({section: values}).

kalishlot's device_state.json (the last-used settings of every device, restored when
a device is reopened) is a SystemState too: kalishlot/server.py keeps one section per
device id ('picoscope:<serial>', 'ximea:0', ...; the id is type:address) holding that
adapter's settings_snapshot(). So a record saved from the lab has the same layout as
that file, and the two can be compared, merged or subset with the methods below.
Device ids can contain dots (an IP address): address them with one-element tuples,
state.get(('rigol_dg:192.168.1.5',)).

Dotted paths split on '.'; for a key that itself contains a dot, pass the path as a
tuple: state.set(('ximea', 'lens 0.5x'), True). Values are checked when set (numpy
scalars and arrays become plain numbers and lists), so a bad value fails at once,
not at save time.

Dependency-free (json / datetime / pathlib), like the other utilities.
"""

import copy
import json
import math
from datetime import datetime
from pathlib import Path

FORMAT = 'os-lab-system-state'
VERSION = 1


def _split(path):
    if isinstance(path, (tuple, list)):
        parts = [str(p) for p in path]
    else:
        parts = str(path).split('.')
    if not parts or any(p == '' for p in parts):
        raise KeyError(f'bad path {path!r}')
    return parts


def to_plain(value, where='value'):
    """value as plain JSON types (dict / list / str / int / float / bool / None).
    numpy scalars and arrays (anything with .item() / .tolist()) are converted."""
    if value is None or type(value) in (bool, str, int, float):     # exact types: np.float64 is a float
        return value
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(f'{where}: keys must be strings, got {k!r}')
            out[k] = to_plain(v, f'{where}.{k}')
        return out
    if isinstance(value, (list, tuple)):
        return [to_plain(v, f'{where}[{i}]') for i, v in enumerate(value)]
    if hasattr(value, 'tolist'):                     # numpy array or scalar
        return to_plain(value.tolist(), where)
    if hasattr(value, 'item'):
        return to_plain(value.item(), where)
    if isinstance(value, Path):
        return str(value)
    for t in (bool, int, float, str):                 # other subclasses of the plain types
        if isinstance(value, t):
            return t(value)
    raise TypeError(f'{where}: {type(value).__name__} is not a JSON value ({value!r})')


def _flatten(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, dict) and v:
            yield from _flatten(v, prefix + (k,))
        else:
            yield prefix + (k,), v


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


class SystemState:
    def __init__(self, values=None, note='', timestamp=None):
        self.values = to_plain(dict(values or {}), 'values')
        self.note = note
        self.timestamp = timestamp or datetime.now().astimezone().isoformat(timespec='seconds')

    # ------------------------------------------------------------ access
    def set(self, path, value):
        """Set one value; missing sections on the way are created."""
        parts = _split(path)
        node = self.values
        for i, p in enumerate(parts[:-1]):
            nxt = node.setdefault(p, {})
            if not isinstance(nxt, dict):
                raise KeyError(f'{".".join(parts[:i + 1])} holds a value, not a section')
            node = nxt
        node[parts[-1]] = to_plain(value, '.'.join(parts))
        return self

    def update(self, section, values=None, **kw):
        """Set several values in one section (nested dicts are merged, not replaced)."""
        for k, v in {**(values or {}), **kw}.items():
            if isinstance(v, dict):
                for sub, leaf in _flatten(v):
                    self.set(_split(section) + [k] + list(sub), leaf)
                if not v:
                    self.set(_split(section) + [k], {})
            else:
                self.set(_split(section) + [k], v)
        return self

    def get(self, path, default=None):
        """The value (or section, as a copy) at path; `default` when not recorded."""
        node = self.values
        for p in _split(path):
            if not isinstance(node, dict) or p not in node:
                return default
            node = node[p]
        return copy.deepcopy(node)

    def stamp(self):
        """Set the timestamp to now (e.g. before saving a record kept up to date)."""
        self.timestamp = datetime.now().astimezone().isoformat(timespec='seconds')
        return self

    def remove(self, path):
        parts = _split(path)
        node = self.values
        for p in parts[:-1]:
            node = node.get(p) if isinstance(node, dict) else None
            if node is None:
                return
        if isinstance(node, dict):
            node.pop(parts[-1], None)

    def __contains__(self, path):
        sentinel = object()
        return self.get(path, sentinel) is not sentinel

    def __getitem__(self, path):
        sentinel = object()
        v = self.get(path, sentinel)
        if v is sentinel:
            raise KeyError(f'{path!r} is not recorded')
        return v

    def __setitem__(self, path, value):
        self.set(path, value)

    def sections(self):
        return list(self.values)

    def flat(self):
        """{'section.key.sub': value} for every recorded value."""
        return {'.'.join(p): copy.deepcopy(v) for p, v in _flatten(self.values)}

    # ----------------------------------------------------- combining
    def subset(self, *paths):
        """A new state with only these sections / values (unrecorded ones are skipped)."""
        out = SystemState(note=self.note, timestamp=self.timestamp)
        for path in paths:
            if path in self:
                out.set(path, self.get(path))
        return out

    def merged(self, other):
        """A new state: this one, with every value recorded in `other` set on top.
        E.g. a camera-off record plus a later camera-only record."""
        out = SystemState(self.values, note=self.note, timestamp=self.timestamp)
        for p, v in _flatten(other.values):
            out.set(p, v)
        return out

    def diff(self, other):
        """{path: (value here, value in other)} for every value that differs or is
        recorded in only one of the two (None on the side where it is missing)."""
        a, b = self.flat(), other.flat()
        return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b))
                if k not in a or k not in b or not _same(a[k], b[k])}

    # -------------------------------------------------------------- JSON
    def to_dict(self):
        return {'format': FORMAT, 'version': VERSION, 'timestamp': self.timestamp, 'note': self.note,
                'values': copy.deepcopy(self.values)}

    @classmethod
    def from_dict(cls, d):
        """A record written by to_dict(), or a plain {section: values} dict."""
        if isinstance(d, dict) and d.get('format') == FORMAT:
            if d.get('version', 1) > VERSION:
                raise ValueError(f'system state version {d["version"]} is newer than this code ({VERSION})')
            return cls(d.get('values') or {}, note=d.get('note', ''), timestamp=d.get('timestamp'))
        if not isinstance(d, dict):
            raise ValueError(f'not a system state: a JSON {type(d).__name__}')
        return cls(d, timestamp=None)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding='utf-8')
        return path

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    def __eq__(self, other):
        return isinstance(other, SystemState) and not self.diff(other)

    def __repr__(self):
        n = len(self.flat())
        return f'SystemState({self.timestamp}, {n} values in {self.sections()}{", " + repr(self.note) if self.note else ""})'
