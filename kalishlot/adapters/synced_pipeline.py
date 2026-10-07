"""The synced video + scope pipeline as a (virtual) kalishlot device.

Not hardware: this adapter owns the parameters of a pico_scope/ mode-video run
and, in the next step of its life, the run itself. It exists as an adapter so
that its box gets everything a device box gets for free - re-attaching after a
page load, a satellite window, saved settings, the same command route.

It replaces editing pico_scope/run_config_local.py. A run is described by

  * `edited`  - the parameters the user changed in the box, {"section.NAME":
                value}; everything else keeps the config file's value, which is
                what the widget shows until it is edited. Only these go to the
                run, as an override layer over the file (run_config.run_params)
  * `adopt`   - per parameter, whether the capture takes the camera/scope box's
                value or decides itself (exposure, gain, ... see ADOPT_ROWS)
  * the camera to use, when more than one is open

PARAMS below is the one list of what the box offers; the frontend draws its
widgets from it and the smoke test checks every name against run_config.SECTIONS.
"""

import collections
import json
import os
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from pico_scope import run_config  # noqa: E402

from . import file_dialogs  # noqa: E402
from .base import DeviceAdapter  # noqa: E402

CAMERA_TYPES = {'ximea_camera': 'ximea', 'basler_camera': 'basler'}
# The formats each camera type offers (their drivers' `formats`, which need the
# camera to answer; the names a capture's PIXEL_FORMAT takes)
PIXEL_FORMATS = {'ximea_camera': ['Mono8', 'Mono10'],
                 'basler_camera': ['Mono8', 'Mono12p', 'Mono12']}
SCOPE_TYPE = 'picoscope'
FOLDER_PARAM = 'capture.OUTPUT_ROOT'
SCOPE_CHANNELS = ('A', 'B', 'C', 'D')
FORBIDDEN_NAME_CHARS = '<>:"/\|?*'     # what Windows will not take in a name

# --- what the box offers ---------------------------------------------------
# (key, kind, label, group, extras). key is "section.NAME" with the names of
# run_config.SECTIONS. kind: float / int / bool / text / choice / optfloat
# (blank = None) / intlist / folder. `auto_of` marks a parameter that only
# matters while that adopt row is unticked (the capture decides, from these).
# group: 'main' always shown, 'advanced' folded away.
PARAMS = []


def _p(key, kind, label, group='main', **extras):
    PARAMS.append({'key': key, 'kind': kind, 'label': label, 'group': group,
                   **extras})


_p('capture.OUTPUT_ROOT', 'folder', 'save folder')
_p('capture.CAPTURE_DURATION_S', 'float', 'measurement length', unit='s',
   min=0.01)
_p('capture.FRAMES_FORMAT', 'choice', 'frames compression',
   choices=['h264', 'lossless'])
_p('capture.H264_CRF', 'int', 'h264 quality (CRF: 0 = lossless, 51 = harshest)',
   min=0, max=51)
_p('capture.BINNING', 'choice', 'binning', choices=[1, 2, 4])
# '' (the first button, "deepest") is None in the run: the camera's deepest.
# A row of buttons, only the ones the chosen camera offers (PIXEL_FORMATS).
_p('capture.PIXEL_FORMAT', 'choice', 'pixel format',
   choices=['', 'Mono8', 'Mono10', 'Mono12', 'Mono12p'], optional=True,
   segmented=True, labels={'': 'deepest'})
_p('capture.LOCATE_FIRST', 'bool', 'locate the mode first (auto ROI)',
   auto_of='roi')
_p('capture.STRICT_LEVELS', 'bool', 'refuse a clipped capture')
_p('capture.SCOPE_CHANNEL', 'choice', 'scope: transmission channel',
   choices=list(SCOPE_CHANNELS))
_p('capture.SCOPE_AUX_CHANNEL', 'choice', 'scope: aux channel',
   choices=list(SCOPE_CHANNELS) + [''])
_p('capture.SCOPE_AUX_LABEL', 'text', 'scope: aux label')
_p('capture.SCOPE_PAD_S', 'float', 'scope: padding either side', unit='s',
   min=0.0)
# blank = no tail; the checkbox only means something with a tail (`needs`)
_p('capture.TRAILING_SCOPE_S', 'optfloat', 'Trailing scope capture', unit='s',
   min=0.01, placeholder='None')
_p('capture.TRAILING_SCOPE_AUX_FG', 'bool',
   'Apply secondary FG channel to trailing capture',
   needs='capture.TRAILING_SCOPE_S')
_p('capture.MASK_THRESHOLD', 'float', 'mode mask threshold', min=0.0, max=1.0)
# only while the matching adopt row is unticked
_p('capture.FRAME_RATE_HZ', 'float', 'requested frame rate', unit='Hz',
   min=1.0, auto_of='frame_rate')
_p('capture.GAIN_DB', 'float', 'starting gain', unit='dB', auto_of='gain')

_p('capture.TARGET_PEAK_FRACTION', 'float', 'gain: target peak', 'advanced',
   min=0.05, max=1.0, auto_of='gain')
_p('capture.MAX_SATURATED_FRACTION', 'float', 'max saturated fraction',
   'advanced', min=0.0, max=1.0)
_p('capture.LEVEL_BURSTS', 'int', 'light check: bursts', 'advanced', min=1)
_p('capture.LEVEL_BURST_FRAMES', 'optfloat', 'light check: frames per burst',
   'advanced')
_p('capture.LEVEL_SAFETY', 'float', 'light check: safety factor', 'advanced',
   min=1.0)
_p('capture.LEVEL_TOO_DIM_FRACTION', 'float', 'light check: too dim below',
   'advanced', min=0.0, max=1.0)
_p('capture.LEVEL_CLIPPED_STEP_DB', 'float', 'light check: step when clipped',
   'advanced', unit='dB', min=0.0)
_p('capture.ROI_WIDTH', 'optfloat', 'ROI width (blank = whole)', 'advanced',
   auto_of='roi')
_p('capture.ROI_HEIGHT_CANDIDATES', 'intlist', 'ROI heights tried',
   'advanced', auto_of='roi')
_p('capture.ROI_MIN_MARGIN_ROWS', 'int', 'ROI margin', 'advanced', min=0,
   auto_of='roi')
_p('capture.ROI_OFFSET_X', 'int', 'ROI x offset', 'advanced', min=0,
   auto_of='roi')
_p('capture.SCOPE_AUTORANGE_PROBE_S', 'float', 'scope auto-range: probe',
   'advanced', unit='s', min=0.01, auto_of='scope_range')
_p('capture.SCOPE_AUTORANGE_MARGIN', 'float', 'scope auto-range: margin',
   'advanced', min=1.0, auto_of='scope_range')
_p('capture.SCOPE_AUTORANGE_MIN_V', 'float', 'scope auto-range: smallest',
   'advanced', unit='V', min=0.0, auto_of='scope_range')
_p('capture.THROUGHPUT_BPS', 'optfloat', 'camera throughput cap', 'advanced',
   unit='B/s')

_p('sync.SEARCH_WINDOW_S', 'float', 'sync: search window', 'sync', unit='s',
   min=0.0)
_p('sync.TIME_COLUMN', 'text', 'sync: time column', 'sync')
_p('sync.SIGNAL_COLUMN', 'text', 'sync: signal column', 'sync')
_p('sync.AUX_COLUMN', 'text', 'sync: aux column', 'sync', optional=True)
_p('sync.AUX_LABEL', 'text', 'sync: aux label', 'sync', optional=True)
_p('show.SHADE_ALPHA', 'float', 'plot: shading alpha', 'sync', min=0.0, max=1.0)
_p('show.FIT_REBINNING', 'int', 'plot: fit rebinning', 'sync', min=1)
_p('show.RENORMALIZE_PERCENTILE', 'float', 'plot: renormalise percentile',
   'sync', min=0.0, max=100.0)

# Asked fresh in the pop-up shown when a capture starts (not a standing box
# parameter - a value left over from the last run must never be saved by
# accident). Both reach the run: the folder label becomes the session folder's
# FOLDER_SUFFIX, and the long arm is recorded in the session for the analysis
# scripts to read instead of asking again. Blank/None = skipped.
FOLDER_LABEL_FIELD = {'key': 'capture.FOLDER_SUFFIX', 'kind': 'foldername',
                      'label': 'folder label'}
LONG_ARM_FIELD = {'key': 'capture.LONG_ARM_CM', 'kind': 'optfloat',
                  'label': 'long arm length', 'min': 0.0}

PARAM_BY_KEY = {p['key']: p for p in PARAMS}

# adopt rows: (key, label). Ticked = the value of the camera/scope box; unticked
# = the capture's own logic (the parameters marked auto_of that row, where it
# has any, feed it).
ADOPT_ROWS = [
    ('exposure', 'exposure'),
    ('gain', 'gain'),
    ('frame_rate', 'frame rate'),
    ('roi', 'ROI'),
    ('scope_range', 'scope range'),
]
assert {k for k, _ in ADOPT_ROWS} == set(run_config.ADOPT_KEYS)

# what "unticked" must say outright, because the config file may pin a value:
# None is how each of these asks the capture to work it out
AUTO_OVERRIDES = {
    'exposure': {'capture.EXPOSURE_US': None},
    'roi': {'capture.MANUAL_ROI': None},
    'scope_range': {'capture.SCOPE_RANGE_V': None},
}
PRESET_EXCLUDED = {FOLDER_PARAM}      # a preset is a recipe, not a place

PIPELINE_SCRIPT = _REPO_ROOT / 'pico_scope' / 'run_mode_video_pipeline.py'
SHOW_SCRIPT = _REPO_ROOT / 'pico_scope' / 'mode_video_sync_show.py'
PLOT_SCOPE_SCRIPT = _REPO_ROOT / 'kalishlot' / 'plot_scope.py'
PLOT_VIDEO_SCRIPT = _REPO_ROOT / 'kalishlot' / 'plot_video.py'
VIDEO_SUFFIXES = ('.npy', '.avi', '.mkv', '.mp4')   # what plot_video reads
SHOW_TAIL = 6                         # lines of a failed viewer's output kept
SESSION_MARKER = 'SESSION_PATH='      # the capture prints where it saved
LOG_KEEP = 600                        # lines kept for a box that attaches late
STOP_WAIT_S = 5.0
# what a run records: everything and syncs it (None), or one instrument alone
# with no sync and no viewer (the pipeline's --only)
ONLY_MODES = ('video', 'scope')
LOAN_BORROWER = 'mode_video_capture.py'


# --- values ----------------------------------------------------------------
def coerce(param, value):
    """`value` as the parameter's type, or ValueError saying what is wrong."""
    kind, label = param['kind'], param['label']
    if kind == 'bool':
        if not isinstance(value, bool):
            raise ValueError(f'{label}: expected true or false')
        return value
    if kind == 'foldername':
        text = '' if value is None else str(value)
        bad = sorted({c for c in text if c in FORBIDDEN_NAME_CHARS or ord(c) < 32})
        if bad:
            raise ValueError(f'{label}: a folder name cannot contain '
                             f'{" ".join(repr(c) for c in bad)}')
        return text.strip().rstrip('.')
    if kind in ('text', 'folder'):
        text = '' if value is None else str(value).strip()
        if not text and param.get('optional'):
            return None
        return text
    if kind == 'choice':
        if value is None and param.get('optional'):
            value = ''
        for choice in param['choices']:
            if str(choice) == str(value):
                return None if choice == '' and param.get('optional') else choice
        raise ValueError(f'{label}: {value!r} is not one of {param["choices"]}')
    if kind == 'intlist':
        items = value if isinstance(value, (list, tuple)) else \
            str(value).replace(';', ',').split(',')
        try:
            numbers = [int(float(str(item).strip())) for item in items
                       if str(item).strip()]
        except ValueError:
            raise ValueError(f'{label}: expected whole numbers like 128, 256')
        if not numbers:
            raise ValueError(f'{label}: at least one number')
        return numbers
    if kind == 'optfloat' and (value is None or str(value).strip() == ''):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{label}: expected a number, got {value!r}')
    if number != number or number in (float('inf'), float('-inf')):
        raise ValueError(f'{label}: expected a finite number')
    if kind == 'int':
        if number != int(number):
            raise ValueError(f'{label}: expected a whole number')
        number = int(number)
    if 'min' in param and number < param['min']:
        raise ValueError(f'{label}: at least {param["min"]}')
    if 'max' in param and number > param['max']:
        raise ValueError(f'{label}: at most {param["max"]}')
    return number


def config_values():
    """The config file's value for every parameter (what a widget shows until
    it is edited). Read each time, so editing the file shows up in the box."""
    values = {}
    for section in sorted({p['key'].split('.')[0] for p in PARAMS}):
        try:
            given = run_config.section_values(section)
        except Exception:
            given = {}
        for param in PARAMS:
            sec, name = param['key'].split('.')
            if sec == section and name in given:
                values[param['key']] = given[name]
    return values


def check_folder(path):
    """(ok, message) for a save folder: absolute, and a place a capture can
    create its session folder. Creates nothing that is left behind."""
    text = str(path or '').strip()
    if not text:
        return False, 'no save folder set'
    folder = Path(text).expanduser()
    if not folder.is_absolute():
        return False, 'the save folder must be a full path'
    probe = folder
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent        # a missing folder is made by the capture,
    if not probe.is_dir():          # so what must exist is its nearest parent
        return False, f'{probe} is not a folder'
    try:
        test = probe / '.kalishlot_write_test'
        test.write_bytes(b'')
        test.unlink()
    except OSError as error:
        return False, f'cannot write to {probe}: {error.strerror or error}'
    note = '' if folder.exists() else ' (will be created)'
    return True, f'{folder}{note}'


def classify_capture(path):
    """(kind, path to open) for something the user picked to show:

      * a folder with one `*_session.json`, or that json itself   -> 'synced'
        (the folder: the synced video + scope viewer)
      * a `*.npz` - one scope recording, a synced capture's block, its
        trailing capture or a stand-alone one                      -> 'scope'
      * a `.npy` / `.avi` / `.mkv` / `.mp4` frames file            -> 'video'
      * a folder with no session but one `*_scope.npz` (the "record scope only"
        output)                                                    -> 'scope'

    ValueError, saying what was expected, for anything else."""
    path = Path(str(path)).expanduser()
    if path.is_dir():
        sessions = sorted(path.glob('*_session.json'))
        if len(sessions) == 1:
            return 'synced', path
        scopes = sorted(path.glob('*_scope.npz')) if not sessions else []
        if len(scopes) == 1:
            return 'scope', scopes[0]
        raise ValueError(
            f'{path} is not a capture folder: expected exactly one '
            f'*_session.json (or one *_scope.npz) in it, found '
            f'{len(sessions) or len(scopes)}')
    if path.is_file():
        if path.name.endswith('_session.json'):
            return 'synced', path.parent
        if path.suffix.lower() == '.npz':
            return 'scope', path
        if path.suffix.lower() in VIDEO_SUFFIXES:
            return 'video', path
    raise ValueError(f'{path} is not a capture folder or file kalishlot can '
                     f'show (a folder of a synced capture, a scope .npz, or a '
                     f'{" / ".join(VIDEO_SUFFIXES)} file)')


_dialog_lock = threading.Lock()


def ask_directory(initial=None):
    """The native folder-browse window, on THIS PC (the lab PC - the same one
    the capture runs on). Blocks until it is closed; returns the chosen path
    or None. Windows' own dialog, via tkinter, opened on top of the browser."""
    import tkinter
    from tkinter import filedialog
    if not _dialog_lock.acquire(blocking=False):
        raise ValueError('a folder window is already open on the lab PC')
    try:
        root = tkinter.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        try:
            start = str(Path(initial).expanduser()) if initial else ''
            if not start or not Path(start).is_dir():
                start = str(Path.home())
            chosen = filedialog.askdirectory(
                parent=root, initialdir=start, mustexist=False,
                title='Save the capture into...')
        finally:
            root.destroy()
    finally:
        _dialog_lock.release()
    return str(Path(chosen)) if chosen else None


# --- the adapter -----------------------------------------------------------
class SyncedPipelineAdapter(DeviceAdapter):
    type_name = 'synced_pipeline'
    display_name = 'Synced video + scope pipeline'
    VIRTUAL = True      # no hardware: the server does not count it as "open"

    # Set by server.py: () -> (devices dict, loans dict). Read without locks on
    # purpose - describe() is called with the server's device lock held.
    registry = None
    pick_folder_fn = staticmethod(ask_directory)
    pick_capture_fn = staticmethod(file_dialogs.ask_capture)
    # Set by server.py: () -> the environment a run starts with, and
    # (borrower) -> hands back whatever that borrower still had on loan.
    child_environment = staticmethod(lambda: dict(os.environ))
    return_loans = staticmethod(lambda borrower: None)
    # what is run; replaced by the tests with a stand-in script
    pipeline_command = staticmethod(
        lambda: [sys.executable, '-u', str(PIPELINE_SCRIPT), '--from-kalishlot'])
    show_command = staticmethod(
        lambda folder: [sys.executable, '-u', str(SHOW_SCRIPT),
                        '--session', folder])
    scope_plot_command = staticmethod(
        lambda path: [sys.executable, '-u', str(PLOT_SCOPE_SCRIPT), path])
    video_plot_command = staticmethod(
        lambda path: [sys.executable, '-u', str(PLOT_VIDEO_SCRIPT), path])

    @staticmethod
    def list_available():
        return [{'address': 'main', 'label': 'synced video + scope pipeline'}]

    def __init__(self, address):
        super().__init__(address)
        self.edited = {}            # "section.NAME" -> value
        self.adopt = {key: True for key in run_config.ADOPT_KEYS}
        self.presets = {}           # name -> {'edited', 'adopt'}
        self.camera_id = None       # the user's pick; None = the first open one
        self._lock = threading.Lock()
        self._process = None
        self._params_path = None
        self._log = collections.deque(maxlen=LOG_KEEP)
        self.show = {'folder': None, 'running': False, 'error': None}
        self._shows = 0             # viewers open now
        self.run = {'running': False, 'returncode': None, 'started': None,
                    'ended': None, 'step': None, 'session': None,
                    'stopped': False, 'error': None, 'only': None}

    def open(self):
        pass

    def close(self):
        if self.run['running']:
            try:
                self._stop()
            except Exception:
                pass

    # ---- what the box needs to know about the rest of the dashboard
    def dependencies(self):
        """The cameras and the scope the dashboard has, each 'open' or 'lent'
        (a device lent to a running pipeline is closed but still counts)."""
        devices, loans = ({}, {})
        if self.registry is not None:
            devices, loans = self.registry()
            devices, loans = dict(devices), dict(loans)
        cameras, scopes = [], []
        for device_id, adapter in devices.items():
            kind = getattr(adapter, 'type_name', None)
            entry = {'device_id': device_id, 'state': 'open'}
            if kind in CAMERA_TYPES:
                cameras.append({**entry, 'type': kind})
            elif kind == SCOPE_TYPE:
                scopes.append(entry)
        for device_id, loan in loans.items():
            entry = {'device_id': device_id, 'state': 'lent'}
            if loan.get('type') in CAMERA_TYPES:
                cameras.append({**entry, 'type': loan['type']})
            elif loan.get('type') == SCOPE_TYPE:
                scopes.append(entry)
        return {'cameras': cameras, 'scopes': scopes}

    def has_function_generator(self):
        devices = self.registry()[0] if self.registry is not None else {}
        return any(getattr(adapter, 'type_name', None) == 'rigol_dg'
                   for adapter in dict(devices).values())

    def chosen_camera(self, dependencies=None):
        cameras = (dependencies or self.dependencies())['cameras']
        for camera in cameras:
            if camera['device_id'] == self.camera_id:
                return camera
        return cameras[0] if cameras else None

    # ---- describe / persistence
    def describe(self):
        dependencies = self.dependencies()
        camera = self.chosen_camera(dependencies)
        configured = config_values()
        return {
            'type': self.type_name,
            'label': 'SYNCED VIDEO + SCOPE PIPELINE',
            'commands': ['set_param', 'reset_param', 'set_adopt',
                         'choose_camera', 'pick_folder', 'check_folder',
                         'preset_save', 'preset_load', 'preset_delete',
                         'start', 'stop', 'show_session'],
            'params': PARAMS,
            'pixel_formats': PIXEL_FORMATS,
            'adopt_rows': [{'key': k, 'label': label}
                           for k, label in ADOPT_ROWS],
            'edited': dict(self.edited),
            'config': configured,
            'adopt': dict(self.adopt),
            'presets': sorted(self.presets),
            'camera_id': camera['device_id'] if camera else None,
            'dependencies': dependencies,
            'ready': bool(camera and dependencies['scopes']),
            'ready_video': camera is not None,
            'ready_scope': bool(dependencies['scopes']),
            'run': dict(self.run),
            'show': dict(self.show),
            'log': list(self._log)[-200:],
        }

    def settings_snapshot(self):
        return {'edited': dict(self.edited), 'adopt': dict(self.adopt),
                'presets': {name: dict(preset)
                            for name, preset in self.presets.items()},
                'camera_id': self.camera_id}

    def restore_settings(self, snapshot):
        edited = {}
        for key, value in (snapshot.get('edited') or {}).items():
            param = PARAM_BY_KEY.get(key)
            if param is None:
                continue                # a parameter since removed
            try:
                edited[key] = coerce(param, value)
            except ValueError:
                continue                # no longer valid: back to the file's
        self.edited = edited
        for key, value in (snapshot.get('adopt') or {}).items():
            if key in self.adopt:
                self.adopt[key] = bool(value)
        self.presets = {str(name): preset for name, preset
                        in (snapshot.get('presets') or {}).items()
                        if isinstance(preset, dict)}
        self.camera_id = snapshot.get('camera_id')

    # ---- commands
    def command(self, name, args):
        with self._lock:
            return self._command(name, args or {})

    def _command(self, name, args):
        if name == 'set_param':
            param = self._param(args.get('key'))
            value = coerce(param, args.get('value'))
            self.edited[param['key']] = value
            self.emit({'type': 'param_applied', 'key': param['key'],
                       'value': value})
            return {'ok': True, 'value': value}
        if name == 'reset_param':
            param = self._param(args.get('key'))
            self.edited.pop(param['key'], None)
            self.emit({'type': 'param_reset', 'key': param['key']})
            return {'ok': True}
        if name == 'set_adopt':
            key = args.get('key')
            if key not in self.adopt:
                raise ValueError(f'unknown adopt row {key!r}')
            self.adopt[key] = bool(args.get('value'))
            self.emit({'type': 'adopt_applied', 'key': key,
                       'value': self.adopt[key]})
            return {'ok': True}
        if name == 'choose_camera':
            self.camera_id = args.get('device_id') or None
            return {'ok': True}
        if name == 'check_folder':
            ok, message = check_folder(args.get('path', self._folder()))
            return {'ok': ok, 'message': message}
        if name == 'pick_folder':
            chosen = self.pick_folder_fn(args.get('start') or self._folder())
            if chosen is not None:
                self.edited[FOLDER_PARAM] = chosen
                self.emit({'type': 'param_applied', 'key': FOLDER_PARAM,
                           'value': chosen})
            return {'ok': True, 'path': chosen}
        if name == 'preset_save':
            label = str(args.get('name') or '').strip()
            if not label:
                raise ValueError('a preset needs a name')
            self.presets[label] = {
                'edited': {k: v for k, v in self.edited.items()
                           if k not in PRESET_EXCLUDED},
                'adopt': dict(self.adopt)}
            return {'ok': True, 'presets': sorted(self.presets)}
        if name == 'preset_load':
            label = args.get('name')
            preset = self.presets.get(label)
            if preset is None:
                raise ValueError(f'no preset {label!r}')
            kept = {k: v for k, v in self.edited.items() if k in PRESET_EXCLUDED}
            self.edited = {**kept, **preset['edited']}
            self.adopt.update(preset['adopt'])
            return {'ok': True}
        if name == 'preset_delete':
            if self.presets.pop(args.get('name'), None) is None:
                raise ValueError(f'no preset {args.get("name")!r}')
            return {'ok': True, 'presets': sorted(self.presets)}
        if name == 'show_session':
            return self._show_session(args.get('path'))
        if name == 'start':
            return self._start(args)
        if name == 'stop':
            return self._stop()
        raise ValueError(f'unknown command {name!r}')

    # ---- showing a past capture
    def _show_session(self, path=None):
        """Open a capture in the interactive viewer (mode_video_sync_show.py):
        the window asking for a folder or a file first, unless a path is given.
        What opens depends on what it is (classify_capture): the synced viewer
        for a folder of a synced capture, kalishlot/plot_scope.py for a scope
        recording, kalishlot/plot_video.py for a frames file. Each is its own
        process with its own window on this PC, and needs no camera or scope,
        so it runs whether or not a pipeline run is going."""
        if path is None:
            path = self.pick_capture_fn(self.show['folder'] or self._folder())
            if path is None:
                return {'ok': True, 'path': None}      # the window was cancelled
        kind, folder = classify_capture(path)
        launch = {'synced': self.show_command, 'scope': self.scope_plot_command,
                  'video': self.video_plot_command}[kind]
        handle, params_path = tempfile.mkstemp(prefix='kalishlot_show_',
                                               suffix='.json')
        shown = {key.split('.', 1)[1]: value for key, value in self.edited.items()
                 if key.startswith('show.') and key in PARAM_BY_KEY}
        with os.fdopen(handle, 'w', encoding='utf-8') as file:
            json.dump({'sections': {'show': shown}} if shown else {}, file)
        environment = dict(self.child_environment())
        environment[run_config.PARAMS_ENV_VAR] = params_path
        environment['PYTHONIOENCODING'] = 'utf-8'
        try:
            process = subprocess.Popen(
                launch(str(folder)), cwd=_REPO_ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                errors='replace', bufsize=1)
        except OSError:
            os.unlink(params_path)
            raise
        self._shows += 1
        self._set_show(folder=str(folder), running=True, error=None)
        threading.Thread(target=self._watch_show,
                         args=(process, params_path, str(folder)),
                         daemon=True).start()
        return {'ok': True, 'path': str(folder)}

    def _set_show(self, **changes):
        self.show.update(changes)
        self.emit({'type': 'show', 'show': dict(self.show)})

    def _watch_show(self, process, params_path, folder):
        tail = collections.deque(maxlen=SHOW_TAIL)
        try:
            for raw in process.stdout:
                tail.append(raw.rstrip('\r\n'))
        except Exception:
            pass
        returncode = process.wait()
        try:
            os.unlink(params_path)
        except OSError:
            pass
        self._shows -= 1
        error = None
        if returncode != 0:
            error = ' | '.join(line for line in tail if line.strip())[-300:] \
                or f'the viewer exited with code {returncode}'
        if self._shows <= 0 or error:
            self._set_show(folder=folder, running=self._shows > 0, error=error)

    # ---- running the pipeline
    def _set_run(self, **changes):
        self.run.update(changes)
        self.emit({'type': 'run', 'run': dict(self.run)})

    def _log_line(self, line):
        self._log.append(line)
        self.emit({'type': 'log', 'line': line})

    def _start(self, args=None):
        if self.run['running']:
            raise ValueError('a run is already in progress')
        args = args or {}
        only = args.get('only') or None
        if only not in (None, *ONLY_MODES):
            raise ValueError(f'unknown run kind {only!r}')
        # from the start-of-capture pop-up: asked fresh every time, so a value
        # that was never (re)typed is skipped rather than reused.
        extra = {
            FOLDER_LABEL_FIELD['key']: coerce(FOLDER_LABEL_FIELD,
                                              args.get('folder_label') or ''),
            LONG_ARM_FIELD['key']: coerce(LONG_ARM_FIELD,
                                          args.get('long_arm_cm')),
        }
        params = self.run_params(extra, only)     # raises when it cannot start
        handle, path = tempfile.mkstemp(prefix='kalishlot_params_',
                                        suffix='.json')
        with os.fdopen(handle, 'w', encoding='utf-8') as file:
            json.dump(params, file, indent=1)
        environment = dict(self.child_environment())
        environment[run_config.PARAMS_ENV_VAR] = path
        environment['PYTHONIOENCODING'] = 'utf-8'
        flags = 0
        if os.name == 'nt':
            flags = (subprocess.CREATE_NO_WINDOW
                     | subprocess.CREATE_NEW_PROCESS_GROUP)
        command = self.pipeline_command()
        if only:
            command = [*command, '--only', only]
        try:
            process = subprocess.Popen(
                command, cwd=_REPO_ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                errors='replace', bufsize=1, creationflags=flags)
        except OSError:
            os.unlink(path)
            raise
        self._process, self._params_path = process, path
        self._log.clear()
        self._set_run(running=True, returncode=None, stopped=False,
                      started=datetime.now().isoformat(timespec='seconds'),
                      ended=None, step=None, session=None, error=None, only=only)
        threading.Thread(target=self._read, args=(process, path),
                         daemon=True).start()
        return {'ok': True}

    def _read(self, process, params_path):
        """Relay the run's output until it ends, then record how it ended."""
        try:
            for raw in process.stdout:
                line = raw.rstrip('\r\n')
                if line.startswith('=== ') and line.endswith(' ==='):
                    self._set_run(step=line.strip('= ').replace('.py', ''))
                elif line.startswith(SESSION_MARKER):
                    self._set_run(session=line[len(SESSION_MARKER):].strip())
                self._log_line(line)
        except Exception as error:      # a reader that dies must not hang
            self._log_line(f'(output reader stopped: {error})')
        returncode = process.wait()
        try:
            os.unlink(params_path)
        except OSError:
            pass
        if process is not self._process or self.run['stopped']:
            return          # a stop finishes the record, once devices are back
        self._set_run(
            running=False, returncode=returncode,
            ended=datetime.now().isoformat(timespec='seconds'),
            error=None if returncode == 0 else f'exited with code {returncode}')

    def _stop(self):
        process = self._process
        if not self.run['running'] or process is None:
            raise ValueError('no run to stop')
        self.run['stopped'] = True
        # the whole tree: the pipeline's capture is a child of a child, and it
        # is the capture that holds the camera and the scope
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           capture_output=True)
        else:
            process.kill()
        try:
            process.wait(STOP_WAIT_S)
        except subprocess.TimeoutExpired:
            process.kill()
        self._log_line('--- stopped from the box ---')
        # a killed borrower returns nothing, so the devices come back here
        try:
            self.return_loans(LOAN_BORROWER)
        finally:
            self._set_run(running=False, returncode=process.returncode,
                          ended=datetime.now().isoformat(timespec='seconds'),
                          error=None)
        return {'ok': True}

    @staticmethod
    def _param(key):
        param = PARAM_BY_KEY.get(key)
        if param is None:
            raise ValueError(f'unknown parameter {key!r}')
        return param

    def _folder(self):
        return self.edited.get(FOLDER_PARAM) or \
            config_values().get(FOLDER_PARAM) or ''

    # ---- what a run is told
    def run_params(self, extra=None, only=None):
        """The JSON a run takes through MODE_VIDEO_PARAMS: the edited values,
        the start-of-capture pop-up's values (`extra`), what each unticked
        adopt row must say outright, the pipeline's own fixed choices, and the
        adopt flags. `only` ('video' or 'scope') asks for that instrument alone,
        which then is the only one needed. Raises ValueError when the run
        cannot start (a missing instrument, a bad save folder)."""
        dependencies = self.dependencies()
        camera = self.chosen_camera(dependencies)
        if only == 'video':
            if camera is None:
                raise ValueError('needs an open camera')
        elif only == 'scope':
            if not dependencies['scopes']:
                raise ValueError('needs an open PicoScope')
        elif camera is None or not dependencies['scopes']:
            raise ValueError('needs an open camera and an open PicoScope')
        folder = self._folder()
        ok, message = check_folder(folder)
        if not ok:
            raise ValueError(message)

        flat = {key: value for key, value in self.edited.items()
                if key in PARAM_BY_KEY}
        flat.update(extra or {})
        if only is None:        # only the full capture has a tail
            configured = config_values()
            effective = lambda key: flat.get(key, configured.get(key))
            if effective('capture.TRAILING_SCOPE_S') and \
                    effective('capture.TRAILING_SCOPE_AUX_FG') and \
                    not self.has_function_generator():
                raise ValueError('the trailing capture applies the secondary '
                                 'FG channel: open the function generator '
                                 'first (or untick it)')
        for row, overrides in AUTO_OVERRIDES.items():
            if not self.adopt[row]:
                flat.update(overrides)
        flat.update({
            'capture.ACTION': 'capture',
            'capture.DRIVE_SCOPE': True,
            'capture.PROMPT_FOR_OUTPUT_ROOT': False,
            'capture.OUTPUT_ROOT': str(Path(folder).expanduser()),
        })
        if only != 'scope':
            flat.update({
                'capture.CAMERA': CAMERA_TYPES[camera['type']],
                'capture.SERIAL_NUMBER': camera['device_id'].split(':', 1)[1],
            })
        if only == 'video':
            flat['capture.DRIVE_SCOPE'] = False
        sections = {}
        for key, value in flat.items():
            section, name = key.split('.', 1)
            sections.setdefault(section, {})[name] = value
        return {'sections': sections, 'adopt': dict(self.adopt)}
