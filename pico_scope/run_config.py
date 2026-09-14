"""One file of run parameters for the mode-video pipeline, outside git.

The four pipeline scripts used to carry their settings as constants at the top
of each file, so changing a frame rate or an ROI meant editing a tracked file -
and then either committing a per-run value or leaving the working tree dirty.
`MANUAL_ROI` was the worst of it: a dict typed from the camera GUI, correct
only until the cavity was realigned and silently wrong afterwards.

Now those constants are *defaults*. The values actually used come from

    pico_scope/run_config_local.py      (git-ignored - this is the file to edit)

which is created for you, from `run_config_local_template.py`, the first time
anything here runs. It holds every parameter of all four scripts, one class per
script, so there is one place to look and nothing tracked to modify.

## What this costs, and why it is still worth it

Settings are part of the record: while they lived in git, `git log` could say
what any measurement was taken with. A git-ignored file throws that away, so
every capture writes its own copy instead - see `dump_into()`, called from
`mode_video_capture.save_session()`, which puts

    run_config_used.py         the config file verbatim, comments and all
    run_config_resolved.json   the values actually in force, derived ones
                               included (the exposure, the ROI that was picked)

next to the frames. That is better provenance than a commit ever was: it is
per-capture rather than per-commit, and it records what was *resolved*, not
just what was typed.

## Using another config

    python pico_scope/run_mode_video_pipeline.py --config my_experiment.py

keeps several named configs side by side. The pipeline passes it to every step
through the MODE_VIDEO_CONFIG environment variable, and the individual scripts
take `--config` too. It has to be read at import time - the scripts' constants
are read by module-level code in some of them - so it is picked up from
sys.argv and the environment directly, before argparse ever runs.

## Adding a parameter

Add it to the template under the right class, and to SECTIONS below. The
self-test checks that the two agree and that every name in them really is a
constant of the script it claims, so a parameter that is registered but
misspelled fails loudly here rather than silently doing nothing during a run.
"""

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE_PATH = HERE / 'run_config_local_template.py'
LOCAL_PATH = HERE / 'run_config_local.py'
ENV_VAR = 'MODE_VIDEO_CONFIG'

# Which names each script takes from the config, keyed by the class that holds
# them. The names are exactly the module-level constants of that script, and
# the self-test checks it: a typo here would otherwise mean a setting that is
# accepted, reported as applied, and ignored.
SECTIONS = {
    'capture': ('mode_video_capture', (
        'ACTION', 'CAMERA', 'DRIVE_SCOPE', 'LOCATE_FIRST', 'STRICT_LEVELS',
        'SERIAL_NUMBER', 'FRAME_RATE_HZ', 'EXPOSURE_US', 'N_FRAMES',
        'PIXEL_FORMAT', 'GAIN_DB', 'BINNING', 'THROUGHPUT_BPS',
        'MANUAL_ROI', 'ROI_WIDTH', 'ROI_HEIGHT_CANDIDATES',
        'ROI_MIN_MARGIN_ROWS', 'ROI_OFFSET_X',
        'SCOPE_CHANNEL', 'SCOPE_RANGE_V', 'SCOPE_COUPLING',
        'SCOPE_SAMPLE_INTERVAL_S', 'SCOPE_PAD_S', 'SCOPE_AUTORANGE_PROBE_S',
        'SCOPE_AUTORANGE_MARGIN', 'SCOPE_AUTORANGE_MIN_V',
        'SCOPE_AUX_CHANNEL', 'SCOPE_AUX_LABEL', 'SCOPE_AUX_RANGE_V',
        'SCOPE_AUX_COUPLING',
        'MASK_THRESHOLD', 'OUTPUT_ROOT', 'PROMPT_FOR_OUTPUT_ROOT',
        'MAX_SATURATED_FRACTION', 'TARGET_PEAK_FRACTION', 'LEVEL_BURSTS',
        'LEVEL_BURST_FRAMES', 'LEVEL_SAFETY', 'LEVEL_TOO_DIM_FRACTION',
        'LEVEL_CLIPPED_STEP_DB', 'HOST_T0_BIAS_S',
    )),
    'sync': ('mode_video_sync', (
        'ACTION', 'SESSION', 'SCOPE_FILE', 'SEARCH_WINDOW_S',
        'TIME_COLUMN', 'SIGNAL_COLUMN', 'AUX_COLUMN', 'AUX_LABEL',
    )),
    'show': ('mode_video_sync_show', (
        'ACTION', 'SESSION', 'SCOPE_FILE',
        'SHADE_ALPHA', 'FIT_REBINNING', 'FIT_MAX_LEVEL',
    )),
    'mark': ('mode_video_sync_mark', (
        'ACTION', 'SESSION', 'SCOPE_FILE',
        'CAVITY_ELEMENTS', 'SHORT_ARM_LENGTHS', 'MID_ARM_LENGTH',
        'N_points', 'SHORT_ARM_LENGTH',
    )),
}

_loaded = None          # the config module, once imported
_loaded_path = None     # where it came from, for the report and the dump


# %% ------------------------------------------------------------- finding it
def _path_from_argv(argv):
    """`--config <path>` or `--config=<path>` in a raw argument list.

    argparse cannot do this: the backend choice in mode_video_sync_show.py and
    the ACTION checks in mode_video_sync_mark.py run at import, long before any
    parser exists, so the config has to be in force by then.
    """
    for i, arg in enumerate(argv):
        if arg == '--config' and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith('--config='):
            return arg[len('--config='):]
    return None


def config_path(explicit=None):
    """Which config file to read: an explicit path, then --config, then
    MODE_VIDEO_CONFIG, then the local file beside this one."""
    for candidate in (explicit, _path_from_argv(sys.argv), os.environ.get(ENV_VAR)):
        if candidate:
            return Path(candidate).expanduser().resolve()
    return LOCAL_PATH


def ensure_local_config():
    """Create run_config_local.py from the template on first use.

    Loudly, because the values it starts with are the scripts' own defaults and
    the point of the file is that they are meant to be edited.
    """
    if LOCAL_PATH.exists():
        return LOCAL_PATH
    if not TEMPLATE_PATH.exists():        # pragma: no cover - a broken checkout
        raise FileNotFoundError(
            f'neither {LOCAL_PATH.name} nor {TEMPLATE_PATH.name} is present in '
            f'{HERE} - the template is tracked, so restore it from git')
    shutil.copyfile(TEMPLATE_PATH, LOCAL_PATH)
    print(f'created {LOCAL_PATH}\n'
          f'  This is where the pipeline\'s run parameters live now, and it is '
          f'git-ignored.\n'
          f'  It starts as a copy of {TEMPLATE_PATH.name} - edit it instead of '
          f'the scripts.')
    return LOCAL_PATH


def load(explicit=None, force=False):
    """Import the config file and return it, remembering it for later calls."""
    global _loaded, _loaded_path
    path = config_path(explicit)
    if _loaded is not None and not force and path == _loaded_path:
        return _loaded
    if path == LOCAL_PATH:
        ensure_local_config()
    if not path.is_file():
        raise FileNotFoundError(f'no config file at {path}')

    # Loaded by path rather than by name so that --config can point anywhere,
    # and under a private name so it cannot shadow a real module.
    spec = importlib.util.spec_from_file_location('_mode_video_run_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _loaded, _loaded_path = module, path
    return module


def loaded_path():
    """Where the config in force came from; None until something loads one."""
    return _loaded_path


# %% -------------------------------------------------------------- using it
def section_values(section, config=None):
    """The settings one script takes from the config, as a plain dict.

    Only names the section is registered for. Anything else in the class is an
    error and says so: a misspelled setting that was quietly ignored would look
    exactly like one that was applied, and the capture would run on the old
    value without a word.
    """
    if section not in SECTIONS:
        raise KeyError(f'unknown config section {section!r}; '
                       f'expected one of {sorted(SECTIONS)}')
    _, allowed = SECTIONS[section]
    config = load() if config is None else config
    block = getattr(config, section, None)
    if block is None:
        return {}
    given = {name: value for name, value in vars(block).items()
             if not name.startswith('_')}
    unknown = sorted(set(given) - set(allowed))
    if unknown:
        raise KeyError(
            f'{Path(_loaded_path).name if _loaded_path else "config"}: '
            f'class {section} sets {", ".join(unknown)}, which '
            f'{SECTIONS[section][0]}.py has no such setting for. '
            f'Valid names are: {", ".join(sorted(allowed))}')
    return given


def apply(section, namespace, config=None):
    """Overwrite a script's constants with the config's, and report what moved.

    `namespace` is the script's own globals(). The constants stay declared in
    the script - they are the defaults, they carry the comments explaining each
    one, and they keep the self-tests running with no config file at all - and
    this replaces the ones the config names.

    Returns the names that were actually changed, so a caller can say so.
    """
    values = section_values(section, config)
    changed = {}
    for name, value in values.items():
        if name not in namespace:
            raise KeyError(
                f'{SECTIONS[section][0]}.py has no constant {name}, but '
                f'run_config lists it under section {section!r}. One of the '
                f'two was renamed; the self-test catches this.')
        if namespace[name] != value or type(namespace[name]) is not type(value):
            changed[name] = value
        namespace[name] = value
    return changed


def describe(section, changed):
    """One line naming the config in force and what it overrode, or nothing
    when it matched the defaults exactly."""
    where = Path(_loaded_path).name if _loaded_path else 'defaults'
    if not changed:
        return f'config: {where} (all values match the script defaults)'
    return (f'config: {where} -> ' +
            ', '.join(f'{name}={value!r}' for name, value in sorted(changed.items())))


# %% --------------------------------------------------------- recording it
def _json_safe(value):
    """Anything the config can hold, in a form json will take."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def dump_into(folder, resolved=None, section='capture'):
    """Record the run parameters beside a capture.

    Two files, because they answer different questions. The verbatim copy keeps
    the comments and shows what was *typed*, including the alternatives left
    commented out; the JSON is what was actually *in force* by the time the
    camera ran, which is not the same thing - the exposure is derived from the
    frame rate, and the ROI that gets used may be one the script chose rather
    than one that was typed.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []

    if _loaded_path and Path(_loaded_path).is_file():
        copy = folder / 'run_config_used.py'
        shutil.copyfile(_loaded_path, copy)
        written.append(copy)

    values = dict(resolved or {})
    if not values:
        values = section_values(section)
    record = {
        'source': str(_loaded_path) if _loaded_path else None,
        'section': section,
        'values': {name: _json_safe(value) for name, value in sorted(values.items())},
    }
    path = folder / 'run_config_resolved.json'
    path.write_text(json.dumps(record, indent=1), encoding='utf-8')
    written.append(path)
    return written


def resolved_values(section, namespace):
    """What a script's registered settings actually are right now.

    Read back out of the module after everything has been applied and derived,
    so the exposure computed from the frame rate is the number recorded, not
    the placeholder that asked for it.
    """
    _, allowed = SECTIONS[section]
    return {name: namespace[name] for name in allowed if name in namespace}


# %% -------------------------------------------------------------- self-test
def _self_test():
    print('run_config self-test')

    # --config wins over the environment, and either over the local file
    assert _path_from_argv(['x', '--config', 'a.py']) == 'a.py'
    assert _path_from_argv(['x', '--config=b.py']) == 'b.py'
    assert _path_from_argv(['x', '--config']) is None      # no value: not a path
    assert _path_from_argv(['--session', 'f']) is None
    print('  --config is read straight from argv, in both spellings')

    # the template parses, and covers exactly what SECTIONS registers
    assert TEMPLATE_PATH.is_file(), TEMPLATE_PATH
    spec = importlib.util.spec_from_file_location('_template_check', TEMPLATE_PATH)
    template = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(template)
    for section, (module_name, allowed) in SECTIONS.items():
        block = getattr(template, section, None)
        assert block is not None, f'the template has no class {section}'
        given = {n for n in vars(block) if not n.startswith('_')}
        assert given == set(allowed), (
            f'{section}: template and SECTIONS disagree - '
            f'only in template: {sorted(given - set(allowed))}, '
            f'only in SECTIONS: {sorted(set(allowed) - given)}')
    print('  the template holds every registered parameter and no others')

    # every registered name is really a constant of the script it belongs to
    sys.path.insert(0, str(HERE.parent))
    for section, (module_name, allowed) in SECTIONS.items():
        source = (HERE / f'{module_name}.py').read_text(encoding='utf-8')
        for name in allowed:
            assert f'\n{name} = ' in source or source.startswith(f'{name} = '), (
                f'{module_name}.py declares no constant {name}, which '
                f'run_config registers under {section!r}')
    print('  every registered name is a module constant of its script')

    # an unknown key is refused rather than ignored
    class _Config:
        class capture:
            BINNING = 4
            BINING = 2          # the typo this check exists for

    try:
        section_values('capture', _Config)
    except KeyError as error:
        assert 'BINING' in str(error), error
    else:                                                   # pragma: no cover
        raise AssertionError('a misspelled setting was accepted')
    print('  a name the script does not have is an error, not a silent no-op')

    # apply() overwrites and reports only what it changed
    class _Good:
        class capture:
            BINNING = 4
            N_FRAMES = 120

    namespace = {'BINNING': 2, 'N_FRAMES': 120}
    changed = apply('capture', namespace, _Good)
    assert namespace['BINNING'] == 4 and namespace['N_FRAMES'] == 120
    assert changed == {'BINNING': 4}, changed
    print('  apply() overwrites the constants and names only what it moved')

    # a registered name the script does not define is caught at apply time
    try:
        apply('capture', {}, _Good)
    except KeyError as error:
        assert 'BINNING' in str(error), error
    else:                                                   # pragma: no cover
        raise AssertionError('a setting with no matching constant was applied')
    print('  a setting with no constant behind it fails instead of vanishing')

    # values survive the round trip into a capture folder
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        written = dump_into(tmp, resolved={'BINNING': 2, 'OUTPUT_ROOT': Path(tmp),
                                           'ROI_HEIGHT_CANDIDATES': (128, 256),
                                           'MANUAL_ROI': None})
        record = json.loads((Path(tmp) / 'run_config_resolved.json')
                            .read_text(encoding='utf-8'))
        assert record['values']['BINNING'] == 2
        assert record['values']['ROI_HEIGHT_CANDIDATES'] == [128, 256]
        assert record['values']['MANUAL_ROI'] is None
        assert record['values']['OUTPUT_ROOT'] == str(Path(tmp))
        assert any(p.name == 'run_config_resolved.json' for p in written)
    print('  the resolved values are written beside the capture as json')

    print('self-test passed')


if __name__ == '__main__':
    _self_test()
