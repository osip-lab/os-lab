"""One spectrum trace, from a PicoScope CSV export or a mode-video capture.

The spectrum scripts

    pico_scope/extract_df_and_fsr_from_scope_csv.py
    pico_scope/mode_spacing_extraction_sidebands.py
    pico_scope/mode_map_2d.py

read a scope trace as (time, transmission). It used to come only from a
PicoScope 7 export (.psdata, converted to CSV). A capture folder written by
pico_scope/run_mode_video_pipeline.py (mode_video_capture.py) has no .psdata:
its scope record is '<stamp>_scope.npz' (t [s], signal [V], optional aux)
described by the 'scope' block of '<stamp>_session.json'. Here the capture
FOLDER plays the part of the .psdata - it is what the user copies, what the
marks sidecar and the results line belong to - and its npz plays the part of
the CSV a waveform buffer was converted to.

The npz's 'signal' is the cavity transmission by construction (the capture's
SCOPE_CHANNEL), so it is the trace these scripts analyse whatever their
SIGNAL_COLUMN says; signal_column_for() names it after the channel it was
recorded on, which is what the marks sidecar remembers.

Deliberately standalone rather than reusing mode_video_sync.load_session_trace:
importing mode_video_sync applies the run config at import, which an analysis
script has no business doing.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SESSION_GLOB = '*_session.json'


def session_json(path):
    """The '*_session.json' of a capture folder (or the json itself), or None
    when `path` is not a capture."""
    path = Path(path)
    if path.is_file() and path.name.endswith('_session.json'):
        return path
    if path.is_dir():
        candidates = sorted(path.glob(SESSION_GLOB))
        if len(candidates) == 1:
            return candidates[0]
    return None


def is_capture_session(path):
    return session_json(path) is not None


def _scope_block(path):
    json_path = session_json(path)
    if json_path is None:
        raise FileNotFoundError(
            f'{path} is not a mode-video capture: expected a folder holding '
            f'exactly one {SESSION_GLOB}')
    session = json.loads(json_path.read_text(encoding='utf-8'))
    scope = session.get('scope')
    if not scope:
        raise FileNotFoundError(
            f'{json_path.parent.name} has no scope trace of its own - it was '
            f'recorded with PicoScope 7 driving the scope, so its spectrum is '
            f'that .psdata: copy the .psdata instead.')
    return json_path, scope


def session_scope_file(path):
    """The capture's '<stamp>_scope.npz'."""
    json_path, scope = _scope_block(path)
    return json_path.parent / scope['file']


def session_signal_column(path):
    """The transmission's channel, named as a PicoScope CSV column."""
    return f"Channel {_scope_block(path)[1]['channel']}"


def signal_column_for(data_path, default):
    """The column a trace of `data_path` is marked on: the recorded channel
    for a capture, `default` (the script's SIGNAL_COLUMN) for anything else."""
    if is_capture_session(data_path):
        return session_signal_column(data_path)
    return default


def load_trace(trace_path, signal_column, time_column='Time'):
    """(time, transmission) float arrays from a CSV export or a capture's npz.

    The CSV branch is what the scripts did inline before: rows 1 and 2 of a
    PicoScope export are the unit / blank header rows, and the time unit is
    whatever the export used - every analysis here works in ratios of it.
    The npz is in seconds.
    """
    trace_path = Path(trace_path)
    if trace_path.suffix.lower() == '.npz':
        _warn_if_clipped(trace_path.parent)
        with np.load(trace_path) as data:
            return (np.asarray(data['t'], dtype=float),
                    np.asarray(data['signal'], dtype=float))
    raw = pd.read_csv(trace_path, skiprows=[1, 2])
    raw = raw.loc[:, [time_column, signal_column]].dropna()
    return (raw[time_column].to_numpy(dtype=float),
            raw[signal_column].to_numpy(dtype=float))


def _warn_if_clipped(folder):
    try:
        _, scope = _scope_block(folder)
    except FileNotFoundError:
        return
    if scope['channel'] in (scope.get('overflow_channels') or []):
        print(f"  ! channel {scope['channel']} went over its "
              f"+-{scope.get('range_v', '?')} V range during this capture - "
              f"the tallest peaks are clipped")


# ------------------------------------------------------------------ self-test
def _self_test():
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        folder = Path(root) / '2026-10-04_120000'
        folder.mkdir()
        t = np.linspace(0.0, 1.0, 64)
        signal = 0.02 * np.sin(2 * np.pi * 3 * t)

        assert not is_capture_session(folder), 'an empty folder is no capture'
        assert signal_column_for(folder, 'Channel D') == 'Channel D'

        # a Phase-1 capture: a session, but the spectrum was a .psdata
        session = folder / '2026-10-04_120000_session.json'
        session.write_text(json.dumps({'sync': {}}), encoding='utf-8')
        assert is_capture_session(folder)
        try:
            session_scope_file(folder)
        except FileNotFoundError as error:
            assert '.psdata' in str(error), error
        else:
            raise AssertionError('a capture with no scope trace must fail')

        np.savez_compressed(folder / '2026-10-04_120000_scope.npz',
                            t=t, signal=signal, aux=2 * signal)
        session.write_text(json.dumps({'scope': {
            'file': '2026-10-04_120000_scope.npz', 'channel': 'C',
            'range_v': 0.05, 'overflow_channels': ['C']}}), encoding='utf-8')
        npz = session_scope_file(folder)
        assert npz.name == '2026-10-04_120000_scope.npz', npz
        assert session_scope_file(session) == npz, 'the json names it too'
        assert signal_column_for(folder, 'Channel D') == 'Channel C'
        x, y = load_trace(npz, 'Channel D')   # the transmission regardless
        assert np.allclose(x, t) and np.allclose(y, signal)

        csv = Path(root) / 'trace.csv'
        csv.write_text('Time,Channel D\n(ms),(V)\n\n0,1\n1,2\n',
                       encoding='utf-8')
        assert not is_capture_session(csv)
        x, y = load_trace(csv, 'Channel D')
        assert list(x) == [0.0, 1.0] and list(y) == [1.0, 2.0]
    print('scope_trace self-test passed')


if __name__ == '__main__':
    _self_test()
