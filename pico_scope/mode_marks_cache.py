"""The '<data file>.modemarks.json' sidecar: one marking, shared by the scripts.

Marking a trace by hand is the slow part of every analysis, so the marks are
written next to the data file they belong to and read back by whichever script
runs next. Both markers write the same sidecar:

    pico_scope/extract_df_and_fsr_from_scope_csv.py   one file, df / FSR / NA
    pico_scope/mode_map_2d.py                         a whole sweep, as a map

which is the point of this module: the marking done at the bench, while the
measurement is being taken, is the marking the 2D map later reuses - it is no
longer thrown away and done again.

The record is a plain JSON dict:

    version        CACHE_VERSION, so an incompatible layout is ignored
    source_name    the data file's name, for reading the sidecar by eye
    source_mtime   its mtime when it was marked; a sidecar older than the data
                   belongs to a different measurement of the same name
    csv_path       the CSV (i.e. which waveform buffer) the marks are positions
                   on - psdata files hold several
    signal_column  the CSV column that was marked ('Channel D'); absent in
                   sidecars written before this field existed
    long_arm_m     the cavity's long arm [m], which sets the FSR
    marks          the pairs, exactly as mode_marking.mark_pairs returns them
    key            the swept parameter's value - only mode_map_2d has one

Everything a consumer must agree on before reusing a marking (which buffer,
which column, how many pairs) is checked by load_cached_marks(); a record that
fails a check is not returned, and the caller marks the file again.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# repo root on sys.path so `utilities.*` resolves even when the importer was
# run as a script from inside pico_scope/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.utils import psdata_to_csv  # noqa: E402

CACHE_SUFFIX = '.modemarks.json'
CACHE_VERSION = 1


def complete_pairs(marks):
    """Only the fully marked pairs - a window closed halfway leaves one open."""
    return [pair for pair in marks if len(pair) == 2]


def trace_csv_path(path):
    """The readable CSV for `path`; .psdata is converted (and its waveform
    buffer chosen) by the same helper the other scope scripts use."""
    path = Path(path)
    if path.suffix.lower() == '.psdata':
        return psdata_to_csv(path)
    return str(path)


def cache_file(data_path):
    """The marks sidecar for a data file: '<stem>.modemarks.json' beside it."""
    data_path = Path(data_path)
    return data_path.with_name(data_path.stem + CACHE_SUFFIX)


def make_record(data_path, csv_path, marks, long_arm_m, signal_column=None,
                key=None):
    """The record to save for one marked file (see the module docstring).

    Only the complete pairs are kept: a half-marked pair is an artefact of how
    the window was closed, not something to hand the next script.
    """
    data_path = Path(data_path)
    record = {'version': CACHE_VERSION,
              'source_name': data_path.name,
              'source_mtime': data_path.stat().st_mtime,
              'csv_path': str(csv_path),
              'long_arm_m': float(long_arm_m),
              'marks': complete_pairs(marks)}
    if signal_column is not None:
        record['signal_column'] = signal_column
    if key is not None:
        record['key'] = float(key)
    return record


def load_cached_marks(data_path, min_pairs=1, signal_column=None):
    """The cached marking of `data_path`, or None if there is no usable one.

    Usable means: the layout is the current one, the data file has not been
    written since the marking (a cache older than its data belongs to a
    different measurement that happened to have the same name), it holds at
    least `min_pairs` complete pairs, and it was made on `signal_column` if
    both that and the sidecar say which column was marked. Anything else
    returns None with a line saying why, and the caller marks the file again.
    """
    path = cache_file(data_path)
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        print(f"  ignoring unreadable {path.name}: {error}")
        return None
    if record.get('version') != CACHE_VERSION:
        return None
    if record.get('source_mtime') != Path(data_path).stat().st_mtime:
        print(f"  {Path(data_path).name} changed since {path.name} was written "
              "- marking it again.")
        return None

    # Older sidecars do not say which column was marked; they all come from
    # scripts whose column was the default, so an absent field is not a reason
    # to throw the marking away.
    marked_column = record.get('signal_column')
    if signal_column and marked_column and marked_column != signal_column:
        print(f"  {path.name} was marked on '{marked_column}', not "
              f"'{signal_column}' - marking it again.")
        return None

    record['marks'] = complete_pairs(record.get('marks') or [])
    if len(record['marks']) < min_pairs:
        print(f"  {path.name} holds only {len(record['marks'])} complete "
              f"pair(s), {min_pairs} needed - marking it again.")
        return None
    return record


def ask_use_cached_marks(record, data_path):
    """Ask whether to reuse a cached marking, or to mark the file again.

    Marking is the slow part of a run, so the cache is what one usually wants
    and is the default here - but a marking one is no longer happy with can
    only be replaced by hand, and deleting the sidecar (or setting REMARK_KEYS
    and starting over) is a detour. A run with no console to answer from keeps
    the cache; the callers' REMARK flags are how such a run asks for the other
    choice.

    The long arm is part of what is printed because it is saved with the marks
    and reused with them: it is the one number that would otherwise be typed
    again on every run, and answering 'n' is how a wrong one is corrected.
    """
    path = cache_file(data_path)
    marked = datetime.fromtimestamp(path.stat().st_mtime)
    print(f"  {path.name} holds a marking of this file: "
          f"{len(record['marks'])} pairs, long arm "
          f"{record['long_arm_m'] * 100:.4g} cm, marked {marked:%Y-%m-%d %H:%M}")
    while True:
        try:
            answer = input("  Use it? y = yes, n = mark the file again "
                           "[default y]: ").strip().lower()
        except EOFError:
            return True
        if answer in ('', 'y', 'yes'):
            return True
        if answer in ('n', 'no'):
            print("  marking it again - the new marks replace the cached ones")
            return False
        print("  Please answer 'y' or 'n'.")


def save_marks(data_path, record):
    path = cache_file(data_path)
    path.write_text(json.dumps(record, indent=1), encoding='utf-8')
    print(f"  marks cached in {path.name}")
    return path


def resolve_csv(record, data_path):
    """The CSV a cached marking was made on, reconverting it if it is gone.

    The psdata -> CSV cache expires (PSDATA_CSV_CACHE_MAX_AGE_DAYS), while the
    marks sidecar does not. The marks are positions on one waveform buffer, so
    a reconversion is only usable if it yields the same buffer file.
    """
    csv_path = Path(record['csv_path'])
    if csv_path.is_file():
        return str(csv_path)
    print(f"  the CSV this marking was made on is gone ({csv_path.name}) "
          "- converting again")
    new_path = Path(trace_csv_path(data_path))
    if new_path.name != csv_path.name:
        raise RuntimeError(
            f"the marks in {cache_file(data_path).name} were made on waveform "
            f"buffer '{csv_path.name}', but '{new_path.name}' was chosen - "
            "rerun and pick the same buffer, or delete the sidecar to mark the "
            "file again.")
    record['csv_path'] = str(new_path)
    save_marks(data_path, record)
    return str(new_path)


# ------------------------------------------------------------------ self-test
def _self_test():
    """Round-trip a sidecar through a temporary data file - no GUI, no scope."""
    import os
    import tempfile

    from pico_scope.mode_marking import peak_record

    with tempfile.TemporaryDirectory() as folder:
        data_path = Path(folder) / 'without EOM.csv'
        data_path.write_text('Time,Channel D\n0,0\n', encoding='utf-8')

        assert cache_file(data_path).name == 'without EOM.modemarks.json'
        assert load_cached_marks(data_path) is None, 'no sidecar, no marks'

        marks = [[peak_record(0.0, 1e-6, 1.0, 0.0),
                  peak_record(2e-4, 1e-6, 1.0, 0.0)],
                 [peak_record(1e-3, 1e-6, 1.0, 0.0),
                  peak_record(1.2e-3, 1e-6, 1.0, 0.0)],
                 [peak_record(2e-3)]]  # a pair the window was closed halfway
        record = make_record(data_path, data_path, marks, 0.33,
                             signal_column='Channel D')
        assert len(record['marks']) == 2, 'the half pair was kept'
        assert 'key' not in record, 'a keyless record grew a key'
        save_marks(data_path, record)

        # what the sibling script reads back
        loaded = load_cached_marks(data_path, min_pairs=2,
                                   signal_column='Channel D')
        assert loaded is not None and len(loaded['marks']) == 2, loaded
        assert loaded['long_arm_m'] == 0.33, loaded
        assert loaded['marks'][1][1]['x0'] == 1.2e-3, loaded['marks']
        assert resolve_csv(loaded, data_path) == str(data_path)

        # ... and what it refuses to read back
        assert load_cached_marks(data_path, min_pairs=3) is None, 'too few pairs'
        assert load_cached_marks(data_path, signal_column='Channel B') is None, \
            'marks from another channel were accepted'

        # a sidecar written before the data file it claims to describe
        stat = data_path.stat()
        os.utime(data_path, (stat.st_atime, stat.st_mtime + 10))
        assert load_cached_marks(data_path) is None, 'a stale sidecar was used'

        # an unreadable sidecar is a reason to mark again, not to crash
        cache_file(data_path).write_text('{not json', encoding='utf-8')
        assert load_cached_marks(data_path) is None

        # the map script's extra field survives the round trip
        keyed = make_record(data_path, data_path, marks, 0.33, key=36)
        save_marks(data_path, keyed)
        assert load_cached_marks(data_path)['key'] == 36.0

    print('mode_marks_cache self-test passed')


if __name__ == '__main__':
    _self_test()
