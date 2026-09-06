"""Explore a synced capture's spectrum against its video, then mark it.

    python pico_scope/mode_video_sync_mark.py --session <capture folder>
    python pico_scope/mode_video_sync_mark.py --session <folder> --scope <file>.psdata
    python pico_scope/mode_video_sync_mark.py --self-test

Two separate windows, one after the other - not at once:

1. **Explore.** The same viewer as `mode_video_sync_show.py`: hover the
   spectrum, see which transverse mode produced each peak, close the window
   once you know what you are looking at.
2. **Annotate.** The same marking window as
   `extract_df_and_fsr_from_scope_csv.py`: drag pairs of peaks, get df / FSR
   / NA extracted exactly the same way, on the very same trace.

The two do not need to run together - by the time you are marking, you
already know what each peak is from step 1. See the docstrings of
`mode_video_sync_show.py` and `extract_df_and_fsr_from_scope_csv.py` for the
controls of each window.

The marking is cached as a '<data file>.modemarks.json' sidecar (see
`pico_scope/mode_marks_cache.py`), next to the .psdata for a Phase 1 capture
(the same file `extract_df_and_fsr_from_scope_csv.py` would use for it), or
next to the session's own recorded scope trace for a Phase 2 capture - so a
marking made here is found again the same way as any other marked
measurement.
"""

import argparse
import json
import sys
from pathlib import Path

# --- what happens when this file is run (edit these, then press Run) -------
# Nothing here needs the command line; the arguments exist for scripting.
ACTION = 'mark'        # 'mark' | 'self-test'
SESSION = r"C:\Users\OsipLab\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Daily measurements and notes\2026-09-07\40cm\Second trial\2026-09-06_163607"
SCOPE_FILE = ''        # the .psdata of a Phase 1 capture; '' for Phase 2
SNAP_TO_BRIGHTEST = True

# --- the cavity being measured (edit this when the setup changes) ----------
# Kept as its own block rather than imported from extract_df_and_fsr_from_scope_csv.py:
# each offline script defines its own (see mode_analysis.CAVITY_ELEMENTS's
# docstring) since different sessions can be measuring different cavities.
CAVITY_ELEMENTS = [
    'LASER_OPTIK_MIRROR',
    'EDMUND_4MM_ASPHERIC_16701',
    'COASTLINE_20CM_MIRROR',
]
SHORT_ARM_LENGTHS = (0.5e-4, 2e-4)  # [m] lens-scan span around the collimation point
MID_ARM_LENGTH = 1.5e-2           # [m] only used by 4-element cavities
N_points = 300                    # lens positions simulated across SHORT_ARM_LENGTHS
SHORT_ARM_LENGTH = 0.7e-2   # [m] near mirror -> lens (the physical one, not the simulation's scan)

# a literal '--self-test' is ensured below before mode_video_sync_show is
# imported, so its own backend check (which only looks at sys.argv, not this
# file's ACTION) picks the same backend and does not fight this line.
_SELF_TEST = ACTION == 'self-test' or '--self-test' in sys.argv
if _SELF_TEST and '--self-test' not in sys.argv:
    sys.argv.append('--self-test')

import matplotlib
matplotlib.use('Agg' if _SELF_TEST else 'Qt5Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pico_scope.mode_analysis import (cavity_fsr_mhz, get_na_interpolators,  # noqa: E402
                                      pair_positions_results, pair_summary)
from pico_scope.mode_marking import mark_pairs, positions_and_widths  # noqa: E402
from pico_scope.mode_marks_cache import (ask_use_cached_marks, complete_pairs,  # noqa: E402
                                         load_cached_marks, make_record,
                                         save_marks)
from pico_scope.mode_video_sync import (SIGNAL_COLUMN, latest_session,  # noqa: E402
                                        release_frames)
from pico_scope.mode_video_sync_show import (ModeSpectrumViewer,  # noqa: E402
                                             camera_label, load_synced_trace,
                                             session_pixel_size_mm)
from utilities.utils import append_numerical_result_line, ask_long_arm_length  # noqa: E402


def explore(session_path, trace, frames, windows, brightness, session,
           source, snap):
    """Open the video-synced viewer; block until the user closes it."""
    title = f'{Path(session_path).name} - {source}'
    viewer = ModeSpectrumViewer(trace, frames, windows, brightness, title, snap,
                                pixel_size_mm=session_pixel_size_mm(session),
                                camera_label=camera_label(session))
    print('Explore the synced video - hover the spectrum to see each mode, '
         'close the window when you are ready to annotate.')
    plt.show(block=True)
    viewer.close_fit()


def annotate(trace, mark_target_path):
    """Mark the mode pairs on `trace`, reusing a cached marking if there is
    one - exactly extract_df_and_fsr_from_scope_csv.py's marking step, but
    on an already-loaded trace instead of a CSV read from disk."""
    cached = load_cached_marks(mark_target_path, min_pairs=2,
                               signal_column=SIGNAL_COLUMN)
    if cached is not None and ask_use_cached_marks(cached, mark_target_path):
        print("  using the cached marks")
        return cached['marks'], cached['long_arm_m']

    raw_marks = mark_pairs(trace.t, trace.signal,
                           title=Path(mark_target_path).name)
    marks = complete_pairs(raw_marks)
    if len(marks) != len(raw_marks):
        print(f"Ignoring {len(raw_marks) - len(marks)} incomplete pair(s).")
    long_arm_length = ask_long_arm_length()  # [m], prompted in cm
    if marks:
        # next to the trace's own file, so mode_map_2d.py or another run of
        # this script reuses it instead of asking again
        save_marks(mark_target_path, make_record(
            mark_target_path, mark_target_path, marks, long_arm_length,
            signal_column=SIGNAL_COLUMN))
    return marks, long_arm_length


def compute_and_record(marks, long_arm_length, mark_target_path):
    """df / FSR / NA from `marks`, printed and appended to the results log -
    the same computation and recording as
    extract_df_and_fsr_from_scope_csv.py, factored out so main() reads
    linearly."""
    lorentzian_positions, lorentzian_widths = positions_and_widths(marks)
    print("Marked pairs:", lorentzian_positions)

    fsr_mhz = cavity_fsr_mhz(long_arm=long_arm_length, mid_arm=MID_ARM_LENGTH,
                             short_arm=SHORT_ARM_LENGTH)

    if len(lorentzian_positions) < 2:
        print("Not enough data to calculate FSR and df.")
        return

    measured_mode_spacing_MHz = pair_summary(
        pair_positions_results(lorentzian_positions, fsr_mhz=fsr_mhz)
    )['df_MHz_mean']
    print(f"Measured mode spacing: {measured_mode_spacing_MHz:.4f} MHz")

    mode_spacing_interp, mode_spacing_over_fsr_interp, na_error = get_na_interpolators(
        elements=CAVITY_ELEMENTS, long_arm=long_arm_length, mid_arm=MID_ARM_LENGTH,
        short_arm_lengths=SHORT_ARM_LENGTHS, N_points=N_points,
        measured_mode_spacing_MHz=measured_mode_spacing_MHz,
        plot_system=True)
    if mode_spacing_over_fsr_interp is None:
        raise RuntimeError(f'cavity-design NA simulation unavailable: {na_error}')

    rows = pair_positions_results(lorentzian_positions, fsr_mhz=fsr_mhz,
                                  na_over_fsr_interp=mode_spacing_over_fsr_interp,
                                  widths=lorentzian_widths)
    results_df = pd.DataFrame(rows)
    print(results_df)
    summary = pair_summary(rows)

    na_text = (f"{summary['NA_mean']:.4f}" if summary["NA_mean"] is not None
              else "unavailable (df/FSR outside the simulated range)")
    df_mhz_text = (f"{summary['df_MHz_mean']:.4f} MHz"
                  if summary["df_MHz_mean"] is not None else "unavailable")
    linewidth_text = ", ".join(
        f"linewidth_{i} = " + (f"{summary[f'fwhm_{i}_MHz_mean']:.4f} MHz"
                               if summary[f"fwhm_{i}_MHz_mean"] is not None
                               else "unavailable")
        for i in (0, 1))
    results_text = (f"long_arm_length = {long_arm_length:.4g} m, "
                    f"n_mode_pairs = {summary['n_pairs']}, "
                    f"mode_spacing = {df_mhz_text}, "
                    f"df_over_fsr = {summary['df_over_fsr_mean']:.4f}, "
                    f"{linewidth_text}, "
                    f"NA = {na_text}")
    if summary["df_over_fsr_std"] is not None:
        results_text += f" (std over pairs: df_over_fsr {summary['df_over_fsr_std']:.4f}"
        if summary["df_MHz_std"] is not None:
            results_text += f", mode_spacing {summary['df_MHz_std']:.4f} MHz"
        for i in (0, 1):
            if summary[f"fwhm_{i}_MHz_std"] is not None:
                results_text += f", linewidth_{i} {summary[f'fwhm_{i}_MHz_std']:.4f} MHz"
        if summary["NA_std"] is not None:
            results_text += f", NA {summary['NA_std']:.4f}"
        results_text += ")"
    append_numerical_result_line(mark_target_path, results_text)


# --------------------------------------------------------------- self-test
def _self_test():
    """No hardware: load_synced_trace on a synthetic Phase 2 session, build
    the viewer from it, and round-trip the marks cache on the file it names -
    the marking window itself (mark_pairs) is exercised by
    pico_scope/mode_marking.py's own self-test, not here."""
    import tempfile

    import numpy as np
    from camera_core import burst_timing
    from pico_scope.mode_marking import peak_record
    from pico_scope.mode_video_capture import save_session

    print('mode_video_sync_mark self-test')
    rng = np.random.default_rng(7)
    n, h, w = 20, 8, 10
    frames = rng.integers(0, 5, size=(n, h, w)).astype(np.uint8)
    meta = [{'block_id': i, 'camera_timestamp_ns': int(i * 1e7),
             'host_time_s': 1.0 * i} for i in range(n)]
    timing = burst_timing(meta, expected_rate_hz=100.0)
    camera_info = {'serial_number': 'x', 'make': 'basler', 'pixel_size_mm': 5.5e-3}

    with tempfile.TemporaryDirectory() as folder:
        stem = 'self_test'
        session_path, _ = save_session(Path(folder), stem, frames, meta, timing,
                                       {'burst_s': 0.12}, camera_info, None)
        session = json.loads(session_path.read_text(encoding='utf-8'))
        t_scope = np.linspace(0, 0.2, 2000)
        signal = rng.normal(0, 1.0, t_scope.size)
        scope_file = f'{stem}_scope.npz'
        np.savez_compressed(Path(folder) / scope_file, t=t_scope, signal=signal)
        session['scope'] = {'file': scope_file}
        session['sync'].update({'t0_host_s': 0.01})
        session_path.write_text(json.dumps(session, indent=1), encoding='utf-8')

        (trace, loaded_frames, windows, brightness, session_dict, source,
         mark_target_path) = load_synced_trace(session_path)
        assert mark_target_path == Path(folder) / scope_file, mark_target_path
        assert mark_target_path.is_file()
        assert windows.shape == (n, 2), windows.shape
        assert brightness.shape == (n,), brightness.shape
        assert 'calibrated host clock' in source, source
        print('  load_synced_trace resolves the Phase 2 scope file and windows')

        viewer = ModeSpectrumViewer(trace, loaded_frames, windows, brightness,
                                    'self-test', snap=False)
        assert viewer.index == 0
        release_frames(loaded_frames)
        print('  the viewer builds from the loaded trace/frames/windows')

        marks = [[peak_record(0.01, 1e-4, 1.0, 0.0), peak_record(0.03, 1e-4, 1.0, 0.0)],
                 [peak_record(0.05, 1e-4, 1.0, 0.0), peak_record(0.07, 1e-4, 1.0, 0.0)]]
        save_marks(mark_target_path, make_record(
            mark_target_path, mark_target_path, marks, 0.34,
            signal_column=SIGNAL_COLUMN))
        loaded = load_cached_marks(mark_target_path, min_pairs=2,
                                   signal_column=SIGNAL_COLUMN)
        assert loaded is not None and len(loaded['marks']) == 2, loaded
        positions, _widths = positions_and_widths(loaded['marks'])
        assert len(positions) == 2, positions
        print('  the marks cache round-trips on the scope file load_synced_trace named')

    assert ACTION in ('mark', 'self-test'), ACTION
    print('self-test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--self-test', action='store_true',
                        help='run the offline checks and exit')
    parser.add_argument('--session', default=SESSION or None,
                        help='capture folder or *_session.json; defaults to '
                             'SESSION in this file, or the newest capture')
    parser.add_argument('--scope', default=SCOPE_FILE or None,
                        help='the .psdata recorded alongside a Phase 1 capture; '
                             'omit for a Phase 2 capture, which carries its own')
    parser.add_argument('--no-snap', action='store_true',
                        help='show the frame the offset names, without snapping '
                             'to the brightest neighbour')
    args = parser.parse_args()

    if args.self_test or ACTION == 'self-test':
        _self_test()
        return
    if ACTION != 'mark':
        raise SystemExit(f'ACTION must be mark or self-test, not {ACTION!r}')

    session = args.session or latest_session()
    print(f'session: {session}')
    snap = SNAP_TO_BRIGHTEST and not args.no_snap
    (trace, frames, windows, brightness, session_dict, source,
     mark_target_path) = load_synced_trace(session, args.scope)

    explore(session, trace, frames, windows, brightness, session_dict, source, snap)
    marks, long_arm_length = annotate(trace, mark_target_path)
    release_frames(frames)
    compute_and_record(marks, long_arm_length, mark_target_path)

    # See extract_df_and_fsr_from_scope_csv.py: keeps the (non-blocking)
    # cavity-design system plot on screen instead of it flashing and closing.
    plt.show(block=True)


if __name__ == '__main__':
    main()
