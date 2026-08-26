"""Record a mode video to sit alongside a PicoScope spectrum recording.

    python pico_scope/mode_video_capture.py            # locate the mode, then capture
    python pico_scope/mode_video_capture.py --locate   # just find the mode, no capture
    python pico_scope/mode_video_capture.py --self-test # no hardware; checks the file format

## How a capture goes

1. Start the PicoScope 7 recording.
2. Press Enter here.

That is the whole protocol, and the loose ordering is deliberate: nothing needs
to be started at a known instant, because the two records are aligned afterwards
from the light itself. The camera and the Channel D photodiode watch the same
cavity transmission, so each frame's brightness is the scope trace integrated
over that frame's exposure - and `pico_scope/mode_video_sync.py` recovers the
one unknown offset by fitting it. See SYNCHRONIZED_VIDEO_SPECTRUM.md.

The only real requirement is **overlap**: the burst must sit inside the scope
record, so record the scope for comfortably longer than the burst lasts and
start it first. The script prints the burst duration before asking.

## What comes out

A session folder holding

    <stem>_frames.npy    the frame stack, uint8 or uint16
    <stem>_mask.npy      the pixels the mode actually lit
    <stem>_session.json  camera settings, per-frame timing, brightness series

The scope side stays a `.psdata` exported from PicoScope 7, exactly as before,
so every existing loader and analysis script keeps working untouched.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'basler_cam'))
from basler_cameras import BaslerCamera, burst_timing  # noqa: E402
from pico_scope.mode_video_sync import (frame_brightness,  # noqa: E402
                                        varying_pixel_mask)

# --- the camera and how it is driven (this is the block to edit) -----------
SERIAL_NUMBER = '25173136'      # the camera pointed at the cavity mode
FRAME_RATE_HZ = 100.0           # see the peak-blending check below
EXPOSURE_US = 9900.0            # just under the frame period, never equal to it
N_FRAMES = 120                  # 1.2 s at 100 Hz
PIXEL_FORMAT = 'Mono8'
GAIN_DB = 0.0                   # measured: gain only makes the noise worse
BINNING = 2                     # firmware mode is Sum on this model: 4x signal
THROUGHPUT_BPS = 212_352_571    # the cameras' own default; 150 MB/s caps us at 74 Hz

# ROI in BINNED pixels. Width is free - readout is paced per row - so keep the
# whole sensor width and spend the budget on rows.
ROI_WIDTH = 1024                # = full 2048 sensor columns at binning 2
ROI_HEIGHT = 512                # = 1024 sensor rows
ROI_OFFSET_X = 0
ROI_OFFSET_Y = None             # None: found by locate_mode() at every run

# --- what the capture is checked against -----------------------------------
# The tightest 0th->1st spacing measured across the 2026-08-23 mode maps. Two
# resonances closer together than one frame period blend into a single image,
# so the frame period has to stay well under this.
TIGHTEST_PEAK_SPACING_S = 0.0247
MIN_FRAMES_BETWEEN_PEAKS = 2.0
MASK_THRESHOLD = 0.15           # fraction of the peak-to-peak that counts as lit
PIXEL_SIZE_MM = 5.5 / 1000.0    # acA2040 pitch; effective pitch is this x binning

# --- where captures are written --------------------------------------------
try:
    from local_config import PATH_DATA_LOCAL
    OUTPUT_ROOT = Path(PATH_DATA_LOCAL) / 'mode_video'
except ImportError:                                   # pragma: no cover
    OUTPUT_ROOT = Path.cwd() / 'mode_video'


# %% [Step 1] Finding the mode ----------------------------------------------
def locate_mode(cam, n_frames=40, threshold=0.1):
    """Where on the sensor does the transmitted mode sit?

    Takes a whole-sensor burst and looks at what *changes* during it, which
    isolates the sweeping mode from any static background or stray light. The
    answer moves whenever the cavity is realigned, so this runs before every
    capture rather than being written down as a constant.

    Returns a dict in binned pixels; `centre_row` is what the ROI is centred on.
    """
    cam.set_binning(BINNING)
    cam.set_roi_full()
    cam.exposure_us = EXPOSURE_US
    cam.gain_db = GAIN_DB
    cam.frame_rate_hz = cam.resulting_frame_rate
    frames, _ = cam.record_burst(n_frames)

    span = (frames.max(axis=0).astype(np.float32)
            - frames.min(axis=0).astype(np.float32))
    if span.max() <= 0:
        raise RuntimeError(
            'nothing on the sensor changed during the reconnaissance burst - '
            'is the laser on and the cavity transmitting?')
    rows = np.nonzero(span.max(axis=1) > threshold * span.max())[0]
    cols = np.nonzero(span.max(axis=0) > threshold * span.max())[0]
    found = {
        'row_min': int(rows.min()), 'row_max': int(rows.max()),
        'col_min': int(cols.min()), 'col_max': int(cols.max()),
        'centre_row': int((rows.min() + rows.max()) // 2),
        'centre_col': int((cols.min() + cols.max()) // 2),
        'peak_pixel': int(frames.max()),
        'saturated_fraction': float((frames >= cam.saturation_level).mean()),
    }
    found['height'] = found['row_max'] - found['row_min'] + 1
    found['width'] = found['col_max'] - found['col_min'] + 1
    return found


def report_mode_location(found, roi_height=ROI_HEIGHT):
    """Print the reconnaissance, and warn if the ROI would clip the mode."""
    print(f"  mode spans rows {found['row_min']}-{found['row_max']} "
          f"({found['height']} binned rows), cols {found['col_min']}-"
          f"{found['col_max']} ({found['width']} binned cols)")
    print(f"  centred at row {found['centre_row']}, col {found['centre_col']}")
    print(f"  peak pixel {found['peak_pixel']}, saturated "
          f"{found['saturated_fraction']:.3%}")
    margin = (roi_height - found['height']) // 2
    if margin < found['height']:
        print(f'  ! only {margin} binned rows of margin around the mode. Higher '
              f'orders are larger than the 0th - consider more rows, at the '
              f'cost of frame rate.')
    if found['saturated_fraction'] > 0.01:
        print('  ! more than 1% of pixels are saturated. The offset fit '
              'degrades badly past ~5%; shorten the exposure or attenuate.')
    return margin


# %% [Step 2] Configuring the camera ----------------------------------------
def configure(cam, offset_y):
    """Apply the capture settings and print every check worth failing on."""
    cam.set_pixel_format(PIXEL_FORMAT)
    cam.set_throughput_limit(THROUGHPUT_BPS)
    binning_info = cam.set_binning(BINNING)
    roi = cam.set_roi(ROI_WIDTH, ROI_HEIGHT, ROI_OFFSET_X, offset_y)
    cam.exposure_us = EXPOSURE_US
    cam.gain_db = GAIN_DB
    cam.frame_rate_hz = FRAME_RATE_HZ
    chunks = cam.enable_chunks()

    period = 1.0 / FRAME_RATE_HZ
    burst = N_FRAMES * period
    sensor_w = roi['width'] * BINNING
    sensor_h = roi['height'] * BINNING
    link_max = cam.max_frame_rate_for(sensor_w, sensor_h, PIXEL_FORMAT, BINNING)
    frames_per_gap = TIGHTEST_PEAK_SPACING_S / period

    print(f'\n--- camera {cam.serial_number} ({cam.model}) ---')
    print(f'  frame {roi["width"]}x{roi["height"]} {PIXEL_FORMAT} at offset '
          f'({roi["offset_x"]}, {roi["offset_y"]}) = sensor '
          f'{sensor_w}x{sensor_h}, rows {roi["offset_y"] * BINNING}-'
          f'{(roi["offset_y"] + roi["height"]) * BINNING}')
    print(f'  binning {binning_info["binning"]} (mode not selectable on this '
          f'model; measured to be Sum), effective pixel '
          f'{PIXEL_SIZE_MM * BINNING * 1000:.1f} um')
    print(f'  exposure {cam.exposure_us:.0f} us, gain {cam.gain_db:.1f} dB, '
          f'chunks {chunks}')

    print(f'\n--- bandwidth (check 5) ---')
    print(f'  {roi["width"] * roi["height"] / 1e6:.3f} MB/frame, link allows '
          f'{link_max:.1f} Hz at {THROUGHPUT_BPS / 1e6:.0f} MB/s')
    print(f'  camera reports ResultingFrameRate {cam.resulting_frame_rate:.2f} Hz')
    resulting = cam.assert_frame_rate_reachable(FRAME_RATE_HZ)

    print(f'\n--- peak blending (check 6) ---')
    print(f'  frame period {period * 1e3:.2f} ms against the tightest measured '
          f'0th->1st spacing of {TIGHTEST_PEAK_SPACING_S * 1e3:.1f} ms')
    print(f'  {frames_per_gap:.1f} frames between the closest pair '
          f'(want at least {MIN_FRAMES_BETWEEN_PEAKS:.0f})')
    if frames_per_gap < MIN_FRAMES_BETWEEN_PEAKS:
        raise RuntimeError(
            f'at {FRAME_RATE_HZ:g} Hz only {frames_per_gap:.1f} frames separate '
            f'the closest 0th and 1st orders, so they will blend into one '
            f'image. Raise the frame rate (and cut ROI rows to afford it).')
    print(f'  burst {burst:.3f} s for {N_FRAMES} frames, '
          f'{N_FRAMES * roi["width"] * roi["height"] / 1e6:.0f} MB')
    return {'roi': roi, 'binning': binning_info, 'chunks': chunks,
            'link_max_hz': link_max, 'resulting_hz': resulting,
            'burst_s': burst, 'frames_between_closest_peaks': frames_per_gap}


# %% [Step 3] Recording and saving -------------------------------------------
def save_session(folder, stem, frames, meta, timing, checks, camera_info,
                 mode_location):
    """Write the frame stack, the mask and the session record."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    mask = varying_pixel_mask(frames, MASK_THRESHOLD)

    frames_name, mask_name = f'{stem}_frames.npy', f'{stem}_mask.npy'
    np.save(folder / frames_name, frames)
    np.save(folder / mask_name, mask)

    session = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'sync': {
            'method': 'optical',
            'description': 'frame brightness fitted against the scope trace; '
                           'see pico_scope/mode_video_sync.py',
            'signal_column': 'Channel D',
        },
        'frames_file': frames_name,
        'mask_file': mask_name,
        'frames_shape': list(frames.shape),
        'frames_dtype': str(frames.dtype),
        'camera': camera_info,
        'binning': BINNING,
        'effective_pixel_size_mm': PIXEL_SIZE_MM * BINNING,
        'requested_frame_rate_hz': FRAME_RATE_HZ,
        'exposure_s': EXPOSURE_US / 1e6,
        'checks': checks,
        'mode_location': mode_location,
        'mask_threshold': MASK_THRESHOLD,
        'mask_pixels': int(mask.sum()),
        'timing': timing,
        'meta': meta,
        # Both series on purpose: masking cuts the noise 5-7x but excludes light
        # the photodiode still sees, so the fit compares its margin on each.
        'brightness_full': frame_brightness(frames).tolist(),
        'brightness_masked': frame_brightness(frames, mask).tolist(),
    }
    session_path = folder / f'{stem}_session.json'
    session_path.write_text(json.dumps(session, indent=1), encoding='utf-8')
    return session_path, mask


def capture(serial_number=SERIAL_NUMBER, output_root=OUTPUT_ROOT,
            locate=True, prompt=True):
    """Locate the mode, configure, wait for the scope, record, save."""
    cam = BaslerCamera(serial_number)
    cam.open()
    try:
        mode_location = None
        offset_y = ROI_OFFSET_Y
        if locate:
            print('--- locating the mode (whole sensor) ---')
            mode_location = locate_mode(cam)
            report_mode_location(mode_location)
            offset_y = int(np.clip(
                mode_location['centre_row'] - ROI_HEIGHT // 2,
                0, cam.max_frame_size[1] - ROI_HEIGHT))
            print(f'  -> ROI offset_y {offset_y} (binned)')

        checks = configure(cam, offset_y)

        if prompt:
            print(f'\nStart the PicoScope recording now - it must run for '
                  f'longer than the {checks["burst_s"]:.2f} s burst and must '
                  f'already be running when the burst starts.')
            input('Press Enter to record the burst... ')

        print(f'recording {N_FRAMES} frames ...')
        tic = time.time()
        frames, meta = cam.record_burst(N_FRAMES)
        wall = time.time() - tic
        timing = burst_timing(meta, expected_rate_hz=FRAME_RATE_HZ)
        print(f'  {frames.shape} {frames.dtype} in {wall:.2f} s')
        print(f'  dropped {timing["n_dropped"]} {timing["dropped"]}')
        print(f'  period {timing["period_s_median"] * 1e3:.4f} ms '
              f'+- {timing["period_s_std"] * 1e3:.4f} ms over '
              f'{timing["duration_s"]:.3f} s')
        if timing['n_dropped']:
            print('  ! frames were dropped. The fit still works - it uses the '
                  'camera timestamps, not a uniform grid - but the video has '
                  'gaps.')

        stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        folder = Path(output_root) / stamp
        session_path, mask = save_session(
            folder, stamp, frames, meta, timing, checks, cam.describe(),
            mode_location)
        print(f'\n  mask covers {int(mask.sum())} of {mask.size} pixels '
              f'({mask.mean():.2%})')
        print(f'  saved {session_path}')
        print(f'\nNow stop and save the PicoScope recording as .psdata, then:')
        print(f'  python pico_scope/mode_video_sync.py --session '
              f'"{folder}" --scope "<that file>.psdata"')
        return session_path
    finally:
        cam.close()


# %% [Step 4] Self-test -------------------------------------------------------
def _self_test():
    """No hardware: check the session round-trips and the checks bite."""
    import tempfile
    from pico_scope.mode_video_sync import load_session, release_frames

    print('mode_video_capture self-test')
    rng = np.random.default_rng(3)
    n, h, w = 12, 8, 10
    frames = rng.integers(0, 5, size=(n, h, w)).astype(np.uint8)
    frames[:, 3:5, 4:6] += (np.arange(n, dtype=np.uint8) * 15)[:, None, None]
    meta = [{'block_id': i, 'camera_timestamp_ns': int(i * 1e7),
             'host_time_s': 1.0 * i} for i in range(n)]
    timing = burst_timing(meta, expected_rate_hz=100.0)
    assert timing['n_dropped'] == 0 and timing['timestamps_look_like_ns']

    with tempfile.TemporaryDirectory() as folder:
        path, mask = save_session(folder, 'test', frames, meta, timing,
                                  {'burst_s': 0.12}, {'serial_number': 'x'},
                                  None)
        session, loaded = load_session(path)
        assert loaded.shape == frames.shape, loaded.shape
        assert np.array_equal(np.asarray(loaded), frames)
        assert session['frames_dtype'] == 'uint8'
        assert len(session['meta']) == n
        assert len(session['brightness_full']) == n
        assert len(session['brightness_masked']) == n
        assert session['mask_pixels'] == int(mask.sum()) > 0
        assert session['effective_pixel_size_mm'] == PIXEL_SIZE_MM * BINNING
        # the folder form of load_session finds the same file
        assert load_session(Path(folder), mmap=False)[0]['created'] == \
            session['created']
        masked = np.array(session['brightness_masked'])
        full = np.array(session['brightness_full'])
        assert masked.max() - masked.min() > full.max() - full.min(), \
            'masking should raise the dynamic range'
        # Windows keeps a memory-mapped file locked, so the session folder
        # cannot be removed until the frames are released.
        release_frames(loaded)
    print('  session round-trips, both brightness series present')

    # the configured frame rate must actually resolve the closest peaks
    period = 1.0 / FRAME_RATE_HZ
    assert TIGHTEST_PEAK_SPACING_S / period >= MIN_FRAMES_BETWEEN_PEAKS, \
        f'the configured {FRAME_RATE_HZ} Hz cannot resolve the closest orders'
    assert EXPOSURE_US / 1e6 < period, \
        'exposure must be shorter than the frame period, or it becomes the cap'
    print(f'  {FRAME_RATE_HZ:g} Hz gives '
          f'{TIGHTEST_PEAK_SPACING_S / period:.1f} frames between the closest '
          f'orders, exposure {EXPOSURE_US / 1e3:.1f} ms < period '
          f'{period * 1e3:.1f} ms')
    print('self-test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--self-test', action='store_true',
                        help='run the offline checks and exit')
    parser.add_argument('--locate', action='store_true',
                        help='find the mode and report, without capturing')
    parser.add_argument('--serial', default=SERIAL_NUMBER,
                        help=f'camera serial number (default {SERIAL_NUMBER})')
    parser.add_argument('--frames', type=int, default=None,
                        help=f'frames to record (default {N_FRAMES})')
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.frames:
        globals()['N_FRAMES'] = args.frames
    if args.locate:
        cam = BaslerCamera(args.serial)
        cam.open()
        try:
            print('--- locating the mode (whole sensor) ---')
            report_mode_location(locate_mode(cam))
        finally:
            cam.set_binning(1)
            cam.set_roi_full()
            cam.close()
        return
    capture(args.serial)


if __name__ == '__main__':
    main()
