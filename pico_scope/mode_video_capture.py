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
# Mono12, not Mono8: as the laser warms the transmission climbs, and a
# clipped peak stops the camera tracking the photodiode - the one thing that
# breaks the alignment fit. 12 bits buy 16x the headroom. It costs frame rate
# (12-bit readout is slower per row), which is paid for by fewer ROI rows.
PIXEL_FORMAT = 'Mono12'
GAIN_DB = 0.0                   # measured: gain only makes the noise worse
BINNING = 2                     # firmware mode is Sum on this model: 4x signal
THROUGHPUT_BPS = 212_352_571    # the cameras' own default; 150 MB/s caps us at 74 Hz

# ROI in BINNED pixels. Width is free - readout is paced per row - so keep the
# whole sensor width and spend the budget on rows.
ROI_WIDTH = 1024                # = full 2048 sensor columns at binning 2
ROI_HEIGHT = 384                # = 768 sensor rows; 99.1 Hz in Mono12
ROI_OFFSET_X = 0
# None means locate_mode() finds it at every run, which is the right default -
# the mode moves whenever the cavity is realigned. The number is the fallback
# used by --no-locate: measured 2026-08-26, mode at binned rows 529-640 centred
# on 584, so 584 - ROI_HEIGHT // 2. Re-run --locate after any realignment.
ROI_OFFSET_Y = 406

# --- the scope, when this script drives it too (Phase 2) -------------------
# Only one program can own the scope, so PicoScope 7 must be closed. The block
# is made just long enough to contain the burst plus the few tens of
# milliseconds it takes to get from RunBlock to the first exposure: every extra
# second of slack would add another ~4 free-spectral-range aliases for the
# optional fine alignment to sort out.
SCOPE_CHANNEL = 'D'             # cavity transmission, as everywhere else
SCOPE_RANGE_V = 0.01            # +-10 mV; peaks sit around 1-2 mV
SCOPE_COUPLING = 'DC'
SCOPE_SAMPLE_INTERVAL_S = 1e-5  # 100 kS/s, the rate the lab already uses
SCOPE_PAD_S = 0.30              # recorded before and after the burst

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
def _extent(profile, threshold, n_sigma=5.0):
    """Where a 1-D profile rises above its own baseline, as (min, max) index.

    A profile is the span image summed along one axis, not maximised along it:
    summing averages the per-pixel noise down over a thousand pixels while the
    mode adds coherently. The baseline is the median, so it is set by the empty
    majority of the sensor rather than by the mode.

    Even summed, the empty part of the sensor is not flat - it is a pedestal
    with real scatter - so a threshold set purely as a fraction of the peak
    dips into the noise and reports the mode as filling the sensor. The cut is
    therefore the stricter of two: `threshold` of the way from baseline to peak,
    and `n_sigma` robust standard deviations above the baseline.
    """
    profile = np.asarray(profile, dtype=float)
    baseline = float(np.median(profile))
    peak = float(profile.max())
    if peak <= baseline:
        return None
    # median absolute deviation -> sigma, unaffected by the mode itself
    sigma = 1.4826 * float(np.median(np.abs(profile - baseline)))
    cut = max(baseline + threshold * (peak - baseline),
              baseline + n_sigma * sigma)
    lit = np.nonzero(profile > cut)[0]
    return (int(lit.min()), int(lit.max())) if lit.size else None


def locate_mode(cam, n_frames=150, threshold=0.1):
    """Where on the sensor does the transmitted mode sit?

    Takes a whole-sensor burst and looks at what *changes* during it, which
    isolates the sweeping mode from any static background or stray light. The
    answer moves whenever the cavity is realigned, so this runs before every
    capture rather than being written down as a constant.

    Uses the same pixel format, gain and binning as the capture: in Mono12 the
    read noise is resolved rather than truncated away, which moves the threshold
    this has to clear.

    The burst has to be long enough to catch the higher-order modes and not just
    the 0th - they are larger and displaced, and they are the ones the ROI must
    not clip. At the whole-sensor frame rate 150 frames covers a couple of
    seconds, i.e. several free spectral ranges.

    Returns a dict in binned pixels; `centre_row` is what the ROI is centred on.
    """
    cam.set_pixel_format(PIXEL_FORMAT)
    cam.set_throughput_limit(THROUGHPUT_BPS)
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
    rows = _extent(span.sum(axis=1), threshold)
    cols = _extent(span.sum(axis=0), threshold)
    if rows is None or cols is None:
        raise RuntimeError(
            'no part of the sensor stands out above the noise during the '
            'sweep. Is the cavity transmitting, and does the burst cover a '
            'resonance?')
    found = {
        'row_min': rows[0], 'row_max': rows[1],
        'col_min': cols[0], 'col_max': cols[1],
        'centre_row': (rows[0] + rows[1]) // 2,
        'centre_col': (cols[0] + cols[1]) // 2,
        'peak_pixel': int(frames.max()),
        'saturation_level': int(cam.saturation_level),
        'pixel_format': cam.pixel_format,
        'saturated_fraction': float((frames >= cam.saturation_level).mean()),
        'span_max': float(span.max()),
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
    print(f"  peak pixel {found['peak_pixel']} of "
          f"{found['saturation_level']} ({found['pixel_format']}), saturated "
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


# %% [Step 1b] Checking the light level --------------------------------------
# A clipped peak is the one thing that reliably breaks the alignment fit: the
# camera stops tracking the photodiode exactly where the signal is strongest.
# Measured earlier on this setup, 1% of samples clipped is survivable and 5%
# is not, so the gate is set well below that.
MAX_SATURATED_FRACTION = 0.001   # 0.1% of pixel samples
TARGET_PEAK_FRACTION = 0.7       # aim the brightest pixel here, of full scale


def check_light_level(cam, n_frames=60, adjust_gain=True):
    """Measure the peak level and trim the gain until it is safe, or explain.

    Runs before the real capture, because a saturated burst cannot be rescued
    afterwards. Gain is the only knob this may touch: the exposure is pinned to
    just under the frame period (shortening it opens dead time in which a
    1.5 ms resonance disappears entirely), and the pixel format is chosen for
    headroom already.

    Returns a dict describing the level. When the light is too bright even at
    minimum gain, `ok` is False and `advice` says what has to change in the
    optics - there is no software fix at that point.
    """
    low, high = cam.gain_limits_db
    saturation = cam.saturation_level
    history = []
    for _ in range(4):
        frames, _ = cam.record_burst(n_frames)
        peak = int(frames.max())
        fraction = float((frames >= saturation).mean())
        history.append({'gain_db': cam.gain_db, 'peak': peak,
                        'peak_fraction': peak / saturation,
                        'saturated_fraction': fraction})
        print(f'  gain {cam.gain_db:5.1f} dB -> peak {peak:5d}/{saturation} '
              f'({peak / saturation:5.1%}), saturated {fraction:.4%}')
        if fraction <= MAX_SATURATED_FRACTION:
            break
        if not adjust_gain or cam.gain_db <= low + 1e-6:
            break
        # drop the gain by the ratio needed to bring the peak to target
        overshoot = max(peak / (TARGET_PEAK_FRACTION * saturation), 1.01)
        cam.gain_db = max(low, cam.gain_db - 20 * np.log10(overshoot))

    last = history[-1]
    # Both tests matter, and the fraction alone is not enough: a peak pinned at
    # full scale means the brightest resonances are clipped through their cores,
    # flattening exactly the features the alignment fit reads, even when only a
    # handful of pixels are involved.
    ok = (last['saturated_fraction'] <= MAX_SATURATED_FRACTION
          and last['peak_fraction'] < 0.99)
    advice = None
    if not ok:
        at_min = cam.gain_db <= low + 1e-6
        advice = (
            f'the peak sits at {last["peak_fraction"]:.1%} of full scale with '
            f'{last["saturated_fraction"]:.3%} of pixels saturated'
            + (f' at the minimum gain of {low:.1f} dB' if at_min else '')
            + '. Nothing in software can fix this: the exposure is pinned to '
              f'{cam.exposure_us / 1000:.1f} ms by the {1e6 / cam.exposure_us:.0f} Hz '
              'frame rate, and shortening it would open dead time in which a '
              'resonance can hide. Attenuate the light reaching the camera - an '
              'ND filter, or a weaker split off the transmission.')
    return {'ok': ok, 'history': history, 'advice': advice,
            'gain_db': cam.gain_db, 'saturation_level': saturation,
            **last}


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
            try:
                input('Press Enter to record the burst... ')
            except EOFError:
                raise RuntimeError(
                    'no console to prompt on. Start the PicoScope recording '
                    'first and re-run with --no-prompt, which records '
                    'immediately - and give the scope record enough length to '
                    'cover the delay before this starts.')

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


# %% [Step 3b] Driving both instruments (Phase 2) -----------------------------
def capture_synchronized(serial_number=SERIAL_NUMBER, output_root=OUTPUT_ROOT,
                         locate=True, n_frames=None, scope_serial=None,
                         adjust_gain=True, require_level=True):
    """Record the spectrum and the mode video from one process.

    The camera is configured first and the scope block started last, so that as
    little as possible happens between the scope beginning to record and the
    first exposure. That gap is what `t0_host` estimates, and keeping it small
    is what makes the fine alignment optional: the smaller the gap, the smaller
    the window the fit has to search, and the better the nominal offset is on
    its own.

    Nothing is aligned here. The session records the scope trace, the frames,
    and `t0_host` from the two host timestamps; `mode_video_sync.py` refines
    that offset later if it is worth refining.
    """
    from pico_scope.ps4000a_scope import PicoScope4000A

    n_frames = N_FRAMES if n_frames is None else n_frames
    cam = BaslerCamera(serial_number)
    cam.open()
    scope = PicoScope4000A(scope_serial)
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
        burst_s = n_frames / FRAME_RATE_HZ

        print('\n--- light level ---')
        level = check_light_level(cam, adjust_gain=adjust_gain)
        checks['light_level'] = level
        if not level['ok']:
            if require_level:
                raise RuntimeError('too bright to capture: ' + level['advice'])
            print(f'  ! {level["advice"]}')
            print('  ! continuing anyway (--allow-saturated); the alignment '
                  'fit will probably not lock.')

        scope.open()
        scope.configure_channel(SCOPE_CHANNEL, enabled=True,
                                coupling=SCOPE_COUPLING, range_v=SCOPE_RANGE_V)
        for name in ('A', 'B', 'C', 'D'):
            if name != SCOPE_CHANNEL:
                scope.configure_channel(name, enabled=False)
        scope.configure_trigger(enabled=False)   # start immediately
        duration = burst_s + 2 * SCOPE_PAD_S
        print(f'\n--- scope {scope.variant} s/n {scope.serial} ---')
        print(f'  channel {SCOPE_CHANNEL}, +-{SCOPE_RANGE_V * 1e3:g} mV '
              f'{SCOPE_COUPLING}, {SCOPE_SAMPLE_INTERVAL_S * 1e6:g} us/sample')
        print(f'  block {duration:.3f} s = {burst_s:.3f} s burst + '
              f'2 x {SCOPE_PAD_S:.2f} s pad')

        block = scope.start_block(duration, SCOPE_SAMPLE_INTERVAL_S)
        host_scope_start = block['host_start_s']
        print(f'  recording; starting the burst ...')
        host_before_burst = time.time()
        frames, meta = cam.record_burst(n_frames)
        host_after_burst = time.time()
        scope.wait_block(timeout_s=duration * 3 + 10)
        t_scope, volts, block_info = scope.read_block()
        # Read the camera's settings while it is still open - everything below
        # happens after the finally clause has closed it.
        camera_info = cam.describe()
        scope_info = {'serial': scope.serial, 'variant': scope.variant}
    finally:
        scope.close()
        cam.close()

    timing = burst_timing(meta, expected_rate_hz=FRAME_RATE_HZ)
    # Scope t = 0 is the trigger, i.e. the start of the block, so the host-clock
    # estimate of where frame 0 sits is simply the delay between the two calls.
    # It carries whatever latency RunBlock and StartGrabbing add, which is the
    # error the fine alignment exists to remove.
    t0_host = host_before_burst - host_scope_start
    print(f'\n  {frames.shape} {frames.dtype}, dropped {timing["n_dropped"]}')
    print(f'  frame period {timing["period_s_median"] * 1e3:.4f} ms '
          f'+- {timing["period_s_std"] * 1e3:.4f} ms')
    print(f'  scope {block_info["n_collected"]} samples at '
          f'{block_info["interval_s"] * 1e9:.0f} ns, overflow '
          f'{block_info["overflow_channels"] or "none"}')
    print(f'  t0 from the host clocks: {t0_host * 1e3:.2f} ms into the block')

    stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    folder = Path(output_root) / stamp
    session_path, mask = save_session(
        folder, stamp, frames, meta, timing, checks, camera_info,
        mode_location)

    # the scope trace lives with the capture, so no .psdata is needed
    signal = volts[SCOPE_CHANNEL]
    np.savez_compressed(folder / f'{stamp}_scope.npz', t=t_scope, signal=signal)
    session = json.loads(session_path.read_text(encoding='utf-8'))
    session['scope'] = {
        'file': f'{stamp}_scope.npz',
        'channel': SCOPE_CHANNEL,
        'range_v': SCOPE_RANGE_V,
        'coupling': SCOPE_COUPLING,
        'sample_interval_s': block_info['interval_s'],
        'n_samples': block_info['n_collected'],
        'duration_s': duration,
        'overflow_channels': block_info['overflow_channels'],
        'serial': scope_info['serial'],
        'variant': scope_info['variant'],
    }
    session['sync'].update({
        'method': 'host_clock',
        't0_host_s': t0_host,
        'host_scope_start_s': host_scope_start,
        'host_before_burst_s': host_before_burst,
        'host_after_burst_s': host_after_burst,
        'description': 'both instruments driven from one process; t0_host is '
                       'the delay between RunBlock and the burst starting. '
                       'Refine with mode_video_sync.py --refine if needed.',
    })
    session_path.write_text(json.dumps(session, indent=1), encoding='utf-8')
    print(f'  mask covers {int(mask.sum())} of {mask.size} pixels '
          f'({mask.mean():.2%})')
    print(f'  saved {session_path}')
    print(f'\nOptional fine alignment:')
    print(f'  python pico_scope/mode_video_sync.py --session "{folder}" --refine')
    return session_path


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
    parser.add_argument('--no-prompt', action='store_true',
                        help='record immediately instead of waiting for Enter; '
                             'the PicoScope recording must already be running '
                             'and long enough to cover the delay')
    parser.add_argument('--no-locate', action='store_true',
                        help='skip the reconnaissance and use ROI_OFFSET_Y')
    parser.add_argument('--allow-saturated', action='store_true',
                        help='capture even if the camera is clipping, instead '
                             'of refusing; the alignment fit will likely fail')
    parser.add_argument('--levels', action='store_true',
                        help='measure the light level and exit, without '
                             'capturing')
    parser.add_argument('--scope', action='store_true',
                        help='drive the scope from here too, instead of '
                             'recording it by hand in PicoScope 7 (which must '
                             'then be closed - only one program can own it)')
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
    if args.levels:
        cam = BaslerCamera(args.serial)
        cam.open()
        try:
            offset_y = ROI_OFFSET_Y
            if not args.no_locate:
                offset_y = int(np.clip(
                    locate_mode(cam)['centre_row'] - ROI_HEIGHT // 2,
                    0, cam.max_frame_size[1] - ROI_HEIGHT))
            configure(cam, offset_y)
            print('\n--- light level ---')
            level = check_light_level(cam)
            print(f"  {'OK' if level['ok'] else 'TOO BRIGHT'}: "
                  f"{level['advice'] or 'peak is in range'}")
        finally:
            cam.close()
        return
    if args.scope:
        capture_synchronized(args.serial, locate=not args.no_locate,
                             require_level=not args.allow_saturated)
    else:
        capture(args.serial, locate=not args.no_locate,
                prompt=not args.no_prompt)


if __name__ == '__main__':
    main()
