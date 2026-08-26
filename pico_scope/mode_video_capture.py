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
# ROI_HEIGHT is only the fallback for --no-locate. Normally choose_roi() picks
# the height from the mode it just measured, because the camera gets moved and
# the mode's size and position move with it.
ROI_HEIGHT_CANDIDATES = (128, 192, 256, 320, 384, 448, 512, 640, 768, 1024)
# The higher orders are larger than the 0th and are the ones that must not be
# clipped, so the margin around what was actually seen is at least as wide as
# the mode itself, and never less than this.
ROI_MIN_MARGIN_ROWS = 48
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

# ps4000aRunBlock returns before the scope has actually begun sampling, so the
# host-clock estimate of where frame 0 sits is systematically early. Measured
# over 12 captures on 2026-08-26: +39.9 ms with a standard deviation of 7.8 ms,
# i.e. a 4.0-frame bias with 0.78 frames of jitter. Subtracting it puts 83% of
# captures within one frame without any fitting, which is what makes the fine
# alignment optional. Re-measure with --calibrate if the driver or the timing
# configuration changes.
HOST_T0_BIAS_S = 0.0399

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
        saturation = cam.saturation_level
        raise RuntimeError(
            f'no part of the sensor stands out above the noise during the '
            f'sweep. The brightest pixel reached {int(frames.max())} of '
            f'{saturation} ({frames.max() / saturation:.1%} of full scale) and '
            f'the largest change during the burst was {span.max():.0f} counts. '
            f'Either the cavity is not transmitting, or the light is too far '
            f'attenuated - aim for a peak near '
            f'{TARGET_PEAK_FRACTION:.0%} of full scale.')
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


def choose_roi(cam, found, target_hz=None, candidates=ROI_HEIGHT_CANDIDATES):
    """Pick the ROI height and vertical offset from the mode just measured.

    Two constraints pull against each other. The height must cover the mode
    with room for the larger higher orders, and it must be small enough that
    the sensor still reads out at the target rate - readout is paced per row,
    so rows are exactly what frame rate costs.

    Resolves them by preferring the smallest height that covers the mode, then
    checking it against the camera's own ResultingFrameRate; if nothing that
    covers the mode is fast enough, it takes the largest height that *is* fast
    enough and says so, rather than silently dropping either requirement.

    Returns a dict with `height`, `offset_y`, `covers` and `resulting_hz`.
    """
    target_hz = FRAME_RATE_HZ if target_hz is None else target_hz
    max_width, max_height = cam.max_frame_size
    margin = max(found['height'], ROI_MIN_MARGIN_ROWS)
    needed = found['height'] + 2 * margin

    def place(height):
        height = min(height, max_height)
        offset = int(np.clip(found['centre_row'] - height // 2,
                             0, max_height - height))
        return height, offset

    def rate_for(height, offset):
        cam.set_roi(ROI_WIDTH, height, ROI_OFFSET_X, offset)
        cam.exposure_us = EXPOSURE_US
        cam.frame_rate_hz = target_hz
        return cam.resulting_frame_rate

    usable = [h for h in sorted(candidates) if h <= max_height]
    fast_enough, covering = [], []
    for candidate in usable:
        height, offset = place(candidate)
        rate = rate_for(height, offset)
        covers = (offset <= found['row_min']
                  and offset + height >= found['row_max'] + 1)
        if rate >= target_hz * 0.98:
            fast_enough.append((height, offset, rate, covers))
        if covers and height >= needed:
            covering.append((height, offset, rate, covers))

    both = [c for c in fast_enough if c[0] >= needed and c[3]]
    if both:
        height, offset, rate, covers = min(both, key=lambda c: c[0])
        note = None
    elif fast_enough:
        height, offset, rate, covers = max(fast_enough, key=lambda c: c[0])
        note = (f'no height that both covers the mode with its margin '
                f'({needed} binned rows) and sustains {target_hz:g} Hz; took '
                f'the tallest that keeps the rate. '
                + ('The mode still fits, with less margin than wanted.'
                   if covers else
                   'THE MODE DOES NOT FIT - it will be clipped. Move the '
                   'camera so the mode sits nearer the sensor centre, or '
                   'accept a lower frame rate.'))
    else:
        raise RuntimeError(
            f'no ROI height sustains {target_hz:g} Hz at '
            f'{cam.pixel_format}. Lower FRAME_RATE_HZ, or use Mono8, whose '
            f'readout is about 2.4x faster per row.')

    cam.set_roi(ROI_WIDTH, height, ROI_OFFSET_X, offset)
    cam.exposure_us = EXPOSURE_US
    cam.frame_rate_hz = target_hz
    result = {'height': height, 'offset_y': offset, 'covers': covers,
              'resulting_hz': cam.resulting_frame_rate, 'needed_rows': needed,
              'margin_rows': min(found['row_min'] - offset,
                                 offset + height - found['row_max'] - 1),
              'note': note}
    print(f'  -> ROI {ROI_WIDTH}x{height} at offset_y {offset} '
          f'(binned rows {offset}-{offset + height}), '
          f'{result["resulting_hz"]:.1f} Hz')
    print(f'     mode occupies {found["row_min"]}-{found["row_max"]}, '
          f'{result["margin_rows"]} rows of margin')
    if note:
        print(f'     ! {note}')
    return result


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
# One burst is not enough to judge the level. At a fixed light level the peak
# varies about 2.3x from burst to burst, because it depends on which resonance
# that burst happened to catch - measured over 12 bursts on 2026-08-26, peak
# 1778 to 4095 while the mean stayed within 27-32. A check made from a single
# burst therefore passes and then lets the real capture clip, which is exactly
# what happened twice. So several bursts are taken and the verdict is formed
# from the worst of them, with headroom for a future burst brighter still.
LEVEL_BURSTS = 4
# The pre-flight bursts must be as long as the capture. A shorter one samples
# fewer free spectral ranges and so has fewer chances to catch a strong
# resonance, which biases the predicted peak low: measured, 120-frame bursts
# reach about 15% higher than 40-frame ones at the same light level. None means
# "same as the capture".
LEVEL_BURST_FRAMES = None
LEVEL_SAFETY = 1.3               # margin above the brightest burst yet seen
LEVEL_TOO_DIM_FRACTION = 0.10    # below this the capture works but wastes range
LEVEL_CLIPPED_STEP_DB = 6.0      # blind back-off while the peak is censored


def measure_light_level(cam, n_bursts=LEVEL_BURSTS, n_frames=None):
    """Peak and saturation statistics over several independent bursts.

    Returns the per-burst peaks along with the summary the verdict uses. The
    figure that matters is the *worst* burst, not the average one: the capture
    only has to clip once to be spoiled.
    """
    n_frames = n_frames or LEVEL_BURST_FRAMES or N_FRAMES
    saturation = cam.saturation_level
    peaks, fractions = [], []
    for _ in range(n_bursts):
        frames, _ = cam.record_burst(n_frames)
        peaks.append(int(frames.max()))
        fractions.append(float((frames >= saturation).mean()))
    peaks = np.array(peaks)
    return {
        'gain_db': cam.gain_db,
        'saturation_level': saturation,
        'peaks': peaks.tolist(),
        'peak_max': int(peaks.max()),
        'peak_median': float(np.median(peaks)),
        'peak_fraction': float(peaks.max() / saturation),
        'peak_spread': float(peaks.max() / max(peaks.min(), 1)),
        'saturated_fraction': float(max(fractions)),
        'n_bursts': n_bursts,
        'n_frames': n_frames,
    }


def check_light_level(cam, adjust_gain=True, n_bursts=LEVEL_BURSTS,
                      n_frames=None):
    """Measure the light level over several bursts and trim gain, or explain.

    Runs before the real capture, because a saturated burst cannot be rescued
    afterwards. Gain is the only knob this may touch: the exposure is pinned to
    just under the frame period (shortening it opens dead time in which a
    1.5 ms resonance disappears entirely), and the pixel format is chosen for
    headroom already.

    The verdict allows LEVEL_SAFETY of headroom above the brightest burst seen,
    since the capture itself is one more draw from the same spread and may land
    higher than anything measured here.

    Returns a dict describing the level. When the light is too bright even at
    minimum gain, `ok` is False and `advice` says by what factor the optics
    have to be attenuated - there is no software fix at that point.
    """
    low, _high = cam.gain_limits_db
    history = []
    for _ in range(5):
        level = measure_light_level(cam, n_bursts, n_frames)
        history.append(level)
        print(f'  gain {level["gain_db"]:5.1f} dB, {level["n_frames"]}-frame '
              f'bursts -> peaks '
              f'{level["peaks"]} of {level["saturation_level"]} '
              f'({level["peak_fraction"]:.1%} worst, '
              f'{level["peak_spread"]:.1f}x spread), saturated '
              f'{level["saturated_fraction"]:.4%}')
        expected_worst = level['peak_max'] * LEVEL_SAFETY
        if (expected_worst < level['saturation_level']
                and level['saturated_fraction'] <= MAX_SATURATED_FRACTION):
            break
        if not adjust_gain or cam.gain_db <= low + 1e-6:
            break
        if level['peak_max'] >= level['saturation_level']:
            # Pinned at full scale: the measurement is censored, so how far
            # over we are is unknown and the computed step would understate it.
            # Back off by a fixed stride instead and measure again.
            cam.gain_db = max(low, cam.gain_db - LEVEL_CLIPPED_STEP_DB)
        else:
            overshoot = expected_worst / (TARGET_PEAK_FRACTION
                                          * level['saturation_level'])
            cam.gain_db = max(low,
                              cam.gain_db - 20 * np.log10(max(overshoot, 1.01)))

    level = history[-1]
    saturation = level['saturation_level']
    expected_worst = level['peak_max'] * LEVEL_SAFETY
    ok = (expected_worst < saturation
          and level['saturated_fraction'] <= MAX_SATURATED_FRACTION)
    advice = None
    if not ok:
        at_min = cam.gain_db <= low + 1e-6
        reduce_by = expected_worst / (TARGET_PEAK_FRACTION * saturation)
        advice = (
            f'the worst of {level["n_bursts"]} bursts peaked at '
            f'{level["peak_max"]} of {saturation} '
            f'({level["peak_fraction"]:.1%} of full scale) with '
            f'{level["saturated_fraction"]:.3%} of pixels saturated'
            + (f', at the minimum gain of {low:.1f} dB' if at_min else '')
            + f'. Allowing {LEVEL_SAFETY:.1f}x for a brighter burst than any '
              f'seen, that clips. Attenuate the light by about '
              f'{reduce_by:.1f}x - the exposure is pinned to '
              f'{cam.exposure_us / 1000:.1f} ms by the frame rate, and '
              f'shortening it would open dead time in which a resonance can '
              f'hide.')
    elif level['peak_fraction'] < LEVEL_TOO_DIM_FRACTION:
        advice = (f'usable, but dim: the brightest burst reached only '
                  f'{level["peak_fraction"]:.1%} of full scale, so most of the '
                  f'range is unused. About '
                  f'{TARGET_PEAK_FRACTION / level["peak_fraction"]:.1f}x more '
                  f'light would improve the brightness SNR.')
    # A copy, because `level` *is* history[-1]: putting `history` into it would
    # make the dict contain itself, which json.dumps rejects as a circular
    # reference - and it did, after the frames had been written.
    result = dict(level)
    result.update({'ok': ok, 'advice': advice, 'history': history,
                   'expected_worst': expected_worst})
    return result


# %% [Step 2] Configuring the camera ----------------------------------------
def configure(cam, offset_y, roi_height=None):
    """Apply the capture settings and print every check worth failing on."""
    roi_height = ROI_HEIGHT if roi_height is None else roi_height
    cam.set_pixel_format(PIXEL_FORMAT)
    cam.set_throughput_limit(THROUGHPUT_BPS)
    binning_info = cam.set_binning(BINNING)
    roi = cam.set_roi(ROI_WIDTH, roi_height, ROI_OFFSET_X, offset_y)
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
def _json_default(value):
    """Make numpy scalars and arrays serialisable.

    Without this a single numpy integer anywhere in the metadata - and they
    arrive from every measurement - raises part-way through writing the session
    file, after the frame stack has already been saved. The capture then leaves
    a folder of arrays with nothing describing them, which is unrecoverable.
    Losing a session to a type is not a trade worth making.
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'{type(value).__name__} is not JSON serialisable')



def save_session(folder, stem, frames, meta, timing, checks, camera_info,
                 mode_location):
    """Write the frame stack, the mask and the session record."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    mask = varying_pixel_mask(frames, MASK_THRESHOLD)
    frames_name, mask_name = f'{stem}_frames.npy', f'{stem}_mask.npy'

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
    # Serialise before writing anything large. The metadata is the fragile
    # part - it is assembled from a dozen measurements, any of which can carry
    # a type json refuses - and it is also the irreplaceable part, since the
    # per-frame timestamps and the offset exist nowhere else. Twice now a
    # capture has written 94 MB of frames and then failed here, leaving arrays
    # nothing could interpret. Failing before the arrays exist costs a rerun;
    # failing after costs the data.
    text = json.dumps(session, indent=1, default=_json_default)
    np.save(folder / frames_name, frames)
    np.save(folder / mask_name, mask)
    session_path = folder / f'{stem}_session.json'
    session_path.write_text(text, encoding='utf-8')
    return session_path, mask


def capture(serial_number=SERIAL_NUMBER, output_root=OUTPUT_ROOT,
            locate=True, prompt=True):
    """Locate the mode, configure, wait for the scope, record, save."""
    cam = BaslerCamera(serial_number)
    cam.open()
    try:
        mode_location = None
        # the --no-locate fallbacks; choose_roi() overrides both when it runs
        offset_y, roi_height = ROI_OFFSET_Y, ROI_HEIGHT
        if locate:
            print('--- locating the mode (whole sensor) ---')
            mode_location = locate_mode(cam)
            report_mode_location(mode_location)
            cam.set_pixel_format(PIXEL_FORMAT)
            cam.set_binning(BINNING)
            roi_choice = choose_roi(cam, mode_location)
            offset_y, roi_height = roi_choice['offset_y'], roi_choice['height']

        checks = configure(cam, offset_y, roi_height)

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
        # the --no-locate fallbacks; choose_roi() overrides both when it runs
        offset_y, roi_height = ROI_OFFSET_Y, ROI_HEIGHT
        if locate:
            print('--- locating the mode (whole sensor) ---')
            mode_location = locate_mode(cam)
            report_mode_location(mode_location)
            cam.set_pixel_format(PIXEL_FORMAT)
            cam.set_binning(BINNING)
            roi_choice = choose_roi(cam, mode_location)
            offset_y, roi_height = roi_choice['offset_y'], roi_choice['height']
        checks = configure(cam, offset_y, roi_height)
        if mode_location is not None:
            checks['roi_choice'] = roi_choice
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
    t0_host_raw = host_before_burst - host_scope_start
    t0_host = t0_host_raw + HOST_T0_BIAS_S
    print(f'\n  {frames.shape} {frames.dtype}, dropped {timing["n_dropped"]}')
    print(f'  frame period {timing["period_s_median"] * 1e3:.4f} ms '
          f'+- {timing["period_s_std"] * 1e3:.4f} ms')
    print(f'  scope {block_info["n_collected"]} samples at '
          f'{block_info["interval_s"] * 1e9:.0f} ns, overflow '
          f'{block_info["overflow_channels"] or "none"}')
    print(f'  t0 from the host clocks: {t0_host_raw * 1e3:.2f} ms raw, '
          f'{t0_host * 1e3:.2f} ms after the {HOST_T0_BIAS_S * 1e3:+.1f} ms '
          f'calibration')

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
        't0_host_raw_s': t0_host_raw,
        'host_t0_bias_s': HOST_T0_BIAS_S,
        'host_scope_start_s': host_scope_start,
        'host_before_burst_s': host_before_burst,
        'host_after_burst_s': host_after_burst,
        'description': 'both instruments driven from one process. t0_host is '
                       'the delay between RunBlock and the burst starting, '
                       'plus the calibrated RunBlock bias. Good to about one '
                       'frame on its own; mode_video_sync.py --refine takes it '
                       'to a hundredth of one.',
    })
    session_path.write_text(json.dumps(session, indent=1, default=_json_default),
                            encoding='utf-8')
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

    # numpy scalars must survive, and a cycle must not be constructible: both
    # have cost a capture its session file after the frames were written
    with tempfile.TemporaryDirectory() as folder:
        numpy_checks = {'peak_max': np.int64(3), 'worst': np.float64(4.0),
                        'covers': np.bool_(True), 'peaks': np.arange(3)}
        path, _ = save_session(folder, 'np', frames, meta, timing,
                               numpy_checks, {'serial_number': 'x'}, None)
        stored = json.loads(path.read_text())['checks']
        assert stored == {'peak_max': 3, 'worst': 4.0, 'covers': True,
                          'peaks': [0, 1, 2]}, stored

    # and when the metadata cannot be written, nothing large is left behind
    with tempfile.TemporaryDirectory() as folder:
        try:
            save_session(folder, 'bad', frames, meta, timing,
                         {'unserialisable': object()}, {'serial_number': 'x'},
                         None)
        except TypeError:
            pass
        else:
            raise AssertionError('unserialisable metadata should have raised')
        leftovers = list(Path(folder).glob('*.npy'))
        assert not leftovers, f'frames written despite failed metadata: {leftovers}'
    print('  numpy metadata survives, and a metadata failure leaves no orphan '
          'arrays')

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
    # --- the ROI chooser, against a stand-in camera ------------------------
    # Rows are what frame rate costs, so a fake camera whose rate is inversely
    # proportional to ROI height exercises the real trade-off without hardware.
    class _FakeCam:
        max_frame_size = (1024, 1024)
        pixel_format = 'Mono12'
        exposure_us = EXPOSURE_US

        def __init__(self, seconds_per_row=1.29e-5):
            self.seconds_per_row = seconds_per_row
            self.height = None
            self.offset_y = None

        def set_roi(self, width, height, offset_x, offset_y):
            self.height, self.offset_y = height, offset_y
            return {'width': width, 'height': height, 'offset_y': offset_y}

        frame_rate_hz = property(lambda self: 0.0, lambda self, value: None)

        @property
        def resulting_frame_rate(self):
            return 1.0 / (self.height * self.seconds_per_row)

    def _found(row_min, row_max):
        return {'row_min': row_min, 'row_max': row_max,
                'centre_row': (row_min + row_max) // 2,
                'height': row_max - row_min + 1}

    # a small central mode: the smallest height that still covers it wins
    small_mode = _found(480, 560)
    choice = choose_roi(_FakeCam(), small_mode, target_hz=100.0)
    assert choice['covers'], choice
    assert choice['resulting_hz'] >= 98.0, choice
    assert choice['margin_rows'] >= small_mode['height'], choice
    assert choice['note'] is None, choice

    # a larger mode has to be given a taller ROI
    bigger = choose_roi(_FakeCam(), _found(400, 640), target_hz=100.0)
    assert bigger['height'] > choice['height'], (choice, bigger)

    # a mode near the top edge is followed rather than centred, and still fits
    edge = choose_roi(_FakeCam(), _found(20, 100), target_hz=100.0)
    assert edge['offset_y'] == 0, edge
    assert edge['covers'], edge

    # when margin and frame rate cannot both be had, the rate is kept and the
    # compromise is reported rather than made silently
    tight = choose_roi(_FakeCam(), _found(300, 740), target_hz=100.0)
    assert tight['note'] is not None, tight
    assert tight['resulting_hz'] >= 98.0, 'the frame rate is the hard constraint'
    print('  choose_roi: covers the mode, follows it to the sensor edge, and '
          'reports the compromise when the rate and the margin conflict')

    # --- the light-level pre-flight, against a stand-in camera -------------
    # The camera this mimics is the real failure: burst peaks that vary about
    # 2.3x at a fixed light level, so no single burst clips while the next one
    # well might. A one-burst check passes here; the multi-burst one must not.
    class _LevelCam:
        exposure_us = EXPOSURE_US

        def __init__(self, base_peaks, saturation=4095, gain_db=0.0):
            self.base_peaks = list(base_peaks)
            self._saturation = saturation
            self.gain_db = gain_db
            self._next = 0

        gain_limits_db = property(lambda self: (0.0, 23.1))
        saturation_level = property(lambda self: self._saturation)

        def record_burst(self, n_frames):
            base = self.base_peaks[self._next % len(self.base_peaks)]
            self._next += 1
            value = min(base * 10 ** (self.gain_db / 20.0), self._saturation)
            frame = np.zeros((n_frames, 10, 10), dtype=np.uint16)
            frame[:, 5, 5] = int(round(value))
            return frame, None

    # peaks that never clip on their own, but leave no headroom for the next
    marginal = _LevelCam([1800, 3300, 2600, 4000])
    verdict = check_light_level(marginal, adjust_gain=False, n_bursts=4,
                                n_frames=4)
    assert max(verdict['peaks']) < verdict['saturation_level'], \
        'the premise: no single burst actually clips'
    assert verdict['saturated_fraction'] == 0.0
    assert not verdict['ok'], 'no headroom for a brighter burst - must refuse'
    assert 'Attenuate' in verdict['advice']
    assert verdict['peak_spread'] > 2.0, verdict['peak_spread']

    # comfortable level: passes
    comfortable = _LevelCam([1500, 1800, 1650, 2000])
    good = check_light_level(comfortable, adjust_gain=False, n_bursts=4,
                             n_frames=4)
    assert good['ok'] and good['advice'] is None, good

    # far too dim: usable, but said so
    faint = _LevelCam([180, 260, 210, 300])
    dim = check_light_level(faint, adjust_gain=False, n_bursts=4, n_frames=4)
    assert dim['ok'] and dim['advice'] and 'dim' in dim['advice'], dim

    # gain is trimmed when it can help, and the verdict then passes
    # safe at 0 dB, clipping at 12 dB: exactly the case gain can rescue
    hot = _LevelCam([700, 1200, 1000, 1500], gain_db=12.0)
    trimmed = check_light_level(hot, adjust_gain=True, n_bursts=4, n_frames=4)
    assert hot.gain_db < 12.0, 'gain should have been reduced'
    assert trimmed['ok'], trimmed['advice']

    # at minimum gain there is nothing left to try, and it says so
    pinned = _LevelCam([4095, 4095, 4095, 4095], gain_db=0.0)
    stuck = check_light_level(pinned, adjust_gain=True, n_bursts=2, n_frames=4)
    assert not stuck['ok']
    assert 'minimum gain' in stuck['advice'], stuck['advice']
    print('  the pre-flight judges from the worst of several bursts, so a level '
          'that no single burst clips at is still refused when it has no '
          'headroom')

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
            offset_y, roi_height = ROI_OFFSET_Y, ROI_HEIGHT
            if not args.no_locate:
                found = locate_mode(cam)
                report_mode_location(found)
                cam.set_pixel_format(PIXEL_FORMAT)
                cam.set_binning(BINNING)
                choice = choose_roi(cam, found)
                offset_y, roi_height = choice['offset_y'], choice['height']
            configure(cam, offset_y, roi_height)
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
