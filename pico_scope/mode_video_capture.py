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
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_core import burst_timing  # noqa: E402
from pico_scope.mode_video_sync import (SESSION_ROOT,  # noqa: E402
                                        frame_brightness, varying_pixel_mask)

# The camera makes this script can drive. Imported one at a time and only when
# needed: a machine with just one SDK installed must still run, and importing
# an absent SDK at module scope would take the whole file down with it.
CAMERA_BACKENDS = {
    'basler': ('basler_cam', 'basler_cameras', 'BaslerCamera'),
    'ximea': ('ximea_cam', 'ximea_cameras', 'XimeaCamera'),
}

# --- what happens when this file is run (edit these, then press Run) -------
# Nothing here needs the command line; the arguments exist for scripting and
# override these when given.
ACTION = 'capture'      # 'capture' | 'levels' | 'locate' | 'self-test'
CAMERA = None           # 'basler' | 'ximea' | None = the only one connected
DRIVE_SCOPE = True      # False: you record the scope yourself in PicoScope 7
LOCATE_FIRST = True     # locate the mode first; False reuses the last ROI
ALLOW_SATURATED = False # capture even when the light is too bright

# --- the camera and how it is driven (this is the block to edit) -----------
# None means whichever camera is connected, of either make, which is right
# whenever there is only one - the serial that used to sit here belonged to a
# camera that is not always the one plugged in. Name a serial only to pick
# between cameras that are both connected; resolve_camera() then says which.
SERIAL_NUMBER = None
FRAME_RATE_HZ = 100.0           # see the peak-blending check below
# The exposure follows the frame rate rather than being typed out beside it:
# as long as the period allows, less the gap the sensor needs between frames.
# Asking for the whole period does not fail loudly, it quietly lowers the rate,
# and the inverse-minus-a-bit had to be recomputed by hand at every new rate.
# The floor is what stops the gap vanishing at high rates, where 1% of a short
# period is less than the sensor wants.
EXPOSURE_GAP_US = max(100.0, 0.01 * 1e6 / FRAME_RATE_HZ)
EXPOSURE_US = 1e6 / FRAME_RATE_HZ - EXPOSURE_GAP_US   # 9900 us at 100 Hz
N_FRAMES = 120                  # 1.2 s at 100 Hz
# None: the deepest format the camera offers - Mono12 on the Basler, Mono10 on
# the XIMEA, whose sensor has no more to give. Depth is wanted for headroom: as
# the laser warms the transmission climbs, and a clipped peak makes a poor
# image of the mode. Name a format to force one (Mono8 reads out faster).
PIXEL_FORMAT = None
GAIN_DB = 0.0                   # measured: gain only makes the noise worse
# N x N sum. The Basler does it in firmware, before the link; the XIMEA has no
# firmware binning at all, so its wrapper sums on the host. Either way the
# signal goes up by N**2 and the data goes down by it.
BINNING = 2
# None: as much of the link as the camera may have. A number caps it, which is
# only wanted when two cameras share a bus - and this capture drives one.
THROUGHPUT_BPS = None

# ROI in BINNED pixels. None: the full sensor width. On the Basler width is
# free - readout is paced per row - so the budget is spent on rows; choose_roi
# narrows the width only if a camera turns out to charge for columns too.
ROI_WIDTH = None
ROI_HEIGHT_CANDIDATES = (128, 192, 256, 320, 384, 448, 512, 640, 768, 1024)
# The higher orders are larger than the 0th and are the ones that must not be
# clipped, so the margin around what was actually seen is at least as wide as
# the mode itself, and never less than this.
ROI_MIN_MARGIN_ROWS = 48
ROI_OFFSET_X = 0

# None, and meant to stay None. The mode moves whenever the cavity is realigned
# or the camera is nudged, so locate_mode() measures where it is at every run
# and choose_roi() sizes the ROI around what it found. A height and an offset
# written here would be right only until the next time the setup is touched,
# and then wrong silently: the capture would still run, on rows the mode has
# left. Set them only to pin the ROI deliberately - comparing two captures
# frame for frame, say. Otherwise --no-locate reuses the ROI of the last
# capture, which at least was measured; see fallback_roi().
ROI_OFFSET_Y = None
ROI_HEIGHT = None

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
# host-clock estimate of where frame 0 sits is systematically early. Part of
# that delay is the camera's own arming time, so the bias is per make and
# measured, never borrowed: applying one camera's number to another would
# misalign every capture by an unknown constant while still claiming sub-frame
# accuracy, and nothing downstream would show it.
#
# basler: measured over 12 captures on 2026-08-26, +39.9 ms with a standard
# deviation of 7.8 ms - a 4.0-frame bias with 0.78 frames of jitter.
# Subtracting it puts 83% of captures within one frame with no fitting at all,
# which is what makes the fine alignment optional.
#
# None means not yet measured. The capture still runs and still records the raw
# host clock; it just says so, and that --refine is not optional for it.
HOST_T0_BIAS_S = {'basler': 0.0399, 'ximea': None}

# --- what the capture is checked against -----------------------------------
# The tightest 0th->1st spacing measured across the 2026-08-23 mode maps. Two
# resonances closer together than one frame period blend into a single image,
# so the frame period has to stay well under this.
TIGHTEST_PEAK_SPACING_S = 0.0247
MIN_FRAMES_BETWEEN_PEAKS = 2.0
MASK_THRESHOLD = 0.15           # fraction of the peak-to-peak that counts as lit

# --- where captures are written --------------------------------------------
# Shared with mode_video_sync, so that leaving its SESSION empty finds the
# capture this script just wrote.
OUTPUT_ROOT = SESSION_ROOT


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

    Uses the same pixel format, gain and binning as the capture: in the deeper
    formats the read noise is resolved rather than truncated away, which moves
    the threshold this has to clear.

    The burst has to be long enough to catch the higher-order modes and not just
    the 0th - they are larger and displaced, and they are the ones the ROI must
    not clip. At the whole-sensor frame rate 150 frames covers a couple of
    seconds, i.e. several free spectral ranges.

    Returns a dict in binned pixels; `centre_row` is what the ROI is centred on.
    """
    apply_camera_basics(cam)
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
    """Pick the ROI from the mode just measured.

    Two constraints pull against each other. The ROI must cover the mode with
    room for the larger higher orders, and it must be small enough that the
    camera still delivers at the target rate.

    Rows are spent first, because on the Basler width is free - readout is
    paced per row - so the whole sensor width costs nothing there. A camera
    that pays for columns too (its rate limited by data volume rather than by
    rows) gets a second resort: the width is narrowed around the mode's own
    columns rather than giving up the frame rate. Which camera is which is not
    assumed - it falls out of probing the rate.

    Prefers the smallest ROI that covers the mode; if nothing that covers it is
    fast enough, takes the largest that *is* fast enough and says so, rather
    than silently dropping either requirement.

    Returns a dict with `height`, `width`, `offset_y`, `offset_x`, `covers`
    and `resulting_hz`.
    """
    target_hz = FRAME_RATE_HZ if target_hz is None else target_hz
    max_width, max_height = cam.max_frame_size
    margin = max(found['height'], ROI_MIN_MARGIN_ROWS)
    needed = found['height'] + 2 * margin
    full_width = min(roi_width_for(cam), max_width)

    def place(height, width):
        height, width = min(height, max_height), min(width, max_width)
        offset_y = int(np.clip(found['centre_row'] - height // 2,
                               0, max_height - height))
        offset_x = int(np.clip(found['centre_col'] - width // 2,
                               0, max_width - width))
        return height, width, offset_y, offset_x

    def probe(height, width):
        height, width, offset_y, offset_x = place(height, width)
        cam.set_roi(width, height, offset_x, offset_y)
        cam.exposure_us = EXPOSURE_US
        cam.frame_rate_hz = target_hz
        return {'height': height, 'width': width,
                'offset_y': offset_y, 'offset_x': offset_x,
                'rate': cam.resulting_frame_rate,
                'covers': (offset_y <= found['row_min']
                           and offset_y + height >= found['row_max'] + 1
                           and offset_x <= found['col_min']
                           and offset_x + width >= found['col_max'] + 1)}

    # Widths to try, widest first. The narrower ones still leave the mode a
    # margin as wide as itself, so a narrowed ROI never clips what it was
    # sized around.
    needed_cols = found['width'] + 2 * max(found['width'], ROI_MIN_MARGIN_ROWS)
    widths = [full_width]
    for factor in (2, 4):
        narrower = max(needed_cols, full_width // factor)
        if narrower < widths[-1]:
            widths.append(narrower)

    usable = [h for h in sorted(candidates) if h <= max_height]
    at_full_width = []
    for width in widths:
        options = [probe(candidate, width) for candidate in usable]
        if width == full_width:
            at_full_width = options
        both = [o for o in options if o['covers'] and o['height'] >= needed
                and o['rate'] >= target_hz * 0.98]
        if both:
            best = min(both, key=lambda o: o['height'])
            note = None if width == full_width else (
                f'narrowed the ROI to {width} of {full_width} columns: at the '
                f'full width no height both covered the mode and kept '
                f'{target_hz:g} Hz. This camera pays for columns as well as '
                f'rows.')
            return _finish_roi(cam, found, best, needed, note, target_hz)

    # Nothing covers the mode at the rate, at any width. Fall back to the
    # widest view that at least keeps the rate, and say what was given up.
    fast_enough = [o for o in at_full_width if o['rate'] >= target_hz * 0.98]
    if not fast_enough:
        shallower = [f for f in cam.formats if f != cam.pixel_format]
        raise RuntimeError(
            f'no ROI that covers the mode sustains {target_hz:g} Hz at '
            f'{cam.pixel_format}. Lower FRAME_RATE_HZ, shorten the exposure, '
            f'or capture in a shallower format '
            f'({", ".join(shallower) or "none available"}), which costs fewer '
            f'bytes per pixel.')
    best = max(fast_enough, key=lambda o: o['height'])
    note = (f'no height that both covers the mode with its margin '
            f'({needed} binned rows) and sustains {target_hz:g} Hz; took '
            f'the tallest that keeps the rate. '
            + ('The mode still fits, with less margin than wanted.'
               if best['covers'] else
               'THE MODE DOES NOT FIT - it will be clipped. Move the '
               'camera so the mode sits nearer the sensor centre, or '
               'accept a lower frame rate.'))
    return _finish_roi(cam, found, best, needed, note, target_hz)


def _finish_roi(cam, found, best, needed, note, target_hz):
    """Apply the chosen ROI, report it, and return the record of the choice."""
    cam.set_roi(best['width'], best['height'], best['offset_x'],
                best['offset_y'])
    cam.exposure_us = EXPOSURE_US
    cam.frame_rate_hz = target_hz
    offset, height = best['offset_y'], best['height']
    result = {'height': height, 'width': best['width'], 'offset_y': offset,
              'offset_x': best['offset_x'], 'covers': best['covers'],
              'resulting_hz': cam.resulting_frame_rate, 'needed_rows': needed,
              'margin_rows': min(found['row_min'] - offset,
                                 offset + height - found['row_max'] - 1),
              'note': note}
    print(f'  -> ROI {best["width"]}x{height} at offset '
          f'({best["offset_x"]}, {offset}) '
          f'(binned rows {offset}-{offset + height}), '
          f'{result["resulting_hz"]:.1f} Hz')
    print(f'     mode occupies {found["row_min"]}-{found["row_max"]}, '
          f'{result["margin_rows"]} rows of margin')
    if note:
        print(f'     ! {note}')
    return result


def report_mode_location(found, roi_height=None):
    """Print the reconnaissance, and warn if the ROI would clip the mode."""
    print(f"  mode spans rows {found['row_min']}-{found['row_max']} "
          f"({found['height']} binned rows), cols {found['col_min']}-"
          f"{found['col_max']} ({found['width']} binned cols)")
    print(f"  centred at row {found['centre_row']}, col {found['centre_col']}")
    print(f"  peak pixel {found['peak_pixel']} of "
          f"{found['saturation_level']} ({found['pixel_format']}), saturated "
          f"{found['saturated_fraction']:.3%}")
    # Without a height there is no margin to judge yet - this runs before
    # choose_roi(), which measures the real margin against the ROI it picks.
    margin = None if roi_height is None else (roi_height - found['height']) // 2
    if margin is not None and margin < found['height']:
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


def camera_class(make):
    """Import one make's device layer and return its camera class.

    Deferred to here so that a missing SDK disables that make alone. Both
    device modules are imported flat, from their own folder, which is also
    what keeps ximea_cam's PyQt-importing package __init__ out of the way.
    """
    folder, module_name, class_name = CAMERA_BACKENDS[make]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / folder))
    return getattr(importlib.import_module(module_name), class_name)


def resolve_camera(make=None, serial=None):
    """Which camera to use: the one named, or the only one connected.

    Naming a camera in the file is a promise about what is plugged in today,
    and that promise goes stale - a camera gets unplugged, or swapped for the
    other one. Falling back on "the only camera there is" is both what is
    usually meant and impossible to get silently wrong.

    Returns `(camera_class, serial_number, make)`.
    """
    make = CAMERA if make is None else make
    serial = SERIAL_NUMBER if serial is None else serial
    if make and make not in CAMERA_BACKENDS:
        raise RuntimeError(f'unknown camera make {make!r}; this script drives '
                           f'{" and ".join(CAMERA_BACKENDS)}')

    found, unavailable = [], {}
    for name in ([make] if make else list(CAMERA_BACKENDS)):
        try:
            cls = camera_class(name)
        except Exception as error:
            unavailable[name] = error      # SDK not installed on this machine
            continue
        for device in cls.list_devices():
            if serial and str(device['serial_number']) != str(serial):
                continue
            found.append((cls, str(device['serial_number']), name,
                          device.get('model', '')))

    if not found:
        detail = ''
        if serial:
            detail = f' with serial {serial}'
        elif make:
            detail = f' of make {make}'
        missing = [f'{name} support is unavailable here ({error})'
                   for name, error in unavailable.items()]
        raise RuntimeError('; '.join(
            [f'no camera is connected{detail}'] + missing))
    if len(found) > 1:
        listing = ', '.join(f'{name}:{sn}' for _, sn, name, _ in found)
        raise RuntimeError(
            f'{len(found)} cameras are connected, so which one watches the '
            f'cavity mode has to be said: set CAMERA to a make, or '
            f'SERIAL_NUMBER (or --serial) to one of {listing}.')

    cls, serial_number, name, model = found[0]
    print(f'  camera {serial_number} ({model}, {name}), the only one connected')
    return cls, serial_number, name


def pixel_format_for(cam):
    """The format to capture in: the deepest the camera offers unless pinned."""
    return PIXEL_FORMAT or cam.deepest_format


def roi_width_for(cam):
    """ROI width in binned pixels: the whole sensor unless pinned."""
    return ROI_WIDTH or cam.max_frame_size[0]


def host_t0_bias_s(make):
    """The calibrated arming delay for one camera make, or 0.0 if unmeasured.

    Unmeasured means unmeasured: no number is invented and none is borrowed
    from the other camera. The caller is told, and says so in its own output.
    """
    return HOST_T0_BIAS_S.get(make) or 0.0


def apply_camera_basics(cam):
    """Format, link limit and binning - the settings ROI choices depend on.

    Binning last and before any ROI: it changes what one pixel means, so every
    size and offset after it is in different units.
    """
    fmt = cam.set_pixel_format(pixel_format_for(cam))
    # None means "as much of the link as this camera may have". Both wrappers
    # clip to their own maximum, so infinity asks for all of it. The Basler
    # opens at a deliberately low 150 MB/s, on the assumption that two cameras
    # share the bus; that cap alone drops a 384-row ROI from 99 Hz to 50, and
    # this capture drives one camera at a time.
    cam.set_throughput_limit(float('inf') if THROUGHPUT_BPS is None
                             else THROUGHPUT_BPS)
    return fmt, cam.set_binning(BINNING)


def previous_roi(root=None):
    """The ROI of the most recent capture on disk, with the file it came from.

    Where the mode sits is a property of the alignment, not of this file, so
    the only honest record of it is the last time it was actually measured.

    Returns `(offset_y, height, path)`, or None if nothing has been captured.
    """
    root = Path(OUTPUT_ROOT if root is None else root)
    sessions = sorted(root.glob('*/*_session.json'),
                      key=lambda q: q.stat().st_mtime, reverse=True)
    for path in sessions:
        try:
            roi = json.loads(path.read_text(encoding='utf-8'))['checks']['roi']
            return int(roi['offset_y']), int(roi['height']), path
        except (ValueError, KeyError, OSError, TypeError):
            continue                     # a half-written session, not a stop
    return None


def fallback_roi():
    """The ROI for a run that skips the reconnaissance.

    Pinned by ROI_OFFSET_Y and ROI_HEIGHT if they are set; otherwise the last
    capture's, which was at least measured at some point. Never a number left
    over from whenever this file happened to be written.
    """
    if (ROI_OFFSET_Y is None) != (ROI_HEIGHT is None):
        raise RuntimeError('pin both ROI_OFFSET_Y and ROI_HEIGHT or neither - '
                           'half a pinned ROI is not enough to place one.')
    if ROI_OFFSET_Y is not None:
        print(f'  ROI pinned in the file: {ROI_HEIGHT} rows at offset_y '
              f'{ROI_OFFSET_Y}')
        return ROI_OFFSET_Y, ROI_HEIGHT
    found = previous_roi()
    if found is None:
        raise RuntimeError(
            'no ROI to fall back on: the mode has not been located and no '
            'previous capture is on disk. Locate it first - LOCATE_FIRST = '
            'True, or drop --no-locate - which is the normal way round.')
    offset_y, height, path = found
    print(f'  reusing the ROI measured for {path.parent.name}: {height} rows '
          f'at offset_y {offset_y}')
    return offset_y, height


# %% [Step 2] Configuring the camera ----------------------------------------
def configure(cam, offset_y, roi_height):
    """Apply the capture settings and print every check worth failing on."""
    if offset_y is None or roi_height is None:
        raise ValueError('configure() needs a measured ROI: locate the mode '
                         'first, or take one from fallback_roi().')
    pixel_format, binning_info = apply_camera_basics(cam)
    roi = cam.set_roi(roi_width_for(cam), roi_height, ROI_OFFSET_X, offset_y)
    cam.exposure_us = EXPOSURE_US
    cam.gain_db = GAIN_DB
    cam.frame_rate_hz = FRAME_RATE_HZ
    stamps = cam.enable_timestamps()

    period = 1.0 / FRAME_RATE_HZ
    burst = N_FRAMES * period
    sensor_w = roi['width'] * BINNING
    sensor_h = roi['height'] * BINNING
    link_max = cam.max_frame_rate_for(sensor_w, sensor_h, pixel_format, BINNING)
    frames_per_gap = TIGHTEST_PEAK_SPACING_S / period

    print(f'\n--- camera {cam.serial_number} ({cam.model}) ---')
    print(f'  frame {roi["width"]}x{roi["height"]} {pixel_format} at offset '
          f'({roi["offset_x"]}, {roi["offset_y"]}) = sensor '
          f'{sensor_w}x{sensor_h}, rows {roi["offset_y"] * BINNING}-'
          f'{(roi["offset_y"] + roi["height"]) * BINNING}')
    print(f'  binning {binning_info["binning"]} ({cam.binning_mode}), full '
          f'scale {cam.saturation_level}, effective pixel '
          f'{cam.pixel_size_mm * BINNING * 1000:.1f} um')
    print(f'  exposure {cam.exposure_us:.0f} us, gain {cam.gain_db:.1f} dB, '
          f'timestamps {stamps or "carried by every frame"}')

    print(f'\n--- bandwidth (check 5) ---')
    print(f'  {sensor_w * sensor_h * cam.BYTES_PER_PIXEL[pixel_format] / 1e6:.3f} '
          f'MB/frame on the link, allowing {link_max:.1f} Hz at '
          f'{cam.throughput_limit_bps / 1e6:.0f} MB/s')
    print(f'  camera can sustain {cam.resulting_frame_rate:.2f} Hz as configured')
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
    return {'roi': roi, 'binning': binning_info, 'timestamps': list(stamps),
            'pixel_format': pixel_format, 'binning_mode': cam.binning_mode,
            'saturation_level': int(cam.saturation_level),
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
        'effective_pixel_size_mm': camera_info['pixel_size_mm'] * BINNING,
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


def capture(serial_number=None, output_root=OUTPUT_ROOT,
            locate=True, prompt=True, make=None):
    """Locate the mode, configure, wait for the scope, record, save."""
    camera_cls, serial_number, make = resolve_camera(make, serial_number)
    cam = camera_cls(serial_number)
    cam.open()
    try:
        mode_location = None
        offset_y = roi_height = None
        if not locate:
            offset_y, roi_height = fallback_roi()
        if locate:
            print('--- locating the mode (whole sensor) ---')
            mode_location = locate_mode(cam)
            report_mode_location(mode_location)
            apply_camera_basics(cam)
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
def capture_synchronized(serial_number=None, output_root=OUTPUT_ROOT,
                         locate=True, n_frames=None, scope_serial=None,
                         adjust_gain=True, require_level=True, make=None):
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
    camera_cls, serial_number, make = resolve_camera(make, serial_number)
    cam = camera_cls(serial_number)
    cam.open()
    scope = PicoScope4000A(scope_serial)
    try:
        mode_location = None
        offset_y = roi_height = None
        if not locate:
            offset_y, roi_height = fallback_roi()
        if locate:
            print('--- locating the mode (whole sensor) ---')
            mode_location = locate_mode(cam)
            report_mode_location(mode_location)
            apply_camera_basics(cam)
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
    bias = host_t0_bias_s(make)
    t0_host = t0_host_raw + bias
    print(f'\n  {frames.shape} {frames.dtype}, dropped {timing["n_dropped"]}')
    print(f'  frame period {timing["period_s_median"] * 1e3:.4f} ms '
          f'+- {timing["period_s_std"] * 1e3:.4f} ms')
    print(f'  scope {block_info["n_collected"]} samples at '
          f'{block_info["interval_s"] * 1e9:.0f} ns, overflow '
          f'{block_info["overflow_channels"] or "none"}')
    if HOST_T0_BIAS_S.get(make) is None:
        print(f'  t0 from the host clocks: {t0_host_raw * 1e3:.2f} ms, with no '
              f'calibration - the {make} arming delay has not been measured')
        print(f'  ! run mode_video_sync.py --refine on this capture. For the '
              f'{make} the nominal offset is not yet good to a frame, because '
              f'nobody has measured how long it takes to arm.')
    else:
        print(f'  t0 from the host clocks: {t0_host_raw * 1e3:.2f} ms raw, '
              f'{t0_host * 1e3:.2f} ms after the {bias * 1e3:+.1f} ms '
              f'{make} calibration')

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
        'host_t0_bias_s': bias,
        'host_t0_bias_calibrated': HOST_T0_BIAS_S.get(make) is not None,
        'camera_make': make,
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

    pixel_size = 5.5 / 1000.0
    fake_camera_info = {'serial_number': 'x', 'make': 'basler',
                        'pixel_size_mm': pixel_size}
    with tempfile.TemporaryDirectory() as folder:
        path, mask = save_session(folder, 'test', frames, meta, timing,
                                  {'burst_s': 0.12}, fake_camera_info, None)
        session, loaded = load_session(path)
        assert loaded.shape == frames.shape, loaded.shape
        assert np.array_equal(np.asarray(loaded), frames)
        assert session['frames_dtype'] == 'uint8'
        assert len(session['meta']) == n
        assert len(session['brightness_full']) == n
        assert len(session['brightness_masked']) == n
        assert session['mask_pixels'] == int(mask.sum()) > 0
        assert session['effective_pixel_size_mm'] == pixel_size * BINNING
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
                               numpy_checks, fake_camera_info, None)
        stored = json.loads(path.read_text())['checks']
        assert stored == {'peak_max': 3, 'worst': 4.0, 'covers': True,
                          'peaks': [0, 1, 2]}, stored

    # and when the metadata cannot be written, nothing large is left behind
    with tempfile.TemporaryDirectory() as folder:
        try:
            save_session(folder, 'bad', frames, meta, timing,
                         {'unserialisable': object()}, fake_camera_info,
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
        """A camera whose rate is paced per row, as the Basler's is.

        `seconds_per_pixel` makes it charge for columns too, which is how the
        width-narrowing path gets exercised without a camera that needs it.
        """
        max_frame_size = (1024, 1024)
        pixel_format = 'Mono12'
        formats = ('Mono8', 'Mono12')
        exposure_us = EXPOSURE_US

        def __init__(self, seconds_per_row=1.29e-5, seconds_per_pixel=0.0):
            self.seconds_per_row = seconds_per_row
            self.seconds_per_pixel = seconds_per_pixel
            self.height = self.width = None
            self.offset_y = self.offset_x = None

        def set_roi(self, width, height, offset_x, offset_y):
            self.height, self.offset_y = height, offset_y
            self.width, self.offset_x = width, offset_x
            return {'width': width, 'height': height,
                    'offset_x': offset_x, 'offset_y': offset_y}

        frame_rate_hz = property(lambda self: 0.0, lambda self, value: None)

        @property
        def resulting_frame_rate(self):
            seconds = (self.height * self.seconds_per_row
                       + self.height * self.width * self.seconds_per_pixel)
            return 1.0 / seconds

    def _found(row_min, row_max, col_min=300, col_max=700):
        return {'row_min': row_min, 'row_max': row_max,
                'col_min': col_min, 'col_max': col_max,
                'centre_row': (row_min + row_max) // 2,
                'centre_col': (col_min + col_max) // 2,
                'height': row_max - row_min + 1,
                'width': col_max - col_min + 1}

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
    # a camera that charges for columns narrows the width rather than
    # giving up the frame rate, and says that is what it did
    # 4e-8 s/pixel is chosen so that the covering height makes 100 Hz at half
    # the width and misses it at full width - the case the narrowing exists for
    wide = choose_roi(_FakeCam(seconds_per_row=1.29e-5, seconds_per_pixel=4e-8),
                      _found(480, 560, col_min=450, col_max=560),
                      target_hz=100.0)
    assert wide['width'] < 1024, wide
    assert wide['covers'] and wide['resulting_hz'] >= 98.0, wide
    assert 'columns as well as rows' in (wide['note'] or ''), wide
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

    # the exposure follows the frame rate and always leaves room to read out
    period_us = 1e6 / FRAME_RATE_HZ
    assert 0 < EXPOSURE_US < period_us, (EXPOSURE_US, period_us)
    assert period_us - EXPOSURE_US >= 100.0, 'no room between frames'
    print(f'  {FRAME_RATE_HZ:g} Hz -> exposure {EXPOSURE_US:.0f} us, '
          f'{EXPOSURE_GAP_US:.0f} us of gap, derived not typed')

    # the ROI is measured at every run, not remembered from whenever this file
    # was written; the only fallback is a previous capture
    assert (ROI_OFFSET_Y is None) == (ROI_HEIGHT is None), 'pin both or neither'
    with tempfile.TemporaryDirectory() as empty:
        assert previous_roi(empty) is None
        older = Path(empty) / 'a'
        older.mkdir()
        (older / 'a_session.json').write_text(
            json.dumps({'checks': {'roi': {'offset_y': 111, 'height': 256}}}),
            encoding='utf-8')
        newer = Path(empty) / 'b'
        newer.mkdir()
        (newer / 'b_session.json').write_text(
            json.dumps({'checks': {'roi': {'offset_y': 222, 'height': 320}}}),
            encoding='utf-8')
        offset_y, height, path = previous_roi(empty)
        assert (offset_y, height) == (222, 320), (offset_y, height)
        assert path.parent.name == 'b'
    print('  with no ROI in the file, --no-locate falls back on the last '
          'capture rather than on a number from 2026-08-26')

    try:
        configure(None, None, None)
    except ValueError as error:
        assert 'measured ROI' in str(error), error
    else:
        raise AssertionError('configure accepted a ROI it was never given')

    # Both makes must satisfy the shared contract. Checked on the classes,
    # so it needs no camera - only the SDK, and a make whose SDK is absent is
    # skipped rather than failing a machine that will never use it.
    from camera_core import check_camera_surface
    checked = []
    for make in CAMERA_BACKENDS:
        try:
            cls = camera_class(make)
        except Exception as error:
            print(f'  {make}: SDK not installed here ({type(error).__name__})')
            continue
        check_camera_surface(cls)
        checked.append(make)
    assert checked, 'no camera SDK is installed, so nothing could be checked'
    print(f'  {" and ".join(checked)} satisfy the shared camera surface, so '
          f'this script never asks which one it is holding')

    # a bias is per make and never borrowed from the other camera
    assert set(HOST_T0_BIAS_S) == set(CAMERA_BACKENDS), HOST_T0_BIAS_S
    assert host_t0_bias_s('nonexistent-make') == 0.0
    for make, bias in HOST_T0_BIAS_S.items():
        assert bias is None or 0.0 <= bias < 1.0, (make, bias)
    print('  the host-clock bias is per camera; an unmeasured one stays 0 and '
          "says so rather than borrowing the other camera's")

    # the run-button configuration has to name something this file can do
    assert ACTION in ('capture', 'levels', 'locate', 'self-test'), ACTION
    assert CAMERA is None or CAMERA in CAMERA_BACKENDS, CAMERA
    assert LEVEL_BURST_FRAMES is None or LEVEL_BURST_FRAMES > 0
    print('self-test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--self-test', action='store_true',
                        help='run the offline checks and exit')
    parser.add_argument('--locate', action='store_true',
                        help='find the mode and report, without capturing')
    parser.add_argument('--camera', default=None,
                        choices=sorted(CAMERA_BACKENDS),
                        help='camera make; defaults to CAMERA in this file, '
                             'or the only camera connected')
    parser.add_argument('--serial', default=None,
                        help='camera serial number; defaults to SERIAL_NUMBER '
                             'in this file, or the only camera connected')
    parser.add_argument('--frames', type=int, default=None,
                        help=f'frames to record (default {N_FRAMES})')
    parser.add_argument('--no-prompt', action='store_true',
                        help='record immediately instead of waiting for Enter; '
                             'the PicoScope recording must already be running '
                             'and long enough to cover the delay')
    parser.add_argument('--no-locate', action='store_true',
                        help='skip the reconnaissance and reuse the ROI '
                             'of the last capture')
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

    # No arguments: do what the block at the top of the file says.
    action = ACTION
    if args.self_test:
        action = 'self-test'
    elif args.levels:
        action = 'levels'
    elif args.locate:
        action = 'locate'
    locate = LOCATE_FIRST and not args.no_locate
    allow_saturated = ALLOW_SATURATED or args.allow_saturated
    if args.frames:
        globals()['N_FRAMES'] = args.frames

    if action == 'self-test':
        _self_test()
        return

    camera_cls, serial, make = resolve_camera(args.camera, args.serial)

    if action == 'locate':
        cam = camera_cls(serial)
        cam.open()
        try:
            print('--- locating the mode (whole sensor) ---')
            report_mode_location(locate_mode(cam))
        finally:
            cam.set_binning(1)
            cam.set_roi_full()
            cam.close()
        return

    if action == 'levels':
        cam = camera_cls(serial)
        cam.open()
        try:
            offset_y = roi_height = None
            if not locate:
                offset_y, roi_height = fallback_roi()
            if locate:
                found = locate_mode(cam)
                report_mode_location(found)
                apply_camera_basics(cam)
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

    if action != 'capture':
        raise SystemExit(f'ACTION must be capture, levels, locate or '
                         f'self-test, not {ACTION!r}')

    if DRIVE_SCOPE or args.scope:
        capture_synchronized(serial, locate=locate, make=make,
                             require_level=not allow_saturated)
    else:
        capture(serial, locate=locate, make=make, prompt=not args.no_prompt)

if __name__ == '__main__':
    main()
