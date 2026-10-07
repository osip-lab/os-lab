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

## Alongside kalishlot

If the kalishlot web GUI is running and holds the camera (or the scope, when
this script drives it), the script borrows them for the run: kalishlot closes
them, their boxes show "on loan", and they are handed back - re-opened with
their settings - when the script ends, also on an error or Ctrl+C. A script
killed outright cannot hand them back; press "reconnect" in the box. See
kalishlot/loan_client.py.

With --from-kalishlot - which is how the camera box's "mode video" button runs
the pipeline - the boxes also lend their settings: the ROI, exposure, gain and
frame rate from the camera box, the channel ranges, couplings and sample rate
from the PicoScope box when one is open. Only what kalishlot cannot set (the
capture duration, binning, ...) comes from the config - see
adopt_kalishlot_settings(). When kalishlot also has the function generator
open, its channels (waveform, frequency, amplitude, offset, on/off) are read -
not borrowed, it keeps scanning - and recorded as 'function_generator' in the
session JSON and in run_config_resolved.json, so the capture says how fast the
laser was being scanned. With no generator box, nothing is added.

## What comes out

A session folder holding

    <stem>_frames.mp4    the frame stack - .mkv when FRAMES_FORMAT is
                         'lossless'; older captures hold a raw .npy
    <stem>_mask.npy      the pixels the mode actually lit
    <stem>_session.json  camera settings, per-frame timing, brightness series

The scope side is `<stem>_scope.npz` when this script drives the scope (a
Phase 1 capture kept the `.psdata` exported from PicoScope 7 instead); the
spectrum scripts read either - see pico_scope/scope_trace.py.
"""

import argparse
import importlib
import json
import os
import sys
import time
import warnings
import xml.etree.ElementTree as ElementTree
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_core import burst_timing  # noqa: E402
from pico_scope import run_config  # noqa: E402
from pico_scope.frame_codec import FORMATS as FRAMES_FORMATS, save_frames, save_preview  # noqa: E402
from pico_scope.mode_video_sync import (SESSION_ROOT,  # noqa: E402
                                        frame_brightness, varying_pixel_mask)
from kalishlot.loan_client import (borrow_from_kalishlot, device_command,  # noqa: E402
                                   open_devices)

# The camera makes this script can drive. Imported one at a time and only when
# needed: a machine with just one SDK installed must still run, and importing
# an absent SDK at module scope would take the whole file down with it.
CAMERA_BACKENDS = {
    'basler': ('basler_cam', 'basler_cameras', 'BaslerCamera'),
    'ximea': ('ximea_cam', 'ximea_cameras', 'XimeaCamera'),
}
# The same makes as kalishlot device types, for borrowing them from it.
KALISHLOT_CAMERA_TYPES = {'basler': 'basler_camera', 'ximea': 'ximea_camera'}

# --- the run parameters, and where they really come from -------------------
# Everything from here to the end of the calibration block is a *default*.
# The values a run actually uses come from pico_scope/run_config_local.py,
# which is git-ignored and created from run_config_local_template.py on first
# use - see run_config.py. The declarations stay here because they carry the
# reasoning for each number, and because the self-tests must run with no config
# file at all. run_config.apply() overwrites them at the end of this block.
ACTION = 'capture'      # 'capture' | 'levels' | 'locate' | 'self-test'
CAMERA = None           # 'basler' | 'ximea' | None = the only one connected
DRIVE_SCOPE = True      # False: you record the scope yourself in PicoScope 7
LOCATE_FIRST = True     # locate the mode first; False reuses the last ROI
                        # (both are ignored while MANUAL_ROI is set)
STRICT_LEVELS = False   # True: refuse to capture when the light clips.
                        # Off by default: clipping is monotone, so it flattens
                        # the peaks without moving them, and the alignment fit
                        # (a centred, normalised inner product) is unchanged by
                        # it. What saturation really costs is the *image* - the
                        # lobes merge into one blob - so it is reported loudly
                        # and left to you to judge.

# --- the camera and how it is driven (this is the block to edit) -----------
# None means whichever camera is connected, of either make, which is right
# whenever there is only one - the serial that used to sit here belonged to a
# camera that is not always the one plugged in. Name a serial only to pick
# between cameras that are both connected; resolve_camera() then says which.
SERIAL_NUMBER = None
FRAME_RATE_HZ = 100           # see the peak-blending check below
# None: the exposure follows the frame rate rather than being typed out beside
# it - as long as the period allows, less the gap the sensor needs between
# frames. Asking for the whole period does not fail loudly, it quietly lowers
# the rate, and the inverse-minus-a-bit had to be recomputed by hand at every
# new rate. Deriving it matters more now that the rate is set from a config
# file: a typed exposure left over from another rate would silently cap it.
# derive_exposure() below turns None into the number; a value pins it instead.
EXPOSURE_US = None              # 9900 us at 100 Hz
EXPOSURE_GAP_US = None          # the gap that leaves; derived alongside
EXPOSURE_DERIVED = False        # whether EXPOSURE_US was derived, not typed
# How long the camera records, in seconds. The number of frames follows from
# it and the frame rate the camera actually reached (see frames_for_duration),
# rounded to the nearest whole frame - so a rate changed in the config, or
# taken from kalishlot's box, keeps the same measurement length instead of
# silently stretching or shrinking it.
CAPTURE_DURATION_S = 1.2        # 120 frames at 100 Hz
N_FRAMES = None                 # derived by configure(); not a setting
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

# --- the ROI by hand, typed straight from the camera GUI ------------------
# The four numbers exactly as the ROI dialog shows them - xiCamTool on the
# XIMEA, pylon Viewer on the Basler - in SENSOR pixels, which is the unit both
# dialogs report. manual_roi() converts them to the binned pixels the camera
# wrappers take: offsets round down, sizes round up, so the ROI applied is
# never smaller than the box that was drawn there (it can be a few pixels
# larger; the numbers actually applied are printed at every run).
#
# None gives the original behaviour back, and is the normal way to run:
# LOCATE_FIRST = True measures where the mode is now, False reuses the ROI of
# the last capture. A dict here overrides both, and the reconnaissance is
# skipped. Typed numbers are right only until the cavity is realigned or the
# camera nudged, and then wrong silently - the capture still runs, on rows the
# mode has left - so set this back to None when the comparison it was pinned
# for is done. This is the setting the config file exists for: it belongs to a
# day's alignment, not to the repository.
MANUAL_ROI = 'xicamtool'
# MANUAL_ROI = 'xicamtool' takes whatever ROI was last set in xiCamTool - see
# xicamtool_roi(). The four numbers typed here are the ones that dialog shows,
# so reading them from the file it already writes saves transcribing them.
# When kalishlot is running and holds the XIMEA, the ROI of its camera box is
# taken instead - that is the one being looked at - see kalishlot_roi().

# ROI in BINNED pixels, for the runs that size it themselves; MANUAL_ROI wins
# over both when it is set. None: the full sensor width. On the Basler width is
# free - readout is paced per row - so the budget is spent on rows; choose_roi
# narrows the width only if a camera turns out to charge for columns too.
ROI_WIDTH = None
ROI_HEIGHT_CANDIDATES = (128, 192, 256, 320, 384, 448, 512, 640, 768, 1024)
# The higher orders are larger than the 0th and are the ones that must not be
# clipped, so the margin around what was actually seen is at least as wide as
# the mode itself, and never less than this.
ROI_MIN_MARGIN_ROWS = 48
ROI_OFFSET_X = 0

# --- the scope, when this script drives it too (Phase 2) -------------------
# Only one program can own the scope, so PicoScope 7 must be closed. The block
# is made just long enough to contain the burst plus the few tens of
# milliseconds it takes to get from RunBlock to the first exposure: every extra
# second of slack would add another ~4 free-spectral-range aliases for the
# optional fine alignment to sort out.
SCOPE_CHANNEL = 'D'             # cavity transmission, as everywhere else
SCOPE_RANGE_V = None            # None: auto-range from a short probe instead
                                # (see auto_range_scope) - useful when the
                                # transmission level is not known ahead of time
SCOPE_COUPLING = 'DC'
SCOPE_SAMPLE_INTERVAL_S = 1e-5  # 100 kS/s, the rate the lab already uses
SCOPE_PAD_S = 0.30              # recorded before and after the burst
SCOPE_AUTORANGE_PROBE_S = 0.2   # seconds sampled to auto-range, when
                                # SCOPE_RANGE_V is None
SCOPE_AUTORANGE_MARGIN = 1.5    # target range = this x the probe's largest
                                # magnitude
SCOPE_AUTORANGE_MIN_V = 0.02    # floor, so a probe that caught no signal (a
                                # blocked beam, say) does not pick the most
                                # sensitive range available

# A second channel, recorded alongside the transmission and shown with it in
# the viewer: the ramp driving the laser temperature, which is what says which
# way the scan is going at any instant. It is never fitted against - the
# alignment uses the transmission alone - so a channel that turns out to be
# unconnected costs a flat line and nothing else. None switches it off, and
# captures made without it load and plot exactly as they did.
SCOPE_AUX_CHANNEL = 'B'
SCOPE_AUX_LABEL = 'Temperature modulation Voltage'
SCOPE_AUX_RANGE_V = 5.0         # it swings about 5 Vpp; the scope snaps this
                                # to the nearest range that still contains it
SCOPE_AUX_COUPLING = 'DC'       # the level matters, not just the swing

# More scope data after the video is over, as a second block saved beside the
# first (<stamp>_scope_tail.npz) - the main trace and the sync are untouched.
# None = no tail. With TRAILING_SCOPE_AUX_FG the function generator's channel
# TRAILING_FG_CHANNEL is switched on (through kalishlot) for the tail and back
# to what it was afterwards. Only the full video + scope capture has a tail.
TRAILING_SCOPE_S = None
TRAILING_SCOPE_AUX_FG = False
TRAILING_FG_CHANNEL = 2

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
# ximea: measured over 26 captures on 2026-09-01, of which 11 gave a fit that
# locked, -145.1 ms with a standard deviation of 2.3 ms - a 14.5-frame bias
# with 0.23 frames of jitter. Negative because this camera arms far faster than
# the host round-trip that estimates t0, where the Basler arms more slowly.
# Only locked fits (depth > 1.5) were averaged: the laser was drifting through
# resonances thermally rather than being scanned, so two bursts in three saw no
# resonance at all and returned a meaningless offset. That the 11 that did lock
# agree to a couple of ms, across bursts whose resonances fell at unrelated
# times, is what rules out a common alias.
#
# None means not yet measured. The capture still runs and still records the raw
# host clock; it just says so, and that --refine is not optional for it.
HOST_T0_BIAS_S = {'basler': 0.0399, 'ximea': -0.1451}

# --- what the capture is checked against -----------------------------------
MASK_THRESHOLD = 0.15           # fraction of the peak-to-peak that counts as lit

# How the frame stack is stored. 'h264' is 8-bit and lossy (about 200x smaller
# than raw; a 12-bit stack is rescaled to 0-255, ~8 counts rms of 3010 lost),
# 'lossless' is FFV1, bit-exact and about 3x smaller. See frame_codec.py.
FRAMES_FORMAT = 'h264'
# H.264 quality (FRAMES_FORMAT 'h264' only): 0 is lossless, 51 the harshest.
# 18 is visually transparent; lower is bigger, and past ~10 gains nothing,
# since the 8-bit step dominates. Recorded in the session as frames_crf.
H264_CRF = 18

# Text added after the timestamp in the session folder's name, e.g. 'no_EOM'
# gives 2026-10-05_112839_no_EOM. The files inside keep the bare timestamp.
FOLDER_SUFFIX = ''

# The cavity's long arm [cm] for this capture, set from the kalishlot box's
# start-of-capture pop-up. Recorded in the session (as long_arm_m) for the
# analysis scripts to read instead of asking again; it is never itself used by
# the capture. None = not given.
LONG_ARM_CM = None

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

# --- where captures are written --------------------------------------------
# None: the local bank mode_video_sync uses, so that leaving its SESSION empty
# finds the capture this script just wrote. A path overrides it. Resolved
# through output_root() rather than read directly, so that a config file
# leaving it None still lands in the shared bank.
OUTPUT_ROOT = None

# Prompt for the Dropbox measurement folder to save each capture into,
# instead of the fixed local OUTPUT_ROOT above - data is identified by its
# Dropbox path elsewhere in the lab, not by a local timestamp bank. Set False
# to go back to saving under OUTPUT_ROOT with no prompt (e.g. quick local
# testing); leaving --session-style auto-discovery under OUTPUT_ROOT working
# only when this is False.
PROMPT_FOR_OUTPUT_ROOT = True

# --- the config file replaces the defaults above ---------------------------
# Before any def below, because several of them bind a constant as a default
# argument (measure_light_level's n_bursts=LEVEL_BURSTS), and a default
# argument is fixed when the def runs, not when it is called. Applying the
# config afterwards would leave those bound to the values this file ships with
# while every other use saw the config's - the kind of split that would show up
# as one setting mysteriously not taking effect.
CONFIG_CHANGES = run_config.apply('capture', globals())


def derive_exposure():
    """Turn EXPOSURE_US = None into the exposure the frame rate allows.

    The whole period less the gap the sensor needs between frames - 1% of the
    period, floored at 100 us so the gap does not vanish at high rates, where
    1% of a short period is less than the sensor wants. Asking for the whole
    period does not fail loudly, it quietly lowers the frame rate.

    Derived after the config is applied rather than beside FRAME_RATE_HZ, so
    that a config raising the rate raises the exposure with it. A config that
    names an exposure keeps it, and only the gap is worked back out.
    """
    global EXPOSURE_US, EXPOSURE_GAP_US, EXPOSURE_DERIVED
    EXPOSURE_DERIVED = EXPOSURE_US is None
    if EXPOSURE_US is None:
        EXPOSURE_GAP_US = max(100.0, 0.01 * 1e6 / FRAME_RATE_HZ)
        EXPOSURE_US = 1e6 / FRAME_RATE_HZ - EXPOSURE_GAP_US
    else:
        EXPOSURE_GAP_US = 1e6 / FRAME_RATE_HZ - EXPOSURE_US
    return EXPOSURE_US


derive_exposure()


def folder_name(stamp):
    """The session folder's name: the timestamp, then FOLDER_SUFFIX if any."""
    suffix = (FOLDER_SUFFIX or '').strip()
    return f'{stamp}_{suffix}' if suffix else stamp


def default_output_root():
    """Where captures go when nothing is prompted for: OUTPUT_ROOT if it names
    somewhere, else the local bank mode_video_sync searches.

    Not called `output_root`: both capture functions take an argument of that
    name, which would shadow this and resolve to None - and they only reach the
    fallback *after* the burst has been recorded, so the failure would cost the
    capture rather than the run.
    """
    return Path(OUTPUT_ROOT) if OUTPUT_ROOT else SESSION_ROOT


def prompt_for_output_root():
    """Where to save this capture's session.

    The Dropbox measurement folder it belongs with, not the local scratch
    bank, so the capture is identified by its Dropbox path like everything
    else about the measurement.
    """
    # imported here: utilities.utils pulls in matplotlib and scipy (over a
    # second at startup) for this one prompt, which most runs never reach
    from utilities.utils import wait_for_path_from_clipboard
    return Path(wait_for_path_from_clipboard(
        filetype='folder',
        instructions_message='Copy the path of the measurement folder to '
                             'save this capture into...'))


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


def choose_roi(cam, found, target_hz=None, candidates=ROI_HEIGHT_CANDIDATES,
               full_width=None):
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

    `full_width` is the widest ROI to consider, in binned pixels; None takes
    it from ROI_WIDTH, or the sensor. It is a parameter so that the self-test
    can size its own synthetic sensor without a pinned ROI_WIDTH changing what
    is being tested.

    Returns a dict with `height`, `width`, `offset_y`, `offset_x`, `covers`
    and `resulting_hz`.
    """
    target_hz = FRAME_RATE_HZ if target_hz is None else target_hz
    max_width, max_height = cam.max_frame_size
    margin = max(found['height'], ROI_MIN_MARGIN_ROWS)
    needed = found['height'] + 2 * margin
    full_width = min(roi_width_for(cam) if full_width is None else full_width,
                     max_width)

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
# Its constants are in the block at the top of the file with the rest of the
# run parameters: measure_light_level() binds LEVEL_BURSTS as a default
# argument, which is fixed when the def runs, so they must be settled before
# any def in this file - and the config is applied up there.


def measure_light_level(cam, n_bursts=LEVEL_BURSTS, n_frames=None):
    """Peak and saturation statistics over several independent bursts.

    Returns the per-burst peaks along with the summary the verdict uses. The
    figure that matters is the *worst* burst, not the average one: the capture
    only has to clip once to be spoiled.
    """
    n_frames = (n_frames or LEVEL_BURST_FRAMES or N_FRAMES
                or frames_for_duration(cam.resulting_frame_rate))
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


def capture_light_level(frames, saturation, gain_db):
    """The light-level record, measured on the captured burst itself.

    For a run whose pre-flight check was skipped (see capture_synchronized):
    the same peak and saturation figures check_light_level() reports, from
    the frames that were actually kept, with a warning printed if they clip.
    """
    peak = int(frames.max())
    saturated = float((frames >= saturation).mean())
    ok = saturated <= MAX_SATURATED_FRACTION and peak < saturation
    advice = None
    if not ok:
        advice = (f'the capture peaked at {peak} of {saturation} with '
                  f'{saturated:.3%} of pixels saturated: the brightest frames '
                  f'are clipped. The timing fit is unaffected; the images of '
                  f'the mode are not - attenuate the light, or lower the '
                  f"gain in kalishlot's box, for the next capture.")
    elif peak / saturation < LEVEL_TOO_DIM_FRACTION:
        advice = (f'usable, but dim: the capture peaked at '
                  f'{peak / saturation:.1%} of full scale.')
    print(f'  light: peak {peak} of {saturation} ({peak / saturation:.1%}), '
          f'saturated {saturated:.4%}' + (f'\n  ! {advice}' if advice else ''))
    return {'measured_on': 'capture', 'gain_db': gain_db,
            'saturation_level': saturation, 'peak_max': peak,
            'peak_fraction': peak / saturation, 'saturated_fraction': saturated,
            'n_bursts': 1, 'n_frames': int(len(frames)),
            'ok': ok, 'advice': advice}


class _start_in_background:
    """Run `function` on a thread now; wait() joins it and re-raises what
    it raised, so a failure surfaces where the result is first needed."""

    def __init__(self, function):
        import threading
        self._error = None

        def run():
            try:
                function()
            except BaseException as error:      # handed to wait()
                self._error = error
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def wait(self, raise_error=True):
        self._thread.join()
        if raise_error and self._error is not None:
            raise self._error


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
    if make and serial:
        # Fully named - by the config, or by kalishlot, which says which
        # camera it lent: nothing to choose, so no enumeration (a second or
        # more per call). A camera that is not there fails at open(), which
        # lists the ones that are.
        print(f'  camera {serial} ({make})')
        return camera_class(make), str(serial), make

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



# xiCamTool writes one of these per camera serial when it closes, and it is
# where the four numbers that used to be transcribed into MANUAL_ROI by hand
# already live. Host-side, so reading it needs no camera: it works with the
# camera powered off and cannot contend with whatever else has the device open.
XICAMTOOL = 'xicamtool'          # the MANUAL_ROI sentinel that asks for it
XICAMTOOL_PARAMVAL = Path(os.environ.get('APPDATA', '')) / 'xiCamTool' / 'paramval'


def xicamtool_roi(serial, directory=None):
    """The ROI xiCamTool last had for this camera, in SENSOR pixels.

    Returned in the same shape and the same units as a hand-typed MANUAL_ROI,
    so it goes on through manual_roi() unchanged.

    The file is per serial, and there is usually more than one - this machine
    has a file for a camera last opened months ago - so it is chosen by the
    serial of the camera actually resolved, never by which file is newest.

    Only the ROI is taken. That file also records the exposure, the frame rate
    and the gain, all of which this script sets deliberately: the exposure is
    derived from the frame rate, and the gain is pinned at 0 because measuring
    showed it only adds noise. Reading them back would quietly undo both.

    The numbers are in xiAPI's downsampled coordinates, which is sensor pixels
    only while downsampling is 1. It always has been here, but multiplying is
    what makes that an assumption the code states rather than one it relies on.
    """
    directory = XICAMTOOL_PARAMVAL if directory is None else Path(directory)
    path = Path(directory) / f'camera_values_{serial}.xml'
    if not path.is_file():
        raise FileNotFoundError(
            f"MANUAL_ROI = {XICAMTOOL!r} reads the ROI xiCamTool saved for "
            f"camera {serial}, but {path} does not exist. Open that camera in "
            f"xiCamTool once and close it again, or set MANUAL_ROI to None to "
            f"locate the mode instead.")

    values = ElementTree.parse(path).getroot().find('Values')
    if values is None:
        raise ValueError(f'{path} has no <Values> block; xiCamTool may have '
                         f'been interrupted while writing it')

    def number(tag):
        node = values.find(tag)
        if node is None or not (node.text or '').strip():
            raise ValueError(
                f'{path} records no {tag}, so the ROI it holds is incomplete. '
                f'Set the ROI in xiCamTool and close it, or set MANUAL_ROI to '
                f'None.')
        return int(node.text)

    scale = number('downsampling') if values.find('downsampling') is not None else 1
    roi = {'offset_x': number('offsetX') * scale,
           'offset_y': number('offsetY') * scale,
           'width': number('width') * scale,
           'height': number('height') * scale}
    saved = datetime.fromtimestamp(path.stat().st_mtime)
    print(f'  ROI from xiCamTool ({serial}), saved '
          f'{saved.strftime("%Y-%m-%d %H:%M")}: sensor '
          f'{roi["width"]}x{roi["height"]} at ({roi["offset_x"]}, '
          f'{roi["offset_y"]})')
    return roi


def kalishlot_roi(describe):
    """The ROI of a kalishlot camera box, in SENSOR pixels, shaped like a
    hand-typed MANUAL_ROI. `describe` is the box's device as kalishlot lists
    it; no ROI there means the box shows the whole sensor, and so is this.

    kalishlot's XIMEA runs unbinned, so its ROI is already in sensor pixels.
    """
    roi = describe.get('roi')
    if roi is None:
        height, width = describe['sensor_full']
        roi = {'x': 0, 'y': 0, 'width': width, 'height': height}
    spec = {'offset_x': int(roi['x']), 'offset_y': int(roi['y']),
            'width': int(roi['width']), 'height': int(roi['height'])}
    print(f"  ROI from kalishlot's {describe['device_id']} box: sensor "
          f"{spec['width']}x{spec['height']} at ({spec['offset_x']}, "
          f"{spec['offset_y']})"
          + ('' if describe.get('roi') else ' (not cropped there)'))
    return spec


def resolve_manual_roi(serial=None, make=None, kalishlot=None):
    """Turn MANUAL_ROI = 'xicamtool' into the four numbers it stands for.

    From kalishlot rather than from xiCamTool's file when kalishlot was
    holding this camera: `kalishlot` is what borrow_from_kalishlot() lent,
    {device_id: its describe()} as it was just before the loan. The box being
    watched is where the ROI was last drawn; the file is only as fresh as the
    last time xiCamTool was closed.

    Done once, as soon as the camera is known, and written back into the
    module: every later manual_roi() call then sees an ordinary typed ROI, and
    the numbers actually used are what run_config_resolved.json records beside
    the capture rather than the sentinel that asked for them.
    """
    global MANUAL_ROI
    if MANUAL_ROI != XICAMTOOL:
        return MANUAL_ROI
    if make is not None and make != 'ximea':
        raise RuntimeError(
            f'MANUAL_ROI = {XICAMTOOL!r} reads a file xiCamTool writes, so it '
            f'is for the XIMEA; this run is on the {make}. pylon Viewer keeps '
            f'no equivalent - its settings are saved by hand, to .pfs files - '
            f'so type the ROI into MANUAL_ROI or set it to None.')
    if not serial:
        raise RuntimeError(f'MANUAL_ROI = {XICAMTOOL!r} needs the serial of '
                           f'the camera to know which file to read')
    held = (kalishlot or {}).get(f"{KALISHLOT_CAMERA_TYPES['ximea']}:{serial}")
    MANUAL_ROI = kalishlot_roi(held) if held else xicamtool_roi(serial)
    return MANUAL_ROI


# Set by kalishlot's "mode video" button (through run_mode_video_pipeline.py
# --from-kalishlot), or by --from-kalishlot here: the run then takes every
# setting kalishlot's boxes can set from the boxes - see
# adopt_kalishlot_settings() - and only the rest from the config.
FROM_KALISHLOT_ENV = 'MODE_VIDEO_FROM_KALISHLOT'


def from_kalishlot():
    return os.environ.get(FROM_KALISHLOT_ENV) == '1'


def gain_from_kalishlot():
    """True when the gain is the box's, so the capture must not trim it."""
    return from_kalishlot() and run_config.adopt_flags()['gain']


def _box_setting(describe, name, zero_is_unset=True):
    """A setting's value as a kalishlot box shows it, or None when the box
    has none (the Basler has no frame rate) or has not read it yet - which it
    shows as 0, except where 0 is a real value (a gain)."""
    for setting in describe.get('settings') or []:
        value = setting.get('value')
        if setting.get('name') == name and value is not None                 and (value or not zero_is_unset):
            return float(value)
    return None


# The function generator's state as kalishlot's box showed it when the run
# started, or None - no kalishlot, no generator box, or not --from-kalishlot.
# Read, never borrowed: the generator has to keep scanning during the capture.
KALISHLOT_FUNCTION_GENERATOR = None


def read_kalishlot_function_generator():
    """The open function-generator box's channels, for the capture record.

    It is how fast and how far the laser was being scanned, which the
    spectrum's time axis means nothing without. Read through kalishlot's
    device list rather than lent: describe() asks the instrument itself, so
    a change made on the front panel is in it too. None when kalishlot is not
    running or has no generator open - the record is then left as it was.
    """
    generator = next((device for device in open_devices() or []
                      if device.get('type') == 'rigol_dg'), None)
    if generator is None:
        return None
    channels = [dict(state, channel=number) for number, state
                in enumerate(generator.get('channels') or [], start=1)]
    for state in channels:
        print(f"  function generator CH{state['channel']}: "
              f"{'on' if state.get('on') else 'off'}, {state.get('waveform')}, "
              f"{state.get('frequency_hz', 0):g} Hz, "
              f"{state.get('amplitude_vpp', 0):g} Vpp, "
              f"offset {state.get('offset_v', 0):g} V")
    return {'device_id': generator.get('device_id'),
            'label': generator.get('label'),
            'read_at': datetime.now().isoformat(timespec='seconds'),
            'channels': channels}


def adopt_kalishlot_settings(kalishlot, serial, make):
    """Take the run's settings from kalishlot's boxes, as they were just
    before the loan; whatever a box cannot set stays the config's.

    `kalishlot` is what borrow_from_kalishlot() lent, as in
    resolve_manual_roi(). From the camera box: the ROI, the exposure, the
    gain and - on the XIMEA, the only one with a rate in its box - the frame
    rate. From the PicoScope box, when one is open: the range and coupling of
    the transmission and aux channels, and the sample rate. Binning, pixel
    format, the capture duration, which channel is which and the pads are not
    in kalishlot, so they stay the config's.

    The ROI replaces MANUAL_ROI and the reconnaissance alike: the box is where
    the mode was just being looked at. The scope range replaces auto-ranging
    for the same reason - it is the one known not to clip.

    Which of those the capture takes is `run_config.adopt_flags()`: roi,
    exposure, gain, frame_rate and scope_range each default to taken, and a
    run parameter file (kalishlot's synced-pipeline box) can turn any off. Off
    means the capture's own logic decides instead - the ROI is located, the
    exposure derived from the frame rate, the gain trimmed by the light-level
    check, the scope range found by auto-ranging - from the config's values.
    Coupling, the aux channel and the sample rate have no such logic, so they
    always come from the boxes.

    Written back into the module, like MANUAL_ROI, so run_config_resolved.json
    records the values actually used.
    """
    global MANUAL_ROI, EXPOSURE_US, GAIN_DB, FRAME_RATE_HZ
    global SCOPE_RANGE_V, SCOPE_COUPLING, SCOPE_AUX_RANGE_V, SCOPE_AUX_COUPLING
    global SCOPE_SAMPLE_INTERVAL_S
    kalishlot = kalishlot or {}
    adopt = run_config.adopt_flags()
    print('--- settings from kalishlot ---')

    camera = kalishlot.get(f'{KALISHLOT_CAMERA_TYPES[make]}:{serial}') \
        if make else None
    if make is None:
        pass                        # a scope-only run: no camera to adopt from
    elif camera is None:
        print(f'  ! kalishlot holds no {make} {serial}; the camera settings '
              f"are the config's")
    else:
        if adopt['roi']:
            MANUAL_ROI = kalishlot_roi(camera)
        exposure = _box_setting(camera, 'exposure')
        gain = _box_setting(camera, 'gain', zero_is_unset=False)
        rate = _box_setting(camera, 'framerate')
        if rate is not None and adopt['frame_rate']:
            FRAME_RATE_HZ = rate
        # None (not read yet) derives it from the rate in force, rather than
        # keeping one derived for the config's rate, which may not fit
        if adopt['exposure']:
            EXPOSURE_US = exposure
        elif EXPOSURE_DERIVED:
            EXPOSURE_US = None      # derived for another rate; derive afresh
        derive_exposure()
        if gain is not None and adopt['gain']:
            GAIN_DB = gain
        taken = [name for name in ('roi', 'exposure', 'gain', 'frame_rate')
                 if adopt[name]]
        print(f'  exposure {EXPOSURE_US:.0f} us, gain {GAIN_DB:.1f} dB, frame '
              f'rate {FRAME_RATE_HZ:g} Hz'
              + ('' if rate is not None or not adopt['frame_rate']
                 else " (the config's - the box has no frame rate)"))
        print(f'  taken from the box: {", ".join(taken) or "nothing"}; the '
              f"rest is the capture's own")

    scope = next((device for device in kalishlot.values()
                  if device.get('type') == 'picoscope'), None)
    if scope is None:
        print("  no PicoScope box open; the scope settings are the config's")
        return
    rate = _box_setting(scope, 'sample_rate_hz')
    if rate is not None:
        SCOPE_SAMPLE_INTERVAL_S = 1.0 / rate
        print(f'  scope {rate:g} S/s')
    channels = scope.get('channels') or {}
    for role, channel in (('signal', SCOPE_CHANNEL), ('aux', SCOPE_AUX_CHANNEL)):
        config = channels.get(channel) if channel else None
        if not config or not config.get('enabled'):
            if channel:
                print(f"  scope channel {channel}: off in kalishlot's box, "
                      f"keeping the config's range and coupling")
            continue
        if role == 'signal':
            if adopt['scope_range']:
                SCOPE_RANGE_V = config['range_v']
            SCOPE_COUPLING = config['coupling']
        else:
            SCOPE_AUX_RANGE_V, SCOPE_AUX_COUPLING = (config['range_v'],
                                                     config['coupling'])
        print(f"  scope channel {channel}: +-{config['range_v']:g} V "
              f"{config['coupling']}")


_TAKE_FROM_FILE = object()   # so that manual_roi(None) can mean 'none typed'


def manual_roi(spec=_TAKE_FROM_FILE):
    """MANUAL_ROI in BINNED pixels, or None when the block is not in use.

    The block at the top is typed in sensor pixels, as the camera GUIs report
    them, and both wrappers take binned ones - so every number is divided by
    BINNING here. Offsets round down and sizes up to a multiple of 4: the
    wrappers snap *down* to whatever increment the camera enforces (4 and 2 on
    these two), so a size rounded up survives that snap while one rounded down
    would lose another row. The ROI applied can therefore be a few sensor
    pixels larger than the one typed, and is never smaller.

    Raises rather than guessing if a key is missing or a size is not positive:
    a half-written ROI would otherwise capture the wrong rows in silence.
    """
    spec = MANUAL_ROI if spec is _TAKE_FROM_FILE else spec
    if spec is None:
        return None
    keys = ('offset_x', 'offset_y', 'width', 'height')
    if set(spec) != set(keys):
        raise ValueError(f'MANUAL_ROI takes exactly {keys} in sensor pixels, '
                         f'as the camera GUI shows them; got {sorted(spec)}')
    if min(int(spec[key]) for key in keys) < 0:
        raise ValueError(f'MANUAL_ROI cannot be negative: {spec}')
    if int(spec['width']) <= 0 or int(spec['height']) <= 0:
        raise ValueError(f'MANUAL_ROI needs a positive width and height: {spec}')
    roi = {}
    for offset_key, size_key in (('offset_x', 'width'), ('offset_y', 'height')):
        offset, size = int(spec[offset_key]), int(spec[size_key])
        start = offset // BINNING
        end = -(-(offset + size) // BINNING)          # ceil, to cover the box
        roi[offset_key] = start
        roi[size_key] = -(-(end - start) // 4) * 4    # ceil to the increment
    return roi


def roi_width_for(cam):
    """ROI width in binned pixels: manual, pinned, or the whole sensor."""
    manual = manual_roi()
    if manual is not None:
        return manual['width']
    return ROI_WIDTH or cam.max_frame_size[0]


def roi_offset_x_for():
    """ROI x offset in binned pixels: manual if it is set, else ROI_OFFSET_X."""
    manual = manual_roi()
    return ROI_OFFSET_X if manual is None else manual['offset_x']


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
    root = default_output_root() if root is None else Path(root)
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
    """The ROI for a run that skips the reconnaissance and has none typed.

    The last capture's, which was at least measured at some point - never a
    number left over from whenever this file happened to be written. An ROI
    typed into MANUAL_ROI is handled before this, in resolve_roi().
    """
    found = previous_roi()
    if found is None:
        raise RuntimeError(
            'no ROI to fall back on: the mode has not been located and no '
            'previous capture is on disk. Locate it first - LOCATE_FIRST = '
            'True, or drop --no-locate - which is the normal way round; or '
            'type one into MANUAL_ROI.')
    offset_y, height, path = found
    print(f'  reusing the ROI measured for {path.parent.name}: {height} rows '
          f'at offset_y {offset_y}')
    return offset_y, height


def resolve_roi(cam, locate):
    """Where the ROI for this run comes from, in binned pixels.

    Three sources, in this order: an ROI typed into MANUAL_ROI, the
    reconnaissance, or the ROI of the last capture. Returns
    (offset_y, roi_height, mode_location, roi_choice); the last two are None
    unless the mode was actually located - nothing downstream may pretend it
    was measured when it was typed - and roi_choice is the choose_roi() dict,
    with the margin and the rate it settled for, which the session records.
    """
    manual = manual_roi()
    if manual is not None:
        print(f"  ROI typed into MANUAL_ROI: {manual['width']}x"
              f"{manual['height']} binned at offset "
              f"({manual['offset_x']}, {manual['offset_y']}) = sensor "
              f"{manual['width'] * BINNING}x{manual['height'] * BINNING} at "
              f"({manual['offset_x'] * BINNING}, "
              f"{manual['offset_y'] * BINNING}); the mode is not located")
        return manual['offset_y'], manual['height'], None, None
    if not locate:
        offset_y, roi_height = fallback_roi()
        return offset_y, roi_height, None, None
    print('--- locating the mode (whole sensor) ---')
    mode_location = locate_mode(cam)
    report_mode_location(mode_location)
    apply_camera_basics(cam)
    choice = choose_roi(cam, mode_location)
    return choice['offset_y'], choice['height'], mode_location, choice


# %% [Step 2] Configuring the camera ----------------------------------------
def frames_for_duration(rate_hz, duration_s=None):
    """The frames CAPTURE_DURATION_S takes at `rate_hz`, rounded to the
    nearest whole frame (halves up), and never fewer than one."""
    duration_s = CAPTURE_DURATION_S if duration_s is None else duration_s
    return max(1, int(np.floor(duration_s * rate_hz + 0.5)))


def configure(cam, offset_y, roi_height):
    """Apply the capture settings and print every check worth failing on."""
    if offset_y is None or roi_height is None:
        raise ValueError('configure() needs a measured ROI: locate the mode '
                         'first, or take one from fallback_roi().')
    pixel_format, binning_info = apply_camera_basics(cam)
    roi = cam.set_roi(roi_width_for(cam), roi_height,
                      roi_offset_x_for(), offset_y)
    cam.exposure_us = EXPOSURE_US
    cam.gain_db = GAIN_DB
    cam.frame_rate_hz = FRAME_RATE_HZ
    stamps = cam.enable_timestamps()

    sensor_w = roi['width'] * BINNING
    sensor_h = roi['height'] * BINNING
    link_max = cam.max_frame_rate_for(sensor_w, sensor_h, pixel_format, BINNING)

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
    try:
        actual_hz = cam.assert_frame_rate_reachable(FRAME_RATE_HZ)
    except RuntimeError as exc:
        # Falling back rather than refusing: a session at the achievable rate
        # is worth having, and the warning is loud enough that it won't be
        # mistaken for the rate that was actually asked for.
        actual_hz = cam.resulting_frame_rate
        message = (f'FRAME_RATE_HZ={FRAME_RATE_HZ:g} is not reachable - the '
                   f'script dropped it to {actual_hz:.1f} Hz automatically. '
                   f'{exc}')
        warnings.warn(message)
        print(f'  ! {message}')

    # the frame count is settled here, from the rate the camera actually
    # runs at, and written back so every later step records that number
    global N_FRAMES
    N_FRAMES = frames_for_duration(actual_hz)
    period = 1.0 / actual_hz
    burst = N_FRAMES * period

    print(f'  burst {burst:.3f} s ({CAPTURE_DURATION_S:g} s asked) = '
          f'{N_FRAMES} frames at {actual_hz:.1f} Hz, '
          f'{N_FRAMES * roi["width"] * roi["height"] / 1e6:.0f} MB')
    return {'roi': roi, 'binning': binning_info, 'timestamps': list(stamps),
            'pixel_format': pixel_format, 'binning_mode': cam.binning_mode,
            'saturation_level': int(cam.saturation_level),
            'link_max_hz': link_max, 'resulting_hz': actual_hz,
            'burst_s': burst}


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
    if FRAMES_FORMAT not in FRAMES_FORMATS:
        raise ValueError(f'FRAMES_FORMAT {FRAMES_FORMAT!r} is not one of '
                         f'{FRAMES_FORMATS}')
    mask_name = f'{stem}_mask.npy'

    session = {
        'created': datetime.now().isoformat(timespec='seconds'),
        # [m], from the kalishlot box's start-of-capture pop-up; None when it
        # was skipped. Lets the analysis scripts (extract_df_and_fsr_from_
        # scope_csv.py, mode_map_2d.py) take the cavity's long arm from here
        # instead of prompting for it again.
        'long_arm_m': LONG_ARM_CM / 100 if LONG_ARM_CM is not None else None,
        'sync': {
            'method': 'optical',
            'description': 'frame brightness fitted against the scope trace; '
                           'see pico_scope/mode_video_sync.py',
            'signal_column': 'Channel D',
        },
        'frames_file': None,        # filled in once the file is written
        'frames_format': FRAMES_FORMAT,
        'frames_scale': None,
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
    if KALISHLOT_FUNCTION_GENERATOR is not None:
        session['function_generator'] = KALISHLOT_FUNCTION_GENERATOR
    # Serialise before writing anything large. The metadata is the fragile
    # part - it is assembled from a dozen measurements, any of which can carry
    # a type json refuses - and it is also the irreplaceable part, since the
    # per-frame timestamps and the offset exist nowhere else. Twice now a
    # capture has written 94 MB of frames and then failed here, leaving arrays
    # nothing could interpret. Failing before the arrays exist costs a rerun;
    # failing after costs the data.
    text = json.dumps(session, indent=1, default=_json_default)
    fps = timing.get('period_s_median')
    frames_path, info = save_frames(folder / f'{stem}_frames', frames,
                                    FRAMES_FORMAT, 1 / fps if fps else 30.0,
                                    H264_CRF)
    if FRAMES_FORMAT == 'h264':
        session['frames_crf'] = H264_CRF
    session['frames_file'] = frames_path.name
    session['frames_scale'] = info['scale']
    text = json.dumps(session, indent=1, default=_json_default)
    np.save(folder / mask_name, mask)
    session_path = folder / f'{stem}_session.json'
    session_path.write_text(text, encoding='utf-8')

    # The run parameters, beside the data they produced. They used to be in
    # git, so a commit could say what a measurement was taken with; they live
    # in a git-ignored config file now, and this is what replaces that record -
    # per capture rather than per commit, and holding what was resolved rather
    # than only what was typed. Both capture paths come through here, so a
    # capture run straight from this script is recorded the same as one the
    # pipeline drove.
    extra = ({'function_generator': KALISHLOT_FUNCTION_GENERATOR}
             if KALISHLOT_FUNCTION_GENERATOR is not None else None)
    run_config.dump_into(folder,
                         resolved=run_config.resolved_values('capture', globals()),
                         extra=extra)
    return session_path, mask


def capture(serial_number=None, output_root=None,
            locate=True, prompt=True, make=None, preview=False):
    """Locate the mode, configure, wait for the scope, record, save.

    `preview` also writes <stamp>_preview.mp4 beside the frames: a stretched
    8-bit copy at the capture's own frame rate, for watching in a media player
    (the frames file is raw sensor counts, which a player shows as black)."""
    camera_cls, serial_number, make = resolve_camera(make, serial_number)
    resolve_manual_roi(serial_number, make)
    cam = camera_cls(serial_number)
    cam.open()
    try:
        offset_y, roi_height, mode_location, _ = resolve_roi(cam, locate)

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

        root = output_root if output_root is not None else (
            prompt_for_output_root() if PROMPT_FOR_OUTPUT_ROOT else default_output_root())
        stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        folder = Path(root) / folder_name(stamp)
        session_path, mask = save_session(
            folder, stamp, frames, meta, timing, checks, cam.describe(),
            mode_location)
        print(f'\n  mask covers {int(mask.sum())} of {mask.size} pixels '
              f'({mask.mean():.2%})')
        print(f'  saved {session_path}')
        if preview:
            period = timing['period_s_median']
            preview_path = save_preview(
                folder / f'{stamp}_preview.mp4', frames,
                1 / period if period else checks['resulting_hz'])
            print(f'  saved {preview_path} (stretched, plays in real time)')
        if prompt:
            print(f'\nNow stop and save the PicoScope recording as .psdata, '
                  f'then:')
            print(f'  python pico_scope/mode_video_sync.py --session '
                  f'"{folder}" --scope "<that file>.psdata"')
        print(f'SESSION_PATH={session_path}')
        return session_path
    finally:
        cam.close()


def auto_range_scope(scope, channel, coupling, probe_s=SCOPE_AUTORANGE_PROBE_S,
                     margin=SCOPE_AUTORANGE_MARGIN, min_v=SCOPE_AUTORANGE_MIN_V):
    """Probe `channel` briefly and return `margin` times the largest
    magnitude seen - the range to ask configure_channel() for next, which
    then snaps it up to the nearest one the hardware actually offers.

    Used when SCOPE_RANGE_V is None: the transmission level depends on the
    day's alignment and gain, so a fixed guess either clips the peaks (too
    narrow) or wastes most of the ADC's resolution (too wide). The probe
    itself is taken at the widest available range so it cannot clip.
    """
    from pico_scope.ps4000a_scope import CHANNEL_NAMES, RANGES

    scope.configure_channel(channel, enabled=True, coupling=coupling,
                            range_v=max(RANGES.values()))
    for name in CHANNEL_NAMES:
        if name != channel:
            scope.configure_channel(name, enabled=False)
    scope.configure_trigger(enabled=False)
    _t, volts, _info = scope.capture_block(probe_s, SCOPE_SAMPLE_INTERVAL_S)
    peak_v = float(np.max(np.abs(volts[channel])))
    range_v = max(margin * peak_v, min_v)
    print(f'  probed +-{peak_v * 1e3:.1f} mV, asking for +-{range_v * 1e3:.1f} '
         f'mV ({margin:g}x)')
    return range_v


# %% [Step 3b] Driving both instruments (Phase 2) -----------------------------
def configure_scope_channels(scope):
    """Set up the transmission channel (auto-ranged when SCOPE_RANGE_V is
    None), the aux channel if any, switch the others off and the trigger off.
    Returns (range_v as the hardware snapped it, aux channel or '', the aux
    channel's configuration or None)."""
    if SCOPE_RANGE_V is None:
        print(f'  auto-ranging channel {SCOPE_CHANNEL} ...')
        range_v = auto_range_scope(scope, SCOPE_CHANNEL, SCOPE_COUPLING)
    else:
        range_v = SCOPE_RANGE_V
    channel_config = scope.configure_channel(
        SCOPE_CHANNEL, enabled=True, coupling=SCOPE_COUPLING,
        range_v=range_v)
    range_v = channel_config['range_v']  # snapped to what the hardware offers
    aux_channel = SCOPE_AUX_CHANNEL
    if aux_channel == SCOPE_CHANNEL:
        raise ValueError(
            f'SCOPE_AUX_CHANNEL is {aux_channel!r}, the same channel as '
            f'SCOPE_CHANNEL - the transmission and the temperature ramp '
            f'are two different signals and need two channels')
    aux_config = None
    if aux_channel:
        aux_config = scope.configure_channel(
            aux_channel, enabled=True, coupling=SCOPE_AUX_COUPLING,
            range_v=SCOPE_AUX_RANGE_V)
        print(f'  channel {aux_channel}, +-{aux_config["range_v"]:g} V '
              f'{SCOPE_AUX_COUPLING}, {SCOPE_AUX_LABEL}')
    for name in ('A', 'B', 'C', 'D'):
        if name not in (SCOPE_CHANNEL, aux_channel):
            scope.configure_channel(name, enabled=False)
    scope.configure_trigger(enabled=False)   # start immediately
    return range_v, aux_channel, aux_config


def record_scope_tail(scope, duration_s):
    """Record `duration_s` more of the scope, right after the main block, with
    the function generator's channel TRAILING_FG_CHANNEL on for it when
    TRAILING_SCOPE_AUX_FG. Returns (t, volts, block_info, fg), `fg` being what
    was done to the generator (None when nothing was).

    The block starts first and the channel is switched on just after, so the
    trace shows the moment it came on (`fg['on_after_start_s']`). The channel
    goes back to the state it was found in, even if the recording fails."""
    generator = fg = None
    if TRAILING_SCOPE_AUX_FG:
        generator = next((device for device in open_devices() or []
                          if device.get('type') == 'rigol_dg'), None)
        if generator is None:
            print('  ! no function generator open in kalishlot: the tail is '
                  'recorded without switching its channel on')
    try:
        block = scope.start_block(duration_s, SCOPE_SAMPLE_INTERVAL_S)
        if generator is not None:
            channels = generator.get('channels') or []
            was_on = bool(channels[TRAILING_FG_CHANNEL - 1].get('on')) \
                if len(channels) >= TRAILING_FG_CHANNEL else False
            fg = {'device_id': generator['device_id'],
                  'channel': TRAILING_FG_CHANNEL, 'was_on': was_on}
            device_command(generator['device_id'], 'set_channel',
                           {'channel': TRAILING_FG_CHANNEL, 'name': 'output',
                            'value': True})
            fg['on_after_start_s'] = time.time() - block['host_start_s']
            print(f'  function generator CH{TRAILING_FG_CHANNEL} on, '
                  f'{fg["on_after_start_s"] * 1e3:.0f} ms into the tail')
        scope.wait_block(timeout_s=duration_s * 3 + 10)
        t, volts, info = scope.read_block()
    finally:
        if fg is not None and not fg['was_on']:
            try:
                device_command(fg['device_id'], 'set_channel',
                               {'channel': fg['channel'], 'name': 'output',
                                'value': False})
                print(f'  function generator CH{fg["channel"]} off again')
            except Exception as error:
                print(f'  ! could not switch CH{fg["channel"]} off again '
                      f'({error}) - do it in the generator box')
    return t, volts, info, fg


def capture_synchronized(serial_number=None, output_root=None,
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

    setup_start = time.time()
    camera_cls, serial_number, make = resolve_camera(make, serial_number)
    resolve_manual_roi(serial_number, make)
    # The scope takes a couple of seconds to open and needs nothing from the
    # camera, so it opens in the background while the camera is set up. It
    # is only touched from here on once that has finished (join below).
    scope = PicoScope4000A(scope_serial)
    scope_opening = _start_in_background(scope.open)
    cam = None
    try:
        cam = camera_cls(serial_number)
        cam.open()
        offset_y, roi_height, mode_location, roi_choice = resolve_roi(
            cam, locate)
        checks = configure(cam, offset_y, roi_height)
        if roi_choice is not None:
            checks['roi_choice'] = roi_choice
        if n_frames is None:
            n_frames = N_FRAMES       # from CAPTURE_DURATION_S, by configure()
        burst_s = n_frames / checks['resulting_hz']

        # The pre-flight bursts exist to set the gain, or to refuse a capture
        # that would clip. From kalishlot neither may happen - the gain is the
        # box's, and STRICT_LEVELS is off - so they would only print advice,
        # at the cost of LEVEL_BURSTS bursts as long as the capture itself.
        # The same numbers are measured on the captured frames instead (see
        # capture_light_level), which is free and describes the real data.
        # Run any other way, the check runs exactly as before.
        pre_check = not (from_kalishlot() and not adjust_gain
                         and not require_level)
        level = None
        if pre_check:
            print('\n--- light level ---')
            level = check_light_level(cam, adjust_gain=adjust_gain)
            checks['light_level'] = level
        else:
            print('\n--- light level: measured on the capture itself '
                  '(settings from kalishlot) ---')
        if level is not None and not level['ok']:
            if require_level:
                raise RuntimeError('too bright to capture: ' + level['advice'])
            print(f'  ! {level["advice"]}')
            # Not a reason to stop: clipping flattens the peaks without moving
            # them, and the fit maximises a centred, normalised inner product,
            # which that leaves alone. The stored frames are what suffer.
            print('  ! capturing anyway. The timing fit is unaffected by '
                  'clipping; the images are, so the lobes may be merged.')

        scope_opening.wait()           # raises here if the open failed
        print(f'\n--- scope {scope.variant} s/n {scope.serial} ---')
        range_v, aux_channel, aux_config = configure_scope_channels(scope)
        duration = burst_s + 2 * SCOPE_PAD_S
        print(f'  channel {SCOPE_CHANNEL}, +-{range_v * 1e3:g} mV '
              f'{SCOPE_COUPLING}, {SCOPE_SAMPLE_INTERVAL_S * 1e6:g} us/sample')
        print(f'  block {duration:.3f} s = {burst_s:.3f} s burst + '
              f'2 x {SCOPE_PAD_S:.2f} s pad')

        saturation = cam.saturation_level
        gain_db = cam.gain_db
        print(f'  setup took {time.time() - setup_start:.1f} s')
        block = scope.start_block(duration, SCOPE_SAMPLE_INTERVAL_S)
        host_scope_start = block['host_start_s']
        print(f'  recording; starting the burst ...')
        host_before_burst = time.time()
        frames, meta = cam.record_burst(n_frames)
        host_after_burst = time.time()
        scope.wait_block(timeout_s=duration * 3 + 10)
        t_scope, volts, block_info = scope.read_block()
        tail = None
        if TRAILING_SCOPE_S:
            print(f'\n--- trailing scope capture, {TRAILING_SCOPE_S:g} s ---')
            try:
                tail = record_scope_tail(scope, TRAILING_SCOPE_S)
            except Exception as error:      # the video is worth more than it
                print(f'  ! the trailing capture failed ({error}); '
                      f'saving the video without it')
        # Read the camera's settings while it is still open - everything below
        # happens after the finally clause has closed it.
        camera_info = cam.describe()
        scope_info = {'serial': scope.serial, 'variant': scope.variant}
    finally:
        # the background open may still be running if the camera failed
        # first; let it finish so the scope is closed rather than left open
        scope_opening.wait(raise_error=False)
        scope.close()
        if cam is not None:
            cam.close()

    if not pre_check:
        checks['light_level'] = capture_light_level(frames, saturation, gain_db)

    timing = burst_timing(meta, expected_rate_hz=checks['resulting_hz'])
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

    root = output_root if output_root is not None else (
        prompt_for_output_root() if PROMPT_FOR_OUTPUT_ROOT else default_output_root())
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    folder = Path(root) / folder_name(stamp)
    session_path, mask = save_session(
        folder, stamp, frames, meta, timing, checks, camera_info,
        mode_location)

    # the scope trace lives with the capture, so no .psdata is needed. The aux
    # array is simply absent when no aux channel was recorded, which is what
    # every earlier capture looks like and what the loader already expects.
    signal = volts[SCOPE_CHANNEL]
    arrays = {'t': t_scope, 'signal': signal}
    if aux_config is not None and aux_channel in volts:
        arrays['aux'] = volts[aux_channel]
    np.savez_compressed(folder / f'{stamp}_scope.npz', **arrays)
    session = json.loads(session_path.read_text(encoding='utf-8'))
    if tail is not None:
        # t in the tail file starts at 0 at its own first sample; the main
        # trace's t = 0 is `start_offset_s` earlier than that, by the host clock
        tail_t, tail_volts, tail_info, tail_fg = tail
        tail_arrays = {'t': tail_t, 'signal': tail_volts[SCOPE_CHANNEL]}
        if 'aux' in arrays:
            tail_arrays['aux'] = tail_volts[aux_channel]
        np.savez_compressed(folder / f'{stamp}_scope_tail.npz', **tail_arrays)
        session['scope_tail'] = {
            'file': f'{stamp}_scope_tail.npz',
            'start_offset_s': tail_info['host_start_s'] - host_scope_start,
            'sample_interval_s': tail_info['interval_s'],
            'n_samples': tail_info['n_collected'],
            'duration_s': TRAILING_SCOPE_S,
            'overflow_channels': tail_info['overflow_channels'],
            'function_generator': tail_fg,
        }
    session['scope'] = {
        'file': f'{stamp}_scope.npz',
        'channel': SCOPE_CHANNEL,
        'range_v': range_v,
        'coupling': SCOPE_COUPLING,
        'sample_interval_s': block_info['interval_s'],
        'n_samples': block_info['n_collected'],
        'duration_s': duration,
        'overflow_channels': block_info['overflow_channels'],
        'serial': scope_info['serial'],
        'variant': scope_info['variant'],
    }
    if 'aux' in arrays:
        session['scope']['aux'] = {
            'channel': aux_channel,
            'label': SCOPE_AUX_LABEL,
            'range_v': aux_config['range_v'],
            'coupling': SCOPE_AUX_COUPLING,
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
    print(f'SESSION_PATH={session_path}')
    return session_path


def capture_scope_only(output_root=None, scope_serial=None):
    """Record the scope alone, for CAPTURE_DURATION_S, with no camera and so
    nothing to sync: no padding either, which only exists to give the sync fit
    room. Saved as <stamp>_scope.npz (t, signal and, if recorded, aux) with a
    <stamp>_scope.json describing it - not *_session.json, which is what marks
    a folder as a video capture to the viewer and to mode_video_sync."""
    from pico_scope.ps4000a_scope import PicoScope4000A

    scope = PicoScope4000A(scope_serial)
    scope.open()
    try:
        print(f'\n--- scope {scope.variant} s/n {scope.serial} ---')
        range_v, aux_channel, aux_config = configure_scope_channels(scope)
        print(f'  channel {SCOPE_CHANNEL}, +-{range_v * 1e3:g} mV '
              f'{SCOPE_COUPLING}, {SCOPE_SAMPLE_INTERVAL_S * 1e6:g} us/sample')
        duration = CAPTURE_DURATION_S
        print(f'  recording {duration:.3f} s ...')
        scope.start_block(duration, SCOPE_SAMPLE_INTERVAL_S)
        scope.wait_block(timeout_s=duration * 3 + 10)
        t_scope, volts, block_info = scope.read_block()
        scope_info = {'serial': scope.serial, 'variant': scope.variant}
    finally:
        scope.close()
    print(f'  {block_info["n_collected"]} samples at '
          f'{block_info["interval_s"] * 1e9:.0f} ns, overflow '
          f'{block_info["overflow_channels"] or "none"}')

    root = output_root if output_root is not None else (
        prompt_for_output_root() if PROMPT_FOR_OUTPUT_ROOT else default_output_root())
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    folder = Path(root) / folder_name(stamp)
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {'t': t_scope, 'signal': volts[SCOPE_CHANNEL]}
    if aux_config is not None and aux_channel in volts:
        arrays['aux'] = volts[aux_channel]
    np.savez_compressed(folder / f'{stamp}_scope.npz', **arrays)
    record = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'long_arm_m': LONG_ARM_CM / 100 if LONG_ARM_CM is not None else None,
        'scope': {
            'file': f'{stamp}_scope.npz',
            'channel': SCOPE_CHANNEL,
            'range_v': range_v,
            'coupling': SCOPE_COUPLING,
            'sample_interval_s': block_info['interval_s'],
            'n_samples': block_info['n_collected'],
            'duration_s': duration,
            'overflow_channels': block_info['overflow_channels'],
            **scope_info,
        },
    }
    if 'aux' in arrays:
        record['scope']['aux'] = {
            'channel': aux_channel, 'label': SCOPE_AUX_LABEL,
            'range_v': aux_config['range_v'], 'coupling': SCOPE_AUX_COUPLING}
    if KALISHLOT_FUNCTION_GENERATOR is not None:
        record['function_generator'] = KALISHLOT_FUNCTION_GENERATOR
    record_path = folder / f'{stamp}_scope.json'
    record_path.write_text(json.dumps(record, indent=1, default=_json_default),
                           encoding='utf-8')
    extra = ({'function_generator': KALISHLOT_FUNCTION_GENERATOR}
             if KALISHLOT_FUNCTION_GENERATOR is not None else None)
    run_config.dump_into(folder,
                         resolved=run_config.resolved_values('capture', globals()),
                         extra=extra)
    print(f'  saved {record_path}')
    print(f'SESSION_PATH={record_path}')
    return record_path


# %% [Step 4] Self-test -------------------------------------------------------
def _self_test():
    global MANUAL_ROI, EXPOSURE_US, GAIN_DB, FRAME_RATE_HZ, SCOPE_RANGE_V
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
    global FRAMES_FORMAT
    configured_format, FRAMES_FORMAT = FRAMES_FORMAT, 'h264'  # not the config's
    with tempfile.TemporaryDirectory() as folder:
        path, mask = save_session(folder, 'test', frames, meta, timing,
                                  {'burst_s': 0.12}, fake_camera_info, None)
        session, loaded = load_session(path)
        assert loaded.shape == frames.shape, loaded.shape
        assert session['frames_format'] == 'h264'
        assert loaded.dtype == frames.dtype
        # lossy, but close: well under one grey level per 20 on average
        err = np.abs(np.asarray(loaded).astype(int) - frames)
        assert err.mean() < 8 and err.max() < 60, (err.mean(), err.max())
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

    # lossless is bit-exact, and a 12-bit stack keeps its counts under h264
    chosen = FRAMES_FORMAT
    try:
        FRAMES_FORMAT = 'lossless'
        with tempfile.TemporaryDirectory() as folder:
            path, _ = save_session(folder, 'll', frames, meta, timing,
                                   {'burst_s': 0.12}, fake_camera_info, None)
            session, loaded = load_session(path)
            assert session['frames_file'].endswith('.mkv')
            assert np.array_equal(loaded, frames)
        deep = (frames.astype(np.uint16) * 16)
        FRAMES_FORMAT = 'h264'
        with tempfile.TemporaryDirectory() as folder:
            path, _ = save_session(folder, 'deep', deep, meta, timing,
                                   {'burst_s': 0.12}, fake_camera_info, None)
            session, loaded = load_session(path)
            assert loaded.dtype == np.uint16 and session['frames_scale'] > 1
            assert np.abs(loaded.astype(int) - deep).max() < 16 * 60
    finally:
        FRAMES_FORMAT = configured_format
    print('  lossless is exact; a 16-bit stack is rescaled and restored')

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
        leftovers = list(Path(folder).glob('*_frames.*'))
        assert not leftovers, f'frames written despite failed metadata: {leftovers}'
    print('  numpy metadata survives, and a metadata failure leaves no orphan '
          'arrays')

    # the exposure must fit inside the frame period, or it becomes the cap
    period = 1.0 / FRAME_RATE_HZ
    assert EXPOSURE_US / 1e6 < period, \
        'exposure must be shorter than the frame period, or it becomes the cap'
    print(f'  {FRAME_RATE_HZ:g} Hz -> exposure {EXPOSURE_US / 1e3:.1f} ms < '
          f'period {period * 1e3:.1f} ms')
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
    choice = choose_roi(_FakeCam(), small_mode, target_hz=100.0,
                          full_width=_FakeCam.max_frame_size[0])
    assert choice['covers'], choice
    assert choice['resulting_hz'] >= 98.0, choice
    assert choice['margin_rows'] >= small_mode['height'], choice
    assert choice['note'] is None, choice

    # a larger mode has to be given a taller ROI
    bigger = choose_roi(_FakeCam(), _found(400, 640), target_hz=100.0,
                       full_width=_FakeCam.max_frame_size[0])
    assert bigger['height'] > choice['height'], (choice, bigger)

    # a mode near the top edge is followed rather than centred, and still fits
    edge = choose_roi(_FakeCam(), _found(20, 100), target_hz=100.0,
                     full_width=_FakeCam.max_frame_size[0])
    assert edge['offset_y'] == 0, edge
    assert edge['covers'], edge

    # when margin and frame rate cannot both be had, the rate is kept and the
    # compromise is reported rather than made silently
    tight = choose_roi(_FakeCam(), _found(300, 740), target_hz=100.0,
                      full_width=_FakeCam.max_frame_size[0])
    assert tight['note'] is not None, tight
    assert tight['resulting_hz'] >= 98.0, 'the frame rate is the hard constraint'
    # a camera that charges for columns narrows the width rather than
    # giving up the frame rate, and says that is what it did
    # 4e-8 s/pixel is chosen so that the covering height makes 100 Hz at half
    # the width and misses it at full width - the case the narrowing exists for
    wide = choose_roi(_FakeCam(seconds_per_row=1.29e-5, seconds_per_pixel=4e-8),
                      _found(480, 560, col_min=450, col_max=560),
                      target_hz=100.0,
                      full_width=_FakeCam.max_frame_size[0])
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
          'that no single burst clips at is still flagged when it has no '
          'headroom')

    # the exposure follows the frame rate and always leaves room to read out
    period_us = 1e6 / FRAME_RATE_HZ
    assert 0 < EXPOSURE_US < period_us, (EXPOSURE_US, period_us)
    assert period_us - EXPOSURE_US >= 100.0, 'no room between frames'
    print(f'  {FRAME_RATE_HZ:g} Hz -> exposure {EXPOSURE_US:.0f} us, '
          f'{EXPOSURE_GAP_US:.0f} us of gap, derived not typed')

    # an ROI typed from the camera GUI is in sensor pixels and lands on binned
    # ones that cover it - never a box smaller than the one that was drawn
    assert manual_roi(None) is None, 'MANUAL_ROI = None means locate as usual'
    typed = {'offset_x': 760, 'offset_y': 1208, 'width': 732, 'height': 734}
    binned = manual_roi(typed)
    for offset_key, size_key in (('offset_x', 'width'), ('offset_y', 'height')):
        assert binned[offset_key] * BINNING <= typed[offset_key], binned
        assert ((binned[offset_key] + binned[size_key]) * BINNING
                >= typed[offset_key] + typed[size_key]), binned
        assert binned[size_key] % 4 == 0, binned   # survives the snap down
    # Spelled out from BINNING rather than written as a literal: the binning
    # is a config setting now, and a literal that only held at 2 would fail
    # this test for a run that legitimately uses another one.
    expected = {}
    for offset_key, size_key in (('offset_x', 'width'), ('offset_y', 'height')):
        start = typed[offset_key] // BINNING
        end = -(-(typed[offset_key] + typed[size_key]) // BINNING)
        expected[offset_key] = start
        expected[size_key] = -(-(end - start) // 4) * 4
    assert binned == expected, (binned, expected)
    for bad in ({'offset_x': 0, 'width': 8},                  # half a box
                {'offset_x': 0, 'offset_y': 0, 'width': 0, 'height': 8},
                {'offset_x': -4, 'offset_y': 0, 'width': 8, 'height': 8}):
        try:
            manual_roi(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'MANUAL_ROI {bad} should have been refused')
    print(f'  MANUAL_ROI is typed in sensor pixels as the camera GUI shows '
          f'them: {typed["width"]}x{typed["height"]} at '
          f'({typed["offset_x"]}, {typed["offset_y"]}) -> binned '
          f'{binned["width"]}x{binned["height"]} at ({binned["offset_x"]}, '
          f'{binned["offset_y"]}), rounded out, never in')

    # the ROI xiCamTool saved, read from the file it writes on close
    with tempfile.TemporaryDirectory() as paramval:
        xml = """<CameraParameterValues serial="TEST001">
 <Values>
  <downsampling type="int">1</downsampling>
  <width type="int">492</width>
  <offsetX type="int">560</offsetX>
  <height type="int">544</height>
  <offsetY type="int">272</offsetY>
  <exposure type="float">5779</exposure>
  <gain type="float">0.375</gain>
 </Values>
</CameraParameterValues>"""
        (Path(paramval) / 'camera_values_TEST001.xml').write_text(xml,
                                                                 encoding='utf-8')
        from_tool = xicamtool_roi('TEST001', paramval)
        assert from_tool == {'offset_x': 560, 'offset_y': 272,
                             'width': 492, 'height': 544}, from_tool
        # the same shape a hand-typed ROI has, so it goes on through unchanged
        assert set(from_tool) == {'offset_x', 'offset_y', 'width', 'height'}
        assert manual_roi(from_tool) == manual_roi(
            {'offset_x': 560, 'offset_y': 272, 'width': 492, 'height': 544})

        # downsampling is a multiplier, not a decoration: xiAPI reports the ROI
        # in downsampled pixels and manual_roi() takes sensor ones
        (Path(paramval) / 'camera_values_TEST002.xml').write_text(
            xml.replace('"TEST001"', '"TEST002"')
               .replace('<downsampling type="int">1<', '<downsampling type="int">2<'),
            encoding='utf-8')
        assert xicamtool_roi('TEST002', paramval) == {
            'offset_x': 1120, 'offset_y': 544, 'width': 984, 'height': 1088}

        # a file for another camera is not silently used instead
        try:
            xicamtool_roi('NOSUCH', paramval)
        except FileNotFoundError as error:
            assert 'NOSUCH' in str(error), error
        else:
            raise AssertionError('a missing camera file should have been refused')

        # an incomplete file is refused rather than half-read
        (Path(paramval) / 'camera_values_TEST003.xml').write_text(
            '<CameraParameterValues><Values><width type="int">8</width>'
            '</Values></CameraParameterValues>', encoding='utf-8')
        try:
            xicamtool_roi('TEST003', paramval)
        except ValueError as error:
            assert 'offsetX' in str(error) or 'height' in str(error), error
        else:
            raise AssertionError('a half-written file should have been refused')
    print('  the ROI xiCamTool saved is read from its own per-serial file, in '
          'sensor pixels, and only the ROI - not the exposure or gain it also '
          'holds, which this script sets itself')

    # the sentinel is for the XIMEA; the Basler has no file like it
    saved_roi = MANUAL_ROI
    try:
        globals()['MANUAL_ROI'] = XICAMTOOL
        try:
            resolve_manual_roi('ANY', 'basler')
        except RuntimeError as error:
            assert 'pylon' in str(error), error
        else:
            raise AssertionError('the sentinel should not apply to a Basler')
    finally:
        globals()['MANUAL_ROI'] = saved_roi
    print("  MANUAL_ROI = 'xicamtool' says so plainly on a Basler rather "
          "than quietly locating instead")

    # kalishlot holding the XIMEA: its box's ROI wins over xiCamTool's file
    # (which does not exist for this serial, so reading it would raise)
    held = {'ximea_camera:TEST009': {
        'device_id': 'ximea_camera:TEST009', 'sensor_full': [2048, 2048],
        'roi': {'x': 100, 'y': 640, 'width': 1800, 'height': 360}}}
    try:
        globals()['MANUAL_ROI'] = XICAMTOOL
        assert resolve_manual_roi('TEST009', 'ximea', held) == {
            'offset_x': 100, 'offset_y': 640, 'width': 1800, 'height': 360}
        globals()['MANUAL_ROI'] = XICAMTOOL
        held['ximea_camera:TEST009']['roi'] = None      # uncropped box
        assert resolve_manual_roi('TEST009', 'ximea', held) == {
            'offset_x': 0, 'offset_y': 0, 'width': 2048, 'height': 2048}
    finally:
        globals()['MANUAL_ROI'] = saved_roi
    print("  with kalishlot holding the XIMEA, MANUAL_ROI = 'xicamtool' takes "
          "the ROI of its box instead of xiCamTool's file")

    # with none typed, the ROI is measured at every run and not remembered
    # from whenever this file was written; the only fallback is a past capture
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
    # Either sign: the bias is the gap between where the host thinks frame 0
    # sits and where it is, so a camera that arms faster than the host
    # round-trip that estimates t0 has a negative one. The XIMEA measured
    # -145 ms against the Basler's +40 ms.
    for make, bias in HOST_T0_BIAS_S.items():
        assert bias is None or -1.0 < bias < 1.0, (make, bias)
    print('  the host-clock bias is per camera and of either sign; an '
          "unmeasured one stays 0 and says so rather than borrowing the "
          "other camera's")

    # Which box values are taken is per parameter (the synced-pipeline box's
    # checkboxes): off hands the decision back to the capture's own logic
    import json as _json
    import tempfile as _tempfile
    saved = (MANUAL_ROI, EXPOSURE_US, GAIN_DB, FRAME_RATE_HZ, SCOPE_RANGE_V,
             os.environ.get(run_config.PARAMS_ENV_VAR))
    box = {
        'ximea_camera:T': {
            'type': 'ximea_camera', 'device_id': 'ximea_camera:T',
            'roi': {'x': 100, 'y': 200, 'width': 400, 'height': 300},
            'sensor_full': [2048, 2048],
            'settings': [{'name': 'exposure', 'value': 5000.0},
                         {'name': 'gain', 'value': 6.0},
                         {'name': 'framerate', 'value': 200.0}]},
        'picoscope:S': {
            'type': 'picoscope', 'device_id': 'picoscope:S',
            'settings': [{'name': 'sample_rate_hz', 'value': 100000.0}],
            'channels': {SCOPE_CHANNEL: {'enabled': True, 'range_v': 0.5,
                                         'coupling': 'AC'}}},
    }

    def adopt_with(adopt):
        global MANUAL_ROI, EXPOSURE_US, GAIN_DB, FRAME_RATE_HZ, SCOPE_RANGE_V
        MANUAL_ROI, EXPOSURE_US, GAIN_DB, FRAME_RATE_HZ, SCOPE_RANGE_V = (
            None, None, 1.0, 100.0, None)
        derive_exposure()
        with _tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'params.json'
            path.write_text(_json.dumps({'adopt': adopt}), encoding='utf-8')
            os.environ[run_config.PARAMS_ENV_VAR] = str(path)
            run_config._params_cache = (None, None)
            adopt_kalishlot_settings(box, 'T', 'ximea')
    try:
        adopt_with({})                                    # all taken, as before
        assert MANUAL_ROI is not None and EXPOSURE_US == 5000.0
        assert GAIN_DB == 6.0 and FRAME_RATE_HZ == 200.0 and SCOPE_RANGE_V == 0.5
        adopt_with({'roi': False, 'exposure': False, 'gain': False,
                    'frame_rate': False, 'scope_range': False})
        assert MANUAL_ROI is None, 'roi off: the mode is located instead'
        assert GAIN_DB == 1.0 and FRAME_RATE_HZ == 100.0, 'gain and rate stay the configs'
        assert SCOPE_RANGE_V is None, 'range off: auto-ranged'
        assert abs(EXPOSURE_US - 9900.0) < 1e-6, EXPOSURE_US  # derived at 100 Hz
        adopt_with({'exposure': False, 'frame_rate': True})   # derived at 200 Hz, 100 us gap
        assert FRAME_RATE_HZ == 200.0 and abs(EXPOSURE_US - 4900.0) < 1e-6,             EXPOSURE_US
        assert MANUAL_ROI is not None and GAIN_DB == 6.0
    finally:
        (MANUAL_ROI, EXPOSURE_US, GAIN_DB, FRAME_RATE_HZ, SCOPE_RANGE_V,
         env_value) = saved
        if env_value is None:
            os.environ.pop(run_config.PARAMS_ENV_VAR, None)
        else:
            os.environ[run_config.PARAMS_ENV_VAR] = env_value
        run_config._params_cache = (None, None)
        derive_exposure()
    print('  each box value is taken, or left to the capture own logic')

    # The function generator is read from kalishlot's device list, never lent,
    # and recorded only when a box is open: a run without one writes the
    # same run_config_resolved.json it always did.
    global open_devices
    real_open_devices = open_devices
    try:
        open_devices = lambda: None                   # kalishlot not running
        assert read_kalishlot_function_generator() is None
        open_devices = lambda: [{'device_id': 'ximea_camera:1',
                                 'type': 'ximea_camera'}]
        assert read_kalishlot_function_generator() is None
        open_devices = lambda: [{
            'device_id': 'rigol_dg:USB0::1', 'type': 'rigol_dg',
            'label': 'Rigol DG822',
            'channels': [{'on': True, 'waveform': 'ramp', 'frequency_hz': 2.0,
                          'amplitude_vpp': 4.0, 'offset_v': 0.0},
                         {'on': False, 'waveform': 'sine',
                          'frequency_hz': 1e3, 'amplitude_vpp': 1.0,
                          'offset_v': 0.0}]}]
        generator = read_kalishlot_function_generator()
    finally:
        open_devices = real_open_devices
    assert [c['channel'] for c in generator['channels']] == [1, 2], generator
    assert generator['channels'][0]['frequency_hz'] == 2.0, generator
    with tempfile.TemporaryDirectory() as folder:
        run_config.dump_into(folder, resolved={'CAPTURE_DURATION_S': 1.0})
        plain = json.loads((Path(folder) / 'run_config_resolved.json')
                           .read_text(encoding='utf-8'))
        assert 'function_generator' not in plain, plain
        run_config.dump_into(folder, resolved={'CAPTURE_DURATION_S': 1.0},
                             extra={'function_generator': generator})
        record = json.loads((Path(folder) / 'run_config_resolved.json')
                            .read_text(encoding='utf-8'))
        assert record['function_generator']['channels'][0]['waveform'] == 'ramp'
    print("  the function generator's channels are recorded when kalishlot "
          'has its box open, and nothing changes when it does not')

    # kalishlot names the camera it lent, so no enumeration is needed; with
    # none (or, oddly, two) lent, the camera is found the usual way
    assert lent_camera({'ximea_camera:QX1': {'type': 'ximea_camera'},
                        'picoscope:JO1': {'type': 'picoscope'}}) == ('ximea', 'QX1')
    assert lent_camera({'picoscope:JO1': {'type': 'picoscope'}}) is None
    assert lent_camera(None) is None
    # the light level measured on the capture, for a run that skipped the
    # pre-flight bursts: a clipped burst is flagged, a fine one is not
    clipped = np.zeros((3, 10, 10), dtype=np.uint16)
    clipped[1, :2, :] = 4095                       # 20 of 300 pixels at the rail
    record = capture_light_level(clipped, 4095, 0.0)
    assert not record['ok'] and record['advice'], record
    fine = np.full((3, 10, 10), 2000, dtype=np.uint16)
    assert capture_light_level(fine, 4095, 0.0)['ok']
    # a background start re-raises where it is waited for, not before
    failing = _start_in_background(lambda: 1 / 0)
    try:
        failing.wait()
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError('a failed background open was swallowed')
    failing.wait(raise_error=False)                # the cleanup path: quiet
    print('  a lent camera is used as named, the light level can be read off '
          'the capture, and a background open reports its failure')

    # the duration becomes the nearest whole number of frames at the rate
    # the camera reached, halves rounding up, and never none
    assert frames_for_duration(100.0, 1.2) == 120
    assert frames_for_duration(150.0, 6.667) == 1000        # 1000.05
    assert frames_for_duration(30.0, 0.05) == 2             # 1.5 rounds up
    assert frames_for_duration(97.3, 1.2) == 117            # 116.76
    assert frames_for_duration(10.0, 0.01) == 1
    print('  the capture duration becomes the nearest whole number of frames')

    # the run-button configuration has to name something this file can do
    assert ACTION in ('capture', 'levels', 'locate', 'self-test'), ACTION
    assert CAMERA is None or CAMERA in CAMERA_BACKENDS, CAMERA
    assert LEVEL_BURST_FRAMES is None or LEVEL_BURST_FRAMES > 0
    print('self-test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default=None,
                        help='config file to run from, instead of '
                             'run_config_local.py; read at import, so it is '
                             'already in force by the time this is parsed')
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
    parser.add_argument('--duration', type=float, default=None,
                        help=f'seconds to record (default '
                             f'{CAPTURE_DURATION_S:g}); the number of frames '
                             f'follows from the frame rate')
    parser.add_argument('--no-prompt', action='store_true',
                        help='record immediately instead of waiting for Enter; '
                             'the PicoScope recording must already be running '
                             'and long enough to cover the delay')
    parser.add_argument('--no-locate', action='store_true',
                        help='skip the reconnaissance and reuse the ROI '
                             'of the last capture')
    parser.add_argument('--strict-levels', action='store_true',
                        help='refuse to capture if the camera is clipping, '
                             'instead of only warning')
    parser.add_argument('--levels', action='store_true',
                        help='measure the light level and exit, without '
                             'capturing')
    parser.add_argument('--scope', action='store_true',
                        help='drive the scope from here too, instead of '
                             'recording it by hand in PicoScope 7 (which must '
                             'then be closed - only one program can own it)')
    parser.add_argument('--no-scope', action='store_true',
                        help='do not drive the scope even if DRIVE_SCOPE says '
                             'to: record the video alone')
    parser.add_argument('--scope-only', action='store_true',
                        help='record only the scope, for the capture '
                             'duration, with no camera and no sync')
    parser.add_argument('--from-kalishlot', action='store_true',
                        help='take the ROI, exposure, gain, frame rate and the '
                             'scope settings from the kalishlot boxes, as its '
                             '"mode video" button does')
    args = parser.parse_args()
    if args.from_kalishlot:
        os.environ[FROM_KALISHLOT_ENV] = '1'
    print(run_config.describe('capture', CONFIG_CHANGES))

    # No arguments: do what the config says, which is the block at the top of
    # this file as the config file overrode it.
    action = ACTION
    if args.self_test:
        action = 'self-test'
    elif args.levels:
        action = 'levels'
    elif args.locate:
        action = 'locate'
    locate = LOCATE_FIRST and not args.no_locate
    strict_levels = STRICT_LEVELS or args.strict_levels
    if args.duration:
        globals()['CAPTURE_DURATION_S'] = args.duration  # the command line wins over
                                               # the config, which wins over
                                               # the default declared above

    if action == 'self-test':
        _self_test()
        return
    if args.scope_only:
        with borrow_from_kalishlot(
                kalishlot_wants(drive_scope=True, camera=False),
                borrower='mode_video_capture.py') as lent:
            run_scope_only(lent)
        return
    if action not in ('capture', 'levels', 'locate'):
        raise SystemExit(f'ACTION must be capture, levels, locate or '
                         f'self-test, not {ACTION!r}')

    drive_scope = (action == 'capture' and not args.no_scope
                   and (DRIVE_SCOPE or args.scope))
    with borrow_from_kalishlot(
            kalishlot_wants(args.camera, args.serial, drive_scope),
            borrower='mode_video_capture.py') as lent:
        run_action(action, args, locate, strict_levels, drive_scope, lent)


def lent_camera(lent):
    """(make, serial) of the camera kalishlot lent, or None - when it lent
    none, or (by an odd configuration) more than one."""
    makes = {type_name: make for make, type_name in KALISHLOT_CAMERA_TYPES.items()}
    cameras = [(makes[device.get('type')], device_id.split(':', 1)[1])
               for device_id, device in (lent or {}).items()
               if device.get('type') in makes]
    return cameras[0] if len(cameras) == 1 else None


def kalishlot_wants(make=None, serial=None, drive_scope=False, camera=True):
    """Which of the devices a running kalishlot holds this run needs.

    The camera it will open - the one named, or any of a make it can drive
    when it is to take the only one connected (which kalishlot then holds) -
    and, when it drives the scope itself, the scope. Borrowed before the
    camera is resolved: a camera held elsewhere may not enumerate.
    """
    make = CAMERA if make is None else make
    serial = SERIAL_NUMBER if serial is None else serial
    camera_types = {KALISHLOT_CAMERA_TYPES[name]
                    for name in ([make] if make else KALISHLOT_CAMERA_TYPES)}

    def wants(device):
        type_name = device.get('type')
        if type_name == 'picoscope':
            return drive_scope
        address = device['device_id'].split(':', 1)[1]
        return camera and type_name in camera_types and (
            not serial or address == str(serial))
    return wants


def run_scope_only(lent=None):
    global KALISHLOT_FUNCTION_GENERATOR
    if from_kalishlot():
        adopt_kalishlot_settings(lent, None, None)
        KALISHLOT_FUNCTION_GENERATOR = read_kalishlot_function_generator()
    capture_scope_only()


def run_action(action, args, locate, strict_levels, drive_scope, lent=None):
    global KALISHLOT_FUNCTION_GENERATOR
    make, serial = args.camera, args.serial
    if lent and not (make or serial or CAMERA or SERIAL_NUMBER):
        # kalishlot has just said which camera it lent: use it as named
        held = lent_camera(lent)
        if held is not None:
            make, serial = held
    camera_cls, serial, make = resolve_camera(make, serial)
    if from_kalishlot():
        adopt_kalishlot_settings(lent, serial, make)
        KALISHLOT_FUNCTION_GENERATOR = read_kalishlot_function_generator()
    # As soon as the camera is known, so the 'levels' and 'locate' paths below
    # see the same ROI a capture would. Idempotent: once it has resolved,
    # MANUAL_ROI is an ordinary dict and the capture paths leave it alone.
    resolve_manual_roi(serial, make, lent)

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
            offset_y, roi_height, _, _ = resolve_roi(cam, locate)
            configure(cam, offset_y, roi_height)
            print('\n--- light level ---')
            level = check_light_level(cam, adjust_gain=not gain_from_kalishlot())
            print(f"  {'OK' if level['ok'] else 'TOO BRIGHT'}: "
                  f"{level['advice'] or 'peak is in range'}")
        finally:
            cam.close()
        return

    if drive_scope:
        # the gain set in kalishlot is kept; too much light is then reported
        # rather than trimmed away
        capture_synchronized(serial, locate=locate, make=make,
                             require_level=strict_levels,
                             adjust_gain=not gain_from_kalishlot())
    else:
        capture(serial, locate=locate, make=make, prompt=not args.no_prompt,
                preview=args.no_scope)

if __name__ == '__main__':
    main()
