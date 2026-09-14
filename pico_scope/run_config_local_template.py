"""Every run parameter of the mode-video pipeline, in one place.

COPIED AUTOMATICALLY TO run_config_local.py ON FIRST RUN. Edit that copy, not
this one: it is git-ignored, so changing a frame rate or an ROI no longer means
modifying a tracked script or leaving the working tree dirty.

One class per script, because the names collide on purpose - all four have an
ACTION, three have a SESSION, and they mean different things:

    capture   pico_scope/mode_video_capture.py       record the burst
    sync      pico_scope/mode_video_sync.py          refine the time offset
    show      pico_scope/mode_video_sync_show.py     the viewer
    mark      pico_scope/mode_video_sync_mark.py     the viewer plus annotation

A name a script does not have is an error, not a silent no-op - delete a
setting to fall back to the script's own default, but do not misspell one.

Every capture writes the config it ran under next to its frames
(run_config_used.py and run_config_resolved.json), so the settings stay part of
the record even though this file is outside git.
"""


class capture:
    """pico_scope/mode_video_capture.py"""

    # --- what happens when the script is run ------------------------------
    # The command line overrides these when given; nothing here needs it.
    ACTION = 'capture'      # 'capture' | 'levels' | 'locate' | 'self-test'
    CAMERA = None           # 'basler' | 'ximea' | None = the only one connected
    DRIVE_SCOPE = True      # False: you record the scope yourself in PicoScope 7
    LOCATE_FIRST = True     # locate the mode first; False reuses the last ROI
                            # (both are ignored while MANUAL_ROI is set)
    STRICT_LEVELS = False   # True: refuse to capture when the light clips.
                            # Off by default: clipping is monotone, so it
                            # flattens the peaks without moving them, and the
                            # alignment fit (a centred, normalised inner
                            # product) is unchanged by it. What saturation
                            # really costs is the *image* - the lobes merge
                            # into one blob - so it is reported loudly and left
                            # to you to judge.

    # --- the camera and how it is driven ----------------------------------
    # None means whichever camera is connected, of either make, which is right
    # whenever there is only one. Name a serial only to pick between cameras
    # that are both connected; the script then says which it chose.
    SERIAL_NUMBER = None
    FRAME_RATE_HZ = 100
    # None derives the exposure from the frame rate: the whole period less the
    # gap the sensor needs between frames (1% of the period, floored at 100 us
    # so the gap does not vanish at high rates) - 9900 us at 100 Hz. Asking for
    # the whole period does not fail loudly, it quietly lowers the rate, which
    # is why this is derived rather than typed beside the rate and forgotten at
    # the next change. Put a number here only to pin a shorter exposure.
    EXPOSURE_US = None
    N_FRAMES = 120                  # 1.2 s at 100 Hz
    # None: the deepest format the camera offers - Mono12 on the Basler, Mono10
    # on the XIMEA, whose sensor has no more to give. Depth is wanted for
    # headroom: as the laser warms the transmission climbs, and a clipped peak
    # makes a poor image of the mode. Name a format to force one (Mono8 reads
    # out faster).
    PIXEL_FORMAT = None
    GAIN_DB = 0.0                   # measured: gain only makes the noise worse
    # N x N sum. The Basler does it in firmware, before the link; the XIMEA has
    # no firmware binning at all, so its wrapper sums on the host. Either way
    # the signal goes up by N**2 and the data goes down by it.
    BINNING = 2
    # None: as much of the link as the camera may have. A number caps it, which
    # is only wanted when two cameras share a bus - and this capture drives one.
    THROUGHPUT_BPS = None

    # --- the ROI by hand, typed straight from the camera GUI ---------------
    # The four numbers exactly as the ROI dialog shows them - xiCamTool on the
    # XIMEA, pylon Viewer on the Basler - in SENSOR pixels, which is the unit
    # both dialogs report. They are converted to the binned pixels the camera
    # wrappers take: offsets round down, sizes round up, so the ROI applied is
    # never smaller than the box that was drawn there (it can be a few pixels
    # larger; the numbers actually applied are printed at every run).
    #
    # None is the normal way to run: LOCATE_FIRST = True measures where the
    # mode is now, False reuses the ROI of the last capture. A dict here
    # overrides both, and the reconnaissance is skipped. Typed numbers are
    # right only until the cavity is realigned or the camera nudged, and then
    # wrong silently - the capture still runs, on rows the mode has left - so
    # set this back to None when the comparison it was pinned for is done.
    # 'xicamtool' takes whatever ROI was last set in xiCamTool, read from the
    # per-serial file it writes when it closes - the same four numbers its ROI
    # dialog shows, without transcribing them. XIMEA only: pylon Viewer keeps
    # no equivalent, and says so rather than locating instead.
    MANUAL_ROI = None
    # MANUAL_ROI = 'xicamtool'
    # MANUAL_ROI = {'offset_x': 560, 'offset_y': 272, 'width': 492, 'height': 544}

    # ROI in BINNED pixels, for the runs that size it themselves; MANUAL_ROI
    # wins over both when it is set. None: the full sensor width. On the Basler
    # width is free - readout is paced per row - so the budget is spent on rows.
    ROI_WIDTH = None
    ROI_HEIGHT_CANDIDATES = (128, 192, 256, 320, 384, 448, 512, 640, 768, 1024)
    # The higher orders are larger than the 0th and are the ones that must not
    # be clipped, so the margin around what was actually seen is at least as
    # wide as the mode itself, and never less than this.
    ROI_MIN_MARGIN_ROWS = 48
    ROI_OFFSET_X = 0

    # --- the scope, when the script drives it too (Phase 2) ----------------
    # Only one program can own the scope, so PicoScope 7 must be closed. The
    # block is made just long enough to contain the burst plus the few tens of
    # milliseconds it takes to get from RunBlock to the first exposure: every
    # extra second of slack would add another ~4 free-spectral-range aliases
    # for the optional fine alignment to sort out.
    SCOPE_CHANNEL = 'D'             # cavity transmission, as everywhere else
    SCOPE_RANGE_V = 0.05            # None: auto-range from a short probe
                                    # instead - useful when the transmission
                                    # level is not known ahead of time
    SCOPE_COUPLING = 'DC'
    SCOPE_SAMPLE_INTERVAL_S = 1e-5  # 100 kS/s, the rate the lab already uses
    SCOPE_PAD_S = 0.30              # recorded before and after the burst
    SCOPE_AUTORANGE_PROBE_S = 0.2   # seconds sampled to auto-range, when
                                    # SCOPE_RANGE_V is None
    SCOPE_AUTORANGE_MARGIN = 1.5    # target range = this x the probe's largest
                                    # magnitude
    SCOPE_AUTORANGE_MIN_V = 0.02    # floor, so a probe that caught no signal
                                    # (a blocked beam, say) does not pick the
                                    # most sensitive range available

    # --- what the capture is checked against -------------------------------
    MASK_THRESHOLD = 0.15     # fraction of the peak-to-peak that counts as lit
    # A clipped peak is the one thing that reliably breaks the alignment fit:
    # the camera stops tracking the photodiode exactly where the signal is
    # strongest. Measured on this setup, 1% of samples clipped is survivable
    # and 5% is not, so the gate is set well below that.
    MAX_SATURATED_FRACTION = 0.001   # 0.1% of pixel samples
    TARGET_PEAK_FRACTION = 0.7       # aim the brightest pixel here, of full scale
    # One burst is not enough to judge the level. At a fixed light level the
    # peak varies about 2.3x from burst to burst, because it depends on which
    # resonance that burst happened to catch - measured over 12 bursts on
    # 2026-08-26, peak 1778 to 4095 while the mean stayed within 27-32. A check
    # made from a single burst therefore passes and then lets the real capture
    # clip, which is exactly what happened twice. So several bursts are taken
    # and the verdict is formed from the worst of them.
    LEVEL_BURSTS = 4
    # The pre-flight bursts must be as long as the capture. A shorter one
    # samples fewer free spectral ranges and so has fewer chances to catch a
    # strong resonance, which biases the predicted peak low: measured, 120-frame
    # bursts reach about 15% higher than 40-frame ones at the same light level.
    # None means "same as the capture".
    LEVEL_BURST_FRAMES = None
    LEVEL_SAFETY = 1.3             # margin above the brightest burst yet seen
    LEVEL_TOO_DIM_FRACTION = 0.10  # below this the capture works but wastes range
    LEVEL_CLIPPED_STEP_DB = 6.0    # blind back-off while the peak is censored

    # --- where captures are written ----------------------------------------
    # True prompts for the Dropbox measurement folder to save each capture
    # into - data is identified by its Dropbox path elsewhere in the lab, not
    # by a local timestamp bank. False saves under OUTPUT_ROOT with no prompt
    # (quick local testing), which is also the only case where the --session
    # auto-discovery of the later steps can find the capture on its own.
    PROMPT_FOR_OUTPUT_ROOT = True
    # None: the local bank, PATH_DATA_LOCAL/mode_video from local_config.py,
    # shared with mode_video_sync so an empty SESSION finds the newest capture.
    # A path here overrides it.
    OUTPUT_ROOT = None

    # --- calibration: measured, and rarely touched -------------------------
    # ps4000aRunBlock returns before the scope has actually begun sampling, so
    # the host-clock estimate of where frame 0 sits is systematically early.
    # Part of that delay is the camera's own arming time, so the bias is per
    # make and measured, never borrowed: applying one camera's number to
    # another would misalign every capture by an unknown constant while still
    # claiming sub-frame accuracy, and nothing downstream would show it.
    #
    # basler: measured over 12 captures on 2026-08-26, +39.9 ms with a standard
    # deviation of 7.8 ms - a 4.0-frame bias with 0.78 frames of jitter.
    # Subtracting it puts 83% of captures within one frame with no fitting at
    # all, which is what makes the fine alignment optional.
    #
    # ximea: measured over 26 captures on 2026-09-01, of which 11 gave a fit
    # that locked, -145.1 ms with a standard deviation of 2.3 ms - a 14.5-frame
    # bias with 0.23 frames of jitter. Negative because this camera arms far
    # faster than the host round-trip that estimates t0, where the Basler arms
    # more slowly. Only locked fits (depth > 1.5) were averaged: the laser was
    # drifting through resonances thermally rather than being scanned, so two
    # bursts in three saw no resonance at all and returned a meaningless
    # offset. That the 11 that did lock agree to a couple of ms, across bursts
    # whose resonances fell at unrelated times, is what rules out a common alias.
    #
    # None means not yet measured. The capture still runs and still records the
    # raw host clock; it just says so, and that --refine is not optional for it.
    HOST_T0_BIAS_S = {'basler': 0.0399, 'ximea': -0.1451}


class sync:
    """pico_scope/mode_video_sync.py"""

    ACTION = 'refine'       # 'refine' | 'fit' | 'self-test'
    SESSION = ''            # capture folder; '' means the most recent one
    SCOPE_FILE = ''         # the .psdata of a Phase 1 capture; '' for Phase 2
    SEARCH_WINDOW_S = 0.25  # half-width of the offset search, when refining

    TIME_COLUMN = 'Time'
    SIGNAL_COLUMN = 'Channel D'   # cavity transmission, as in mode_map_2d.py


class show:
    """pico_scope/mode_video_sync_show.py"""

    ACTION = 'show'      # 'show' | 'self-test'
    SESSION = ''         # capture folder; '' means the most recent one
    SCOPE_FILE = ''      # the .psdata of a Phase 1 capture; '' for Phase 2

    SHADE_ALPHA = 0.06   # faint: at 120 frames these are stripes until you zoom
    # Matches kalishlot's CameraFitMixin.FIT_REBINNING, so the browser and this
    # viewer fit the same frame the same way. 4x costs ~140 ms on a 448x1024
    # frame and agrees with 2x to better than 1% on the widths.
    FIT_REBINNING = 4
    # fit_gaussian bounds amplitude and offset at 4095, the 12-bit full scale it
    # was written for. A binned frame can exceed that - 2x2 summing 10-bit XIMEA
    # pixels reaches 4092, but 4x4 would reach 16368 - so a frame that scales
    # past it is divided down before fitting and the amplitude scaled back.
    FIT_MAX_LEVEL = 4095


class mark:
    """pico_scope/mode_video_sync_mark.py"""

    ACTION = 'mark'        # 'mark' | 'self-test'
    # 'clipboard' asks for the folder by clipboard when the script runs;
    # '' means the most recent capture; or put a path here.
    SESSION = 'clipboard'
    SCOPE_FILE = ''        # the .psdata of a Phase 1 capture; '' for Phase 2

    # --- the cavity being measured (edit when the setup changes) -----------
    # Kept per-script rather than shared: different sessions can be measuring
    # different cavities (see mode_analysis.CAVITY_ELEMENTS's docstring).
    CAVITY_ELEMENTS = [
        'LASER_OPTIK_MIRROR',
        'EDMUND_4MM_ASPHERIC_16701',
        'COASTLINE_20CM_MIRROR',
    ]
    SHORT_ARM_LENGTHS = (0.5e-4, 2e-4)  # [m] lens-scan span around collimation
    MID_ARM_LENGTH = 1.5e-2             # [m] only used by 4-element cavities
    N_points = 300              # lens positions simulated across SHORT_ARM_LENGTHS
    SHORT_ARM_LENGTH = 0.7e-2   # [m] near mirror -> lens (the physical one, not
                                # the simulation's scan)
