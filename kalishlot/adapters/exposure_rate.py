"""Exposure and frame rate that follow each other, for cameras that have both.

Setting one sets the other to the most it can be:

  rate asked for      -> the exposure fills the frame period, as far as the
                         camera can still keep that rate
  exposure asked for  -> the rate goes to the camera's maximum at that exposure

Written against the device-layer camera surface only (exposure_us,
frame_rate_hz, frame_rate_limits_hz), so it holds no SDK and runs on the
streaming thread inside the adapter's apply(). The camera's own ceiling on the
rate is what decides - it includes the readout and link overheads that no
formula here would know - so the exposure is found by asking it rather than by
subtracting a guessed gap from the period.
"""

MAX_TRIES = 6
MARGIN_US = 1.0  # beyond the measured overshoot, so the next try lands inside


def rate_for_exposure(camera, exposure_us):
    """Set the exposure, then the fastest rate the camera allows with it."""
    camera.exposure_us = exposure_us
    camera.frame_rate_hz = camera.frame_rate_limits_hz[1]


def exposure_for_rate(camera, rate_hz):
    """Set the rate, then the longest exposure that still keeps it.

    The rate first goes to what was asked for, shortening the exposure first
    when the current one is too long to allow it. Then the exposure starts at
    the whole period (what the camera can actually keep, if the rate was
    clipped) and is shortened by however far the camera's ceiling falls short
    of the rate - a step or two, since the ceiling moves one-for-one with the
    exposure. Should that not settle, the exposure is left at the longest
    that was seen to fit.
    """
    period = 1e6 / rate_hz
    if camera.exposure_us > period:
        camera.exposure_us = period
    if camera.frame_rate_limits_hz[1] < rate_hz:
        camera.exposure_us = min(camera.exposure_us, 0.5 * period)
    camera.frame_rate_hz = rate_hz
    rate = camera.frame_rate_hz          # clipped, if even that was too fast
    period = 1e6 / rate

    fits = camera.exposure_us            # the rate is being kept with this
    exposure = period
    for _ in range(MAX_TRIES):
        camera.exposure_us = exposure
        ceiling = camera.frame_rate_limits_hz[1]
        if ceiling >= rate * (1 - 1e-6):
            fits = camera.exposure_us
            break
        exposure = camera.exposure_us - (1e6 / ceiling - period) - MARGIN_US
        if exposure <= fits:
            break
    camera.exposure_us = fits
    camera.frame_rate_hz = rate          # a longer try may have pulled it down
