"""What every camera in this lab has to look like, and the parts they share.

Two cameras now feed the same code: a Basler acA2040 through pypylon and a
XIMEA MQ042 through xiapi. `basler_cam/basler_cameras.py` and
`ximea_cam/ximea_cameras.py` each wrap their own SDK, but everything above
them - the kalishlot adapters, and the synchronized capture in
`pico_scope/mode_video_capture.py` - is written against the one surface
described here and never asks which make it is holding.

Nothing in this module imports an SDK, so it can be imported to check the
contract on a machine with neither camera installed.

## The surface

    open(), close(), is_open, serial_number, model, pixel_size_mm
    pixel_format, set_pixel_format(fmt), formats, deepest_format,
        saturation_level
    exposure_us, gain_db, exposure_limits_us, gain_limits_db
    frame_rate_hz, resulting_frame_rate
    binning, set_binning(n), max_binning, binning_mode
    set_roi(w, h, ox, oy), set_roi_full(), max_frame_size,
        max_frame_rate_for(w, h, fmt, binning), assert_frame_rate_reachable(hz)
    throughput_limit_bps, set_throughput_limit(bps)
    record_burst(n) -> (frames, meta), describe()
    list_devices()                                          [static]
    start_streaming(), get_frame(), stop_streaming()        [for CameraStreamer]

Two conventions are worth stating because they are easy to get subtly wrong:

**Sizes are in binned pixels.** `set_roi` and `max_frame_size` speak the
camera's coordinates *after* binning; only `max_frame_rate_for` takes sensor
pixels, because link bandwidth is charged on what crosses the wire.

**Binning means an N x N sum**, whoever does it. The Basler has it in firmware;
the XIMEA MQ042 offers no downsampling at all, so its wrapper sums on the host
as frames arrive. `binning_mode` says which, and `saturation_level` grows with
the square of the factor to match. Callers get four times the signal either
way and never branch on it.

## The synchronization contract

`record_burst(n)` returns `(frames, meta)` where `frames` is `(n, h, w)` and
`meta` is one dict per frame:

    {'block_id':            int, increments by exactly 1 per delivered frame,
     'camera_timestamp_ns': int, stamped by the camera at exposure, in
                                 nanoseconds, on an arbitrary but stable epoch,
     'host_time_s':         float, time.time() on arrival}

These two integers are the whole of it. `mode_video_sync.frame_start_times()`
reads `camera_timestamp_ns` and nothing else to place every frame on the scope's
time axis, and `dropped_frames()` reads `block_id` and nothing else to know a
frame went missing. A host-side timestamp substituted for a camera-side one
would still look plausible and would quietly carry the USB latency into the
alignment, so `burst_timing()` below measures the implied period rather than
trusting the unit.
"""

import queue
import threading

import numpy as np

# Everything a camera must expose to be usable by the capture script and the
# kalishlot adapters. Checked by check_camera_surface(), which both device
# modules run in their self-tests - so a method missed while adding a third
# camera fails on a laptop, not at the bench.
CAMERA_SURFACE = (
    'open', 'close', 'is_open', 'model', 'pixel_size_mm',
    'pixel_format', 'set_pixel_format', 'formats', 'deepest_format',
    'saturation_level',
    'exposure_us', 'gain_db', 'exposure_limits_us', 'gain_limits_db',
    'frame_rate_hz', 'resulting_frame_rate',
    'binning', 'set_binning', 'max_binning', 'binning_mode',
    'set_roi', 'set_roi_full', 'max_frame_size', 'max_frame_rate_for',
    'assert_frame_rate_reachable',
    'throughput_limit_bps', 'set_throughput_limit',
    'record_burst', 'describe', 'list_devices',
    'start_streaming', 'get_frame', 'stop_streaming',
)

# Set in __init__ rather than on the class, so they are only checkable once a
# camera has been constructed.
CAMERA_INSTANCE_ATTRS = ('serial_number',)

# Keys every row of a record_burst() meta list carries.
BURST_META_KEYS = ('block_id', 'camera_timestamp_ns', 'host_time_s')


def check_camera_surface(camera, name=None):
    """Raise unless `camera` (a class or an instance) has the whole surface.

    Takes a class on purpose: the check then costs nothing and needs no
    hardware, so it can run in a self-test on any machine. Given a class it
    can only see what the class carries, so the handful of attributes that
    are assigned in __init__ are checked only when given an instance.
    """
    name = name or getattr(camera, '__name__', type(camera).__name__)
    expected = CAMERA_SURFACE
    if not isinstance(camera, type):
        expected = expected + CAMERA_INSTANCE_ATTRS
    missing = [item for item in expected if not hasattr(camera, item)]
    if missing:
        raise AssertionError(
            f'{name} is missing {len(missing)} of the camera surface: '
            f'{", ".join(missing)}. See camera_core.py for what each means.')
    return True


def check_burst_meta(meta, name='meta'):
    """Raise unless a record_burst() meta list can drive the synchronization."""
    if not meta:
        raise AssertionError(f'{name} is empty')
    for i, row in enumerate(meta):
        absent = [key for key in BURST_META_KEYS if key not in row]
        if absent:
            raise AssertionError(f'{name}[{i}] lacks {", ".join(absent)}')
    stamps = [row['camera_timestamp_ns'] for row in meta]
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise AssertionError(
            f'{name} timestamps are not strictly increasing, so they cannot '
            f'place frames on a time axis')
    return True


def burst_timing(meta, expected_rate_hz=None):
    """Sanity-check a record_burst() metadata list; return a report dict.

    Three things are worth knowing about a burst before it is trusted:
    whether any frame was dropped (a gap in block_id), how steady the frame
    period was, and whether the camera timestamps really are in nanoseconds -
    which is checked by comparing the period they imply against the frame rate
    that was asked for, rather than assumed.
    """
    block_ids = np.array([row['block_id'] for row in meta], dtype=np.int64)
    stamps = np.array([row['camera_timestamp_ns'] for row in meta],
                      dtype=np.float64)
    gaps = np.diff(block_ids)
    missing = [{'after_frame': int(i), 'after_block_id': int(block_ids[i]),
                'n_missing': int(gap - 1)}
               for i, gap in enumerate(gaps) if gap != 1]
    periods_s = np.diff(stamps) / 1e9
    report = {
        'n_frames': int(block_ids.size),
        'dropped': missing,
        'n_dropped': int(sum(m['n_missing'] for m in missing)),
        'period_s_median': float(np.median(periods_s)) if periods_s.size else None,
        'period_s_std': float(np.std(periods_s)) if periods_s.size else None,
        'duration_s': float((stamps[-1] - stamps[0]) / 1e9) if stamps.size > 1 else 0.0,
    }
    if expected_rate_hz and report['period_s_median']:
        expected = 1.0 / expected_rate_hz
        report['period_ratio_to_expected'] = report['period_s_median'] / expected
        # A ratio far from 1 usually means the timestamp tick is not a
        # nanosecond on this model, not that the camera missed its rate.
        report['timestamps_look_like_ns'] = bool(
            0.9 < report['period_ratio_to_expected'] < 1.1)
    return report


class CameraStreamer:
    """Grab frames continuously from an open camera in a background thread
    and deliver them through a callback.

    GUI-agnostic on purpose: any interface (desktop GUI, web server) supplies
    `on_frame(image)` and optionally `on_error(exception)`; both are called
    from the streaming thread, so the interface is responsible for handing
    the data over to its own event loop if needed.

    Works with any camera meeting the contract above. Neither pylon nor xiapi
    is thread-safe, so `submit()` is not a convenience - it is the only legal
    way to touch the camera while the stream is running.
    """

    # Fallback join timeout for a camera that does not declare its own.
    GRAB_TIMEOUT_MS = 10000

    def __init__(self, camera, on_frame, on_error=None):
        self.camera = camera
        self.on_frame = on_frame
        self.on_error = on_error
        self._thread = None
        self._playing = threading.Event()
        self._stopping = threading.Event()
        self._single_request = threading.Event()
        self._commands = queue.Queue()

    @property
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_paused(self):
        return not self._playing.is_set()

    def start(self):
        if self.is_running:
            self._playing.set()
            return
        self._stopping.clear()
        self._playing.set()
        self.camera.start_streaming()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f'stream-{self.camera.serial_number}')
        self._thread.start()

    def pause(self):
        """Stop delivering frames; acquisition thread stays alive."""
        self._playing.clear()

    def resume(self):
        self._playing.set()

    def snap(self):
        """Deliver one frame while paused.

        All camera access stays in the streaming thread, so this only posts
        a request; the frame arrives through on_frame like any other.
        While playing this is a no-op (frames are coming anyway).
        """
        if self.is_paused:
            self._single_request.set()

    def submit(self, command):
        """Run `command(camera)` in the streaming thread between grabs.

        The SDK camera object is not thread-safe, so while the streamer is
        running, all camera access (e.g. changing exposure/gain) must go
        through here instead of calling the camera directly. The command is
        responsible for its own error handling; an uncaught exception is
        reported through on_error but does not stop the stream.
        """
        self._commands.put(command)

    def _run_commands(self):
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                command(self.camera)
            except Exception as error:
                if self.on_error is not None:
                    self.on_error(error)

    def stop(self):
        """Stop the thread and the camera's acquisition."""
        self._stopping.set()
        self._playing.set()  # release a paused loop so it can exit
        # Ask the camera how long a grab may take, rather than naming one
        # camera's constant here - the two SDKs do not agree on it.
        timeout_ms = getattr(self.camera, 'GRAB_TIMEOUT_MS', self.GRAB_TIMEOUT_MS)
        if self._thread is not None:
            self._thread.join(timeout=2 * timeout_ms / 1000)
            self._thread = None
        if self.camera.is_open:
            self.camera.stop_streaming()

    def _loop(self):
        while not self._stopping.is_set():
            self._run_commands()
            single = False
            if not self._playing.wait(timeout=0.1):
                if not self._single_request.is_set():
                    continue  # paused; keep checking for resume/stop/snap
                self._single_request.clear()
                single = True
            if self._stopping.is_set():
                break
            try:
                frame = self.camera.get_frame()
            except Exception as error:
                if self._stopping.is_set():
                    break
                if self.on_error is not None:
                    self.on_error(error)
                break
            if self._stopping.is_set():
                break
            if single or self._playing.is_set():
                self.on_frame(frame)


def sum_bin(frames, factor):
    """Sum non-overlapping factor x factor blocks. Works on one frame or a stack.

    The host-side equivalent of the Basler's firmware Sum binning, for cameras
    that have none. Summing rather than averaging is the point: it is what
    multiplies the signal by factor**2 and buys the extra bits of range, which
    is the whole reason the Basler bins.

    Trailing rows and columns that do not fill a block are dropped, so the
    caller should size the ROI in multiples of `factor`.
    """
    frames = np.asarray(frames)
    if factor == 1:
        return frames
    h, w = frames.shape[-2:]
    h, w = (h // factor) * factor, (w // factor) * factor
    trimmed = frames[..., :h, :w]
    shape = trimmed.shape[:-2] + (h // factor, factor, w // factor, factor)
    # int64 first: a sum of four 10-bit pixels overflows nothing, but a sum of
    # four 16-bit ones would wrap silently in the input dtype.
    return trimmed.reshape(shape).sum(axis=(-3, -1), dtype=np.int64)


def _self_test():
    """No hardware: the shared helpers, and the checks that guard the contract."""
    print('camera_core self-test')

    meta = [{'block_id': 10 + i,
             'camera_timestamp_ns': 1_000_000_000 + i * 10_000_000,
             'host_time_s': 1.0 + i * 0.01} for i in range(20)]
    check_burst_meta(meta)
    report = burst_timing(meta, expected_rate_hz=100.0)
    assert report['n_frames'] == 20 and report['n_dropped'] == 0
    assert abs(report['period_s_median'] - 0.01) < 1e-12, report
    assert report['timestamps_look_like_ns'], report
    print(f"  {report['n_frames']} frames, period "
          f"{report['period_s_median'] * 1e3:.2f} ms, none dropped")

    # a gap in block_id is a dropped frame, wherever the timestamps landed
    gapped = meta[:5] + [dict(row, block_id=row['block_id'] + 3)
                         for row in meta[5:]]
    assert burst_timing(gapped)['n_dropped'] == 3, burst_timing(gapped)
    assert burst_timing(gapped)['dropped'][0]['after_frame'] == 4
    print('  a gap in block_id is counted, not smoothed over')

    # a camera whose tick is not a nanosecond is caught, not believed
    microseconds = [dict(row, camera_timestamp_ns=row['camera_timestamp_ns'] // 1000)
                    for row in meta]
    assert not burst_timing(microseconds, 100.0)['timestamps_look_like_ns']
    print('  a timestamp tick that is not a nanosecond is caught by the '
          'implied period, not assumed correct')

    # the surface check names what is missing rather than failing late
    class Incomplete:
        pass
    try:
        check_camera_surface(Incomplete)
    except AssertionError as error:
        assert 'record_burst' in str(error), error
    else:
        raise AssertionError('the surface check passed an empty class')
    print('  the surface check names the missing methods')

    # host-side binning sums, and grows the full scale by the square
    frame = np.ones((8, 10), dtype=np.uint16)
    binned = sum_bin(frame, 2)
    assert binned.shape == (4, 5) and binned.max() == 4, binned
    stack = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)
    assert sum_bin(stack, 2).shape == (2, 2, 2)
    assert sum_bin(stack, 2)[0, 0, 0] == 0 + 1 + 4 + 5
    # odd sizes lose the remainder rather than raising
    assert sum_bin(np.ones((5, 5), dtype=np.uint16), 2).shape == (2, 2)
    print('  host-side 2x2 binning sums (4x signal), as the Basler firmware does')

    print('self-test passed')


if __name__ == '__main__':
    _self_test()
