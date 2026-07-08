"""Reliable interface to Basler cameras via pypylon.

This module provides a thin, robust wrapper around pypylon that supports
several cameras open simultaneously. It is meant to be the camera backend
for a future GUI, but can also be run directly as a connectivity self-test:

    python basler_cameras.py

The self-test enumerates all connected Basler cameras, opens all of them at
the same time, sets exposure/gain, grabs a frame from each, prints image
statistics and closes everything cleanly.
"""

import queue
import threading
import time

import numpy as np
from pypylon import pylon


class BaslerCamera:
    """A single Basler camera, addressed by serial number.

    Usage:
        cam = BaslerCamera('24756778')
        cam.open()
        cam.exposure_us = 3000
        cam.gain_db = 0.0
        img = cam.grab()
        cam.close()

    or as a context manager:
        with BaslerCamera('24756778') as cam:
            img = cam.grab()
    """

    GRAB_TIMEOUT_MS = 10000
    MAX_FRAME_RATE = 10.0  # Hz, per camera
    THROUGHPUT_LIMIT_BPS = 150_000_000  # bytes/s per camera, two cameras fit in USB3
    GRAB_RETRIES = 3  # discarded-frame retries during streaming

    def __init__(self, serial_number):
        self.serial_number = str(serial_number)
        self._cam = None

    # ---------------------------------------------------------------- device
    @staticmethod
    def list_devices():
        """Return info dicts for all Basler cameras connected to this PC."""
        factory = pylon.TlFactory.GetInstance()
        return [{'serial_number': d.GetSerialNumber(),
                 'model': d.GetModelName(),
                 'friendly_name': d.GetFriendlyName()}
                for d in factory.EnumerateDevices()]

    @property
    def is_open(self):
        return self._cam is not None and self._cam.IsOpen()

    def open(self, pixel_format='Mono12'):
        """Connect to the camera and apply sane defaults."""
        if self.is_open:
            return
        factory = pylon.TlFactory.GetInstance()
        matches = [d for d in factory.EnumerateDevices()
                   if d.GetSerialNumber() == self.serial_number]
        if not matches:
            available = [d['serial_number'] for d in self.list_devices()]
            raise RuntimeError(f'camera s/n {self.serial_number} not found; '
                               f'available: {available}')
        self._cam = pylon.InstantCamera(factory.CreateDevice(matches[0]))
        self._cam.Open()

        self._cam.ExposureMode.SetValue('Timed')
        self._cam.ExposureAuto.SetValue('Off')
        self._cam.GainSelector.SetValue('All')
        self._cam.GainAuto.SetValue('Off')
        self._cam.PixelFormat.SetValue(pixel_format)

        # Two cameras free-running at full speed oversubscribe the USB3 bus
        # and frames get discarded ("payload data has been discarded").
        # Basler's recommended fix is to limit each camera's link throughput
        # so the sum over all cameras stays below the host controller
        # capacity (~380 MB/s practical for USB3). This paces the data on the
        # wire; capping the frame rate alone does not prevent burst collisions.
        self._cam.DeviceLinkThroughputLimitMode.SetValue('On')
        self._cam.DeviceLinkThroughputLimit.SetValue(self.THROUGHPUT_LIMIT_BPS)
        self._cam.AcquisitionFrameRateEnable.SetValue(True)
        self._cam.AcquisitionFrameRate.SetValue(self.MAX_FRAME_RATE)

    def close(self):
        if self._cam is not None:
            try:
                if self._cam.IsGrabbing():
                    self._cam.StopGrabbing()
                if self._cam.IsOpen():
                    self._cam.Close()
            finally:
                self._cam = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # ------------------------------------------------------------- settings
    @property
    def model(self):
        return self._cam.DeviceModelName.GetValue()

    @property
    def frame_shape(self):
        """(height, width) of the frames the camera delivers."""
        return self._cam.Height.GetValue(), self._cam.Width.GetValue()

    @property
    def exposure_us(self):
        return self._cam.ExposureTime.GetValue()

    @exposure_us.setter
    def exposure_us(self, value):
        node = self._cam.ExposureTime
        self._cam.ExposureTime.SetValue(float(np.clip(value, node.Min, node.Max)))

    @property
    def gain_db(self):
        return self._cam.Gain.GetValue()

    @gain_db.setter
    def gain_db(self, value):
        node = self._cam.Gain
        self._cam.Gain.SetValue(float(np.clip(value, node.Min, node.Max)))

    @property
    def exposure_limits_us(self):
        node = self._cam.ExposureTime
        return node.Min, node.Max

    @property
    def gain_limits_db(self):
        node = self._cam.Gain
        return node.Min, node.Max

    # ------------------------------------------------------------- grabbing
    def grab(self):
        """Grab a single frame and return it as a numpy array."""
        result = self._cam.GrabOne(self.GRAB_TIMEOUT_MS)
        try:
            if not result.GrabSucceeded():
                raise RuntimeError(f'grab failed on camera {self.serial_number}: '
                                   f'{result.ErrorCode} {result.ErrorDescription}')
            return result.Array.copy()
        finally:
            result.Release()

    def start_streaming(self):
        """Start continuous acquisition; retrieve frames with get_frame()."""
        if not self._cam.IsGrabbing():
            self._cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def get_frame(self):
        """Retrieve the latest frame while streaming.

        A frame the camera discarded for lack of USB bandwidth is skipped and
        the next one is awaited, up to GRAB_RETRIES times.
        """
        last_error = ''
        for _ in range(self.GRAB_RETRIES + 1):
            result = self._cam.RetrieveResult(self.GRAB_TIMEOUT_MS,
                                              pylon.TimeoutHandling_ThrowException)
            try:
                if result.GrabSucceeded():
                    return result.Array.copy()
                last_error = f'{result.ErrorCode} {result.ErrorDescription}'
            finally:
                result.Release()
        raise RuntimeError(f'grab failed on camera {self.serial_number}: {last_error}')

    def stop_streaming(self):
        if self._cam.IsGrabbing():
            self._cam.StopGrabbing()


class CameraStreamer:
    """Grab frames continuously from an open BaslerCamera in a background
    thread and deliver them through a callback.

    GUI-agnostic on purpose: any interface (desktop GUI, web server) supplies
    `on_frame(image)` and optionally `on_error(exception)`; both are called
    from the streaming thread, so the interface is responsible for handing
    the data over to its own event loop if needed.
    """

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

        The pylon camera object is not thread-safe, so while the streamer is
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
        if self._thread is not None:
            self._thread.join(timeout=2 * BaslerCamera.GRAB_TIMEOUT_MS / 1000)
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


def self_test():
    """Open all connected cameras simultaneously and grab a frame from each."""
    devices = BaslerCamera.list_devices()
    if not devices:
        print('no Basler cameras found')
        return

    print(f'found {len(devices)} camera(s):')
    for d in devices:
        print(f"  s/n {d['serial_number']} - {d['model']}")

    cameras = [BaslerCamera(d['serial_number']) for d in devices]
    try:
        # open all cameras at the same time
        for cam in cameras:
            cam.open()
            print(f'\nopened {cam.serial_number} ({cam.model})')
            print(f'  exposure range: {cam.exposure_limits_us[0]:.0f} - '
                  f'{cam.exposure_limits_us[1]:.0f} us')
            print(f'  gain range: {cam.gain_limits_db[0]:.1f} - '
                  f'{cam.gain_limits_db[1]:.1f} dB')

        # set and read back exposure / gain on every camera
        for cam in cameras:
            cam.exposure_us = 3000
            cam.gain_db = 0.0
            print(f'{cam.serial_number}: set exposure -> read back '
                  f'{cam.exposure_us:.1f} us, gain -> {cam.gain_db:.1f} dB')

        # grab a single frame from each camera while all are open
        for cam in cameras:
            tic = time.time()
            img = cam.grab()
            dt = time.time() - tic
            print(f'{cam.serial_number}: grabbed {img.shape} {img.dtype} in {dt:.2f} s, '
                  f'min {img.min()}, max {img.max()}, mean {img.mean():.1f}')

        # short burst of continuous grabbing from all cameras interleaved
        for cam in cameras:
            cam.start_streaming()
        tic = time.time()
        n_frames = 5
        for i in range(n_frames):
            for cam in cameras:
                img = cam.get_frame()
            print(f'streaming frame {i + 1}/{n_frames} from all cameras ok')
        dt = time.time() - tic
        print(f'streamed {n_frames} frames from {len(cameras)} camera(s) '
              f'in {dt:.2f} s ({n_frames * len(cameras) / dt:.1f} frames/s total)')
        for cam in cameras:
            cam.stop_streaming()

        print('\nself-test passed')
    finally:
        for cam in cameras:
            cam.close()
        print('all cameras closed')


if __name__ == '__main__':
    self_test()
