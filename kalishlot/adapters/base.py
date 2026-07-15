"""Base interface for device adapters.

An adapter wraps one device instance for the web server: it exposes a uniform
vocabulary (open/close, describe, command, latest display frame, events) so the
canvas frontend and the server never need device-specific code. New instrument
types are added by subclassing DeviceAdapter and registering the class in
server.DEVICE_TYPES; the pure device layer (e.g. basler_cam/basler_cameras.py)
stays untouched.

Threading model: adapters produce frames/events from their own background
threads; the server reads them from asyncio handlers. Everything crossing that
boundary goes through the small thread-safe helpers here (frame under a lock,
events through per-listener queues).
"""

import queue
import threading


class DeviceAdapter:
    # subclasses override these two class attributes
    type_name = 'base'
    display_name = 'base device'

    @staticmethod
    def list_available():
        """Return [{'address': str, 'label': str}, ...] of connectable devices."""
        raise NotImplementedError

    def __init__(self, address):
        self.address = str(address)
        self._frame_lock = threading.Lock()
        self._display_frame = None  # uint8 grayscale, ready for JPEG encoding
        self._frame_id = 0
        self._listeners = []
        self._listeners_lock = threading.Lock()

    # ------------------------------------------------------------ lifecycle
    def open(self):
        """Connect to the device and start producing frames/events."""
        raise NotImplementedError

    def close(self):
        """Stop threads and release the device."""
        raise NotImplementedError

    def describe(self):
        """Static description for the frontend box: label, frame shape,
        supported commands, settings schema with current values."""
        raise NotImplementedError

    # ------------------------------------------------------------- commands
    def command(self, name, args):
        """Execute a command (play/pause/snap/set_setting/...); return a
        JSON-able result dict. Raise ValueError for unknown commands."""
        raise NotImplementedError

    # -------------------------------------------------- settings persistence
    # The server keeps a per-device record of these snapshots on disk and
    # calls restore_settings() right after open() when the device was used
    # before, so a re-opened device comes back with its last-used settings.
    def settings_snapshot(self):
        """JSON-able dict of the settings worth persisting across opens, or
        None when the device keeps its own state (e.g. a bench instrument
        that remembers its configuration — restoring would overwrite it)."""
        return None

    def restore_settings(self, snapshot):
        """Apply a snapshot produced by settings_snapshot(); called after
        open(), before the first describe(). Must tolerate stale/partial
        snapshots (settings may have changed between versions)."""

    # --------------------------------------------------------------- frames
    def _store_display_frame(self, display_frame):
        """Called by the adapter's producing thread with a display-ready
        (uint8 grayscale) frame."""
        with self._frame_lock:
            self._display_frame = display_frame
            self._frame_id += 1

    def latest_display_frame(self):
        """Return (frame_id, uint8 image) of the newest frame, or (0, None).
        Consumers compare frame_id to skip already-sent frames — the same
        newest-frame-only rule as the desktop GUI, so slow consumers never
        build a backlog."""
        with self._frame_lock:
            return self._frame_id, self._display_frame

    # --------------------------------------------------------------- events
    def add_listener(self):
        """Register a consumer; returns a Queue receiving JSON-able event
        dicts (settings applied, status changes, fit results, errors)."""
        listener = queue.Queue(maxsize=100)
        with self._listeners_lock:
            self._listeners.append(listener)
        return listener

    def remove_listener(self, listener):
        with self._listeners_lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def emit(self, event):
        """Send an event dict to all listeners (dropped for a listener whose
        queue is full — a stuck consumer must not block the device)."""
        with self._listeners_lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener.put_nowait(event)
            except queue.Full:
                pass
