"""Pure-Python interface to Rigol DG800-series function generators (SCPI over
VISA/USB). No GUI imports — this is the device layer for any interface
(kalishlot web GUI, scripts), per the lab's device/GUI decoupling rule.

Run directly for a read-only connectivity self-test:

    python rigol_dg.py

Notes:
- Opening the device performs no writes: connecting never disturbs a running
  output. Only the explicit setters write.
- While being controlled the instrument shows 'remote' and ignores its front
  panel; to_local() (or a front-panel key press) releases it.
"""

import threading

import pyvisa

RIGOL_DG_PREFIX = 'USB0::0x1AB1::0x0643::'  # vendor Rigol, product DG800 series
TIMEOUT_MS = 5000

# GUI-friendly name -> SCPI token (as returned by :SOUR<n>:FUNC?)
WAVEFORMS = {'sine': 'SIN', 'square': 'SQU', 'ramp': 'RAMP',
             'pulse': 'PULS', 'noise': 'NOIS', 'dc': 'DC'}
_TOKEN_TO_NAME = {token: name for name, token in WAVEFORMS.items()}


class RigolDG:
    """One Rigol DG800-series generator, addressed by its VISA resource string.

    All VISA I/O is serialized behind a lock, so the instance may be used
    from several threads (e.g. parallel web request handlers).
    """

    CHANNELS = (1, 2)

    def __init__(self, resource):
        self.resource = str(resource)
        self.identity = ''
        self._inst = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- device
    @staticmethod
    def list_devices():
        """Return info dicts for all connected Rigol DG800-series generators."""
        manager = pyvisa.ResourceManager()
        devices = []
        for resource in manager.list_resources('USB?*::INSTR'):
            if not resource.startswith(RIGOL_DG_PREFIX):
                continue
            try:
                inst = manager.open_resource(resource)
                inst.timeout = TIMEOUT_MS
                identity = inst.query('*IDN?').strip()
                inst.close()
            except Exception:
                continue  # present on the bus but not answering (in use?)
            # e.g. 'Rigol Technologies,DG822,DG8A262000837,00.02.06.00.01'
            fields = identity.split(',')
            model = fields[1] if len(fields) > 1 else 'DG800'
            serial = fields[2] if len(fields) > 2 else '?'
            devices.append({'resource': resource, 'model': model,
                            'serial': serial, 'identity': identity})
        return devices

    @property
    def is_open(self):
        return self._inst is not None

    def open(self):
        """Connect and read the identity. Performs no writes: a running
        output is never disturbed by connecting."""
        if self.is_open:
            return
        manager = pyvisa.ResourceManager()
        self._inst = manager.open_resource(self.resource)
        self._inst.timeout = TIMEOUT_MS
        self.identity = self._inst.query('*IDN?').strip()

    def close(self):
        if self._inst is not None:
            try:
                self._inst.close()
            finally:
                self._inst = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _query(self, command):
        with self._lock:
            return self._inst.query(command).strip()

    def _write(self, command):
        with self._lock:
            self._inst.write(command)

    # --------------------------------------------------------- channel state
    def get_state(self, channel):
        """Current settings of one channel as a JSON-friendly dict."""
        token = self._query(f':SOUR{channel}:FUNC?')
        return {'on': self._query(f':OUTP{channel}?') == 'ON',
                'waveform': _TOKEN_TO_NAME.get(token, token.lower()),
                'frequency_hz': float(self._query(f':SOUR{channel}:FREQ?')),
                'amplitude_vpp': float(self._query(f':SOUR{channel}:VOLT?')),
                'offset_v': float(self._query(f':SOUR{channel}:VOLT:OFFS?'))}

    # -------------------------------------------------------------- setters
    # Each setter reads back and returns the value the instrument actually
    # accepted (it clamps out-of-range requests), like the camera layer.
    def set_output(self, channel, on):
        self._write(f':OUTP{channel} {"ON" if on else "OFF"}')
        return self._query(f':OUTP{channel}?') == 'ON'

    def set_frequency(self, channel, hertz):
        self._write(f':SOUR{channel}:FREQ {float(hertz):.6f}')
        return float(self._query(f':SOUR{channel}:FREQ?'))

    def set_amplitude(self, channel, vpp):
        self._write(f':SOUR{channel}:VOLT:UNIT VPP')
        self._write(f':SOUR{channel}:VOLT {float(vpp):.4f}')
        return float(self._query(f':SOUR{channel}:VOLT?'))

    def set_offset(self, channel, volts):
        self._write(f':SOUR{channel}:VOLT:OFFS {float(volts):.4f}')
        return float(self._query(f':SOUR{channel}:VOLT:OFFS?'))

    def set_waveform(self, channel, name):
        token = WAVEFORMS.get(str(name).lower())
        if token is None:
            raise ValueError(f'unknown waveform {name!r}; '
                             f'choose from {sorted(WAVEFORMS)}')
        self._write(f':SOUR{channel}:FUNC {token}')
        accepted = self._query(f':SOUR{channel}:FUNC?')
        return _TOKEN_TO_NAME.get(accepted, accepted.lower())

    # ----------------------------------------------------------------- misc
    def to_local(self):
        """Give the front panel back to the user standing at the instrument."""
        try:
            self._write(':SYST:LOC')
        except Exception:
            pass  # some firmware lacks it; a front-panel key press also works


def self_test():
    """Read-only: list generators, open each, print both channels."""
    devices = RigolDG.list_devices()
    if not devices:
        print('no Rigol DG800-series generators found')
        return
    for info in devices:
        print(f"found {info['model']} s/n {info['serial']} at {info['resource']}")
        with RigolDG(info['resource']) as gen:
            print(f'  identity: {gen.identity}')
            for channel in gen.CHANNELS:
                state = gen.get_state(channel)
                print(f"  CH{channel}: {'ON ' if state['on'] else 'off'} "
                      f"{state['waveform']} {state['frequency_hz']:g} Hz, "
                      f"{state['amplitude_vpp']:g} Vpp, "
                      f"offset {state['offset_v']:g} V")
            gen.to_local()
    print('self-test passed (read-only)')


if __name__ == '__main__':
    self_test()
