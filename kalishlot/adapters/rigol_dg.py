"""Rigol DG800-series function generator adapter.

First control-only device type (no frames): the box is settings and state
only. Wraps the pure device layer in rigol_fg/rigol_dg.py. Opening performs
no writes — CH1 may be driving an experiment when someone adds the box.
"""

import sys
from pathlib import Path

# the device layer lives at the repo root, outside kalishlot/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'rigol_fg'))
from rigol_dg import WAVEFORMS, RigolDG  # noqa: E402

from .base import DeviceAdapter  # noqa: E402

SETTERS = {'output': RigolDG.set_output,
           'frequency': RigolDG.set_frequency,
           'amplitude': RigolDG.set_amplitude,
           'offset': RigolDG.set_offset,
           'waveform': RigolDG.set_waveform}

# Lab safety cap, channel -> max amplitude in Vpp. CH1 drives the laser
# temperature scan: above ~5 Vpp the laser is cooled so much that humidity
# condenses on it. Requests beyond the cap are clamped, never forwarded.
AMPLITUDE_LIMIT_VPP = {1: 5.0}


class RigolDGAdapter(DeviceAdapter):
    type_name = 'rigol_dg'
    display_name = 'Rigol function generator'

    @staticmethod
    def list_available():
        return [{'address': d['resource'],
                 'label': f"{d['model']} s/n {d['serial']}"}
                for d in RigolDG.list_devices()]

    def __init__(self, address):
        super().__init__(address)
        self.generator = RigolDG(address)
        self._label = self.display_name

    # ------------------------------------------------------------ lifecycle
    def open(self):
        self.generator.open()
        fields = self.generator.identity.split(',')
        if len(fields) > 2:
            self._label = f'{fields[1]} — s/n {fields[2]}'

    def close(self):
        self.generator.close()

    def describe(self):
        return {'type': self.type_name,
                'label': self._label,
                'commands': ['set_channel', 'refresh', 'local'],
                'waveforms': sorted(WAVEFORMS),
                'channels': [self.generator.get_state(channel)
                             for channel in self.generator.CHANNELS]}

    # ------------------------------------------------------------- commands
    def _emit_channel(self, channel):
        self.emit({'type': 'channel', 'channel': channel,
                   'state': self.generator.get_state(channel)})

    def command(self, name, args):
        if name == 'set_channel':
            channel = int(args['channel'])
            if channel not in self.generator.CHANNELS:
                raise ValueError(f'no channel {channel}')
            setter = SETTERS.get(args['name'])
            if setter is None:
                raise ValueError(f"unknown channel setting {args['name']!r}")
            value = args['value']
            if args['name'] == 'amplitude' and channel in AMPLITUDE_LIMIT_VPP:
                value = min(float(value), AMPLITUDE_LIMIT_VPP[channel])
            setter(self.generator, channel, value)
            # broadcast the state the instrument actually accepted
            self._emit_channel(channel)
            return {'ok': True}
        if name == 'refresh':
            # pick up changes made on the front panel
            for channel in self.generator.CHANNELS:
                self._emit_channel(channel)
            return {'ok': True}
        if name == 'local':
            self.generator.to_local()
            return {'ok': True}
        raise ValueError(f'unknown command {name!r}')
