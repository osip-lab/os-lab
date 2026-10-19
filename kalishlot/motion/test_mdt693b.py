"""Hardware-free tests of the MDT693B driver against a fake instrument.

    python -m motion.test_mdt693b        (from kalishlot/)

The fake replies byte-for-byte like the real controller (firmware 1.17,
recorded 2026-10-19), including the echo and the ~+0.07 V readback offset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion import build_axis  # noqa: E402
from motion.safety import AxisLimits, SafetyError  # noqa: E402
from motion.thorlabs_mdt693b import MDT693BController, visa_address  # noqa: E402

ID_REPLY = ('Model MDT693B Piezo Control Module\rFirmware Version: 1.17\rVoltage Range: 0V to 150V\r'
            'Serial#:2408096171-09\rFriendly Name:MDT693B\rThorlabs, Inc. Newton, NJ 07860\r'
            'www.thorlabs.com\r')


class FakeInstrument:
    def __init__(self, vlimit=150):
        self.out = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.vlimit = vlimit
        self.writes = []
        self._buf = b''
        self.closed = False

    def write(self, cmd):
        self.writes.append(cmd)
        if cmd == 'id?':
            reply = '\r\r\r' + ID_REPLY + '\r'
        elif cmd == 'vlimit?':
            reply = f'[ {self.vlimit}]\r'
        elif cmd.endswith('voltage?'):
            v = self.out[cmd[0]]
            reply = f'[{v + 0.07 if v else 0:6.2f}]\r'
        elif '=' in cmd:
            ch, v = cmd[0], float(cmd.split('=')[1])
            self.out[ch] = min(v, self.vlimit)
            reply = ''
        else:
            reply = 'CMD_NOT_DEFINED\r'
        self._buf += (cmd + '\r' + reply + '>').encode('ascii')

    def read_bytes(self, n):
        b, self._buf = self._buf[:n], self._buf[n:]
        if not b:
            raise TimeoutError('fake: nothing to read')
        return b

    def close(self):
        self.closed = True


def make_config(fake, max_v=75.0):
    MDT693BController._instances.clear()
    ctrl = MDT693BController.shared('COM9')
    ctrl._open_instrument = lambda: fake
    lim = AxisLimits(min=0.0, max=max_v, max_step=0.5, max_slew=1e6)
    return {'px': {'driver': 'thorlabs_mdt693b', 'address': 'COM9', 'options': {'channel': 'x'},
                   'limits': lim},
            'py': {'driver': 'thorlabs_mdt693b', 'address': 'ASRL9::INSTR', 'options': {'channel': 'y'},
                   'limits': lim}}


def test_address():
    assert visa_address('COM4') == 'ASRL4::INSTR' == visa_address('ASRL4::INSTR')


def test_open_reads_only_and_close_never_writes():
    fake = FakeInstrument()
    fake.out['x'] = 5.0
    cfg = make_config(fake)
    x, y = build_axis('px', cfg), build_axis('py', cfg)
    x.open(), y.open()
    assert abs(x.position - 5.07) < 1e-9 and y.position == 0.0
    assert not any('=' in w for w in fake.writes), fake.writes
    assert x._raw.controller is y._raw.controller       # one session for the two channels
    x.close()
    assert not fake.closed                               # y still uses it
    y.close()
    assert fake.closed and not any('=' in w for w in fake.writes)


def test_ramped_move():
    fake = FakeInstrument()
    cfg = make_config(fake)
    x = build_axis('px', cfg, sleep=lambda s: None)
    x.open()
    r = x.move_to(2.0, who='test')
    sets = [w for w in fake.writes if '=' in w]
    assert sets == ['xvoltage=0.500', 'xvoltage=1.000', 'xvoltage=1.500', 'xvoltage=2.000'], sets
    assert abs(r.readback - 2.07) < 1e-9 and fake.out['y'] == 0.0


def test_out_of_range_rejected():
    fake = FakeInstrument()
    x = build_axis('px', make_config(fake), sleep=lambda s: None)
    x.open()
    for bad in (-0.1, 75.1, float('nan')):
        try:
            x.move_to(bad, who='test')
        except SafetyError:
            pass
        else:
            raise AssertionError(bad)
    assert not any('=' in w for w in fake.writes)


def test_soft_max_above_limit_switch_refused():
    fake = FakeInstrument(vlimit=75)
    x = build_axis('px', make_config(fake, max_v=100.0))
    try:
        x.open()
    except RuntimeError as e:
        assert 'limit switch' in str(e)
    else:
        raise AssertionError('opened with soft max above the limit switch')
    assert fake.closed


def test_front_panel_knob_refuses_move():
    fake = FakeInstrument()
    x = build_axis('px', make_config(fake), sleep=lambda s: None)
    x.open()
    x.move_to(1.0, who='test')
    fake.out['x'] = 20.0                                 # someone turned the knob
    n = len(fake.writes)
    try:
        x.move_to(1.5, who='test')
    except RuntimeError as e:
        assert 'knob' in str(e)
    else:
        raise AssertionError('moved from a stale voltage')
    assert not any('=' in w for w in fake.writes[n:])
    assert fake.out['x'] == 20.0


def test_not_an_mdt693b():
    fake = FakeInstrument()
    fake.write = lambda cmd: setattr(fake, '_buf', fake._buf + (cmd + '\rHELLO\r>').encode())
    x = build_axis('px', make_config(fake))
    try:
        x.open()
    except RuntimeError as e:
        assert 'not an MDT693B' in str(e)
    else:
        raise AssertionError('accepted an unknown instrument')


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print(f'ok  {name}')
    print(f'{len(tests)} MDT693B tests passed')


if __name__ == '__main__':
    main()
