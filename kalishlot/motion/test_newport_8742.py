"""Hardware-free tests of the 8742 Picomotor driver against a fake controller.

    python -m motion.test_newport_8742        (from kalishlot/)

The fake answers like the real controller (firmware 3.04, recorded 2026-11-23):
replies without CR LF (the transport strips it), PA moves that take time.
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion import build_axis  # noqa: E402
from motion.newport_8742 import Picomotor8742Controller, parse_address  # noqa: E402
from motion.safety import AxisLimits, SafetyError  # noqa: E402

IDN = 'New_Focus 8742 v3.04 09/09/16 103147'


class FakeController:
    """Moves `steps_per_poll` steps every time MD? is asked while moving."""

    def __init__(self, motor_types=(3, 0, 0, 0), position=100, steps_per_poll=150):
        self.qm = dict(zip((1, 2, 3, 4), motor_types))
        self.tp = {1: position, 2: 0, 3: 0, 4: 0}
        self.va = {c: 2000 for c in (1, 2, 3, 4)}
        self.target = {}
        self.errors = []
        self.writes = []
        self.reply = None
        self.closed = False
        self.steps_per_poll = steps_per_poll
        self.stall_at = None        # counter value where a move gets stuck (timeout test)

    def write(self, cmd):
        self.writes.append(cmd)
        self.reply = None
        if cmd == '*IDN?':
            self.reply = IDN
        elif cmd == 'TB?':
            self.reply = self.errors.pop(0) if self.errors else '0, NO ERROR DETECTED'
        elif cmd == 'AB':
            self.target.clear()
        else:
            ch, op, arg = int(cmd[0]), cmd[1:3], cmd[3:]
            if op == 'QM':
                self.reply = str(self.qm[ch])
            elif op == 'TP':
                self.reply = str(self.tp[ch])
            elif op == 'VA' and arg == '?':
                self.reply = str(self.va[ch])
            elif op == 'VA':
                self.va[ch] = int(arg)
            elif op == 'AC':
                pass
            elif op == 'MD':
                self._advance(ch)
                self.reply = '0' if ch in self.target else '1'
            elif op == 'PA':
                self.target[ch] = int(arg)
            else:
                self.errors.append('6, COMMAND DOES NOT EXIST')

    def _advance(self, ch):
        if ch not in self.target:
            return
        t, p = self.target[ch], self.tp[ch]
        if self.stall_at is not None and p == self.stall_at:
            return
        step = max(-self.steps_per_poll, min(self.steps_per_poll, t - p))
        nxt = p + step
        if self.stall_at is not None and min(p, nxt) < self.stall_at < max(p, nxt):
            nxt = self.stall_at
        self.tp[ch] = nxt
        if nxt == t:
            del self.target[ch]

    def read(self):
        if self.reply is None:
            raise TimeoutError('fake: no reply pending')
        r, self.reply = self.reply, None
        return r

    def close(self):
        self.closed = True


def make_config(fake, lo=-10000, hi=10000):
    Picomotor8742Controller._instances.clear()
    ctrl = Picomotor8742Controller.shared('usb:103147')
    ctrl._open_transport = lambda: fake
    lim = AxisLimits(min=lo, max=hi, max_step=200, max_slew=1e9)
    return {'p1': {'driver': 'newport_8742', 'address': 'usb:103147', 'options': {'channel': 1},
                   'limits': lim},
            'p2': {'driver': 'newport_8742', 'address': 'USB:103147', 'options': {'channel': 2},
                   'limits': lim}}


def moves(fake):
    return [w for w in fake.writes if w[1:3] == 'PA']


def opened(fake, name='p1', **cfg_kw):
    axis = build_axis(name, make_config(fake, **cfg_kw), sleep=lambda s: None)
    axis._raw._sleep = lambda s: None
    axis.open()
    return axis


def test_address():
    assert parse_address('usb') is None and parse_address('USB:103147') == '103147'
    try:
        parse_address('COM3')
    except ValueError:
        pass
    else:
        raise AssertionError('accepted COM3')


def test_open_reads_only_and_close_never_moves():
    fake = FakeController()
    p1 = opened(fake)
    assert p1.position == 100 and p1.position_known
    assert not moves(fake) and 'AB' not in fake.writes
    p1.close()
    assert fake.closed and not moves(fake)


def test_no_motor_refused():
    fake = FakeController()
    p2 = build_axis('p2', make_config(fake))
    try:
        p2.open()
    except RuntimeError as e:
        assert 'no motor' in str(e)
    else:
        raise AssertionError('opened a channel without a motor')
    assert fake.closed


def test_ramped_move_lands_on_target():
    fake = FakeController()
    p1 = opened(fake)
    r = p1.move_to(600, who='test')
    assert moves(fake) == ['1PA267', '1PA433', '1PA600'], moves(fake)
    assert fake.tp[1] == 600 and r.commanded == 600 and r.readback is None
    p1.move_by(-150, who='test')
    assert fake.tp[1] == 450


def test_out_of_range_rejected():
    fake = FakeController()
    p1 = opened(fake, lo=-500, hi=500)
    for bad in (-501, 501, float('nan'), 1e9):
        try:
            p1.move_to(bad, who='test')
        except SafetyError:
            pass
        else:
            raise AssertionError(bad)
    assert not moves(fake)


def test_moved_elsewhere_refuses():
    fake = FakeController()
    p1 = opened(fake)
    p1.move_to(300, who='test')
    fake.tp[1] = 2000                       # jogged from the Picomotor Application
    n = len(fake.writes)
    try:
        p1.move_to(400, who='test')
    except RuntimeError as e:
        assert 'Picomotor Application' in str(e)
    else:
        raise AssertionError('moved from a stale count')
    assert not any(w[1:3] == 'PA' for w in fake.writes[n:])
    assert fake.tp[1] == 2000


def test_stuck_move_times_out_and_aborts():
    fake = FakeController(steps_per_poll=50)
    fake.stall_at = 150
    p1 = opened(fake)
    clock = [0.0]
    p1._raw._now = lambda: clock[0]
    p1._raw._sleep = lambda s: clock.__setitem__(0, clock[0] + 1.0)
    try:
        p1.move_by(100, who='test')
    except RuntimeError as e:
        assert 'did not finish' in str(e)
    else:
        raise AssertionError('no timeout')
    assert 'AB' in fake.writes and fake.tp[1] == 150 and p1._raw._last == 150


def test_estop_mid_ramp():
    fake = FakeController()
    p1 = opened(fake)
    calls = [0]

    def sleep(s):
        calls[0] += 1
        if calls[0] == 2:                   # after the second ramp piece
            threading.Thread(target=p1.estop, args=('test',)).start()
            threading.Event().wait(0.05)

    p1._sleep = sleep
    try:
        p1.move_to(1100, who='test')
    except SafetyError as e:
        assert 'E-stop' in str(e)
    else:
        raise AssertionError('E-stop ignored')
    assert 'AB' in fake.writes and fake.tp[1] < 1100
    try:
        p1.move_by(10, who='test')
    except SafetyError:
        pass
    else:
        raise AssertionError('moved while E-stopped')


def test_controller_error_raises():
    fake = FakeController()
    p1 = opened(fake)
    real_write = fake.write

    def write(cmd):
        real_write(cmd)
        if cmd.startswith('1PA'):
            fake.errors.append('10, VELOCITY OUT OF RANGE')
    fake.write = write
    try:
        p1.move_by(50, who='test')
    except RuntimeError as e:
        assert 'VELOCITY OUT OF RANGE' in str(e)
    else:
        raise AssertionError('controller error ignored')


def test_tiny_motor_velocity_capped():
    fake = FakeController(motor_types=(2, 0, 0, 0))
    opened(fake)
    assert fake.va[1] == 1750, fake.va


def test_not_an_8742():
    fake = FakeController()
    real_write = fake.write
    fake.write = lambda cmd: (real_write(cmd), setattr(fake, 'reply', 'HELLO')
                              if cmd == '*IDN?' else None)
    try:
        build_axis('p1', make_config(fake)).open()
    except RuntimeError as e:
        assert 'not an 8742' in str(e)
    else:
        raise AssertionError('accepted an unknown instrument')


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print(f'ok  {name}')
    print(f'{len(tests)} 8742 tests passed')


if __name__ == '__main__':
    main()
