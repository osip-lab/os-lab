"""Hardware-free test of RigolDG's safety envelope (CH1: 5 Vpp, |offset| + amp/2 <= 2.5 V).

A fake VISA instrument stands in for the generator, so this runs anywhere:

    python rigol_fg/test_safety_limits.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rigol_dg import RigolDG  # noqa: E402


class FakeInstrument:
    """Answers the SCPI subset RigolDG uses, and records every write."""

    def __init__(self):
        self.state = {ch: {'VOLT': 3.0, 'OFFS': 0.0, 'FREQ': 0.05, 'FUNC': 'RAMP', 'OUTP': 'ON'}
                      for ch in (1, 2)}
        self.writes = []

    def write(self, cmd):
        self.writes.append(cmd)
        head, _, val = cmd.partition(' ')
        ch = int(head[5]) if head.startswith(':SOUR') else int(head[5]) if head.startswith(':OUTP') else None
        if head.endswith(':VOLT:OFFS'):
            self.state[ch]['OFFS'] = float(val)
        elif head.endswith(':VOLT'):
            self.state[ch]['VOLT'] = float(val)

    def query(self, cmd):
        ch = int(cmd[5])
        if cmd.endswith(':VOLT:OFFS?'):
            return str(self.state[ch]['OFFS'])
        if cmd.endswith(':VOLT?'):
            return str(self.state[ch]['VOLT'])
        raise KeyError(cmd)


def expect_refused(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError(f'{fn.__name__}{args} was not refused')


def main():
    gen = RigolDG('FAKE')
    gen._inst = inst = FakeInstrument()

    assert gen.set_amplitude(1, 5.0) == 5.0                 # at the limit: fine
    n = len(inst.writes)
    expect_refused(gen.set_amplitude, 1, 5.01)              # amplitude cap
    expect_refused(gen.set_offset, 1, 0.1)                  # 5 Vpp + any offset breaks the peak cap
    assert len(inst.writes) == n, 'a refused request reached the instrument'

    gen.set_amplitude(1, 3.0)
    assert gen.set_offset(1, 1.0) == 1.0                    # 1 + 1.5 = 2.5: fine
    expect_refused(gen.set_offset, 1, -1.01)
    expect_refused(gen.set_amplitude, 1, 3.1)               # 1 + 1.55 > 2.5
    assert abs(gen.max_amplitude(1) - 3.0) < 1e-9
    assert abs(gen.max_offset(1) - 1.0) < 1e-9

    assert gen.set_amplitude(2, 10.0) == 10.0               # CH2 has no limits
    assert gen.set_offset(2, 3.0) == 3.0

    free = RigolDG('FAKE', limits={})                       # explicit opt-out still possible
    free._inst = FakeInstrument()
    assert free.set_amplitude(1, 8.0) == 8.0
    print('rigol safety limits OK')


if __name__ == '__main__':
    main()
