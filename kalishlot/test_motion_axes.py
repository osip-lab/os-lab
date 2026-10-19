"""Standalone check of the motorized-axes adapter on the simulated axes.

Runs without the server, hardware or websockets (smoke_test.py calls run()
as check_motion_axes on machines that have its dependencies):

    python kalishlot/test_motion_axes.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters.motion_axes import MotionAxesAdapter  # noqa: E402


def _drain(listener):
    events = []
    while not listener.empty():
        events.append(listener.get_nowait())
    return events


def _refused(adapter, name, args):
    try:
        adapter.command(name, args)
    except ValueError as error:
        return str(error)
    raise AssertionError(f'{name} {args} was not refused')


def run():
    available = MotionAxesAdapter.list_available()
    assert any(a['address'] == 'simulated' for a in available), available

    adapter = MotionAxesAdapter('simulated')
    listener = adapter.add_listener()
    adapter.open()
    d = adapter.describe()
    names = [a['name'] for a in d['axes']]
    assert 'sim_mirror_phi' in names and all(a['position_known'] for a in d['axes']), d
    tilt = next(a for a in d['axes'] if a['name'] == 'sim_mirror_phi')
    start = tilt['position']

    r = adapter.command('move_by', {'axis': 'sim_mirror_phi', 'delta': 0.5, 'wait': True})
    assert abs(r['result']['commanded'] - (start + 0.5)) < 1e-9, r

    # out of range, unknown axis, frozen: HTTP 400 (ValueError), nothing moves
    _refused(adapter, 'move_to', {'axis': 'sim_mirror_phi', 'target': 1e6})
    _refused(adapter, 'move_to', {'axis': 'nope', 'target': 1})
    adapter.command('freeze', {'axis': 'sim_mirror_phi', 'frozen': True})
    _refused(adapter, 'move_by', {'axis': 'sim_mirror_phi', 'delta': 0.1})
    adapter.command('freeze', {'axis': 'sim_mirror_phi', 'frozen': False})

    # background move: returns at once, emits axis events, ends at the target
    _drain(listener)
    r = adapter.command('move_by', {'axis': 'sim_mirror_phi', 'delta': 2.0})
    assert r.get('started'), r
    _refused(adapter, 'move_by', {'axis': 'sim_mirror_phi', 'delta': 0.1})   # busy
    deadline = time.time() + 5
    while adapter._moving('sim_mirror_phi') and time.time() < deadline:
        time.sleep(0.02)
    states = [e['state'] for e in _drain(listener) if e['type'] == 'axis']
    assert states and states[0]['moving'] and not states[-1]['moving'], states
    assert abs(states[-1]['commanded'] - (start + 2.5)) < 1e-9, states[-1]

    # E-stop holds and refuses until reset
    adapter.command('estop', {})
    _refused(adapter, 'move_by', {'axis': 'sim_mirror_phi', 'delta': -0.1})
    adapter.command('reset_estop', {})
    adapter.command('move_to', {'axis': 'sim_mirror_phi', 'target': start, 'wait': True})

    raw = {n: a._raw for n, a in adapter.axes.items()}
    moves_before = {n: r.moves for n, r in raw.items()}
    adapter.close()
    assert {n: r.moves for n, r in raw.items()} == moves_before, 'close() moved an axis'
    print('motion axes adapter ok (moves, refusals, background move, E-stop, close holds)')


if __name__ == '__main__':
    run()
