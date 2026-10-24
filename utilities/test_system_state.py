"""Tests of utilities/system_state.py.

    python utilities/test_system_state.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utilities.system_state import SystemState  # noqa: E402


def lab_state():
    s = SystemState(note='before walking the lens')
    s.set('large_mirror_piezo.x_v', 37.5)
    s.set('large_mirror_piezo.y_v', 41.0)
    s.update('lens_stage', encoder_position_um=-12.3, steps_from_home=4120)
    s.update('ximea', exposure_us=2000, frame_rate_hz=14.28, gain_db=0.0,
             roi={'offset_x': 366, 'offset_y': 544, 'width': 432, 'height': 400})
    s['picoscope'] = {'sample_rate_hz': 1e6, 'window_s': 2.0,
                      'channels': {'A': {'enabled': True, 'range_v': 2.0, 'coupling': 'DC'}}}
    s.update('rigol_dg', ch1={'shape': 'RAMP', 'frequency_hz': 0.05, 'amplitude_vpp': 3.0, 'output': True})
    s.set('manual.long_arm_ruler_mm', 451.5)
    return s


def test_set_get_and_missing():
    s = lab_state()
    assert s.get('ximea.exposure_us') == 2000
    assert s['picoscope.channels.A.range_v'] == 2.0
    assert s.get('ximea.roi') == {'offset_x': 366, 'offset_y': 544, 'width': 432, 'height': 400}
    assert s.get('basler.exposure_us') is None and s.get('basler.exposure_us', -1) == -1
    assert 'manual.long_arm_ruler_mm' in s and 'manual.other' not in s
    try:
        s['basler.exposure_us']
        raise AssertionError('expected KeyError')
    except KeyError:
        pass
    got = s.get('ximea.roi')
    got['width'] = 1                                    # a copy: the record does not change
    assert s.get('ximea.roi.width') == 432


def test_round_trip_json():
    s = lab_state()
    with tempfile.TemporaryDirectory() as d:
        path = s.save(Path(d) / 'sub' / 'state.json')  # creates the folder
        back = SystemState.load(path)
        raw = json.loads(path.read_text(encoding='utf-8'))
    assert back == s and back.note == s.note and back.timestamp == s.timestamp
    assert raw['format'] == 'os-lab-system-state' and raw['values']['manual']['long_arm_ruler_mm'] == 451.5


def test_partial_record():
    """Camera off: nothing about it is saved, and loading does not invent it."""
    s = lab_state()
    s.remove('ximea')
    with tempfile.TemporaryDirectory() as d:
        back = SystemState.load(s.save(Path(d) / 'no_camera.json'))
    assert 'ximea' not in back.sections() and back.get('ximea.exposure_us') is None
    assert back.get('picoscope.sample_rate_hz') == 1e6


def test_new_parameters_need_no_code():
    s = SystemState()
    s.set('new_device.some_setting.deep.value_nm', 3)
    s.update('another', anything=[1, 2, 3], label='x')
    assert s.flat() == {'new_device.some_setting.deep.value_nm': 3, 'another.anything': [1, 2, 3],
                        'another.label': 'x'}


def test_update_merges_nested():
    s = lab_state()
    s.update('rigol_dg', ch1={'frequency_hz': 0.1})          # only the frequency changes
    assert s.get('rigol_dg.ch1') == {'shape': 'RAMP', 'frequency_hz': 0.1, 'amplitude_vpp': 3.0, 'output': True}


def test_diff_merge_subset():
    a = lab_state()
    b = SystemState.from_dict(a.to_dict())
    b.set('large_mirror_piezo.x_v', 38.0)
    b.remove('manual')
    b.set('basler.exposure_us', 500)
    assert a.diff(b) == {'basler.exposure_us': (None, 500), 'large_mirror_piezo.x_v': (37.5, 38.0),
                         'manual.long_arm_ruler_mm': (451.5, None)}
    camera_off = a.subset('picoscope', 'large_mirror_piezo')
    assert camera_off.sections() == ['picoscope', 'large_mirror_piezo']
    later = SystemState({'ximea': {'exposure_us': 3000}})
    both = camera_off.merged(later)
    assert both.get('ximea.exposure_us') == 3000 and both.get('picoscope.window_s') == 2.0
    assert camera_off.get('ximea') is None                   # merged() returns a new record


def test_values_checked_when_set():
    s = SystemState()
    try:
        s.set('x.y', object())
        raise AssertionError('expected TypeError')
    except TypeError:
        pass
    try:
        import numpy as np
    except ImportError:
        return
    s.set('scope.trace_v', np.array([0.1, 0.2]))
    s.set('scope.rate_hz', np.float64(1e6))
    s.set('cam.count', np.int64(7))
    assert s.get('scope.trace_v') == [0.1, 0.2] and type(s.get('scope.rate_hz')) is float
    assert type(s.get('cam.count')) is int
    json.dumps(s.to_dict())


def test_dotted_keys_and_section_conflict():
    s = SystemState()
    s.set(('ximea', 'lens 0.5x'), True)
    assert s.get(('ximea', 'lens 0.5x')) is True and s.sections() == ['ximea']
    s.set('a.b', 1)
    try:
        s.set('a.b.c', 2)                                    # a.b is a value, not a section
        raise AssertionError('expected KeyError')
    except KeyError:
        pass


def test_reads_plain_device_snapshots():
    """kalishlot's device_state.json: {device: settings snapshot}, no wrapper."""
    plain = {'picoscope': {'sample_rate_hz': 1e6}, 'ximea': {'exposure_us': 2000, 'markers': []}}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'device_state.json'
        p.write_text(json.dumps(plain))
        s = SystemState.load(p)
    assert s.get('ximea.exposure_us') == 2000 and s.sections() == ['picoscope', 'ximea']


def test_kalishlot_device_state_like_the_server():
    """What kalishlot/server.py does with device_state.json: device ids type:address
    (dots possible), record only on change, restore on open; old plain files load."""
    from utilities.system_state import to_plain
    rigol = 'rigol_dg:TCPIP0::192.168.1.5::INSTR'
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'device_state.json'
        path.write_text(json.dumps({'picoscope:JY123': {'sample_rate_hz': 1e6, 'pair': [1, 2]}}))  # old format
        saved = SystemState.load(path)
        assert saved.get(('picoscope:JY123',)) == {'sample_rate_hz': 1e6, 'pair': [1, 2]}
        snapshot = to_plain({'sample_rate_hz': 1e6, 'pair': (1, 2)})      # a tuple: equal after to_plain
        assert saved.get(('picoscope:JY123',)) == snapshot                 # unchanged: the server skips the write
        saved.set((rigol,), {'ch1': {'frequency_hz': 0.05}})
        saved.stamp().save(path)
        back = SystemState.load(path)
    assert back.get((rigol,)) == {'ch1': {'frequency_hz': 0.05}}
    assert back.sections() == ['picoscope:JY123', rigol]
    assert back.get(rigol) is None                       # a dotted string splits: use the tuple


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print(f'ok  {name}')
    print(f'{len(tests)} system state tests passed')


if __name__ == '__main__':
    main()
