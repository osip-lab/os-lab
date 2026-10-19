"""Motorized optomechanics: vendor-neutral axes behind a safety layer.

    axis.py         raw Axis interface (drivers implement it; nobody else touches it)
    safety.py       SafeAxis: limits, ramps, freeze, E-stop, anti-backlash, move log
    virtual.py      VirtualAxis: linear combinations of axes as one DoF
    simulated.py    SimulatedAxis: backlash, random landing error, drift
    axes_config.py  versioned limits, tiers, virtual axes
    test_safety.py  hardware-free tests of every safety rule

Pure Python (no GUI), per kalishlot/ADDING_DEVICES.md. Design and decisions:
kalishlot/MECHANICAL_CONTROL.md.
"""

from motion.safety import AxisLimits, MoveLog, SafeAxis, SafetyError  # noqa: F401


def _driver_registry():
    from motion.simulated import SimulatedAxis
    return {'simulated': lambda name, address, **opt: SimulatedAxis(name, **opt)}
    # real drivers are added here as they are written, e.g.
    # 'thorlabs_mdt693b': lambda name, address, **opt: MDT693BAxis(name, address, **opt)


def build_axis(name, config=None, log=None, **safe_kwargs):
    """A SafeAxis for the configured axis `name` (not opened yet)."""
    if config is None:
        from motion.axes_config import AXES as config
    entry = config[name]
    raw = _driver_registry()[entry['driver']](name, entry['address'], **entry.get('options', {}))
    return SafeAxis(raw, entry['limits'], log=log, **safe_kwargs)
