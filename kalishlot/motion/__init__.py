"""Motorized optomechanics: vendor-neutral axes behind a safety layer.

    axis.py         raw Axis interface (drivers implement it; nobody else touches it)
    safety.py       SafeAxis: limits, ramps, freeze, E-stop, anti-backlash, move log
    virtual.py      VirtualAxis: linear combinations of axes as one DoF
    simulated.py    SimulatedAxis: backlash, random landing error, drift
    thorlabs_mdt693b.py  MDT693BAxis: one channel of a Thorlabs MDT693B piezo driver
    axes_config.py  versioned limits, tiers, virtual axes
    test_safety.py  hardware-free tests of every safety rule

Pure Python (no GUI), per kalishlot/ADDING_DEVICES.md. Design and decisions:
kalishlot/MECHANICAL_CONTROL.md.
"""

from motion.safety import AxisLimits, MoveLog, SafeAxis, SafetyError  # noqa: F401


def _driver_registry():
    def simulated(name, address, limits, **opt):
        from motion.simulated import SimulatedAxis
        return SimulatedAxis(name, **opt)

    def mdt693b(name, address, limits, **opt):
        from motion.thorlabs_mdt693b import MDT693BAxis
        opt.setdefault('max_allowed', limits.max)   # checked against the limit switch on open
        return MDT693BAxis(name, address, **opt)

    return {'simulated': simulated, 'thorlabs_mdt693b': mdt693b}


def build_axis(name, config=None, log=None, **safe_kwargs):
    """A SafeAxis for the configured axis `name` (not opened yet)."""
    if config is None:
        from motion.axes_config import AXES as config
    entry = config[name]
    raw = _driver_registry()[entry['driver']](name, entry['address'], entry['limits'],
                                              **entry.get('options', {}))
    return SafeAxis(raw, entry['limits'], log=log, **safe_kwargs)
