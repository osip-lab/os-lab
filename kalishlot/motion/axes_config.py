"""Versioned axis configuration: drivers, safety limits, tiers, virtual axes.

This file is the single source of truth for what may move and how far. It is
deliberately NOT editable from the kalishlot UI: change it here, commit it,
and the change is reviewable. (The UI can only freeze/unfreeze at run time,
and cannot unfreeze an axis frozen here.)

Real axes so far: the large mirror's two tilts on a Thorlabs MDT693B. Other
candidates (Standa positioners, PI E-872 PiezoMike, Thorlabs Kinesis) get a
driver when they are used. The simulated axes stay for the hardware-free tests.

Limits to MEASURE in the lab before trusting a real axis (see the
characterization script in phase 6): repeatability_sigma, settle_s,
vibration_settle_s, backlash (-> approach_overshoot), safe range.
"""

from motion.safety import AxisLimits

# name -> {'driver': registry key, 'address': driver-specific, 'options': driver kwargs,
#          'limits': AxisLimits}
AXES = {
    'sim_mirror_phi': {
        'driver': 'simulated', 'address': 'sim:0',
        'options': dict(units='V', kind='voltage', start=37.5, has_readback=True,
                        backlash=0.6, repeatability_sigma=0.05, drift_per_s=0.0005),
        'limits': AxisLimits(min=0.0, max=75.0, max_step=1.0, max_slew=10.0, settle_s=0.02,
                             vibration_settle_s=0.5, repeatability_sigma=0.05,
                             approach=+1, approach_overshoot=1.0),
    },
    'sim_mirror_theta': {
        'driver': 'simulated', 'address': 'sim:1',
        'options': dict(units='V', kind='voltage', start=37.5, has_readback=True,
                        backlash=0.6, repeatability_sigma=0.05, drift_per_s=-0.0003),
        'limits': AxisLimits(min=0.0, max=75.0, max_step=1.0, max_slew=10.0, settle_s=0.02,
                             vibration_settle_s=0.5, repeatability_sigma=0.05,
                             approach=+1, approach_overshoot=1.0),
    },
    'sim_lens_x': {
        'driver': 'simulated', 'address': 'sim:2',
        'options': dict(units='um', kind='position', start=0.0, closed_loop=True,
                        repeatability_sigma=0.3, sensor_noise=0.05),
        'limits': AxisLimits(min=-500.0, max=500.0, max_step=20.0, max_slew=100.0,
                             settle_s=0.05, vibration_settle_s=1.0, repeatability_sigma=0.3),
    },
    'sim_lens_y': {     # the lens across the beam: with sim_mirror_phi a narrow ridge (cavity_physics)
        'driver': 'simulated', 'address': 'sim:3',
        'options': dict(units='um', kind='position', start=0.0, closed_loop=True,
                        repeatability_sigma=0.3, sensor_noise=0.05),
        'limits': AxisLimits(min=-500.0, max=500.0, max_step=20.0, max_slew=100.0,
                             settle_s=0.05, vibration_settle_s=1.0, repeatability_sigma=0.3),
    },
    # Large mirror in a Thorlabs POLARIS-K1S3P (3 piezo actuators), driven by a Thorlabs
    # MDT693B (s/n 2408096171-09, USB virtual COM port, limit switch at 150 V).
    # Names follow cavity_design: phi = xy-plane (horizontal) tilt, theta = xz-plane
    # (vertical) tilt. The controller's Z output is not used.
    # PLACEHOLDER limits (2026-10-19): range kept to half the 150 V the piezos take until the
    # tilt per volt is measured; approach off until hysteresis is characterized; settle and
    # repeatability still to be measured. No settle delays: piezos do not shake the setup
    # (lab, 2026-10-19), unlike motors.
    'large_mirror_phi': {
        'driver': 'thorlabs_mdt693b', 'address': 'COM4', 'options': {'channel': 'x'},
        'limits': AxisLimits(min=0.0, max=75.0, max_step=0.5, max_slew=5.0, settle_s=0.0,
                             vibration_settle_s=0.0),
    },
    'large_mirror_theta': {
        'driver': 'thorlabs_mdt693b', 'address': 'COM4', 'options': {'channel': 'y'},
        'limits': AxisLimits(min=0.0, max=75.0, max_step=0.5, max_slew=5.0, settle_s=0.0,
                             vibration_settle_s=0.0),
    },
}

# Optimization order: the loop works on TIERS[0] alone, then TIERS[0] + TIERS[1], ...
TIERS = [['sim_mirror_phi', 'sim_mirror_theta'], ['sim_lens_x']]

# name -> [(axis name, coefficient), ...]
VIRTUAL_AXES = {
    # 'beam_walk_y': [('m1_phi', 1.0), ('m2_phi', -1.0)],
}
