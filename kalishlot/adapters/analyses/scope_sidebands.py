"""Sidebands (NA) analysis for the PicoScope box.

6-Lorentzian sideband fit on the paused snapshot -> mode spacing, linewidths
and NA. Same math as the offline script
(pico_scope/mode_spacing_extraction_sidebands.py) via mode_analysis.

The host (the adapter) provides snapshot_region(args) and emit(event);
see kalishlot/ADDING_ANALYSES.md for the extension contract.
"""

import numpy as np

from .util import finite_or_none, na_interpolators, warm_na_interpolators
from pico_scope.mode_analysis import (DEFAULT_SIDEBAND_FREQ_MHZ,  # noqa: E402
                                      SIX_LORENTZIAN_PARAMS,
                                      fit_six_lorentzians, sideband_results,
                                      six_lorentzian_model)


class SidebandsAnalysis:
    COMMANDS = ('analyze_sidebands',)

    def __init__(self, host):
        self.host = host
        self._last = None  # last 'analysis_result' event, for reattach
        warm_na_interpolators()

    def describe_state(self):
        return {'analysis': self._last,
                'sideband_freq_default_mhz': DEFAULT_SIDEBAND_FREQ_MHZ}

    def reset(self):
        """Called on play: the result belongs to the discarded snapshot."""
        self._last = None

    def command(self, name, args):
        if name != 'analyze_sidebands':
            return None
        return self._analyze(args)

    def _analyze(self, args):
        channel = args['channel']
        x_fit, y_fit, t_min, t_max = self.host.snapshot_region(args)
        x0_guess, x1_guess = float(args['x0']), float(args['x1'])
        d_guess = abs(float(args['x_sb']) - x0_guess)
        if d_guess <= 0:
            raise ValueError('the sideband mark coincides with the 0th-order mark')
        f_sb_mhz = float(args.get('f_sb_mhz') or DEFAULT_SIDEBAND_FREQ_MHZ)

        try:
            params, errors = fit_six_lorentzians(
                x_fit, y_fit, x0_guess=x0_guess, x1_guess=x1_guess,
                d_guess=d_guess, region=(t_min, t_max))
        except RuntimeError as error:
            raise ValueError(f'fit did not converge: {error}')
        # pre-warmed at construction; a fit clicked mid-build waits for it
        na_interp, _, na_error = na_interpolators()
        results = sideband_results(params['x0'], params['x1'], params['d'],
                                   params['s0'], params['s1'], f_sb_mhz,
                                   na_interp=na_interp)
        if na_interp is not None and results['NA'] is None:
            na_error = (f"mode spacing {results['mode_spacing_MHz']:.1f} MHz "
                        f"is outside the simulated NA range")
        t_dense = np.linspace(t_min, t_max, 500)
        curve = six_lorentzian_model(
            t_dense, *(params[name] for name in SIX_LORENTZIAN_PARAMS))
        event = {'type': 'analysis_result',
                 'channel': channel,
                 'f_sb_mhz': f_sb_mhz,
                 'params': finite_or_none(params),
                 'errors': finite_or_none(errors),
                 'results': {**{key: (None if value is None else float(value))
                                for key, value in results.items()},
                             'na_error': na_error},
                 'curve': {'t': [round(float(v), 9) for v in t_dense],
                           'v': [round(float(v), 5) for v in curve]}}
        self._last = event
        self.host.emit(event)
        return {'ok': True, 'results': event['results']}
