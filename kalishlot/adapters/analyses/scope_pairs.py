"""Pairs (df/FSR) analysis for the PicoScope box.

One double-Lorentzian fit per marked pair; the fitted pairs accumulate and
the df/FSR results (needing >= 2 pairs) are re-broadcast after every change.
Same math as the offline script
(pico_scope/extract_df_and_fsr_from_scope_csv.py) via mode_analysis.

The host (the adapter) provides snapshot_region(args) and emit(event);
see kalishlot/ADDING_ANALYSES.md for the extension contract.
"""

import numpy as np

from .util import finite_or_none, na_interpolators, warm_na_interpolators
from pico_scope.mode_analysis import (DOUBLE_LORENTZIAN_PARAMS,  # noqa: E402
                                      cavity_fsr_mhz, double_lorentzian,
                                      fit_lorentzian_pair,
                                      pair_positions_results, pair_summary)


class PairsAnalysis:
    COMMANDS = ('fit_pair', 'undo_pair', 'clear_pairs')

    def __init__(self, host):
        self.host = host
        self._pairs = []  # fitted Lorentzian pairs on the current snapshot
        self._last = None  # last 'analysis_pairs' event, for reattach
        warm_na_interpolators()

    def describe_state(self):
        return {'analysis_pairs': self._last}

    def reset(self):
        """Called on play: the pairs belong to the discarded snapshot."""
        self._pairs = []
        self._last = None

    def command(self, name, args):
        if name == 'fit_pair':
            return self._fit_pair(args)
        if name == 'undo_pair':
            if self._pairs:
                self._pairs.pop()
            return self._broadcast()
        if name == 'clear_pairs':
            self._pairs = []
            return self._broadcast()
        return None

    def _fit_pair(self, args):
        channel = args['channel']
        x_fit, y_fit, t_min, t_max = self.host.snapshot_region(args)
        x1_guess, x2_guess = float(args['x1']), float(args['x2'])
        if x1_guess == x2_guess:
            raise ValueError('the two peak marks coincide')
        try:
            params, errors = fit_lorentzian_pair(
                x_fit, y_fit, x1_guess=x1_guess, x2_guess=x2_guess,
                region=(t_min, t_max))
        except RuntimeError as error:
            raise ValueError(f'fit did not converge: {error}')
        t_dense = np.linspace(t_min, t_max, 300)
        curve = double_lorentzian(
            t_dense, *(params[name] for name in DOUBLE_LORENTZIAN_PARAMS))
        self._pairs.append(
            {'channel': channel,
             'x01': float(params['x01']), 'x02': float(params['x02']),
             'params': finite_or_none(params),
             'errors': finite_or_none(errors),
             'curve': {'t': [round(float(v), 9) for v in t_dense],
                       'v': [round(float(v), 5) for v in curve]}})
        return self._broadcast()

    def _broadcast(self):
        """Recompute the df/FSR results from all fitted pairs and broadcast
        the full pairs state (every viewer draws the same overlay)."""
        results = None
        if len(self._pairs) >= 2:
            # normalized scan order: pairs sorted left to right, each pair's
            # first peak the leftmost — the pair-to-pair spacing (the FSR)
            # then always relates like peaks
            positions = sorted((min(pair['x01'], pair['x02']),
                                max(pair['x01'], pair['x02']))
                               for pair in self._pairs)
            _, na_over_fsr_interp, na_error = na_interpolators()
            try:
                rows = pair_positions_results(
                    positions, fsr_mhz=cavity_fsr_mhz(),
                    na_over_fsr_interp=na_over_fsr_interp)
                summary = pair_summary(rows)
                if na_over_fsr_interp is not None and summary['NA_mean'] is None:
                    na_error = ('the fitted df/FSR is outside the simulated '
                                'NA range')
                results = {'rows': rows, **summary,
                           'fsr_mhz': cavity_fsr_mhz(), 'na_error': na_error}
            except ValueError as error:  # e.g. two pairs at the same position
                results = {'error': str(error)}
        event = {'type': 'analysis_pairs', 'pairs': self._pairs,
                 'results': results}
        self._last = event
        self.host.emit(event)
        return {'ok': True, 'n_pairs': len(self._pairs), 'results': results}
