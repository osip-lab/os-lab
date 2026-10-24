"""Nested alignment: step the outer axes, re-optimize the inner ones at every step.

    levels = [Level(['sim_mirror_phi'], Objective('intensity')),
              Level(['sim_lens_y'], Objective('intensity'))]
    summary = NestedLoop(axes, measure, levels).run()

Level 0 is the outermost. For every trial position of level k's axes, the levels
below are optimized again (recursively; the innermost is a plain AlignmentLoop)
and only then is level k's objective measured:

    theta_k* = argmax_k  F_k(theta_k, theta_below*(theta_k))

Two uses (kalishlot/MECHANICAL_CONTROL.md, section 3.e):
- Same objective at every level: a narrow ridge. Almost every direction drops
  the coupling sharply and only one combination of axes gains slowly, so a
  joint search steps off the crest every time and concludes it is at the top.
  Re-peaking the stiff axes (inner) after each step of the soft ones (outer)
  turns the ridge into a smooth profile. The inner axes must be the stiff ones:
  an inner problem without a sharp optimum makes the outer profile noise.
- Different objectives: maximize the coupling while KEEPING the mode symmetric.
  The inner level gets e.g. Objective('split_linewidths', maximize=False,
  good_enough=1.0): it does nothing while the mode is still symmetric enough,
  and restores it when an outer step broke it. Maximizing a metric that already
  sits at its (flat) peak would only chase noise.

Predictor. After each outer trial the optimized inner positions are known, so
d(theta_below*)/d(theta_k) is learned as a matrix K (least squares over the
last predictor_window trials, accepted or not). The next trial starts the levels below at
theta_below* + K delta. On a ridge, (delta, K delta) is the ridge direction
(implicit function theorem: K -> -H_bb^-1 H_bk), so the inner runs shrink to
small corrections; that is where the evaluations are saved.

Outer search per level: the coordinate search of AlignmentLoop on the profile
(+step, then -step; x1.5 on success, x0.5 on failure; never below 3 sigma of
the repeatability), a parabola through (-step, 0, +step) when both sides fail,
and a joint step along the profile gradient when the level has >= 2 axes.

Noise. The noise of every level's objective comes from noise_repeats captures
in place at the start. A level's value at a point is the mean over the last
capture of the levels below (taken there) plus outer_repeats more, not the
inner's best capture (that one is biased upwards). Accepting needs
accept_sigmas x that level's noise; large jumps are confirmed in place; the
incumbent is re-measured every remeasure_every trials.

Profile noise (profile_noise=True, the default). The levels below never end
exactly at their optimum, and where they end differs from run to run, so a
level's value at one fixed position scatters more than the measurement noise
alone. Without counting that, a lucky run of the levels below looks like an
improvement; on a flat profile every step can, and the level never stops. So
the first time a level is optimized, the levels below are solved a second time
at the same position, from a start shifted by one step on each of their axes;
the difference d of the two values gives the profile noise |d| / sqrt(2), and
the level's noise (margin, jump confirmation) is the larger of the two. It
costs one more solve of the levels below per level, once per run. A single
difference is a rough estimate; it only ever raises the margin.

Budget, stop() and abort are shared by all levels: one evaluation budget, one
history (every capture, tagged with the level that asked for it), and the best
point is the full position (all levels' axes). Every capture waits for ALL
axes to settle, since the outer ones moved too.
"""

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from alignment.loop import AlignmentLoop, Evaluation, Objective, _check, _default_step, _pos, _sigma


@dataclass
class Level:
    axes: list
    objective: object = field(default_factory=Objective)
    steps: dict = field(default_factory=dict)    # initial step per axis (default: as AlignmentLoop)
    max_evaluations: int | None = None           # per optimization of this level (captures, levels below included)


class _Stopped(Exception):
    pass


class NestedLoop:
    def __init__(self, axes, measure, levels, outer_repeats=1, accept_sigmas=1.0, min_step_sigmas=3.0,
                 noise_repeats=4, remeasure_every=6, max_evaluations=400, confirm_jump=10.0,
                 predictor=True, predictor_window=10, inner_stall_passes=1, profile_noise=True,
                 on_abort='best', log_dir=None,
                 who='nested alignment', print_fn=print):
        self.axes = dict(axes)
        self.measure_fn = measure
        self.levels = [lv if isinstance(lv, Level) else Level(*lv) for lv in levels]
        seen = set()
        for lv in self.levels:
            for n in lv.axes:
                if n not in self.axes:
                    raise ValueError(f'level axis {n!r} is not among the axes')
                if n in seen:
                    raise ValueError(f'axis {n!r} is in two levels')
                seen.add(n)
        self.outer_repeats = outer_repeats
        self.accept_sigmas = accept_sigmas
        self.min_step_sigmas = min_step_sigmas
        self.noise_repeats = noise_repeats
        self.remeasure_every = remeasure_every
        self.max_evaluations = max_evaluations
        self.confirm_jump = confirm_jump
        self.predictor = predictor
        self.window = predictor_window      # outer trials the predictor is fitted to
        self._pairs = {}                    # level -> [(outer move, shift of the optimum below)]
        self.inner_stall_passes = inner_stall_passes   # warm inner runs: one failed pass is enough
        self.profile_noise = profile_noise             # also count the scatter of the levels below
        self.profile_sigma = {}                        # level -> estimated profile noise
        self.on_abort = on_abort
        self.who = who
        self.print = print_fn or (lambda *a, **k: None)
        self.history = []                 # every capture: dict(index, level, kind, positions, metrics)
        self.noise = [0.0] * len(self.levels)
        self.best = None                  # Evaluation of level 0 at the best full position
        self.K = {}                       # level -> (below axis names, matrix rows=below, cols=level axes)
        self.profile = {k: [] for k in range(len(self.levels))}   # level -> outer trials
        self._steps = {}
        self._min_steps = {n: max(min_step_sigmas * _sigma(a), 1e-12) for n, a in self.axes.items()}
        for lv in self.levels:
            for n in lv.axes:
                self._steps[n] = lv.steps.get(n, _default_step(self.axes[n]))
        self._last_cm = None
        self._last_pos = None
        self._inner = None
        self._stop = threading.Event()
        self._log_path = None
        if log_dir:
            d = Path(log_dir)
            d.mkdir(parents=True, exist_ok=True)
            self._log_path = d / 'nested_evaluations.jsonl'

    # ------------------------------------------------------------ plumbing
    def stop(self):
        self._stop.set()
        if self._inner is not None:
            self._inner.stop()

    def positions(self):
        return {n: _pos(a) for n, a in self.axes.items()}

    def _below(self, k):
        return [n for lv in self.levels[k + 1:] for n in lv.axes]

    def _free(self, names):
        return [n for n in names if not self.axes[n].frozen]

    def _budget(self):
        return self.max_evaluations - len(self.history)

    def _check_running(self):
        if self._stop.is_set():
            raise _Stopped()
        if self._budget() <= 0:
            raise _Stopped('evaluation budget used up')

    def _capture(self, level, kind):
        """Settle every axis, capture once, record it. Returns CaptureMetrics.
        The shared budget is a hard limit here (AlignmentLoop's final verify
        capture does not check its own)."""
        if self._budget() <= 0:
            raise _Stopped('evaluation budget used up')
        for a in self.axes.values():
            a.wait_until_settled()
        cm = self.measure_fn()
        pos = self.positions()
        self._last_cm, self._last_pos = cm, pos
        rec = {'index': len(self.history), 'level': level, 'kind': kind, 'positions': pos,
               'metrics': {k: (m.value if m.valid else None) for k, m in cm.metrics.items()},
               'time': time.time()}
        self.history.append(rec)
        if self._log_path:
            with self._log_path.open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(rec, default=str) + '\n')
        return cm

    def _move_to(self, target):
        for n, t in target.items():
            if abs(_pos(self.axes[n]) - t) > 1e-12:
                self.axes[n].move_to(t, who=self.who)

    def _allowed(self, target):
        try:
            for n, t in target.items():
                _check(self.axes[n], t)
        except ValueError:
            return False
        return True

    def _score_here(self, k, kind, extra=None):
        """Level k's objective at the current position: the last capture if it was
        taken here, plus outer_repeats (or `extra`) more. Returns an Evaluation."""
        obj = self.levels[k].objective
        pos = self.positions()
        cms = []
        if self._last_pos is not None and all(abs(self._last_pos[n] - pos[n]) < 1e-12 for n in pos):
            cms.append(self._last_cm)
        want = len(cms) + (self.outer_repeats if extra is None else extra)
        while len(cms) < max(want, 1):
            self._check_running()
            cms.append(self._capture(k, kind))
        results = [obj.evaluate(cm) for cm in cms]
        scores = [r[0] for r in results]
        metrics = {}
        for cm in cms:
            for name, m in cm.metrics.items():
                if m.valid:
                    metrics.setdefault(name, []).append(m.value)
        metrics = {name: sum(v) / len(v) for name, v in metrics.items()}
        if any(sc is None for sc in scores):
            reason = next(r[2] for r in results if r[0] is None)
            return Evaluation(len(self.history), pos, None, False, reason, metrics, kind)
        feasible = all(r[1] for r in results)
        reason = '; '.join(sorted({r[2] for r in results if r[2]}))
        ev = Evaluation(len(self.history), pos, sum(scores) / len(scores), feasible, reason, metrics, kind)
        ev.n = len(scores)
        return ev

    # --------------------------------------------------------------- levels
    def _optimize(self, k):
        """Optimize levels k.. with the axes above k fixed; leave the axes at the
        result. Raises _Stopped when stopped or out of budget."""
        self._check_running()
        if k == len(self.levels) - 1:
            return self._innermost(k)
        return self._profile_search(k)

    def _innermost(self, k):
        lv = self.levels[k]
        names = self._free(lv.axes)
        if not names:
            return
        budget = self._budget()
        if lv.max_evaluations is not None:
            budget = min(budget, lv.max_evaluations)
        steps = {n: self._steps[n] for n in names}
        loop = AlignmentLoop({n: self.axes[n] for n in names}, lambda: self._capture(k, 'inner'),
                             lv.objective, tiers=[names], steps=steps,
                             accept_sigmas=self.accept_sigmas, min_step_sigmas=self.min_step_sigmas,
                             max_evaluations=max(budget, 1), on_abort='stay', known_noise=self.noise[k],
                             stall_passes=self.inner_stall_passes,
                             who=self.who, print_fn=None)
        self._inner = loop
        try:
            if self._stop.is_set():
                raise _Stopped()
            loop.run()
        finally:
            self._inner = None
        for n in names:                           # warm start for the next call, not below the minimum
            self._steps[n] = max(min(loop.steps[n], steps[n] * 2), 2 * self._min_steps[n])
        self._check_running()

    def _predict(self, k, delta):
        """Start positions of the levels below k for an outer move `delta` (dict)."""
        below = self._free(self._below(k))
        x0 = self.best_at[k].positions
        if not self.predictor or k not in self.K or not below:
            return {}
        names_b, cols, K = self.K[k]
        d = np.array([delta.get(n, 0.0) for n in cols])
        pred = K @ d
        out = {}
        for i, n in enumerate(names_b):
            if n in below and pred[i] != 0:
                t = x0[n] + float(pred[i])
                if self._allowed({n: t}):
                    out[n] = t
        return out

    def _learn(self, k, delta, x0, x1):
        """Refit K from one more outer trial: delta on level k's axes moved the
        optimum of the axes below from x0 to x1. K is the least-squares fit
        D K^T = Y over the last predictor_window trials: every single trial's
        shift is noisy (the levels below end only near their optimum), and a
        fit over many trials averages that out, where an update that follows
        the latest trials does not (2026-10-24: 0.48 vs 0.67, true 0.5)."""
        below = self._free(self._below(k))
        cols = self._free(self.levels[k].axes)
        if not below or not cols:
            return
        if k not in self.K:
            self.K[k] = (below, cols, np.zeros((len(below), len(cols))))
            self._pairs[k] = []
        names_b, cols, K = self.K[k]
        d = np.array([delta.get(n, 0.0) for n in cols])
        if d @ d == 0:
            return
        self._pairs[k] = (self._pairs[k] + [(d, np.array([x1[n] - x0[n] for n in names_b]))])[-self.window:]
        D = np.array([a for a, _ in self._pairs[k]])
        Y = np.array([b for _, b in self._pairs[k]])
        K[:] = np.linalg.lstsq(D, Y, rcond=None)[0].T

    def _trial(self, k, delta, kind='trial'):
        """Move level k's axes by delta (predicted moves below), optimize below,
        score level k. Keep it if better; else return to the incumbent."""
        inc = self.best_at[k]
        x0 = dict(inc.positions)
        target = {n: x0[n] + dv for n, dv in delta.items()}
        if not self._allowed(target):
            return None, False
        n_before = len(self.history)
        self._move_to({**target, **self._predict(k, delta)})
        try:
            self._optimize(k + 1)
        except RuntimeError as e:                 # the levels below lost the signal there
            self.print(f'{"  " * k}  level {k} trial lost the signal ({e}); back')
            self._move_to(x0)
            return None, False
        x1 = self.positions()
        ev = self._confirmed(k, self._score_here(k, kind))
        self._learn(k, delta, x0, x1)
        accepted = ev.better_than(inc, self.margin(k))
        self.profile[k].append({'delta': dict(delta), 'kind': kind, 'score': ev.score, 'feasible': ev.feasible,
                                'accepted': accepted, 'evaluations': len(self.history) - n_before,
                                'start': x0, 'positions': x1, 'metrics': ev.metrics})
        self._trials_since[k] += 1
        if accepted:
            self.best_at[k] = ev
            return ev, True
        self._move_to(x0)
        if self._trials_since[k] >= self.remeasure_every:
            self._remeasure(k)
        return ev, False

    def level_noise(self, k):
        """Level k's noise: the measurement noise, or the profile noise when larger."""
        return max(self.noise[k], self.profile_sigma.get(k, 0.0))

    def margin(self, k):
        return self.accept_sigmas * self.level_noise(k)

    def _confirmed(self, k, ev):
        inc = self.best_at[k]
        if ev.score is None or inc.score is None or self._budget() <= 0:
            return ev
        jump = ev.score - inc.score
        if jump <= max(self.confirm_jump * self.level_noise(k), 0.05 * abs(inc.score)):
            return ev
        again = self._score_here(k, 'confirm', extra=1 + self.outer_repeats)
        if again.score is None:
            return again
        n1, n2 = getattr(ev, 'n', 1), getattr(again, 'n', 1)
        again.score = (ev.score * n1 + again.score * n2) / (n1 + n2)
        again.feasible = ev.feasible and again.feasible
        return again

    def _remeasure(self, k):
        """Drift guard: score the incumbent again in place (the axes are there)."""
        self._trials_since[k] = 0
        ev = self._score_here(k, 'remeasure', extra=1 + self.outer_repeats)
        if ev.score is not None:
            self.best_at[k] = ev

    def _estimate_profile_noise(self, k):
        """Solve the levels below again at the same level-k position, from a start
        shifted by one step on each of their axes; keep the better of the two."""
        first = self.best_at[k]
        shifted = {}
        for n in self._free(self._below(k)):
            for sign in (+1, -1):
                t = first.positions[n] + sign * self._steps[n]
                if self._allowed({n: t}):
                    shifted[n] = t
                    break
        if not shifted:
            return
        self._move_to(shifted)
        try:
            self._optimize(k + 1)
        except RuntimeError:                      # lost the signal from the shifted start
            self._move_to(first.positions)
            return
        second = self._score_here(k, 'profile noise')
        if second.score is None:
            self._move_to(first.positions)
            return
        self.profile_sigma[k] = abs(second.score - first.score) / math.sqrt(2)
        self.print(f'{"  " * k}  level {k}: profile noise {self.profile_sigma[k]:.2g} '
                   f'(measurement noise {self.noise[k]:.2g})')
        if second.better_than(first, 0.0):
            self.best_at[k] = second
        else:
            self._move_to(first.positions)

    def _profile_search(self, k):
        ind = '  ' * k
        names = self._free(self.levels[k].axes)
        self._optimize(k + 1)                     # the levels below, here
        self.best_at[k] = self._score_here(k, 'start')
        self._trials_since[k] = 0
        if self.best_at[k].score is None:
            raise RuntimeError(f'signal lost at level {k}: {self.best_at[k].reason}')
        self.print(f'{ind}level {k} ({", ".join(names)}): start {self.best_at[k].score:.4g}')
        if self.profile_noise and k not in self.profile_sigma:
            self._estimate_profile_noise(k)
        stall = 0
        n0 = len(self.history)
        cap = self.levels[k].max_evaluations
        while stall < 2 and names:
            if all(self._steps[n] < self._min_steps[n] for n in names):
                break
            if cap is not None and len(self.history) - n0 >= cap:
                break
            improved = False
            grad = {}
            for n in names:
                step = self._steps[n]
                if step < self._min_steps[n]:
                    continue
                f0 = self.best_at[k].score
                sides = {}
                for sign in (+1, -1):
                    ev, ok = self._trial(k, {n: sign * step})
                    sides[sign] = ev
                    if ok:
                        self._steps[n] *= 1.5
                        improved = True
                        break
                else:
                    if all(sides[s] is not None and sides[s].score is not None for s in sides):
                        fp, fm = sides[1].score, sides[-1].score
                        grad[n] = (fp - fm) / (2 * step)
                        curv = fp - 2 * f0 + fm
                        if curv < 0:                       # parabola vertex inside the bracket
                            off = step * (fm - fp) / (2 * curv)
                            if self._min_steps[n] <= abs(off) < step:
                                ev, ok = self._trial(k, {n: off}, kind='parabola')
                                if ok:
                                    improved = True
                                    self._steps[n] = max(abs(off), self._min_steps[n] * 1.01)
                                    continue
                    self._steps[n] *= 0.5
            if len(grad) >= 2:
                norm = math.sqrt(sum((g * self._steps[n]) ** 2 for n, g in grad.items()))
                if norm > 0:
                    delta = {n: 1.5 * self._steps[n] * (g * self._steps[n]) / norm for n, g in grad.items()}
                    _, ok = self._trial(k, delta, kind='gradient')
                    improved |= ok
            stall = 0 if improved else stall + 1
            b = self.best_at[k]
            self.print(f'{ind}  level {k}: {len(self.history):3d} evals  best {b.score:.4g}  steps '
                       + ' '.join(f'{n}={self._steps[n]:.3g}' for n in names))
        self._move_to(self.best_at[k].positions)

    # ------------------------------------------------------------------ run
    def _noise_floor(self):
        cms = [self._capture(None, 'noise') for _ in range(self.noise_repeats)]
        for k, lv in enumerate(self.levels):
            scores = [lv.objective.evaluate(cm)[0] for cm in cms]
            scores = [s for s in scores if s is not None]
            if len(scores) < 2:
                raise RuntimeError(f'signal lost before starting ({lv.objective.metric} invalid)')
            mean = sum(scores) / len(scores)
            self.noise[k] = math.sqrt(sum((s - mean) ** 2 for s in scores) / (len(scores) - 1))
            self.print(f'noise floor, level {k}: {lv.objective.metric} = {mean:.4g} +- {self.noise[k]:.2g}')

    def run(self):
        start = self.positions()
        t0 = time.time()
        aborted = False
        reason = ''
        self.best_at = {}
        self._trials_since = {}
        try:
            self._noise_floor()
            try:
                self._optimize(0)
                if len(self.levels) == 1 or 0 not in self.best_at:
                    self.best_at[0] = self._score_here(0, 'final')
                else:                              # verify the best point reproduces
                    self._move_to(self.best_at[0].positions)
                    ev = self._score_here(0, 'verify', extra=1 + self.outer_repeats)
                    self.print(f'verified best: {ev.score} (recorded {self.best_at[0].score:.4g})')
                    if ev.score is not None:
                        self.best_at[0] = ev
            except _Stopped as e:
                reason = str(e) or 'stopped'
                aborted = self._stop.is_set()
                self.print(f'nested loop: {reason}')
        except Exception:
            aborted = True
            raise
        finally:
            best = self.best_at.get(0)
            if best is None and self.best_at:      # stopped inside the first level-0 point
                best = self.best_at[min(self.best_at)]
            self.best = best
            if aborted or reason:
                target = start if (self.on_abort == 'start' or best is None) else best.positions
                if self.on_abort in ('best', 'start') or reason:
                    self._stop.clear()
                    try:
                        self._move_to(target)
                    except Exception as e:     # never mask the original error
                        self.print(f'return after abort failed: {e}')
        per_trial = {k: [p['evaluations'] for p in self.profile[k]] for k in self.profile}
        return {'start': start, 'best': self.best.positions if self.best else None,
                'best_score': self.best.score if self.best else None,
                'noise': list(self.noise), 'profile_noise': dict(self.profile_sigma),
                'evaluations': len(self.history),
                'evaluations_per_trial': per_trial,
                'predictor': {k: {'below': b, 'axes': c, 'K': K.tolist()} for k, (b, c, K) in self.K.items()},
                'aborted': aborted, 'stopped_because': reason, 'seconds': time.time() - t0}
