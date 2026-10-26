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

Probe steps. When the levels below must follow closely (a narrow ridge: an
outer step without the right inner correction loses the mode), the first
large steps fail before K is known, and the steps only ever shrink. With
Level.probe_steps = {axis: small step}, each axis first takes one small trial
(the levels below still re-optimized, the result judged like any trial), which
gives K; the larger steps then start already predicted.

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

Procedure targets (section 3.f). Level(..., measure=procedure): the level's value
at a point comes from procedure(run) -> CaptureMetrics instead of one capture,
e.g. alignment/procedures.py: move_and_compare (capture, move the large mirror's
stage, capture, move back: the drift of the mode on the mirror). run.capture()
counts against the shared budget; every axis the procedure moved is put back
when it ends, also on stop() or an error. Its noise floor comes from
noise_repeats procedure calls.

Vector targets (alignment/targets.py: VectorTarget, e.g. the drift (dy, dz) -> 0).
A level whose objective reports a position and a goal is steered first: probe
each axis, fit vector = a + J x over all the level's trials, move to where the
fit meets the goal, repeat; then keep the fitted zero (_steer). A fit over many
trials averages the scatter of the levels below, which comparing two single
trials does not. If the goal is still not reached, the coordinate search goes on.

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
    probe_steps: dict = field(default_factory=dict)  # small first trial per axis, to learn the predictor
                                                     # before the (larger) steps; see the module docstring
    max_evaluations: int | None = None           # per optimization of this level (captures, levels below included)
    measure: object = None                       # procedure(run) -> CaptureMetrics, instead of one capture;
                                                 # see "Procedure targets" in the module docstring


class _Stopped(Exception):
    pass


class _Run:
    """What a procedure gets: capture, move (through the safe axes), positions.
    Every axis it moves is put back by the loop when the procedure ends."""

    def __init__(self, loop, level, kind):
        self._loop, self._level, self._kind = loop, level, kind
        self.home = {}                              # axis -> position before the procedure moved it

    @property
    def axes(self):
        return self._loop.axes

    def positions(self):
        return self._loop.positions()

    def capture(self, label=''):
        """Settle every axis, capture once (counted in the shared budget). CaptureMetrics."""
        self._loop._check_running()
        return self._loop._capture(self._level, f'{self._kind} {label}'.strip())

    def move_to(self, targets):
        for n in targets:
            self.home.setdefault(n, _pos(self._loop.axes[n]))
        self._loop._move_to(targets)

    def move_by(self, deltas):
        self.move_to({n: _pos(self._loop.axes[n]) + d for n, d in deltas.items()})

    def move_back(self):
        self._loop._move_to(self.home)


class NestedLoop:
    def __init__(self, axes, measure, levels, outer_repeats=1, accept_sigmas=1.0, min_step_sigmas=3.0,
                 noise_repeats=4, remeasure_every=6, max_evaluations=400, confirm_jump=10.0,
                 predictor=True, predictor_window=10, inner_stall_passes=1, profile_noise=True,
                 steer=True, on_abort='best', log_dir=None,
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
        self.steer = steer                             # Jacobian steering for vector targets
        self._reached = {}                             # level -> its target reached (this optimization)
        self._noise_points = {}                        # level -> vector-target positions at the noise floor
        self._points = {}                              # level -> every Evaluation of this optimization
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

    def _call_procedure(self, k, kind):
        """Run level k's procedure here; every axis it moved goes back, also on
        stop or error. Returns its CaptureMetrics."""
        self._check_running()
        run = _Run(self, k, kind)
        try:
            cm = self.levels[k].measure(run)
        finally:
            run.move_back()
            self._last_cm = self._last_pos = None      # the last capture was not taken here
        return cm

    def _measure_level(self, k, kind):
        if self.levels[k].measure is not None:
            return self._call_procedure(k, kind)
        self._check_running()
        return self._capture(k, kind)

    def _score_here(self, k, kind, extra=None):
        """Level k's objective at the current position: the last capture if it was
        taken here (not for a procedure level), plus outer_repeats (or `extra`)
        more. Returns an Evaluation."""
        obj = self.levels[k].objective
        pos = self.positions()
        cms = []
        if (self.levels[k].measure is None and self._last_pos is not None
                and all(abs(self._last_pos[n] - pos[n]) < 1e-12 for n in pos)):
            cms.append(self._last_cm)
        want = len(cms) + (self.outer_repeats if extra is None else extra)
        while len(cms) < max(want, 1):
            cms.append(self._measure_level(k, kind))
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
        loop = AlignmentLoop({n: self.axes[n] for n in names}, lambda: self._measure_level(k, 'inner'),
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

    def _trial(self, k, delta, kind='trial', adopt=False):
        """Move level k's axes by delta (predicted moves below), optimize below,
        score level k. Keep it if better (or `adopt`: keep it whatever it
        measured, when a fit over many trials says it is better); else return
        to the incumbent."""
        inc = self.best_at[k]
        x0 = dict(inc.positions)
        target = {n: x0[n] + dv for n, dv in delta.items()}
        if not self._allowed(target):
            return None, False
        n_before = len(self.history)
        predicted = self._predict(k, delta)
        self._move_to({**target, **predicted})
        try:
            self._optimize(k + 1)
        except RuntimeError as e:                 # the levels below lost the signal there
            self.print(f'{"  " * k}  level {k} trial lost the signal ({e}); back')
            self._move_to(x0)
            return None, False
        x1 = self.positions()
        ev = self._confirmed(k, self._score_here(k, kind))
        self._points.setdefault(k, []).append(ev)
        self._learn(k, delta, x0, x1)
        accepted = ev.better_than(inc, self.margin(k)) or (adopt and ev.score is not None)
        K = self.K.get(k)
        self.profile[k].append({'delta': dict(delta), 'kind': kind, 'score': ev.score, 'feasible': ev.feasible,
                                'accepted': accepted, 'evaluations': len(self.history) - n_before,
                                'first_capture': n_before, 'last_capture': len(self.history) - 1,
                                'start': x0, 'positions': x1, 'metrics': ev.metrics,
                                'predicted': predicted, 'K': K[2].copy() if K else None})
        self._trials_since[k] += 1
        if accepted:
            self.best_at[k] = ev
            self._check_reached(k)
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

    def _check_reached(self, k):
        """A target objective (good_enough, VectorTarget, ...) whose incumbent looks
        reached is measured confirm_repeats more times in place; if the mean still
        is, level k stops there."""
        obj = self.levels[k].objective
        inc = self.best_at[k]
        if not hasattr(obj, 'reached') or inc.score is None or not obj.reached([inc]):
            return False
        n = getattr(obj, 'confirm_repeats', 0)
        if n:
            again = self._score_here(k, 'confirm target', extra=n)
            if again.score is None or not obj.reached([inc] + [again] * n):
                return False
            n1, n2 = getattr(inc, 'n', 1), getattr(again, 'n', 1)
            again.score = (inc.score * n1 + again.score * n2) / (n1 + n2)
            again.feasible = inc.feasible and again.feasible
            self.best_at[k] = again
        self._reached[k] = True
        self.print(f'{"  " * k}  level {k}: target reached ({self._shown(k, self.best_at[k].score)})')
        return True

    def _shown(self, k, score):
        """A score as the user reads it (the metric, not its negation)."""
        if score is None:
            return 'invalid'
        return f'{getattr(self.levels[k].objective, "value_of", lambda s: s)(score):.4g}'

    def _position_noise(self, k):
        """rms scatter of a vector target's position over level k's noise-floor captures."""
        pts = self._noise_points.get(k) or []
        if len(pts) < 2:
            return 0.0
        mean = [sum(q[i] for q in pts) / len(pts) for i in range(len(pts[0]))]
        return math.sqrt(sum(sum((a - b) ** 2 for a, b in zip(q, mean)) for q in pts) / (len(pts) - 1))

    def _steer(self, k, probe_doublings=4, max_moves=12, rounds=3):
        """Steering of level k towards objective.goal, for a measured vector (e.g. the
        drift (dy, dz)) that is close to linear in the level's axes. Probe each axis
        (the step doubled until the vector moves by more than 3x its noise; the
        levels below re-optimized, as in every trial), then repeatedly move to where
        a straight-line fit over all trials so far meets the goal (_fit_vector),
        until that point stops moving; then go there and keep it (`adopt`: the fit
        over all trials is a better judge than one more noisy measurement). Up to
        `rounds` times, each fit using the trials of the rounds before.
        Returns True when the target was reached; otherwise the coordinate search
        goes on from the fitted zero."""
        obj = self.levels[k].objective
        names = self._free(self.levels[k].axes)
        goal = np.array(obj.goal, dtype=float)

        def where(ev):
            q = obj.position(ev) if ev is not None and ev.score is not None else None
            return None if q is None else np.array(q, dtype=float)
        if where(self.best_at[k]) is None or not names:
            return False
        noise = max(self._position_noise(k), self.profile_sigma.get(k, 0.0))
        J = np.zeros((goal.size, len(names)))
        probes = {}
        for i, n in enumerate(names):                # how does each axis move the vector?
            h = self._steps[n]
            for attempt in range(probe_doublings + 1):
                base = where(self.best_at[k])
                ev, _ = self._trial(k, {n: h}, kind='probe')
                q = where(ev)
                if q is None:
                    if attempt == 0:                 # not allowed / lost there: the other side
                        h = -h
                        continue
                    break
                d = q - base
                J[:, i] = d / h
                probes[n] = abs(h)
                if self._reached.get(k) or np.linalg.norm(d) > 3 * noise:
                    break
                h *= 2
            if self._reached.get(k):
                return True
        if not probes or not np.any(J):
            self.print(f'{"  " * k}  level {k}: steering: the axes do not move {obj.metric} measurably')
            return False
        self.print(f'{"  " * k}  level {k}: steering, probed ' + ' '.join(f'{n} ({h:.3g})' for n, h in probes.items()))
        mins = np.array([self._min_steps[n] for n in names])
        limit = np.array([8 * probes.get(n, self._steps[n]) for n in names])
        for _ in range(rounds):
            last_root = None
            for _ in range(max_moves):
                if self._reached.get(k):
                    return True
                model = self._fit_vector(k)
                if model is None:
                    return False
                root, _, _ = model
                dp = root - self._level_x(k, self.best_at[k])
                dp *= min(1.0, float(np.min(limit / np.maximum(np.abs(dp), 1e-12))))   # trust region
                settled = last_root is not None and np.all(np.abs(root - last_root) < mins)
                if len(self._points[k]) >= 2 * (len(names) + 1) and (settled or np.all(np.abs(dp) < mins)):
                    break                            # enough trials, and the fitted zero stopped moving
                if np.all(np.abs(dp) < mins):        # already there: one more point at the zero, to average
                    dp = np.where(np.abs(dp) < mins, mins * np.sign(dp + 1e-300), dp)
                last_root = root
                ev, _ = self._trial(k, {n: float(dp[i]) for i, n in enumerate(names)}, kind='steer')
                if where(ev) is None:                # lost the levels below there: smaller moves
                    limit = limit / 2
                self.print(f'{"  " * k}  level {k}: {len(self.history):3d} evals  steering, '
                           f'{obj.metric} {self._shown(k, self.best_at[k].score)}')
            if self._reached.get(k):
                return True
            model = self._fit_vector(k)              # go to the fitted zero, and keep it
            if model is None:
                return False
            root, at_root, at_incumbent = model
            dp = root - self._level_x(k, self.best_at[k])
            if not (np.any(np.abs(dp) >= mins) and at_root + self.margin(k) < at_incumbent):
                break                                # the incumbent is the fitted zero already
            self._trial(k, {n: float(dp[i]) for i, n in enumerate(names)}, kind='fitted zero', adopt=True)
            self.print(f'{"  " * k}  level {k}: at the fitted zero ({at_root:.3g} by the fit), measured '
                       f'{self._shown(k, self.best_at[k].score)}')
        return bool(self._reached.get(k))

    def _level_x(self, k, ev):
        return np.array([ev.positions[n] for n in self._free(self.levels[k].axes)], dtype=float)

    def _fit_vector(self, k):
        """Straight-line fit, measured vector = a + J x (x: level k's axes), over every
        trial of this optimization of level k. One trial's vector carries the scatter
        of the levels below (they never end exactly at their optimum); the fit
        averages it over all trials. Returns (x where the fit meets the goal, fitted
        distance there, fitted distance at the incumbent), or None when the trials
        do not span the axes."""
        obj = self.levels[k].objective
        names = self._free(self.levels[k].axes)
        pts = [(self._level_x(k, ev), np.array(obj.position(ev), dtype=float)) for ev in self._points.get(k, [])
               if ev.score is not None and obj.position(ev) is not None]
        if len(pts) < len(names) + 1:
            return None
        X = np.array([np.concatenate(([1.0], x)) for x, _ in pts])
        if np.linalg.matrix_rank(X) < len(names) + 1:
            return None
        B = np.linalg.lstsq(X, np.array([q for _, q in pts]), rcond=None)[0]
        a, J = B[0], B[1:].T
        goal = np.array(obj.goal, dtype=float)
        root = np.linalg.lstsq(J, goal - a, rcond=None)[0]
        x_inc = self._level_x(k, self.best_at[k])
        return root, float(np.linalg.norm(a + J @ root - goal)), float(np.linalg.norm(a + J @ x_inc - goal))

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
        self._reached[k] = False
        self._points[k] = [self.best_at[k]]
        if self.best_at[k].score is None:
            raise RuntimeError(f'signal lost at level {k}: {self.best_at[k].reason}')
        obj = self.levels[k].objective
        self.print(f'{ind}level {k} ({", ".join(names)}): start {obj.metric} {self._shown(k, self.best_at[k].score)}')
        if self._check_reached(k):
            return
        if self.profile_noise and k not in self.profile_sigma:
            self._estimate_profile_noise(k)
        for n in names:                           # learn the predictor with small steps first
            h = self.levels[k].probe_steps.get(n)
            if h:
                ev, ok = self._trial(k, {n: h}, kind='probe')
                if not ok and (ev is None or ev.score is None):
                    self._trial(k, {n: -h}, kind='probe')
        steered = False
        if self.steer and hasattr(obj, 'position') and hasattr(obj, 'goal') and not self._reached[k]:
            steered = self._steer(k)
        stall = 0
        n0 = len(self.history)
        cap = self.levels[k].max_evaluations
        while stall < 2 and names and not self._reached[k] and not steered:
            if all(self._steps[n] < self._min_steps[n] for n in names):
                break
            if cap is not None and len(self.history) - n0 >= cap:
                break
            improved = False
            grad = {}
            for n in names:
                step = self._steps[n]
                if step < self._min_steps[n] or self._reached[k]:
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
            if len(grad) >= 2 and not self._reached[k]:
                norm = math.sqrt(sum((g * self._steps[n]) ** 2 for n, g in grad.items()))
                if norm > 0:
                    delta = {n: 1.5 * self._steps[n] * (g * self._steps[n]) / norm for n, g in grad.items()}
                    _, ok = self._trial(k, delta, kind='gradient')
                    improved |= ok
            stall = 0 if improved else stall + 1
            b = self.best_at[k]
            self.print(f'{ind}  level {k}: {len(self.history):3d} evals  best {self._shown(k, b.score)}  steps '
                       + ' '.join(f'{n}={self._steps[n]:.3g}' for n in names))
        self._move_to(self.best_at[k].positions)

    # ------------------------------------------------------------------ run
    def _noise_floor(self):
        """noise_repeats captures in place, shared by the levels that use one capture;
        a procedure level runs its procedure noise_repeats times instead."""
        shared = None
        for k, lv in enumerate(self.levels):
            if lv.measure is not None:
                cms = [self._call_procedure(k, 'noise') for _ in range(self.noise_repeats)]
            else:
                if shared is None:
                    shared = [self._capture(None, 'noise') for _ in range(self.noise_repeats)]
                cms = shared
            scores = [lv.objective.evaluate(cm)[0] for cm in cms]
            scores = [s for s in scores if s is not None]
            if len(scores) < 2:
                raise RuntimeError(f'signal lost before starting ({lv.objective.metric} invalid)')
            mean = sum(scores) / len(scores)
            self.noise[k] = math.sqrt(sum((s - mean) ** 2 for s in scores) / (len(scores) - 1))
            if hasattr(lv.objective, 'position'):
                pts = [lv.objective.position(Evaluation(0, {}, 0.0, True, '', {n: m.value for n, m in cm.metrics.items()
                                                                               if m.valid}, 'noise')) for cm in cms]
                self._noise_points[k] = [q for q in pts if q is not None]
            self.print(f'noise floor, level {k}: {lv.objective.metric} = {self._shown(k, mean)} '
                       f'+- {self.noise[k]:.2g}')

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
                'target_reached': bool(self._reached.get(0)),
                'aborted': aborted, 'stopped_because': reason, 'seconds': time.time() - t0}
