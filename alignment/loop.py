"""The alignment loop: move axes, measure, keep what helps.

    loop = AlignmentLoop(axes, measure, Objective('intensity'), tiers=[['tx', 'ty'], ['z']])
    summary = loop.run()

`axes`    {name: SafeAxis or VirtualAxis}. Every move goes through them, so the
          safety rules (range, ramps, freeze, E-stop) bind the optimizer too.
`measure` callable() -> alignment.metrics.CaptureMetrics, called only after
          every axis has finished moving AND its vibration settle delay has
          passed (wait_until_settled), because motion shakes the setup.

Design (kalishlot/MECHANICAL_CONTROL.md, section 3):
- Stages follow the tiers: tier 1 alone, then tiers 1+2, ... Frozen axes are
  skipped (and SafeAxis would refuse them anyway).
- Within a stage: coordinate search. Each axis tries +step then -step; an
  improvement is accepted only if it beats the incumbent by
  accept_sigmas x the measured noise floor. Success grows the step (x1.5),
  failure shrinks it (x0.5). After a pass over >= 2 axes the finite-difference
  gradient gives a joint move along a linear combination of the axes.
- Repeatability: no axis is stepped by less than min_step_sigmas x its
  repeatability_sigma (smaller steps are lost in the landing error). The
  metric, not the commanded position, is the truth: the incumbent is
  re-measured every `remeasure_every` evaluations (drift, landing error), and
  the final return to the best point is verified, with one more fine pass if
  it does not reproduce.
- Outliers: a suspiciously large improvement is re-measured before it is
  accepted (confirm_jump), so one bad capture cannot become the incumbent.
- Invalid measurements (no peak, saturated, ...) count as "worse", never as
  zero; an invalid incumbent stops the loop (signal lost).
- stop() aborts between evaluations; the axes then return to the best point
  (or the start, or stay: `on_abort`).
- Every evaluation is logged (JSON lines in log_dir when given).
- Target objectives (alignment/targets.py, e.g. MarkerTarget) have a goal
  rather than an optimum: when a new incumbent satisfies objective.reached(),
  it is measured confirm_repeats more times in place, and if the mean still
  satisfies it the loop stops there (summary['target_reached']).
  When the objective also reports a measured position (objective.position(ev),
  objective.goal), each stage first steers: probe every axis with a step grown
  until its effect clears the position noise, then move straight at the goal
  by least squares, refining the Jacobian with every move (Broyden). The
  coordinate search follows only if steering stalls. Small steps that drown in
  the noise would otherwise make the coordinate search shrink and give up.
"""

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


# ------------------------------------------------------------------ objective

@dataclass
class Objective:
    metric: str = 'intensity'
    maximize: bool = True
    constraints: tuple = ()          # (metric, '>=' | '<=', threshold), e.g. ('gaussian_overlap', '>=', 0.9)

    def evaluate(self, cm):
        """(score, feasible, reason). score is None when the metric is invalid.
        Higher score is always better (minimized metrics are negated)."""
        m = cm.metrics.get(self.metric)
        if m is None or not m.valid:
            return None, False, f'{self.metric} invalid: {m.reason if m else "missing"}'
        score = m.value if self.maximize else -m.value
        for name, op, thr in self.constraints:
            c = cm.metrics.get(name)
            if c is None or not c.valid:
                return score, False, f'constraint {name} invalid'
            ok = c.value >= thr if op == '>=' else c.value <= thr
            if not ok:
                return score, False, f'constraint {name} {op} {thr} violated ({c.value:.4g})'
        return score, True, ''


@dataclass
class Evaluation:
    index: int
    positions: dict
    score: float | None
    feasible: bool
    reason: str
    metrics: dict
    kind: str                         # 'noise', 'trial', 'remeasure', 'verify', ...
    time: float = field(default_factory=time.time)

    def better_than(self, other, margin):
        """Feasible beats infeasible; then score by more than `margin`."""
        if self.score is None:
            return False
        if other is None or other.score is None:
            return True
        if self.feasible != other.feasible:
            return self.feasible
        return self.score > other.score + margin


# ----------------------------------------------------------------- axis glue

def _pos(axis):
    return axis.position if hasattr(axis, 'components') else axis.commanded


def _sigma(axis):
    if hasattr(axis, 'components'):        # virtual: the worst component, scaled
        return max(abs(c) * a.limits.repeatability_sigma for a, c in axis.components if c)
    return axis.limits.repeatability_sigma


def _check(axis, target):
    """Raise SafetyError (a ValueError) if moving `axis` to `target` would be refused."""
    if hasattr(axis, 'components'):
        axis.check_delta(target - axis.position)
    else:
        axis.check_target(target)


def _default_step(axis):
    if hasattr(axis, 'components'):
        return min(a.limits.max_step / max(abs(c), 1e-12) for a, c in axis.components if c)
    return max(axis.limits.max_step, 10 * axis.limits.repeatability_sigma)


class AlignmentLoop:
    def __init__(self, axes, measure, objective=None, tiers=None, steps=None,
                 accept_sigmas=1.0, min_step_sigmas=3.0, noise_repeats=4,
                 remeasure_every=8, max_evaluations=150, on_abort='best', confirm_jump=10.0,
                 log_dir=None, who='alignment loop', print_fn=print):
        self.axes = dict(axes)
        self.measure_fn = measure
        self.objective = objective or Objective()
        self.tiers = tiers or [list(self.axes)]
        self.steps = {n: (steps or {}).get(n, _default_step(a)) for n, a in self.axes.items()}
        self.min_steps = {n: max(min_step_sigmas * _sigma(a), 1e-12) for n, a in self.axes.items()}
        self.accept_sigmas = accept_sigmas
        self.confirm_jump = confirm_jump
        self.noise_repeats = noise_repeats
        self.remeasure_every = remeasure_every
        self.max_evaluations = max_evaluations
        self.on_abort = on_abort
        self.who = who
        self.print = print_fn or (lambda *a, **k: None)
        self.history = []
        self.noise = 0.0
        self.best = None
        self.target_reached = False
        self._last_remeasure = 0
        self._stop = threading.Event()
        self._log_path = None
        if log_dir:
            d = Path(log_dir)
            d.mkdir(parents=True, exist_ok=True)
            self._log_path = d / 'evaluations.jsonl'

    # ------------------------------------------------------------ plumbing
    def stop(self):
        self._stop.set()

    def positions(self):
        return {n: _pos(a) for n, a in self.axes.items()}

    def _settle(self):
        for a in self.axes.values():
            a.wait_until_settled()

    def measure(self, kind):
        self._settle()
        cm = self.measure_fn()
        score, feasible, reason = self.objective.evaluate(cm)
        ev = Evaluation(len(self.history), self.positions(), score, feasible, reason,
                        {k: (m.value if m.valid else None) for k, m in cm.metrics.items()}, kind)
        self.history.append(ev)
        if self._log_path:
            with self._log_path.open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(ev.__dict__, default=str) + '\n')
        return ev

    def _move_to(self, target_positions):
        for n, target in target_positions.items():
            a = self.axes[n]
            if abs(_pos(a) - target) > 1e-12:
                a.move_to(target, who=self.who)

    def _budget_left(self):
        return (len(self.history) < self.max_evaluations and not self._stop.is_set()
                and not self.target_reached)

    @property
    def margin(self):
        return self.accept_sigmas * self.noise

    # --------------------------------------------------------------- stages
    def _noise_floor(self):
        evs = [self.measure('noise') for _ in range(self.noise_repeats)]
        scores = [e.score for e in evs if e.score is not None]
        if len(scores) < 2:
            raise RuntimeError('signal lost before starting: '
                               + '; '.join(sorted({e.reason for e in evs if e.reason})))
        mean = sum(scores) / len(scores)
        self.noise = math.sqrt(sum((s - mean) ** 2 for s in scores) / (len(scores) - 1))
        best = max(evs, key=lambda e: (e.feasible, e.score if e.score is not None else -math.inf))
        best.score = mean                      # the incumbent's value is the mean, not a lucky shot
        self.print(f'noise floor: {self.objective.metric} = {mean:.4g} +- {self.noise:.2g}')
        if hasattr(self.objective, 'reached') and self.objective.reached(evs):
            self.target_reached = True
            self.print('target already reached at the start')
        return best

    def _accept(self, ev):
        """Make ev the incumbent; for a target objective, confirm a hit in place."""
        self.best = ev
        if not hasattr(self.objective, 'reached') or not self.objective.reached([ev]):
            return
        repeats = [ev]
        for _ in range(getattr(self.objective, 'confirm_repeats', 2)):
            if not self._budget_left():
                break
            repeats.append(self.measure('confirm target'))
        scores = [e.score for e in repeats if e.score is not None]
        ev.score = sum(scores) / len(scores)          # the incumbent is the mean, not the lucky shot
        if len(repeats) > getattr(self.objective, 'confirm_repeats', 2) and self.objective.reached(repeats):
            self.target_reached = True
            self.print(f'target reached: {self.objective.metric} = {-ev.score:.3g} '
                       f'(mean of {len(repeats)} captures)')

    def _confirmed(self, ev):
        """A suspiciously large improvement (> confirm_jump x noise and > 5% of the
        incumbent) is measured once more in place; the trial then counts with the
        mean of both, so one outlier capture cannot become the incumbent."""
        if (ev.score is None or self.best is None or self.best.score is None
                or not self._budget_left()):
            return ev
        jump = ev.score - self.best.score
        if jump <= max(self.confirm_jump * self.noise, 0.05 * abs(self.best.score)):
            return ev
        again = self.measure('confirm')
        if again.score is None:
            return again
        again.score = (ev.score + again.score) / 2
        again.feasible = ev.feasible and again.feasible
        return again

    def _remeasure_incumbent(self):
        self._last_remeasure = len(self.history)
        ev = self.measure('remeasure')
        if ev.score is None:
            raise RuntimeError(f'signal lost at the incumbent: {ev.reason}')
        self.best = ev

    def _coordinate_pass(self, active):
        """One pass over the active axes; returns (improved, gradient estimates)."""
        improved = False
        grad = {}
        for n in active:
            if not self._budget_left():
                break
            if self.axes[n].frozen or self.steps[n] < self.min_steps[n]:
                continue
            x0 = self.best.positions[n]
            step = self.steps[n]
            results = {}
            accepted = False
            for sign in (+1, -1):
                target = x0 + sign * step
                try:
                    _check(self.axes[n], target)
                except ValueError:
                    continue                    # the edge of the allowed range: try the other side
                self._move_to({n: target})
                ev = self._confirmed(self.measure('trial'))
                results[sign] = ev
                if ev.better_than(self.best, self.margin):
                    self._accept(ev)
                    self.steps[n] *= 1.5
                    improved = accepted = True
                    break
                if not self._budget_left():
                    break
            if not accepted:
                self._move_to({n: x0})
                self.steps[n] *= 0.5
            if len(results) == 2 and all(r.score is not None for r in results.values()):
                grad[n] = (results[1].score - results[-1].score) / (2 * step)
            if len(self.history) - self._last_remeasure >= self.remeasure_every and self._budget_left():
                self._move_to(self.best.positions)
                self._remeasure_incumbent()
        return improved, grad

    def _gradient_step(self, grad):
        """A joint move along the estimated gradient (a linear combination of axes),
        each axis scaled by its own current step."""
        norm = math.sqrt(sum((g * self.steps[n]) ** 2 for n, g in grad.items()))
        if norm == 0 or not self._budget_left():
            return False
        x0 = dict(self.best.positions)
        target = {n: x0[n] + 1.5 * self.steps[n] * (g * self.steps[n]) / norm for n, g in grad.items()}
        try:
            for n, t in target.items():
                _check(self.axes[n], t)
        except ValueError:
            return False
        self._move_to(target)
        ev = self._confirmed(self.measure('gradient'))
        if ev.better_than(self.best, self.margin):
            self._accept(ev)
            return True
        self._move_to(x0)
        return False

    def _position_noise(self):
        """rms 2D scatter of the position over the noise-floor evaluations."""
        pts = [self.objective.position(e) for e in self.history if e.kind == 'noise']
        pts = [q for q in pts if q is not None]
        if len(pts) < 2:
            return 0.0, (pts[0] if pts else None)
        mx, my = sum(q[0] for q in pts) / len(pts), sum(q[1] for q in pts) / len(pts)
        rms = math.sqrt(sum((q[0] - mx) ** 2 + (q[1] - my) ** 2 for q in pts) / (len(pts) - 1))
        return rms, (mx, my)

    def _steer_stage(self, active, probe_doublings=4, max_moves=20):
        """Jacobian steering towards objective.goal (see the module docstring)."""
        import numpy as np
        axes = [n for n in active if not self.axes[n].frozen]
        if not axes or self.best is None:
            return
        position = self.objective.position
        noise, mean_at_start = self._position_noise()
        cur = self.best
        c = np.array(position(cur) if len(self.history) > self.noise_repeats or mean_at_start is None
                     else mean_at_start, dtype=float)
        goal = np.array(self.objective.goal, dtype=float)
        J = np.zeros((2, len(axes)))
        probes = {}
        for i, n in enumerate(axes):                    # probe: how does each axis move the position?
            h = self.steps[n]
            for _ in range(probe_doublings + 1):
                if not self._budget_left():
                    return
                target = cur.positions[n] + h
                try:
                    _check(self.axes[n], target)
                except ValueError:
                    h = -h
                    target = cur.positions[n] + h
                    try:
                        _check(self.axes[n], target)
                    except ValueError:
                        break
                self._move_to({n: target})
                q = position(self.measure('probe'))
                if q is None:
                    break
                d = np.array(q) - c
                J[:, i] = d / h
                probes[n] = abs(h)
                if math.hypot(*d) > 3 * noise:
                    break
                h *= 2
            self._move_to(cur.positions)
        if not probes or np.linalg.norm(J) == 0:
            self.print('  steering: the axes do not move the position measurably')
            return
        self.print('  steering: probed ' + ' '.join(f'{n} ({h:.3g})' for n, h in probes.items()))
        for _ in range(max_moves):
            if not self._budget_left():
                return
            dp = np.linalg.lstsq(J, goal - c, rcond=None)[0]
            limit = np.array([8 * probes.get(n, self.steps[n]) for n in axes])
            dp *= min(1.0, float(np.min(limit / np.maximum(np.abs(dp), 1e-12))))   # trust region
            moved = False
            for scale in (1.0, 0.5, 0.25):
                target = {n: cur.positions[n] + scale * dp[i] for i, n in enumerate(axes)}
                try:
                    for n, t in target.items():
                        _check(self.axes[n], t)
                except ValueError:
                    continue
                self._move_to(target)
                ev = self.measure('steer')
                q = position(ev)
                if q is None:
                    self._move_to(cur.positions)
                    continue
                step = np.array([target[n] - cur.positions[n] for n in axes])
                if step @ step > 0:                     # Broyden: learn from every move
                    J += np.outer((np.array(q) - c) - J @ step, step) / (step @ step)
                if ev.better_than(self.best, self.margin):
                    self._accept(ev)
                    cur, c, moved = ev, np.array(q, dtype=float), True
                    break
                self._move_to(cur.positions)
            if not moved or self.target_reached:
                return

    def _optimize_stage(self, active):
        self.print(f'stage: {", ".join(active)}')
        if hasattr(self.objective, 'position') and hasattr(self.objective, 'goal'):
            self._steer_stage(active)
            if self.target_reached:
                return
        stall = 0
        while self._budget_left() and stall < 2:
            if all(self.steps[n] < self.min_steps[n] for n in active if not self.axes[n].frozen):
                break
            improved, grad = self._coordinate_pass(active)
            if len(grad) >= 2:
                improved |= self._gradient_step(grad)
            stall = 0 if improved else stall + 1
            self.print(f'  {len(self.history):3d} evals  best {self.best.score:.4g}  '
                       f'steps ' + ' '.join(f'{n}={self.steps[n]:.3g}' for n in active))

    # ------------------------------------------------------------------ run
    def run(self):
        start = self.positions()
        t0 = time.time()
        aborted = False
        try:
            self.best = self._noise_floor()
            self._move_to(self.best.positions)
            seen = []
            for tier in self.tiers:
                seen += [n for n in tier if n in self.axes and n not in seen]
                active = [n for n in seen if not self.axes[n].frozen]
                if active:
                    self._optimize_stage(active)
                if self._stop.is_set() or self.target_reached:
                    break
            aborted = self._stop.is_set()
            if not aborted and not self.target_reached:   # a reached target was confirmed in place
                self._finish()
        except Exception:
            aborted = True
            raise
        finally:
            if aborted and self.on_abort in ('best', 'start'):
                target = start if self.on_abort == 'start' or self.best is None else self.best.positions
                self._stop.clear()
                try:
                    self._move_to(target)
                except Exception as e:      # never mask the original error
                    self.print(f'return after abort failed: {e}')
        return {'start': start, 'best': self.best.positions if self.best else None,
                'best_score': self.best.score if self.best else None,
                'noise': self.noise, 'evaluations': len(self.history),
                'target_reached': self.target_reached,
                'aborted': aborted, 'seconds': time.time() - t0}

    def _finish(self):
        """Go to the best point and verify it reproduces; one fine pass if not."""
        self._move_to(self.best.positions)
        recorded = self.best.score
        ev = self.measure('verify')
        if ev.score is not None and ev.score >= recorded - 2 * max(self.noise, 1e-12):
            self.best = ev
            self.print(f'verified best: {ev.score:.4g} (recorded {recorded:.4g})')
            return
        self.print(f'best did not reproduce ({ev.score} vs {recorded:.4g}); fine pass')
        self.best = ev
        for n in self.axes:
            self.steps[n] = max(self.min_steps[n] * 1.01, self.steps[n])
        active = [n for n in self.axes if not self.axes[n].frozen]
        self._coordinate_pass(active)
        self._move_to(self.best.positions)


# ------------------------------------------------------------------ 1D scan

def scan_axis(axis, positions, measure, objective=None, repeats=1, return_to='best',
              who='1D scan'):
    """Measure the objective at each position of one axis (sorted ascending,
    so every point is approached from the same side), after the vibration
    settle delay. Returns a list of dicts and leaves the axis at the best point
    (return_to='best'), back at the start ('start'), or at the last point (None).

    Also the lab's characterization tool: repeat a scan up and down to see the
    hysteresis, or with repeats > 1 at fixed points to see the repeatability."""
    objective = objective or Objective()
    start = _pos(axis)
    rows = []
    for x in sorted(float(p) for p in positions):
        _check(axis, x)
        axis.move_to(x, who=who)
        for r in range(repeats):
            axis.wait_until_settled()
            cm = measure()
            score, feasible, reason = objective.evaluate(cm)
            rows.append({'target': x, 'commanded': _pos(axis), 'repeat': r, 'score': score,
                         'feasible': feasible, 'reason': reason,
                         'metrics': {k: (m.value if m.valid else None) for k, m in cm.metrics.items()}})
    if return_to == 'start':
        axis.move_to(start, who=who)
    elif return_to == 'best':
        good = [r for r in rows if r['score'] is not None]
        if good:
            axis.move_to(max(good, key=lambda r: (r['feasible'], r['score']))['target'], who=who)
    return rows
