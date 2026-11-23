"""An alignment as the aligner box describes it, turned into loop objects.

A level = a table of rows plus the controls (axes) it may move:

    {'rows': [{'signal': 'basler_camera:1/com_x', 'goal': 'goto',
               'value': {'kind': 'marker', 'device_id': 'basler_camera:1',
                         'marker': 'Mirror center', 'coord': 'x'},
               'tolerance': 1.0}, ...],
     'controls': [{'axis': 'picomotor_ch1', 'step': None}, ...]}

A row with 'enabled': False stays in the table (greyed in the box) but takes
no part: active(level) drops it before anything is validated, measured or
built. Rows keep their table numbers ('_row') in error messages.

Goals per row:
    goto      bring the signal to the value, within tolerance
    max, min  optimize the signal (value optional: "good enough" threshold)
    ge, le    guard: the signal must stay >= / <= the value (a constraint)

What one level may hold is what the loops can run: either goto rows only
(a VectorTarget with per-signal tolerances, steered by its Jacobian), or
exactly one max/min row (an Objective); guard rows go with either.
validate() says why a level does not fit. Several levels = the nested loop
(alignment/nested.py: one Level per box level, outer first); not offered yet.
"""

from alignment.loop import Objective
from alignment.targets import VectorTarget

GOALS = {'goto': 'go to', 'max': 'maximize', 'min': 'minimize', 'ge': '≥ (guard)', 'le': '≤ (guard)'}
VALUE_KINDS = ('number', 'marker', 'hold')


def _is_number(v):
    try:
        return v is not None and float(v) == float(v)
    except (TypeError, ValueError):
        return False


def active(level):
    """The level with its disabled rows dropped; each kept row knows its table number."""
    rows = [dict(r, _row=r.get('_row', n)) for n, r in enumerate(level.get('rows') or [], 1)
            if r.get('enabled', True)]
    return {**level, 'rows': rows}


def validate(level):
    """Raise ValueError with a readable reason if the level cannot run (disabled rows ignored)."""
    all_rows = level.get('rows') or []
    rows = active(level)['rows']
    controls = [c for c in level.get('controls') or [] if c.get('axis')]
    if not rows:
        raise ValueError('every row is disabled' if all_rows else 'add at least one signal row')
    if not controls:
        raise ValueError('tick at least one control')
    for r in rows:
        i = r['_row']
        if not r.get('signal'):
            raise ValueError(f'row {i}: choose a signal')
        if r.get('goal') not in GOALS:
            raise ValueError(f'row {i}: choose a goal')
        kind = (r.get('value') or {}).get('kind')
        if r['goal'] in ('goto', 'ge', 'le'):
            if kind not in VALUE_KINDS:
                raise ValueError(f'row {i}: give a value')
            if kind == 'number' and not _is_number((r.get('value') or {}).get('value')):
                raise ValueError(f'row {i}: give a number')
            if kind == 'marker' and not (r['value'].get('marker') and r['value'].get('device_id')):
                raise ValueError(f'row {i}: choose a marker')
            if kind == 'hold' and r['goal'] != 'goto':
                raise ValueError(f'row {i}: "hold" is only for go-to rows')
        if r['goal'] == 'goto':
            tol = r.get('tolerance')
            if tol is None or not float(tol) > 0:
                raise ValueError(f'row {i}: a go-to row needs a tolerance > 0')
    goto = [r for r in rows if r['goal'] == 'goto']
    opt = [r for r in rows if r['goal'] in ('max', 'min')]
    if goto and opt:
        raise ValueError('one level either goes to values or optimizes one signal, not both '
                         '(the other belongs in a nested level)')
    if len(opt) > 1:
        raise ValueError('only one signal can be maximized or minimized per level')
    if not goto and not opt:
        raise ValueError('guards alone give nothing to do: add a go-to or a max/min row')
    sigs = [r['signal'] for r in goto]
    if len(set(sigs)) != len(sigs):
        raise ValueError('the same signal appears in two go-to rows')
    axes = [c['axis'] for c in controls]
    if len(set(axes)) != len(axes):
        raise ValueError('the same control is ticked twice')


def signals_of(level):
    return list(dict.fromkeys(r['signal'] for r in level['rows']))


def needs_start_values(level):
    return any((r.get('value') or {}).get('kind') == 'hold' for r in level['rows'])


def build_objective(level, values, label=''):
    """The loop objective of a validated level. values: {row index: resolved float}."""
    rows = level['rows']
    constraints = tuple((r['signal'], '>=' if r['goal'] == 'ge' else '<=', values[i])
                        for i, r in enumerate(rows) if r['goal'] in ('ge', 'le'))
    goto = [(i, r) for i, r in enumerate(rows) if r['goal'] == 'goto']
    if goto:
        return VectorTarget(tuple(r['signal'] for _, r in goto), tuple(values[i] for i, _ in goto),
                            constraints=constraints, label=label,
                            tolerances=tuple(float(r['tolerance']) for _, r in goto))
    i, r = next((i, r) for i, r in enumerate(rows) if r['goal'] in ('max', 'min'))
    good = values.get(i)
    return Objective(r['signal'], maximize=r['goal'] == 'max', constraints=constraints,
                     good_enough=good, confirm_repeats=2 if good is not None else 0)
