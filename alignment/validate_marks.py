"""Run the alignment metrics on saved synced sessions and compare with hand marks.

For every session folder under ROOTS: split sweeps, find 0th-order peaks and
the first-order family, score the camera frames, and compare with the
session's *_scope.modemarks.json when present (pairs of [0th, 1st] Lorentzian
fits clicked in mode_marking.py).

Prints one block per session; with SAVE_JSON also writes
<session>/alignment_metrics.json next to it.

Run from PyCharm (edit the globals below) or:
    python alignment/validate_marks.py [session_or_day_folder ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.metrics import MetricsConfig, evaluate_session  # noqa: E402

DAILY = Path(r'C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox'
             r'\Laser Phase Plate\Daily measurements and notes')
ROOTS = [DAILY / '2026-10-11', DAILY / '2026-10-12', DAILY / '2026-10-15']
WITH_FRAMES = True
SAVE_JSON = False
VERBOSE = False              # one line per peak and frame
MATCH_TOL_S = 0.003          # a detected peak matches a hand mark within this
CONFIG = MetricsConfig()


def session_folders(roots):
    for root in roots:
        root = Path(root)
        if list(root.glob('*_session.json')):
            yield root
        else:
            for sub in sorted(p for p in root.iterdir() if p.is_dir()):
                if list(sub.glob('*_session.json')):
                    yield sub


def compare_with_marks(folder, result):
    marks_files = list(folder.glob('*_scope.modemarks.json'))
    if not marks_files:
        return None
    marks = json.loads(marks_files[0].read_text(encoding='utf-8'))['marks']
    lines = []
    fundamentals = [(sw['sweep'], f) for sw in result.sweeps for f in sw['fundamentals']]
    for pair in marks:
        t0, t1 = pair[0]['x0'], pair[1]['x0']
        hit = [(sw, f) for sw, f in fundamentals if abs(f['t'] - t0) < MATCH_TOL_S]
        if not hit:
            in_sweep = [s for s in result.sweeps
                        if s['sweep']['t_start'] <= t0 <= s['sweep']['t_end']]
            lines.append(f'    mark 0th @ {t0:.4f}: NOT detected'
                         f' ({"inside" if in_sweep else "outside"} the usable sweeps)')
            continue
        sw, f = hit[0]
        s_mark = (t1 - t0) / f['local_fsr'] * sw['direction']
        lines.append(f'    mark 0th @ {t0:.4f}: detected (height {f["height"]:.3f} V, '
                     f'mark {pair[0]["height"]:.3f}); mark 1st at s={s_mark:+.3f} '
                     f'(dir {sw["direction"]:+d})')
    return lines


def describe(folder, result):
    print(f'\n== {folder.name}')
    for sw in result.sweeps:
        s = sw['sweep']
        fam = sw['family_s']
        fu = sw['fundamentals']
        n_ok = sum(f['usable'] for f in fu)
        whys = sorted({f['why'] for f in fu if not f['usable']})
        print(f'  sweep {s["t_start"]:.2f}-{s["t_end"]:.2f}s dir {s["direction"]:+d}: '
              f'FSR {sw["fsr_time_s"] or float("nan"):.3f}s, 0th {len(fu)} ({n_ok} usable'
              f'{", " + "/".join(whys) if whys else ""}), family s='
              f'{"none" if fam is None else f"{fam:+.3f}"}, 1st {len(sw["first_orders"])}')
        if sw.get('family_candidates'):
            print('      families (s: weight):', ', '.join(
                f'{c:+.3f}: {w:.2f}' for c, w in sw['family_candidates']),
                f'| chosen by {sw.get("family_choice")}',
                (f'camera scores {[round(x, 2) for x in sw["family_camera_scores"]]}'
                 if sw.get('family_camera_scores') else ''))
        conf = [fo.get('camera_confirmed') for fo in sw['first_orders']]
        if any(c is not None for c in conf):
            print(f'      first orders confirmed on camera: {sum(bool(c) for c in conf)}/{len(conf)}')
        for fo in (sw['first_orders'] if VERBOSE else []):
            if 'fundamental_likeness' in fo:
                print(f'      1st @ {fo["peak"]["t"]:.4f}: likeness {fo["fundamental_likeness"]:.2f}'
                      f' purity {fo["purity"]:.2f} confirmed {fo["camera_confirmed"]}'
                      f' split {fo["split"]} ({fo["split_linewidths"]:.2f} lw)')
        for f in (fu if VERBOSE else []):
            if f.get('gauss') and f['gauss']['valid']:
                g = f['gauss']
                print(f'      0th @ {f["t"]:.4f}: frame {f["frame"]} overlap {g["overlap"]:.3f}'
                      f' ell {g["ellipticity"]:.2f} purity {f["purity"]:.2f}'
                      f' usable {f["frame_usable"]}')
            elif f.get('frame_why'):
                print(f'      0th @ {f["t"]:.4f}: frame {f.get("frame")} - {f["frame_why"]}')
    for name, m in result.metrics.items():
        val = f'{m.value:.4g} +- {m.spread:.2g} (n={m.n})' if m.n else '-'
        print(f'  {name:17s} {val:28s} {"ok" if m.valid else "INVALID: " + m.reason}')
    cmp_lines = compare_with_marks(folder, result)
    if cmp_lines:
        print('  hand marks:')
        print('\n'.join(cmp_lines))


def main(roots):
    for folder in session_folders(roots):
        try:
            result = evaluate_session(folder, CONFIG, with_frames=WITH_FRAMES)
        except Exception as e:      # keep going over a day's sessions
            print(f'\n== {folder.name}\n  FAILED: {type(e).__name__}: {e}')
            continue
        describe(folder, result)
        if SAVE_JSON:
            (folder / 'alignment_metrics.json').write_text(
                json.dumps(result.to_dict(), indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    main([Path(a) for a in sys.argv[1:]] or ROOTS)
