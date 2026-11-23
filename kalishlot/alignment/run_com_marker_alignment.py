"""Steer camera spots onto kalishlot markers with motors, by the centre of mass.

    python alignment/run_com_marker_alignment.py \
        --target "basler_camera:24916690=Mirror center" \
        --target "ximea_camera:QXNAA2506000=Aperture" \
        --axes picomotor_ch1 picomotor_ch2 picomotor_ch3 picomotor_ch4

Each --target is CAMERA_DEVICE_ID=MARKER_LABEL, read live from the running
kalishlot (the cameras stay there; their centre of mass is switched on). The
axes are opened HERE, so their box must not be open in kalishlot (refused
otherwise). The loop is alignment/loop.py's AlignmentLoop with a VectorTarget
over every camera's centre of mass (alignment/com_measure.py): it measures the
noise, probes each axis, steers by least squares with a Jacobian it refines
after every move, and stops once the mean position over 3 captures is within
--tolerance px of the markers (combined over the cameras).

--dry-run measures 3 times and prints the miss, without opening any axis.
Ctrl+C stops between evaluations and returns the axes to the best point.
Every evaluation is logged to alignment/runs/<time>/evaluations.jsonl.
"""

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.com_measure import KALISHLOT_URL, ComMeasure  # noqa: E402
from alignment.loop import AlignmentLoop  # noqa: E402
from motion import build_axis  # noqa: E402
from motion.axes_config import AXES  # noqa: E402

RUNS_DIR = Path(__file__).resolve().parent / 'runs'


def parse_targets(items):
    """['basler_camera:249=Mirror center', ...] -> cameras {name: id}, markers {name: label}."""
    cameras, markers = {}, {}
    for item in items:
        device_id, sep, marker = item.partition('=')
        if not sep or not device_id or not marker:
            raise SystemExit(f'--target must be CAMERA_DEVICE_ID=MARKER_LABEL, got {item!r}')
        name = device_id.split(':')[0].replace('_camera', '')
        while name in cameras:
            name += '2'
        cameras[name], markers[name] = device_id, marker
    return cameras, markers


def describe(cm, target):
    parts = []
    for i in range(0, len(target.metrics), 2):
        mx, my = cm.metrics[target.metrics[i]], cm.metrics[target.metrics[i + 1]]
        cam = target.metrics[i].rsplit('_com_', 1)[0]
        if mx.valid and my.valid:
            parts.append(f'{cam} ({mx.value:7.1f}, {my.value:7.1f}) '
                         f'miss ({mx.value - target.goal[i]:+6.1f}, {my.value - target.goal[i + 1]:+6.1f})')
        else:
            parts.append(f'{cam} invalid: {mx.reason}')
    score, _, _ = target.evaluate(cm)
    return '  '.join(parts) + (f'  |  distance {-score:.2f} px' if score is not None else '')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', action='append', required=True, help='CAMERA_DEVICE_ID=MARKER_LABEL')
    ap.add_argument('--axes', nargs='+', default=['picomotor_ch1', 'picomotor_ch2',
                                                   'picomotor_ch3', 'picomotor_ch4'])
    ap.add_argument('--tolerance', type=float, default=3.0, help='px, combined over the cameras')
    ap.add_argument('--max-evaluations', type=int, default=80)
    ap.add_argument('--samples', type=int, default=10, help='CoM results per camera per evaluation')
    ap.add_argument('--delay', type=float, default=0.3, help='s after a move before frames count')
    ap.add_argument('--step', type=float, default=None,
                    help='first probe step for every axis (default: the axis max_step)')
    ap.add_argument('--dry-run', action='store_true', help='measure only, move nothing')
    ap.add_argument('--url', default=KALISHLOT_URL)
    args = ap.parse_args()

    cameras, markers = parse_targets(args.target)
    measure = ComMeasure(cameras, samples=args.samples, delay_s=args.delay, url=args.url)
    measure.ensure_on()
    target = measure.marker_target(markers, tolerance_px=args.tolerance)
    print(f'target: {target.label}  goal {tuple(round(g, 1) for g in target.goal)}  '
          f'tolerance {args.tolerance} px')

    if args.dry_run:
        for _ in range(3):
            print(describe(measure(), target))
        return

    drivers = {AXES[n]['driver'] for n in args.axes}
    held = [d for d in measure.devices() if d in {f'motion_axes:{drv}' for drv in drivers}]
    if held:
        raise SystemExit(f'kalishlot has {held} open: close that box first (one owner per motor)')

    axes = {n: build_axis(n) for n in args.axes}
    for a in axes.values():
        a.open()
    try:
        print('start: ' + '  '.join(f'{n}={a.position:g}' for n, a in axes.items()))
        count = [0]

        def logged_measure():
            cm = measure()
            count[0] += 1
            pos = ' '.join(f'{a.commanded:g}' for a in axes.values())
            print(f'  #{count[0]:3d} [{pos}]  {describe(cm, target)}')
            return cm

        run_dir = RUNS_DIR / time.strftime('%Y%m%d-%H%M%S')
        loop = AlignmentLoop(axes, logged_measure, target, tiers=[list(axes)],
                             steps={n: args.step for n in axes} if args.step else None,
                             max_evaluations=args.max_evaluations, log_dir=run_dir,
                             who='com marker alignment')
        result = {}
        worker = threading.Thread(target=lambda: result.update(summary=loop.run(), error=None),
                                  daemon=True)
        worker.start()
        try:
            while worker.is_alive():
                worker.join(0.2)
        except KeyboardInterrupt:
            print('stopping: returning to the best point ...')
            loop.stop()
            worker.join()
        summary = result.get('summary')
        if summary is None:
            raise SystemExit('the loop failed (see the traceback above); axes returned to the best point')
        reason = '' if summary['target_reached'] else f" ({summary.get('end_reason') or 'budget'})"
        print(f"\n{'TARGET REACHED' if summary['target_reached'] else 'target NOT reached'}{reason}"
              f"{' (aborted)' if summary['aborted'] else ''}: {summary['evaluations']} evaluations, "
              f"{summary['seconds']:.0f} s")
        print('end:   ' + '  '.join(f'{n}={a.position:g}' for n, a in axes.items()))
        print('steps: ' + '  '.join(f'{n}={a.travel_used:g}' for n, a in axes.items())
              + f'  (total {sum(a.travel_used for a in axes.values()):g})')
        print(f'log:   {run_dir / "evaluations.jsonl"}')
    finally:
        for a in axes.values():
            a.close()


if __name__ == '__main__':
    main()
