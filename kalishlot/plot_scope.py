"""Plot one PicoScope capture saved by kalishlot's pipeline.

    python kalishlot/plot_scope.py <capture>_scope.npz
    python kalishlot/plot_scope.py <capture>_scope_tail.npz --save plot.png

Takes the .npz of a single scope recording, whichever way it was made: the scope
block of a synced video capture (`<stamp>_scope.npz` beside a `*_session.json`),
its trailing capture (`<stamp>_scope_tail.npz`), or a stand-alone "record scope
only" (`<stamp>_scope.npz` beside `<stamp>_scope.json`). Each channel gets a y
axis of its own - the transmission is tens of millivolts and the temperature ramp
volts - and the title says how the scope was set: sampling rate, samples,
duration, and each channel's range and coupling.

The conditions come from the record next to the file (the session or scope
json); with none, the title says only what the data itself shows.

GUI-free apart from the window: `describe_capture` and `build_figure` need no
display, which is how the self-test runs them.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CHANNEL_COLORS = ('tab:blue', 'tab:green', 'tab:red', 'tab:orange')
SPINE_STEP = 0.11           # axes widths between the right-hand y axes


def find_record(npz_path):
    """(record dict, role) for the json next to `npz_path`, or (None, None).

    role is 'main' (the video's scope block, or a stand-alone scope capture) or
    'tail' (the trailing capture): whichever block of whichever json names this
    very file."""
    npz_path = Path(npz_path)
    for path in sorted(npz_path.parent.glob('*.json')):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        if (record.get('scope') or {}).get('file') == npz_path.name:
            return record, 'main'
        if (record.get('scope_tail') or {}).get('file') == npz_path.name:
            return record, 'tail'
    return None, None


def _range_text(volts):
    return f'±{volts * 1e3:g} mV' if volts < 1 else f'±{volts:g} V'


def describe_capture(npz_path):
    """Everything the plot needs, as a dict: `t`, `channels` (a list of dicts
    with name, label, volts, range_v, coupling), the sampling conditions and
    the title lines."""
    npz_path = Path(npz_path)
    data = np.load(npz_path)
    t = np.asarray(data['t'], dtype=float)
    record, role = find_record(npz_path)
    scope = (record or {}).get('scope') or {}
    block = (record or {}).get('scope_tail') if role == 'tail' else scope
    block = block or {}

    interval = block.get('sample_interval_s') or (
        float(np.median(np.diff(t))) if t.size > 1 else None)
    n_samples = block.get('n_samples') or int(t.size)
    duration = float(t[-1] - t[0]) if t.size > 1 else 0.0

    channels = []
    for key in ('signal', 'aux'):
        if key not in data.files:
            continue
        meta = scope if key == 'signal' else (scope.get('aux') or {})
        letter = meta.get('channel')
        label = ('transmission' if key == 'signal'
                 else meta.get('label') or 'aux channel')
        channels.append({
            'key': key, 'volts': np.asarray(data[key], dtype=float),
            'channel': letter, 'label': label,
            'range_v': meta.get('range_v'), 'coupling': meta.get('coupling')})

    lines = []
    head = npz_path.name
    if role == 'tail':
        head += ' - trailing capture'
        offset = block.get('start_offset_s')
        if offset is not None:
            head += f', starts {offset:.3f} s after the main trace'
        fg = block.get('function_generator') or {}
        if fg:
            sent = fg.get('on_sent_after_start_s', fg.get('on_after_start_s'))
            done = fg.get('on_done_after_start_s')
            text = f'function generator CH{fg.get("channel", "?")} on'
            if sent is not None:
                text += (f' (command sent {sent * 1e3:.0f} ms in'
                         + (f', returned at {done * 1e3:.0f} ms' if done else '')
                         + ')')
                if sent > duration:
                    text += ' - after this recording had ended'
            lines.append(text)
    elif role is None:
        head += ' - no record found next to it'
    lines.insert(0, head)

    rate = f'{1 / interval / 1e3:g} kS/s ({interval * 1e6:g} µs/sample)' \
        if interval else 'sample rate unknown'
    lines.insert(1, f'{n_samples} samples at {rate}, {duration:.4g} s'
                    + (f'; scope {scope["variant"]} s/n {scope["serial"]}'
                       if scope.get('variant') else ''))
    settings = []
    for channel in channels:
        name = channel['channel'] or ('?' if channel['key'] == 'signal' else 'aux')
        if channel['range_v']:
            settings.append(f'{name} {_range_text(channel["range_v"])} '
                            f'{channel["coupling"] or ""}'.rstrip())
        else:
            settings.append(f'{name} range unknown')
    if settings:
        lines.insert(2, 'channels: ' + ', '.join(settings))
    return {'t': t, 'channels': channels, 'lines': lines, 'role': role,
            'sample_interval_s': interval, 'n_samples': n_samples}


def build_figure(npz_path, figsize=(13, 5.5)):
    """The figure for one capture: a shared time axis, a y axis per channel."""
    import matplotlib.pyplot as plt

    capture = describe_capture(npz_path)
    fig, host = plt.subplots(figsize=figsize)
    axes = []
    handles = []
    for index, channel in enumerate(capture['channels']):
        color = CHANNEL_COLORS[index % len(CHANNEL_COLORS)]
        if index == 0:
            ax = host
        else:
            ax = host.twinx()
            ax.spines['right'].set_position(('axes', 1 + SPINE_STEP * (index - 1)))
        label = (f'{channel["channel"]}: ' if channel['channel'] else '') \
            + channel['label']
        scale, unit = ((1e3, 'mV') if np.max(np.abs(channel['volts']), initial=0) < 0.5
                       else (1.0, 'V'))
        line, = ax.plot(capture['t'], channel['volts'] * scale, lw=0.8,
                        color=color, label=label)
        ax.set_ylabel(f'{label} [{unit}]', color=color, fontsize=9)
        ax.tick_params(axis='y', colors=color, labelsize=8)
        axes.append(ax)
        handles.append(line)
    host.set_xlabel('time in this recording [s]')
    if handles:
        host.legend(handles=handles, loc='upper right', fontsize=8,
                    framealpha=0.7)
    fig.suptitle('\n'.join(capture['lines']), fontsize=9)
    fig.subplots_adjust(top=0.84 - 0.01 * len(capture['lines']),
                        right=max(0.9 - SPINE_STEP * (len(axes) - 1) * 0.6, 0.6))
    return fig, axes


def _self_test():
    import tempfile
    import matplotlib
    matplotlib.use('Agg')

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        t = np.arange(1000) * 1e-4
        np.savez(folder / 's_scope.npz', t=t, signal=np.sin(t * 40) * 0.02,
                 aux=t * 4 - 2)
        np.savez(folder / 's_scope_tail.npz', t=t[:400], signal=t[:400] * 0.01,
                 aux=t[:400])
        scope = {'file': 's_scope.npz', 'channel': 'D', 'range_v': 0.05,
                 'coupling': 'DC', 'sample_interval_s': 1e-4, 'n_samples': 1000,
                 'variant': '4424A', 'serial': 'X1',
                 'aux': {'channel': 'B', 'label': 'ramp', 'range_v': 2.0,
                         'coupling': 'DC'}}
        tail = {'file': 's_scope_tail.npz', 'start_offset_s': 3.8,
                'sample_interval_s': 1e-4, 'n_samples': 400,
                'function_generator': {'channel': 2, 'on_sent_after_start_s': 0.01,
                                       'on_done_after_start_s': 0.9}}

        # a record that is neither: ignored, not an error
        (folder / 'run_params_used.json').write_text('{"sections": {}}')
        (folder / 'x_session.json').write_text(
            json.dumps({'scope': scope, 'scope_tail': tail}))
        main = describe_capture(folder / 's_scope.npz')
        assert main['role'] == 'main' and len(main['channels']) == 2
        text = '\n'.join(main['lines'])
        assert '10 kS/s' in text and '±50 mV DC' in text and '±2 V DC' in text, text
        assert 'D ' in text and 'B ' in text and '4424A' in text, text
        tail_capture = describe_capture(folder / 's_scope_tail.npz')
        assert tail_capture['role'] == 'tail'
        tail_text = '\n'.join(tail_capture['lines'])
        assert '400 samples' in tail_text and 'CH2 on' in tail_text, tail_text
        assert 'starts 3.800 s' in tail_text, tail_text
        # the channel ranges of the tail are the main block's
        assert '±50 mV DC' in tail_text, tail_text
        fig, axes = build_figure(folder / 's_scope.npz')
        assert len(axes) == 2 and axes[0] is not axes[1], 'one y axis per channel'
        import matplotlib.pyplot as plt
        plt.close(fig)

        # a stand-alone scope capture, and a file with no record at all
        (folder / 'x_session.json').unlink()
        (folder / 'x_scope.json').write_text(json.dumps({'scope': scope}))
        assert describe_capture(folder / 's_scope.npz')['role'] == 'main'
        (folder / 'x_scope.json').unlink()
        bare = describe_capture(folder / 's_scope.npz')
        assert bare['role'] is None and 'no record' in bare['lines'][0]
        assert 'range unknown' in '\n'.join(bare['lines'])
        plt.close(build_figure(folder / 's_scope.npz')[0])
    print('plot_scope self-test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('path', nargs='?', help='a *_scope.npz or '
                                                '*_scope_tail.npz')
    parser.add_argument('--save', default=None, help='write the figure here '
                                                     'instead of opening a window')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.path:
        parser.error('give the .npz of a scope capture')
    import matplotlib
    if args.save:
        matplotlib.use('Agg')
    else:
        matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    fig, _ = build_figure(args.path)
    if args.save:
        fig.savefig(args.save, dpi=100)
        print(f'saved {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
