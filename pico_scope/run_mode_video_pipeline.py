"""Run the synchronized-video-spectrum pipeline end to end.

    python pico_scope/run_mode_video_pipeline.py

Runs, in order, on the newest capture:

    mode_video_capture.py     locate the mode and record it
    mode_video_sync.py        refine the frame/scope time offset
    mode_video_sync_show.py   open the result in the viewer

Each step runs with no arguments, so it uses whatever ACTION/SESSION/etc. are
set at the top of that script. Stops after the first step that exits non-zero.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ['mode_video_capture.py', 'mode_video_sync.py', 'mode_video_sync_show.py']


def main():
    for step in STEPS:
        print(f'\n=== {step} ===')
        result = subprocess.run([sys.executable, str(HERE / step)])
        if result.returncode != 0:
            sys.exit(f'{step} exited with code {result.returncode}; stopping.')


if __name__ == '__main__':
    main()
