"""Run the synchronized-video-spectrum pipeline end to end.

    python pico_scope/run_mode_video_pipeline.py

Runs, in order:

    mode_video_capture.py     locate the mode and record it
    mode_video_sync.py        refine the frame/scope time offset
    mode_video_sync_show.py   open the result in the viewer

mode_video_capture.py runs with no arguments, so it uses whatever
ACTION/CAMERA/etc. are set at the top of that script - including where it
prompts to save the capture (see PROMPT_FOR_OUTPUT_ROOT there). The session
folder it actually wrote is scraped from its output (the 'SESSION_PATH=...'
line it prints) and passed on explicitly as --session to the next two steps,
since a capture saved to a folder the user just pasted (rather than the
fixed local bank) is no longer the "newest capture" those steps would find
on their own. Stops after the first step that exits non-zero.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ['mode_video_capture.py', 'mode_video_sync.py', 'mode_video_sync_show.py']
SESSION_MARKER = 'SESSION_PATH='


def run_capture_step(step):
    """Run mode_video_capture.py, relaying its output live and returning the
    session path it printed via the SESSION_PATH= marker (see its capture()/
    capture_synchronized()), or None if it never printed one.

    -u disables Python's output buffering: stdout is piped here (not a
    terminal), which would otherwise block-buffer and delay prompts like
    "Press Enter to record the burst..." until the buffer filled or the
    process exited.
    """
    process = subprocess.Popen(
        [sys.executable, '-u', str(HERE / step)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    session_path = None
    for line in process.stdout:
        print(line, end='')
        if line.startswith(SESSION_MARKER):
            session_path = line[len(SESSION_MARKER):].strip()
    process.wait()
    return process.returncode, session_path


def main():
    session_path = None
    for step in STEPS:
        print(f'\n=== {step} ===')
        if step == 'mode_video_capture.py':
            returncode, session_path = run_capture_step(step)
        else:
            if session_path is None:
                sys.exit(f'{step}: no session path was captured from '
                         f'mode_video_capture.py - it must have printed a '
                         f'{SESSION_MARKER!r} line for this to continue.')
            command = [sys.executable, str(HERE / step), '--session', session_path]
            returncode = subprocess.run(command).returncode
        if returncode != 0:
            sys.exit(f'{step} exited with code {returncode}; stopping.')


if __name__ == '__main__':
    main()
