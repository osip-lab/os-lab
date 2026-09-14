"""Run the synchronized-video-spectrum pipeline end to end.

    python pico_scope/run_mode_video_pipeline.py
    python pico_scope/run_mode_video_pipeline.py --config my_experiment.py

Runs, in order:

    mode_video_capture.py     locate the mode and record it
    mode_video_sync.py        refine the frame/scope time offset
    mode_video_sync_show.py   open the result in the viewer

Every step runs with no arguments of its own, so each one does what the config
file says - which is pico_scope/run_config_local.py unless --config names
another. That file is git-ignored and holds the run parameters of all four
pipeline scripts; see pico_scope/run_config.py. It reaches the subprocesses
through the MODE_VIDEO_CONFIG environment variable rather than an argument,
because the scripts have to read it at import, before their own argparse runs.

The session folder the capture actually wrote is scraped from its output (the
'SESSION_PATH=...' line it prints) and passed on explicitly as --session to the
next two steps, since a capture saved to a folder the user just pasted (rather
than the fixed local bank) is no longer the "newest capture" those steps would
find on their own. Stops after the first step that exits non-zero.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pico_scope import run_config  # noqa: E402

STEPS = ['mode_video_capture.py', 'mode_video_sync.py', 'mode_video_sync_show.py']
SESSION_MARKER = 'SESSION_PATH='


def step_environment(config_path):
    """The environment the steps run in, carrying the config they share.

    One resolved path for the whole run, so that a --config given here and a
    relative path cannot be re-resolved differently by a subprocess that
    happens to have another working directory.
    """
    environment = dict(os.environ)
    environment[run_config.ENV_VAR] = str(config_path)
    return environment


def run_capture_step(step, environment):
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
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        env=environment)
    session_path = None
    for line in process.stdout:
        print(line, end='')
        if line.startswith(SESSION_MARKER):
            session_path = line[len(SESSION_MARKER):].strip()
    process.wait()
    return process.returncode, session_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default=None,
                        help='config file to run from, instead of '
                             'pico_scope/run_config_local.py')
    args = parser.parse_args()

    # Resolved here, and created from the template here if it does not exist
    # yet, so the notice about a new config file is printed once at the top of
    # the run instead of from inside the first subprocess.
    config_path = run_config.config_path(args.config)
    if config_path == run_config.LOCAL_PATH:
        run_config.ensure_local_config()
    if not config_path.is_file():
        sys.exit(f'no config file at {config_path}')
    print(f'config: {config_path}')
    environment = step_environment(config_path)

    session_path = None
    for step in STEPS:
        print(f'\n=== {step} ===')
        if step == 'mode_video_capture.py':
            returncode, session_path = run_capture_step(step, environment)
        else:
            if session_path is None:
                sys.exit(f'{step}: no session path was captured from '
                         f'mode_video_capture.py - it must have printed a '
                         f'{SESSION_MARKER!r} line for this to continue.')
            command = [sys.executable, str(HERE / step), '--session', session_path]
            returncode = subprocess.run(command, env=environment).returncode
        if returncode != 0:
            sys.exit(f'{step} exited with code {returncode}; stopping.')


if __name__ == '__main__':
    main()
