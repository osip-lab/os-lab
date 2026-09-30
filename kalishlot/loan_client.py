"""Borrow devices from a running kalishlot server, for standalone scripts.

Only one program can own a camera or a scope, so a script that needs one
kalishlot holds would fail to open it. Instead it borrows it:

    from kalishlot.loan_client import borrow_from_kalishlot

    with borrow_from_kalishlot(lambda d: d['type'] == 'picoscope',
                               borrower='mode_video_capture.py'):
        ...  # open and use the scope as usual

On entry every open kalishlot device the predicate accepts is lent: kalishlot
closes it (freeing the hardware) and its boxes in the browser show who has it.
On exit - normal, exception or Ctrl+C - each is returned: kalishlot re-opens it
with the settings it had, and the boxes re-attach by themselves. See "loans"
in server.py.

When kalishlot is not running there is nothing to borrow and the block simply
runs. If the script is killed before it can return a device, the device stays
closed until "reconnect" is pressed in its box.

Standard library only, so any script can use it without kalishlot's
dependencies installed.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager

KALISHLOT_URL = 'http://localhost:8090'
TIMEOUT_S = 15         # lending waits for the hardware to be released
RETURN_TRIES = 5       # a device the borrower has only just let go of can
RETURN_RETRY_S = 1.0   # still be busy for a moment; the return is retried


class KalishlotError(RuntimeError):
    pass


def _call(url, path, method='GET', body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f'{url}{path}', data=data, method=method,
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read()).get('detail', error.reason)
        except Exception:
            detail = error.reason
        raise KalishlotError(f'{method} {path}: {detail}') from error


def _device_path(device_id):
    return '/api/devices/' + urllib.parse.quote(device_id, safe=':/')


def open_devices(url=KALISHLOT_URL):
    """The devices kalishlot holds, or None when it is not running."""
    try:
        return _call(url, '/api/devices')
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return None


def lend(device_id, borrower, url=KALISHLOT_URL):
    _call(url, _device_path(device_id) + '/lend', 'POST', {'borrower': borrower})


def give_back(device_id, url=KALISHLOT_URL):
    """Return a lent device; retried while its hardware is still busy.
    Returns False when the loan was cancelled meanwhile (box closed, idle
    shutdown) - then the device is meant to stay closed."""
    for attempt in range(RETURN_TRIES):
        try:
            _call(url, _device_path(device_id) + '/return', 'POST')
            return True
        except KalishlotError as error:
            if 'not on loan' in str(error):
                return False
            if attempt == RETURN_TRIES - 1:
                raise
            time.sleep(RETURN_RETRY_S)


@contextmanager
def borrow_from_kalishlot(accept, borrower, url=KALISHLOT_URL, log=print):
    """Lend every open kalishlot device `accept(describe)` is true for, for
    the duration of the block; yields {device_id: describe} of those lent,
    as each device was just before the loan (its settings, a camera's ROI)."""
    devices = open_devices(url)
    wanted = [d for d in devices or [] if accept(d)]
    lent = {}
    try:
        for device in wanted:
            device_id = device['device_id']
            lend(device_id, borrower, url)
            lent[device_id] = device
            log(f'  kalishlot: borrowed {device_id}')
        yield lent
    finally:
        for device_id in lent:
            try:
                if give_back(device_id, url):
                    log(f'  kalishlot: returned {device_id}')
                else:
                    log(f'  kalishlot: {device_id} was closed meanwhile, '
                        f'left closed')
            except Exception as error:
                log(f'  kalishlot: could not return {device_id} ({error}); '
                    f'press "reconnect" in its box')
