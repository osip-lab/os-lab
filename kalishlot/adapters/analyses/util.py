"""Shared bits for the server-side analysis extensions.

Importing this module also puts the repo root on sys.path, so the analysis
modules can import the shared math package (pico_scope.mode_analysis) that
the offline scripts use — the math is never duplicated here.
"""

import sys
import threading
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pico_scope.mode_analysis import get_na_interpolators  # noqa: E402


def finite_or_none(values):
    """Fit params/errors can be inf/NaN (singular covariance); neither
    survives JSON, so send them as null."""
    return {key: (value if np.isfinite(value) else None)
            for key, value in values.items()}


# ------------------------------------------------------- NA interpolators
# Building them runs the whole cavity-design lens-position simulation (slow),
# so it happens once, in the background, as soon as an analysis needing them
# is constructed — by the time the first fit is clicked they are ready.
# get_na_interpolators() caches but does not lock, so all access goes through
# na_interpolators() here: a fit clicked mid-build waits for the build instead
# of starting a second one.
_na_lock = threading.Lock()
_warm_started = False


def na_interpolators():
    """Thread-safe (mode_spacing_interp, df_over_fsr_interp, error)."""
    with _na_lock:
        return get_na_interpolators()


def warm_na_interpolators():
    """Start the one-time background build; idempotent, never blocks."""
    global _warm_started
    if _warm_started:
        return
    _warm_started = True
    threading.Thread(target=na_interpolators, daemon=True,
                     name='na-interpolators-warmup').start()
