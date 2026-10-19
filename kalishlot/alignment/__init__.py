"""Optical alignment metrics and the alignment loop.

Pure computation: numpy/scipy only, no GUI, no device I/O. Works on a capture
in memory (scope trace + optional camera frames) or on a saved synced session.
See kalishlot/MECHANICAL_CONTROL.md for the design.

    sweeps.py     split a capture into monotonic scan sweeps (aux ramp channel)
    spectrum.py   0th-order peaks, local FSR, first-order family, splitting
    camera.py     Gaussian-likeness of a frame, first-order lobe check
    metrics.py    one capture -> metric values with validity flags
    validate_marks.py   offline check against hand-made *.modemarks.json
"""
