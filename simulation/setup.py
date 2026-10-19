"""Simulated synced capture: scope trace + aux ramp + camera frames from simulated axes.

    setup = SimulatedSetup.from_config()          # simulated axes of motion/axes_config.py
    cap = setup.capture()                         # dict: t, signal, aux, frames, windows, mask, truth
    alignment.metrics.evaluate_capture(cap['t'], cap['signal'], cap['aux'],
                                       cap['frames'], cap['windows'], cap['mask'])

What it reproduces (all seen in the 2026-10 sessions):
- the laser frequency follows the temperature drive through a thermal lag
  (first-order low-pass, tau ~0.35 s), so after every ramp turnaround the
  frequency stalls (no peaks), then the time-per-FSR is nonlinear;
- 0th-order peak height = coupling x peak_v, clipped at the scope range;
- transverse modes HG_nm at n*s_x + m*s_y with exact Gouy spacing (the
  convention of cavity_design's Cavity.plot_spectrum, extended to astigmatism):
  the first order is fed by the misalignment (1 - coupling), higher orders get
  dummy decaying amplitudes; see mode_amplitudes();
- camera frames integrate each mode's light over the exposure window; the
  0th order is an elliptical Gaussian whose centroid moves with the beam tilt;
- vibrations: a capture starting within `vibration_settle_s` of the last move
  of any axis gets a decaying frequency jitter (messy spectrum), which is why
  real captures must wait until the axes have settled.
"""

import time
from dataclasses import dataclass

import numpy as np

from simulation.cavity_physics import make_physics


@dataclass
class ScanModel:
    dt: float = 1e-4                # scope sample interval [s]
    ramp_hz: float = 0.05
    ramp_vpp: float = 3.0
    thermal_tau_s: float = 0.35     # laser frequency lag behind the drive
    fsr_per_volt: float = 7.4       # 0.3 V/s ramp -> ~0.45 s per FSR
    linewidth_fsr: float = 0.0022   # HWHM as FSR fraction (~1 ms at 0.45 s/FSR)
    peak_v: float = 1.6             # 0th-order height at perfect coupling [V]
    first_order_gain: float = 0.8   # first-order height per unit of lost coupling
    first_order_floor: float = 0.03 # first-order height even when aligned (mode matching) [V]
    max_order: int = 3              # highest transverse order n + m synthesized
    higher_order_decay: float = 1.0 # dummy amplitude decay per order (as in plot_spectrum)
    baseline_v: float = 0.02
    noise_v: float = 0.003
    range_v: float = 2.0            # scope clips here
    aux_noise_v: float = 0.003
    vibration_fsr: float = 0.05     # jitter amplitude right after a move [FSR]
    vibration_hz: float = 35.0
    vibration_decay_s: float = 0.4


@dataclass
class CameraModel:
    shape: tuple = (120, 128)
    frame_rate_hz: float = 14.28
    exposure_s: float = 0.07
    sigma_px: float = 8.0
    centre_px: tuple = (64.0, 60.0)     # (x, y) when aligned
    px_per_tilt: float = 8.0            # centroid shift per unit tilt (tilt is in NA units)
    counts_per_v_s: float = 2.0e4       # integrated scope volt-seconds -> peak counts
    background: float = 60.0
    noise: float = 4.0


def _lowpass(x, dt, tau):
    """First-order IIR low-pass (exact discretization)."""
    a = np.exp(-dt / tau)
    y = np.empty_like(x)
    acc = x[0]
    for i in range(x.size):        # ~1e5 samples: fast enough without numba
        acc = a * acc + (1 - a) * x[i]
        y[i] = acc
    return y


def mode_amplitudes(state, sc):
    """{(n, m): peak height [V]} of the HG_nm modes in the synthetic spectrum.

    Positions follow cavity_design's Cavity.plot_spectrum convention - exact
    spacing from the Gouy phase, order k at k x the transverse spacing - but
    per axis (HG_nm at n*s_x + m*s_y), so astigmatic (tilted) cavities work;
    plot_spectrum itself only plots and raises for astigmatic cavities.
    Amplitudes: the 0th order is physical (coupling); the first order is fed
    by the lost coupling, split between HG10/HG01 by the tilt direction;
    orders >= 2 get plot_spectrum-style dummy amplitudes, decaying as
    exp(-higher_order_decay * order) from the first-order floor."""
    lost = 1.0 - state.coupling
    tx2, ty2 = state.tilt_y ** 2, state.tilt_z ** 2
    share_x = tx2 / (tx2 + ty2) if (tx2 + ty2) > 1e-24 else 0.5
    amps = {(0, 0): sc.peak_v * state.coupling,
            (1, 0): sc.first_order_floor + sc.peak_v * sc.first_order_gain * lost * share_x,
            (0, 1): sc.first_order_floor + sc.peak_v * sc.first_order_gain * lost * (1 - share_x)}
    for order in range(2, sc.max_order + 1):
        h = sc.first_order_floor * np.exp(-sc.higher_order_decay * (order - 1))
        for n in range(order + 1):
            amps[(n, order - n)] = h
    return amps


def _lorentz_wrapped(nu, s, gamma):
    d = ((nu - s + 0.5) % 1.0) - 0.5
    return gamma ** 2 / (d ** 2 + gamma ** 2)


class SimulatedSetup:
    def __init__(self, raw_axes, physics=None, scan=None, camera=None, now=time.monotonic,
                 vibration_settle_s=None, seed=0):
        """raw_axes: {name: SimulatedAxis} (the raw drivers behind the SafeAxes)."""
        self.raw_axes = dict(raw_axes)
        self.physics = physics or make_physics()
        self.scan = scan or ScanModel()
        self.camera = camera or CameraModel()
        self._now = now
        self.vibration_settle_s = vibration_settle_s or {}
        self._rng = np.random.default_rng(seed)
        self.captures = 0

    @classmethod
    def from_config(cls, safe_axes=None, **kw):
        """Build from SafeAxes (their raw simulated drivers), or fresh simulated
        axes of motion/axes_config.py when none are given."""
        if safe_axes is None:
            from motion import build_axis
            from motion.axes_config import AXES
            safe_axes = {n: build_axis(n) for n, e in AXES.items() if e['driver'] == 'simulated'}
            for a in safe_axes.values():
                a.open()
        raw = {n: a._raw for n, a in safe_axes.items()}
        settle = {n: a.limits.vibration_settle_s for n, a in safe_axes.items()}
        setup = cls(raw, vibration_settle_s=settle, **kw)
        setup.safe_axes = safe_axes
        return setup

    def true_positions(self):
        return {n: a.true_position for n, a in self.raw_axes.items()}

    def mode_state(self):
        return self.physics.evaluate(self.true_positions())

    def aligned_first_order_s(self):
        """Signed first-order offset (FSR fraction, wrapped to [-0.5, 0.5)) of the
        aligned cavity: what the lab would put in MetricsConfig.first_order_s."""
        st = self.physics.evaluate(dict(self.physics.aligned))
        s = (st.s_x + st.s_y) / 2
        return ((s + 0.5) % 1.0) - 0.5

    def _vibration(self, t_abs):
        """Frequency jitter [FSR] from axes that moved recently."""
        sc = self.scan
        jitter = np.zeros_like(t_abs)
        for n, a in self.raw_axes.items():
            since = t_abs - a.last_move_t
            if since[0] < self.vibration_settle_s.get(n, 0.5) + 3 * sc.vibration_decay_s:
                env = np.where(since >= 0, np.exp(-np.clip(since, 0, None) / sc.vibration_decay_s), 0)
                jitter += sc.vibration_fsr * env * np.sin(2 * np.pi * sc.vibration_hz * since
                                                          + self._rng.uniform(0, 2 * np.pi))
        return jitter

    def capture(self, duration=5.6, with_frames=True):
        sc, cam = self.scan, self.camera
        t0_abs = self._now()
        state = self.mode_state()
        pre = 6 * sc.thermal_tau_s
        t_full = np.arange(-pre, duration, sc.dt)
        t_abs = t0_abs + t_full
        phase = (t_abs * sc.ramp_hz) % 1.0
        drive = sc.ramp_vpp / 2 * (4 * np.abs(phase - 0.5) - 1)
        nu_drive = _lowpass(drive, sc.dt, sc.thermal_tau_s) * sc.fsr_per_volt
        keep = t_full >= 0
        t = t_full[keep]
        nu = nu_drive[keep] + self._vibration(t_abs[keep])
        aux = drive[keep] + self._rng.normal(0, sc.aux_noise_v, t.size)

        g = sc.linewidth_fsr
        modes = []     # (name, trace [V])
        if state.valid:
            for (n, m), h in mode_amplitudes(state, sc).items():
                modes.append((f'HG{n}{m}', h * _lorentz_wrapped(nu, n * state.s_x + m * state.s_y, g)))
        clean = sc.baseline_v + sum(m[1] for m in modes) if modes else np.full(t.size, sc.baseline_v)
        signal = np.minimum(clean + self._rng.normal(0, sc.noise_v, t.size), sc.range_v)

        out = {'t': t, 'signal': signal, 'aux': aux, 'truth': state,
               'range_v': sc.range_v, 't0_abs': t0_abs,
               'vibrating': bool(np.any(np.abs(self._vibration(t_abs[keep][:1])) > 0))}
        if with_frames:
            out.update(self._frames(t, modes, state, duration))
        self.captures += 1
        self.last_capture = out           # for visualizations that replay what was measured
        return out

    def _frames(self, t, modes, state, duration):
        cam = self.camera
        starts = np.arange(0.01, duration - cam.exposure_s, 1.0 / cam.frame_rate_hz)
        windows = np.column_stack([starts, starts + cam.exposure_s])
        h, w = cam.shape
        yy, xx = np.indices(cam.shape, dtype=float)
        e = state.ellipticity if state.valid and np.isfinite(state.ellipticity) else 1.0
        sx, sy = cam.sigma_px / np.sqrt(e), cam.sigma_px * np.sqrt(e)
        cx = cam.centre_px[0] + cam.px_per_tilt * (state.tilt_y if state.valid else 0)
        cy = cam.centre_px[1] + cam.px_per_tilt * (state.tilt_z if state.valid else 0)
        u, v = (xx - cx) / sx, (yy - cy) / sy
        gauss = np.exp(-(u ** 2 + v ** 2) / 2)
        from numpy.polynomial.hermite import hermval

        def hg(n, m):     # HG_nm intensity in the (u, v) beam coordinates
            hx = hermval(u, [0] * n + [1])
            hy = hermval(v, [0] * m + [1])
            norm = 2.0 ** (n + m) * np.prod(range(1, n + 1)) * np.prod(range(1, m + 1))
            return hx ** 2 * hy ** 2 * gauss / norm

        templates = {name: hg(int(name[2]), int(name[3])) for name, _ in modes}
        cum = {name: np.concatenate([[0.0], np.cumsum(trace) * (t[1] - t[0])]) for name, trace in modes}
        idx = np.searchsorted(t, windows)
        frames = np.empty((len(starts), h, w), dtype=np.uint16)
        for k, (a, b) in enumerate(idx):
            img = np.full(cam.shape, cam.background)
            for name, c in cum.items():
                img = img + cam.counts_per_v_s * (c[b] - c[a]) * templates[name]
            img = img + self._rng.normal(0, cam.noise, cam.shape)
            frames[k] = np.clip(img, 0, 65535).astype(np.uint16)
        mask = ((xx - cam.centre_px[0]) ** 2 + (yy - cam.centre_px[1]) ** 2) <= (5 * cam.sigma_px) ** 2
        return {'frames': frames, 'windows': windows, 'mask': mask}


def _self_test():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from alignment.metrics import MetricsConfig, evaluate_capture
    from simulation.cavity_physics import AnalyticPhysics

    class Clock:
        t = 1000.0

        def now(self):
            return self.t

    clock = Clock()
    from motion.simulated import SimulatedAxis
    axes = {'sim_mirror_phi': SimulatedAxis('sim_mirror_phi', start=37.5, now=clock.now),
            'sim_mirror_theta': SimulatedAxis('sim_mirror_theta', start=37.5, now=clock.now),
            'sim_lens_x': SimulatedAxis('sim_lens_x', units='um', start=0.0, now=clock.now)}
    for a in axes.values():
        a.open()
    physics = AnalyticPhysics(split_per_unit={'sim_mirror_phi': 2e-3, 'sim_mirror_theta': 0.0})
    setup = SimulatedSetup(axes, physics=physics, now=clock.now)
    cfg = MetricsConfig(saturation_v=0.995 * setup.scan.range_v)

    cap = setup.capture()
    m = evaluate_capture(cap['t'], cap['signal'], cap['aux'], cap['frames'], cap['windows'],
                         cap['mask'], cfg)
    aligned_I = m.value('intensity')
    assert aligned_I is not None and abs(aligned_I - 1.6) < 0.1, m.metrics['intensity']
    assert m.value('gaussian_overlap') > 0.95, m.metrics['gaussian_overlap']

    axes['sim_mirror_phi'].move_to(37.5 + 6)      # split 0.012 FSR (~3 linewidths)
    axes['sim_mirror_theta'].move_to(37.5 + 5)      # both first-order modes get light
    clock.t += 30                                    # well after the move: settled
    m2 = evaluate_capture(**{k: v for k, v in setup.capture().items()
                             if k in ('t', 'signal', 'aux', 'frames', 'windows', 'mask')}, cfg=cfg)
    assert m2.value('intensity') < aligned_I * 0.8, (m2.metrics['intensity'], aligned_I)
    assert m2.value('split_fsr') is not None and m2.value('split_fsr') > 0, m2.metrics['split_fsr']

    axes['sim_mirror_phi'].move_to(37.5)          # capture right after a move: vibrating
    cap3 = setup.capture()
    assert cap3['vibrating']
    print(f'setup self-test OK (aligned I={aligned_I:.3f} V, misaligned I={m2.value("intensity"):.3f} V,'
          f' split={m2.value("split_fsr"):.4f} FSR)')


if __name__ == '__main__':
    _self_test()
