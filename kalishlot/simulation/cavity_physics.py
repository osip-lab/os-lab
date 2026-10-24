"""Cavity physics for the simulator: motor positions -> mode state.

Built on the cavity-design project (cavity_design), reached through
utilities/cavity_design_bridge.py (path in local_config.py). For every set of
element perturbations it builds the perturbed cavity and returns:

- coupling     power coupling of the fixed input beam into the 0th order:
               |overlap(aligned mode, perturbed mode)|^2. The input beam is the
               aligned cavity's mode, i.e. perfectly mode-matched at zero.
- s_x, s_y     positions of the HG10 / HG01 modes relative to the 0th order,
               as FSR fractions, from the round-trip Gouy phase per axis
               (s = -2 * sum(phi_axis) / 2pi, mod 1). Their difference is the
               astigmatic first-order splitting.
- ellipticity  w_min / w_max of the perturbed mode at its waist.
- tilt_y/_z    beam direction change in units of the mode's divergence (NA):
               the camera sees this as a centroid shift in beam widths.
- valid        False when the perturbed cavity has no stable mode (NaN): the
               light then does not resonate at all.

`AnalyticPhysics` is a dependency-free stand-in (quadratic coupling loss,
linear astigmatism) for tests and for machines without cavity-design.

Axes follow cavity_design: the optical axis is x. A mirror's `phi` tilt turns
it in the xy plane (the beam is deflected along y), `theta` in the xz plane
(along z); an element's `x` moves it along the optical axis. tilt_y / tilt_z
are therefore the transverse beam-angle components; the camera's columns are
lab y and its rows lab z. s_x / s_y belong to the perturbed mode's two
transverse principal axes (cavity_design's per-axis Gouy phase), i.e. the
HG10 and HG01 families, whatever their orientation.

This module has no os-lab dependencies beyond the bridge, so the cavity-design
project can import it as well (see kalishlot/MECHANICAL_CONTROL.md).
"""

import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]


@dataclass
class ModeState:
    coupling: float
    s_x: float
    s_y: float
    ellipticity: float = 1.0
    tilt_y: float = 0.0
    tilt_z: float = 0.0
    valid: bool = True

    @property
    def split(self):
        """Signed first-order splitting s_x - s_y, wrapped to [-0.5, 0.5) FSR."""
        return ((self.s_x - self.s_y + 0.5) % 1.0) - 0.5

    def to_dict(self):
        d = asdict(self)
        d['split'] = self.split
        return d


INVALID = ModeState(0.0, float('nan'), float('nan'), float('nan'), 0.0, 0.0, False)


@dataclass(frozen=True)
class DoF:
    """How one motor axis perturbs the cavity: element, parameter, SI per axis unit."""
    element_index: int
    parameter: str          # 'x', 'y', 'phi', ... as cavity_design names them
    si_per_unit: float      # e.g. 1e-6 rad per volt for a piezo tilt


def example_cavity():
    """The mirror-lens-mirror cavity of the cavity-calculations skill example
    (examples/perturbation_tolerance_analysis.py). Replace with the lab's
    actual cavity once the motorized elements are chosen."""
    from cavity_design import (Cavity, SphericalMirror, LASER_OPTIK_MIRROR,
                               EKSMA_LENS_20MM_ASPHERIC, DUMMY_LENS)
    return Cavity(
        elements=[
            LASER_OPTIK_MIRROR.to_position(np.array([-0.005, 0.0, 0.0])),
            EKSMA_LENS_20MM_ASPHERIC.to_position(np.array([0.017623230771841976, 0.0, 0.0])),
            DUMMY_LENS.to_position(np.array([0.03305723077184197, 0.0, 0.0])),
            SphericalMirror(name='End mirror', radius=0.2, outwards_normal=np.array([1.0, 0.0, 0.0]),
                            center=np.array([0.4511688875799871, 0.0, 0.0]), curvature_sign=-1,
                            diameter=0.0254,
                            material_properties=LASER_OPTIK_MIRROR.material_properties),
        ],
        standing_wave=True, lambda_0_laser=1.064e-06, t_is_trivial=True, p_is_trivial=True,
        use_paraxial_ray_tracing=True)


# Example DoF map for the simulated axes in motion/axes_config.py. The example
# cavity is near the stability edge: ~10 urad of end-mirror tilt or ~10 um of
# lens travel already costs a few percent of coupling, and ~3x that loses the mode.
EXAMPLE_DOFS = {
    'sim_mirror_phi': DoF(3, 'phi', 1e-6),      # end mirror, xy-plane tilt: 1 V -> 1 urad (37.5 V = aligned)
    'sim_mirror_theta': DoF(3, 'theta', 1e-6),  # end mirror, xz-plane tilt
    'sim_lens_x': DoF(1, 'x', 1e-7),            # lens along the optical axis: 1 um of stage -> 0.1 um
    'sim_lens_y': DoF(1, 'y', 1e-7),            # lens across the optical axis (in the xy plane)
}
EXAMPLE_ALIGNED = {'sim_mirror_phi': 37.5, 'sim_mirror_theta': 37.5, 'sim_lens_x': 0.0, 'sim_lens_y': 0.0}
# With sim_lens_y the end-mirror phi tilt has a partner: a lens shift re-centres the mode, so the
# coupling loss is a narrow ridge in (phi, lens_y), curvature ratio ~2400, soft direction about
# 0.98 phi + 0.18 lens_y in axis units (finite differences, 2026-10-24). Moving phi alone looks
# much worse than moving along the ridge: the case alignment/nested.py is for.


def _gouy_s(cavity):
    g = np.sum([a.acquired_gouy_phase_per_axis for a in cavity.arms], axis=0)
    return (-2 * np.asarray(g, dtype=float) / (2 * np.pi)) % 1.0


class CavityPhysics:
    """Mode state of a cavity whose elements are perturbed by motor axes."""

    def __init__(self, cavity_factory=None, dofs=None, aligned=None, cache_decimals=6):
        repo = str(_REPO)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from utilities.cavity_design_bridge import add_cavity_design_to_path
        add_cavity_design_to_path()
        import cavity_design  # noqa: F401  (fail early if the project is missing)
        self._cd = cavity_design
        self.reference = (cavity_factory or example_cavity)()
        self.dofs = dict(EXAMPLE_DOFS if dofs is None else dofs)
        self.aligned = dict(EXAMPLE_ALIGNED if aligned is None else aligned)
        self._s0 = _gouy_s(self.reference)
        mp = self.reference.arms[0].mode_parameters
        self._k0 = np.asarray(mp.k_vector, dtype=float)
        self._na = float(np.mean(mp.NA))
        self._cache = {}
        self._decimals = cache_decimals

    def evaluate(self, positions):
        """positions: {axis name: position in axis units}. Missing axes = aligned."""
        offsets = self._offsets(positions)
        key = tuple(round(offsets[n], self._decimals) for n in sorted(offsets))
        if key not in self._cache:
            if len(self._cache) > 20000:
                self._cache.clear()
            self._cache[key] = self._evaluate(offsets)
        return self._cache[key]

    def _offsets(self, positions):
        return {n: float(positions.get(n, self.aligned.get(n, 0.0))) - self.aligned.get(n, 0.0)
                for n in self.dofs}

    def _pointers(self, offsets):
        cd = self._cd
        return [cd.PerturbationPointer(element_index=d.element_index, parameter_name=d.parameter,
                                       perturbation_value=offsets[n] * d.si_per_unit)
                for n, d in self.dofs.items() if offsets[n] != 0]

    def perturbed_cavity(self, positions):
        """The cavity_design Cavity for these motor positions (the reference when aligned),
        e.g. for Cavity.plot or plot_2_cavity_perturbation_overlap."""
        pointers = self._pointers(self._offsets(positions))
        if not pointers:
            return self.reference
        with np.errstate(all='ignore'):
            return self._cd.perturb_cavity(self.reference, pointers)

    def _evaluate(self, offsets):
        cd = self._cd
        pointers = self._pointers(offsets)
        if not pointers:
            return ModeState(1.0, float(self._s0[0]), float(self._s0[1]), 1.0, 0.0, 0.0, True)
        with np.errstate(all='ignore'):
            try:
                pc = cd.perturb_cavity(self.reference, pointers)
                ov = cd.calculate_cavities_overlap(self.reference, pc)
                s = _gouy_s(pc)
                mp = pc.arms[0].mode_parameters
                w0 = np.asarray(mp.w_0, dtype=float)
                k = np.asarray(mp.k_vector, dtype=float)
            except Exception:
                return INVALID
        if not (np.isfinite(ov) and np.all(np.isfinite(s)) and np.all(np.isfinite(w0))):
            return INVALID
        dk = k - self._k0
        return ModeState(coupling=float(abs(ov) ** 2), s_x=float(s[0]), s_y=float(s[1]),
                         ellipticity=float(w0.min() / w0.max()),
                         tilt_y=float(dk[1] / self._na), tilt_z=float(dk[2] / self._na),
                         valid=True)


class AnalyticPhysics:
    """Dependency-free stand-in: coupling = exp(-sum((d/scale)^2)), astigmatism
    linear in the tilt axes, tilt proportional to the offsets. Same interface.

    curvature   optional (names, matrix M): coupling = exp(-d M d) over those axes
                instead of the separable scales, e.g. a rotated, elongated quadratic
                (a narrow ridge, see ridge_curvature).
    signed_split optional {axis: FSR per unit}: the splitting is sum(k * d), signed,
                so one axis can cancel the astigmatism another one makes; it
                replaces split_per_unit (which grows with |d| of every axis)."""

    def __init__(self, aligned=None, scales=None, s0=0.3, split_per_unit=None,
                 tilt_per_unit=None, lost_beyond=3.0, curvature=None, signed_split=None):
        self.aligned = dict(EXAMPLE_ALIGNED if aligned is None else aligned)
        self.scales = dict({'sim_mirror_phi': 8.0, 'sim_mirror_theta': 8.0, 'sim_lens_x': 60.0,
                            'sim_lens_y': 15.0}
                           if scales is None else scales)
        self.curvature = None
        if curvature is not None:
            names, m = curvature
            self.curvature = (list(names), np.asarray(m, dtype=float))
        self.signed_split = dict(signed_split) if signed_split else None
        self.s0 = s0
        self.split_per_unit = dict({'sim_mirror_phi': 4e-4, 'sim_mirror_theta': -4e-4}
                                   if split_per_unit is None else split_per_unit)
        self.tilt_per_unit = dict({'sim_mirror_phi': (0.03, 0.0), 'sim_mirror_theta': (0.0, 0.03)}
                                  if tilt_per_unit is None else tilt_per_unit)
        self.lost_beyond = lost_beyond

    def evaluate(self, positions):
        d = {n: float(positions.get(n, a)) - a for n, a in self.aligned.items()}
        coupled = set(self.curvature[0]) if self.curvature else set()
        r2 = sum((d[n] / self.scales[n]) ** 2 for n in self.scales if n in d and n not in coupled)
        if self.curvature:
            v = np.array([d.get(n, 0.0) for n in self.curvature[0]])
            r2 += float(v @ self.curvature[1] @ v)
        if r2 > self.lost_beyond ** 2:
            return INVALID
        if self.signed_split is not None:
            split = sum(d.get(n, 0.0) * k for n, k in self.signed_split.items())
        else:
            split = sum(abs(d.get(n, 0.0)) * k for n, k in self.split_per_unit.items())
        tx = sum(d.get(n, 0.0) * v[0] for n, v in self.tilt_per_unit.items())
        ty = sum(d.get(n, 0.0) * v[1] for n, v in self.tilt_per_unit.items())
        return ModeState(float(np.exp(-r2)), self.s0 + split / 2, self.s0 - split / 2,
                         float(1.0 / (1.0 + 5 * abs(split))), tx, ty, True)


def ridge_curvature(names, soft_direction, soft_width, stiff_width):
    """(names, M) for AnalyticPhysics(curvature=...): coupling falls by 1/e at
    soft_width along soft_direction and at stiff_width across it (two axes)."""
    u = np.asarray(soft_direction, dtype=float)
    u /= np.linalg.norm(u)
    w = np.array([-u[1], u[0]])
    return list(names), np.outer(u, u) / soft_width ** 2 + np.outer(w, w) / stiff_width ** 2


def make_physics(prefer_cavity_design=True, **kw):
    """CavityPhysics when the cavity-design project is reachable, else AnalyticPhysics."""
    if prefer_cavity_design:
        try:
            return CavityPhysics(**kw)
        except Exception as e:      # missing project / path: fall back, say so
            print(f'cavity_physics: cavity-design unavailable ({type(e).__name__}: {e}); '
                  f'using AnalyticPhysics')
    return AnalyticPhysics()


def _self_test():
    a = AnalyticPhysics()
    st = a.evaluate(EXAMPLE_ALIGNED)
    assert st.valid and abs(st.coupling - 1) < 1e-12 and abs(st.split) < 1e-12
    off = a.evaluate({**EXAMPLE_ALIGNED, 'sim_mirror_phi': 37.5 + 8})
    assert off.valid and abs(off.coupling - np.exp(-1)) < 1e-9 and off.split > 0
    assert not a.evaluate({**EXAMPLE_ALIGNED, 'sim_lens_x': 1000}).valid
    names, m = ridge_curvature(['sim_mirror_phi', 'sim_lens_y'], (1.0, 0.2), 25.0, 1.5)
    r = AnalyticPhysics(curvature=(names, m))
    along = r.evaluate({**EXAMPLE_ALIGNED, 'sim_mirror_phi': 37.5 + 10, 'sim_lens_y': 2.0})
    phi_only = r.evaluate({**EXAMPLE_ALIGNED, 'sim_mirror_phi': 37.5 + 10})
    assert along.coupling > 0.8 > 0.2 > phi_only.coupling, (along.coupling, phi_only.coupling)
    sp = AnalyticPhysics(signed_split={'sim_mirror_phi': 1e-3, 'sim_lens_y': -1e-3})
    assert abs(sp.evaluate({**EXAMPLE_ALIGNED, 'sim_mirror_phi': 40.5, 'sim_lens_y': 3.0}).split) < 1e-12
    try:
        p = CavityPhysics()
    except Exception as e:
        print(f'cavity-design not reachable, skipped CavityPhysics checks ({e})')
    else:
        z = p.evaluate(EXAMPLE_ALIGNED)
        assert z.valid and abs(z.coupling - 1) < 1e-9, z
        t = p.evaluate({**EXAMPLE_ALIGNED, 'sim_mirror_phi': 37.5 + 10})
        assert t.valid and 0.95 < t.coupling < 0.995 and abs(t.split) > 1e-4, t
        assert not p.evaluate({**EXAMPLE_ALIGNED, 'sim_mirror_phi': 37.5 + 100}).valid
        print(f'CavityPhysics: aligned s = {z.s_x:.4f}; 10 urad tilt -> coupling {t.coupling:.4f}, '
              f'split {t.split:+.5f} FSR, tilt {t.tilt_y:+.3f} NA')
    print('cavity_physics self-test OK')


if __name__ == '__main__':
    _self_test()
