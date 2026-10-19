"""Virtual axes: a fixed linear combination of real axes, moved as one DoF.

Examples: "beam walk" = tilt M1 by +d and M2 by -k*d; translation + tilt about
a pivot. Moving a virtual axis by `delta` moves every component axis by
`coefficient * delta`. All component targets are checked BEFORE any axis
moves, so a refused component means nothing moved. Each component then ramps
through its own SafeAxis, so every per-axis rule (range, step, slew, frozen,
E-stop, approach) still applies.

The virtual position is the accumulated delta since creation (0 at creation).
"""

from motion.safety import SafetyError


class VirtualAxis:
    def __init__(self, name, components, units='a.u.'):
        """`components`: list of (SafeAxis, coefficient)."""
        if not components:
            raise ValueError('a virtual axis needs at least one component')
        self.name = name
        self.units = units
        self.components = [(a, float(c)) for a, c in components]
        self._position = 0.0

    @property
    def position(self):
        return self._position

    @property
    def frozen(self):
        return any(a.frozen for a, c in self.components if c)

    @property
    def ready_at(self):
        return max(a.ready_at for a, _ in self.components)

    def check_delta(self, delta):
        """Raise SafetyError if any component would refuse. Moves nothing."""
        for axis, c in self.components:
            if c == 0:
                continue
            if axis.commanded is None:
                raise SafetyError(f'{self.name}: {axis.name} position unknown')
            axis.check_target(axis.commanded + c * float(delta))

    def move_by(self, delta, who):
        delta = float(delta)
        self.check_delta(delta)
        results = []
        for axis, c in self.components:
            if c:
                results.append(axis.move_by(c * delta, who=f'{who} via {self.name}'))
        self._position += delta
        return results

    def move_to(self, value, who):
        return self.move_by(float(value) - self._position, who)

    def wait_until_settled(self):
        for axis, _ in self.components:
            axis.wait_until_settled()
