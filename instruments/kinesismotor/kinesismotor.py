from pylablib.devices import Thorlabs


class KinesisMotor:
    """Thin wrapper around pylablib's ``Thorlabs.KinesisMotor``.

    ``motor_type`` is passed straight through as pylablib's ``scale=`` argument, so
    it accepts anything pylablib does: ``"step"`` (raw steps), ``"stage"``
    (autodetect), a specific stage model name (e.g. ``"MTS25-Z8"``, ``"K10CR1"``,
    ``"PRM1-Z8"``, ``"NR360"``, ...), a numeric steps-per-unit factor, or a
    ``(pos, vel, acc)`` scale tuple.

    Behavior is driven by the *actual* units pylablib reports
    (:attr:`units`, from ``get_scale_units()``) rather than the ``motor_type``
    string, so it generalizes to any supported stage:

    - Linear stages (units ``'m'``) are commanded/reported in **mm**.
    - Rotary stages (units ``'deg'``) are commanded/reported in **deg**, and get
      homing-via-limit-switch configured automatically (as rotary stages need it
      to establish an absolute reference after power-up).
    - ``"step"`` stages are left in raw steps.
    """

    def __init__(self, serial: str, motor_type: str = 'stage'):
        self.km = Thorlabs.KinesisMotor(serial, scale=motor_type)
        self.motor_type = motor_type
        self.units = self.km.get_scale_units()
        self.scale = 1e-3 if self.units == 'm' else 1

        if self.units == 'deg':
            self.km.setup_limit_switch(hw_kind_cw='make_home', hw_kind_ccw='make_home')
            self.km.setup_homing(velocity=10)

    def home(self, force: bool = False):
        self.km.home(sync=True, force=force)

    def is_homed(self) -> bool:
        return self.km.is_homed()

    def move_to(self, position: float):
        self.km.move_to(position * self.scale)
        self.km.wait_move()

    def move_by(self, position: float):
        self.km.move_by(position * self.scale)
        self.km.wait_move()

    def get_position(self) -> float:
        return self.km.get_position() / self.scale

    def set_velocity(self, max_velocity: float = None, acceleration: float = None):
        if max_velocity is not None:
            max_velocity *= self.scale
        if acceleration is not None:
            acceleration *= self.scale
        self.km.setup_velocity(max_velocity=max_velocity, acceleration=acceleration)

    def setup_homing(self, **kwargs):
        """Passthrough to pylablib's ``setup_homing`` (native units)."""
        self.km.setup_homing(**kwargs)

    def setup_limit_switch(self, **kwargs):
        """Passthrough to pylablib's ``setup_limit_switch`` (native units)."""
        self.km.setup_limit_switch(**kwargs)

    def stop(self):
        self.km.stop()

    def close(self):
        self.km.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
