from __future__ import annotations

import os
import time


class KinesisMotor:
    """Thin wrapper around the Thorlabs Kinesis .NET API for KDC101 (linear, mm) and
    K10CR1 (rotary, deg) stages.

    This class drives the Kinesis .NET assemblies directly via ``pythonnet``, mirroring
    :class:`instruments.precisionpiezo.PrecisionPiezoCT1P`.

    Windows only: requires **Kinesis** to be installed (default
    ``C:\\Program Files\\Thorlabs\\Kinesis``, override with the ``KINESIS_DIR`` env var
    or the ``kinesis_dir=`` constructor argument) and the ``pythonnet`` package.

    The controller type is auto-detected from the serial number's two-digit prefix
    (``27`` -> KCube DC Servo, ``55`` -> K10CR1 cage rotator). The Kinesis .NET API
    then auto-detects the physical stage attached (e.g. Z925B) and reports/accepts
    positions directly in real-world units -- **mm** for linear stages, **deg** for
    rotary stages (:attr:`units`). No scale/unit bookkeeping is needed on top of that.

    ``motor_type`` is accepted for backwards compatibility with existing call sites
    but is otherwise unused -- the .NET API needs no hint to identify the stage.
    """

    DEFAULT_KINESIS_DIR = r"C:\Program Files\Thorlabs\Kinesis"
    _HOME_TIMEOUT_MS = 60_000
    _MOVE_TIMEOUT_MS = 60_000
    _SETTINGS_TIMEOUT_MS = 10_000

    def __init__(self, serial: str, motor_type: str = 'stage', *, kinesis_dir: str | None = None):
        self._closed = True  # guard: set False only after a successful open

        # ------------------------------------------------------------------
        # 1. DLL path setup (Windows only, before any .NET reference is added)
        # ------------------------------------------------------------------
        if os.name == "nt":
            kinesis_dir = kinesis_dir or os.environ.get("KINESIS_DIR", self.DEFAULT_KINESIS_DIR)
            os.add_dll_directory(kinesis_dir)
        else:
            kinesis_dir = kinesis_dir or self.DEFAULT_KINESIS_DIR

        # ------------------------------------------------------------------
        # 2. Deferred .NET imports -- must come after the path setup above
        # ------------------------------------------------------------------
        import clr  # noqa: PLC0415

        clr.AddReference(os.path.join(kinesis_dir, "Thorlabs.MotionControl.DeviceManagerCLI.dll"))
        clr.AddReference(os.path.join(kinesis_dir, "Thorlabs.MotionControl.GenericMotorCLI.dll"))
        clr.AddReference(os.path.join(kinesis_dir, "Thorlabs.MotionControl.KCube.DCServoCLI.dll"))
        clr.AddReference(os.path.join(kinesis_dir, "Thorlabs.MotionControl.IntegratedStepperMotorsCLI.dll"))

        from System import Decimal  # noqa: PLC0415
        from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI  # noqa: PLC0415
        from Thorlabs.MotionControl.KCube.DCServoCLI import KCubeDCServo  # noqa: PLC0415
        from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import CageRotator  # noqa: PLC0415

        self._Decimal = Decimal

        # Controller type is selected from the serial's two-digit prefix (as an int --
        # DevicePrefix is exposed as an int, not a string); the .NET API itself then
        # auto-detects the physical stage attached (e.g. Z925B, K10CR1/M).
        controllers = {
            KCubeDCServo.DevicePrefix: (KCubeDCServo.CreateKCubeDCServo, 'mm'),
            CageRotator.DevicePrefix: (CageRotator.CreateCageRotator, 'deg'),
        }

        # ------------------------------------------------------------------
        # 3. Discover and connect
        # ------------------------------------------------------------------
        DeviceManagerCLI.BuildDeviceList()
        available = list(DeviceManagerCLI.GetDeviceList())
        if serial not in available:
            raise RuntimeError(f"No Kinesis device with serial {serial!r} found. Available: {available}")

        try:
            prefix = int(serial[:2])
            factory, units = controllers[prefix]
        except (KeyError, ValueError):
            raise RuntimeError(
                f"Serial {serial!r} has an unrecognized controller prefix. "
                f"Known prefixes: {sorted(controllers)}"
            ) from None
        self.units = units

        self._device = factory(serial)
        self._device.Connect(serial)
        if not self._device.IsSettingsInitialized():
            self._device.WaitForSettingsInitialized(self._SETTINGS_TIMEOUT_MS)
            if not self._device.IsSettingsInitialized():
                raise RuntimeError(f"Settings for Kinesis device {serial!r} did not initialize in time.")

        # Loads the attached stage's real-world unit conversion (mm / deg).
        self._device.LoadMotorConfiguration(serial)

        self._device.StartPolling(250)
        time.sleep(0.25)
        self._device.EnableDevice()
        time.sleep(0.25)  # wait for the device to enable

        self.serial = serial
        self.motor_type = motor_type  # accepted for backwards compatibility; unused
        self.model = self._device.GetDeviceInfo().Name
        self._closed = False
        print(f"Connected to {self.model} (SN {self.serial})")

    # ------------------------------------------------------------------ #
    # Decimal helpers (System.Decimal <-> float)                           #
    # ------------------------------------------------------------------ #

    def _to_decimal(self, value: float):
        return self._Decimal(value)

    def _from_decimal(self, value) -> float:
        # System.Decimal.ToDouble is a *static* method (Decimal.ToDouble(d)), not an
        # instance method, even though pythonnet makes it visible via hasattr(value, ...).
        return float(self._Decimal.ToDouble(value)) if isinstance(value, self._Decimal) else float(value)

    # ------------------------------------------------------------------ #
    # Motion (native units: mm for linear stages, deg for rotary stages)   #
    # ------------------------------------------------------------------ #

    def home(self, force: bool = False):
        if not force and self.is_homed():
            return
        self._device.Home(self._HOME_TIMEOUT_MS)

    def is_homed(self) -> bool:
        return bool(self._device.Status.IsHomed)

    def move_to(self, position: float):
        self._device.MoveTo(self._to_decimal(position), self._MOVE_TIMEOUT_MS)
        print(f"Motor: {self.serial} moved to {position:.6f} {self.units}")

    def move_by(self, position: float):
        self.move_to(self.get_position() + position)
        print(f"Motor: {self.serial} moved by {position:.6f} {self.units}")

    def get_position(self) -> float:
        return self._from_decimal(self._device.Position)

    def wait_move(self):
        """No-op: :meth:`move_to`/:meth:`move_by` already block until the move completes.

        Kept so callers that issue a move and then unconditionally call ``wait_move()``
        (e.g. :func:`instruments.camera.utils.autofocus`) keep working.
        """

    def set_velocity(self, max_velocity: float = None, acceleration: float = None):
        params = self._device.GetVelocityParams()
        if max_velocity is not None:
            params.MaxVelocity = self._to_decimal(max_velocity)
        if acceleration is not None:
            params.Acceleration = self._to_decimal(acceleration)
        self._device.SetVelocityParams(params)

    def setup_homing(self, **kwargs):
        """Set homing parameters. Currently only ``velocity`` (native units/s) is supported."""
        if 'velocity' in kwargs:
            params = self._device.GetHomingParams()
            params.Velocity = self._to_decimal(kwargs['velocity'])
            self._device.SetHomingParams(params)

    def setup_limit_switch(self, **kwargs):
        """No-op: the Kinesis .NET API homes rotary stages natively, so no limit-switch
        configuration is needed. Kept as a no-op for backwards compatibility with existing
        call sites."""

    def stop(self):
        self._device.StopImmediate()

    # ------------------------------------------------------------------ #
    # Teardown                                                             #
    # ------------------------------------------------------------------ #

    def close(self):
        if not self._closed:
            self._device.StopPolling()
            self._device.Disconnect()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
