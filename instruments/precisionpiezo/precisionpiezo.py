from __future__ import annotations

import os
import time


class PrecisionPiezoCT1P:
    """Thin wrapper around the Thorlabs Kinesis .NET API for the CT1P cage-compatible
    z-axis translation stage with integrated precision piezo actuator.

    The CT1P has a piezo driver built directly into the stage (no separate KIM/K-Cube
    controller) and connects straight to the PC over USB. Kinesis exposes it as an
    ``IntegratedPrecisionPiezo`` device, distinct from the benchtop ``PPC00X`` piezo
    controllers, so it is *not* reachable through pylablib's ``Thorlabs.KinesisMotor``
    (used by :class:`instruments.kinesismotor.KinesisMotor` for KDC101/K10CR1 stages).
    Instead this class drives the Kinesis .NET assemblies directly via ``pythonnet``.

    Windows only: requires **Kinesis** to be installed (default
    ``C:\\Program Files\\Thorlabs\\Kinesis``, override with the ``KINESIS_DIR`` env var
    or the ``kinesis_dir=`` constructor argument) and the ``pythonnet`` package.

    Two independent control modes, matching the physical actuator:

    - **Closed loop** (default): absolute position in **micrometers** (µm), 0 to
      :attr:`max_travel_um` (~160 µm). Use :meth:`move_to` / :meth:`move_by` /
      :meth:`get_position`.
    - **Open loop**: direct drive voltage in **volts**. Use :meth:`set_voltage` /
      :meth:`get_voltage`. No position feedback in this mode.

    Example::

        from instruments.precisionpiezo import PrecisionPiezoCT1P

        with PrecisionPiezoCT1P("44000001") as piezo:
            piezo.zero()
            piezo.move_to(80.0)          # µm, closed loop; blocks until reached
            print(piezo.get_position())
    """

    DEFAULT_KINESIS_DIR = r"C:\Program Files\Thorlabs\Kinesis"

    # The Kinesis .NET API's unit conventions differ between position and voltage:
    #   - GetMaxTravel is in raw steps of 100 nm (10000 == 1 mm).
    #   - GetPosition/SetPosition are a signed percentage of max travel, scaled to
    #     +/-32767 (== +/-100%).
    #   - GetOutputVoltage/SetOutputVoltage/GetMinOutputVoltage/GetMaxOutputVoltage,
    #     by contrast, already report/accept real volts directly (confirmed empirically:
    #     GetMaxOutputVoltage/GetMinOutputVoltage read exactly 140.0/-10.0, matching the
    #     CT1P's documented -10..140 V range with no extra scaling) -- no conversion needed.
    _TRAVEL_UNIT_UM = 0.1
    _FULL_SCALE = 32767

    def __init__(
        self,
        serial: str,
        *,
        kinesis_dir: str | None = None,
        polling_ms: int = 250,
        simulate: bool = False,
    ):
        self._closed = True  # guard: set False only after a successful open
        self._simulating = False

        # ------------------------------------------------------------------
        # 1. DLL path setup (Windows only, before any .NET reference is added)
        # ------------------------------------------------------------------
        if os.name == "nt":
            kinesis_dir = kinesis_dir or os.environ.get("KINESIS_DIR", self.DEFAULT_KINESIS_DIR)
            os.add_dll_directory(kinesis_dir)
        else:
            kinesis_dir = kinesis_dir or self.DEFAULT_KINESIS_DIR

        # ------------------------------------------------------------------
        # 2. Deferred .NET imports — must come after the path setup above
        # ------------------------------------------------------------------
        import clr  # noqa: PLC0415

        clr.AddReference(os.path.join(kinesis_dir, "Thorlabs.MotionControl.DeviceManagerCLI.dll"))
        clr.AddReference(os.path.join(kinesis_dir, "Thorlabs.MotionControl.GenericPiezoCLI.dll"))
        clr.AddReference(os.path.join(kinesis_dir, "Thorlabs.MotionControl.IntegratedPrecisionPiezoCLI.dll"))

        from System import Decimal  # noqa: PLC0415
        from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI, SimulationManager  # noqa: PLC0415
        from Thorlabs.MotionControl.GenericPiezoCLI.Piezo import PiezoControlModeTypes  # noqa: PLC0415
        from Thorlabs.MotionControl.IntegratedPrecisionPiezoCLI import IntegratedPrecisionPiezo  # noqa: PLC0415

        self._Decimal = Decimal
        self._PiezoControlModeTypes = PiezoControlModeTypes

        # ------------------------------------------------------------------
        # 3. Optional hardware-free simulation (Kinesis Simulator)
        # ------------------------------------------------------------------
        if simulate:
            SimulationManager.Instance.InitializeSimulations()
            self._simulating = True
        self._SimulationManager = SimulationManager

        # ------------------------------------------------------------------
        # 4. Discover and connect
        # ------------------------------------------------------------------
        DeviceManagerCLI.BuildDeviceList()
        available = list(DeviceManagerCLI.GetDeviceList(IntegratedPrecisionPiezo.DevicePrefix))
        if serial not in available:
            raise RuntimeError(
                f"No CT1P (IntegratedPrecisionPiezo) with serial {serial!r} found. "
                f"Available: {available}"
            )

        self._device = IntegratedPrecisionPiezo.CreateIntegratedPrecisionPiezo(serial)
        self._device.Connect(serial)

        if not self._device.IsSettingsInitialized():
            self._device.WaitForSettingsInitialized(10000)  # 10 second timeout
            if not self._device.IsSettingsInitialized():
                raise RuntimeError(f"Settings for CT1P {serial!r} did not initialize in time.")

        # Load the stage/controller configuration needed for real-world units.
        self._config = self._device.GetPiezoConfiguration(serial)

        self._device.StartPolling(polling_ms)
        time.sleep(0.25)
        self._device.EnableDevice()
        time.sleep(0.25)  # wait for the device to enable

        # ------------------------------------------------------------------
        # 5. Cache metadata
        # ------------------------------------------------------------------
        self.serial = serial
        device_info = self._device.GetDeviceInfo()
        self.model = device_info.Description
        self.max_travel_um = self._from_decimal(self._device.GetMaxTravel()) * self._TRAVEL_UNIT_UM

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
    # Closed-loop position control (micrometers)                           #
    # ------------------------------------------------------------------ #

    def move_to(self, position_um: float, timeout: float = 30.0) -> None:
        """Move to an absolute position in micrometers (closed loop). Blocks until reached.

        Args:
            position_um: Target position, 0 to :attr:`max_travel_um`.
            timeout: Maximum time to wait for the move to complete (seconds).
        """
        if not 0 <= position_um <= self.max_travel_um:
            raise ValueError(f"position_um must be 0-{self.max_travel_um}, got {position_um}")
        self.set_closed_loop()
        raw = round(position_um / self.max_travel_um * self._FULL_SCALE)
        raw = max(0, min(self._FULL_SCALE, raw))
        self._device.SetPosition(self._to_decimal(raw))
        self._wait_position(position_um, timeout=timeout)

    def move_by(self, delta_um: float, timeout: float = 30.0) -> None:
        """Move by a relative offset in micrometers (closed loop). Blocks until reached."""
        self.move_to(self.get_position() + delta_um, timeout=timeout)

    def get_position(self) -> float:
        """Return the current closed-loop position in micrometers."""
        raw = self._from_decimal(self._device.GetPosition())
        return raw / self._FULL_SCALE * self.max_travel_um

    def get_max_travel(self) -> float:
        """Return the maximum closed-loop travel in micrometers (~160 µm for the CT1P)."""
        return self.max_travel_um

    def _wait_position(self, target: float, tol: float = 0.5, timeout: float = 30.0) -> None:
        """Block until get_position() is within tol of target, or raise TimeoutError."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if abs(self.get_position() - target) <= tol:
                return
            time.sleep(0.1)
        raise TimeoutError(f"CT1P did not reach {target}um within {timeout}s")

    # ------------------------------------------------------------------ #
    # Control mode                                                          #
    # ------------------------------------------------------------------ #

    def set_closed_loop(self) -> None:
        """Switch to closed-loop (position) control mode."""
        self._device.SetPositionControlMode(self._PiezoControlModeTypes.CloseLoop)

    def set_open_loop(self) -> None:
        """Switch to open-loop (voltage) control mode."""
        self._device.SetPositionControlMode(self._PiezoControlModeTypes.OpenLoop)

    # ------------------------------------------------------------------ #
    # Open-loop voltage control (volts)                                    #
    # ------------------------------------------------------------------ #

    def set_voltage(self, volts: float) -> None:
        """Set the output drive voltage directly (open loop, no position feedback)."""
        min_volts = self.get_min_voltage()
        max_volts = self.get_max_voltage()
        if not min_volts <= volts <= max_volts:
            raise ValueError(f"volts must be {min_volts}-{max_volts}, got {volts}")
        self.set_open_loop()
        self._device.SetOutputVoltage(self._to_decimal(volts))

    def get_voltage(self) -> float:
        """Return the current output drive voltage."""
        return self._from_decimal(self._device.GetOutputVoltage())

    def get_min_voltage(self) -> float:
        """Return the minimum output drive voltage (negative for the CT1P, e.g. -10 V)."""
        return self._from_decimal(self._device.GetMinOutputVoltage())

    def get_max_voltage(self) -> float:
        """Return the maximum output drive voltage."""
        return self._from_decimal(self._device.GetMaxOutputVoltage())

    # ------------------------------------------------------------------ #
    # Zeroing                                                              #
    # ------------------------------------------------------------------ #

    def zero(self) -> None:
        """Run the piezo zeroing routine to (re-)establish the closed-loop reference.

        The stage should be free to move over its full travel before calling this.
        Blocks until the routine completes.
        """
        self._device.SetZero()

    # ------------------------------------------------------------------ #
    # Info                                                                 #
    # ------------------------------------------------------------------ #

    def identify(self) -> dict:
        """Return model, serial number, and description."""
        return {
            "model": self.model,
            "serial": self.serial,
            "max_travel_um": self.max_travel_um,
        }

    # ------------------------------------------------------------------ #
    # Teardown                                                             #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        if not self._closed:
            self._device.StopPolling()
            self._device.Disconnect()
            if self._simulating:
                self._SimulationManager.Instance.UninitializeSimulations()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
