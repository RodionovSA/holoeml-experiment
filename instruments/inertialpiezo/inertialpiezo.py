from __future__ import annotations

import os
import time


class KIM101:
    """Thin wrapper around the Thorlabs Kinesis .NET API for the KIM101 K-Cube Inertial
    Motor controller (also accepts its single-channel sibling, KIM001).

    The KIM101 is a separate 4-channel controller driving **open-loop, encoder-less** inertial
    ("slip-stick") actuators such as the PIAK10 -- Kinesis exposes it as ``KCubeInertialMotor``,
    and every motion call is addressed to a channel. Like the other classes in this package,
    this one drives the Kinesis .NET assemblies directly via ``pythonnet``.

    This class represents the **controller** (the USB connection) and owns no motion methods
    itself -- one KIM101 can drive up to four independent actuators, so motion lives on
    :class:`KIM101Axis`, obtained per channel via :meth:`axis`. This mirrors the Kinesis .NET
    API itself, which splits the same way: ``Connect``/``StartPolling``/``EnableDevice`` take
    no channel, while ``MoveTo``/``GetPosition``/``Stop``/etc. all take one.

    Windows only: requires **Kinesis** to be installed (default
    ``C:\\Program Files\\Thorlabs\\Kinesis``, override with the ``KINESIS_DIR`` env var
    or the ``kinesis_dir=`` constructor argument) and the ``pythonnet`` package.

    Example::

        from instruments.inertialpiezo import KIM101

        with KIM101("97000001") as kim:            # controller: one connection
            z = kim.axis(1)                        # axis: the actual actuator
            z.zero()
            z.move_by(500)                          # steps; blocks until reached
            print(z.get_position())
    """

    DEFAULT_KINESIS_DIR = r"C:\Program Files\Thorlabs\Kinesis"
    _SETTINGS_TIMEOUT_MS = 10_000
    MAX_CHANNELS = 4

    def __init__(
        self,
        serial: str,
        *,
        kinesis_dir: str | None = None,
        polling_ms: int = 250,
        stage_type: str | None = "PIA",
        simulate: bool = False,
    ):
        self._closed = True  # guard: set False only after a successful open
        self._simulating = False
        self._axes: dict[int, KIM101Axis] = {}

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
        clr.AddReference(os.path.join(kinesis_dir, "Thorlabs.MotionControl.KCube.InertialMotorCLI.dll"))

        from System import Int32  # noqa: PLC0415
        from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI, SimulationManager  # noqa: PLC0415
        from Thorlabs.MotionControl.KCube.InertialMotorCLI import (  # noqa: PLC0415
            InertialMotorJogDirection,
            InertialMotorStatus,
            KCubeInertialMotor,
            KIMStages,
        )

        self._Int32 = Int32
        self._MotorChannels = InertialMotorStatus.MotorChannels
        self._JogDirection = InertialMotorJogDirection
        self._KIMStages = KIMStages

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
        available = list(DeviceManagerCLI.GetDeviceList(KCubeInertialMotor.DevicePrefix_KIM101))
        available += list(DeviceManagerCLI.GetDeviceList(KCubeInertialMotor.DevicePrefix_KIM001))
        if serial not in available:
            raise RuntimeError(
                f"No KIM101/KIM001 (KCubeInertialMotor) with serial {serial!r} found. "
                f"Available: {available}"
            )

        self._device = KCubeInertialMotor.CreateKCubeInertialMotor(serial)
        self._device.Connect(serial)

        if not self._device.IsSettingsInitialized():
            self._device.WaitForSettingsInitialized(self._SETTINGS_TIMEOUT_MS)
            if not self._device.IsSettingsInitialized():
                raise RuntimeError(f"Settings for KIM101 {serial!r} did not initialize in time.")

        # Loads the controller/actuator configuration.
        self._device.GetInertialMotorConfiguration(serial)

        self._device.StartPolling(polling_ms)
        time.sleep(0.25)
        self._device.EnableDevice()
        time.sleep(0.25)  # wait for the device to enable

        if stage_type is not None:
            self._device.SetStageType(getattr(self._KIMStages, stage_type))
        self._stage_type = stage_type

        # ------------------------------------------------------------------
        # 5. Resolve overloaded MoveTo/MoveBy to their Int32 (step-count) form.
        #    KCubeInertialMotor.MoveTo/MoveBy each have an Int32 and a Decimal
        #    overload; pythonnet needs the explicit signature to avoid picking
        #    the Decimal one for a plain Python int argument.
        # ------------------------------------------------------------------
        self._move_to = self._device.MoveTo.Overloads[self._MotorChannels, Int32, Int32]
        self._move_by = self._device.MoveBy.Overloads[self._MotorChannels, Int32, Int32]

        # ------------------------------------------------------------------
        # 6. Cache metadata
        # ------------------------------------------------------------------
        self.serial = serial
        self.model = self._device.GetDeviceInfo().Description
        self.n_channels = 1 if self._device.IsSingleChannelDevice() else self.MAX_CHANNELS

        self._closed = False
        print(f"Connected to {self.model} (SN {self.serial})")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _require_open(self) -> "KIM101":
        if self._closed:
            raise RuntimeError(f"KIM101 {self.serial!r} is closed.")
        return self

    def _net_channel(self, channel: int):
        if not 1 <= channel <= self.n_channels:
            raise ValueError(f"channel must be 1-{self.n_channels}, got {channel}")
        return getattr(self._MotorChannels, f"Channel{channel}")

    # ------------------------------------------------------------------ #
    # Axes                                                                 #
    # ------------------------------------------------------------------ #

    def axis(self, channel: int) -> "KIM101Axis":
        """Return the :class:`KIM101Axis` for ``channel`` (1-:attr:`n_channels`).

        Axes are created lazily and cached -- calling this twice with the same channel
        returns the same object.
        """
        self._net_channel(channel)  # validate before creating/caching
        if channel not in self._axes:
            self._axes[channel] = KIM101Axis(self, channel)
        return self._axes[channel]

    # ------------------------------------------------------------------ #
    # Device-wide control                                                  #
    # ------------------------------------------------------------------ #

    def enable(self) -> None:
        self._require_open()._device.EnableDevice()

    def disable(self) -> None:
        self._require_open()._device.DisableDevice()

    def stop_all(self) -> None:
        """Stop every channel (1-:attr:`n_channels`)."""
        device = self._require_open()._device
        for channel in range(1, self.n_channels + 1):
            device.Stop(self._net_channel(channel))

    def is_any_moving(self) -> bool:
        return bool(self._require_open()._device.IsAnyChannelMoving())

    def set_dual_channel_mode(self, enable: bool) -> None:
        self._require_open()._device.SetDualChannelMode(bool(enable))

    def is_dual_channel_mode(self) -> bool:
        return bool(self._require_open()._device.IsDualChannelMode())

    def get_stage_type(self) -> str:
        return str(self._require_open()._device.GetStageType())

    def set_stage_type(self, stage_type: str) -> None:
        self._require_open()._device.SetStageType(getattr(self._KIMStages, stage_type))
        self._stage_type = stage_type

    # ------------------------------------------------------------------ #
    # Info                                                                 #
    # ------------------------------------------------------------------ #

    def identify(self) -> dict:
        """Return model, serial number, channel count, and configured stage type."""
        return {
            "model": self.model,
            "serial": self.serial,
            "n_channels": self.n_channels,
            "stage_type": self._stage_type,
        }

    def __repr__(self) -> str:
        return f"KIM101(SN {self.serial!r})"

    # ------------------------------------------------------------------ #
    # Teardown                                                             #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        if not self._closed:
            self._device.StopPolling()
            self._device.Disconnect()
            if self._simulating:
                self._SimulationManager.Instance.UninitializeSimulations()
            self._axes.clear()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class KIM101Axis:
    """One channel of a :class:`KIM101` controller -- i.e. one PIAK10 piezo inertial
    actuator. Obtain instances via :meth:`KIM101.axis`, never construct directly.

    **Units**: positions are plain **inertial steps** (signed 32-bit integers), not real-world
    distance -- the PIA actuator's effective step size drifts with load, drive voltage and
    direction, so there is no fixed step-to-distance conversion to give. Convert to physical
    units yourself if you calibrate one for your setup.

    An axis has no :meth:`close`/context-manager protocol of its own -- its lifecycle belongs
    to the owning :class:`KIM101`, so it cannot accidentally disconnect a device other code may
    still be using.
    """

    def __init__(self, kim: KIM101, channel: int):
        self._kim = kim
        self.channel = channel
        self._net_channel = kim._net_channel(channel)
        self.units = "steps"

    @property
    def serial(self) -> str:
        return self._kim.serial

    def _require_open(self) -> KIM101:
        return self._kim._require_open()

    # ------------------------------------------------------------------ #
    # Motion (native units: inertial steps)                                #
    # ------------------------------------------------------------------ #

    def move_to(self, position: int, timeout: float = 30.0) -> None:
        """Move to an absolute position in steps. Blocks until reached.

        A no-op if already at ``position`` -- Kinesis's ``MoveTo`` blocks on an
        internal move-complete signal that a zero-delta move never fires, so
        issuing it anyway would hang for the full ``timeout`` and raise
        ``MoveTimeoutException`` instead of returning immediately.

        Args:
            position: Target position in inertial steps (signed 32-bit).
            timeout: Maximum time to wait for the move to complete (seconds).
        """
        kim = self._require_open()
        position = int(position)
        if position == self.get_position():
            print(f"KIM101 {self.serial} ch{self.channel} already at {position} steps; no move needed")
            return
        kim._move_to(self._net_channel, position, int(timeout * 1000))
        print(f"KIM101 {self.serial} ch{self.channel} moved to {position} steps")

    def move_by(self, steps: int, timeout: float = 30.0) -> None:
        """Move by a relative offset in steps. Blocks until reached.

        A no-op for ``steps == 0`` -- see :meth:`move_to` for why a zero-delta
        move must be skipped rather than issued to the controller.
        """
        kim = self._require_open()
        steps = int(steps)
        if steps == 0:
            print(f"KIM101 {self.serial} ch{self.channel} move_by(0); no move needed")
            return
        kim._move_by(self._net_channel, steps, int(timeout * 1000))
        print(f"KIM101 {self.serial} ch{self.channel} moved by {steps} steps")

    def jog(self, direction: str, timeout: float = 30.0) -> None:
        """Jog one step-burst in ``direction`` (``'+'``/``'increase'`` or ``'-'``/``'decrease'``),
        using this channel's configured jog parameters (see :meth:`set_jog_parameters`)."""
        kim = self._require_open()
        direction = str(direction).lower()
        if direction in ("+", "increase", "inc"):
            net_direction = kim._JogDirection.Increase
        elif direction in ("-", "decrease", "dec"):
            net_direction = kim._JogDirection.Decrease
        else:
            raise ValueError(f"direction must be '+'/'increase' or '-'/'decrease', got {direction!r}")
        kim._device.Jog(self._net_channel, net_direction, int(timeout * 1000))

    def stop(self) -> None:
        self._require_open()._device.Stop(self._net_channel)

    # ------------------------------------------------------------------ #
    # Position                                                             #
    # ------------------------------------------------------------------ #

    def get_position(self) -> int:
        """Return the current position in inertial steps."""
        return int(self._require_open()._device.GetPosition(self._net_channel))

    def set_position_as(self, position: int) -> None:
        """Redefine the current physical position as ``position`` steps, without moving."""
        self._require_open()._device.SetPositionAs(self._net_channel, int(position))

    def zero(self) -> None:
        """Reset the step counter to 0 at the current physical position.

        The PIA actuator is open loop with no absolute reference, so this only zeroes the
        counter -- it does not move the stage or find a hardware home.
        """
        self._require_open()._device.SetPositionToZero(self._net_channel)

    # ------------------------------------------------------------------ #
    # Status                                                               #
    # ------------------------------------------------------------------ #

    def is_moving(self) -> bool:
        return bool(self._require_open()._device.ChannelStatus(self._net_channel).IsMoving)

    def can_move(self) -> bool:
        return bool(self._require_open()._device.CanMoveChannel(self._net_channel))

    def wait_move(self, timeout: float = 30.0) -> None:
        """Block until this channel stops moving, or raise :class:`TimeoutError`.

        :meth:`move_to`/:meth:`move_by` already block internally, so this is only needed
        after :meth:`jog` or to wait out a move issued elsewhere.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_moving():
                return
            time.sleep(0.05)
        raise TimeoutError(f"KIM101 {self.serial} ch{self.channel} did not stop moving within {timeout}s")

    def get_status(self) -> dict:
        """Return ``{position, is_moving, is_enabled, is_error}`` for this channel."""
        status = self._require_open()._device.ChannelStatus(self._net_channel)
        return {
            "position": int(status.Position),
            "is_moving": bool(status.IsMoving),
            "is_enabled": bool(status.IsEnabled),
            "is_error": bool(status.IsError),
        }

    # ------------------------------------------------------------------ #
    # Drive / jog parameters                                               #
    # ------------------------------------------------------------------ #

    def get_drive_parameters(self) -> dict:
        """Return ``{step_rate, step_acceleration, max_voltage}`` for this channel."""
        params = self._require_open()._device.GetDriveParameters(self._net_channel)
        return {
            "step_rate": int(params.StepRate),
            "step_acceleration": int(params.StepAcceleration),
            "max_voltage": int(params.MaxVoltage),
        }

    def set_drive_parameters(
        self,
        step_rate: int = None,
        step_acceleration: int = None,
        max_voltage: int = None,
    ) -> None:
        """Set drive parameters; any argument left ``None`` keeps its current value."""
        device = self._require_open()._device
        params = device.GetDriveParameters(self._net_channel)
        if step_rate is not None:
            params.StepRate = int(step_rate)
        if step_acceleration is not None:
            params.StepAcceleration = int(step_acceleration)
        if max_voltage is not None:
            params.MaxVoltage = int(max_voltage)
        device.SetDriveParameters(self._net_channel, params)

    def get_jog_parameters(self) -> dict:
        """Return ``{jog_rate, jog_acceleration, jog_step_fwd, jog_step_rev}`` for this channel."""
        params = self._require_open()._device.GetJogParameters(self._net_channel)
        return {
            "jog_rate": int(params.JogRate),
            "jog_acceleration": int(params.JogAcceleration),
            "jog_step_fwd": int(params.JogStepFwd),
            "jog_step_rev": int(params.JogStepRev),
        }

    def set_jog_parameters(
        self,
        jog_rate: int = None,
        jog_acceleration: int = None,
        jog_step_fwd: int = None,
        jog_step_rev: int = None,
    ) -> None:
        """Set jog parameters; any argument left ``None`` keeps its current value."""
        device = self._require_open()._device
        params = device.GetJogParameters(self._net_channel)
        if jog_rate is not None:
            params.JogRate = int(jog_rate)
        if jog_acceleration is not None:
            params.JogAcceleration = int(jog_acceleration)
        if jog_step_fwd is not None:
            params.JogStepFwd = int(jog_step_fwd)
        if jog_step_rev is not None:
            params.JogStepRev = int(jog_step_rev)
        device.SetJogParameters(self._net_channel, params)

    # ------------------------------------------------------------------ #
    # Channel enable/disable                                               #
    # ------------------------------------------------------------------ #

    def enable(self) -> None:
        self._require_open()._device.EnableChannel(self._net_channel)

    def disable(self) -> None:
        self._require_open()._device.DisableChannel(self._net_channel)

    # ------------------------------------------------------------------ #
    # Info                                                                 #
    # ------------------------------------------------------------------ #

    def identify(self) -> dict:
        """Return serial number, channel, units, and current position."""
        return {
            "serial": self.serial,
            "channel": self.channel,
            "units": self.units,
            "position": self.get_position(),
        }

    def __repr__(self) -> str:
        return f"KIM101Axis(SN {self.serial!r}, ch{self.channel})"
