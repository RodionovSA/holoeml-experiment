"""Hardware device bundle for the phase experiment.

``Instruments`` owns the connections (and their teardown) for whichever
devices it was asked to open -- camera, monochromator, filter wheel, focus
motor, polarizer, sample stage x/y, and/or the KIM101 inertial piezo. It has
no knowledge of :class:`~phasectl.config.Config` measurement policy -- that
lives in :class:`~phasectl.control.Control`. Devices are opened selectively,
via the ``devices`` argument to :meth:`Instruments.from_config`; each
device's SDK is imported lazily inside its own opener, so requesting a
subset doesn't require the others' SDKs to be installed.

Connecting several devices is a multi-step process where any step can fail
after earlier ones already opened a handle. ``ExitStack`` makes that safe:
each successfully opened device is registered immediately, so a failure
partway through unwinds only what was actually opened, in reverse order.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from instruments.camera import Camera
    from instruments.filterwheel import FilterWheelControl
    from instruments.kinesismotor import KinesisMotor
    from instruments.monochromator.mono import MonochromatorControl
    from instruments.inertialpiezo import KIM101, KIM101Axis

    from phasectl.config import Config

DEVICE_NAMES = ("camera", "mono", "filterwheel", "focus", "polarizer",
                "stage_x", "stage_y", "kim")
ALL_DEVICES = frozenset(DEVICE_NAMES)

_ALIASES = {"piezo": "kim"}


def _normalize(devices: Iterable[str]) -> set[str]:
    """Resolve aliases and validate *devices* against :data:`DEVICE_NAMES`."""
    names = {_ALIASES.get(d, d) for d in devices}
    unknown = names - ALL_DEVICES
    if unknown:
        raise ValueError(
            f"Unknown device name(s) {sorted(unknown)}; expected a subset of "
            f"{sorted(ALL_DEVICES)} (or the alias 'piezo' for 'kim')."
        )
    return names


def _open_camera(config: Config, stack: contextlib.ExitStack) -> Camera:
    from instruments.camera import open_camera
    camera = open_camera(config, stack)
    return stack.enter_context(camera)


def _open_mono(config: Config, stack: contextlib.ExitStack) -> MonochromatorControl:
    from instruments.monochromator.mono import MonochromatorControl
    return stack.enter_context(MonochromatorControl(port=config.monochromator_port))


def _open_filterwheel(config: Config, stack: contextlib.ExitStack) -> FilterWheelControl:
    from instruments.filterwheel import FilterWheelControl
    return stack.enter_context(FilterWheelControl(config.filterwheel_address))


def _open_focus(config: Config, stack: contextlib.ExitStack) -> KinesisMotor:
    from instruments.kinesismotor import KinesisMotor
    return stack.enter_context(KinesisMotor(config.focus_serial, motor_type='stage'))


def _open_polarizer(config: Config, stack: contextlib.ExitStack) -> KinesisMotor:
    from instruments.kinesismotor import KinesisMotor
    return stack.enter_context(KinesisMotor(config.polarizer_serial, motor_type="K10CR1"))


def _open_stage_x(config: Config, stack: contextlib.ExitStack) -> KinesisMotor:
    from instruments.kinesismotor import KinesisMotor
    return stack.enter_context(KinesisMotor(config.stage_x_serial, motor_type='stage'))


def _open_stage_y(config: Config, stack: contextlib.ExitStack) -> KinesisMotor:
    from instruments.kinesismotor import KinesisMotor
    return stack.enter_context(KinesisMotor(config.stage_y_serial, motor_type='stage'))


def _open_kim(config: Config, stack: contextlib.ExitStack) -> KIM101:
    from instruments.inertialpiezo import KIM101
    return stack.enter_context(KIM101(config.kim_serial))


_OPENERS = {
    "camera": _open_camera,
    "mono": _open_mono,
    "filterwheel": _open_filterwheel,
    "focus": _open_focus,
    "polarizer": _open_polarizer,
    "stage_x": _open_stage_x,
    "stage_y": _open_stage_y,
    "kim": _open_kim,
}


@dataclass
class Instruments:
    """Bundle of connected hardware devices, plus their combined lifecycle.

    Only the devices requested via :meth:`from_config`'s ``devices`` argument
    are opened; accessing a device that was not requested raises
    :class:`RuntimeError` naming the fix, rather than an ``AttributeError``
    on ``None`` further down in a routine.
    """

    _devices: dict[str, object] = field(default_factory=dict)
    _stack: contextlib.ExitStack = field(default_factory=contextlib.ExitStack, repr=False)

    @classmethod
    def from_config(cls, config: Config, devices: Iterable[str]) -> Instruments:
        """Connect to the requested devices, described by *config*.

        Args:
            config: Protocol configuration (serials, addresses, ports).
            devices: Names to open -- any subset of :data:`DEVICE_NAMES`
                (or :data:`ALL_DEVICES`). ``"piezo"`` is an alias for
                ``"kim"``.

        If a later device fails to connect, every device opened so far is
        cleanly closed before the exception propagates. No settings are
        applied here -- that is :class:`~phasectl.control.Control`'s job.
        """
        names = _normalize(devices)

        stack = contextlib.ExitStack()
        opened: dict[str, object] = {}
        try:
            for name in DEVICE_NAMES:
                if name not in names:
                    continue
                opened[name] = _OPENERS[name](config, stack)
            if "kim" in opened:
                opened["piezo"] = opened["kim"].axis(config.kim_channel)
        except Exception:
            stack.close()
            raise

        return cls(_devices=opened, _stack=stack)

    def _require(self, name: str):
        try:
            return self._devices[name]
        except KeyError:
            raise RuntimeError(
                f"Device {name!r} was not opened. Pass it in devices= when calling "
                f"Instruments.from_config (opened: {sorted(self._devices)})."
            ) from None

    @property
    def camera(self) -> Camera:
        return self._require("camera")

    @property
    def mono(self) -> MonochromatorControl:
        return self._require("mono")

    @property
    def filterwheel(self) -> FilterWheelControl:
        return self._require("filterwheel")

    @property
    def focus(self) -> KinesisMotor:
        return self._require("focus")

    @property
    def polarizer(self) -> KinesisMotor:
        return self._require("polarizer")

    @property
    def stage_x(self) -> KinesisMotor:
        return self._require("stage_x")

    @property
    def stage_y(self) -> KinesisMotor:
        return self._require("stage_y")

    @property
    def kim(self) -> KIM101:
        return self._require("kim")

    @property
    def piezo(self) -> KIM101Axis:
        return self._require("piezo")

    def close(self) -> None:
        """Close every opened device. Safe to call more than once."""
        self._stack.close()

    def __enter__(self) -> Instruments:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @contextlib.contextmanager
    def armed(self):
        """Arm the camera for capture, disarming again on exit (even on error)."""
        self.camera.arm()
        time.sleep(0.1)
        try:
            yield
        finally:
            self.camera.disarm()
