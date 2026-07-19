"""Hardware device bundle for the amplitude experiment.

``Instruments`` owns the camera, monochromator, focus motor, polarizer, and
filter wheel connections and their teardown. It has no knowledge of
:class:`~amplitude.config.Config` measurement policy (exposure settings,
wavelength sweeps, etc.) -- that lives in :class:`~amplitude.control.Control`.

Connecting all five devices is a multi-step process where any step can fail
after earlier ones already opened a handle. ``ExitStack`` makes that safe:
each successfully opened device is registered immediately, so a failure
partway through unwinds only what was actually opened, in reverse order --
and calling ``close()`` more than once is already a no-op, so no ``_closed``
flag is needed.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field

import numpy as np

from instruments.filterwheel import FilterWheelControl
from instruments.kinesismotor import KinesisMotor
from instruments.monochromator.mono import MonochromatorControl
from instruments.pythorcam.thorcam import ThorlabsCamera, create_camera_sdk

from amplitude.config import Config


@dataclass
class Instruments:
    """Bundle of connected hardware devices, plus their combined lifecycle."""

    camera: ThorlabsCamera
    mono: MonochromatorControl
    focus: KinesisMotor
    polarizer: KinesisMotor
    filterwheel: FilterWheelControl
    _stack: contextlib.ExitStack = field(repr=False)

    @classmethod
    def from_config(cls, config: Config) -> Instruments:
        """Connect to every device described by *config*.

        Devices are entered into an :class:`~contextlib.ExitStack` as each
        connection succeeds, so if a later device fails to connect, every
        device opened so far is cleanly closed (in reverse order) before the
        exception propagates.
        """
        stack = contextlib.ExitStack()
        try:
            sdk = create_camera_sdk()
            stack.callback(sdk.dispose)

            camera = stack.enter_context(ThorlabsCamera(sdk, config.camera_serial))
            camera.set_settings(
                exposure_time_us=config.calib_initial_exposure_ms * 1000,
                gain=config.calib_initial_gain,
                black_level=config.camera_black_level,
                bit_depth=getattr(np, config.camera_bit_depth),
                out_bit_depth=getattr(np, config.camera_out_bit_depth),
            )

            mono = stack.enter_context(MonochromatorControl(port=config.monochromator_port))
            filterwheel = stack.enter_context(FilterWheelControl(config.filterwheel_address))
            focus = stack.enter_context(KinesisMotor(config.focus_serial, motor_type='stage'))
            polarizer = stack.enter_context(KinesisMotor(config.polarizer_serial, motor_type="K10CR1"))
        except Exception:
            stack.close()
            raise

        return cls(camera, mono, focus, polarizer, filterwheel, stack)

    def close(self) -> None:
        """Close every device. Safe to call more than once."""
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
