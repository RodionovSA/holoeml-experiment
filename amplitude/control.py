"""Measurement/calibration orchestration for the amplitude experiment.

``Control`` owns none of the hardware connections, settings-file state, or
measurement algorithms directly:

- device connection and lifecycle live in
  :class:`~amplitude.devices.Instruments`;
- exposure/focus settings persistence lives in
  :class:`~amplitude.settings_store.ExposureStore` /
  :class:`~amplitude.settings_store.FocusStore`;
- the measurement/calibration algorithms themselves live in
  :mod:`amplitude.measurements` (one module per routine -- ``default_state``,
  ``brightness``, ``reference``, ``black``, ``sample``), each operating on a
  shared :class:`~amplitude.measurements.context.MeasurementContext`.

``Control`` composes all of the above and is left with just the object
lifecycle (connect, prepare, close) and thin public methods that guard on
readiness and delegate to the matching routine module.
"""

from pathlib import Path

import numpy as np

from amplitude.config import Config, ExposureSettings, FocusSettings
from amplitude.devices import Instruments
from amplitude.measurements import black, brightness, default_state, reference, sample
from amplitude.measurements.context import MeasurementContext
from amplitude.settings_store import ExposureStore, FocusStore, Polarization


class Control:
    def __init__(self,
                 instruments: Instruments,
                 config: Config,
                 exposure_store: ExposureStore,
                 focus_store: FocusStore):
        self.instruments = instruments
        self.config = config
        self.exposure_store = exposure_store
        self.focus_store = focus_store

        self._ctx = MeasurementContext(instruments, config, exposure_store, focus_store)
        self._prepared = False

    @classmethod
    def from_config(cls, config: Config) -> "Control":
        instruments = Instruments.from_config(config)
        exposure_store = ExposureStore.from_config(config)
        focus_store = FocusStore.from_config(config)
        return cls(instruments, config, exposure_store, focus_store)

    def _require_prepared(self) -> None:
        if not self._prepared:
            raise RuntimeError("Devices not prepared; call prepare_devices() first.")

    def prepare_devices(self):
        """ Set settings for the hardware and home devices"""
        inst = self.instruments
        cfg = self.config

        # Monochromator
        inst.mono.initialize_arduino()
        inst.mono.home_motor()
        inst.mono.select_grating_mode(mode=cfg.monochromator_grating_mode)

        # Focus
        if not inst.focus.is_homed():
            inst.focus.home()
        v = cfg.default_focus_max_velocity
        a = cfg.default_focus_acceleration
        if v is not None or a is not None:
            inst.focus.set_velocity(max_velocity=v, acceleration=a)

        # Polarizer
        if cfg.default_polarizer_home_velocity is not None:
            inst.polarizer.setup_homing(velocity=cfg.default_polarizer_home_velocity)
        if not inst.polarizer.is_homed():
            inst.polarizer.home()
        v = cfg.default_polarizer_max_velocity
        a = cfg.default_polarizer_acceleration
        if v is not None or a is not None:
            inst.polarizer.set_velocity(max_velocity=v, acceleration=a)

        # Camera
        inst.camera.set_settings(
            exposure_time_us=cfg.calib_initial_exposure_ms * 1000,
            gain=cfg.calib_initial_gain,
            black_level=cfg.camera_black_level,
            bit_depth=getattr(np, cfg.camera_bit_depth),
            out_bit_depth=getattr(np, cfg.camera_out_bit_depth),
        )
        self._prepared = True

    def close(self):
        self.instruments.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Settings helpers                                                     #
    # ------------------------------------------------------------------ #

    def load_exposure_settings(self, xpol: bool | None = None) -> ExposureSettings:
        return self.exposure_store.reload(Polarization.from_xpol(xpol))

    def save_exposure_settings(self, xpol: bool | None = None) -> None:
        self.exposure_store.save(Polarization.from_xpol(xpol))

    def load_focus_settings(self) -> FocusSettings:
        return self.focus_store.reload()

    def save_focus_settings(self) -> None:
        self.focus_store.save()

    # ------------------------------------------------------------------ #
    # Public methods -- guard readiness, delegate to amplitude.measurements #
    # ------------------------------------------------------------------ #

    def go_to_default_state(self) -> None:
        """Move hardware to the canonical resting state. See :func:`amplitude.measurements.default_state.run`."""
        self._require_prepared()
        return default_state.run(self._ctx)

    def brightness_calibration(self,
                               override: bool = False,
                               use_current_as_initial: bool = False,
                               xpol: bool | None = True) -> None:
        """Run per-wavelength brightness calibration. See :func:`amplitude.measurements.brightness.run`."""
        self._require_prepared()
        return brightness.run(self._ctx, override, use_current_as_initial, xpol)

    def reference_measurement(self, xpol: bool | None = None) -> Path:
        """Capture a reference image stack. See :func:`amplitude.measurements.reference.run`."""
        self._require_prepared()
        return reference.run(self._ctx, xpol)

    def black_measurement(self, xpol: bool | None = None) -> Path:
        """Capture a dark-frame image stack. See :func:`amplitude.measurements.black.run`."""
        self._require_prepared()
        return black.run(self._ctx, xpol)

    def sample_measurement(self, xpol: bool | None = None) -> Path:
        """Capture a sample image stack. See :func:`amplitude.measurements.sample.run`."""
        self._require_prepared()
        return sample.run(self._ctx, xpol)
