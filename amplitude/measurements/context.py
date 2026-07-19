"""Shared operational surface for measurement/calibration routines.

Each routine module in :mod:`amplitude.measurements` (``brightness``,
``reference``, ``black``, ``sample``, ``default_state``) receives a single
:class:`MeasurementContext` instead of the four collaborators
(:class:`~amplitude.devices.Instruments`, :class:`~amplitude.config.Config`,
:class:`~amplitude.settings_store.ExposureStore`,
:class:`~amplitude.settings_store.FocusStore`) individually. It exposes
device shortcuts and the handful of primitives every routine needs
(wavelength sweep, polarizer/filter-wheel positioning, exposure application,
NPZ saving, settings lookup by polarization).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from amplitude.config import Config, ExposureSettings
from amplitude.settings_store import ExposureStore, FocusStore, Polarization

if TYPE_CHECKING:
    # Only needed for the type hint below -- avoided at runtime so this module
    # (and the routines built on it) can be imported/tested without the
    # hardware SDKs that amplitude.devices pulls in.
    from amplitude.devices import Instruments


class MeasurementContext:
    def __init__(self,
                 instruments: Instruments,
                 config: Config,
                 exposure_store: ExposureStore,
                 focus_store: FocusStore):
        self.instruments = instruments
        self.config = config
        self.exposure_store = exposure_store
        self.focus_store = focus_store

    # ------------------------------------------------------------------ #
    # Device shortcuts                                                     #
    # ------------------------------------------------------------------ #

    @property
    def camera(self):
        return self.instruments.camera

    @property
    def mono(self):
        return self.instruments.mono

    @property
    def focus(self):
        return self.instruments.focus

    @property
    def polarizer(self):
        return self.instruments.polarizer

    @property
    def filterwheel(self):
        return self.instruments.filterwheel

    def armed(self):
        return self.instruments.armed()

    # ------------------------------------------------------------------ #
    # Shared primitives                                                    #
    # ------------------------------------------------------------------ #

    def sweep_wavelengths(self) -> np.ndarray:
        cfg = self.config
        return np.linspace(cfg.wvl_start, cfg.wvl_stop, cfg.wvl_num)

    def move_polarizer(self, xpol: bool | None) -> None:
        if xpol is not None:
            angle = self.config.polarizer_x_position + (0 if xpol else 90)
            self.polarizer.move_to(angle)

    def set_filterwheel_for_wvl(self, wvl: float) -> None:
        cfg = self.config
        fw = self.filterwheel
        fw_pos = cfg.longpass_pos if wvl >= cfg.filter_wvl else cfg.filterwheel_empty_pos
        if fw.get_position() != fw_pos:
            fw.set_position(fw_pos)

    def apply_camera_exposure(self, es: ExposureSettings, es_idx: list[int], i: int) -> None:
        camera = self.camera
        camera.set_exposure_time_us(int(es.exposure_ms[es_idx[i]] * 1000))
        camera.set_gain(es.gain[es_idx[i]])

    def save_npz(self, prefix: str, xpol: bool | None,
                 wavelengths: np.ndarray, images: list) -> Path:
        pol_label = {True: 'xpol_', False: 'ypol_', None: ''}[xpol]
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = save_dir / f'{prefix}_{pol_label}{timestamp}.npz'
        np.savez(save_path, wavelengths=wavelengths, images=np.stack(images))
        return save_path

    def map_wavelengths(self,
                        settings_wavelengths: list[float],
                        config_wavelengths: np.ndarray,
                        name: str,
                        atol: float = 0.01) -> list[int]:
        """Return indices into *settings_wavelengths* for each wavelength in *config_wavelengths*.

        Raises RuntimeError if any config wavelength has no match within *atol* nm.
        """
        settings_arr = np.array(settings_wavelengths)
        if settings_arr.size == 0:
            raise RuntimeError(
                f"{name} settings are empty. Run calibration first."
            )
        indices = []
        for wvl in config_wavelengths:
            diffs = np.abs(settings_arr - wvl)
            idx = int(np.argmin(diffs))
            if diffs[idx] > atol:
                raise RuntimeError(
                    f"{name} settings do not cover wavelength {wvl:.2f} nm. "
                    "Run calibration first."
                )
            indices.append(idx)
        return indices

    def exposure_for(self, xpol: bool | None) -> ExposureSettings:
        return self.exposure_store.get(Polarization.from_xpol(xpol))
