"""Per-polarization settings persistence for the amplitude experiment.

Pulls the exposure/focus settings-file bookkeeping out of the orchestrator
(:class:`~amplitude.control.Control`): loading/creating the JSON files,
picking the right one for a given polarization, and merging freshly
calibrated values into the stored settings.

``Polarization`` replaces the repeated ``if xpol is True / False / else``
three-way branches that used to appear at every call site. Measurement
methods still take a plain ``xpol: bool | None`` (matching the existing
CLI/config surface); convert with :meth:`Polarization.from_xpol` at the
boundary.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np

from amplitude.config import Config, ExposureSettings, FocusSettings


class Polarization(Enum):
    """Which polarization state a measurement/calibration applies to."""

    X = "x"
    Y = "y"
    NONE = "none"

    @classmethod
    def from_xpol(cls, xpol: bool | None) -> Polarization:
        if xpol is True:
            return cls.X
        if xpol is False:
            return cls.Y
        return cls.NONE


def _load_or_init(path_str: str, cls):
    path = Path(path_str)
    if not path.exists():
        cls().save(path)
    return path, cls.load(path)


class ExposureStore:
    """Owns the three per-wavelength exposure-settings files (x/y/none)."""

    def __init__(self,
                 paths: dict[Polarization, Path],
                 settings: dict[Polarization, ExposureSettings]):
        self._paths = paths
        self._settings = settings

    @classmethod
    def from_config(cls, config: Config) -> ExposureStore:
        none_path, none_settings = _load_or_init(config.exposure_settings_path, ExposureSettings)
        x_path, x_settings = _load_or_init(config.exposure_settings_xpol_path, ExposureSettings)
        y_path, y_settings = _load_or_init(config.exposure_settings_ypol_path, ExposureSettings)
        return cls(
            paths={Polarization.NONE: none_path, Polarization.X: x_path, Polarization.Y: y_path},
            settings={Polarization.NONE: none_settings, Polarization.X: x_settings, Polarization.Y: y_settings},
        )

    def get(self, pol: Polarization) -> ExposureSettings:
        return self._settings[pol]

    def reload(self, pol: Polarization) -> ExposureSettings:
        settings = ExposureSettings.load(self._paths[pol])
        self._settings[pol] = settings
        return settings

    def save(self, pol: Polarization) -> None:
        self._settings[pol].save(self._paths[pol])

    def merge_calibration(self,
                          pol: Polarization,
                          wavelengths: list[float],
                          exposures_ms: list[float],
                          gains: list[int],
                          brightnesses: list[float]) -> None:
        """Merge freshly calibrated per-wavelength values into the stored settings.

        Existing wavelengths (within 0.01 nm) are updated in place; new
        wavelengths are inserted in sorted order. Saves the result and makes
        it the current settings for *pol*.
        """
        current = self._settings[pol]
        stored_wvls = np.array(current.wavelengths)
        merged = ExposureSettings(
            wavelengths=list(current.wavelengths),
            exposure_ms=list(current.exposure_ms),
            gain=list(current.gain),
            best_brightness=list(current.best_brightness),
        )
        for wvl, exp_ms, gain, brightness in zip(wavelengths, exposures_ms, gains, brightnesses):
            diffs = np.abs(stored_wvls - wvl) if len(stored_wvls) else np.array([float('inf')])
            idx = int(np.argmin(diffs))
            if len(stored_wvls) and diffs[idx] <= 0.01:
                merged.exposure_ms[idx] = exp_ms
                merged.gain[idx] = gain
                merged.best_brightness[idx] = brightness
            else:
                # New wavelength not previously in file -- insert in sorted order.
                pos = int(np.searchsorted(stored_wvls, wvl))
                merged.wavelengths.insert(pos, float(wvl))
                merged.exposure_ms.insert(pos, exp_ms)
                merged.gain.insert(pos, gain)
                merged.best_brightness.insert(pos, brightness)
                stored_wvls = np.insert(stored_wvls, pos, wvl)

        self._settings[pol] = merged
        self.save(pol)


class FocusStore:
    """Owns the per-wavelength focus-offsets file."""

    def __init__(self, path: Path, settings: FocusSettings):
        self._path = path
        self.settings = settings

    @classmethod
    def from_config(cls, config: Config) -> FocusStore:
        path, settings = _load_or_init(config.focus_settings_path, FocusSettings)
        return cls(path, settings)

    def get(self) -> FocusSettings:
        return self.settings

    def reload(self) -> FocusSettings:
        self.settings = FocusSettings.load(self._path)
        return self.settings

    def save(self) -> None:
        self.settings.save(self._path)
