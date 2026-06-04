from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class ExposureSettings:
    """Per-wavelength camera exposure settings persisted as a JSON file."""

    wavelengths: list[float] = field(default_factory=list)
    exposure_ms: list[float] = field(default_factory=list)
    gain: list[int] = field(default_factory=list)
    best_brightness: list[float] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> ExposureSettings:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class Config:
    """Experiment configuration for all hardware devices.

    Holds connection parameters and sweep settings for the monochromator,
    camera, focus motor, and filter wheel.  Use :meth:`to_yaml` /
    :meth:`from_yaml` to persist and restore a configuration.
    """

    # Monochromator
    monochromator_port: str
    """Serial port for the Arduino monochromator controller (e.g. ``'COM4'``)."""
    monochromator_grating_mode: str
    """Active grating mode: ``'VIS Grating'``, ``'IR Grating'``, or ``'Switch Mode'``."""
    wvl_start: float
    """Start wavelength of the sweep (nm)."""
    wvl_stop: float
    """Stop wavelength of the sweep (nm)."""
    wvl_num: int
    """Number of wavelength steps in the sweep."""
    filter_wvl: float
    """Wavelength to turn on the logpass filter"""

    # Camera
    camera_serial: str
    """Serial number string of the Thorlabs camera."""
    exposure_settings_path: str = "exposure_settings.json"
    """Path to the per-wavelength exposure settings JSON file."""
    camera_black_level: int = 0
    """Camera black level offset."""
    camera_bit_depth: str = 'uint16'
    """Raw frame dtype passed to numpy (e.g. ``'uint16'``, ``'uint8'``)."""
    camera_out_bit_depth: str = 'float32'
    """Output image dtype after averaging (e.g. ``'float32'``, ``'float64'``)."""

    # Brightness calibration
    calib_target_brightness: float = 0.5
    """Target mean pixel value (normalised 0–1) for brightness calibration."""
    calib_tolerance: float = 0.02
    """Acceptable deviation from target before calibration stops."""
    calib_increment: float = 0.1
    """Multiplicative step size (±10% per iteration) for exposure time adjustment."""
    calib_gain_step_db: float = 0.5
    """Fixed gain step in dB per iteration (1 dB = 10 SDK units). Additive, not multiplicative."""
    calib_max_steps: int = 50
    """Maximum iterations per wavelength before giving up."""
    calib_max_exposure_ms: int = 500
    """Upper exposure limit in milliseconds."""
    calib_initial_exposure_ms: int = 1
    """Starting exposure in milliseconds for the first wavelength."""
    calib_initial_gain: int = 0
    """Starting gain for the first wavelength."""
    calib_priority: str = 'exposure_time'
    """Which control to exhaust first: ``'exposure_time'`` or ``'gain'``."""
    calib_num_frames_to_average: int = 1
    """Frames to average per brightness measurement."""
    calib_num_frames_to_drop: int = 5
    """Frames to discard before averaging."""
    calib_delay: float = 0.0
    """Delay in seconds between frame captures during calibration."""

    # Default state
    default_wavelength: float = 550.0
    """Wavelength (nm) the system returns to after init, calibration, and measurement."""

    # z-translation motor
    focus_serial: str = ""
    """Serial number string of the Thorlabs Kinesis focus motor."""

    # Filterwheel
    filterwheel_address: str = ""
    """VISA / serial address of the filter wheel controller."""
    filterwheel_timeout: int = 2000
    """Communication timeout for the filter wheel (ms)."""
    black_pos: int = 4  # 1-6
    """Filter wheel position (1–6) that blocks all light (beam block)."""
    longpass_pos: int = 3  # 1-6
    """Filter wheel position (1–6) of the long-pass filter."""
    filterwheel_empty_pos: int = 1  # 1-6
    """Filter wheel position (1–6) of the open/transmission window."""

    # Measurements (reference, black, sample)
    save_dir: str = '.'
    """Directory where all measurement NPZ files are saved (reference, black, sample)."""
    num_frames_to_average: int = 1
    """Frames to average per wavelength for all measurement types."""
    num_frames_to_drop: int = 5
    """Frames to discard before averaging for all measurement types."""
    capture_delay: float = 0.0
    """Delay in seconds between frame captures for all measurement types."""

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_yaml(self, path: str | Path) -> None:
        """Write this configuration to a YAML file.

        Args:
            path: Destination file path.  The file is created or overwritten.
        """
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a configuration from a YAML file produced by :meth:`to_yaml`.

        Args:
            path: Path to the YAML file.

        Returns:
            A fully populated :class:`Config` instance.
        """
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
