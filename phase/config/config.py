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
class FocusSettings:
    """Per-wavelength focus offsets (mm) relative to ``Config.default_focus_position``."""

    wavelengths: list[float] = field(default_factory=list)
    offsets: list[float] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> FocusSettings:
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

    # ── Wavelength sweep (required) ───────────────────────────────────────────
    wvl_start: float
    """Start wavelength of the sweep (nm)."""
    wvl_stop: float
    """Stop wavelength of the sweep (nm)."""
    wvl_num: int
    """Number of wavelength steps in the sweep (linspace)."""
    filter_wvl: float
    """Wavelength (nm) at which the long-pass filter is inserted."""

    # ── Monochromator (required) ──────────────────────────────────────────────
    monochromator_port: str
    """Serial port for the Arduino monochromator controller (e.g. ``'COM4'``). Leave empty to be prompted at runtime."""
    monochromator_grating_mode: str
    """Active grating mode: ``'VIS Grating'``, ``'IR Grating'``, or ``'Switch Mode'``."""

    # ── Camera (required) ─────────────────────────────────────────────────────
    camera_serial: str
    """Serial number string of the Thorlabs camera."""

    # ─────────────────────────────────────────────────────────────────────────
    # Optional fields — all parameters below have defaults
    # ─────────────────────────────────────────────────────────────────────────

    # ── Wavelength sweep ──────────────────────────────────────────────────────
    default_wavelength: float = 550.0
    """Wavelength (nm) the system returns to after init, calibration, and measurement."""

    # ── Camera ────────────────────────────────────────────────────────────────
    camera_black_level: int = 0
    """Camera black level offset."""
    camera_bit_depth: str = 'uint16'
    """Raw frame dtype passed to numpy (e.g. ``'uint16'``, ``'uint8'``)."""
    camera_out_bit_depth: str = 'float32'
    """Output image dtype after averaging (e.g. ``'float32'``, ``'float64'``)."""

    # ── Focus motor ───────────────────────────────────────────────────────────
    focus_serial: str = ""
    """Serial number string of the Thorlabs Kinesis focus motor."""
    default_focus_position: float | None = None
    """Focus motor position (mm) the system returns to between operations. ``None`` skips focus movement."""
    default_focus_max_velocity: float | None = None
    """Focus motor max velocity (mm/s). ``None`` skips."""
    default_focus_acceleration: float | None = None
    """Focus motor acceleration (mm/s²). ``None`` skips."""
    focus_use_current_position: bool = False
    """When ``True``, use the motor's position at sweep start as the focus baseline instead of ``default_focus_position``.  All per-wavelength offsets are applied relative to that position."""

    # ── Polarizer motor ───────────────────────────────────────────────────────
    polarizer_serial: str = ""
    """Serial number string of the Thorlabs Kinesis polarizer rotation stage."""
    default_polarizer_home_velocity: float | None = None
    """Polarizer homing velocity (deg/s). ``None`` skips focus movement."""
    default_polarizer_max_velocity: float | None = None
    """Polarizer motor max velocity (deg/s). ``None`` skips."""
    default_polarizer_acceleration: float | None = None
    """Polarizer motor acceleration (deg/s²). ``None`` skips."""
    polarizer_x_position: float | None = None
    """Polarizer position for x-polarization state. ``None`` skips."""

    # ── Filter wheel ──────────────────────────────────────────────────────────
    filterwheel_address: str = ""
    """VISA / USB address of the Newport filter wheel (e.g. ``'USB0::...'``)."""
    filterwheel_timeout: int = 2000
    """Communication timeout for the filter wheel (ms)."""
    filterwheel_empty_pos: int = 1
    """Filter wheel position (1–6) of the open/transmission window."""
    longpass_pos: int = 3
    """Filter wheel position (1–6) of the long-pass filter."""
    black_pos: int = 4
    """Filter wheel position (1–6) that blocks all light (beam block)."""

    # ── Brightness calibration ────────────────────────────────────────────────
    calib_target_brightness: float = 0.5
    """Target mean pixel value (normalised 0–1) for brightness calibration."""
    calib_tolerance: float = 0.02
    """Acceptable deviation from target before calibration stops."""
    calib_increment: float = 0.1
    """Multiplicative step size (±10% per iteration) for exposure time adjustment."""
    calib_gain_step_db: float = 0.5
    """Fixed gain step in dB per iteration (1 dB = 10 SDK units). Additive, not multiplicative."""
    calib_priority: str = 'exposure_time'
    """Which control to exhaust first: ``'exposure_time'`` or ``'gain'``."""
    calib_max_steps: int = 50
    """Maximum iterations per wavelength before giving up."""
    calib_max_exposure_ms: int = 500
    """Upper exposure limit in milliseconds."""
    calib_initial_exposure_ms: int = 1
    """Starting exposure in milliseconds for the first wavelength."""
    calib_initial_gain: int = 0
    """Starting gain for the first wavelength."""
    calib_num_frames_to_average: int = 1
    """Frames to average per brightness measurement."""
    calib_num_frames_to_drop: int = 5
    """Frames to discard before averaging."""
    calib_delay: float = 0.0
    """Delay in seconds between frame captures during calibration."""
    calib_roi_fraction: float | None = None
    """Central fraction of image (0–1) used for brightness averaging during calibration. ``None`` uses the full frame."""

    # ── Measurements ──────────────────────────────────────────────────────────
    save_dir: str = '.'
    """Directory where all measurement NPZ files are saved (reference, black, sample)."""
    num_frames_to_average: int = 1
    """Frames to average per wavelength for all measurement types."""
    num_frames_to_drop: int = 5
    """Frames to discard before averaging for all measurement types."""
    capture_delay: float = 0.0
    """Delay in seconds between frame captures for all measurement types."""
    exposure_settings_path: str = "exposure_settings.json"
    """Path to the per-wavelength exposure settings JSON file."""
    exposure_settings_xpol_path: str = "exposure_settings_xpol.json"
    """Path to the per-wavelength exposure settings JSON file for x-polarization."""
    exposure_settings_ypol_path: str = "exposure_settings_ypol.json"
    """Path to the per-wavelength exposure settings JSON file for y-polarization."""
    focus_settings_path: str = "focus_settings.json"
    """Path to the per-wavelength focus positions JSON file."""

    # ── Autofocus ─────────────────────────────────────────────────────────────
    autofocus_enabled: bool = False
    """When ``True``, ``sample_measurement`` runs a focus scan at the first wavelength of each focus range."""
    autofocus_range_boundaries: list[float] = field(default_factory=lambda: [440.0, 465.0])
    """Wavelength boundaries (nm) separating focus regimes (ascending). Creates N+1 ranges."""
    autofocus_scan_half_range_mm: float = 0.05
    """Half-width of the position scan around the stored focus target (mm)."""
    autofocus_step_size_mm: float = 0.002
    """Motor step size during the autofocus scan (mm)."""
    autofocus_velocity_mm_s: float = 0.1
    """Motor velocity during the autofocus scan (mm/s)."""
    autofocus_num_frames_to_drop: int = 3
    """Frames to discard at each scan position before capturing the focus image."""

    # ─────────────────────────────────────────────────────────────────────────
    # Serialisation helpers
    # ─────────────────────────────────────────────────────────────────────────

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
