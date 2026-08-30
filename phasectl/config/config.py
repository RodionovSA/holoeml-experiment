from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from instruments.config import load_equipment


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
    """Serial number string of the camera."""
    num_frames_to_average: int = 1
    """Frames to average per wavelength for all measurement types."""
    num_frames_to_drop: int = 5
    """Frames to discard before averaging for all measurement types."""
    capture_delay: float = 0.0
    """Delay in seconds between frame captures for all measurement types."""

    # ─────────────────────────────────────────────────────────────────────────
    # Optional fields — all parameters below have defaults
    # ─────────────────────────────────────────────────────────────────────────

    # ── Wavelength sweep ──────────────────────────────────────────────────────
    default_wavelength: float = 550.0
    """Wavelength (nm) the system returns to after init, calibration, and measurement."""

    # ── Camera ────────────────────────────────────────────────────────────────
    camera_vendor: str = "thorlabs"
    """Camera SDK to use: ``'thorlabs'`` or ``'pco'``. See ``instruments.camera.open_camera``."""
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
    
    # ── Stage xy ───────────────────────────────────────────────────────────
    stage_x_serial: str = ""
    """Serial number string of the Thorlabs Kinesis stage x motor."""
    stage_y_serial: str = ""
    """Serial number string of the Thorlabs Kinesis stage y motor."""
    default_stage_max_velocity: float | None = None
    """Stage xy motor max velocity (mm/s). ``None`` skips."""
    default_stage_acceleration: float | None = None
    """Stage xy motor acceleration (mm/s²). ``None`` skips."""
    
    # ── Piezo (phase protocol) ───────────────────────────────────────────────
    piezo_serial: str = ""
    """Serial number string of the CT1P (IntegratedPrecisionPiezo) controller."""

    # ── Inertial piezo (KIM101 / PIAK10) ─────────────────────────────────────
    kim_serial: str = ""
    """Serial number string of the KIM101 inertial piezo controller."""
    kim_channel: int = 4
    """KIM101 channel (1-4) the PIAK10 phase-shifter actuator is wired to."""
    kim_step_rate: int = 200
    """KIM101 drive step rate (steps/s)."""
    kim_step_acceleration: int = 500
    """KIM101 drive step acceleration (steps/s²)."""
    kim_max_voltage: int = 125
    """KIM101 drive max voltage (V)."""

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
    exposure_settings_path: str = "exposure_settings.json"
    """Path to the per-wavelength exposure settings JSON file."""
    exposure_settings_xpol_path: str = "exposure_settings_xpol.json"
    """Path to the per-wavelength exposure settings JSON file for x-polarization."""
    exposure_settings_ypol_path: str = "exposure_settings_ypol.json"
    """Path to the per-wavelength exposure settings JSON file for y-polarization."""
    focus_settings_path: str = "focus_settings.json"
    """Path to the per-wavelength focus positions JSON file."""

    # ── Phase acquisition (piezo phase-shift sweep) ────────────────────────────
    num_piezo_steps: int = 10
    """Camera frames (and piezo positions) per phase-shift sweep."""
    num_piezo_return_steps: int = 4
    """Backward moves of ``piezo_step_size_bwd`` walked per average after all
    forward repeats, to bring the actuator back (see phase_measurement.py)."""
    piezo_step_size_fwd: int = 5
    """Commanded steps per forward phase-shift move."""
    piezo_step_size_bwd: int = 5
    """Commanded steps per backward move. Calibrated independently of
    ``piezo_step_size_fwd`` -- the slip-stick actuator covers different physical
    distance per commanded step depending on direction."""
    piezo_settle_s: float = 0.05
    """Pause after each piezo move before capturing (mechanical settling)."""
    piezo_dry_num: int = 10
    """Backlash-compensation dry-run moves before each direction change."""
    piezo_dry_step: int = 1
    """Commanded steps per dry-run move."""
    num_averages: int = 2
    """Independent phase-shift repeats combined per phase measurement."""

    # ── Reference position (sample stage offset) ────────────────────────────────
    reference_x_by: float = 0.0
    """Relative stage-x move (mm) from sample to reference position."""
    reference_y_by: float = 0.8
    """Relative stage-y move (mm) from sample to reference position."""

    # ── Image ROI ────────────────────────────────────────────────────────────
    roi_x0: int = 0
    roi_y0: int = 0
    roi_x1: int | None = None
    roi_y1: int | None = None
    """Crop applied to each captured frame: ``[y0:y1, x0:x1]``. ``None`` = full extent."""

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
    def from_yaml(cls, path: str | Path, equipment_path: str | Path | None = None) -> Config:
        """Load a configuration from a YAML file produced by :meth:`to_yaml`.

        Shared equipment settings (device serials, filter wheel timeout/positions,
        polarizer position/velocities, focus velocities, camera bit depth, etc.)
        are loaded from the shared :mod:`instruments.config` store and merged
        underneath ``path``, so ``path`` only needs to specify protocol-specific
        parameters and may override any equipment value if needed.

        Args:
            path: Path to the protocol YAML file.
            equipment_path: Path to the shared equipment YAML file. Defaults to
                ``instruments/config/config.yaml`` (see
                :func:`instruments.config.load_equipment`).

        Returns:
            A fully populated :class:`Config` instance.
        """
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        equipment = asdict(load_equipment(equipment_path))
        merged = {**equipment, **data}
        valid_fields = {f.name for f in fields(cls)}
        merged = {k: v for k, v in merged.items() if k in valid_fields}
        return cls(**merged)
