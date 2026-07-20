from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_EQUIPMENT_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass
class EquipmentConfig:
    """Shared, protocol-agnostic equipment configuration.

    Holds device serial numbers and hardware settings that are properties of
    the physical devices themselves (not of a particular measurement
    protocol), so they only need to be set once instead of being copied into
    every protocol's :class:`~amplitude.config.Config`.
    """

    # ── Serials / addresses ─────────────────────────────────────────────────
    camera_serial: str = ""
    """Serial number string of the Thorlabs camera."""
    focus_serial: str = ""
    """Serial number string of the Thorlabs Kinesis focus motor."""
    polarizer_serial: str = ""
    """Serial number string of the Thorlabs Kinesis polarizer rotation stage."""
    piezo_serial: str = ""
    """Serial number string of the CT1P (IntegratedPrecisionPiezo) controller."""
    powermeter_serial: str = ""
    """Serial number string of the Thorlabs power meter."""
    spectrometer_serial: str = ""
    """Serial number string of the spectrometer."""
    filterwheel_address: str = ""
    """VISA / USB address of the Newport filter wheel (e.g. ``'USB0::...'``)."""

    # ── Monochromator ────────────────────────────────────────────────────────
    monochromator_port: str = ""
    """Serial port for the Arduino monochromator controller (e.g. ``'COM4'``)."""
    monochromator_grating_mode: str = "VIS Grating"
    """Active grating mode: ``'VIS Grating'``, ``'IR Grating'``, or ``'Switch Mode'``."""

    # ── Camera ────────────────────────────────────────────────────────────────
    camera_black_level: int = 0
    """Camera black level offset."""
    camera_bit_depth: str = "uint16"
    """Raw frame dtype passed to numpy (e.g. ``'uint16'``, ``'uint8'``)."""
    camera_out_bit_depth: str = "float32"
    """Output image dtype after averaging (e.g. ``'float32'``, ``'float64'``)."""

    # ── Focus motor (kinematics only — reference position is protocol-specific) ─
    default_focus_max_velocity: float | None = None
    """Focus motor max velocity (mm/s). ``None`` skips."""
    default_focus_acceleration: float | None = None
    """Focus motor acceleration (mm/s²). ``None`` skips."""

    # ── Polarizer motor ──────────────────────────────────────────────────────
    polarizer_x_position: float | None = None
    """Polarizer position for x-polarization state. ``None`` skips."""
    default_polarizer_home_velocity: float | None = None
    """Polarizer homing velocity (deg/s). ``None`` skips."""
    default_polarizer_max_velocity: float | None = None
    """Polarizer motor max velocity (deg/s). ``None`` skips."""
    default_polarizer_acceleration: float | None = None
    """Polarizer motor acceleration (deg/s²). ``None`` skips."""

    # ── Filter wheel ──────────────────────────────────────────────────────────
    filterwheel_timeout: int = 2000
    """Communication timeout for the filter wheel (ms)."""
    filterwheel_empty_pos: int = 1
    """Filter wheel position (1-6) of the open/transmission window."""
    longpass_pos: int = 3
    """Filter wheel position (1-6) of the long-pass filter."""
    black_pos: int = 4
    """Filter wheel position (1-6) that blocks all light (beam block)."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> EquipmentConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


def load_equipment(path: str | Path | None = None) -> EquipmentConfig:
    """Load the shared equipment configuration.

    Args:
        path: Path to an equipment YAML file. Defaults to
            :data:`DEFAULT_EQUIPMENT_PATH` (``instruments/config/config.yaml``).
    """
    return EquipmentConfig.from_yaml(path or DEFAULT_EQUIPMENT_PATH)
