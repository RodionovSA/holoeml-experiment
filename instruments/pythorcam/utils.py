"""Moved to :mod:`instruments.camera.utils` -- none of this code is
Thorlabs-specific, so it now lives alongside the vendor-neutral `Camera`
interface and works with any camera driver (`ThorlabsCamera`, `PcoCamera`, ...).

Re-exported here so existing imports (`phase/control.py`, `scripts/focus.py`,
`scripts/test/thorcam_noise_bright.py`) keep working unchanged. New code
should import from `instruments.camera.utils` directly.
"""

from instruments.camera.utils import (
    autoexposure,
    autofocus,
    brightness_calibration,
    calculate_focus_measure,
    live_view,
    zoom,
)

__all__ = [
    "autoexposure",
    "autofocus",
    "brightness_calibration",
    "calculate_focus_measure",
    "live_view",
    "zoom",
]
