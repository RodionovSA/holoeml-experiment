from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .spectrometer import Spectrometer


def autoexposure(
    sp: "Spectrometer",
    target: float = 0.80,
    *,
    tolerance: float = 0.05,
    max_iter: int = 20,
) -> int:
    """Adjust integration time so the peak intensity lands near ``target`` fraction
    of the detector's saturation value.

    Uses proportional scaling (new_t = current_t * target / fill) which converges
    in 1–2 steps for a linear detector.  Falls back to doubling when the spectrum
    is dark (all zeros), and halves when saturated.

    Args:
        sp: An open :class:`~instruments.spectrometer.Spectrometer` instance.
        target: Desired peak fill fraction, e.g. ``0.80`` for 80 % of saturation.
        tolerance: Acceptable deviation from *target*.  Stops when
            ``|fill - target| <= tolerance``.
        max_iter: Safety cap on iterations.

    Returns:
        Final integration time in microseconds.
    """
    if not (0.0 < target < 1.0):
        raise ValueError(f"target must be in (0, 1), got {target!r}")

    min_t, max_t = sp.get_integration_time_limits()
    max_counts = sp.get_max_intensity()

    for _ in range(max_iter):
        intensities = sp.get_intensities()
        peak = max(intensities)
        fill = peak / max_counts

        if abs(fill - target) <= tolerance:
            break

        current_t = sp.get_integration_time()

        if peak == 0:
            new_t = min(current_t * 2, max_t)
        else:
            new_t = int(current_t * target / fill)

        new_t = max(min_t, min(max_t, new_t))

        if new_t == current_t:
            break

        sp.set_integration_time(new_t)

    return sp.get_integration_time()
