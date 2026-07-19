"""Reference-image measurement."""

from __future__ import annotations

from pathlib import Path

from amplitude.measurements.context import MeasurementContext


def run(ctx: MeasurementContext, xpol: bool | None = None) -> Path:
    """Capture one reference image per sweep wavelength and save as NPZ.

    Moves the monochromator to each wavelength, applies the per-wavelength
    exposure settings from :attr:`exposure_settings`, positions the filter
    wheel (long-pass above ``filter_wvl``, open/empty below), captures an
    image, and saves the full stack to ``save_dir``.

    Parameters
    ----------
    xpol : bool or None
        ``True`` moves the polarizer to x-polarization and uses
        :attr:`exposure_settings_x`; ``False`` uses y-polarization and
        :attr:`exposure_settings_y`; ``None`` skips polarizer movement and
        uses the unpolarized :attr:`exposure_settings`.

    Returns
    -------
    Path
        Path to the saved NPZ file.

    Raises
    ------
    RuntimeError
        If the selected :attr:`exposure_settings` does not match the config wavelengths.
    """
    cfg = ctx.config
    wavelengths = ctx.sweep_wavelengths()
    ctx.move_polarizer(xpol)

    es = ctx.exposure_for(xpol)
    es_idx = ctx.map_wavelengths(es.wavelengths, wavelengths, "Exposure")

    images = []
    with ctx.armed():
        for i, wvl in enumerate(wavelengths):
            ctx.mono.set_wavelength(wvl)
            ctx.set_filterwheel_for_wvl(wvl)
            ctx.apply_camera_exposure(es, es_idx, i)
            images.append(ctx.camera.get_image(
                num_frames_to_average=cfg.num_frames_to_average,
                num_frames_to_drop=cfg.num_frames_to_drop,
                delay=cfg.capture_delay,
            ))

    return ctx.save_npz('reference', xpol, wavelengths, images)
