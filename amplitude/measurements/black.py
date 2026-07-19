"""Dark-frame (beam-blocked) measurement."""

from __future__ import annotations

from pathlib import Path

from amplitude.measurements.context import MeasurementContext


def run(ctx: MeasurementContext, xpol: bool | None = None) -> Path:
    """Capture one dark frame per sweep wavelength with the beam blocked.

    Moves the filter wheel to ``config.black_pos`` once before capturing,
    then iterates over wavelengths applying per-wavelength exposure settings
    without moving the monochromator.  The polarizer is not moved (beam is
    blocked), but exposure settings are selected to match the intended
    measurement polarization.  Saves the image stack to ``save_dir``.

    Parameters
    ----------
    xpol : bool or None
        ``True`` uses :attr:`exposure_settings_x`; ``False`` uses
        :attr:`exposure_settings_y`; ``None`` uses :attr:`exposure_settings`.

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

    es = ctx.exposure_for(xpol)
    es_idx = ctx.map_wavelengths(es.wavelengths, wavelengths, "Exposure")

    ctx.filterwheel.set_position(cfg.black_pos)
    images = []
    with ctx.armed():
        for i, wvl in enumerate(wavelengths):
            print(f"wavelength {wvl}")
            ctx.apply_camera_exposure(es, es_idx, i)
            images.append(ctx.camera.get_image(
                num_frames_to_average=cfg.num_frames_to_average,
                num_frames_to_drop=cfg.num_frames_to_drop,
                delay=cfg.capture_delay,
            ))

    return ctx.save_npz('black', xpol, wavelengths, images)
