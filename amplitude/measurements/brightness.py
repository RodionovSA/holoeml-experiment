"""Per-wavelength brightness calibration."""

from __future__ import annotations

from instruments.pythorcam.utils import brightness_calibration as _brightness_calibration

from amplitude.measurements.context import MeasurementContext
from amplitude.settings_store import Polarization


def run(ctx: MeasurementContext,
        override: bool = False,
        use_current_as_initial: bool = False,
        xpol: bool | None = True) -> None:
    """Run per-wavelength brightness calibration and store results.

    All calibration parameters are read from :attr:`ctx.config`.  Iterates over
    every wavelength in the config sweep, moves the monochromator and filter
    wheel (matching measurement conditions), and calls
    :func:`brightness_calibration` from ``utils``.

    By default the exposure and gain found at wavelength N seed the next
    wavelength to speed up convergence.  When *use_current_as_initial* is
    ``True`` and stored settings already cover all sweep wavelengths, each
    wavelength is instead seeded from its own stored value, which can
    accelerate re-calibration significantly.

    Returns immediately if the settings file already contains entries for
    all config wavelengths, unless *override* is ``True``.
    """
    cfg = ctx.config
    wavelengths = ctx.sweep_wavelengths()
    ctx.move_polarizer(xpol)

    pol = Polarization.from_xpol(xpol)
    current_settings = ctx.exposure_store.get(pol)

    if not override:
        try:
            ctx.map_wavelengths(current_settings.wavelengths, wavelengths, "Exposure")
            return  # all config wavelengths already calibrated
        except RuntimeError:
            pass

    # Build per-wavelength seeds if using current stored values.
    if use_current_as_initial:
        try:
            seed_idx = ctx.map_wavelengths(current_settings.wavelengths, wavelengths, "Exposure")
            init_exp_us = [int(current_settings.exposure_ms[j] * 1000) for j in seed_idx]
            init_gain = [current_settings.gain[j] for j in seed_idx]
        except RuntimeError:
            init_exp_us = None
            init_gain = None
    else:
        init_exp_us = None
        init_gain = None

    # Cascade seeds for the non-current-initial path.
    exp_us = cfg.calib_initial_exposure_ms * 1000
    gain = cfg.calib_initial_gain

    calibrated_wavelengths: list[float] = []
    calibrated_exposures: list[float] = []
    calibrated_gains: list[int] = []
    calibrated_brightnesses: list[float] = []

    with ctx.armed():
        for i, wvl in enumerate(wavelengths):
            ctx.mono.set_wavelength(wvl)
            ctx.set_filterwheel_for_wvl(wvl)

            if init_exp_us is not None:
                exp_us = init_exp_us[i]
                gain = init_gain[i]

            exp_us, gain, brightness = _brightness_calibration(
                camera=ctx.camera,
                initial_exposure_time=exp_us,
                initial_gain=gain,
                target_brightness=cfg.calib_target_brightness,
                tolerance=cfg.calib_tolerance,
                increment=cfg.calib_increment,
                max_number_of_steps=cfg.calib_max_steps,
                max_exposure_time=cfg.calib_max_exposure_ms * 1000,
                priority=cfg.calib_priority,
                gain_step_db=cfg.calib_gain_step_db,
                num_frames_to_average=cfg.calib_num_frames_to_average,
                num_frames_to_drop=cfg.calib_num_frames_to_drop,
                delay=cfg.calib_delay,
                roi_fraction=cfg.calib_roi_fraction,
            )
            calibrated_wavelengths.append(float(wvl))
            calibrated_exposures.append(exp_us / 1000)
            calibrated_gains.append(gain)
            calibrated_brightnesses.append(brightness)

    ctx.exposure_store.merge_calibration(
        pol, calibrated_wavelengths, calibrated_exposures,
        calibrated_gains, calibrated_brightnesses,
    )
