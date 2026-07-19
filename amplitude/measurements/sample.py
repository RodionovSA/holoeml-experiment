"""Sample-image measurement, with optional per-range autofocus."""

from __future__ import annotations

import bisect
from pathlib import Path

from instruments.pythorcam.utils import autofocus as _autofocus

from amplitude.config import Config
from amplitude.measurements.context import MeasurementContext


def _wavelength_range_idx(config: Config, wavelength_nm: float) -> int:
    """Return the focus-range index for *wavelength_nm*.

    Ranges are defined by ``config.autofocus_range_boundaries`` (ascending nm).
    Index 0 covers everything below the first boundary; index N covers
    everything at or above the last boundary.  The boundary wavelength itself
    belongs to the upper range (bisect_right semantics).
    """
    return bisect.bisect_right(config.autofocus_range_boundaries, wavelength_nm)


def _run_autofocus(ctx: MeasurementContext, center_position_mm: float) -> float:
    """Run a focus scan centred on *center_position_mm* and return the best position in mm.

    The camera must already be armed.  Scan range and step size are taken
    from config.  Motor velocity is set to ``autofocus_velocity_mm_s`` for the
    scan; the caller is responsible for restoring sweep velocity afterwards.
    """
    cfg = ctx.config
    _mm = 1e-3
    start_m = (center_position_mm - cfg.autofocus_scan_half_range_mm) * _mm
    end_m   = (center_position_mm + cfg.autofocus_scan_half_range_mm) * _mm
    step_m  = cfg.autofocus_step_size_mm * _mm
    vel_m   = cfg.autofocus_velocity_mm_s * _mm
    best_pos_m, _curve, _images = _autofocus(
        camera_trans=ctx.camera,
        focus_motor=ctx.focus.km,
        start_position=start_m,
        end_position=end_m,
        step_size=step_m,
        velocity=vel_m,
        camera_ref=None,
        num_frames_to_average=1,
        num_frames_to_drop=cfg.autofocus_num_frames_to_drop,
        delay=0,
    )
    best_mm = best_pos_m / _mm
    print(f'[autofocus] best position: {best_mm:.4f} mm')
    return best_mm


def run(ctx: MeasurementContext, xpol: bool | None = None) -> Path:
    """Capture one sample image per sweep wavelength and save as NPZ.

    Moves the monochromator, filter wheel, and focus motor to per-wavelength
    positions.  Focus movement is skipped when the motor is already within
    0.5 motor units of the target.  Uses per-wavelength exposure and gain
    from the polarization-selected :attr:`exposure_settings` and positions
    from :attr:`focus_settings`.

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
        If either settings file does not match the config wavelengths.
    """
    cfg = ctx.config
    wavelengths = ctx.sweep_wavelengths()
    ctx.move_polarizer(xpol)

    es = ctx.exposure_for(xpol)
    es_idx    = ctx.map_wavelengths(es.wavelengths, wavelengths, "Exposure")
    focus_settings = ctx.focus_store.get()
    focus_idx = ctx.map_wavelengths(focus_settings.wavelengths, wavelengths, "Focus")

    if cfg.focus_use_current_position:
        focus_base_mm = ctx.focus.get_position()
    else:
        if cfg.default_focus_position is None:
            raise RuntimeError(
                "default_focus_position must be set in config to use focus_settings offsets."
            )
        focus_base_mm = cfg.default_focus_position

    images = []
    range_corrections: dict[int, float] = {}  # range_idx -> position correction (mm)
    with ctx.armed():
        for i, wvl in enumerate(wavelengths):
            ctx.mono.set_wavelength(wvl)
            ctx.set_filterwheel_for_wvl(wvl)
            ctx.apply_camera_exposure(es, es_idx, i)

            stored_target_mm = focus_base_mm + focus_settings.offsets[focus_idx[i]]
            if cfg.autofocus_enabled:
                range_idx = _wavelength_range_idx(cfg, wvl)
                if range_idx not in range_corrections:
                    # First wavelength in this range: move to stored target then autofocus.
                    ctx.focus.move_to(stored_target_mm)
                    best_mm = _run_autofocus(ctx, center_position_mm=stored_target_mm)
                    range_corrections[range_idx] = best_mm - stored_target_mm
                    print(f'[autofocus] range {range_idx}  wvl={wvl:.1f} nm  '
                          f'correction={range_corrections[range_idx]:+.4f} mm')
                    # Restore sweep velocity after the slow autofocus scan.
                    _v, _a = cfg.default_focus_max_velocity, cfg.default_focus_acceleration
                    ctx.focus.set_velocity(
                        max_velocity=(_v if _v is not None else None),
                        acceleration=(_a if _a is not None else None),
                    )
                target_mm = (stored_target_mm + range_corrections[range_idx])
            else:
                target_mm = stored_target_mm

            if abs(ctx.focus.get_position() - target_mm) > 2e-3:
                ctx.focus.move_to(target_mm)

            images.append(ctx.camera.get_image(
                num_frames_to_average=cfg.num_frames_to_average,
                num_frames_to_drop=cfg.num_frames_to_drop,
                delay=cfg.capture_delay,
            ))

    if cfg.autofocus_enabled and range_corrections:
        for range_idx, correction_mm in range_corrections.items():
            for j, wvl_j in enumerate(wavelengths):
                if _wavelength_range_idx(cfg, wvl_j) == range_idx:
                    focus_settings.offsets[focus_idx[j]] += correction_mm
        ctx.focus_store.save()

    return ctx.save_npz('sample', xpol, wavelengths, images)
