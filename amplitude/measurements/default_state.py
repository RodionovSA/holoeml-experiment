"""Return hardware to its canonical resting state."""

from __future__ import annotations

import numpy as np

from amplitude.measurements.context import MeasurementContext


def run(ctx: MeasurementContext) -> None:
    """Move hardware to the canonical resting state.

    Sets the monochromator to ``config.default_wavelength``, the filter
    wheel to the empty position, the focus motor to
    ``config.default_focus_position`` (if set), and the camera to the
    initial calibration settings defined in config.
    """
    cfg = ctx.config
    ctx.mono.set_wavelength(cfg.default_wavelength)
    ctx.filterwheel.set_position(cfg.filterwheel_empty_pos)

    if cfg.default_focus_position is not None:
        ctx.focus.move_to(cfg.default_focus_position)
    ctx.camera.set_settings(
        exposure_time_us=cfg.calib_initial_exposure_ms * 1000,
        gain=cfg.calib_initial_gain,
        black_level=cfg.camera_black_level,
        bit_depth=getattr(np, cfg.camera_bit_depth),
        out_bit_depth=getattr(np, cfg.camera_out_bit_depth),
    )

    if cfg.polarizer_x_position is not None:
        ctx.polarizer.move_to(cfg.polarizer_x_position)
