"""Shared operational surface for phase acquisition routines.

Each future routine module in :mod:`phasectl.measurements` will receive a
single :class:`MeasurementContext` instead of the two collaborators
(:class:`~phasectl.devices.Instruments`, :class:`~phasectl.config.Config`)
individually. It exposes device shortcuts and the primitives every phase
routine needs: piezo phase-shift sweeps, backlash-compensation dry runs,
stage referencing, and NPZ saving.
"""

from __future__ import annotations

import contextlib
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from phasectl.config import Config
from phasectl.utils import crop

if TYPE_CHECKING:
    from phasectl.devices import Instruments


class MeasurementContext:
    def __init__(self, instruments: Instruments, config: Config):
        self.instruments = instruments
        self.config = config

    @classmethod
    def from_config(cls, config: Config, devices) -> MeasurementContext:
        """Connect the requested devices and wrap them in a context.

        Args:
            config: Protocol configuration.
            devices: Which devices to open -- see
                :meth:`~phasectl.devices.Instruments.from_config`.
        """
        from phasectl.devices import Instruments
        return cls(Instruments.from_config(config, devices), config)

    def close(self) -> None:
        self.instruments.close()

    def __enter__(self) -> MeasurementContext:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Device shortcuts                                                     #
    # ------------------------------------------------------------------ #

    @property
    def camera(self):
        return self.instruments.camera

    @property
    def mono(self):
        return self.instruments.mono

    @property
    def filterwheel(self):
        return self.instruments.filterwheel

    @property
    def focus(self):
        return self.instruments.focus

    @property
    def polarizer(self):
        return self.instruments.polarizer

    @property
    def stage_x(self):
        return self.instruments.stage_x

    @property
    def stage_y(self):
        return self.instruments.stage_y

    @property
    def kim(self):
        return self.instruments.kim

    @property
    def piezo(self):
        return self.instruments.piezo

    def armed(self):
        return self.instruments.armed()

    # ------------------------------------------------------------------ #
    # Settings primitives                                                  #
    # ------------------------------------------------------------------ #

    def apply_drive_parameters(self) -> dict:
        """Apply the configured KIM101 drive parameters and return what the
        controller actually accepted (``{step_rate, step_acceleration,
        max_voltage}``), so callers can record it alongside a measurement."""
        cfg = self.config
        self.piezo.set_drive_parameters(
            step_rate=cfg.kim_step_rate,
            step_acceleration=cfg.kim_step_acceleration,
            max_voltage=cfg.kim_max_voltage,
        )
        return self.piezo.get_drive_parameters()

    def apply_stage_velocities(self) -> None:
        """Apply ``default_stage_max_velocity``/``default_stage_acceleration``
        to both sample stages. A no-op when both are ``None``."""
        cfg = self.config
        v, a = cfg.default_stage_max_velocity, cfg.default_stage_acceleration
        if v is None and a is None:
            return
        self.stage_x.set_velocity(max_velocity=v, acceleration=a)
        self.stage_y.set_velocity(max_velocity=v, acceleration=a)

    # ------------------------------------------------------------------ #
    # Piezo phase-shift primitives                                        #
    # ------------------------------------------------------------------ #

    def _piezo_step_size(self, direction: int) -> int:
        cfg = self.config
        return cfg.piezo_step_size_fwd if direction >= 0 else -cfg.piezo_step_size_bwd

    def step_piezo(self, direction: int = 1) -> None:
        """Move the piezo one phase-shift step in *direction* (``+1``/``-1``),
        then pause ``piezo_settle_s`` for mechanical settling."""
        self.piezo.move_by(self._piezo_step_size(direction))
        time.sleep(self.config.piezo_settle_s)

    def walk_piezo(self, num_moves: int, direction: int = 1) -> None:
        """Move the piezo *num_moves* steps in *direction*, no capture --
        pure repositioning (e.g. the return sweep)."""
        for _ in range(num_moves):
            self.step_piezo(direction)

    def dry_run(self, direction: int = 1) -> None:
        """Backlash-compensation dry run: ``piezo_dry_num`` moves of
        ``piezo_dry_step`` in *direction*, each followed by the settle pause.
        Run this before every direction change -- the slip-stick actuator
        needs to take up backlash before its commanded steps are
        reproducible."""
        cfg = self.config
        step = cfg.piezo_dry_step if direction >= 0 else -cfg.piezo_dry_step
        for _ in range(cfg.piezo_dry_num):
            self.piezo.move_by(step)
            time.sleep(cfg.piezo_settle_s)

    def capture(self) -> np.ndarray:
        """Capture one averaged, cropped frame using the config's frame
        averaging/drop/delay and ROI settings."""
        cfg = self.config
        img = self.camera.get_image(
            num_frames_to_average=cfg.num_frames_to_average,
            num_frames_to_drop=cfg.num_frames_to_drop,
            delay=cfg.capture_delay,
        )
        return crop(img, cfg.roi_x0, cfg.roi_y0, cfg.roi_x1, cfg.roi_y1)

    def acquire_stack(self, num_steps: int | None = None, direction: int = 1) -> np.ndarray:
        """Capture one phase-shift sweep: alternate :meth:`capture` and
        :meth:`step_piezo`; *num_steps* frames and *num_steps - 1* moves (no
        move after the last frame). Defaults to ``config.num_piezo_steps``.

        The caller is responsible for a leading :meth:`dry_run` -- this
        method covers one sweep's worth of frames only, so it can serve both
        a single measurement and one repeat inside a longer forward walk
        across several calls.
        """
        cfg = self.config
        n = num_steps if num_steps is not None else cfg.num_piezo_steps
        images = []
        for step in range(n):
            images.append(self.capture())
            if step < n - 1:
                self.step_piezo(direction)
        return np.asarray(images)

    def return_sweep(self, num_moves: int | None = None) -> None:
        """Walk the piezo back *num_moves* steps (no capture). Defaults to
        ``config.num_piezo_return_steps``. As with :meth:`acquire_stack`,
        the caller is responsible for a leading ``dry_run(-1)``."""
        cfg = self.config
        n = num_moves if num_moves is not None else cfg.num_piezo_return_steps
        self.walk_piezo(n, -1)

    @contextlib.contextmanager
    def stage_reference(self):
        """Move the sample stages to the reference position for the
        duration of the block, then restore their starting positions --
        even if the block raises, so a failure never strands the stages at
        the reference offset."""
        cfg = self.config
        init_x = self.stage_x.get_position()
        init_y = self.stage_y.get_position()
        try:
            self.stage_x.move_by(cfg.reference_x_by)
            self.stage_y.move_by(cfg.reference_y_by)
            yield
        finally:
            self.stage_x.move_to(init_x)
            self.stage_y.move_to(init_y)

    # ------------------------------------------------------------------ #
    # Saving                                                               #
    # ------------------------------------------------------------------ #

    def save_npz(self, prefix: str, **arrays) -> Path:
        """Save *arrays* to a timestamped NPZ file under ``config.save_dir``."""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = save_dir / f'{prefix}_{timestamp}.npz'
        np.savez(save_path, **arrays)
        return save_path
