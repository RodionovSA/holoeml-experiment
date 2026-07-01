import bisect
import contextlib
from datetime import datetime
from pathlib import Path

import numpy as np
import time

from pylablib.devices import Thorlabs
from src.pythorcam.thorcam import ThorlabsCamera, create_camera_sdk
from src.monochromator.mono import MonochromatorControl
from src.filterwheel import FilterWheelControl
from src.config import Config, ExposureSettings, FocusSettings
from src.pythorcam.utils import brightness_calibration as _brightness_calibration
from src.pythorcam.utils import autofocus as _autofocus


class Control:
    def __init__(self,
                 camera: ThorlabsCamera,
                 monochromator: MonochromatorControl,
                 focus: Thorlabs.KinesisMotor,
                 polarizer: Thorlabs.KinesisMotor,
                 filterwheel: FilterWheelControl,
                 config: Config):

        self.camera = camera
        self.mono = monochromator
        self.focus = focus
        self.polarizer = polarizer
        self.filterwheel = filterwheel
        self.config = config
        self._closed = False

        self._exposure_settings_path, self.exposure_settings = \
            self._init_settings_file(config.exposure_settings_path, ExposureSettings)
        self._exposure_settings_xpol_path, self.exposure_settings_x = \
            self._init_settings_file(config.exposure_settings_xpol_path, ExposureSettings)
        self._exposure_settings_ypol_path, self.exposure_settings_y = \
            self._init_settings_file(config.exposure_settings_ypol_path, ExposureSettings)
        self._focus_settings_path, self.focus_settings = \
            self._init_settings_file(config.focus_settings_path, FocusSettings)

    @staticmethod
    def _init_settings_file(path_str, cls):
        path = Path(path_str)
        if not path.exists():
            cls().save(path)
        return path, cls.load(path)

    @classmethod
    def from_config(cls, config: Config):
        camerasdk = create_camera_sdk()
        camera = ThorlabsCamera(camerasdk, config.camera_serial)
        camera.set_settings(
            exposure_time_us=config.calib_initial_exposure_ms * 1000,
            gain=config.calib_initial_gain,
            black_level=config.camera_black_level,
            bit_depth=getattr(np, config.camera_bit_depth),
            out_bit_depth=getattr(np, config.camera_out_bit_depth),
        )
        monochromator = None
        filterwheel = None
        focus = None
        polarizer = None
        try:
            monochromator = MonochromatorControl(port=config.monochromator_port)
            monochromator.initialize_arduino()
            monochromator.home_motor()
            monochromator.select_grating_mode(mode=config.monochromator_grating_mode)

            filterwheel = FilterWheelControl(config.filterwheel_address)

            focus = Thorlabs.KinesisMotor(config.focus_serial, scale='stage')

            polarizer = Thorlabs.KinesisMotor(config.polarizer_serial, scale="K10CR1")

            ctrl = cls(camera, monochromator, focus, polarizer, filterwheel, config)
            ctrl.go_to_default_state()
            return ctrl
        except Exception:
            if focus is not None:
                focus.close()
            if polarizer is not None:
                polarizer.close()
            if filterwheel is not None:
                filterwheel.close()
            if monochromator is not None:
                monochromator.disconnect()
            camera.close()
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.camera.close()
        self.mono.disconnect()
        self.focus.close()
        self.polarizer.close()
        self.filterwheel.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Settings helpers                                                     #
    # ------------------------------------------------------------------ #

    def _exposure_settings_for(self, xpol: bool | None) -> tuple[ExposureSettings, Path]:
        if xpol is True:
            return self.exposure_settings_x, self._exposure_settings_xpol_path
        if xpol is False:
            return self.exposure_settings_y, self._exposure_settings_ypol_path
        return self.exposure_settings, self._exposure_settings_path

    def load_exposure_settings(self, xpol: bool | None = None) -> ExposureSettings:
        _, path = self._exposure_settings_for(xpol)
        settings = ExposureSettings.load(path)
        if xpol is True:
            self.exposure_settings_x = settings
        elif xpol is False:
            self.exposure_settings_y = settings
        else:
            self.exposure_settings = settings
        return settings

    def save_exposure_settings(self, xpol: bool | None = None) -> None:
        settings, path = self._exposure_settings_for(xpol)
        settings.save(path)

    def load_focus_settings(self) -> FocusSettings:
        self.focus_settings = FocusSettings.load(self._focus_settings_path)
        return self.focus_settings

    def save_focus_settings(self) -> None:
        self.focus_settings.save(self._focus_settings_path)

    # ------------------------------------------------------------------ #
    # Measurement primitives                                               #
    # ------------------------------------------------------------------ #

    def _sweep_wavelengths(self) -> np.ndarray:
        cfg = self.config
        return np.linspace(cfg.wvl_start, cfg.wvl_stop, cfg.wvl_num)

    def _move_polarizer(self, xpol: bool | None) -> None:
        if xpol is not None:
            angle = self.config.polarizer_x_position + (0 if xpol else 90)
            self.polarizer.move_to(angle)
            self.polarizer.wait_move()

    def _set_filterwheel_for_wvl(self, wvl: float) -> None:
        cfg = self.config
        fw_pos = cfg.longpass_pos if wvl >= cfg.filter_wvl else cfg.filterwheel_empty_pos
        if self.filterwheel.get_position() != fw_pos:
            self.filterwheel.set_position(fw_pos)

    def _apply_camera_exposure(self, es: ExposureSettings, es_idx: list[int], i: int) -> None:
        self.camera.set_exposure_time_us(int(es.exposure_ms[es_idx[i]] * 1000))
        self.camera.set_gain(es.gain[es_idx[i]])

    @contextlib.contextmanager
    def _armed_camera(self):
        self.camera.arm()
        time.sleep(0.1)
        try:
            yield
        finally:
            self.camera.disarm()

    def _save_npz(self, prefix: str, xpol: bool | None,
                  wavelengths: np.ndarray, images: list) -> Path:
        pol_label = {True: 'xpol_', False: 'ypol_', None: ''}[xpol]
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = save_dir / f'{prefix}_{pol_label}{timestamp}.npz'
        np.savez(save_path, wavelengths=wavelengths, images=np.stack(images))
        return save_path

    # ------------------------------------------------------------------ #
    # Algorithm helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_wavelength_range_idx(self, wavelength_nm: float) -> int:
        """Return the focus-range index for *wavelength_nm*.

        Ranges are defined by ``config.autofocus_range_boundaries`` (ascending nm).
        Index 0 covers everything below the first boundary; index N covers
        everything at or above the last boundary.  The boundary wavelength itself
        belongs to the upper range (bisect_right semantics).
        """
        return bisect.bisect_right(self.config.autofocus_range_boundaries, wavelength_nm)

    def _run_autofocus(self, center_position_mm: float) -> float:
        """Run a focus scan centred on *center_position_mm* and return the best position in mm.

        The camera must already be armed.  Scan range and step size are taken
        from config.  Motor velocity is set to ``autofocus_velocity_mm_s`` for the
        scan; the caller is responsible for restoring sweep velocity afterwards.
        """
        cfg = self.config
        _mm = 1e-3
        start_m = (center_position_mm - cfg.autofocus_scan_half_range_mm) * _mm
        end_m   = (center_position_mm + cfg.autofocus_scan_half_range_mm) * _mm
        step_m  = cfg.autofocus_step_size_mm * _mm
        vel_m   = cfg.autofocus_velocity_mm_s * _mm
        best_pos_m, _curve, _images = _autofocus(
            camera_trans=self.camera,
            focus_motor=self.focus,
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

    def _map_wavelengths(self,
                         settings_wavelengths: list[float],
                         config_wavelengths: np.ndarray,
                         name: str,
                         atol: float = 0.01) -> list[int]:
        """Return indices into *settings_wavelengths* for each wavelength in *config_wavelengths*.

        Raises RuntimeError if any config wavelength has no match within *atol* nm.
        """
        settings_arr = np.array(settings_wavelengths)
        indices = []
        for wvl in config_wavelengths:
            diffs = np.abs(settings_arr - wvl)
            idx = int(np.argmin(diffs))
            if diffs[idx] > atol:
                raise RuntimeError(
                    f"{name} settings do not cover wavelength {wvl:.2f} nm. "
                    "Run calibration first."
                )
            indices.append(idx)
        return indices

    # ------------------------------------------------------------------ #
    # Public methods                                                       #
    # ------------------------------------------------------------------ #

    def go_to_default_state(self) -> None:
        """Move hardware to the canonical resting state.

        Sets the monochromator to ``config.default_wavelength``, the filter
        wheel to the empty position, the focus motor to
        ``config.default_focus_position`` (if set), and the camera to the
        initial calibration settings defined in config.
        """
        self.mono.set_wavelength(self.config.default_wavelength)
        self.filterwheel.set_position(self.config.filterwheel_empty_pos)
        _mm = 1e-3
        v = self.config.default_focus_max_velocity
        a = self.config.default_focus_acceleration
        if v is not None or a is not None:
            self.focus.setup_velocity(max_velocity=v * _mm, acceleration=a * _mm)
        if self.config.default_focus_position is not None:
            self.focus.move_to(self.config.default_focus_position * _mm)
            self.focus.wait_move()
        self.camera.set_settings(
            exposure_time_us=self.config.calib_initial_exposure_ms * 1000,
            gain=self.config.calib_initial_gain,
            black_level=self.config.camera_black_level,
            bit_depth=getattr(np, self.config.camera_bit_depth),
            out_bit_depth=getattr(np, self.config.camera_out_bit_depth),
        )
        self.polarizer.setup_limit_switch(hw_kind_cw='make_home', hw_kind_ccw='make_home')
        if self.config.default_polarizer_home_velocity is not None:
            self.polarizer.setup_homing(velocity=self.config.default_polarizer_home_velocity)
        if not self.polarizer.is_homed():
            self.polarizer.home(sync=True)
        v = self.config.default_polarizer_max_velocity
        a = self.config.default_polarizer_acceleration
        if v is not None or a is not None:
            self.polarizer.setup_velocity(max_velocity=v, acceleration=a)
        if self.config.polarizer_x_position is not None:
            self.polarizer.move_to(self.config.polarizer_x_position)
            self.polarizer.wait_move()

    def brightness_calibration(self,
                               override: bool = False,
                               use_current_as_initial: bool = False,
                               xpol: bool | None = True) -> None:
        """Run per-wavelength brightness calibration and store results.

        All calibration parameters are read from :attr:`config`.  Iterates over
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
        cfg = self.config
        wavelengths = self._sweep_wavelengths()
        self._move_polarizer(xpol)

        current_settings, settings_path = self._exposure_settings_for(xpol)

        if not override:
            try:
                self._map_wavelengths(current_settings.wavelengths, wavelengths, "Exposure")
                return  # all config wavelengths already calibrated
            except RuntimeError:
                pass

        # Build per-wavelength seeds if using current stored values.
        if use_current_as_initial:
            try:
                seed_idx = self._map_wavelengths(current_settings.wavelengths, wavelengths, "Exposure")
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

        with self._armed_camera():
            for i, wvl in enumerate(wavelengths):
                self.mono.set_wavelength(wvl)
                self._set_filterwheel_for_wvl(wvl)

                if init_exp_us is not None:
                    exp_us = init_exp_us[i]
                    gain = init_gain[i]

                exp_us, gain, brightness = _brightness_calibration(
                    camera=self.camera,
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

        # Merge calibrated results into the existing stored settings.
        stored_wvls = np.array(current_settings.wavelengths)
        merged = ExposureSettings(
            wavelengths=list(current_settings.wavelengths),
            exposure_ms=list(current_settings.exposure_ms),
            gain=list(current_settings.gain),
            best_brightness=list(current_settings.best_brightness),
        )
        for k, wvl in enumerate(calibrated_wavelengths):
            diffs = np.abs(stored_wvls - wvl) if len(stored_wvls) else np.array([float('inf')])
            idx = int(np.argmin(diffs))
            if len(stored_wvls) and diffs[idx] <= 0.01:
                merged.exposure_ms[idx] = calibrated_exposures[k]
                merged.gain[idx] = calibrated_gains[k]
                merged.best_brightness[idx] = calibrated_brightnesses[k]
            else:
                # New wavelength not previously in file — insert in sorted order.
                pos = int(np.searchsorted(stored_wvls, wvl))
                merged.wavelengths.insert(pos, float(wvl))
                merged.exposure_ms.insert(pos, calibrated_exposures[k])
                merged.gain.insert(pos, calibrated_gains[k])
                merged.best_brightness.insert(pos, calibrated_brightnesses[k])
                stored_wvls = np.insert(stored_wvls, pos, wvl)

        if xpol is True:
            self.exposure_settings_x = merged
        elif xpol is False:
            self.exposure_settings_y = merged
        else:
            self.exposure_settings = merged
        merged.save(settings_path)

    def reference_measurement(self, xpol: bool | None = None) -> Path:
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
        cfg = self.config
        wavelengths = self._sweep_wavelengths()
        self._move_polarizer(xpol)

        es, _ = self._exposure_settings_for(xpol)
        es_idx = self._map_wavelengths(es.wavelengths, wavelengths, "Exposure")

        images = []
        with self._armed_camera():
            for i, wvl in enumerate(wavelengths):
                self.mono.set_wavelength(wvl)
                self._set_filterwheel_for_wvl(wvl)
                self._apply_camera_exposure(es, es_idx, i)
                images.append(self.camera.get_image(
                    num_frames_to_average=cfg.num_frames_to_average,
                    num_frames_to_drop=cfg.num_frames_to_drop,
                    delay=cfg.capture_delay,
                ))

        return self._save_npz('reference', xpol, wavelengths, images)

    def black_measurement(self, xpol: bool | None = None) -> Path:
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
        cfg = self.config
        wavelengths = self._sweep_wavelengths()

        es, _ = self._exposure_settings_for(xpol)
        es_idx = self._map_wavelengths(es.wavelengths, wavelengths, "Exposure")

        self.filterwheel.set_position(cfg.black_pos)
        images = []
        with self._armed_camera():
            for i, wvl in enumerate(wavelengths):
                print(f"wavelength {wvl}")
                self._apply_camera_exposure(es, es_idx, i)
                images.append(self.camera.get_image(
                    num_frames_to_average=cfg.num_frames_to_average,
                    num_frames_to_drop=cfg.num_frames_to_drop,
                    delay=cfg.capture_delay,
                ))

        return self._save_npz('black', xpol, wavelengths, images)

    def sample_measurement(self, xpol: bool | None = None) -> Path:
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
        cfg = self.config
        wavelengths = self._sweep_wavelengths()
        self._move_polarizer(xpol)

        es, _ = self._exposure_settings_for(xpol)
        es_idx    = self._map_wavelengths(es.wavelengths, wavelengths, "Exposure")
        focus_idx = self._map_wavelengths(self.focus_settings.wavelengths, wavelengths, "Focus")

        if cfg.focus_use_current_position:
            focus_base_mm = self.focus.get_position() * 1e3
        else:
            if cfg.default_focus_position is None:
                raise RuntimeError(
                    "default_focus_position must be set in config to use focus_settings offsets."
                )
            focus_base_mm = cfg.default_focus_position

        images = []
        range_corrections: dict[int, float] = {}  # range_idx -> position correction (mm)
        with self._armed_camera():
            for i, wvl in enumerate(wavelengths):
                self.mono.set_wavelength(wvl)
                self._set_filterwheel_for_wvl(wvl)
                self._apply_camera_exposure(es, es_idx, i)

                stored_target_mm = focus_base_mm + self.focus_settings.offsets[focus_idx[i]]
                if cfg.autofocus_enabled:
                    range_idx = self._get_wavelength_range_idx(wvl)
                    if range_idx not in range_corrections:
                        # First wavelength in this range: move to stored target then autofocus.
                        self.focus.move_to(stored_target_mm * 1e-3)
                        self.focus.wait_move()
                        best_mm = self._run_autofocus(center_position_mm=stored_target_mm)
                        range_corrections[range_idx] = best_mm - stored_target_mm
                        print(f'[autofocus] range {range_idx}  wvl={wvl:.1f} nm  '
                              f'correction={range_corrections[range_idx]:+.4f} mm')
                        # Restore sweep velocity after the slow autofocus scan.
                        _v, _a = cfg.default_focus_max_velocity, cfg.default_focus_acceleration
                        self.focus.setup_velocity(
                            max_velocity=(_v * 1e-3 if _v is not None else None),
                            acceleration=(_a * 1e-3 if _a is not None else None),
                        )
                    target_m = (stored_target_mm + range_corrections[range_idx]) * 1e-3
                else:
                    target_m = stored_target_mm * 1e-3

                if abs(self.focus.get_position() - target_m) > 2e-6:
                    self.focus.move_to(target_m)
                    self.focus.wait_move()

                images.append(self.camera.get_image(
                    num_frames_to_average=cfg.num_frames_to_average,
                    num_frames_to_drop=cfg.num_frames_to_drop,
                    delay=cfg.capture_delay,
                ))

        if cfg.autofocus_enabled and range_corrections:
            for range_idx, correction_mm in range_corrections.items():
                for j, wvl_j in enumerate(wavelengths):
                    if self._get_wavelength_range_idx(wvl_j) == range_idx:
                        self.focus_settings.offsets[focus_idx[j]] += correction_mm
            self.save_focus_settings()

        return self._save_npz('sample', xpol, wavelengths, images)
