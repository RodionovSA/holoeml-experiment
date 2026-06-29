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

        self._exposure_settings_path = Path(config.exposure_settings_path)
        if not self._exposure_settings_path.exists():
            ExposureSettings().save(self._exposure_settings_path)
        self.exposure_settings = ExposureSettings.load(self._exposure_settings_path)

        self._exposure_settings_xpol_path = Path(config.exposure_settings_xpol_path)
        if not self._exposure_settings_xpol_path.exists():
            ExposureSettings().save(self._exposure_settings_xpol_path)
        self.exposure_settings_x = ExposureSettings.load(self._exposure_settings_xpol_path)

        self._exposure_settings_ypol_path = Path(config.exposure_settings_ypol_path)
        if not self._exposure_settings_ypol_path.exists():
            ExposureSettings().save(self._exposure_settings_ypol_path)
        self.exposure_settings_y = ExposureSettings.load(self._exposure_settings_ypol_path)

        self._focus_settings_path = Path(config.focus_settings_path)
        if not self._focus_settings_path.exists():
            FocusSettings().save(self._focus_settings_path)
        self.focus_settings = FocusSettings.load(self._focus_settings_path)

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

    def _exposure_settings_for(self, xpol: bool | None) -> tuple[ExposureSettings, Path]:
        if xpol is True:
            return self.exposure_settings_x, self._exposure_settings_xpol_path
        if xpol is False:
            return self.exposure_settings_y, self._exposure_settings_ypol_path
        return self.exposure_settings, self._exposure_settings_path

    def load_exposure_settings(self, xpol: bool | None = None) -> ExposureSettings:
        if xpol is True:
            self.exposure_settings_x = ExposureSettings.load(self._exposure_settings_xpol_path)
            return self.exposure_settings_x
        if xpol is False:
            self.exposure_settings_y = ExposureSettings.load(self._exposure_settings_ypol_path)
            return self.exposure_settings_y
        self.exposure_settings = ExposureSettings.load(self._exposure_settings_path)
        return self.exposure_settings

    def save_exposure_settings(self, xpol: bool | None = None) -> None:
        settings, path = self._exposure_settings_for(xpol)
        settings.save(path)

    def load_focus_settings(self) -> FocusSettings:
        self.focus_settings = FocusSettings.load(self._focus_settings_path)
        return self.focus_settings

    def save_focus_settings(self) -> None:
        self.focus_settings.save(self._focus_settings_path)

    def brightness_calibration(self,
                               override: bool = False,
                               use_current_as_initial: bool = False,
                               xpol: bool|None = True) -> None:
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
        wavelengths = np.linspace(cfg.wvl_start, cfg.wvl_stop, cfg.wvl_num)
        
        if xpol is not None:
            angle = self.config.polarizer_x_position + (0 if xpol else 90)
            self.polarizer.move_to(angle)
            self.polarizer.wait_move()

        current_settings, settings_path = self._exposure_settings_for(xpol)

        if not override:
            stored = current_settings.wavelengths
            if (len(stored) == len(wavelengths)
                    and np.allclose(stored, wavelengths, atol=0.01)):
                return

        # Build per-wavelength seeds if using current stored values.
        stored = current_settings
        if (use_current_as_initial
                and len(stored.wavelengths) == len(wavelengths)
                and np.allclose(stored.wavelengths, wavelengths, atol=0.01)):
            init_exp_us = [int(e * 1000) for e in stored.exposure_ms]
            init_gain = list(stored.gain)
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

        self.camera.arm()
        time.sleep(0.1)
        try:
            for i, wvl in enumerate(wavelengths):
                self.mono.set_wavelength(wvl)
                fw_pos = cfg.longpass_pos if wvl >= cfg.filter_wvl else cfg.filterwheel_empty_pos
                self.filterwheel.set_position(fw_pos)

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
        finally:
            self.camera.disarm()

        new_settings = ExposureSettings(
            wavelengths=calibrated_wavelengths,
            exposure_ms=calibrated_exposures,
            gain=calibrated_gains,
            best_brightness=calibrated_brightnesses,
        )
        if xpol is True:
            self.exposure_settings_x = new_settings
        elif xpol is False:
            self.exposure_settings_y = new_settings
        else:
            self.exposure_settings = new_settings
        new_settings.save(settings_path)

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
        wavelengths = np.linspace(cfg.wvl_start, cfg.wvl_stop, cfg.wvl_num)

        if xpol is not None:
            angle = cfg.polarizer_x_position + (0 if xpol else 90)
            self.polarizer.move_to(angle)
            self.polarizer.wait_move()

        es, _ = self._exposure_settings_for(xpol)
        if len(es.wavelengths) != len(wavelengths) or not np.allclose(es.wavelengths, wavelengths, atol=0.01):
            raise RuntimeError(
                "Exposure settings do not match config wavelengths. "
                "Run brightness_calibration() first."
            )

        self.camera.arm()
        time.sleep(0.1)
        images = []
        try:
            for i, wvl in enumerate(wavelengths):
                self.mono.set_wavelength(wvl)

                fw_pos = cfg.longpass_pos if wvl >= cfg.filter_wvl else cfg.filterwheel_empty_pos
                if self.filterwheel.get_position() != fw_pos:
                    self.filterwheel.set_position(fw_pos)

                self.camera.set_exposure_time_us(int(es.exposure_ms[i] * 1000))
                self.camera.set_gain(es.gain[i])

                image = self.camera.get_image(
                    num_frames_to_average=cfg.num_frames_to_average,
                    num_frames_to_drop=cfg.num_frames_to_drop,
                    delay=cfg.capture_delay,
                )
                images.append(image)
        finally:
            self.camera.disarm()

        pol_label = {True: 'xpol_', False: 'ypol_', None: ''}[xpol]
        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = save_dir / f'reference_{pol_label}{timestamp}.npz'
        np.savez(save_path, wavelengths=wavelengths, images=np.stack(images))
        return save_path

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
        wavelengths = np.linspace(cfg.wvl_start, cfg.wvl_stop, cfg.wvl_num)

        es, _ = self._exposure_settings_for(xpol)
        if len(es.wavelengths) != len(wavelengths) or not np.allclose(es.wavelengths, wavelengths, atol=0.01):
            raise RuntimeError(
                "Exposure settings do not match config wavelengths. "
                "Run brightness_calibration() first."
            )

        pol_label = {True: 'xpol_', False: 'ypol_', None: ''}[xpol]
        self.filterwheel.set_position(cfg.black_pos)
        self.camera.arm()
        time.sleep(0.1)
        images = []
        try:
            for i in range(len(wavelengths)):
                self.camera.set_exposure_time_us(int(es.exposure_ms[i] * 1000))
                self.camera.set_gain(es.gain[i])
                image = self.camera.get_image(
                    num_frames_to_average=cfg.num_frames_to_average,
                    num_frames_to_drop=cfg.num_frames_to_drop,
                    delay=cfg.capture_delay,
                )
                images.append(image)
        finally:
            self.camera.disarm()

        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = save_dir / f'black_{pol_label}{timestamp}.npz'
        np.savez(save_path, wavelengths=wavelengths, images=np.stack(images))
        return save_path

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
        wavelengths = np.linspace(cfg.wvl_start, cfg.wvl_stop, cfg.wvl_num)

        if xpol is not None:
            angle = cfg.polarizer_x_position + (0 if xpol else 90)
            self.polarizer.move_to(angle)
            self.polarizer.wait_move()

        es, _ = self._exposure_settings_for(xpol)

        for settings, name in (
            (es.wavelengths, "Exposure"),
            (self.focus_settings.wavelengths, "Focus"),
        ):
            if len(settings) != len(wavelengths) or not np.allclose(settings, wavelengths, atol=0.01):
                raise RuntimeError(
                    f"{name} settings do not match config wavelengths. "
                    "Run calibration first."
                )

        if cfg.default_focus_position is None:
            raise RuntimeError(
                "default_focus_position must be set in config to use focus_settings offsets."
            )

        self.camera.arm()
        time.sleep(0.1)
        images = []
        try:
            for i, wvl in enumerate(wavelengths):
                self.mono.set_wavelength(wvl)

                fw_pos = cfg.longpass_pos if wvl >= cfg.filter_wvl else cfg.filterwheel_empty_pos
                if self.filterwheel.get_position() != fw_pos:
                    self.filterwheel.set_position(fw_pos)

                target_m = (cfg.default_focus_position + self.focus_settings.offsets[i]) * 1e-3
                if abs(self.focus.get_position() - target_m) > 5e-6:
                    self.focus.move_to(target_m)
                    self.focus.wait_move()

                self.camera.set_exposure_time_us(int(es.exposure_ms[i] * 1000))
                self.camera.set_gain(es.gain[i])
                image = self.camera.get_image(
                    num_frames_to_average=cfg.num_frames_to_average,
                    num_frames_to_drop=cfg.num_frames_to_drop,
                    delay=cfg.capture_delay,
                )
                images.append(image)
        finally:
            self.camera.disarm()

        pol_label = {True: 'xpol_', False: 'ypol_', None: ''}[xpol]
        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = save_dir / f'sample_{pol_label}{timestamp}.npz'
        np.savez(save_path, wavelengths=wavelengths, images=np.stack(images))
        return save_path
