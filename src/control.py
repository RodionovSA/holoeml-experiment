from datetime import datetime
from pathlib import Path

import numpy as np

from pylablib.devices import Thorlabs
from src.pythorcam.thorcam import ThorlabsCamera, create_camera_sdk, TLCameraSDK
from src.monochromator.mono import MonochromatorControl
from src.filterwheel import FilterWheelControl
from src.config import Config, ExposureSettings
from src.pythorcam.utils import brightness_calibration as _brightness_calibration

class Control:
    def __init__(self,
                 camerasdk: TLCameraSDK,
                 camera: ThorlabsCamera,
                 monochromator: MonochromatorControl,
                 focus: Thorlabs.KinesisMotor,
                 filterwheel: FilterWheelControl,
                 config: Config):

        self._camerasdk = camerasdk
        self.camera = camera
        self.mono = monochromator
        self.focus = focus
        self.filterwheel = filterwheel
        self.config = config
        self._closed = False

        self._exposure_settings_path = Path(config.exposure_settings_path)
        if not self._exposure_settings_path.exists():
            ExposureSettings().save(self._exposure_settings_path)
        self.exposure_settings = ExposureSettings.load(self._exposure_settings_path)

    @classmethod
    def from_config(cls, config: Config):
        camerasdk = create_camera_sdk()
        camera = ThorlabsCamera(camerasdk, config.camera_serial)
        monochromator = None
        filterwheel = None
        focus = None
        try:
            monochromator = MonochromatorControl(port=config.monochromator_port)
            monochromator.initialize_arduino()
            monochromator.home_motor()
            monochromator.select_grating_mode(mode=config.monochromator_grating_mode)

            filterwheel = FilterWheelControl(config.filterwheel_address)

            focus = Thorlabs.KinesisMotor(config.focus_serial, scale='stage')

            return cls(camerasdk, camera, monochromator, focus, filterwheel, config)
        except Exception:
            if focus is not None:
                focus.close()
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

    def load_exposure_settings(self) -> ExposureSettings:
        self.exposure_settings = ExposureSettings.load(self._exposure_settings_path)
        return self.exposure_settings

    def save_exposure_settings(self) -> None:
        self.exposure_settings.save(self._exposure_settings_path)

    def brightness_calibration(self, override: bool = False) -> None:
        """Run per-wavelength brightness calibration and store results.

        All calibration parameters are read from :attr:`config`.  Iterates over
        every wavelength in the config sweep, moves the monochromator, and calls
        :func:`brightness_calibration` from ``utils``.  The exposure and gain
        found at wavelength N seed the next wavelength to speed up convergence.

        Returns immediately if the settings file already contains entries for
        all config wavelengths, unless *override* is ``True``.
        """
        cfg = self.config
        wavelengths = np.linspace(cfg.wvl_start, cfg.wvl_stop, cfg.wvl_num)

        if not override:
            stored = self.exposure_settings.wavelengths
            if (len(stored) == len(wavelengths)
                    and np.allclose(stored, wavelengths, atol=0.01)):
                return

        exp_us = cfg.calib_initial_exposure_ms * 1000
        gain = cfg.calib_initial_gain
        calibrated_wavelengths: list[float] = []
        calibrated_exposures: list[int] = []
        calibrated_gains: list[int] = []

        for wvl in wavelengths:
            self.mono.set_wavelength(wvl)
            exp_us, gain = _brightness_calibration(
                camera=self.camera,
                initial_exposure_time=exp_us,
                initial_gain=gain,
                target_brightness=cfg.calib_target_brightness,
                tolerance=cfg.calib_tolerance,
                increment=cfg.calib_increment,
                max_number_of_steps=cfg.calib_max_steps,
                max_exposure_time=cfg.calib_max_exposure_ms * 1000,
                priority=cfg.calib_priority,
                num_frames_to_average=cfg.calib_num_frames_to_average,
                num_frames_to_drop=cfg.calib_num_frames_to_drop,
                delay=cfg.calib_delay,
            )
            calibrated_wavelengths.append(float(wvl))
            calibrated_exposures.append(exp_us // 1000)
            calibrated_gains.append(gain)

        self.exposure_settings = ExposureSettings(
            wavelengths=calibrated_wavelengths,
            exposure_ms=calibrated_exposures,
            gain=calibrated_gains,
        )
        self.save_exposure_settings()

    def set_camera_settings(self,
                            exposure_time_ms: int,
                            gain: int,
                            black_level: int | None = None,
                            bit_depth=None,
                            out_bit_depth=None) -> None:
        """Exposure time in ms (converted to µs for the camera).

        ``black_level``, ``bit_depth``, and ``out_bit_depth`` default to the
        values in :attr:`config` when not supplied.
        """
        cfg = self.config
        self.camera.set_settings(
            exposure_time_us=exposure_time_ms * 1000,
            gain=gain,
            black_level=black_level if black_level is not None else cfg.camera_black_level,
            bit_depth=bit_depth if bit_depth is not None else getattr(np, cfg.camera_bit_depth),
            out_bit_depth=out_bit_depth if out_bit_depth is not None else getattr(np, cfg.camera_out_bit_depth),
        )

    def reference_measurement(self) -> Path:
        """Capture one reference image per sweep wavelength and save as NPZ.

        Moves the monochromator to each wavelength, applies the per-wavelength
        exposure settings from :attr:`exposure_settings`, positions the filter
        wheel (long-pass above ``filter_wvl``, open/empty below), captures an
        image, and saves the full stack to ``save_dir``.

        Returns
        -------
        Path
            Path to the saved NPZ file.

        Raises
        ------
        RuntimeError
            If :attr:`exposure_settings` does not match the config wavelengths.
        """
        cfg = self.config
        wavelengths = np.linspace(cfg.wvl_start, cfg.wvl_stop, cfg.wvl_num)

        stored = self.exposure_settings.wavelengths
        if len(stored) != len(wavelengths) or not np.allclose(stored, wavelengths, atol=0.01):
            raise RuntimeError(
                "Exposure settings do not match config wavelengths. "
                "Run brightness_calibration() first."
            )

        images = []
        for i, wvl in enumerate(wavelengths):
            self.mono.set_wavelength(wvl)

            fw_pos = cfg.longpass_pos if wvl >= cfg.filter_wvl else cfg.filterwheel_empty_pos
            if self.filterwheel.get_position() != fw_pos:
                self.filterwheel.set_position(fw_pos)

            self.set_camera_settings(
                exposure_time_ms=self.exposure_settings.exposure_ms[i],
                gain=self.exposure_settings.gain[i],
            )

            image = self.camera.get_image(
                num_frames_to_average=cfg.num_frames_to_average,
                num_frames_to_drop=cfg.num_frames_to_drop,
                delay=cfg.capture_delay,
            )
            images.append(image)

        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = save_dir / f'reference_{timestamp}.npz'
        np.savez(save_path, wavelengths=wavelengths, images=np.stack(images))
        return save_path
        
    



    