import time

import numpy as np
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

from instruments.camera.base import Camera, CameraStream  # noqa: F401 -- re-exported for backward compat (e.g. scripts/focus.py)

def create_camera_sdk():
    try:
        # if on Windows, use the provided setup script to add the DLLs folder to the PATH
        try:
            from .windows_setup import configure_path  # when imported as a package (e.g. from a notebook)
        except ImportError:
            from windows_setup import configure_path   # when run directly as a script
        configure_path()
    except ImportError:
        pass

    sdk = TLCameraSDK()

    return sdk

class ThorlabsCamera(Camera):
    """Thorlabs CS126 / LP126 camera wrapper.

    Construction opens the device. Use as a context manager or call
    `close()` explicitly to release the handle.

    Implements the vendor-neutral `instruments.camera.Camera` interface;
    `get_image()`, `CameraStream`, and the `instruments.camera.utils`
    routines (autofocus, brightness_calibration, live_view, ...) all work
    against this class without knowing it's a Thorlabs camera.

    Example
    -------
    >>> sdk = create_camera_sdk()
    >>> with ThorlabsCamera(sdk, '35595') as cam:
    ...     cam.set_settings(exposure_time_us=10000, gain=0, black_level=5)
    ...     cam.arm()
    ...     img = cam.get_image(num_frames_to_average=5)
    ...     cam.disarm()
    """

    # CS126 hardware limits.
    _MIN_EXPOSURE_MS = 0.028
    _MAX_EXPOSURE_MS = 14700.924
    _MIN_GAIN = 0
    _MAX_GAIN = 480

    def __init__(self,
                 sdk: TLCameraSDK,
                 serial_number: str,
                 cam_type: str = 'MU'):
        """Connect to a Thorlabs CS126/LP126 camera by serial number.

        Parameters
        ----------
        sdk : TLCameraSDK
            Initialized SDK handle from `create_camera_sdk()`.
        serial_number : str
            Camera serial number (must match a device returned by
            `sdk.discover_available_cameras()`).
        cam_type : {'MU', 'CU'}
            'MU' for monochrome, anything else treated as color.

        Raises
        ------
        RuntimeError
            If no cameras are detected or the requested serial is not available.
        """
        super().__init__()
        self.sdk = sdk
        self.serial_number = serial_number
        self.cam_type = cam_type
        self.bit_depth = np.uint16

        available = sdk.discover_available_cameras()
        if not available:
            raise RuntimeError("No Thorlabs cameras detected")
        if serial_number not in available:
            raise RuntimeError(
                f"Camera {serial_number} not found. Available: {available}"
            )
        self._camera = sdk.open_camera(serial_number)
        print(f"Camera {serial_number} is connected")

    def close(self) -> None:
        """Disarm (if armed) and dispose of the camera handle."""
        if self._camera.is_armed:
            self._camera.disarm()
        self._camera.dispose()
        print(f"Camera {self.serial_number} is disconnected")

    @property
    def image_shape(self) -> tuple[int, int]:
        """(height, width) in pixels."""
        return (self._camera.image_height_pixels, self._camera.image_width_pixels)

    @property
    def num_channels(self) -> int:
        return 1 if self.cam_type == 'MU' else 3

    @property
    def pixel_max_value(self) -> int:
        return 4095  # CS126 is a 12-bit sensor stored in uint16

    @property
    def min_exposure_ms(self) -> float:
        return self._MIN_EXPOSURE_MS

    @property
    def max_exposure_ms(self) -> float:
        return self._MAX_EXPOSURE_MS

    @property
    def min_gain(self) -> int:
        return self._MIN_GAIN

    @property
    def max_gain(self) -> int:
        return self._MAX_GAIN

    def set_exposure_ms(self, exposure_ms: float) -> None:
        """Set exposure time in milliseconds."""
        self._camera.exposure_time_us = int(exposure_ms * 1000)

    def _set_gain(self, gain: int) -> None:
        """Set sensor gain in SDK-defined units (see Thorlabs TSI SDK docs;
        typically tenths of a dB -- use `convert_gain_to_decibels()` to translate)."""
        self._camera.gain = gain

    def _set_black_level(self, black_level: int) -> None:
        """Set black-level offset in raw ADC counts."""
        self._camera.black_level = black_level

    def set_raw_bit_depth(self, dtype) -> None:
        self.bit_depth = dtype

    def _set_frame_rate_control(self, fps: float) -> None:
        self._camera.frame_rate_control_value = fps
        self._camera.is_frame_rate_control_enabled = True

    def arm(self) -> None:
        """Arm the camera in mode 2 (software-triggered, 2-frame buffer) and issue a software trigger."""
        self._camera.arm(2)
        self._camera.issue_software_trigger()

    def disarm(self) -> None:
        """Stop frame acquisition."""
        self._camera.disarm()

    def _get_single_frame(self, timeout: float = 5.0) -> np.ndarray:
        """Block until one frame arrives, then return it as (H, W, C) of `self.bit_depth`.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait for a frame before raising TimeoutError.

        Raises
        ------
        TimeoutError
            If no frame is received within `timeout` seconds.
        """
        deadline = time.monotonic() + timeout
        frame = None
        while frame is None:
            frame = self._camera.get_pending_frame_or_null()
            if frame is None:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"No frame received from camera {self.serial_number} "
                        f"within {timeout:.1f}s"
                    )
                time.sleep(0.0005)
        image = np.array(frame.image_buffer, dtype=self.bit_depth).reshape(self.image_shape)
        if self.cam_type == 'MU':
            return image[:, :, np.newaxis]
        return np.stack([image, image, image], axis=-1)


#Test
if __name__=="__main__":
    # Keep in sync with instruments/config/config.yaml's camera_serial by hand
    # (this demo block predates instruments/config and isn't wired to it).
    serial_number = '35596'
    camerasdk = create_camera_sdk()
    with ThorlabsCamera(camerasdk, serial_number) as camera:
        camera.set_settings(exposure_time_us=10000, gain=0, black_level=5, bit_depth=np.uint16)

        camera.arm()
        time.sleep(0.1)
        image = camera.get_image(num_frames_to_average=5)

        #live_view(camera, zoom_value=0.4)

        camera.disarm()

    np.save('test_image.npy', image)
