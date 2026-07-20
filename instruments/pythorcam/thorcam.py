import threading
import time
from collections import deque

import numpy as np
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

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

class ThorlabsCamera:
    """Thorlabs CS126 / LP126 camera wrapper.

    Construction opens the device. Use as a context manager or call
    `close()` explicitly to release the handle.

    Example
    -------
    >>> sdk = create_camera_sdk()
    >>> with ThorlabsCamera(sdk, '35595') as cam:
    ...     cam.set_settings(exposure_time_us=10000, gain=0, black_level=5)
    ...     cam.arm()
    ...     img = cam.get_image(num_frames_to_average=5)
    ...     cam.disarm()
    """

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
        self.sdk = sdk
        self.serial_number = serial_number
        self.cam_type = cam_type
        self.bit_depth = np.uint16
        self.out_bit_depth = np.float32
        self.pixel_max_value: int = 4095  # CS126 is a 12-bit sensor stored in uint16

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def image_height(self) -> int:
        """Frame height in pixels."""
        return self._camera.image_height_pixels

    @property
    def image_width(self) -> int:
        """Frame width in pixels."""
        return self._camera.image_width_pixels

    @property
    def image_shape(self) -> tuple[int, int]:
        """(height, width) in pixels."""
        return (self._camera.image_height_pixels, self._camera.image_width_pixels)

    def set_settings(self,
                     exposure_time_us: int,
                     gain: int,
                     black_level: int,
                     bit_depth=np.uint16,
                     out_bit_depth=np.float32,
                     frame_rate_control_value: float | None = None) -> None:
        """Configure capture parameters.

        Parameters
        ----------
        exposure_time_us : int
            Exposure time in microseconds (µs).
        gain : int
            Sensor gain in SDK-defined units (see Thorlabs TSI SDK docs;
            typically tenths of a dB — use `convert_gain_to_decibels()` to
            translate).
        black_level : int
            Black-level offset in raw ADC counts.
        bit_depth : numpy dtype
            dtype of the raw frame buffer (e.g. np.uint8, np.uint16).
        out_bit_depth : numpy dtype
            dtype of the averaged image returned by `get_image()`
            (e.g. np.float32, np.float64).
        frame_rate_control_value : float, optional
            Target frame rate in frames per second. When provided, frame-rate
            control is enabled; when None, it is left untouched.
        """
        self._camera.exposure_time_us = exposure_time_us
        self._camera.gain = gain
        self._camera.black_level = black_level
        self.bit_depth = bit_depth
        self.out_bit_depth = out_bit_depth
        if frame_rate_control_value is not None:
            self._camera.frame_rate_control_value = frame_rate_control_value
            self._camera.is_frame_rate_control_enabled = True

    def set_exposure_time_us(self, exposure_time: int) -> None:
        """Set exposure time in microseconds (µs)."""
        self._camera.exposure_time_us = exposure_time

    def set_gain(self, gain: int) -> None:
        """Set sensor gain in SDK-defined units (see Thorlabs TSI SDK docs)."""
        self._camera.gain = gain

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

    def get_image(self,
                  num_frames_to_average: int = 1,
                  num_frames_to_drop: int = 0,
                  delay: float = 0) -> np.ndarray:
        """Drop `num_frames_to_drop` frames, then capture and average `num_frames_to_average` frames.

        Parameters
        ----------
        num_frames_to_average : int
            Number of frames to mean together. Output is the per-pixel mean.
        num_frames_to_drop : int
            Number of frames to discard before averaging (lets the sensor settle
            after exposure / gain changes).
        delay : float
            Seconds to sleep between dropped frames.

        Returns
        -------
        np.ndarray
            Shape (H, W, 1) for monochrome or (H, W, 3) for color, dtype `out_bit_depth`.
        """
        for _ in range(num_frames_to_drop):
            self._get_single_frame()
            if delay:
                time.sleep(delay)

        H, W = self.image_shape
        C = 1 if self.cam_type == 'MU' else 3
        accumulator = np.zeros((H, W, C), dtype=np.float64)
        for _ in range(num_frames_to_average):
            accumulator += self._get_single_frame()

        return (accumulator / num_frames_to_average).astype(self.out_bit_depth)

class CameraStream:
    """Runs ThorlabsCamera capture on a background daemon thread.

    Frames are stored in a fixed-size ring buffer. The main thread can
    call get_latest_frame() at any time without blocking on hardware.

    Example
    -------
    >>> with ThorlabsCamera(sdk, '35595') as cam:
    ...     cam.set_settings(exposure_time_us=10000, gain=0, black_level=5)
    ...     cam.arm()
    ...     stream = CameraStream(cam)
    ...     stream.start()
    ...     try:
    ...         frame = stream.get_latest_frame()       # non-blocking, most recent
    ...         avg = stream.get_averaged_frame(n=5)    # blocks until 5 frames are ready
    ...     finally:
    ...         stream.stop()
    ...         cam.disarm()
    """

    def __init__(self, camera: ThorlabsCamera, buffer_size: int = 30):
        self._camera = camera
        self._buffer: deque = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._new_frame = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        """Start the background capture thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the capture thread to exit and wait for it to finish."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def _loop(self):
        while not self._stop_event.is_set():
            frame = self._camera._get_single_frame()
            with self._lock:
                self._buffer.append(frame)
            self._new_frame.set()
            self._new_frame.clear()

    def get_latest_frame(self) -> np.ndarray | None:
        """Return a copy of the most recently captured frame, or None if buffer is empty."""
        with self._lock:
            return np.copy(self._buffer[-1]) if self._buffer else None

    def get_averaged_frame(self, n: int, timeout: float = 10.0) -> np.ndarray | None:
        """Return the mean of the last n frames. Waits until n frames are available."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                frames = list(self._buffer)
            if len(frames) >= n:
                return np.mean(frames[-n:], axis=0, dtype=np.float32)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self._new_frame.wait(timeout=min(remaining, 0.1))


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
