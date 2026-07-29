"""Vendor-neutral camera interface.

``Camera`` defines the operational surface every camera driver in this repo
implements (``ThorlabsCamera``, ``PcoCamera``, ...) so that
``instruments/camera/utils.py`` (autofocus, brightness calibration, live
view) and callers such as ``amplitude/devices.py`` can work with either
vendor interchangeably.

Frame convention: every camera returns images as ``(H, W, C)`` -- ``C=1``
for monochrome. Exposure is in **milliseconds** on the granular interface
below (``set_exposure_ms``); the two Thorlabs-era microsecond entry points
(``set_exposure_time_us``, ``set_settings``) survive as legacy shims because
some call sites outside this refactor's scope still use them positionally
(see ``scripts/phase/piezo_scan.py``).
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import deque

import numpy as np


class Camera(ABC):
    """Common interface for the driver classes under ``instruments/``.

    Subclasses implement device-specific connection, arm/disarm, and
    single-frame capture. ``get_image`` and the legacy shims below are
    shared and should not be overridden.
    """

    #: Whether this camera exposes a gain control. ``brightness_calibration``
    #: skips gain adjustment entirely when this is False (e.g. PcoCamera).
    supports_gain: bool = True
    #: Whether this camera exposes a black-level control.
    supports_black_level: bool = True

    def __init__(self) -> None:
        self.out_bit_depth = np.float32

    # ------------------------------------------------------------------ #
    # Abstract: implemented per vendor                                    #
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def image_shape(self) -> tuple[int, int]:
        """(height, width) in pixels."""

    @property
    @abstractmethod
    def num_channels(self) -> int:
        """Number of channels returned per frame (1 for monochrome)."""

    @property
    @abstractmethod
    def pixel_max_value(self) -> int:
        """Maximum raw pixel value (e.g. 4095 for a 12-bit sensor)."""

    @property
    @abstractmethod
    def min_exposure_ms(self) -> float:
        """Hardware exposure lower bound, in milliseconds."""

    @property
    @abstractmethod
    def max_exposure_ms(self) -> float:
        """Hardware exposure upper bound, in milliseconds."""

    @property
    @abstractmethod
    def min_gain(self) -> int:
        """Hardware gain lower bound. Meaningless when ``supports_gain`` is False."""

    @property
    @abstractmethod
    def max_gain(self) -> int:
        """Hardware gain upper bound. Meaningless when ``supports_gain`` is False."""

    @abstractmethod
    def set_exposure_ms(self, exposure_ms: float) -> None:
        """Set exposure time in milliseconds."""

    @abstractmethod
    def arm(self) -> None:
        """Prepare the camera to deliver frames via ``_get_single_frame``."""

    @abstractmethod
    def disarm(self) -> None:
        """Stop frame acquisition."""

    @abstractmethod
    def _get_single_frame(self, timeout: float = 5.0) -> np.ndarray:
        """Block until one frame arrives; return it as (H, W, C)."""

    @abstractmethod
    def close(self) -> None:
        """Release the device handle."""

    # ------------------------------------------------------------------ #
    # Concrete: shared across vendors                                     #
    # ------------------------------------------------------------------ #

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def image_height(self) -> int:
        """Frame height in pixels."""
        return self.image_shape[0]

    @property
    def image_width(self) -> int:
        """Frame width in pixels."""
        return self.image_shape[1]

    def set_gain(self, gain: int) -> None:
        """Set sensor gain. Raises NotImplementedError if ``supports_gain`` is False."""
        if not self.supports_gain:
            raise NotImplementedError(f"{type(self).__name__} has no gain control")
        self._set_gain(gain)

    def _set_gain(self, gain: int) -> None:
        raise NotImplementedError(f"{type(self).__name__} has no gain control")

    def set_black_level(self, black_level: int) -> None:
        """Set black-level offset. Raises NotImplementedError if ``supports_black_level`` is False."""
        if not self.supports_black_level:
            raise NotImplementedError(f"{type(self).__name__} has no black-level control")
        self._set_black_level(black_level)

    def _set_black_level(self, black_level: int) -> None:
        raise NotImplementedError(f"{type(self).__name__} has no black-level control")

    def set_out_bit_depth(self, dtype) -> None:
        """dtype `get_image()` casts its final averaged output to."""
        self.out_bit_depth = dtype

    def set_raw_bit_depth(self, dtype) -> None:
        """dtype of the raw per-frame buffer, for vendors that need one to
        decode frames (see ``ThorlabsCamera``). No-op otherwise."""
        pass

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
            Shape (H, W, C), dtype `out_bit_depth`.
        """
        for _ in range(num_frames_to_drop):
            self._get_single_frame()
            if delay:
                time.sleep(delay)

        H, W = self.image_shape
        C = self.num_channels
        accumulator = np.zeros((H, W, C), dtype=np.float64)
        for _ in range(num_frames_to_average):
            accumulator += self._get_single_frame()

        return (accumulator / num_frames_to_average).astype(self.out_bit_depth)

    # ------------------------------------------------------------------ #
    # Legacy shims -- kept in microseconds on purpose                     #
    # ------------------------------------------------------------------ #

    def set_exposure_time_us(self, exposure_time: int) -> None:
        """Set exposure time in microseconds (µs). Legacy alias for `set_exposure_ms`."""
        self.set_exposure_ms(exposure_time / 1000)

    def set_settings(self,
                     exposure_time_us: int,
                     gain: int,
                     black_level: int,
                     bit_depth=np.uint16,
                     out_bit_depth=np.float32,
                     frame_rate_control_value: float | None = None) -> None:
        """Legacy convenience wrapper over the granular setters above.

        Kept in microseconds (not milliseconds) deliberately:
        ``scripts/phase/piezo_scan.py`` calls this positionally
        (``set_settings(EXPOSURE_US, GAIN, BLACK_LEVEL)``), and a unit change
        here would silently apply an exposure 1000x too large with no error.
        Prefer `set_exposure_ms` / `set_gain` / `set_black_level` directly in
        new code.
        """
        self.set_exposure_ms(exposure_time_us / 1000)
        if self.supports_gain:
            self.set_gain(gain)
        if self.supports_black_level:
            self.set_black_level(black_level)
        self.set_out_bit_depth(out_bit_depth)
        self.set_raw_bit_depth(bit_depth)
        if frame_rate_control_value is not None:
            self._set_frame_rate_control(frame_rate_control_value)

    def _set_frame_rate_control(self, fps: float) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support frame-rate control")


class CameraStream:
    """Runs Camera capture on a background daemon thread.

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

    def __init__(self, camera: Camera, buffer_size: int = 30):
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
