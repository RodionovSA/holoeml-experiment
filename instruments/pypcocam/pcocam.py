"""PCO camera driver -- implements the shared `instruments.camera.Camera`
interface via the vendor `pco` SDK (https://pypi.org/project/pco/).

Unlike the Thorlabs CS126, this camera has no gain control (only a fixed
"conversion factor"), so `supports_gain` and `supports_black_level` are both
False; `instruments.camera.utils.brightness_calibration` uses that flag to
run exposure-only calibration.

Acquisition uses the SDK's non-blocking 'ring buffer' recorder mode so a
single `_get_single_frame` implementation works uniformly for both one-shot
capture (`get_image`, inherited from `Camera`) and continuous streaming
(`CameraStream`, also inherited).

`set_exposure_ms` applies exposure changes directly via the SDK setter with no
recorder restart, matching `ThorlabsCamera` (which never re-arms on exposure
change either). An earlier version of this driver disarmed and re-armed on every
exposure change on the theory that the SDK only "latches" new settings on arm
(`pco.Camera.arm_camera()`'s docstring does say settings are validated on arm);
but `arm()` here means `pco.Camera.record(...)`, which doesn't just re-validate
settings -- it tears down and reallocates the vendor's DMA-backed ring-buffer
recorder from scratch every time. Doing that once per exposure change was
observed to cause the frame-wait to stall a few cycles later, so exposure changes
now rely purely on the settle-frame-drop mechanism `autoexposure`/`get_image`
already provide (`num_frames_to_drop`) to skip over any transitional frame, same
as the Thorlabs path.
"""

from __future__ import annotations

import numpy as np
import pco

from instruments.camera.base import Camera

#: pco's sentinel for "the most recently recorded image" (see pco.Camera.image()).
_LATEST_IMAGE = 0xFFFFFFFF


class PcoCamera(Camera):
    """`pco.Camera` wrapper, selected by serial number."""

    supports_gain = False
    supports_black_level = False

    #: Depth of the ring buffer used by `arm()` / `_get_single_frame()`.
    RING_BUFFER_SIZE = 4

    def __init__(self, serial: str):
        """Connect to a pco camera by serial number.

        Parameters
        ----------
        serial : str
            Camera serial number. Note ``pco.Camera``'s first positional
            argument is a *name*, not a serial -- selecting by serial
            requires the ``serial=`` keyword with an int, which is what
            this wrapper does.
        """
        super().__init__()
        self.serial_number = serial
        self._cam = pco.Camera(serial=int(serial))
        self._cam.auto_exposure_off()
        self._armed = False

        # Cached once at connect (read-only unless IR sensitivity is toggled, which this
        # driver never does). The vendor `exposure_time` setter's own guard admits values
        # down to one step below the true minimum and the SDK then rejects them outright
        # (CameraException, not ValueError) -- see `set_exposure_ms` -- so the real limits
        # are cached here in integer ns for a clamp that actually holds.
        desc = self._cam.description
        self._min_exposure_ns = round(desc['min exposure time'] * 1e9)
        self._max_exposure_ns = round(desc['max exposure time'] * 1e9)
        self._exposure_step_ns = max(1, round(desc['min exposure step'] * 1e9))

        # Full-sensor dimensions and ROI placement rules, also cached once -- used by
        # `set_roi` to restrict the hardware ROI (by default the camera transfers the
        # entire sensor on every frame, which for this camera's multi-thousand-pixel
        # sensor is tens of MB per frame even when the caller only wants a small crop).
        self._sensor_width = desc['max width']
        self._sensor_height = desc['max height']
        roi_step_x, roi_step_y = desc['roi steps']
        self._roi_step_x = max(1, roi_step_x)
        self._roi_step_y = max(1, roi_step_y)
        self._min_roi_width = desc['min width']
        self._min_roi_height = desc['min height']

        print(f"Camera {serial} is connected")

    def close(self) -> None:
        """Disarm (if armed) and close the camera handle."""
        if self._armed:
            self.disarm()
        self._cam.close()
        print(f"Camera {self.serial_number} is disconnected")

    @property
    def image_shape(self) -> tuple[int, int]:
        """(height, width) in pixels, from the camera's configured ROI."""
        x0, y0, x1, y1 = self._cam.configuration['roi']  # 1-based, inclusive
        return (y1 - y0 + 1, x1 - x0 + 1)

    @property
    def sensor_shape(self) -> tuple[int, int]:
        """(height, width) of the full sensor, independent of the currently configured ROI."""
        return (self._sensor_height, self._sensor_width)

    @property
    def num_channels(self) -> int:
        return 1

    @property
    def pixel_max_value(self) -> int:
        bit_resolution = self._cam.description['bit resolution']
        return 2 ** bit_resolution - 1

    @property
    def min_exposure_ms(self) -> float:
        return self._min_exposure_ns / 1e6

    @property
    def max_exposure_ms(self) -> float:
        return self._max_exposure_ns / 1e6

    @property
    def exposure_ms(self) -> float:
        """Exposure actually applied by the camera (ms), after clamping and quantization."""
        return self._cam.exposure_time * 1000

    @property
    def min_gain(self) -> int:
        return 0

    @property
    def max_gain(self) -> int:
        return 0

    def set_exposure_ms(self, exposure_ms: float) -> None:
        """Set exposure time in milliseconds, clamped to the camera's hardware range.

        The vendor `exposure_time` setter's own guard admits values down to one step
        below the hardware minimum; the SDK then rejects those with a bare "wrong
        parameter" CameraException rather than the ValueError you'd expect, so the
        clamp has to happen here instead. Rounding up to the step grid afterwards is
        belt-and-braces: the firmware quantizes internally, this just keeps us off
        the boundary it's rejecting.

        Applies directly with no recorder restart -- any frame still in flight under
        the old exposure is the caller's problem to drop via `num_frames_to_drop`
        (`autoexposure`/`get_image` already do this), same as `ThorlabsCamera`.
        """
        ns = round(exposure_ms * 1e6)
        ns = min(max(ns, self._min_exposure_ns), self._max_exposure_ns)
        ns = min(-(-ns // self._exposure_step_ns) * self._exposure_step_ns, self._max_exposure_ns)

        # The vendor setter re-derives its own timebase with int(), which truncates;
        # the half-nanosecond nudge stops float round-trip error from dropping the
        # applied value a count below `ns` (and back under the hardware minimum).
        self._cam.exposure_time = (ns + 0.5) / 1e9

    def set_roi(self, width: int, height: int) -> None:
        """Restrict the hardware ROI to a centered window of approximately
        (width, height) pixels, instead of transferring the full sensor every frame.

        The requested size is clamped to the sensor and rounded up to the ROI step
        grid the sensor's readout architecture requires (`description['roi steps']`);
        call `image_shape` afterwards for the actual applied size. The window is
        always centered on the sensor, which satisfies this camera's ROI symmetry
        requirement (`description['roi is horz/vert symmetric']`) whether or not it's
        actually enforced.

        Like `set_exposure_ms`, this disarms and re-arms if the camera is currently
        armed -- the SDK rejects ROI changes while a recording is in progress.
        """
        width = min(max(width, self._min_roi_width), self._sensor_width)
        height = min(max(height, self._min_roi_height), self._sensor_height)
        width = min(-(-width // self._roi_step_x) * self._roi_step_x, self._sensor_width)
        height = min(-(-height // self._roi_step_y) * self._roi_step_y, self._sensor_height)

        x0 = (self._sensor_width - width) // 2 + 1     # 1-based, inclusive
        y0 = (self._sensor_height - height) // 2 + 1
        x1 = x0 + width - 1
        y1 = y0 + height - 1

        was_armed = self._armed
        if was_armed:
            self.disarm()
        self._cam.sdk.set_roi(x0, y0, x1, y1)
        if was_armed:
            self.arm()

    def arm(self) -> None:
        """Start continuous acquisition into a ring buffer."""
        self._cam.record(number_of_images=self.RING_BUFFER_SIZE, mode="ring buffer")
        self._armed = True

    def disarm(self) -> None:
        """Stop acquisition."""
        self._cam.stop()
        self._armed = False

    #: Extra margin (s) added on top of the configured exposure for the default
    #: per-frame wait timeout, covering sensor readout/transfer and general system
    #: jitter. A flat 5.0s default timeout (independent of exposure) was observed to
    #: intermittently expire around ~1.8s exposures on this camera's full-sensor
    #: frame size -- see `set_roi` for shrinking that transfer, and this margin for
    #: not depending on transfer time being negligible in the first place.
    _FRAME_TIMEOUT_MARGIN_S = 5.0

    def _get_single_frame(self, timeout: float | None = None) -> np.ndarray:
        """Block until a new frame is available, then return it as (H, W, 1).

        Parameters
        ----------
        timeout : float or None
            Maximum seconds to wait for a new frame before raising TimeoutError.
            Defaults to the currently configured exposure time plus
            `_FRAME_TIMEOUT_MARGIN_S` -- `Camera.get_image()` (the only other
            caller of this method) never overrides it, so a fixed default has to
            scale with exposure itself rather than assume exposures stay short.
        """
        if timeout is None:
            timeout = self._cam.exposure_time + self._FRAME_TIMEOUT_MARGIN_S
        self._cam.wait_for_new_image(delay=True, timeout=timeout)
        image, _meta = self._cam.image(_LATEST_IMAGE)
        return image[:, :, np.newaxis]
