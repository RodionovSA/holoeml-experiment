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
instead rely on an explicit settle barrier tracked in this class (see
`EXPOSURE_SETTLE_FRAMES` / `_settle_target` below).

That barrier exists because the vendor's own `wait_for_new_image` is not
sufficient on its own: it only blocks until the recorder's processed-image
count exceeds the index of the last frame *this process copied*
(`pco.Camera._image_number`, updated inside `image()`). In a free-running ring
buffer, that count has usually already advanced past the caller by the time
`set_exposure_ms` runs (e.g. after a monochromator move), so the very next
`image(_LATEST_IMAGE)` call returns instantly with a frame that started
integrating under the *old* exposure -- confirmed empirically in
`scripts/test/test_pco_camera.ipynb` (every other frame read back saturated
or under-exposed by exactly one exposure setting). The barrier fixes this by
tracking, in addition to "has a new frame arrived", "has the recorder processed
at least N more frames since I last changed something that affects the frame's
content" -- exposure (`set_exposure_ms`) or anything upstream of the camera
that a caller signals via `_flush()` (e.g. `sample.py` moving the focus motor
*after* applying the new exposure but before capturing).
"""

from __future__ import annotations

import time

import numpy as np
import pco

from instruments.camera.base import Camera

#: pco's sentinel for "the most recently recorded image" (see pco.Camera.image()).
_LATEST_IMAGE = 0xFFFFFFFF

#: `dwProcImgCount` is a uint32 recorder frame counter (`pco.Recorder.get_status`).
_UINT32_MOD = 1 << 32


def _reached(count: int, target: int) -> bool:
    """Wrap-safe ``count >= target`` for the recorder's uint32 frame counter."""
    return ((count - target) % _UINT32_MOD) < (_UINT32_MOD >> 1)


class PcoCamera(Camera):
    """`pco.Camera` wrapper, selected by serial number."""

    supports_gain = False
    supports_black_level = False

    #: Depth of the ring buffer used by `arm()` / `_get_single_frame()`.
    RING_BUFFER_SIZE = 4

    #: Frames the recorder must process after an exposure change before a grab
    #: is trusted: one for whatever frame was already in flight under the old
    #: exposure, one more so the new timing isn't caught mid-latch at a frame
    #: boundary. Not underscore-prefixed so a verification/calibration script
    #: can override it on the class or an instance (e.g. set to 0 to reproduce
    #: pre-fix behaviour, or sweep it to find the minimum this camera needs).
    #: See `scripts/test/pco_exposure_settle.py`.
    EXPOSURE_SETTLE_FRAMES = 2

    #: Frames the recorder must process after `_flush()` before a grab is
    #: trusted. Covers callers that move other hardware (monochromator, filter
    #: wheel, focus motor) between an exposure change and the actual capture --
    #: `amplitude/measurements/sample.py` moves focus *after*
    #: `apply_camera_exposure`, for example. Smaller than
    #: `EXPOSURE_SETTLE_FRAMES` since no firmware reprogramming is involved,
    #: just draining whatever the free-running recorder already produced.
    FLUSH_SETTLE_FRAMES = 1

    #: Poll interval while waiting for the settle barrier, matching the vendor
    #: `wait_for_new_image`'s own polling cadence.
    _SETTLE_POLL_S = 0.001

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
        #: Recorder frame index (`dwProcImgCount`, mod 2**32) that must be
        #: reached before a grab is considered settled, or `None` if no
        #: barrier is pending. Set by `_require_fresh_frames` (called from
        #: `set_exposure_ms` / `_flush`), consumed by `_await_settle` (called
        #: from `_get_single_frame`). Cleared on `arm`/`disarm` since
        #: `record()`/`stop()` reset the vendor's own counter to 0.
        self._settle_target: int | None = None
        self._settle_timeout_s = 0.0

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

        Applies directly with no recorder restart. Any frame still in flight under the
        old exposure -- or already sitting in the ring buffer -- is handled by the
        settle barrier: this arms it so `_get_single_frame` won't return a grab
        until `EXPOSURE_SETTLE_FRAMES` fresh frames have been processed, the same
        idea `ThorlabsCamera._flush` implements for its FIFO queue.
        """
        previous_exposure_s = self._cam.exposure_time

        ns = round(exposure_ms * 1e6)
        ns = min(max(ns, self._min_exposure_ns), self._max_exposure_ns)
        ns = min(-(-ns // self._exposure_step_ns) * self._exposure_step_ns, self._max_exposure_ns)

        # The vendor setter re-derives its own timebase with int(), which truncates;
        # the half-nanosecond nudge stops float round-trip error from dropping the
        # applied value a count below `ns` (and back under the hardware minimum).
        self._cam.exposure_time = (ns + 0.5) / 1e9

        # The frame already in flight when this runs was exposed for as long as the
        # *old* setting, not the new one -- e.g. going 1000ms -> 200ms, that frame
        # still takes up to 1s to finish. Size the settle timeout on whichever is
        # longer so a big drop in exposure can't time out waiting on its own barrier.
        self._require_fresh_frames(self.EXPOSURE_SETTLE_FRAMES,
                                    exposure_s=max(previous_exposure_s, ns / 1e9))

    def set_roi(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Restrict the hardware ROI to (x0, y0, x1, y1), instead of transferring
        the full sensor every frame.

        Same 1-based, inclusive convention as the vendor SDK itself
        (`pco.Camera.configuration['roi']`, `sdk.get_roi()` / `sdk.set_roi()`),
        so a value read from one can be fed straight into this.

        The window is clamped to the sensor, then its width/height are rounded
        up to the ROI step grid the sensor's readout architecture requires
        (`description['roi steps']`), growing toward increasing x1/y1 (x0, y0
        are left as given) and clamped back to the sensor if that would run
        off the edge; call `image_shape` afterwards for the actual applied
        size. Some pco sensors additionally require the ROI to be centered /
        symmetric about the sensor (`description['roi is horz/vert
        symmetric']`); that is not enforced here -- the vendor SDK is, and
        rejects a violating window with a CameraException.

        Unlike `set_exposure_ms`, this disarms and re-arms if the camera is currently
        armed -- the SDK rejects ROI changes while a recording is in progress. That
        re-arm already resets the settle barrier (see `arm`), so no separate
        settle call is needed here.
        """
        x0 = min(max(x0, 1), self._sensor_width)
        y0 = min(max(y0, 1), self._sensor_height)
        x1 = min(max(x1, x0), self._sensor_width)
        y1 = min(max(y1, y0), self._sensor_height)

        width = max(self._min_roi_width, x1 - x0 + 1)
        height = max(self._min_roi_height, y1 - y0 + 1)
        width = min(-(-width // self._roi_step_x) * self._roi_step_x, self._sensor_width - x0 + 1)
        height = min(-(-height // self._roi_step_y) * self._roi_step_y, self._sensor_height - y0 + 1)
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
        # record() resets the vendor's own dwProcImgCount to 0; a barrier armed
        # against the previous recording's counter would be stale (either an
        # instant false-pass or a phantom multi-second wait).
        self._settle_target = None
        self._settle_timeout_s = 0.0

    def disarm(self) -> None:
        """Stop acquisition."""
        self._cam.stop()
        self._armed = False
        self._settle_target = None
        self._settle_timeout_s = 0.0

    def _frames_processed(self) -> int:
        """Recorder's running count of images processed since `arm()` (uint32)."""
        return self._cam.rec.get_status()["dwProcImgCount"]

    def _require_fresh_frames(self, n_frames: int, exposure_s: float | None = None) -> None:
        """Arm (or extend) the settle barrier: the next `_get_single_frame()`
        call blocks until `n_frames` more frames have been processed.

        No-op while disarmed -- `arm()` calls `record()`, which latches current
        settings and resets the frame counter, so there is nothing to wait out
        yet. If a barrier is already pending, keep whichever target is further
        in the future rather than overwriting it (covers `set_exposure_ms`
        immediately followed by `_flush()`, or vice versa).
        """
        if not self._armed or n_frames <= 0:
            return
        target = (self._frames_processed() + n_frames) % _UINT32_MOD
        if self._settle_target is None or _reached(target, self._settle_target):
            self._settle_target = target
        if exposure_s is None:
            exposure_s = self._cam.exposure_time
        self._settle_timeout_s = max(
            self._settle_timeout_s,
            n_frames * (exposure_s + self._FRAME_TIMEOUT_MARGIN_S),
        )

    def _await_settle(self) -> None:
        """Block until the pending settle barrier (if any) is satisfied."""
        target = self._settle_target
        if target is None:
            return
        deadline = time.perf_counter() + self._settle_timeout_s
        try:
            while True:
                status = self._cam.rec.get_status()
                if not status["bIsRunning"]:
                    # Recording stopped from under us (e.g. another thread
                    # disarmed); nothing left to wait for -- let the caller's
                    # own wait_for_new_image/timeout handle what comes next.
                    return
                if _reached(status["dwProcImgCount"], target):
                    return
                if time.perf_counter() > deadline:
                    raise TimeoutError(
                        f"Camera {self.serial_number}: exposure/flush settle barrier "
                        f"(target frame {target}, now {status['dwProcImgCount']}) not "
                        f"reached within {self._settle_timeout_s:.1f}s"
                    )
                time.sleep(self._SETTLE_POLL_S)
        finally:
            self._settle_target = None
            self._settle_timeout_s = 0.0

    def _flush(self) -> None:
        """Discard frames already in flight or buffered before this call.

        Called by `Camera.get_image()` before its drop/average loop. Arms a
        (smaller) settle barrier so a caller that moved other hardware --
        monochromator, filter wheel, focus motor -- between the last exposure
        change and this capture still gets a frame acquired after that move,
        not whatever was already sitting in the free-running ring buffer.
        """
        self._require_fresh_frames(self.FLUSH_SETTLE_FRAMES)

    #: Extra margin (s) added on top of the configured exposure for the default
    #: per-frame wait timeout, covering sensor readout/transfer and general system
    #: jitter. A flat 5.0s default timeout (independent of exposure) was observed to
    #: intermittently expire around ~1.8s exposures on this camera's full-sensor
    #: frame size -- see `set_roi` for shrinking that transfer, and this margin for
    #: not depending on transfer time being negligible in the first place.
    _FRAME_TIMEOUT_MARGIN_S = 5.0

    def _get_single_frame(self, timeout: float | None = None) -> np.ndarray:
        """Block until a new, settled frame is available, then return it as (H, W, 1).

        First waits out any pending settle barrier (see `_require_fresh_frames`),
        then waits for that settled frame to actually be copyable.

        Parameters
        ----------
        timeout : float or None
            Maximum seconds to wait for a new frame *after* the settle barrier
            clears, before raising TimeoutError. Defaults to the currently
            configured exposure time plus `_FRAME_TIMEOUT_MARGIN_S` --
            `Camera.get_image()` (the only other caller of this method) never
            overrides it, so a fixed default has to scale with exposure itself
            rather than assume exposures stay short.
        """
        self._await_settle()
        if timeout is None:
            timeout = self._cam.exposure_time + self._FRAME_TIMEOUT_MARGIN_S
        self._cam.wait_for_new_image(delay=True, timeout=timeout)
        image, _meta = self._cam.image(_LATEST_IMAGE)
        return image[:, :, np.newaxis]
