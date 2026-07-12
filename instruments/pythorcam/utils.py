import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .thorcam import CameraStream, ThorlabsCamera


def zoom(img: np.ndarray, scaling_factor: float) -> np.ndarray:
    """Resize `img` by `scaling_factor` using OpenCV bilinear interpolation."""
    return cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor)


def live_view(camera: ThorlabsCamera,
              zoom_value: float = 0.5,
              overlay_lines: list[tuple[str, float, str]] | None = None,
              refresh_period: float = 0.05) -> None:
    """Real-time matplotlib viewer backed by a `CameraStream`.

    Capture runs on a background thread; the main thread only redraws.
    Stops on KeyboardInterrupt (Ctrl-C in the terminal, "Interrupt" in a notebook).

    Parameters
    ----------
    camera : ThorlabsCamera
        Already-armed camera.
    zoom_value : float
        Display zoom factor passed to `zoom()`.
    overlay_lines : list of (orientation, position_px, color), optional
        Each entry draws one reference line on the display. `orientation` is
        'h' for a horizontal line (`axhline`) or 'v' for vertical (`axvline`).
        `position_px` is in original (un-zoomed) image pixels.
    refresh_period : float
        Seconds between display redraws.
    """
    stream = CameraStream(camera)
    stream.start()
    try:
        plt.ion()
        fig, ax = plt.subplots()
        img = ax.imshow(zoom(camera.get_image(), zoom_value), cmap='gray')
        for orientation, position, color in (overlay_lines or []):
            scaled = int(zoom_value * position)
            if orientation == 'h':
                ax.axhline(y=scaled, color=color, linestyle='-')
            else:
                ax.axvline(x=scaled, color=color, linestyle='-')
        while True:
            frame = stream.get_latest_frame()
            if frame is not None:
                img.set_data(zoom(frame, zoom_value))
                fig.canvas.draw()
                fig.canvas.flush_events()
            time.sleep(refresh_period)
    except KeyboardInterrupt:
        print("loop terminated")
    finally:
        stream.stop()
        plt.ioff()
        plt.show()


def calculate_focus_measure(image: np.ndarray) -> float:
    """Laplacian-variance focus metric. Higher = sharper."""
    img = image[:, :, 0] if image.ndim == 3 else image
    return cv2.Laplacian(img, cv2.CV_32F).var()


def autofocus(camera_trans: ThorlabsCamera,
              focus_motor,
              start_position: float,
              end_position: float,
              step_size: float,
              velocity: float = 0.1,
              camera_ref: ThorlabsCamera | None = None,
              num_frames_to_average: int = 1,
              num_frames_to_drop: int = 5,
              delay: float = 0) -> tuple[float, np.ndarray, np.ndarray]:
    """Sweep the focus motor and return the position with the sharpest image.

    When *camera_ref* is provided the transmission image is divided by the
    reference image before the focus metric is computed (cancels illumination
    structure).  When *camera_ref* is ``None`` the metric is computed directly
    on the transmission image.

    Parameters
    ----------
    camera_trans : ThorlabsCamera
        Sample-arm (or sole) camera.  Must already be armed.
    focus_motor : pylablib KinesisMotor
        Must expose ``setup_velocity``, ``move_to``, ``wait_move``, and
        ``get_position`` (standard pylablib KinesisMotor API).
    start_position, end_position : float
        Sweep range in motor native units (metres for KinesisMotor with scale='stage').
    step_size : float
        Step in motor native units.
    velocity : float
        Motor velocity in motor native units / s.
    camera_ref : ThorlabsCamera or None
        Reference-arm camera.  When ``None``, normalization is skipped.
    num_frames_to_average, num_frames_to_drop : int
        Forwarded to ``camera.get_image()``.
    delay : float
        Forwarded to ``camera.get_image()``.

    Returns
    -------
    best_position : float
        Motor position (motor native units) with the highest focus metric.
    focus_curve : np.ndarray, shape (2, steps)
        Row 0 is positions, row 1 is focus metric values.
    images : np.ndarray, shape (steps, H, W)
        Captured transmission images at each step (channel 0 only).
    """
    best_focus = -1.0
    best_position = start_position

    steps = int(round((end_position - start_position) / step_size))
    positions = np.linspace(start_position, end_position, steps)

    focus_values = -np.ones(steps)
    images = np.zeros([steps, camera_trans.image_height, camera_trans.image_width])

    focus_motor.setup_velocity(max_velocity=velocity)
    focus_motor.move_to(float(start_position))
    focus_motor.wait_move()
    time.sleep(0.5)

    for step in range(steps):
        image_trans = camera_trans.get_image(num_frames_to_average=int(num_frames_to_average),
                                             num_frames_to_drop=int(num_frames_to_drop),
                                             delay=delay)

        if camera_ref is not None:
            image_ref = camera_ref.get_image(num_frames_to_average=int(num_frames_to_average),
                                             num_frames_to_drop=int(num_frames_to_drop),
                                             delay=delay)
            metric_image = image_trans / image_ref
        else:
            metric_image = image_trans

        current_focus = calculate_focus_measure(metric_image) * 1000
        focus_values[step] = current_focus
        images[step, :, :] = image_trans[:, :, 0]

        print(f'step {step}/{steps}  position={focus_motor.get_position():.6f}  focus={current_focus:.3f}')

        if (current_focus > best_focus) and (step != 0):
            best_focus = current_focus
            best_position = focus_motor.get_position()

        if step < steps - 1:
            focus_motor.move_to(float(positions[step + 1]))
            focus_motor.wait_move()

    focus_curve = np.array([positions, focus_values])

    focus_motor.move_to(float(best_position))
    focus_motor.wait_move()
    time.sleep(0.5)

    return best_position, focus_curve, images


def brightness_calibration(camera: ThorlabsCamera,
                           initial_exposure_time: int,
                           initial_gain: int,
                           target_brightness: float,
                           tolerance: float,
                           increment: float,
                           max_number_of_steps: int,
                           max_exposure_time: int,
                           priority: str = 'exposure_time',
                           gain_step_db: float = 0.5,
                           min_exposure_time: int = 28,
                           min_gain: int = 0,
                           max_gain: int = 480,
                           num_frames_to_average: int = 1,
                           num_frames_to_drop: int = 5,
                           delay: float = 0,
                           roi_fraction: float | None = None) -> tuple[int, int, float]:
    """Iteratively adjust exposure time and gain until mean brightness hits target.

    One control is treated as primary (the `priority` knob): it is exhausted
    before the secondary knob is touched.  Exposure time uses a multiplicative
    step ``(1 ± increment)``; gain uses a fixed additive step in dB to avoid
    the large brightness jumps that multiplicative stepping causes at high gain.

    Parameters
    ----------
    camera : ThorlabsCamera
    initial_exposure_time : int
        Starting exposure in microseconds (µs). Must be armed!
    initial_gain : int
        Starting gain in SDK-defined units.
    target_brightness : float
        Desired mean brightness normalised to [0, 1] relative to
        ``camera.pixel_max_value``.
    tolerance : float
        Stop once ``|normalised_mean - target| < tolerance``.
    increment : float
        Multiplicative step for exposure time (e.g. 0.1 means ±10% per iteration).
    max_number_of_steps : int
        Bail out after this many iterations even if target not reached.
    max_exposure_time : int
        Upper limit for exposure in µs (user-defined ceiling).
    priority : {'exposure_time', 'gain'}
        Which control to exhaust first when brightness needs to increase.
        The opposite control is used first when brightness needs to decrease.
    gain_step_db : float
        Fixed gain step in dB per iteration (1 dB = 10 SDK units).
        Additive stepping avoids large brightness jumps at high gain values.
    min_exposure_time : int
        Lower limit for exposure in µs. Defaults to CS126 hardware minimum (28).
    min_gain, max_gain : int
        Gain range in SDK-defined units. Defaults to CS126 range [0, 480].
    num_frames_to_average, num_frames_to_drop : int
        Forwarded to ``camera.get_image()``.
    delay : float
        Forwarded to ``camera.get_image()``.
    roi_fraction : float or None
        If given, brightness is measured over the central ``roi_fraction``
        fraction of image height and width (e.g. 0.5 = central 50%×50%).
        ``None`` uses the full frame.

    Returns
    -------
    tuple[int, int, float]
        Final ``(exposure_time_us, gain, best_brightness)`` after calibration.
        ``best_brightness`` is the normalised mean brightness at the returned settings.
    """
    if priority not in ('exposure_time', 'gain'):
        raise ValueError("priority must be 'exposure_time' or 'gain'")

    max_exposure_time = min(max_exposure_time, 14700924)
    gain_step = max(1, round(gain_step_db * 10))  # SDK units; 1 unit = 0.1 dB

    current_exposure = initial_exposure_time
    current_gain = initial_gain
    camera.set_exposure_time_us(current_exposure)
    camera.set_gain(current_gain)

    final_brightness = 0.0
    for _ in range(max_number_of_steps):
        image = camera.get_image(num_frames_to_average=int(num_frames_to_average),
                                 num_frames_to_drop=int(num_frames_to_drop),
                                 delay=delay)
        if roi_fraction is not None:
            h, w = image.shape[0], image.shape[1]
            cy, cx = h // 2, w // 2
            ry, rx = max(1, int(h * roi_fraction / 2)), max(1, int(w * roi_fraction / 2))
            region = image[cy - ry : cy + ry, cx - rx : cx + rx]
            mean_brightness = region.mean() / camera.pixel_max_value
        else:
            mean_brightness = image.mean() / camera.pixel_max_value
        final_brightness = mean_brightness

        if abs(mean_brightness - target_brightness) < tolerance:
            print(f'Target brightness reached: {mean_brightness:.2f}')
            break

        if mean_brightness < target_brightness:
            # Need more brightness — use primary knob first, then secondary.
            if priority == 'exposure_time':
                if current_exposure < max_exposure_time:
                    current_exposure = int(min(max(current_exposure * (1 + increment), current_exposure + 1), max_exposure_time))
                    camera.set_exposure_time_us(current_exposure)
                elif current_gain < max_gain:
                    current_gain = min(current_gain + gain_step, max_gain)
                    camera.set_gain(current_gain)
            else:  # priority == 'gain'
                if current_gain < max_gain:
                    current_gain = min(current_gain + gain_step, max_gain)
                    camera.set_gain(current_gain)
                elif current_exposure < max_exposure_time:
                    current_exposure = int(min(max(current_exposure * (1 + increment), current_exposure + 1), max_exposure_time))
                    camera.set_exposure_time_us(current_exposure)
        else:
            # Need less brightness — use secondary knob first to recover primary headroom.
            if priority == 'exposure_time':
                if current_gain > min_gain:
                    current_gain = max(current_gain - gain_step, min_gain)
                    camera.set_gain(current_gain)
                elif current_exposure > min_exposure_time:
                    current_exposure = int(max(min(current_exposure * (1 - increment), current_exposure - 1), min_exposure_time))
                    camera.set_exposure_time_us(current_exposure)
            else:  # priority == 'gain'
                if current_exposure > min_exposure_time:
                    current_exposure = int(max(min(current_exposure * (1 - increment), current_exposure - 1), min_exposure_time))
                    camera.set_exposure_time_us(current_exposure)
                elif current_gain > min_gain:
                    current_gain = max(current_gain - gain_step, min_gain)
                    camera.set_gain(current_gain)

        time.sleep(0.5)

    return current_exposure, current_gain, float(final_brightness)


def autoexposure(camera: ThorlabsCamera,
                 initial_exposure_time: int,
                 target_brightness: float,
                 tolerance: float,
                 increment: float,
                 max_number_of_steps: int,
                 num_frames_to_average: int = 1,
                 num_frames_to_drop: int = 5,
                 delay: float = 0,
                 roi_fraction: float | None = None) -> int:
    """Iteratively adjust exposure until the mean brightness hits target.

    Multiplicatively scales the exposure by (1 ± `increment`) each iteration,
    clamped to the CS126 hardware range [28, 14700924] µs.

    Parameters
    ----------
    camera : ThorlabsCamera
    initial_exposure_time : int
        Starting exposure in microseconds (µs).
    target_brightness : float
        Desired mean pixel value of the averaged image.
    tolerance : float
        Stop once `|mean - target| < tolerance`.
    increment : float
        Multiplicative step (e.g. 0.1 means ±10% per iteration).
    max_number_of_steps : int
        Bail out after this many iterations even if target not reached.
    num_frames_to_average, num_frames_to_drop : int
        Forwarded to `camera.get_image()`.
    delay : float
        Forwarded to `camera.get_image()`.
    roi_fraction : float or None
        If given, brightness is measured over the central ``roi_fraction``
        fraction of image height and width (e.g. 0.5 = central 50%×50%).
        ``None`` uses the full frame.

    Returns
    -------
    int
        Final exposure time in microseconds.
    """
    camera.set_exposure_time_us(initial_exposure_time)

    current_exposure_time = initial_exposure_time

    for _ in range(max_number_of_steps):
        image = camera.get_image(num_frames_to_average=int(num_frames_to_average),
                                 num_frames_to_drop=int(num_frames_to_drop),
                                 delay=delay)
        
        if roi_fraction is not None:
            h, w = image.shape[0], image.shape[1]
            cy, cx = h // 2, w // 2
            ry, rx = max(1, int(h * roi_fraction / 2)), max(1, int(w * roi_fraction / 2))
            region = image[cy - ry : cy + ry, cx - rx : cx + rx]
            mean_brightness = region.mean() 
        else:
            mean_brightness = image.mean() 

        if abs(mean_brightness - target_brightness) < tolerance:
            print(f'Target brightness reached: {mean_brightness}')
            break

        if mean_brightness < target_brightness:
            current_exposure_time = int(min(current_exposure_time * (1 + increment), 14700924))
        else:
            current_exposure_time = int(max(current_exposure_time * (1 - increment), 28))

        camera.set_exposure_time_us(current_exposure_time)
        time.sleep(0.5)

    return current_exposure_time
