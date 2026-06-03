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
    return cv2.Laplacian(image, cv2.CV_64F).var()


def autofocus(camera_trans: ThorlabsCamera,
              camera_ref: ThorlabsCamera,
              focus_motor,
              start_position: float,
              end_position: float,
              step_size: float,
              velocity: float = 0.1,
              num_frames_to_average: int = 1,
              num_frames_to_drop: int = 5,
              delay: float = 0) -> tuple[float, np.ndarray, np.ndarray]:
    """Sweep the focus motor and return the position with the sharpest image.

    For each position the transmission image is divided by the reference image
    before the focus metric is computed (cancels out illumination structure).

    Parameters
    ----------
    camera_trans, camera_ref : ThorlabsCamera
        Sample-arm and reference-arm cameras.
    focus_motor : Thorlabs.KinesisMotor (or wrapper)
        Must expose `Move_to_position(position, velocity)` and `Get_position()`.
        NOTE: these names don't match pylablib's KinesisMotor API
        (`move_to` / `get_position`) — see CLAUDE.md, focus-motor wrapper TBD.
    start_position, end_position : float
        Sweep range in motor units (mm for KDC101 + Z8 stage).
    step_size : float
        Step in motor units.
    velocity : float
        Motor velocity in motor units / s.
    num_frames_to_average, num_frames_to_drop : int
        Forwarded to `camera.get_image()`.
    delay : float
        Forwarded to `camera.get_image()`.

    Returns
    -------
    best_position : float
        Motor position with the highest focus metric.
    focus_curve : np.ndarray, shape (2, steps)
        Row 0 is positions, row 1 is focus metric values.
    images : np.ndarray, shape (steps, H, W)
        Captured transmission images at each step (channel 0 only).
    """
    best_focus = -1
    best_position = start_position

    focus_motor.Move_to_position(float(start_position), float(0.1))
    time.sleep(0.5)

    steps = int((end_position - start_position) / step_size)
    positions = np.linspace(start_position, end_position, steps)

    focus_values = -np.ones([steps])
    images = np.zeros([steps, camera_trans.image_height, camera_trans.image_width])
    for step in range(steps):
        image_trans = camera_trans.get_image(num_frames_to_average=int(num_frames_to_average),
                                             num_frames_to_drop=int(num_frames_to_drop),
                                             delay=delay)
        image_ref = camera_ref.get_image(num_frames_to_average=int(num_frames_to_average),
                                         num_frames_to_drop=int(num_frames_to_drop),
                                         delay=delay)

        current_focus = calculate_focus_measure(image_trans / image_ref) * 1000
        focus_values[step] = current_focus
        images[step, :, :] = image_trans[:, :, 0]

        print('current_focus: ', current_focus)

        if (current_focus > best_focus) and (step != 0):
            best_focus = current_focus
            best_position = focus_motor.Get_position()

        focus_motor.Move_to_position(float(start_position + (step + 1) * step_size), float(velocity))
        time.sleep(0.1)

    focus_curve = np.array([positions, focus_values])
    focus_motor.Move_to_position(float(best_position), float(0.05))
    time.sleep(0.5)

    return best_position, focus_curve, images


def autoexposure(camera: ThorlabsCamera,
                 initial_exposure_time: int,
                 target_brightness: float,
                 tolerance: float,
                 increment: float,
                 max_number_of_steps: int,
                 num_frames_to_average: int = 1,
                 num_frames_to_drop: int = 5,
                 delay: float = 0) -> int:
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
