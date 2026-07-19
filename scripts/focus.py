"""Real-time focusing app.

Shows a live camera view with controls for exposure, gain, and the focus
motor (absolute ``move_to`` / relative ``move_by``), plus a live sharpness
readout (Laplacian variance) to help judge focus by eye. Run this before a
measurement to focus the sample; all defaults (serials, exposure/gain
starting point, focus velocity/acceleration) come from the YAML config.

The focus motor is never moved on startup -- only its position is read.

Usage
-----
    python scripts/focus.py
    python scripts/focus.py -c /path/to/config.yaml --zoom 0.75
"""

import argparse
import threading
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox

import amplitude
from amplitude.config import Config
from instruments.kinesismotor import KinesisMotor
from instruments.pythorcam.thorcam import CameraStream, ThorlabsCamera, create_camera_sdk
from instruments.pythorcam.utils import calculate_focus_measure, zoom

PKG_ROOT = Path(amplitude.__file__).resolve().parent  # .../holoeml-experiment/amplitude


class FocusApp:
    """Live camera view + exposure/gain/focus-motor controls (matplotlib widgets)."""

    def __init__(self, camera: ThorlabsCamera, focus: KinesisMotor, config: Config,
                 zoom_factor: float = 0.5):
        self.camera = camera
        self.focus = focus
        self.config = config
        self.zoom_factor = zoom_factor

        self.stream = CameraStream(camera)
        self._moving = False

        self._build_figure()

    # ── figure / widgets ──────────────────────────────────────────────────

    def _build_figure(self) -> None:
        self.fig = plt.figure("Focus", figsize=(7.5, 9))
        self.ax_img = self.fig.add_axes([0.08, 0.40, 0.84, 0.56])
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])

        H, W = self.camera.image_shape
        blank = np.zeros((int(H * self.zoom_factor), int(W * self.zoom_factor)))
        self.img = self.ax_img.imshow(blank, cmap='gray', vmin=0,
                                       vmax=self.camera.pixel_max_value)

        self.status_text = self.fig.text(0.08, 0.34, "", fontsize=10, family='monospace')

        # Exposure slider
        ax_exp = self.fig.add_axes([0.2, 0.27, 0.6, 0.03])
        self.sl_exposure = Slider(
            ax_exp, 'Exposure (ms)',
            valmin=0.03, valmax=max(self.config.calib_max_exposure_ms, 1),
            valinit=self.config.calib_initial_exposure_ms,
        )
        self.sl_exposure.on_changed(self._on_exposure_changed)

        # Gain slider (SDK units are tenths of a dB)
        ax_gain = self.fig.add_axes([0.2, 0.22, 0.6, 0.03])
        self.sl_gain = Slider(
            ax_gain, 'Gain (dB)',
            valmin=0, valmax=48, valinit=self.config.calib_initial_gain / 10,
        )
        self.sl_gain.on_changed(self._on_gain_changed)

        # move_to
        ax_moveto = self.fig.add_axes([0.2, 0.13, 0.35, 0.045])
        self.tb_moveto = TextBox(ax_moveto, 'move_to (mm)   ',
                                  initial=f"{self._safe_get_position():.4f}")
        self.tb_moveto.on_submit(self._submit_move_to)
        ax_go = self.fig.add_axes([0.58, 0.13, 0.12, 0.045])
        self.btn_go = Button(ax_go, 'Go')
        self.btn_go.on_clicked(lambda _event: self._submit_move_to(self.tb_moveto.text))

        # move_by
        ax_step = self.fig.add_axes([0.2, 0.06, 0.35, 0.045])
        self.tb_step = TextBox(ax_step, 'move_by step (mm)', initial='0.001')
        ax_minus = self.fig.add_axes([0.58, 0.06, 0.12, 0.045])
        self.btn_minus = Button(ax_minus, '-')
        self.btn_minus.on_clicked(lambda _event: self._submit_move_by(-1))
        ax_plus = self.fig.add_axes([0.72, 0.06, 0.12, 0.045])
        self.btn_plus = Button(ax_plus, '+')
        self.btn_plus.on_clicked(lambda _event: self._submit_move_by(+1))

        self.fig.canvas.mpl_connect('close_event', self._on_close)

    # ── camera controls ───────────────────────────────────────────────────

    def _on_exposure_changed(self, value_ms: float) -> None:
        self.camera.set_exposure_time_us(int(value_ms * 1000))

    def _on_gain_changed(self, value_db: float) -> None:
        self.camera.set_gain(int(round(value_db * 10)))

    # ── focus motor controls ──────────────────────────────────────────────

    def _safe_get_position(self) -> float:
        try:
            return self.focus.get_position()
        except Exception as exc:
            print(f"[focus] could not read motor position: {exc}")
            return 0.0

    def _submit_move_to(self, text: str) -> None:
        try:
            value = float(text)
        except ValueError:
            print(f"[focus] invalid move_to value: {text!r}")
            return
        self._start_move(self.focus.move_to, value)

    def _submit_move_by(self, sign: int) -> None:
        try:
            step = float(self.tb_step.text)
        except ValueError:
            print(f"[focus] invalid step value: {self.tb_step.text!r}")
            return
        self._start_move(self.focus.move_by, sign * step)

    def _start_move(self, move_fn, value: float) -> None:
        if self._moving:
            print("[focus] motor is already moving; ignoring request")
            return
        self._moving = True
        threading.Thread(target=self._move_worker, args=(move_fn, value), daemon=True).start()

    def _move_worker(self, move_fn, value: float) -> None:
        try:
            move_fn(value)
        except Exception as exc:
            print(f"[focus] motor move failed: {exc}")
        finally:
            self._moving = False

    # ── live view loop ─────────────────────────────────────────────────────

    def _on_tick(self) -> None:
        frame = self.stream.get_latest_frame()
        if frame is not None:
            self.img.set_data(zoom(frame[:, :, 0], self.zoom_factor))
            focus_metric = calculate_focus_measure(frame)
            brightness = float(frame.mean())
            position = "moving..." if self._moving else f"{self._safe_get_position():.4f} mm"
            self.status_text.set_text(
                f"position: {position}   focus: {focus_metric:8.2f}   brightness: {brightness:8.1f}"
            )
        self.fig.canvas.draw_idle()

    def _on_close(self, _event) -> None:
        self.timer.stop()
        self.stream.stop()

    def run(self) -> None:
        self.stream.start()
        self.timer = self.fig.canvas.new_timer(interval=50)
        self.timer.add_callback(self._on_tick)
        self.timer.start()
        plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", type=Path, default=PKG_ROOT / "config/config.yaml")
    p.add_argument("--zoom", type=float, default=0.5, help="display downscale factor")
    args = p.parse_args()

    config = Config.from_yaml(str(args.config))

    sdk = create_camera_sdk()
    try:
        with ThorlabsCamera(sdk, config.camera_serial) as camera, \
             KinesisMotor(config.focus_serial, motor_type='stage') as focus:
            camera.set_settings(
                exposure_time_us=config.calib_initial_exposure_ms * 1000,
                gain=config.calib_initial_gain,
                black_level=config.camera_black_level,
                bit_depth=getattr(np, config.camera_bit_depth),
                out_bit_depth=getattr(np, config.camera_out_bit_depth),
            )
            if config.default_focus_max_velocity or config.default_focus_acceleration:
                focus.set_velocity(max_velocity=config.default_focus_max_velocity,
                                    acceleration=config.default_focus_acceleration)

            camera.arm()
            try:
                app = FocusApp(camera, focus, config, zoom_factor=args.zoom)
                app.run()
            finally:
                camera.disarm()
    finally:
        sdk.dispose()


if __name__ == "__main__":
    main()
