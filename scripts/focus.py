"""Real-time focusing app.

Shows a live camera view with controls for exposure, gain, and the focus
motor (absolute ``move_to`` / relative ``move_by``), plus a live sharpness
readout (Laplacian variance) to help judge focus by eye. Run this before a
measurement to focus the sample; all defaults (serials, exposure/gain
starting point, focus velocity/acceleration) come from the YAML config.

The focus motor is never moved on startup -- only its position is read.

The live view is optimized for speed: the display frame is cheaply
downsampled (strided slicing, not a full-resolution resize), only the image
is repainted each tick (matplotlib blitting) instead of the whole figure, the
focus/brightness metrics are computed on a background thread from a small
central crop, and the motor position is cached instead of queried every
frame.

Usage
-----
    python scripts/focus.py
    python scripts/focus.py -c /path/to/config.yaml --display-width 1200
    python scripts/focus.py --profile   # print live-view timing to stdout
"""

import argparse
import threading
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox

import amplitude
from amplitude.config import Config
from instruments.kinesismotor import KinesisMotor
from instruments.pythorcam.thorcam import CameraStream, ThorlabsCamera, create_camera_sdk
from instruments.pythorcam.utils import calculate_focus_measure

PKG_ROOT = Path(amplitude.__file__).resolve().parent  # .../holoeml-experiment/amplitude

METRIC_CROP_SIZE = 512     # px, central region used for the focus/brightness readout
METRIC_PERIOD_S = 0.1      # how often the background metric worker recomputes
PROFILE_PERIOD_S = 2.0     # how often --profile prints a timing summary


class FocusApp:
    """Live camera view + exposure/gain/focus-motor controls (matplotlib widgets)."""

    def __init__(self, camera: ThorlabsCamera, focus: KinesisMotor, config: Config,
                 display_width: int = 1000, profile: bool = False):
        self.camera = camera
        self.focus = focus
        self.config = config
        self.profile = profile

        H, W = self.camera.image_shape
        self.stride = max(1, W // max(display_width, 1))

        self.stream = CameraStream(camera)
        self._moving = False
        self._position = self._safe_get_position()
        self._focus_metric = 0.0
        self._brightness = 0.0

        self._bg = None            # cached blit background (set on first full draw)
        self.timer = None
        self._stop_metric = threading.Event()
        self._metric_thread = None

        if self.profile:
            self._profile_reset()

        self._build_figure()

    # ── figure / widgets ──────────────────────────────────────────────────

    def _build_figure(self) -> None:
        self.fig = plt.figure("Focus", figsize=(7.5, 9))
        self.ax_img = self.fig.add_axes([0.08, 0.40, 0.84, 0.56])
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])

        H, W = self.camera.image_shape
        blank = np.zeros((H // self.stride, W // self.stride))
        self.img = self.ax_img.imshow(blank, cmap='gray', vmin=0,
                                       vmax=self.camera.pixel_max_value,
                                       animated=True)

        self.overlay = self.ax_img.text(
            0.02, 0.98, "", transform=self.ax_img.transAxes,
            va='top', ha='left', fontsize=9, family='monospace', color='lime',
            bbox=dict(facecolor='black', alpha=0.5, pad=4), animated=True,
        )

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
                                  initial=f"{self._position:.4f}")
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
        self.fig.canvas.mpl_connect('draw_event', self._on_draw)

    # ── blitting ───────────────────────────────────────────────────────────

    def _on_draw(self, _event) -> None:
        """Cache the (widget-free) background after any full figure redraw.

        Fires on the initial draw, on window resize, and whenever a slider /
        text box triggers a full redraw -- so the cached background always
        matches what's currently on screen.
        """
        self._bg = self.fig.canvas.copy_from_bbox(self.ax_img.bbox)

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
            self._position = self._safe_get_position()
        except Exception as exc:
            print(f"[focus] motor move failed: {exc}")
        finally:
            self._moving = False

    # ── background metric worker ────────────────────────────────────────────

    def _central_crop(self, frame: np.ndarray) -> np.ndarray:
        H, W = frame.shape[:2]
        size = min(METRIC_CROP_SIZE, H, W)
        cy, cx = H // 2, W // 2
        half = size // 2
        return frame[cy - half:cy + half, cx - half:cx + half]

    def _metric_loop(self) -> None:
        while not self._stop_metric.is_set():
            frame = self.stream.get_latest_frame()
            if frame is not None:
                crop = self._central_crop(frame)
                self._focus_metric = calculate_focus_measure(crop)
                self._brightness = float(crop.mean())
            self._stop_metric.wait(METRIC_PERIOD_S)

    # ── live view loop ─────────────────────────────────────────────────────

    def _overlay_text(self) -> str:
        position = "moving..." if self._moving else f"{self._position:.4f} mm"
        return (f"position: {position}\n"
                f"focus:    {self._focus_metric:8.2f}\n"
                f"bright:   {self._brightness:8.1f}")

    def _on_tick(self) -> None:
        t0 = time.perf_counter()
        frame = self.stream.get_latest_frame()
        t1 = time.perf_counter()

        if frame is not None:
            disp = frame[::self.stride, ::self.stride, 0]
            self.img.set_data(disp)
            self.overlay.set_text(self._overlay_text())

            canvas = self.fig.canvas
            if self._bg is not None:
                canvas.restore_region(self._bg)
                self.ax_img.draw_artist(self.img)
                self.ax_img.draw_artist(self.overlay)
                canvas.blit(self.ax_img.bbox)
                canvas.flush_events()
            else:
                # No cached background yet (first frame before the initial
                # full draw has happened) -- fall back to a normal redraw.
                canvas.draw_idle()

        t2 = time.perf_counter()
        if self.profile:
            self._profile_record(frame is not None, t1 - t0, t2 - t1)

    # ── profiling ────────────────────────────────────────────────────────

    def _profile_reset(self) -> None:
        self._profile_last_print = time.perf_counter()
        self._profile_n_frames = 0
        self._profile_fetch_ms = []
        self._profile_draw_ms = []

    def _profile_record(self, got_frame: bool, fetch_s: float, draw_s: float) -> None:
        if got_frame:
            self._profile_n_frames += 1
            self._profile_fetch_ms.append(fetch_s * 1000)
            self._profile_draw_ms.append(draw_s * 1000)

        now = time.perf_counter()
        elapsed = now - self._profile_last_print
        if elapsed < PROFILE_PERIOD_S:
            return

        fps = self._profile_n_frames / elapsed
        avg_fetch = sum(self._profile_fetch_ms) / len(self._profile_fetch_ms) if self._profile_fetch_ms else 0.0
        avg_draw = sum(self._profile_draw_ms) / len(self._profile_draw_ms) if self._profile_draw_ms else 0.0
        msg = (f"[focus][profile] display={fps:5.1f} fps   "
               f"fetch={avg_fetch:5.2f} ms   draw={avg_draw:5.2f} ms   "
               f"focus_metric={self._focus_metric:.1f}")
        try:
            camera_fps = self.camera._camera.get_measured_frame_rate_fps()
            msg += f"   camera={camera_fps:.1f} fps"
        except Exception:
            pass
        print(msg)
        self._profile_reset()

    def _on_close(self, _event) -> None:
        if self.timer is not None:
            self.timer.stop()
        self._stop_metric.set()
        self.stream.stop()

    def run(self) -> None:
        self.stream.start()

        self._metric_thread = threading.Thread(target=self._metric_loop, daemon=True)
        self._metric_thread.start()

        self.timer = self.fig.canvas.new_timer(interval=33)  # ~30 fps target
        self.timer.add_callback(self._on_tick)
        self.timer.start()
        plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", type=Path, default=PKG_ROOT / "config/config.yaml")
    p.add_argument("--display-width", type=int, default=1000,
                   help="live view display width in pixels (frame is downsampled to it)")
    p.add_argument("--profile", action="store_true",
                   help="print live-view timing (fps, fetch/draw ms) to stdout")
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
                app = FocusApp(camera, focus, config,
                                display_width=args.display_width, profile=args.profile)
                app.run()
            finally:
                camera.disarm()
    finally:
        sdk.dispose()


if __name__ == "__main__":
    main()
