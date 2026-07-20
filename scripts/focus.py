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
crop around the current view, and the motor position is cached instead of
queried every frame.

A Zoom slider digitally magnifies the live view in real time; clicking
anywhere in the image recenters the zoom on that point (e.g. to inspect an
off-center feature). The focus/brightness readout tracks whatever region is
currently zoomed/centered on.

A red dashed crosshair (independent H-line / V-line position sliders) helps
align the angle of a structure -- e.g. rotate the sample until an edge is
parallel to the horizontal line. The crosshair position is defined in
axes-fraction space, so it stays correctly placed at any zoom/pan level.

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

        self.frame_h, self.frame_w = self.camera.image_shape
        self.display_width = max(display_width, 1)
        self.stride = max(1, self.frame_w // self.display_width)  # used for the initial blank image

        self.stream = CameraStream(camera)
        self._moving = False
        self._position = self._safe_get_position()
        self._focus_metric = 0.0
        self._brightness = 0.0

        # Zoom / pan state (driven by the Zoom slider and clicks on the image).
        self.zoom = 1.0
        self.center_frac = (0.5, 0.5)     # zoom center, as a fraction of the full frame
        self._crop = (0, 0, self.stride)  # (x0, y0, stride) of the crop shown last tick
        self._last_disp_shape = None      # triggers one full redraw on the first frame

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
        self.ax_img = self.fig.add_axes([0.08, 0.42, 0.84, 0.54])
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])

        blank = np.zeros((self.frame_h // self.stride, self.frame_w // self.stride))
        self.img = self.ax_img.imshow(blank, cmap='gray', vmin=0,
                                       vmax=self.camera.pixel_max_value,
                                       interpolation='nearest', animated=True)
        self._last_disp_shape = blank.shape

        # Crosshair -- drawn in AXES-FRACTION coordinates (transAxes), not data
        # coordinates, so it stays correctly positioned at any zoom/pan level
        # with no need to route it through _crop_bounds like the image is.
        self.hline = self.ax_img.plot([0, 1], [0.5, 0.5], transform=self.ax_img.transAxes,
                                       color='red', lw=1, ls='--', alpha=0.8,
                                       animated=True)[0]
        self.vline = self.ax_img.plot([0.5, 0.5], [0, 1], transform=self.ax_img.transAxes,
                                       color='red', lw=1, ls='--', alpha=0.8,
                                       animated=True)[0]

        self.overlay = self.ax_img.text(
            0.02, 0.98, "", transform=self.ax_img.transAxes,
            va='top', ha='left', fontsize=9, family='monospace', color='lime',
            bbox=dict(facecolor='black', alpha=0.5, pad=4), animated=True,
        )

        # Exposure slider
        ax_exp = self.fig.add_axes([0.2, 0.37, 0.6, 0.03])
        self.sl_exposure = Slider(
            ax_exp, 'Exposure (ms)',
            valmin=0.03, valmax=max(self.config.calib_max_exposure_ms, 1),
            valinit=self.config.calib_initial_exposure_ms,
        )
        self.sl_exposure.on_changed(self._on_exposure_changed)

        # Gain slider (SDK units are tenths of a dB)
        ax_gain = self.fig.add_axes([0.2, 0.32, 0.6, 0.03])
        self.sl_gain = Slider(
            ax_gain, 'Gain (dB)',
            valmin=0, valmax=48, valinit=self.config.calib_initial_gain / 10,
        )
        self.sl_gain.on_changed(self._on_gain_changed)

        # Zoom slider -- digitally magnifies the live view; click the image to recenter.
        ax_zoom = self.fig.add_axes([0.2, 0.27, 0.6, 0.03])
        self.sl_zoom = Slider(ax_zoom, 'Zoom (x)', valmin=1.0, valmax=16.0, valinit=self.zoom)
        self.sl_zoom.on_changed(self._on_zoom_changed)

        # Crosshair position sliders (0-100% of the current view, independent of zoom)
        ax_hline = self.fig.add_axes([0.2, 0.22, 0.6, 0.03])
        self.sl_hline = Slider(ax_hline, 'H-line pos (%)', valmin=0, valmax=100, valinit=50)
        self.sl_hline.on_changed(self._on_hline_changed)

        ax_vline = self.fig.add_axes([0.2, 0.17, 0.6, 0.03])
        self.sl_vline = Slider(ax_vline, 'V-line pos (%)', valmin=0, valmax=100, valinit=50)
        self.sl_vline.on_changed(self._on_vline_changed)

        # move_to
        ax_moveto = self.fig.add_axes([0.2, 0.10, 0.35, 0.045])
        self.tb_moveto = TextBox(ax_moveto, 'move_to (mm)   ',
                                  initial=f"{self._position:.4f}")
        self.tb_moveto.on_submit(self._submit_move_to)
        ax_go = self.fig.add_axes([0.58, 0.10, 0.12, 0.045])
        self.btn_go = Button(ax_go, 'Go')
        self.btn_go.on_clicked(lambda _event: self._submit_move_to(self.tb_moveto.text))

        # move_by
        ax_step = self.fig.add_axes([0.2, 0.03, 0.35, 0.045])
        self.tb_step = TextBox(ax_step, 'move_by step (mm)', initial='0.001')
        ax_minus = self.fig.add_axes([0.58, 0.03, 0.12, 0.045])
        self.btn_minus = Button(ax_minus, '-')
        self.btn_minus.on_clicked(lambda _event: self._submit_move_by(-1))
        ax_plus = self.fig.add_axes([0.72, 0.03, 0.12, 0.045])
        self.btn_plus = Button(ax_plus, '+')
        self.btn_plus.on_clicked(lambda _event: self._submit_move_by(+1))

        self.fig.canvas.mpl_connect('close_event', self._on_close)
        self.fig.canvas.mpl_connect('draw_event', self._on_draw)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

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

    # ── zoom / pan ───────────────────────────────────────────────────────

    def _on_zoom_changed(self, value: float) -> None:
        self.zoom = value

    def _on_hline_changed(self, value: float) -> None:
        self.hline.set_ydata([value / 100, value / 100])

    def _on_vline_changed(self, value: float) -> None:
        self.vline.set_xdata([value / 100, value / 100])

    def _on_click(self, event) -> None:
        """Recenter the zoom on wherever the user clicks inside the image."""
        if event.inaxes is not self.ax_img or event.xdata is None or event.ydata is None:
            return  # click landed on a slider/button/textbox, or outside any axes
        x0, y0, stride = self._crop
        full_x = x0 + event.xdata * stride
        full_y = y0 + event.ydata * stride
        cx = min(max(full_x / self.frame_w, 0.0), 1.0)
        cy = min(max(full_y / self.frame_h, 0.0), 1.0)
        self.center_frac = (cx, cy)

    def _crop_bounds(self, crop_w: int, crop_h: int) -> tuple[int, int, int, int]:
        """Center a ``crop_w x crop_h`` window on ``self.center_frac``, clamped to the frame."""
        crop_w = min(crop_w, self.frame_w)
        crop_h = min(crop_h, self.frame_h)
        cx = self.center_frac[0] * self.frame_w
        cy = self.center_frac[1] * self.frame_h
        x0 = int(round(cx - crop_w / 2))
        y0 = int(round(cy - crop_h / 2))
        x0 = max(0, min(x0, self.frame_w - crop_w))
        y0 = max(0, min(y0, self.frame_h - crop_h))
        return x0, y0, crop_w, crop_h

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

    def _metric_window(self, frame: np.ndarray) -> np.ndarray:
        """Crop used for the focus/brightness readout: tracks the current zoom/pan,
        capped at ``METRIC_CROP_SIZE`` so it stays cheap even at zoom=1x."""
        view_w = max(1, int(self.frame_w / self.zoom))
        view_h = max(1, int(self.frame_h / self.zoom))
        size_w = min(METRIC_CROP_SIZE, view_w)
        size_h = min(METRIC_CROP_SIZE, view_h)
        x0, y0, size_w, size_h = self._crop_bounds(size_w, size_h)
        return frame[y0:y0 + size_h, x0:x0 + size_w]

    def _metric_loop(self) -> None:
        while not self._stop_metric.is_set():
            frame = self.stream.get_latest_frame()
            if frame is not None:
                crop = self._metric_window(frame)
                self._focus_metric = calculate_focus_measure(crop)
                self._brightness = float(crop.mean())
            self._stop_metric.wait(METRIC_PERIOD_S)

    # ── live view loop ─────────────────────────────────────────────────────

    def _overlay_text(self) -> str:
        position = "moving..." if self._moving else f"{self._position:.4f} mm"
        return (f"position: {position}   zoom: {self.zoom:4.1f}x\n"
                f"focus:    {self._focus_metric:8.2f}\n"
                f"bright:   {self._brightness:8.1f}")

    def _on_tick(self) -> None:
        t0 = time.perf_counter()
        frame = self.stream.get_latest_frame()
        t1 = time.perf_counter()

        if frame is not None:
            crop_w = max(1, int(self.frame_w / self.zoom))
            crop_h = max(1, int(self.frame_h / self.zoom))
            x0, y0, crop_w, crop_h = self._crop_bounds(crop_w, crop_h)
            stride = max(1, crop_w // self.display_width)
            disp = frame[y0:y0 + crop_h:stride, x0:x0 + crop_w:stride, 0]
            self._crop = (x0, y0, stride)

            self.img.set_data(disp)
            self.overlay.set_text(self._overlay_text())

            canvas = self.fig.canvas
            if disp.shape != self._last_disp_shape:
                # Zoom level changed -> the displayed array size changed. set_data()
                # does NOT update imshow's own extent, so it must be updated
                # explicitly (otherwise the new, differently-shaped array gets
                # mapped onto the old data box, letterboxing the image) along
                # with the axes view limits. Do a synchronous full draw so the
                # blit background (cached via _on_draw / 'draw_event') is
                # guaranteed correct before the next tick tries to blit against
                # it -- avoids a stale-background glitch a deferred draw_idle()
                # could leave for one frame.
                extent = (-0.5, disp.shape[1] - 0.5, disp.shape[0] - 0.5, -0.5)
                self.img.set_extent(extent)
                self.ax_img.set_xlim(extent[0], extent[1])
                self.ax_img.set_ylim(extent[2], extent[3])
                self._last_disp_shape = disp.shape
                canvas.draw()
            elif self._bg is not None:
                canvas.restore_region(self._bg)
                self.ax_img.draw_artist(self.img)
                self.ax_img.draw_artist(self.hline)
                self.ax_img.draw_artist(self.vline)
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
