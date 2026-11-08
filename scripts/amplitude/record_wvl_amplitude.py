"""Record sample-arm amplitude frames over a wavelength sweep.

Sequence (reference arm shutter closed for the whole run):
  0. Both arm shutters closed -> n black frames (own file).
  For each wavelength (monochromator + filter wheel), sample arm open:
    1. Sample position -> n frames.
    2. Reference position (stage offset reference_x_by/reference_y_by)
       -> n frames, then the stage returns to the sample position.
    3. Save one NPZ for this wavelength.
  4. Both arm shutters closed -> n black frames again (own file).
The n frames at each step are averaged; the per-pixel mean and std are saved.
Camera exposure/gain are fixed for the whole run.
"""
import numpy as np
from pathlib import Path
import argparse
import time

import phasectl
from phasectl.config import Config
from phasectl.measurements.context import MeasurementContext

PKG_ROOT = Path(phasectl.__file__).resolve().parent  # .../holoeml-experiment/phasectl

DEFAULT_N = 10
DEFAULT_EXPOSURE_MS = 250.0
DEFAULT_GAIN = 0
DEFAULT_NAME = "WvlAmplitude"
DEFAULT_WVL_START = 500.0
DEFAULT_WVL_STOP = 700.0
DEFAULT_WVL_STEP = 5.0


def capture_mean_std(ctx, n):
    """Capture n frames and return their per-pixel mean and std (float32)."""
    frames = np.asarray([ctx.capture() for _ in range(n)], dtype=np.float64)
    return frames.mean(axis=0).astype(np.float32), frames.std(axis=0).astype(np.float32)


def record_black(ctx, n, name, wavelengths, meta, label):
    ctx.set_arms(ref=False, sample=False)
    black, black_std = capture_mean_std(ctx, n)
    path = ctx.save_npz(f"{name}_black_{label}", black=black, black_std=black_std,
                        wavelengths=wavelengths, **meta)
    print(f"[saved] black frames ({label}), shape={black.shape} -> {path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help="Number of frames averaged per position and for black frames")
    parser.add_argument("--exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS, help="Camera's exposure time in ms")
    parser.add_argument("--gain", type=int, default=DEFAULT_GAIN, help="Camera's gain settings in 0.1 dB units")
    parser.add_argument("--name", type=str, default=DEFAULT_NAME, help="Name prefix of the saved files")
    parser.add_argument("--wvl-start", type=float, default=DEFAULT_WVL_START, help="First wavelength in nm")
    parser.add_argument("--wvl-stop", type=float, default=DEFAULT_WVL_STOP, help="Last wavelength in nm (inclusive)")
    parser.add_argument("--wvl-step", type=float, default=DEFAULT_WVL_STEP, help="Wavelength step in nm")
    args = parser.parse_args()
    n = args.n
    exposure_ms = args.exposure_ms
    gain = args.gain
    name = args.name
    num_wvl = int(np.floor((args.wvl_stop - args.wvl_start) / args.wvl_step + 1e-9)) + 1
    wavelengths = np.round(args.wvl_start + args.wvl_step * np.arange(num_wvl), 3)

    print(f"[config] n={n} exposure_ms={exposure_ms} gain={gain} name={name!r}")
    print(f"[config] {len(wavelengths)} wavelengths: {wavelengths}")

    cfg = Config.from_yaml(PKG_ROOT / "config/config.yaml")
    devices = ["camera", "mono", "filterwheel", "stage_x", "stage_y", "shutter"]

    with MeasurementContext.from_config(cfg, devices) as ctx:
        initial_arms = ctx.get_arms()
        try:
            # Set base parameters
            ctx.camera.set_exposure_ms(exposure_ms)
            ctx.camera.set_gain(gain)
            ctx.camera.set_out_bit_depth(np.dtype("uint16"))
            ctx.apply_stage_velocities()
            ctx.init_mono()
            print("[setup] camera, stage and monochromator ready")

            meta = dict(
                n=n,
                exposure_ms=exposure_ms,
                gain=gain,
                frames_to_average=cfg.num_frames_to_average,
                frames_to_drop=cfg.num_frames_to_drop,
                reference_x_by=cfg.reference_x_by,
                reference_y_by=cfg.reference_y_by,
            )

            with ctx.armed():
                print("[setup] context armed")

                record_black(ctx, n, name, wavelengths, meta, "start")

                # Reference arm stays closed; only the sample arm is used.
                ctx.set_arms(ref=False, sample=True)
                for k, wvl in enumerate(wavelengths):
                    t0 = time.time()
                    print(f"[wvl {k+1}/{len(wavelengths)}] {wvl} nm")
                    ctx.set_wavelength(wvl)

                    sample_pos, sample_pos_std = capture_mean_std(ctx, n)
                    with ctx.stage_reference():
                        ref_pos, ref_pos_std = capture_mean_std(ctx, n)

                    path = ctx.save_npz(
                        f"{name}_{wvl:g}nm",
                        wavelength=wvl,
                        sample_pos=sample_pos,
                        sample_pos_std=sample_pos_std,
                        ref_pos=ref_pos,
                        ref_pos_std=ref_pos_std,
                        **meta,
                    )
                    print(f"[wvl {k+1}/{len(wavelengths)}] saved -> {path.name}, "
                          f"took {time.time()-t0:.1f}s")

                record_black(ctx, n, name, wavelengths, meta, "end")
            print("[done] all wavelengths complete")
        finally:
            ctx.set_arms(*initial_arms)
            ctx.set_wavelength(cfg.default_wavelength)
            print(f"[cleanup] arm shutters restored, monochromator at {cfg.default_wavelength} nm")


if __name__ == "__main__":
    main()
