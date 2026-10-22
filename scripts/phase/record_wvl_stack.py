"""Record phase-shift stacks over a wavelength sweep, with arm-shutter frames.

Sequence:
  0. Once per run: both arm shutters closed -> n black frames (own file).
  For each wavelength (monochromator + filter wheel):
    1. Sample position:
       - both arms open -> piezo stack (dry run, n frames, dry run back,
         n_back return moves), `repeat` times
       - reference arm only (sample shutter closed) -> n frames
       - sample arm only (reference shutter closed) -> n frames
    2. Reference position (stage offset reference_x_by/reference_y_by):
       the same as 1, then the stage returns to the sample position.
    3. Save one NPZ for this wavelength.
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

DEFAULT_N = 16
DEFAULT_N_BACK = 10
DEFAULT_EXPOSURE_MS = 250.0
DEFAULT_GAIN = 0
DEFAULT_MOVE_PIEZO = True
DEFAULT_NAME = "WvlStack"
DEFAULT_REPEAT = 1
DEFAULT_WVL_START = 500.0
DEFAULT_WVL_STOP = 700.0
DEFAULT_WVL_STEP = 5.0
DEFAULT_OUT_BIT_DEPTH = "uint16"  # lossless for num_frames_to_average=1; 'float32' when averaging


def capture_frames(ctx, n):
    return np.asarray([ctx.capture() for _ in range(n)])


def acquire_piezo_stack(ctx, n, n_back, move_piezo, repeat, label):
    all_images = []
    for i in range(repeat):
        t0 = time.time()
        if move_piezo:
            ctx.dry_run(direction=1)
            images = ctx.acquire_stack(n, direction=1)
            ctx.dry_run(direction=-1)
            ctx.return_sweep(n_back)
        else:
            images = capture_frames(ctx, n)
        all_images.append(images)
        print(f"  [{label}] stack repeat {i+1}/{repeat} done, shape={images.shape}, "
              f"took {time.time()-t0:.2f}s")
    return np.stack(all_images)


def record_position(ctx, n, n_back, move_piezo, repeat, label):
    """Stack with both arms open, then reference-arm-only and sample-arm-only frames."""
    ctx.set_arms(ref=True, sample=True)
    stack = acquire_piezo_stack(ctx, n, n_back, move_piezo, repeat, label)
    ctx.set_arms(ref=True, sample=False)
    ref_arm = capture_frames(ctx, n)
    ctx.set_arms(ref=False, sample=True)
    sample_arm = capture_frames(ctx, n)
    print(f"  [{label}] ref-arm and sample-arm frames done, shape={ref_arm.shape}")
    return stack, ref_arm, sample_arm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help="Number of frames per stack, and of black / single-arm frames")
    parser.add_argument("--n-back", type=int, default=DEFAULT_N_BACK, help="Number of steps for piezo to return")
    parser.add_argument("--exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS, help="Camera's exposure time in ms")
    parser.add_argument("--gain", type=int, default=DEFAULT_GAIN, help="Camera's gain settings in 0.1 dB units")
    parser.add_argument("--move-piezo", action=argparse.BooleanOptionalAction, default=DEFAULT_MOVE_PIEZO,
                        help="Whether to move piezo motor after each frame")
    parser.add_argument("--name", type=str, default=DEFAULT_NAME, help="Name prefix of the saved files")
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT,
                        help="Number of times to repeat the stack at each position")
    parser.add_argument("--wvl-start", type=float, default=DEFAULT_WVL_START, help="First wavelength in nm")
    parser.add_argument("--wvl-stop", type=float, default=DEFAULT_WVL_STOP, help="Last wavelength in nm (inclusive)")
    parser.add_argument("--wvl-step", type=float, default=DEFAULT_WVL_STEP, help="Wavelength step in nm")
    parser.add_argument("--out-bit-depth", type=str, default=DEFAULT_OUT_BIT_DEPTH,
                        help="dtype of saved frames, e.g. 'uint16' or 'float32'")
    args = parser.parse_args()
    n = args.n
    n_back = args.n_back
    exposure_ms = args.exposure_ms
    gain = args.gain
    move_piezo = args.move_piezo
    name = args.name
    repeat = args.repeat
    out_bit_depth = np.dtype(args.out_bit_depth)
    num_wvl = int(np.floor((args.wvl_stop - args.wvl_start) / args.wvl_step + 1e-9)) + 1
    wavelengths = np.round(args.wvl_start + args.wvl_step * np.arange(num_wvl), 3)

    print(f"[config] n={n} n_back={n_back} exposure_ms={exposure_ms} gain={gain} "
          f"move_piezo={move_piezo} repeat={repeat} name={name!r} out_bit_depth={out_bit_depth}")
    print(f"[config] {len(wavelengths)} wavelengths: {wavelengths}")

    cfg = Config.from_yaml(PKG_ROOT / "config/config.yaml")
    if np.issubdtype(out_bit_depth, np.integer) and cfg.num_frames_to_average > 1:
        print(f"[warning] num_frames_to_average={cfg.num_frames_to_average} with integer "
              f"output {out_bit_depth}: averaged frames are truncated to whole counts")
    devices = ["camera", "kim", "mono", "filterwheel", "stage_x", "stage_y", "shutter"]

    with MeasurementContext.from_config(cfg, devices) as ctx:
        initial_arms = ctx.get_arms()
        try:
            # Set base parameters
            ctx.camera.set_exposure_ms(exposure_ms)
            ctx.camera.set_gain(gain)
            ctx.camera.set_out_bit_depth(out_bit_depth)
            ctx.apply_drive_parameters()
            ctx.apply_stage_velocities()
            ctx.init_mono()
            print("[setup] camera, drive, stage and monochromator ready")

            meta = dict(
                n=n,
                n_back=n_back,
                exposure_ms=exposure_ms,
                gain=gain,
                move_piezo=move_piezo,
                repeat=repeat,
                frames_to_average=cfg.num_frames_to_average,
                frames_to_drop=cfg.num_frames_to_drop,
                piezo_step_fwd=cfg.piezo_step_size_fwd,
                piezo_step_bwd=cfg.piezo_step_size_bwd,
                reference_x_by=cfg.reference_x_by,
                reference_y_by=cfg.reference_y_by,
            )

            with ctx.armed():
                print("[setup] context armed")

                # Black frames once per run: camera settings are fixed.
                ctx.set_arms(ref=False, sample=False)
                black = capture_frames(ctx, n)
                path = ctx.save_npz(f"{name}_black", black=black, wavelengths=wavelengths, **meta)
                print(f"[saved] black frames, shape={black.shape} -> {path.name}")

                for k, wvl in enumerate(wavelengths):
                    t0 = time.time()
                    print(f"[wvl {k+1}/{len(wavelengths)}] {wvl} nm")
                    ctx.set_wavelength(wvl)

                    stack_s, ref_arm_s, sample_arm_s = record_position(
                        ctx, n, n_back, move_piezo, repeat, "sample pos")
                    with ctx.stage_reference():
                        stack_r, ref_arm_r, sample_arm_r = record_position(
                            ctx, n, n_back, move_piezo, repeat, "ref pos")

                    path = ctx.save_npz(
                        f"{name}_{wvl:g}nm",
                        wavelength=wvl,
                        stack_sample_pos=stack_s,
                        ref_arm_sample_pos=ref_arm_s,
                        sample_arm_sample_pos=sample_arm_s,
                        stack_ref_pos=stack_r,
                        ref_arm_ref_pos=ref_arm_r,
                        sample_arm_ref_pos=sample_arm_r,
                        **meta,
                    )
                    print(f"[wvl {k+1}/{len(wavelengths)}] saved -> {path.name}, "
                          f"took {time.time()-t0:.1f}s")
            print("[done] all wavelengths complete")
        finally:
            ctx.set_arms(*initial_arms)
            ctx.set_wavelength(cfg.default_wavelength)
            print(f"[cleanup] arm shutters restored, monochromator at {cfg.default_wavelength} nm")


if __name__ == "__main__":
    main()
