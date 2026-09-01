import numpy as np
from pathlib import Path
import argparse
import time

import phasectl
from phasectl.config import Config
from phasectl.measurements.context import MeasurementContext

PKG_ROOT = Path(phasectl.__file__).resolve().parent  # .../holoeml-experiment/phasectl

DEFAULT_N = 5
DEFAULT_N_BACK = 3
DEFAULT_EXPOSURE_MS = 250.0
DEFAULT_GAIN = 100
DEFAULT_MOVE_PIEZO = True
DEFAULT_NAME = "Stack"
DEFAULT_REPEAT = 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of frames")
    parser.add_argument("--n-back", type=int, default=DEFAULT_N_BACK, help="Number of steps for piezo to return")
    parser.add_argument("--exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS, help="Camera's exposure time in ms")
    parser.add_argument("--gain", type=int, default=DEFAULT_GAIN, help="Camera's gain settings in 0.1 dB units")
    parser.add_argument("--move-piezo", action=argparse.BooleanOptionalAction, default=DEFAULT_MOVE_PIEZO,
                        help="Whether to move piezo motor after each frame")
    parser.add_argument("--name", type=str, default=DEFAULT_NAME, help="Name of the saving file")
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT,
                        help="Number of time to repeat same measurement")
    args = parser.parse_args()
    n = args.n
    n_back = args.n_back
    exposure_ms = args.exposure_ms
    gain = args.gain
    move_piezo = args.move_piezo
    name = args.name
    repeat = args.repeat
    
    print(f"[config] n={n} n_back={n_back} exposure_ms={exposure_ms} gain={gain} "
        f"move_piezo={move_piezo} repeat={repeat} name={name!r}")
    
    cfg = Config.from_yaml(PKG_ROOT / "config/config.yaml")
    devices = ["camera", "kim"]
    ctx = MeasurementContext.from_config(cfg, devices)
    
    # Set base parameters
    ctx.camera.set_exposure_ms(exposure_ms)
    ctx.camera.set_gain(gain)
    ctx.apply_drive_parameters()
    print("[setup] camera and drive parameters applied")
    
    with ctx.armed():
        print("[setup] context armed")
        all_images = []
        for i in range(repeat):
            t0 = time.time()
            print(f"[repeat {i+1}/{repeat}] starting acquisition...")
            if move_piezo:
                ctx.dry_run(direction=1)
                images = ctx.acquire_stack(n, direction=1)
                ctx.dry_run(direction=-1)
                ctx.return_sweep(n_back)
            else:
                images = []
                for _ in range(n):
                    images.append(ctx.capture())
                images = np.asarray(images)
            all_images.append(images)
            print(f"[repeat {i+1}/{repeat}] done, shape={images.shape}, took {time.time()-t0:.2f}s")
            
        all_images = np.stack(all_images)
        print(f"[done] all repeats complete, final stack shape={all_images.shape}")
    
    ctx.save_npz(
        name,
        stack=all_images,
        n=n,
        n_back=n_back,
        exposure_ms=exposure_ms,
        gain=gain,
        move_piezo=move_piezo,
        frames_to_average=cfg.num_frames_to_average,
        frames_to_drop=cfg.num_frames_to_drop,
        piezo_step_fwd=cfg.piezo_step_size_fwd,
        piezo_step_bwd=cfg.piezo_step_size_bwd,
    )
    print(f"[saved] {name}")
            
if __name__ == "__main__":
    main()