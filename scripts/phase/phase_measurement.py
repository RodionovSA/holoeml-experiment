""" A script for a full phase measurement at a fixed wavelength and polarization"""

import numpy as np
import argparse
from pathlib import Path
import contextlib
import concurrent.futures
import time
from typing import Tuple
from datetime import datetime
from dataclasses import asdict

import instruments.config
from instruments.camera import Camera, open_camera
from instruments.kinesismotor import KinesisMotor
from instruments.inertialpiezo import KIM101, KIM101Axis
from instruments.config import EquipmentConfig
from phase import aia, remove_carrier, subtract_reference
from phase.backend import asnumpy, CUPY_AVAILABLE
from phase.combine import combine_acquisitions, CombinedResult

NUM_PIEZO_STEPS = 20
STEP_SIZE = 5
SETTLE_S = 0.05  # pause after each step before capturing (mechanical settling)
DRY_NUM = 10
DRY_STEP = 1

DRIVE_STEP_RATE = 200          # steps/s
DRIVE_STEP_ACCELERATION = 500 # steps/s^2
DRIVE_MAX_VOLTAGE = 125       # V

NUM_AVERAGES = 2
REPEATS = 1  # number of independent sample+reference measurements per invocation
MAX_DATASET_GB = 4.0  # pre-flight abort threshold for --repeats
EXPOSURE_MS = 250
GAIN = 100
REFERENCE_X_BY = 0.0 # mm
REFERENCE_Y_BY = 0.8 # mm
IMAGE_SHAPE_X0 = 300 
IMAGE_SHAPE_Y0 = 400 
IMAGE_SHAPE_X1 = 3600 
IMAGE_SHAPE_Y1 = 2600 

CONFIG_ROOT = Path(instruments.config.__file__).resolve().parent

@contextlib.contextmanager
def _armed_camera(camera):
    camera.arm()
    time.sleep(0.1)
    try:
        yield
    finally:
        camera.disarm()
        
def _acquire_repeat(camera: Camera,
                     piezo: KIM101Axis,
                     step_size: int,
                     num_piezo_steps: int) -> np.ndarray:
    """One phase-shift sweep's worth of hardware I/O for a single repeat:
    dry-run backlash compensation, the frame-capture loop, dry-run back, and
    return to the start position. Kept on the caller's thread -- the
    piezo/camera driver calls aren't meant to be used concurrently -- while
    `measure_phase` overlaps this with the (CPU/GPU-bound) `aia` solve of
    the *previous* repeat in a background thread instead.
    """

    images = []
    for step in range(num_piezo_steps):
        images.append(camera.get_image()[IMAGE_SHAPE_Y0:IMAGE_SHAPE_Y1,
                                         IMAGE_SHAPE_X0:IMAGE_SHAPE_X1, 0])
        if step < num_piezo_steps - 1:
            piezo.move_by(step_size)
            time.sleep(SETTLE_S)

    images = np.asarray(images)

    return images

def _release_gpu_memory():
    """Return CuPy's pooled (cached, not live) VRAM to the driver.

    The K2200 also drives the display, and its allocator pool plateaus around
    2.9 of the card's 4 GB across repeated AIA solves. Called once per
    `measure_phase`, after results are already copied to the host -- the next
    solve just re-allocates, hidden behind the ~9 s of hardware time per
    repeat.
    """
    if CUPY_AVAILABLE:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()

def _process_repeat(images: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, float, float]:
    """Run AIA on one repeat's frame stack. Meant to be submitted to a
    background thread (see `measure_phase`) so it overlaps with the next
    repeat's hardware acquisition rather than leaving the piezo/camera idle
    for the whole solve."""
    res = aia(images, gain="auto", iters=60, device=device)
    print(f"Converged: {res.converged}")
    print(f"Kappa_p: {res.kappa_p}")
    print(f"Kappa_ps: {res.kappa_ps}")
    return res.a, res.b, res.phi, res.converged, res.kappa_p, res.kappa_ps

def measure_phase(camera: Camera,
                  piezo: KIM101Axis,
                  step_size: int,
                  num_piezo_steps: int,
                  num_averages: int,
                  device: str = "auto") -> Tuple[CombinedResult, np.ndarray, np.ndarray, np.ndarray]:
    """Acquire `num_averages` independent repeats and combine their AIA phase.

    Each repeat's `aia` solve (`_process_repeat`) runs in a background
    thread while the *next* repeat's dry-run + sweep (`_acquire_repeat`)
    proceeds on this thread, instead of the piezo/camera sitting idle for
    the whole solve. A single-worker pool is enough since only one solve is
    ever in flight; with `device="cuda"` that solve becomes fast enough to
    mostly hide behind the ~(2*DRY_NUM + num_piezo_steps)*SETTLE_S seconds
    of hardware time each repeat already takes.

    Returns the combined-repeat result alongside the per-repeat AIA
    diagnostics (`converged`, `kappa_p`, `kappa_ps`), each shape
    `(num_averages,)`, so callers can judge whether `num_piezo_steps` was
    high enough for every repeat to actually converge.
    """

    # dry piezo run forward
    for _ in range(DRY_NUM):
        piezo.move_by(DRY_STEP)
        time.sleep(SETTLE_S)

    phi_list, b_list = [], []
    converged_list, kappa_p_list, kappa_ps_list = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pending = None
        for _ in range(num_averages):
            images = _acquire_repeat(camera, piezo, step_size, num_piezo_steps)
            if pending is not None:
                _, prev_b, prev_phi, prev_conv, prev_kp, prev_kps = pending.result()
                b_list.append(prev_b)
                phi_list.append(prev_phi)
                converged_list.append(prev_conv)
                kappa_p_list.append(prev_kp)
                kappa_ps_list.append(prev_kps)
            pending = pool.submit(_process_repeat, images, device)
        _, last_b, last_phi, last_conv, last_kp, last_kps = pending.result()
        b_list.append(last_b)
        phi_list.append(last_phi)
        converged_list.append(last_conv)
        kappa_p_list.append(last_kp)
        kappa_ps_list.append(last_kps)

    b = np.asarray([asnumpy(x) for x in b_list])
    phi = np.asarray([asnumpy(x) for x in phi_list])
    b_list.clear()
    phi_list.clear()
    _release_gpu_memory()

    res = combine_acquisitions(phi, weights=b, device=device)
    if res.sign_flips:
        print(f"Sign flips among repeats: {res.sign_flips}")

    # dry piezo run backward
    for _ in range(DRY_NUM):
        piezo.move_by(-DRY_STEP)
        time.sleep(SETTLE_S)

    # piezo back with same steps
    for _ in range(num_piezo_steps*num_averages):
        piezo.move_by(-step_size)
        time.sleep(SETTLE_S)

    return (res,
            np.asarray(converged_list),
            np.asarray(kappa_p_list),
            np.asarray(kappa_ps_list))

def _run_once(camera: Camera,
               piezo: KIM101Axis,
               stage_x: KinesisMotor,
               stage_y: KinesisMotor,
               step_size: int,
               num_piezo_steps: int,
               num_averages: int,
               device: str) -> dict:
    """One full sample+reference measurement: arm the camera, measure the
    sample, move to the reference position and measure it, move back, then
    subtract and remove the carrier. Stage positions are read at the start
    and restored at the end, so repeated calls all start from -- and return
    to -- the same nominal state.

    Returns a dict of host-side (numpy, not cupy) values for this one run,
    ready to be stacked across runs and passed to `np.savez`.
    """
    init_x_pos = stage_x.get_position()
    init_y_pos = stage_y.get_position()
    start = time.time()
    with _armed_camera(camera):
        sample_res, sample_converged, sample_kappa_p, sample_kappa_ps = measure_phase(
            camera, piezo, step_size, num_piezo_steps, num_averages, device=device)
        stage_x.move_by(REFERENCE_X_BY)
        stage_y.move_by(REFERENCE_Y_BY)

        reference_res, reference_converged, reference_kappa_p, reference_kappa_ps = measure_phase(
            camera, piezo, step_size, num_piezo_steps, num_averages, device=device)

    stage_x.move_to(init_x_pos)
    stage_y.move_to(init_y_pos)

    # a pixel unreliable in *either* acquisition is unreliable in the
    # difference -- weight both the sign resolution and carrier fit by the
    # joint reliability rather than the sample's alone.
    weight = sample_res.mean_resultant * reference_res.mean_resultant

    diff = subtract_reference(sample_res.phi, reference_res.phi, weight, device=device)
    print(f"Sign branch: sign={diff.sign:+d} ambiguous={diff.ambiguous} "
          f"spread_same={diff.spread_same:.4f} spread_flipped={diff.spread_flipped:.4f}")
    carrier_res = remove_carrier(diff.phi, weight,
                                defocus=True, refine_iters=10, n_blocks=10, device=device)
    end = time.time()

    # bring array-valued fields back to the host explicitly -- np.savez
    # can't serialize a cupy array, and dataclasses.asdict won't do that
    # transfer on its own.
    out = asdict(carrier_res)
    out["phi"] = asnumpy(out["phi"]).astype(np.float32)
    out["weight"] = asnumpy(weight).astype(np.float32)
    out["sign"] = diff.sign
    out["ambiguous"] = diff.ambiguous
    out["spread_same"] = diff.spread_same
    out["spread_flipped"] = diff.spread_flipped
    out["time"] = end - start
    # within-run (across num_averages repeats) scatter of the sample
    # acquisition, as a single scalar figure of merit -- the per-pixel
    # scatter map itself is not kept, to keep the file size down.
    out["within_run_scatter"] = float(np.median(asnumpy(sample_res.scatter)))
    out["converged"] = np.stack([asnumpy(sample_converged), asnumpy(reference_converged)])
    out["kappa_p"] = np.stack([asnumpy(sample_kappa_p), asnumpy(reference_kappa_p)])
    out["kappa_ps"] = np.stack([asnumpy(sample_kappa_ps), asnumpy(reference_kappa_ps)])
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Where to run the AIA/carrier/reference solves "
                             "(see phase.backend) -- 'auto' uses a GPU if one "
                             "is installed and available, else the CPU.")
    parser.add_argument("--num_averages", type=int, default=NUM_AVERAGES,
                        help="Number of independent repeats to acquire and "
                             "combine per phase measurement.")
    parser.add_argument("--num_piezo_steps", type=int, default=NUM_PIEZO_STEPS,
                        help="Number of piezo phase-shift steps per repeat.")
    parser.add_argument("--repeats", type=int, default=REPEATS,
                        help="Number of independent sample+reference measurements "
                             "to run back-to-back and combine into one .npz -- "
                             "use this to judge run-to-run phase fluctuation "
                             "instead of comparing separate files by hand.")
    args = parser.parse_args()
    device = args.device
    num_averages = args.num_averages
    num_piezo_steps = args.num_piezo_steps
    repeats = args.repeats

    # Pre-flight size check -- phi_runs + weight_runs are the two big arrays
    # in the saved file, both float32 (H, W) per repeat -- so we can size
    # the dataset before touching any hardware.
    H = IMAGE_SHAPE_Y1 - IMAGE_SHAPE_Y0
    W = IMAGE_SHAPE_X1 - IMAGE_SHAPE_X0
    bytes_per_run = 2 * H * W * np.dtype(np.float32).itemsize
    estimated_gb = repeats * bytes_per_run / 1e9
    print(f"Planned batch: {repeats} repeat(s) of {H}x{W} float32 phi+weight "
          f"-> ~{estimated_gb:.2f} GB")
    if estimated_gb > MAX_DATASET_GB:
        raise RuntimeError(
            f"Planned dataset (~{estimated_gb:.2f} GB) exceeds MAX_DATASET_GB "
            f"({MAX_DATASET_GB} GB) for --repeats {repeats}. Reduce --repeats "
            f"or raise MAX_DATASET_GB."
        )

    cfg = EquipmentConfig.from_yaml(CONFIG_ROOT / "config.yaml")

    with contextlib.ExitStack() as stack:
        # Init devices
        camera = open_camera(cfg, stack)
        stack.enter_context(camera)
        kim = stack.enter_context(KIM101(cfg.kim_serial))
        piezo = kim.axis(cfg.kim_channel)
        stage_x = stack.enter_context(KinesisMotor(cfg.stage_x_serial))
        stage_y = stack.enter_context(KinesisMotor(cfg.stage_y_serial))

        # Set settings
        camera.set_exposure_ms(EXPOSURE_MS)
        if camera.supports_gain:
            camera.set_gain(GAIN)
        piezo.set_drive_parameters(
            step_rate=DRIVE_STEP_RATE,
            step_acceleration=DRIVE_STEP_ACCELERATION,
            max_voltage=DRIVE_MAX_VOLTAGE,
        )
        applied_drive_params = piezo.get_drive_parameters()
        print(f"Drive parameters applied: {applied_drive_params}")
        stage_x.set_velocity(max_velocity=cfg.default_stage_max_velocity,
                            acceleration=cfg.default_stage_acceleration)
        stage_y.set_velocity(max_velocity=cfg.default_stage_max_velocity,
                            acceleration=cfg.default_stage_acceleration)

        # Run repeats back-to-back on this one hardware session. A failure
        # partway through still leaves the completed runs savable, rather
        # than losing the whole batch -- mirrors kim_step_calibration.py.
        runs = []
        error: Exception | None = None
        batch_start = time.time()
        try:
            for run_idx in range(repeats):
                run_out = _run_once(camera, piezo, stage_x, stage_y,
                                     STEP_SIZE, num_piezo_steps, num_averages, device)
                runs.append(run_out)
                print(f"run {run_idx + 1}/{repeats}: sign={run_out['sign']:+d} "
                      f"within_run_scatter={np.degrees(run_out['within_run_scatter']):.4f} deg "
                      f"time={run_out['time']:.2f}s")
        except Exception as exc:
            error = exc
            print(f"ERROR during acquisition: {exc!r}")
            print(f"Completed {len(runs)}/{repeats} runs before the failure -- "
                  f"saving partial data.")
    batch_end = time.time()
    print(f"Batch of {len(runs)}/{repeats} run(s) completed in "
          f"{batch_end - batch_start:.2f} seconds")

    if not runs:
        if error is not None:
            raise error
        raise RuntimeError("No runs completed -- nothing to save.")

    n = len(runs)
    phi_runs = np.stack([r["phi"] for r in runs]).astype(np.float32)
    weight_runs = np.stack([r["weight"] for r in runs]).astype(np.float32)

    if n >= 2:
        across = combine_acquisitions(phi_runs, weights=weight_runs, device=device)
        if across.sign_flips:
            print(f"Sign flips among runs: {across.sign_flips}")
        phi = asnumpy(across.phi).astype(np.float32)
        scatter = asnumpy(across.scatter).astype(np.float32)
        mean_resultant = asnumpy(across.mean_resultant).astype(np.float32)
        sign_flips = np.asarray(across.sign_flips, dtype=np.int64)
        _release_gpu_memory()

        # figure of merit: weighted mean across-run scatter (degrees),
        # restricted to the more reliable half of the frame so dim/unreliable
        # pixels don't dominate the comparison between parameter settings.
        weight_mean = weight_runs.mean(axis=0)
        mask = weight_mean >= np.median(weight_mean)
        fom_scatter_deg = float(np.degrees(
            np.average(scatter[mask], weights=mean_resultant[mask])))
    else:
        # only one run -- there is no across-run scatter to compute; fall
        # back to the within-run figure as the best available estimate.
        phi = phi_runs[0]
        scatter = np.zeros_like(phi)
        mean_resultant = weight_runs[0]
        sign_flips = np.asarray([], dtype=np.int64)
        fom_scatter_deg = float(np.degrees(runs[0]["within_run_scatter"]))

    print(f"Figure of merit (across-run scatter): {fom_scatter_deg:.4f} deg")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path(__file__).resolve().parent / f"phase_measurement_{timestamp}.npz"
    description = (
        "Phase measurement batch. 'phi' is the across-run combined phase map "
        "(drop-in replacement for the single-run 'phi' of earlier versions of "
        "this script). 'phi_runs'/'weight_runs' are the per-run carrier-removed "
        "phase and joint reliability, letting the across-run combine be redone "
        "offline. 'fom_scatter_deg' is the weighted-mean across-run scatter in "
        "degrees, restricted to the more reliable half of the frame -- the "
        "single number to compare between parameter settings. All other "
        "per-run fields ('sign', 'kx', 'converged', ...) have shape (repeats,), "
        "except 'converged'/'kappa_p'/'kappa_ps' which are "
        "(repeats, 2, num_averages) with axis 1 = sample/reference."
    )
    out = {
        "phi": phi,
        "phi_runs": phi_runs,
        "weight_runs": weight_runs,
        "scatter": scatter,
        "mean_resultant": mean_resultant,
        "sign_flips": sign_flips,
        "fom_scatter_deg": fom_scatter_deg,
        "sign": np.asarray([r["sign"] for r in runs]),
        "ambiguous": np.asarray([r["ambiguous"] for r in runs]),
        "spread_same": np.asarray([r["spread_same"] for r in runs]),
        "spread_flipped": np.asarray([r["spread_flipped"] for r in runs]),
        "time": np.asarray([r["time"] for r in runs]),
        "within_run_scatter": np.asarray([r["within_run_scatter"] for r in runs]),
        "kx": np.asarray([r["kx"] for r in runs]),
        "ky": np.asarray([r["ky"] for r in runs]),
        "fx": np.asarray([r["fx"] for r in runs]),
        "fy": np.asarray([r["fy"] for r in runs]),
        "kxx": np.asarray([r["kxx"] for r in runs]),
        "kyy": np.asarray([r["kyy"] for r in runs]),
        "kxy": np.asarray([r["kxy"] for r in runs]),
        "piston": np.asarray([r["piston"] for r in runs]),
        "converged": np.stack([r["converged"] for r in runs]),
        "kappa_p": np.stack([r["kappa_p"] for r in runs]),
        "kappa_ps": np.stack([r["kappa_ps"] for r in runs]),
        "repeats": n,
        "num_averages": num_averages,
        "num_piezo_steps": num_piezo_steps,
        "step_size": STEP_SIZE,
        "settle_s": SETTLE_S,
        "dry_num": DRY_NUM,
        "dry_step": DRY_STEP,
        "exposure_ms": EXPOSURE_MS,
        "gain": GAIN,
        "device": device,
        "roi": np.array([IMAGE_SHAPE_X0, IMAGE_SHAPE_Y0, IMAGE_SHAPE_X1, IMAGE_SHAPE_Y1]),
        "reference_x_by": REFERENCE_X_BY,
        "reference_y_by": REFERENCE_Y_BY,
        "drive_step_rate": applied_drive_params["step_rate"],
        "drive_step_acceleration": applied_drive_params["step_acceleration"],
        "drive_max_voltage": applied_drive_params["max_voltage"],
        "timestamp": timestamp,
        "description": description,
    }
    np.savez(save_path, **out)
    size_mb = save_path.stat().st_size / 1e6
    print(f"Saved measurement batch ({size_mb:.1f} MB) -> {save_path}")

    if error is not None:
        raise error

if __name__ == "__main__":
    main()
        
    
    
    
    
    