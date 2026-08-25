"""KIM101/PIAK10 coherence-zone scan: map fringe visibility vs. commanded position.

phase_measurement.py's small phase-shift sweep (STEP_SIZE * (NUM_PIEZO_STEPS-1)
commanded steps) rides on top of a much coarser, manually-set piezo position that
keeps the two interferometer arms' optical path lengths matched within the
source's coherence length -- the "good coherence zone". This script maps out
that zone directly: it sweeps the KIM101/PIAK10 across a range of commanded
positions, records a frame at intervals, and computes fringe visibility online
via `phase.measure_frame_visibility` (absolute/comparable across positions,
unlike `measure_frame_contrast` which is only relative within one stack -- see
that function's docstring).

IMPORTANT -- every commanded move in this script, from the very first to the
very last, is exactly STEP_SIZE (matching phase_measurement.py's own phase-shift
step), at the SAME drive parameters (DRIVE_STEP_RATE/DRIVE_STEP_ACCELERATION/
DRIVE_MAX_VOLTAGE) and SETTLE_S. This is deliberate, not an approximation: a
slip-stick inertial actuator commanded to move 1000 steps in one burst runs for
far longer than one commanded to move 5 steps at the same STEP_RATE (which sets
pulse frequency, not burst duration) -- a different mechanical regime, not just
"the same actuator going further". A scan built from a big single jump (even
just to *reach* the scan's start position) would make its own coordinate
inconsistent with the rest of the data from the first move onward. So there is
no jump anywhere: the whole trip is one continuous walk in four STEP_SIZE-only
segments --

    approach (0 -> -RANGE) -> outbound (-RANGE -> +RANGE)
        -> return (+RANGE -> -RANGE) -> retreat (-RANGE -> 0)

-- each preceded by its own DRY_NUM-step dry run in that segment's direction
(matching phase_measurement.py's own convention of always dry-running before a
sweep), including before the very last move back to the start position -- that
move matters as much as any other, since it's what the piezo is actually left at
before your next real measurement. Only outbound and return (the two full-range
segments) are used for the peak/FWHM/dead-zone analysis; approach and retreat
exist to get to and from the scan symmetrically without a jump, but their
visibility is recorded too (free -- the actuator is moving through those
positions regardless).

What's used to bound runtime/dataset size is CAPTURE_STRIDE -- how many of the
STEP_SIZE positions in each segment get a camera capture + visibility
measurement -- never the step size itself.

Before running: position the piezo at your manually-set good coherence zone
first (this script scans *around* the current position, symmetric in RANGE).

Run from the repo root so `instruments`/`phase` are importable:
    python scripts/phase/kim_coherence_scan.py
"""

import contextlib
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from instruments.camera import Camera, open_camera
from instruments.config import load_equipment
from instruments.inertialpiezo import KIM101, KIM101Axis
from phase import measure_frame_visibility

# --------------------------------------------------------------------------- #
# Settings -- edit before running                                             #
# --------------------------------------------------------------------------- #
_EQ = load_equipment()
CAMERA_SERIAL = _EQ.camera_serial
KIM_SERIAL = _EQ.kim_serial
KIM_CHANNEL = _EQ.kim_channel

EXPOSURE_MS = 250.0
GAIN = 100
BLACK_LEVEL = 0
# Small, centered ROI -- one frame per captured point, so this stays fast and
# the saved file stays small, rather than the full phase_measurement.py ROI.
# Centered on that ROI's midpoint (IMAGE_SHAPE_X0..X1, Y0..Y1 = 300,3600 /
# 400,2600 there): x_c=1950, y_c=1500.
ROI = (300, 400, 1500, 1500)  # (x0, y0, w, h) px
FRAMES_TO_AVERAGE = 1
FRAMES_TO_DROP = 0
OUT_BIT_DEPTH = np.uint16

# KIM101 drive parameters -- MUST match phase_measurement.py (effective step
# size depends on drive voltage/rate; a scan at different settings doesn't
# transfer to the measurement script's step units).
DRIVE_STEP_RATE = 200          # steps/s
DRIVE_STEP_ACCELERATION = 500  # steps/s^2
DRIVE_MAX_VOLTAGE = 125        # V

# Scan pattern. STEP_SIZE/SETTLE_S MUST match phase_measurement.py's own values
# exactly (see module docstring) -- this is not a tuning knob for this script.
STEP_SIZE = 5          # commanded steps per move -- MUST match phase_measurement.py's STEP_SIZE
RANGE = 1000            # commanded steps, scanned symmetrically: -RANGE .. +RANGE.
# If the peak sits at the edge of the scan (the script warns when this
# happens), widen RANGE on the *next* run -- don't guess a larger value
# up front; a wider RANGE costs real move+settle time (see the printed
# estimate below), all of it genuine, none of it hidden in a big jump.
# How many STEP_SIZE moves between captured points, within each segment. Every
# move still happens regardless of this (see module docstring) -- only how
# many of those positions get a camera capture + visibility measurement is
# reduced, to bound dataset size and capture/exposure time (not move time,
# which dominates the run and scales with RANGE/STEP_SIZE regardless of this).
CAPTURE_STRIDE = 4
SETTLE_S = 0.05         # pause after each move before capturing -- matches phase_measurement.py
DRY_NUM = 10            # backlash-compensation dry-run steps before each of the 4 segments
DRY_STEP = 1
MOVE_TIMEOUT_S = 30.0
MAX_DATASET_GB = 4.0    # pre-flight abort threshold, sized against captured (not moved) frames

# Segment codes, saved as scan_segment -- scan_direction alone can't tell
# approach from return (both -1) or outbound from retreat (both +1).
SEG_APPROACH, SEG_OUTBOUND, SEG_RETURN, SEG_RETREAT = 0, 1, 2, 3


@contextlib.contextmanager
def _armed_camera(camera: Camera):
    camera.arm()
    time.sleep(0.1)
    try:
        yield
    finally:
        camera.disarm()


def _crop(img: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    """Apply ROI = (x0, y0, w, h) as a software crop. No-op when roi is None."""
    if roi is None:
        return img
    x0, y0, w, h = roi
    H, W = img.shape[0], img.shape[1]
    if x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H:
        raise ValueError(f"ROI {roi} does not fit inside frame of shape {(H, W)}")
    return img[y0:y0 + h, x0:x0 + w]


def _dry_run(axis: KIM101Axis, direction: int) -> int:
    """DRY_NUM backlash-compensation steps of DRY_STEP in `direction` (+1/-1),
    run before every one of the 4 segments (see module docstring) -- including
    before the final return to the start position, which previous versions of
    this script skipped. Returns the net commanded delta (direction*DRY_NUM*
    DRY_STEP) so the caller can keep its own running `commanded` tally exact.
    """
    for _ in range(DRY_NUM):
        axis.move_by(direction * DRY_STEP, timeout=MOVE_TIMEOUT_S)
        time.sleep(SETTLE_S)
    return direction * DRY_NUM * DRY_STEP


def _sweep_segment(camera: Camera, axis: KIM101Axis, direction: int, n_moves: int,
                    segment_id: int, label: str, commanded: int, idx: int,
                    t_start: float, arrays: dict) -> tuple:
    """Move `n_moves` times by `direction * STEP_SIZE` -- every single one of
    them, never a bigger jump (see module docstring) -- capturing a frame and
    computing visibility every `CAPTURE_STRIDE`-th position (including
    position 0, before this segment's first move) into `arrays`, starting at
    `idx`. One implementation shared by all 4 segments rather than 4
    hand-written copies -- duplicated loops are exactly the kind of thing that
    caused the original step-counting bug in phase_measurement.py.

    Caller is responsible for the leading dry run in `direction` (each segment
    starts right after a direction reversal -- see `_dry_run`).

    Returns (commanded, idx) updated to their values after this segment.
    """
    captured = 0
    for m in range(n_moves + 1):
        if m > 0:
            axis.move_by(direction * STEP_SIZE, timeout=MOVE_TIMEOUT_S)
            time.sleep(SETTLE_S)
            commanded += direction * STEP_SIZE
        if m % CAPTURE_STRIDE != 0:
            continue

        img = camera.get_image(FRAMES_TO_AVERAGE, FRAMES_TO_DROP)
        frame = _crop(img, ROI)
        arrays["images"][idx] = frame
        arrays["commanded"][idx] = commanded
        arrays["counter"][idx] = axis.get_position()
        arrays["direction"][idx] = direction
        arrays["segment"][idx] = segment_id
        arrays["visibility"][idx] = float(
            measure_frame_visibility(frame[None, :, :, 0].astype(np.float32))[0])
        arrays["times"][idx] = time.monotonic() - t_start
        idx += 1
        captured += 1

        if captured % 20 == 0 or m == n_moves:
            print(f"    [{label}] move {m}/{n_moves} (capture {captured}) -> "
                  f"commanded {commanded}, visibility {arrays['visibility'][idx - 1]:.5f}")

    return commanded, idx


def _estimate_envelope(commanded: np.ndarray, visibility: np.ndarray) -> tuple:
    """Peak position and FWHM (both in commanded steps) of one sweep's
    visibility-vs-position curve, by linear interpolation of the half-max
    crossings on either side of the peak sample.

    Returns (peak_commanded, fwhm) as floats, or (nan, nan) with a printed
    warning if the curve doesn't have a clear interior peak (e.g. RANGE was
    too small and the envelope is clipped, or it's flat/noise-dominated) --
    same defensive spirit as e.g. `phase.carrier.remove_carrier`'s block-count
    warning, since a silently wrong FWHM from a clipped scan is worse than an
    honest "couldn't tell".
    """
    order = np.argsort(commanded)
    x, y = commanded[order], visibility[order]
    i_peak = int(np.argmax(y))
    if i_peak == 0 or i_peak == len(y) - 1:
        print("WARNING: envelope peak sits at the edge of the scanned range -- "
              "RANGE is too small to see the whole coherence envelope; widen it.")
        return float("nan"), float("nan")

    half = 0.5 * (y[i_peak] + min(y[0], y[-1]))

    def _crossing(lo, hi):
        # linear-interpolate the step index where y crosses `half`, walking
        # outward from the peak; returns None if it never crosses within range.
        for i in range(lo, hi, 1 if hi > lo else -1):
            j = i + (1 if hi > lo else -1)
            if (y[i] - half) * (y[j] - half) <= 0 and y[i] != y[j]:
                t = (half - y[i]) / (y[j] - y[i])
                return x[i] + t * (x[j] - x[i])
        return None

    left = _crossing(i_peak, 0)
    right = _crossing(i_peak, len(y) - 1)
    if left is None or right is None:
        print("WARNING: half-max crossing not found within the scanned range on "
              "at least one side -- RANGE is too small; widen it. Peak position "
              "is still reported, FWHM is not.")
        return float(x[i_peak]), float("nan")
    return float(x[i_peak]), float(right - left)


if __name__ == "__main__":
    t_start = time.monotonic()

    with contextlib.ExitStack() as stack:
        camera = open_camera(_EQ, stack)
        stack.enter_context(camera)
        kim = stack.enter_context(KIM101(KIM_SERIAL))
        axis = kim.axis(KIM_CHANNEL)

        camera.set_exposure_ms(EXPOSURE_MS)
        if camera.supports_gain:
            camera.set_gain(GAIN)
        if camera.supports_black_level:
            camera.set_black_level(BLACK_LEVEL)
        camera.set_raw_bit_depth(getattr(np, _EQ.camera_bit_depth))
        camera.set_out_bit_depth(OUT_BIT_DEPTH)

        axis.set_drive_parameters(
            step_rate=DRIVE_STEP_RATE,
            step_acceleration=DRIVE_STEP_ACCELERATION,
            max_voltage=DRIVE_MAX_VOLTAGE,
        )
        applied_drive_params = axis.get_drive_parameters()
        print(f"Drive parameters applied: {applied_drive_params}")

        # Track position as a commanded delta from wherever the piezo already
        # was (the manually-set good zone) -- zero() makes that delta directly
        # readable off the counter, matching kim_step_calibration.py.
        axis.zero()

        H, W = camera.image_shape
        C = camera.num_channels
        if ROI is not None:
            _, _, crop_w, crop_h = ROI
            H, W = crop_h, crop_w

        # n_moves_* is how many times the actuator actually moves in each
        # segment -- every one of them is a real STEP_SIZE burst (see module
        # docstring). n_capture_* is how many of those positions get a camera
        # capture; dataset size depends on that, not on n_moves_*.
        n_moves_half = RANGE // STEP_SIZE        # approach, retreat: 0 <-> -RANGE
        n_moves_full = 2 * RANGE // STEP_SIZE    # outbound, return: -RANGE <-> +RANGE
        n_capture_half = n_moves_half // CAPTURE_STRIDE + 1
        n_capture_full = n_moves_full // CAPTURE_STRIDE + 1
        n_scan = 2 * n_capture_half + 2 * n_capture_full
        n_moves_total = 2 * n_moves_half + 2 * n_moves_full

        bytes_per_frame = H * W * C * np.dtype(OUT_BIT_DEPTH).itemsize
        estimated_gb = n_scan * bytes_per_frame / 1e9
        est_move_s = n_moves_total * (STEP_SIZE / DRIVE_STEP_RATE + SETTLE_S)
        print(f"Planned scan: {n_scan} captured points across 4 segments "
              f"(approach+outbound+return+retreat) of {H}x{W}x{C} "
              f"({OUT_BIT_DEPTH.__name__}) -> ~{estimated_gb:.2f} GB, from "
              f"{n_moves_total} STEP_SIZE={STEP_SIZE} moves total "
              f"(~{est_move_s:.0f}s of move+settle time alone, everything else "
              f"-- camera exposure, dry runs -- on top of that)")
        if estimated_gb > MAX_DATASET_GB:
            raise RuntimeError(
                f"Planned dataset (~{estimated_gb:.2f} GB) exceeds MAX_DATASET_GB "
                f"({MAX_DATASET_GB} GB). Reduce RANGE, increase CAPTURE_STRIDE, use a "
                f"smaller ROI, or raise MAX_DATASET_GB. (STEP_SIZE is not a dataset-size "
                f"knob -- it must match phase_measurement.py.)"
            )

        scan_images = np.zeros((n_scan, H, W, C), dtype=OUT_BIT_DEPTH)
        scan_commanded = np.zeros(n_scan, dtype=np.int64)
        scan_counter = np.zeros(n_scan, dtype=np.int64)
        scan_direction = np.zeros(n_scan, dtype=np.int8)
        scan_segment = np.zeros(n_scan, dtype=np.int8)
        scan_visibility = np.zeros(n_scan, dtype=np.float64)
        scan_times = np.zeros(n_scan, dtype=np.float64)
        arrays = dict(images=scan_images, commanded=scan_commanded,
                       counter=scan_counter, direction=scan_direction,
                       segment=scan_segment, visibility=scan_visibility,
                       times=scan_times)

        idx = 0
        commanded = 0
        error: Exception | None = None

        try:
            with _armed_camera(camera):
                # Every one of the 4 segments below is preceded by its own
                # dry run in that segment's direction -- including the last
                # one, back to the start position (see module docstring: a
                # previous version of this script skipped that one).
                commanded += _dry_run(axis, -1)
                print(f"Approach: {n_moves_half} moves of {STEP_SIZE} steps "
                      f"to reach the scan start (0 -> {-RANGE}) ...")
                commanded, idx = _sweep_segment(
                    camera, axis, -1, n_moves_half, SEG_APPROACH, "approach",
                    commanded, idx, t_start, arrays)

                commanded += _dry_run(axis, 1)
                print(f"Outbound sweep: {n_moves_full} moves, capturing every "
                      f"{CAPTURE_STRIDE}th ({n_capture_full} points) ...")
                commanded, idx = _sweep_segment(
                    camera, axis, 1, n_moves_full, SEG_OUTBOUND, "outbound",
                    commanded, idx, t_start, arrays)

                commanded += _dry_run(axis, -1)
                print(f"Return sweep: {n_moves_full} moves, capturing every "
                      f"{CAPTURE_STRIDE}th ({n_capture_full} points) ...")
                commanded, idx = _sweep_segment(
                    camera, axis, -1, n_moves_full, SEG_RETURN, "return",
                    commanded, idx, t_start, arrays)

                commanded += _dry_run(axis, 1)
                print(f"Retreat: {n_moves_half} moves back toward the start ...")
                commanded, idx = _sweep_segment(
                    camera, axis, 1, n_moves_half, SEG_RETREAT, "retreat",
                    commanded, idx, t_start, arrays)
        except Exception as exc:
            error = exc
            print(f"ERROR during acquisition: {exc!r}")
            print(f"Captured {idx}/{n_scan} scan points before the failure -- "
                  f"saving partial data.")
        finally:
            # The four segments above are sized to cancel exactly (approach
            # -RANGE, outbound +2*RANGE, return -2*RANGE, retreat +RANGE net
            # to 0), so by this point the commanded counter should already be
            # back near 0 -- off only by whatever the four dry runs' own step
            # count didn't itself cancel (at most 4*DRY_NUM*DRY_STEP=40 steps
            # at these defaults), not the ~2*RANGE-step gap an unmatched-regime
            # jump would have left. move_to(0) below is therefore now a small,
            # honest correction -- not a second instance of the big-jump
            # problem this revision fixes (see module docstring).
            residual = axis.get_position()
            print(f"Returning to start position (commanded 0; residual "
                  f"{residual} steps from the four dry runs) ...")
            try:
                axis.move_to(0, timeout=MOVE_TIMEOUT_S)
            except Exception as move_exc:
                print(f"WARNING: failed to return to start position: {move_exc!r}")
            print(f"Final commanded position: {axis.get_position()}")

    # -------------------------------------------------------------------------
    # Trim to what was actually captured (see kim_step_calibration.py).
    # -------------------------------------------------------------------------
    scan_images = scan_images[:idx]
    scan_commanded = scan_commanded[:idx]
    scan_counter = scan_counter[:idx]
    scan_direction = scan_direction[:idx]
    scan_segment = scan_segment[:idx]
    scan_visibility = scan_visibility[:idx]
    scan_times = scan_times[:idx]

    # -------------------------------------------------------------------------
    # Envelope analysis: peak position + FWHM per direction, and the
    # forward/backward peak offset (dead zone + accumulated asymmetry).
    # Restricted to the two full-range segments (scan_direction alone can't
    # tell outbound from retreat, or approach from return -- scan_segment can).
    # -------------------------------------------------------------------------
    out_mask = scan_segment == SEG_OUTBOUND
    ret_mask = scan_segment == SEG_RETURN
    peak_out = fwhm_out = peak_ret = fwhm_ret = float("nan")
    fwd_back_offset = float("nan")
    if out_mask.sum() >= 3:
        peak_out, fwhm_out = _estimate_envelope(
            scan_commanded[out_mask].astype(float), scan_visibility[out_mask])
    if ret_mask.sum() >= 3:
        peak_ret, fwhm_ret = _estimate_envelope(
            scan_commanded[ret_mask].astype(float), scan_visibility[ret_mask])
    if not (np.isnan(peak_out) or np.isnan(peak_ret)):
        fwd_back_offset = peak_ret - peak_out

    # phase_measurement.py's own phase-shift sweep spans (NUM_PIEZO_STEPS-1)*
    # STEP_SIZE commanded steps; STEP_SIZE above already must match that
    # script's value (see module docstring), so only NUM_PIEZO_STEPS=20 (its
    # default) needs hardcoding here -- kept out of an import to avoid pulling
    # in that script's hardware-driver imports. Keep in sync if it changes.
    sweep_span = STEP_SIZE * (20 - 1)
    print("\n--- Coherence envelope summary ---")
    print(f"Outbound peak: {peak_out:+.1f} steps, FWHM {fwhm_out:.1f} steps")
    print(f"Return   peak: {peak_ret:+.1f} steps, FWHM {fwhm_ret:.1f} steps")
    print(f"Forward/backward peak offset (dead zone + asymmetry): "
          f"{fwd_back_offset:+.1f} steps")
    if not np.isnan(fwhm_out):
        print(f"phase_measurement.py's own sweep span "
              f"(STEP_SIZE*(NUM_PIEZO_STEPS-1)={sweep_span} steps) is "
              f"{100 * sweep_span / fwhm_out:.1f}% of the outbound FWHM.")

    description = (
        "KIM101/PIAK10 coherence-zone scan: fringe visibility (phase."
        "measure_frame_visibility, absolute/comparable across the whole scan -- "
        "not measure_frame_contrast, which only normalizes within one stack) vs. "
        "commanded piezo position. Every commanded move in the whole run is "
        "exactly STEP_SIZE (no jump of any other size, see module docstring), "
        "in 4 segments identified by 'scan_segment': 0=approach (0->-RANGE), "
        "1=outbound (-RANGE->+RANGE), 2=return (+RANGE->-RANGE), 3=retreat "
        "(-RANGE->0). 'scan_direction' is the physical move sign (+1/-1) and "
        "does not by itself distinguish e.g. approach from return (both -1) -- "
        "use 'scan_segment' for that. Peak position and FWHM are computed from "
        "'scan_segment' 1 and 2 only (approach/retreat visibility is recorded "
        "but not used for the envelope fit); the forward/backward peak offset "
        "(= reversal dead zone + any step-size asymmetry, in commanded steps) "
        "is their difference. All are printed and saved as scalars below. "
        "'scan_counter' is the open-loop commanded-step counter "
        "(KIM101Axis.get_position()), not a position sensor. Use the located "
        "peak as the center for a finer re-scan (narrower RANGE and/or smaller "
        "CAPTURE_STRIDE -- STEP_SIZE stays fixed, matching phase_measurement.py), "
        "or as the position to manually re-set the piezo to before a measurement "
        "sequence."
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path(__file__).resolve().parent / f"kim_coherence_scan_{timestamp}.npz"

    np.savez(
        save_path,
        scan_images=scan_images,              # (n_scan, H, W, C)
        scan_commanded=scan_commanded,        # (n_scan,) cumulative commanded steps from start
        scan_counter=scan_counter,            # (n_scan,) axis.get_position() readback
        scan_direction=scan_direction,        # (n_scan,) +1 / -1 physical move sign
        scan_segment=scan_segment,            # (n_scan,) 0=approach 1=outbound 2=return 3=retreat
        scan_visibility=scan_visibility,      # (n_scan,) phase.measure_frame_visibility
        scan_times=scan_times,                # (n_scan,) s since start
        peak_out=peak_out,
        fwhm_out=fwhm_out,
        peak_ret=peak_ret,
        fwhm_ret=fwhm_ret,
        fwd_back_offset=fwd_back_offset,
        description=description,
        timestamp=timestamp,
        exposure_ms=EXPOSURE_MS,
        gain=GAIN,
        black_level=BLACK_LEVEL,
        roi=np.array(ROI) if ROI is not None else np.array([]),
        frames_to_average=FRAMES_TO_AVERAGE,
        frames_to_drop=FRAMES_TO_DROP,
        out_bit_depth=OUT_BIT_DEPTH.__name__,
        range_steps=RANGE,
        step_size=STEP_SIZE,
        capture_stride=CAPTURE_STRIDE,
        settle_s=SETTLE_S,
        dry_num=DRY_NUM,
        dry_step=DRY_STEP,
        drive_step_rate=applied_drive_params["step_rate"],
        drive_step_acceleration=applied_drive_params["step_acceleration"],
        drive_max_voltage=applied_drive_params["max_voltage"],
        camera_serial=CAMERA_SERIAL,
        camera_vendor=_EQ.camera_vendor,
        kim_serial=KIM_SERIAL,
        kim_channel=KIM_CHANNEL,
    )

    size_mb = save_path.stat().st_size / 1e6
    print(f"Saved coherence scan ({size_mb:.1f} MB) -> {save_path}")

    if error is not None:
        raise error
