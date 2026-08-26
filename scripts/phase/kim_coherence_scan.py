"""KIM101/PIAK10 coherence-zone scan: map fringe visibility vs. move index.

phase_measurement.py's small phase-shift sweep (PM_STEP_SIZE * (NUM_PIEZO_STEPS-1)
commanded steps) rides on top of a much coarser, manually-set piezo position that
keeps the two interferometer arms' optical path lengths matched within the
source's coherence length -- the "good coherence zone". This script maps out
that zone directly: it sweeps the KIM101/PIAK10 across a range of commanded
positions, records a frame at intervals, and computes fringe visibility online
via `phase.measure_frame_visibility` (absolute/comparable across positions,
unlike `measure_frame_contrast` which is only relative within one stack -- see
that function's docstring).

IMPORTANT -- every commanded move in this script, from the very first to the
very last, is exactly STEP_SIZE_FWD forward or STEP_SIZE_BWD backward, at the
SAME drive parameters (DRIVE_STEP_RATE/DRIVE_STEP_ACCELERATION/
DRIVE_MAX_VOLTAGE) and SETTLE_S. This is deliberate, not an approximation: a
slip-stick inertial actuator commanded to move 1000 steps in one burst runs for
far longer than one commanded to move 5 steps at the same STEP_RATE (which sets
pulse frequency, not burst duration) -- a different mechanical regime, not just
"the same actuator going further". A scan built from a big single jump (even
just to *reach* the scan's start position) would make its own coordinate
inconsistent with the rest of the data from the first move onward. So there is
no jump anywhere: the whole trip is one continuous walk in four segments, each
made of nothing but STEP_SIZE_FWD/STEP_SIZE_BWD moves in that segment's
direction --

    approach (0 -> -RANGE_bwd) -> outbound (-RANGE_bwd -> +RANGE_fwd)
        -> return (+RANGE_fwd -> -RANGE_bwd) -> retreat (-RANGE_bwd -> 0)

-- each preceded by its own DRY_NUM-step dry run in that segment's direction
(matching phase_measurement.py's own convention of always dry-running before a
sweep), including before the very last move back to the start position -- that
move matters as much as any other, since it's what the piezo is actually left at
before your next real measurement. Only outbound and return (the two full-range
segments) are used for the peak/FWHM/dead-zone analysis; approach and retreat
exist to get to and from the scan symmetrically without a jump, but their
visibility is recorded too (free -- the actuator is moving through those
positions regardless).

STEP_SIZE_FWD and STEP_SIZE_BWD are separate because the actuator was found to
cover different physical distance per commanded step depending on direction
(a real slip-stick asymmetry, not measurement noise) -- calibrate them so that
one forward move and one backward move cover the same physical distance. A
"move" (one STEP_SIZE_FWD or STEP_SIZE_BWD burst) is therefore the script's
physical coordinate, not the raw commanded step count: 'scan_moves' (used for
all analysis) counts moves signed by direction; 'scan_commanded' is the raw
open-loop commanded-step tally sent to the controller, kept only as telemetry
-- with STEP_SIZE_FWD != STEP_SIZE_BWD it does *not* return to 0 at the end of
a symmetric walk (see the `finally` block).

What's used to bound runtime/dataset size is CAPTURE_STRIDE -- how many of the
moves in each segment get a camera capture + visibility measurement -- never
the step sizes themselves.

Before running: position the piezo at your manually-set good coherence zone
first (this script scans *around* the current position, RANGE_bwd commanded
steps backward and RANGE_fwd commanded steps forward -- not necessarily
symmetric, and each measured in its own direction's step units).

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
ROI = (300, 400, 3300, 2200)  # (x0, y0, w, h) px
FRAMES_TO_AVERAGE = 1
FRAMES_TO_DROP = 2
OUT_BIT_DEPTH = np.uint16

# KIM101 drive parameters -- MUST match phase_measurement.py (effective step
# size depends on drive voltage/rate; a scan at different settings doesn't
# transfer to the measurement script's step units).
DRIVE_STEP_RATE = 200          # steps/s
DRIVE_STEP_ACCELERATION = 500  # steps/s^2
DRIVE_MAX_VOLTAGE = 125        # V

# Scan pattern. SETTLE_S MUST match phase_measurement.py's own value exactly
# (see module docstring) -- this is not a tuning knob for this script.
#
# STEP_SIZE_FWD/STEP_SIZE_BWD are commanded steps per move, one per direction
# -- calibrate them so that one forward move and one backward move cover the
# same physical distance (the actuator was found not to be symmetric; see
# module docstring). STEP_SIZE_FWD is the one that must match
# phase_measurement.py's STEP_SIZE (PM_STEP_SIZE below) for the sweep-span
# comparison printed at the end to mean anything.
STEP_SIZE_FWD = 5      # commanded steps per forward (+) move
STEP_SIZE_BWD = 4      # commanded steps per backward (-) move
RANGE_bwd = 2500        # commanded steps scanned backward from start: -RANGE_bwd .. 0
RANGE_fwd = 4000        # commanded steps scanned forward from start: 0 .. +RANGE_fwd
# If the peak sits at the edge of the scan (the script warns when this
# happens), widen RANGE_bwd (peak at the backward edge) or RANGE_fwd (peak at
# the forward edge) on the *next* run -- don't guess a larger value up front;
# a wider range costs real move+settle time (see the printed estimate below),
# all of it genuine, none of it hidden in a big jump.
# How many moves between captured points, within each segment. Every move
# still happens regardless of this (see module docstring) -- only how many of
# those positions get a camera capture + visibility measurement is reduced,
# to bound dataset size and capture/exposure time (not move time, which
# dominates the run and scales with RANGE_bwd/RANGE_fwd/STEP_SIZE_FWD/
# STEP_SIZE_BWD regardless of this).
CAPTURE_STRIDE = 8
SETTLE_S = 0.1         # pause after each move before capturing -- matches phase_measurement.py
DRY_NUM = 10            # backlash-compensation dry-run steps before each of the 4 segments
DRY_STEP = 1
MOVE_TIMEOUT_S = 30.0
MAX_DATASET_GB = 8.0    # pre-flight abort threshold, sized against captured (not moved) frames

# phase_measurement.py's own STEP_SIZE -- kept as a plain constant (not an
# import) to avoid pulling in that script's hardware-driver imports; keep in
# sync if it changes. Used only for the sweep-span comparison printed at the
# end, which is meaningful only when STEP_SIZE_FWD == PM_STEP_SIZE.
PM_STEP_SIZE = 5

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


def _step_size(direction: int) -> int:
    """Commanded steps per scan move in `direction` (+1 forward / -1 backward)
    -- the single place that picks STEP_SIZE_FWD vs STEP_SIZE_BWD, so no
    other code has to re-decide which one applies.
    """
    return STEP_SIZE_FWD if direction > 0 else STEP_SIZE_BWD


def _dry_run(axis: KIM101Axis, direction: int) -> tuple[int, float]:
    """DRY_NUM backlash-compensation steps of DRY_STEP in `direction` (+1/-1),
    run before every one of the 4 segments (see module docstring) -- including
    before the final return to the start position, which previous versions of
    this script skipped. Returns (commanded_delta, moves_delta):
    commanded_delta = direction*DRY_NUM*DRY_STEP (raw steps sent to the
    controller, for the caller's `commanded` tally); moves_delta is that same
    motion expressed in this direction's move units (direction*DRY_NUM*
    DRY_STEP/_step_size(direction)) for the caller's `moves` tally -- DRY_STEP
    is not one full move, so this is generally a fraction of a move.
    """
    for _ in range(DRY_NUM):
        axis.move_by(direction * DRY_STEP, timeout=MOVE_TIMEOUT_S)
        time.sleep(SETTLE_S)
    commanded_delta = direction * DRY_NUM * DRY_STEP
    moves_delta = commanded_delta / _step_size(direction)
    return commanded_delta, moves_delta


def _sweep_segment(camera: Camera, axis: KIM101Axis, direction: int, n_moves: int,
                    segment_id: int, label: str, commanded: int, moves: float,
                    idx: int, t_start: float, arrays: dict) -> tuple:
    """Move `n_moves` times by `direction * _step_size(direction)` -- every
    single one of them, never a bigger jump (see module docstring) --
    capturing a frame and computing visibility every `CAPTURE_STRIDE`-th
    position (including position 0, before this segment's first move) into
    `arrays`, starting at `idx`. One implementation shared by all 4 segments
    rather than 4 hand-written copies -- duplicated loops are exactly the
    kind of thing that caused the original step-counting bug in
    phase_measurement.py.

    Caller is responsible for the leading dry run in `direction` (each segment
    starts right after a direction reversal -- see `_dry_run`).

    `moves` is the move-index coordinate: it increments by exactly
    `direction * 1` per move regardless of `direction`'s step size, so it is
    the physical (direction-independent) coordinate used for analysis, unlike
    `commanded` which is in that direction's own step units (see module
    docstring).

    Returns (commanded, moves, idx) updated to their values after this
    segment.
    """
    step = _step_size(direction)
    captured = 0
    for m in range(n_moves + 1):
        if m > 0:
            axis.move_by(direction * step, timeout=MOVE_TIMEOUT_S)
            time.sleep(SETTLE_S)
            commanded += direction * step
            moves += direction
        if m % CAPTURE_STRIDE != 0:
            continue

        img = camera.get_image(FRAMES_TO_AVERAGE, FRAMES_TO_DROP)
        frame = _crop(img, ROI)
        arrays["images"][idx] = frame
        arrays["commanded"][idx] = commanded
        arrays["moves"][idx] = moves
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
                  f"commanded {commanded}, moves {moves:+.1f}, "
                  f"visibility {arrays['visibility'][idx - 1]:.5f}")

    return commanded, moves, idx


def _estimate_envelope(moves: np.ndarray, visibility: np.ndarray) -> tuple:
    """Peak position and FWHM (both in moves -- one move = one STEP_SIZE_FWD/
    STEP_SIZE_BWD burst, calibrated to equal physical distance either
    direction, see module docstring) of one sweep's visibility-vs-position
    curve, by linear interpolation of the half-max crossings on either side
    of the peak sample.

    Returns (peak_moves, fwhm) as floats, or (nan, nan) with a printed
    warning if the curve doesn't have a clear interior peak (e.g. RANGE_bwd/
    RANGE_fwd was too small and the envelope is clipped, or it's flat/
    noise-dominated) -- same defensive spirit as e.g.
    `phase.carrier.remove_carrier`'s block-count warning, since a silently
    wrong FWHM from a clipped scan is worse than an honest "couldn't tell".
    """
    # moves is ascending after this sort, so index 0 is the backward
    # (-RANGE_bwd) end of the scan and index -1 is the forward (+RANGE_fwd) end.
    order = np.argsort(moves)
    x, y = moves[order], visibility[order]
    i_peak = int(np.argmax(y))
    if i_peak == 0:
        print("WARNING: envelope peak sits at the backward edge of the scanned "
              "range -- RANGE_bwd is too small to see the whole coherence "
              "envelope; widen it.")
        return float("nan"), float("nan")
    if i_peak == len(y) - 1:
        print("WARNING: envelope peak sits at the forward edge of the scanned "
              "range -- RANGE_fwd is too small to see the whole coherence "
              "envelope; widen it.")
        return float("nan"), float("nan")

    # With an asymmetric scan the two tails aren't equally far from the peak,
    # so y[0] and y[-1] need not agree; taking the lower of the two as the
    # baseline keeps the reported half-max (and thus FWHM) conservative.
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
    if left is None:
        print("WARNING: half-max crossing not found on the backward side of "
              "the scanned range -- RANGE_bwd is too small; widen it. Peak "
              "position is still reported, FWHM is not.")
        return float(x[i_peak]), float("nan")
    if right is None:
        print("WARNING: half-max crossing not found on the forward side of "
              "the scanned range -- RANGE_fwd is too small; widen it. Peak "
              "position is still reported, FWHM is not.")
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
        # segment -- every one of them is a real STEP_SIZE_FWD/STEP_SIZE_BWD
        # burst (see module docstring). n_capture_* is how many of those
        # positions get a camera capture; dataset size depends on that, not
        # on n_moves_*.
        #
        # n_moves_full is deliberately the *sum* of the two per-side counts
        # (not (RANGE_bwd + RANGE_fwd) // some single step size) so the four
        # segments cancel to exactly 0 net moves even when RANGE_bwd/
        # RANGE_fwd aren't multiples of their own step size:
        # -n_moves_bwd + n_moves_full - n_moves_full + n_moves_bwd == 0
        # regardless of truncation.
        n_moves_bwd = RANGE_bwd // STEP_SIZE_BWD  # approach, retreat: 0 <-> -RANGE_bwd
        n_moves_fwd = RANGE_fwd // STEP_SIZE_FWD  # forward half of outbound/return
        n_moves_full = n_moves_bwd + n_moves_fwd  # outbound, return: -RANGE_bwd <-> +RANGE_fwd
        n_capture_bwd = n_moves_bwd // CAPTURE_STRIDE + 1
        n_capture_full = n_moves_full // CAPTURE_STRIDE + 1
        n_scan = 2 * n_capture_bwd + 2 * n_capture_full
        n_moves_total = 2 * n_moves_bwd + 2 * n_moves_full

        # Effective range actually walked (may differ from the configured
        # RANGE_bwd/RANGE_fwd if either isn't a multiple of its own step size).
        eff_bwd = n_moves_bwd * STEP_SIZE_BWD
        eff_fwd = n_moves_fwd * STEP_SIZE_FWD
        if eff_bwd != RANGE_bwd or eff_fwd != RANGE_fwd:
            print(f"NOTE: RANGE_bwd/RANGE_fwd aren't multiples of "
                  f"STEP_SIZE_BWD={STEP_SIZE_BWD}/STEP_SIZE_FWD={STEP_SIZE_FWD}; "
                  f"effective scanned range is -{eff_bwd} .. +{eff_fwd} commanded "
                  f"steps (configured -{RANGE_bwd} .. +{RANGE_fwd}).")

        bytes_per_frame = H * W * C * np.dtype(OUT_BIT_DEPTH).itemsize
        estimated_gb = n_scan * bytes_per_frame / 1e9
        # Forward and backward moves have different burst durations now.
        # Outbound+retreat are forward moves; approach+return are backward
        # moves -- by construction both totals equal n_moves_bwd+n_moves_full.
        n_fwd_or_bwd_moves_total = n_moves_bwd + n_moves_full
        est_move_s = n_fwd_or_bwd_moves_total * (
            (STEP_SIZE_FWD + STEP_SIZE_BWD) / DRIVE_STEP_RATE + 2 * SETTLE_S)
        print(f"Planned scan: {n_scan} captured points across 4 segments "
              f"(approach+outbound+return+retreat), span -{n_moves_bwd} .. "
              f"+{n_moves_fwd} moves (-{eff_bwd} bwd / +{eff_fwd} fwd commanded "
              f"steps), of {H}x{W}x{C} ({OUT_BIT_DEPTH.__name__}) -> "
              f"~{estimated_gb:.2f} GB, from {n_moves_total} moves total "
              f"(STEP_SIZE_FWD={STEP_SIZE_FWD}/STEP_SIZE_BWD={STEP_SIZE_BWD}) "
              f"(~{est_move_s:.0f}s of move+settle time alone, everything else "
              f"-- camera exposure, dry runs -- on top of that)")
        if estimated_gb > MAX_DATASET_GB:
            raise RuntimeError(
                f"Planned dataset (~{estimated_gb:.2f} GB) exceeds MAX_DATASET_GB "
                f"({MAX_DATASET_GB} GB). Reduce RANGE_bwd/RANGE_fwd, increase "
                f"CAPTURE_STRIDE, use a smaller ROI, or raise MAX_DATASET_GB. "
                f"(STEP_SIZE_FWD/STEP_SIZE_BWD are not dataset-size knobs -- "
                f"STEP_SIZE_FWD must match phase_measurement.py's STEP_SIZE.)"
            )

        scan_images = np.zeros((n_scan, H, W, C), dtype=OUT_BIT_DEPTH)
        scan_commanded = np.zeros(n_scan, dtype=np.int64)
        scan_moves = np.zeros(n_scan, dtype=np.float64)
        scan_counter = np.zeros(n_scan, dtype=np.int64)
        scan_direction = np.zeros(n_scan, dtype=np.int8)
        scan_segment = np.zeros(n_scan, dtype=np.int8)
        scan_visibility = np.zeros(n_scan, dtype=np.float64)
        scan_times = np.zeros(n_scan, dtype=np.float64)
        arrays = dict(images=scan_images, commanded=scan_commanded,
                       moves=scan_moves, counter=scan_counter,
                       direction=scan_direction, segment=scan_segment,
                       visibility=scan_visibility, times=scan_times)

        idx = 0
        commanded = 0
        moves = 0.0
        error: Exception | None = None

        try:
            with _armed_camera(camera):
                # Every one of the 4 segments below is preceded by its own
                # dry run in that segment's direction -- including the last
                # one, back to the start position (see module docstring: a
                # previous version of this script skipped that one).
                d_cmd, d_mov = _dry_run(axis, -1)
                commanded += d_cmd
                moves += d_mov
                print(f"Approach: {n_moves_bwd} moves of {STEP_SIZE_BWD} steps "
                      f"to reach the scan start (0 -> -{n_moves_bwd} moves, "
                      f"{-eff_bwd} commanded) ...")
                commanded, moves, idx = _sweep_segment(
                    camera, axis, -1, n_moves_bwd, SEG_APPROACH, "approach",
                    commanded, moves, idx, t_start, arrays)

                d_cmd, d_mov = _dry_run(axis, 1)
                commanded += d_cmd
                moves += d_mov
                print(f"Outbound sweep: {n_moves_full} moves (-{n_moves_bwd} -> "
                      f"+{n_moves_fwd} moves), capturing every {CAPTURE_STRIDE}th "
                      f"({n_capture_full} points) ...")
                commanded, moves, idx = _sweep_segment(
                    camera, axis, 1, n_moves_full, SEG_OUTBOUND, "outbound",
                    commanded, moves, idx, t_start, arrays)

                d_cmd, d_mov = _dry_run(axis, -1)
                commanded += d_cmd
                moves += d_mov
                print(f"Return sweep: {n_moves_full} moves (+{n_moves_fwd} -> "
                      f"-{n_moves_bwd} moves), capturing every {CAPTURE_STRIDE}th "
                      f"({n_capture_full} points) ...")
                commanded, moves, idx = _sweep_segment(
                    camera, axis, -1, n_moves_full, SEG_RETURN, "return",
                    commanded, moves, idx, t_start, arrays)

                d_cmd, d_mov = _dry_run(axis, 1)
                commanded += d_cmd
                moves += d_mov
                print(f"Retreat: {n_moves_bwd} moves back toward the start ...")
                commanded, moves, idx = _sweep_segment(
                    camera, axis, 1, n_moves_bwd, SEG_RETREAT, "retreat",
                    commanded, moves, idx, t_start, arrays)
        except Exception as exc:
            error = exc
            print(f"ERROR during acquisition: {exc!r}")
            print(f"Captured {idx}/{n_scan} scan points before the failure -- "
                  f"saving partial data.")
        finally:
            # The four scan segments above are sized to cancel exactly in
            # MOVES (approach -n_moves_bwd, outbound +n_moves_full, return
            # -n_moves_full, retreat +n_moves_bwd net to 0), so the piezo is
            # already back at the start physically, up to the small residual
            # the four dry runs leave (DRY_STEP isn't a whole move, so their
            # net move contribution isn't exactly 0 once STEP_SIZE_FWD !=
            # STEP_SIZE_BWD -- see dry_residual_moves below). No return move
            # is issued here.
            #
            # The raw COMMANDED counter is a different story: each segment's
            # commanded delta is (moves in that segment) * (that segment's
            # step size), and the two step sizes no longer cancel
            # symmetrically once STEP_SIZE_FWD != STEP_SIZE_BWD, so the
            # counter settles at a predictable nonzero offset rather than 0.
            # The four dry runs contribute exactly 0 net commanded steps
            # (two forward, two backward, all DRY_NUM*DRY_STEP), so this
            # offset comes entirely from the scan segments:
            expected = (n_moves_bwd + n_moves_full) * (STEP_SIZE_FWD - STEP_SIZE_BWD)
            dry_residual_moves = 2 * DRY_NUM * DRY_STEP * (
                1.0 / STEP_SIZE_FWD - 1.0 / STEP_SIZE_BWD)
            residual = axis.get_position()
            print(f"No return move issued -- commanding move_to(0) here would "
                  f"both violate the no-jump invariant (see module docstring) "
                  f"and, now that STEP_SIZE_FWD != STEP_SIZE_BWD makes "
                  f"'commanded 0' the wrong target, physically move the piezo "
                  f"rather than restore it.")
            if error is None:
                # The four scan segments completed and cancel exactly in
                # moves, so the piezo is already back at the start physically
                # -- up to the small residual the four dry runs leave, since
                # DRY_STEP isn't a whole move (dry_residual_moves below).
                print(f"Walk completed and is symmetric in moves "
                      f"({moves:+.3f} net, of which {dry_residual_moves:+.3f} "
                      f"is the dry runs' own residual) -- piezo is already "
                      f"back at the start.")
                print(f"Commanded-step counter: {residual} "
                      f"(expected {expected:+d} from the FWD/BWD step-size "
                      f"difference across the scan segments)")
                if residual != expected:
                    print(f"WARNING: commanded counter {residual} does not "
                          f"match the expected offset {expected:+d} -- check "
                          f"for a missed/extra move or a mid-run "
                          f"STEP_SIZE_FWD/STEP_SIZE_BWD change.")
            else:
                # The run was interrupted mid-walk -- the segments have NOT
                # necessarily cancelled, so the piezo is left wherever it
                # stopped (moves={moves:+.3f} net from start), not restored.
                print(f"Run was interrupted -- piezo left at its current "
                      f"position (moves {moves:+.3f} net from start, "
                      f"commanded counter {residual}), NOT restored to the "
                      f"start. Move it back manually before the next run if "
                      f"needed.")

    # -------------------------------------------------------------------------
    # Trim to what was actually captured (see kim_step_calibration.py).
    # -------------------------------------------------------------------------
    scan_images = scan_images[:idx]
    scan_commanded = scan_commanded[:idx]
    scan_moves = scan_moves[:idx]
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
            scan_moves[out_mask], scan_visibility[out_mask])
    if ret_mask.sum() >= 3:
        peak_ret, fwhm_ret = _estimate_envelope(
            scan_moves[ret_mask], scan_visibility[ret_mask])
    if not (np.isnan(peak_out) or np.isnan(peak_ret)):
        fwd_back_offset = peak_ret - peak_out

    # phase_measurement.py's own phase-shift sweep spans NUM_PIEZO_STEPS-1 of
    # its own STEP_SIZE moves (PM_STEP_SIZE above) -- i.e. 19 moves at its
    # default NUM_PIEZO_STEPS=20 (kept out of an import to avoid pulling in
    # that script's hardware-driver imports; keep in sync if it changes).
    # Comparing that against fwhm_out (also in moves) is only a fair,
    # distance-for-distance comparison when this scan's forward move is the
    # same physical distance as that script's move, i.e. STEP_SIZE_FWD ==
    # PM_STEP_SIZE.
    pm_sweep_moves = 20 - 1
    print("\n--- Coherence envelope summary ---")
    print(f"Outbound peak: {peak_out:+.2f} moves, FWHM {fwhm_out:.2f} moves")
    print(f"Return   peak: {peak_ret:+.2f} moves, FWHM {fwhm_ret:.2f} moves")
    print(f"Forward/backward peak offset (dead zone + asymmetry): "
          f"{fwd_back_offset:+.2f} moves")
    if not np.isnan(fwhm_out):
        if STEP_SIZE_FWD != PM_STEP_SIZE:
            print(f"NOTE: STEP_SIZE_FWD ({STEP_SIZE_FWD}) != PM_STEP_SIZE "
                  f"({PM_STEP_SIZE}) -- this scan's moves and "
                  f"phase_measurement.py's moves aren't the same physical "
                  f"distance, so the comparison below doesn't transfer.")
        print(f"phase_measurement.py's own sweep span "
              f"(NUM_PIEZO_STEPS-1={pm_sweep_moves} of its own moves) is "
              f"{100 * pm_sweep_moves / fwhm_out:.1f}% of the outbound FWHM.")

    description = (
        "KIM101/PIAK10 coherence-zone scan: fringe visibility (phase."
        "measure_frame_visibility, absolute/comparable across the whole scan -- "
        "not measure_frame_contrast, which only normalizes within one stack) vs. "
        "move index. Every commanded move in the whole run is exactly "
        "STEP_SIZE_FWD (forward) or STEP_SIZE_BWD (backward) -- no jump of any "
        "other size, see module docstring -- in 4 segments identified by "
        "'scan_segment': 0=approach (0->-RANGE_bwd), 1=outbound "
        "(-RANGE_bwd->+RANGE_fwd), 2=return (+RANGE_fwd->-RANGE_bwd), "
        "3=retreat (-RANGE_bwd->0), RANGE_bwd/RANGE_fwd each in that "
        "direction's own commanded-step units. 'scan_direction' is the "
        "physical move sign (+1/-1) and does not by itself distinguish e.g. "
        "approach from return (both -1) -- use 'scan_segment' for that. "
        "'scan_moves' is the analysis coordinate: net move count signed by "
        "direction, one unit per STEP_SIZE_FWD/STEP_SIZE_BWD burst -- unlike "
        "'scan_commanded' (raw open-loop commanded steps actually sent to the "
        "controller, in mixed units once STEP_SIZE_FWD != STEP_SIZE_BWD, and "
        "which does NOT return to 0 at the end of the run for the same "
        "reason -- see module docstring), 'scan_moves' is directly comparable "
        "across forward and backward segments because it's calibrated to "
        "equal physical distance per unit either direction. Peak position "
        "and FWHM (both in moves) are computed from 'scan_segment' 1 and 2 "
        "only (approach/retreat visibility is recorded but not used for the "
        "envelope fit) from 'scan_moves'; the forward/backward peak offset "
        "(= reversal dead zone + any residual asymmetry, in moves) is their "
        "difference. All are printed and saved as scalars below. "
        "'scan_counter' is the open-loop commanded-step counter "
        "(KIM101Axis.get_position()), not a position sensor. Use the located "
        "peak as the center for a finer re-scan (narrower RANGE_bwd/RANGE_fwd "
        "and/or smaller CAPTURE_STRIDE -- STEP_SIZE_FWD/STEP_SIZE_BWD stay "
        "fixed; STEP_SIZE_FWD should match phase_measurement.py's STEP_SIZE), "
        "or as the position to manually re-set the piezo to before a "
        "measurement sequence."
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path(__file__).resolve().parent / f"kim_coherence_scan_{timestamp}.npz"

    np.savez(
        save_path,
        scan_images=scan_images,              # (n_scan, H, W, C)
        scan_commanded=scan_commanded,        # (n_scan,) cumulative commanded steps from start (raw telemetry -- mixed FWD/BWD units, does not return to 0)
        scan_moves=scan_moves,                # (n_scan,) analysis coordinate: net move count signed by direction (equal physical distance per unit)
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
        range_bwd_steps=RANGE_bwd,
        range_fwd_steps=RANGE_fwd,
        step_size_fwd=STEP_SIZE_FWD,
        step_size_bwd=STEP_SIZE_BWD,
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
