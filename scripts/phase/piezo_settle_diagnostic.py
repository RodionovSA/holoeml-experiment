"""Piezo settle-time diagnostic (v3).

Connects only to the CT1P piezo (no camera). Prints the control mode
(open-loop/closed-loop) at every stage, in addition to raw and um position
readings, and runs two tests:

  Test A (no zero): assert closed loop, then run absolute-move steps
    (0.5/1.0/2.0 um) and poll each ~20s. Measures true closed-loop settle
    behavior without the zeroing complication.

  Test B (zero, then re-assert closed loop): zero() the stage, poll ~20s,
    print the mode right after zero (expected: it dropped to open loop --
    see below), re-assert closed loop, print mode again, then repeat the
    absolute-move steps from ~0 um.

Background: v2 of this script found that after zero() the stage read ~45um
and barely moved in response to SetPosition commands afterward, even though
zero() itself clearly swept the full range and settled at 0um (raw 0). Per
the Thorlabs Kinesis API docs (Thorlabs.MotionControl.Benchtop.PrecisionPiezo.h,
the CT1P's sibling controller): "SetPosition... is ignored if not in closed
loop mode", and "SetZero sets the voltage output to zero" -- i.e. zero()
drops the device into OPEN LOOP mode. v2 set closed loop *before* zero() but
never re-asserted it *after*, so every SetPosition during the move tests was
silently ignored, and the observed drift was just the piezo relaxing after
the zero sweep, not a response to commands. This version confirms that by
reading back the mode, and tests whether re-asserting closed loop after
zero() restores normal, fast servoing.

This is diagnostic-only: it pokes the Kinesis device handle directly (reusing
the class's own raw<->um scaling) so it can poll without being blocked by
PrecisionPiezoCT1P.move_to()'s internal wait.

Run from the repo root:
    python scripts/phase/piezo_settle_diagnostic.py
"""

import time
from datetime import datetime
from pathlib import Path

import numpy as np

from instruments.config import load_equipment
from instruments.precisionpiezo import PrecisionPiezoCT1P

PIEZO_SERIAL = load_equipment().piezo_serial

POLL_DURATION_S = 20.0
POLL_INTERVAL_S = 0.05
SETTLE_TOL_UM = 0.05

# Absolute targets (um) used for both Test A and Test B move sequences.
# Test A starts from wherever the stage currently is; Test B starts from ~0
# (right after zeroing), so the same relative steps (0.5/1.0/2.0) are applied
# on top of each test's own starting point.
STEP_SIZES_UM = [0.5, 1.0, 2.0]


def _raw_position(piezo: PrecisionPiezoCT1P):
    return piezo._from_decimal(piezo._device.GetPosition())


def _raw_max_travel(piezo: PrecisionPiezoCT1P):
    return piezo._from_decimal(piezo._device.GetMaxTravel())


def _mode_name(piezo: PrecisionPiezoCT1P) -> str:
    try:
        mode = piezo._device.GetPositionControlMode()
        return str(mode)
    except Exception as exc:  # noqa: BLE001 -- diagnostic best-effort
        return f"<error: {exc}>"


def _set_position_raw(piezo: PrecisionPiezoCT1P, target_um: float) -> None:
    """Command an absolute move without blocking on the class's wait."""
    raw = round(target_um / piezo.max_travel_um * piezo._FULL_SCALE)
    raw = max(0, min(piezo._FULL_SCALE, raw))
    piezo._device.SetPosition(piezo._to_decimal(raw))


def _poll(piezo: PrecisionPiezoCT1P, duration_s: float, interval_s: float, label: str):
    print(f"  polling '{label}' for {duration_s:.0f}s ...")
    t0 = time.monotonic()
    times, positions, raws = [], [], []
    deadline = t0 + duration_s
    while time.monotonic() < deadline:
        t = time.monotonic() - t0
        pos = piezo.get_position()
        raw = _raw_position(piezo)
        times.append(t)
        positions.append(pos)
        raws.append(raw)
        if len(times) <= 5 or len(times) % 20 == 0:
            print(f"    t={t:6.2f}s  raw={raw:8.1f}  pos={pos:8.3f} um")
        time.sleep(interval_s)
    return np.array(times), np.array(positions), np.array(raws)


def _settle_time(times: np.ndarray, positions: np.ndarray, target_um: float,
                  tol_um: float) -> float | None:
    within = np.abs(positions - target_um) <= tol_um
    for i in range(len(within)):
        if within[i:].all():
            return times[i]
    return None


def _run_move_sequence(piezo: PrecisionPiezoCT1P, prefix: str, start: float, results: dict):
    """Command cumulative absolute moves of STEP_SIZES_UM starting from `start`."""
    current = start
    for step_um in STEP_SIZES_UM:
        target = current + step_um
        if target > piezo.max_travel_um:
            print(f"  skipping step {step_um} um: target {target:.2f} exceeds max travel")
            continue
        print(f"\n  [{prefix}] move -> {target:.3f} um (step {step_um:+.3f} um), "
              f"mode={_mode_name(piezo)}")
        _set_position_raw(piezo, target)
        times, positions, raws = _poll(piezo, POLL_DURATION_S, POLL_INTERVAL_S,
                                        f"{prefix} step {step_um}um")
        final = positions[-1]
        residual = target - final
        settle_t = _settle_time(times, positions, target, SETTLE_TOL_UM)
        print(f"  final: raw={raws[-1]:.1f}  um={final:.3f}  residual={residual:+.3f} um  "
              f"mode={_mode_name(piezo)}")
        if settle_t is not None:
            print(f"  settled within {SETTLE_TOL_UM} um after {settle_t:.2f} s")
        else:
            print(f"  did NOT settle within {SETTLE_TOL_UM} um in {POLL_DURATION_S:.0f} s")
        results[f"{prefix}_step_{step_um}um"] = {
            "target": target, "step": step_um, "times": times, "positions": positions,
            "raws": raws, "final": final, "residual": residual, "settle_t": settle_t,
        }
        current = final


if __name__ == "__main__":
    all_results = {}

    with PrecisionPiezoCT1P(serial=PIEZO_SERIAL) as piezo:
        print(f"max_travel_um       = {piezo.max_travel_um:.3f}")
        print(f"max_travel_um (raw) = {_raw_max_travel(piezo):.1f}  (expect ~1600)")
        print(f"position at connect: raw={_raw_position(piezo):.1f}  "
              f"um={piezo.get_position():.3f}  mode={_mode_name(piezo)}")

        # ------------------------------------------------------------
        # Test A: closed loop, no zero -- baseline servo behavior.
        # ------------------------------------------------------------
        print("\n=== Test A: set closed loop (no zero), then move ===")
        piezo.set_closed_loop()
        time.sleep(0.5)
        print(f"mode after set_closed_loop: {_mode_name(piezo)}")
        start_a = piezo.get_position()
        print(f"starting position: {start_a:.3f} um")
        _run_move_sequence(piezo, "A", start_a, all_results)

        # ------------------------------------------------------------
        # Test B: zero(), observe mode flip, re-assert closed loop, move.
        # ------------------------------------------------------------
        print("\n=== Test B: zero(), then re-assert closed loop, then move ===")
        piezo.zero()
        zero_t, zero_pos, zero_raw = _poll(piezo, POLL_DURATION_S, POLL_INTERVAL_S, "zero()")
        print(f"mode right after zero() settle: {_mode_name(piezo)}  "
              f"(expect open-loop -- SetZero drops the device out of closed loop)")

        piezo.set_closed_loop()
        time.sleep(0.5)
        print(f"mode after re-asserting set_closed_loop: {_mode_name(piezo)}")
        start_b = piezo.get_position()
        print(f"starting position: {start_b:.3f} um")
        _run_move_sequence(piezo, "B", start_b, all_results)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path(__file__).resolve().parent / f"piezo_settle_diagnostic_{timestamp}.npz"
    save_kwargs = {
        "zero_times": zero_t, "zero_positions": zero_pos, "zero_raws": zero_raw,
        "step_sizes_um": np.array(STEP_SIZES_UM),
    }
    for key, r in all_results.items():
        save_kwargs[f"{key}_times"] = r["times"]
        save_kwargs[f"{key}_positions"] = r["positions"]
        save_kwargs[f"{key}_raws"] = r["raws"]
        save_kwargs[f"{key}_target"] = r["target"]
        save_kwargs[f"{key}_step"] = r["step"]
        save_kwargs[f"{key}_final"] = r["final"]
        save_kwargs[f"{key}_residual"] = r["residual"]
        save_kwargs[f"{key}_settle_t"] = r["settle_t"] if r["settle_t"] is not None else np.nan
    np.savez(save_path, **save_kwargs)
    print(f"\nSaved diagnostic traces -> {save_path}")
