"""Piezo open-loop voltage-to-position calibration.

Closed-loop position control on the CT1P is accurate but settles very slowly
(confirmed both in our own diagnostics and directly in the Kinesis GUI).
Open-loop voltage control settles quickly. This script measures the actual
sensor position (get_position() -- the position sensor keeps reporting real
position even while driving in open loop) at a series of commanded voltages,
fits a line position_um = slope * volts + intercept, and saves the fit for
use by piezo_step_response.py to convert desired um steps into volts.

Sweeps a SMALL, LOCAL range of voltage around the current value (not the
device's full -10..140V range): the acquisition script only ever needs
~74 um of total travel, and an earlier attempt at a wide 10-90%-of-max sweep
drove the stage to its travel limit and got stuck (that attempt also used a
since-fixed voltage unit bug in precisionpiezo.py -- see git history). This
version steps outward from the current voltage in both directions and stops
early (in that direction) if the position saturates (stops responding) or
hits the travel limits, and always returns to the starting voltage when done.

Run from the repo root:
    python scripts/phase/piezo_voltage_calibration.py
"""

import time
from datetime import datetime
from pathlib import Path

import numpy as np

from instruments.config import load_equipment
from instruments.precisionpiezo import PrecisionPiezoCT1P

PIEZO_SERIAL = load_equipment().piezo_serial

# Local sweep parameters: step outward from the current voltage by
# VOLT_STEP_V, up to N_STEPS_EACH_SIDE steps in each direction. Kept fine
# (0.2V) because a coarser 1.0V step previously appeared to "saturate"
# instantly in one direction after a single jump -- but the user separately
# confirmed smooth bidirectional motion in the Kinesis GUI using 0.1V steps,
# so that was likely an artifact of too-large/fast a single jump (e.g. a
# settle-timing issue), not a real physical limit. Fine steps avoid that.
VOLT_STEP_V = 0.2
N_STEPS_EACH_SIDE = 20

# Stop stepping further in a direction if the position moved less than this
# between consecutive points, for N_SATURATED_TO_STOP consecutive points in a
# row (a single flat reading isn't trusted -- it can be a settle-timing
# artifact rather than genuine saturation).
SATURATION_TOL_UM = 0.03
N_SATURATED_TO_STOP = 3

# Stop stepping further in a direction if within this margin of the travel
# limits (0 or max_travel_um).
TRAVEL_MARGIN_UM = 2.0

# Settle-detection: wait until N_STABLE consecutive position samples agree
# within STABLE_TOL_UM, or give up after STABLE_TIMEOUT_S.
POLL_INTERVAL_S = 0.02
N_STABLE = 10
STABLE_TOL_UM = 0.01
STABLE_TIMEOUT_S = 5.0


def _wait_stable(piezo: PrecisionPiezoCT1P) -> tuple[float, bool]:
    """Poll get_position() until N_STABLE consecutive samples agree within
    STABLE_TOL_UM, or STABLE_TIMEOUT_S elapses. Returns (final_position, settled)."""
    deadline = time.monotonic() + STABLE_TIMEOUT_S
    window: list[float] = []
    pos = piezo.get_position()
    while time.monotonic() < deadline:
        pos = piezo.get_position()
        window.append(pos)
        if len(window) > N_STABLE:
            window.pop(0)
        if len(window) == N_STABLE and (max(window) - min(window)) <= STABLE_TOL_UM:
            return pos, True
        time.sleep(POLL_INTERVAL_S)
    return pos, False


def _sweep_direction(piezo: PrecisionPiezoCT1P, start_volts: float, sign: int,
                      min_volts: float, max_volts: float) -> tuple[list, list]:
    """Step from start_volts in direction `sign` (+1 or -1), stopping early on
    saturation, travel-limit proximity, or the voltage bound. Returns
    (volts_list, positions_list) NOT including the start point itself."""
    volts_list, positions_list = [], []
    last_pos, _ = _wait_stable(piezo)
    v = start_volts
    n_flat = 0
    for _ in range(N_STEPS_EACH_SIDE):
        v_next = v + sign * VOLT_STEP_V
        if not (min_volts <= v_next <= max_volts):
            print(f"    stop: next voltage {v_next:.2f} V outside [{min_volts:.2f}, "
                  f"{max_volts:.2f}]")
            break
        piezo.set_voltage(v_next)
        pos, settled = _wait_stable(piezo)
        flag = "" if settled else "  (did not settle within timeout)"
        print(f"    {v_next:7.3f} V -> {pos:8.3f} um{flag}")

        if pos <= TRAVEL_MARGIN_UM or pos >= piezo.max_travel_um - TRAVEL_MARGIN_UM:
            print(f"    stop: position {pos:.2f} um within {TRAVEL_MARGIN_UM} um of a "
                  f"travel limit")
            volts_list.append(v_next)
            positions_list.append(pos)
            break

        volts_list.append(v_next)
        positions_list.append(pos)

        if abs(pos - last_pos) <= SATURATION_TOL_UM:
            n_flat += 1
            if n_flat >= N_SATURATED_TO_STOP:
                print(f"    stop: position flat for {N_SATURATED_TO_STOP} consecutive "
                      f"points (<= {SATURATION_TOL_UM} um each)")
                break
        else:
            n_flat = 0

        last_pos = pos
        v = v_next
    return volts_list, positions_list


if __name__ == "__main__":
    with PrecisionPiezoCT1P(serial=PIEZO_SERIAL) as piezo:
        piezo.set_open_loop()
        time.sleep(0.2)

        min_volts = piezo.get_min_voltage()
        max_volts = piezo.get_max_voltage()
        start_volts = piezo.get_voltage()
        start_pos, _ = _wait_stable(piezo)
        print(f"device voltage range: [{min_volts:.2f}, {max_volts:.2f}] V")
        print(f"starting at {start_volts:.3f} V, {start_pos:.3f} um")

        print("\nSweeping up:")
        volts_up, positions_up = _sweep_direction(
            piezo, start_volts, +1, min_volts, max_volts)

        print("Returning to start before sweeping down ...")
        piezo.set_voltage(start_volts)
        _wait_stable(piezo)

        print("\nSweeping down:")
        volts_down, positions_down = _sweep_direction(
            piezo, start_volts, -1, min_volts, max_volts)

        print("\nReturning to starting voltage ...")
        piezo.set_voltage(start_volts)
        final_pos, _ = _wait_stable(piezo)
        print(f"final position: {final_pos:.3f} um (started at {start_pos:.3f} um)")

    # -------------------------------------------------------------------
    # Combine both directions (plus the start point) and fit a line.
    # -------------------------------------------------------------------
    volts = np.array([start_volts] + volts_down[::-1] + volts_up)
    positions = np.array([start_pos] + positions_down[::-1] + positions_up)
    order = np.argsort(volts)
    volts, positions = volts[order], positions[order]

    if len(volts) < 3:
        raise RuntimeError(
            f"Only {len(volts)} usable calibration points (need >= 3) -- the sweep "
            f"stopped almost immediately in both directions. Inspect the printed stop "
            f"reasons above; the local range (VOLT_STEP_V x N_STEPS_EACH_SIDE) may need "
            f"to be smaller, or the stage may be sitting very close to a travel limit."
        )

    slope, intercept = np.polyfit(volts, positions, 1)
    fit = slope * volts + intercept
    residuals = positions - fit
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((positions - positions.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"\nFit over {len(volts)} points ({volts.min():.2f} to {volts.max():.2f} V): "
          f"position_um = {slope:.4f} * volts + {intercept:.4f}")
    print(f"  R^2 = {r_squared:.5f}")
    print(f"  residual: max |.| = {np.abs(residuals).max():.4f} um, "
          f"rms = {np.sqrt(np.mean(residuals ** 2)):.4f} um")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path(__file__).resolve().parent / f"piezo_voltage_calibration_{timestamp}.npz"
    np.savez(
        save_path,
        volts=volts,
        positions=positions,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        min_voltage=min_volts,
        max_voltage=max_volts,
        start_voltage=start_volts,
        start_position_um=start_pos,
        piezo_serial=PIEZO_SERIAL,
    )
    print(f"\nSaved calibration -> {save_path}")
