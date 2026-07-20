"""Read-only piezo voltage/position check -- NO Set* calls, zero motion risk.

The previous calibration run drove the CT1P far and got stuck, traced to a
wrong voltage raw<->volts conversion in precisionpiezo.py (borrowed from the
wrong sibling controller). The correct, device-specific spec, read directly
from the installed C:\\Program Files\\Thorlabs\\Kinesis\\
Thorlabs.MotionControl.IntegratedPrecisionPiezo.h header, says:

    IPP_GetOutputVoltage / IPP_SetOutputVoltage:
        raw range -2184 to 30533  ==  -10V to 140V  (fixed linear mapping)
    IPP_GetMaxOutputVoltage: "(140V)", units of 1 tenth of a volt

But it's still unknown whether the .NET call this class actually uses
(self._device.GetOutputVoltage(), returning a System.Decimal) hands back RAW
counts (needing the linear formula above) or ALREADY-CONVERTED real volts
(Kinesis .NET wrappers often return real-world units directly via Decimal --
this turned out to be true for position). This script only *reads* -- no
SetPosition/SetOutputVoltage/move calls at all -- and prints every relevant
raw value so it can be compared against the Kinesis GUI's live display
(read there manually, without changing anything) to settle the question
empirically before precisionpiezo.py's voltage methods are fixed.

Run from the repo root:
    python scripts/phase/piezo_voltage_readonly_check.py
"""

from instruments.config import load_equipment
from instruments.precisionpiezo import PrecisionPiezoCT1P

PIEZO_SERIAL = load_equipment().piezo_serial

if __name__ == "__main__":
    with PrecisionPiezoCT1P(serial=PIEZO_SERIAL) as piezo:
        print("=== Position ===")
        raw_pos = piezo._from_decimal(piezo._device.GetPosition())
        print(f"  raw GetPosition()      = {raw_pos}")
        print(f"  get_position() (um)    = {piezo.get_position():.4f}")
        print(f"  max_travel_um          = {piezo.max_travel_um:.4f}")

        print("\n=== Voltage ===")
        raw_volt = piezo._from_decimal(piezo._device.GetOutputVoltage())
        print(f"  raw GetOutputVoltage()      = {raw_volt}")
        print(f"  get_voltage() (current model, likely wrong) = {piezo.get_voltage():.4f}")

        raw_max_volt = piezo._from_decimal(piezo._device.GetMaxOutputVoltage())
        print(f"  raw GetMaxOutputVoltage()   = {raw_max_volt}")
        print(f"  get_max_voltage() (current model, x0.1)     = {piezo.get_max_voltage():.4f}")

        if hasattr(piezo._device, "GetMinOutputVoltage"):
            raw_min_volt = piezo._from_decimal(piezo._device.GetMinOutputVoltage())
            print(f"  raw GetMinOutputVoltage()   = {raw_min_volt}")
            print(f"  GetMinOutputVoltage x0.1    = {raw_min_volt * 0.1:.4f}")
        else:
            print("  GetMinOutputVoltage: not found on this .NET object")

        print("\n=== Candidate interpretations of raw GetOutputVoltage ===")
        print(f"  A) already real volts (no conversion): {raw_volt:.4f} V")
        print(f"  B) old percentage-of-max model (raw/32767*max_volts, "
              f"max_volts via x0.1): {piezo.get_voltage():.4f} V")
        # C) IPP header's fixed linear raw<->volts mapping for GetOutputVoltage/SetOutputVoltage
        VOLT_RAW_MIN, VOLT_RAW_MAX = -2184, 30533
        VOLT_MIN_V, VOLT_MAX_V = -10.0, 140.0
        frac = (raw_volt - VOLT_RAW_MIN) / (VOLT_RAW_MAX - VOLT_RAW_MIN)
        volts_c = VOLT_MIN_V + frac * (VOLT_MAX_V - VOLT_MIN_V)
        print(f"  C) IPP header fixed linear mapping (-2184..30533 -> -10..140V): "
              f"{volts_c:.4f} V")

        print("\nNow compare against the Kinesis GUI's live open-loop voltage display "
              "(don't change anything in the GUI) to see which candidate (A/B/C) matches.")
