"""Serial wrapper for the Arduino monochromator firmware.

Windows-only at runtime (uses ``msvcrt`` for non-blocking keypress
detection during wavelength moves). The module imports cleanly on
other platforms for static analysis, but ``set_wavelength`` will raise
if invoked without ``msvcrt`` available.

Talks to ``monochromator_3modes.ino`` over serial using the
``OK:`` / ``ERR:`` / ``INFO:`` line protocol documented in the firmware
header. Typical usage::

    mono = MonochromatorControl(port="COM4")
    mono.initialize_arduino()
    mono.home_motor()
    mono.select_grating_mode("VIS Grating")
    mono.set_wavelength(500)
    mono.disconnect()

Or as a context manager::

    with MonochromatorControl(port="COM4") as mono:
        mono.initialize_arduino()
        ...
"""

import time

import serial
import serial.tools.list_ports

try:
    import msvcrt  # Windows-only
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None


class MonochromatorControl:
    """High-level controller for the Arduino-driven monochromator."""

    # ---- Defaults / constants --------------------------------------
    DEFAULT_BAUD       = 9600
    DEFAULT_TIMEOUT    = 1     # seconds; pyserial read timeout
    RESPONSE_TIMEOUT   = 30    # seconds to wait for a typical OK/ERR
    HOMING_TIMEOUT     = 120   # seconds; homing traverses the full range
    EMERGENCY_STOP_KEY = "e"

    VALID_MODES = ("VIS Grating", "IR Grating", "Switch Mode")
    MODE_VALUES = {"VIS Grating": 0, "IR Grating": 1, "Switch Mode": 2}

    # =================================================================
    # Lifecycle
    # =================================================================
    def __init__(self, port=None, baud_rate=DEFAULT_BAUD, timeout=DEFAULT_TIMEOUT):
        """Open the serial connection to the Arduino.

        If ``port`` is ``None``, interactively prompt the user to pick
        one from the available COM ports.
        """
        self.ser = None
        self.arduino_initialized = False
        self.motor_homed = False
        self.grating_selected = False

        if port is None:
            port = self._prompt_for_port()
            if port is None:
                return
        self.connect_to_monochromator(port, baud_rate, timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()

    # =================================================================
    # COM port management
    # =================================================================
    @staticmethod
    def list_available_ports():
        """Return a list of detected serial port device names."""
        return [p.device for p in serial.tools.list_ports.comports()]

    def connect_to_monochromator(self, selected_port=None,
                                 baud_rate=DEFAULT_BAUD,
                                 timeout=DEFAULT_TIMEOUT):
        """Open the serial connection. Prompts for a port if not given;
        retries on failure."""
        while not self.ser:
            if not selected_port:
                selected_port = self._prompt_for_port()
                if not selected_port:
                    return
            try:
                self.ser = serial.Serial(selected_port, baud_rate, timeout=timeout)
                time.sleep(0.1)  # let the OS settle the connection
                print(f"Connected to {selected_port}; you can now initialize the Monochromator.")
            except serial.SerialException:
                print(f"Could not open {selected_port}. Please try again.")
                selected_port = None

    def disconnect(self):
        """Close the serial connection. Safe to call multiple times."""
        if self.ser is None:
            return
        try:
            self.ser.close()
            print("Disconnected from Arduino.")
        except Exception as e:
            print(f"Error disconnecting: {e}")
        finally:
            self.ser = None

    # =================================================================
    # Commands
    # =================================================================
    def initialize_arduino(self):
        """Drain the Arduino's boot banner and mark the device ready."""
        if not self._ready():
            return
        # Arduino resets on serial open (DTR) and takes ~1-2 s to boot.
        # Wait long enough for the boot banner to fully arrive before draining.
        time.sleep(2.0)
        while self.ser.in_waiting > 0:
            line = self.ser.readline().decode(errors="replace").strip()
            if line:
                print(line)
        self.arduino_initialized = True

    def home_motor(self):
        """Home the stepper against the photodiode flag. Blocks until
        the Arduino reports ``OK: homed`` (or times out / errors)."""
        if not self._ready(require_init=True):
            return
        self._send("home")
        outcome, _ = self._wait_for(("OK: homed",), timeout=self.HOMING_TIMEOUT)
        if outcome == "ok":
            self.motor_homed = True

    def select_grating_mode(self, mode=None):
        """Select ``"VIS Grating"``, ``"IR Grating"``, or ``"Switch Mode"``.
        Prompts interactively if ``mode`` is ``None``."""
        if not self._ready(require_init=True, require_homed=True):
            return

        if mode is None:
            mode = self._prompt_for_mode()
            if mode is None:
                return
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid grating mode {mode!r}. Must be one of {list(self.VALID_MODES)}."
            )

        self._send(f"mode {self.MODE_VALUES[mode]}")
        outcome, _ = self._wait_for(("OK: mode",))
        if outcome == "ok":
            self.grating_selected = True

    def set_wavelength(self, wavelength=None):
        """Move the stepper to ``wavelength`` (nm). Prompts if not given.
        Press the emergency-stop key ('e' by default) during the move
        to interrupt it."""
        if not self._ready(require_init=True, require_homed=True, require_mode=True):
            return
        if msvcrt is None:
            raise RuntimeError(
                "set_wavelength requires Windows (msvcrt unavailable)."
            )

        if wavelength is None:
            wavelength = input("Enter the target wavelength (nm): ").strip()
        try:
            wavelength = float(wavelength)
        except (TypeError, ValueError):
            print(f"Invalid wavelength: {wavelength!r}")
            return

        self._send(f"wavelength {wavelength}")

        # Busy-poll so the emergency-stop keypress is detected with
        # minimal latency. _wait_for is not used here because its
        # blocking readline() would delay the keypress check.
        terminal_prefixes = ("OK: wavelength", "OK: stopped", "ERR:")
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode("utf-8", errors="ignore").lower()
                if key == self.EMERGENCY_STOP_KEY:
                    self._send("stop")
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode(errors="replace").strip()
                if not line:
                    continue
                print(line)
                if any(line.startswith(p) for p in terminal_prefixes):
                    return

    def stop(self):
        """Send an unsolicited ``stop``. Useful from another thread."""
        if self.ser is not None:
            self._send("stop")

    def status(self):
        """Query Arduino state. Returns the raw ``INFO:`` status line,
        or ``None`` if no connection / timeout."""
        if not self._ready():
            return None
        self._send("status")
        _, line = self._wait_for(("INFO: mode=",))
        return line

    # =================================================================
    # Internals
    # =================================================================
    def _ready(self, *, require_init=False, require_homed=False, require_mode=False):
        """Guard helper: print a hint and return False if a precondition
        for the calling command is not met."""
        if not self.ser:
            print("COM port is not selected.")
            return False
        if require_init and not self.arduino_initialized:
            print("The Arduino is off, please press START.")
            return False
        if require_homed and not self.motor_homed:
            print("Please home the motor first to start operation.")
            return False
        if require_mode and not self.grating_selected:
            print("Please select a grating mode first.")
            return False
        return True

    def _send(self, command):
        """Write a single newline-terminated command to the Arduino."""
        self.ser.write(f"{command}\n".encode())

    def _wait_for(self, ok_prefixes, err_prefixes=("ERR:",), timeout=None):
        """Read serial lines (printing each) until one starts with a
        member of ``ok_prefixes`` or ``err_prefixes``, or until
        ``timeout`` seconds elapse.

        Returns ``(status, line)`` where ``status`` is one of
        ``"ok"``, ``"err"``, ``"timeout"`` and ``line`` is the matching
        line (``None`` on timeout).
        """
        deadline = time.monotonic() + (timeout if timeout is not None else self.RESPONSE_TIMEOUT)
        while time.monotonic() < deadline:
            line = self.ser.readline().decode(errors="replace").strip()
            if not line:
                continue
            print(line)
            if any(line.startswith(p) for p in ok_prefixes):
                return "ok", line
            if any(line.startswith(p) for p in err_prefixes):
                return "err", line
        print(f"Timed out waiting for one of {list(ok_prefixes)}")
        return "timeout", None

    @classmethod
    def _prompt_for_port(cls):
        """Interactively ask the user to pick a COM port."""
        ports = cls.list_available_ports()
        if not ports:
            print("No COM ports available.")
            return None
        print(f"Available COM ports: {ports}")
        return input("Enter the COM port to connect to (e.g., COM3): ").strip() or None

    def _prompt_for_mode(self):
        """Interactively ask the user to pick a grating mode. Returns
        ``None`` if the user declines to change an already-selected mode."""
        if self.grating_selected:
            confirm = input(
                "A grating mode is already selected. Change it? (y/n): "
            ).strip().lower()
            if confirm != "y":
                print("Grating mode change cancelled.")
                return None
        while True:
            mode = input(f"Select a grating mode {list(self.VALID_MODES)}: ").strip()
            if mode in self.VALID_MODES:
                return mode
            print("Invalid grating mode. Please try again.")
