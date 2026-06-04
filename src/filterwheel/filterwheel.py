import time
import warnings
import pyvisa


class FilterWheelControl:
    """Newport USFW-100 Universal Filter Wheel driver via PyVISA USB RAW.

    Confirmed protocol (tested 2026-06):
      FILT?      -> 'FILTn'  current position (n=1-6), 'FILT0' while moving
      FILTn      -> 'FILTn'  start move to position n (response is immediate)
    Positions 1-6 are valid. Device sends no termination bytes; reads rely on
    USB bulk-packet boundaries.
    """

    NUM_POSITIONS = 6

    def __init__(self, address: str):
        rm = pyvisa.ResourceManager()
        self._inst = rm.open_resource(address)
        self._inst.write_termination = '\r\n'
        self._inst.read_termination = '\r\n'
        self._inst.timeout = 3000

    def _cmd(self, cmd: str) -> str:
        """Write a command and return the response string."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            self._inst.write(cmd)
            return self._inst.read()

    def get_position(self) -> int:
        """Return current filter position (1-6), or 0 if the wheel is moving."""
        resp = self._cmd('FILT?')
        return int(resp.replace('FILT', ''))

    def set_position(self, pos: int, wait: bool = True, timeout: float = 5.0):
        """Move wheel to position pos (1-6).

        If wait=True, blocks until the wheel arrives or timeout (seconds) elapses.
        """
        if not 1 <= pos <= self.NUM_POSITIONS:
            raise ValueError(f'Position must be 1-{self.NUM_POSITIONS}, got {pos}')
        self._cmd(f'FILT{pos}')
        if wait:
            self._wait_move(pos, timeout)

    def _wait_move(self, target: int, timeout: float):
        """Block until FILT? returns target position.

        The device returns FILT0 while moving, but there is a brief window after
        issuing the command where FILT? still echoes the old position. Polling
        for the specific target avoids a false-early-exit in that window.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.get_position() == target:
                return
            time.sleep(0.1)
        raise TimeoutError(f'Filter wheel did not reach position {target} within {timeout}s')

    def close(self):
        self._inst.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
