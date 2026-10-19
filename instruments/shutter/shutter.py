import time
import serial


class ShutterControl:
    """Thorlabs SC30 two-channel shutter controller driver via pyserial.

    The SC30 enumerates as a USB virtual COM port (VID 0x1313, PID 0x100E).
    Thorlabs ships only a 32-bit C++ SDK (TL_DeviceSDK), but that SDK merely
    forwards ASCII "HyperTerminal" commands to the COM port, so we talk to the
    port directly.

    Confirmed protocol (tested 2026-10, firmware Mainboard v1.20):
      Commands are ASCII terminated by '\\r'; the device answers with
      '\\r<CMD>? <value>\\r>' for queries and '\\r>' for set commands.
      'UNKNOWN COMMAND!' -> unknown/garbled command (no trailing prompt)
      '!'                -> valid command refused in the current state
      MODE?           -> 0 Manual, 1 External, 2 Internal, 3 Triggered.
                         Global to both channels; MODE= is only accepted
                         while locked (UNLOCKED=0), and disables both channels.
      UNLOCKED?       -> 0 when the master lock forces all shutters closed;
                         ENABLE= is refused while locked, and UNLOCKED=n is
                         refused when n equals the current state
      ENABLEx=n       -> Manual: opens (1) / closes (0) shutter x.
                         Internal: starts (1) / stops (0) the free-run.
                         Triggered: arms (1) / disarms (0) the channel.
      SHUTTERSTATEx?  -> 0 closed, 1 open (actual shutter state)
      SHUTTERCYCCOMPx? -> cycles completed in the current run (reset on start)
    Out-of-range timing values are clamped silently by the controller to the
    shutter's limits (SHB05: max 25 Hz, duty range narrows with frequency,
    e.g. 28-77 % at 10 Hz), so configure_* methods return the read-back values.
    Send '?' over a terminal for the full command menu.
    """

    CHANNELS = (1, 2)
    MODES = {'manual': 0, 'external': 1, 'internal': 2, 'triggered': 3}
    TRIGGER_SOURCES = {'t1': 0, 't2': 1, 'sw': 2}
    PROMPT = b'>'

    def __init__(self, port: str, timeout: float = 1.0):
        self._ser = serial.Serial(port, 115200, timeout=timeout)
        self._ser.reset_input_buffer()

    def _read_reply(self) -> str:
        """Read until the '>' prompt or a '!' rejection marker (or timeout)."""
        buf = bytearray()
        while True:
            b = self._ser.read(1)
            if not b:
                break
            buf += b
            if b in (self.PROMPT, b'!'):
                break
        return buf.decode(errors='replace')

    def _cmd(self, cmd: str, retries: int = 1) -> str:
        """Write a command and return the response with echo/prompt stripped.

        Queries are retried once on a garbled reply (seen once during a fast
        poll while a shutter was cycling); set commands are never retried.
        """
        is_query = cmd.endswith('?')
        self._ser.write(f'{cmd}\r'.encode())
        raw = self._read_reply()
        resp = raw[:-1].strip()
        # Queries echo the command: 'SHUTTERSTATE1? 1' -> '1'
        garbled = 'UNKNOWN COMMAND' in raw or (
            is_query and raw.endswith('>') and not resp.upper().startswith(cmd.upper()))
        if garbled and is_query and retries > 0:
            time.sleep(0.05)
            self._ser.reset_input_buffer()
            return self._cmd(cmd, retries - 1)
        if 'UNKNOWN COMMAND' in raw:
            raise ValueError(f'SC30 rejected command {cmd!r}')
        if raw.endswith('!'):
            raise RuntimeError(f'SC30 refused {cmd!r} in its current state '
                               f'(mode={self._cmd("mode?")}, unlocked={self._cmd("unlocked?")})')
        if not raw.endswith('>'):
            raise TimeoutError(f'No response from SC30 to {cmd!r} (got {raw!r})')
        if is_query:
            resp = resp[len(cmd):].strip()
        return resp

    def _check_channel(self, channel: int):
        if channel not in self.CHANNELS:
            raise ValueError(f'Channel must be one of {self.CHANNELS}, got {channel}')

    # ── Device info ──────────────────────────────────────────────────────
    def identify(self) -> str:
        """Return the ID string, e.g. 'SC30 v1.20 : Display v1.20 : Serial Number ...'."""
        return self._cmd('id?')

    def get_serial(self) -> str:
        return self._cmd('serial?')

    def get_shutter_model(self, channel: int) -> str:
        """Return the shutter model configured on a channel (e.g. 'SHB05')."""
        self._check_channel(channel)
        return self._cmd(f'shutter{channel}?')

    def get_error(self, channel: int) -> int:
        """Return the active shutter error bitfield for a channel."""
        self._check_channel(channel)
        return int(self._cmd(f'shuttererror{channel}?'))

    def is_unlocked(self) -> bool:
        """False when the master lock forces all shutters closed."""
        return self._cmd('unlocked?') == '1'

    def lock(self):
        """Engage the master lock: all shutters are forced closed."""
        # Re-sending the current lock state is refused with '!'
        if self.is_unlocked():
            self._cmd('unlocked=0')

    def unlock(self):
        """Release the master lock so channels can be enabled again."""
        if not self.is_unlocked():
            self._cmd('unlocked=1')

    # ── Mode ─────────────────────────────────────────────────────────────
    def get_mode(self) -> str:
        code = int(self._cmd('mode?'))
        return next(name for name, value in self.MODES.items() if value == code)

    def set_mode(self, mode: str, unlock: bool = True):
        """Set operating mode: 'manual', 'external', 'internal' or 'triggered'.

        The mode is global to both channels. The SC30 only accepts a mode
        change while locked, so this locks the controller first, which forces
        BOTH shutters closed, and both channels come out of the switch
        disabled. With unlock=True (default) the lock is released afterwards
        so open()/start() work.
        """
        if mode not in self.MODES:
            raise ValueError(f'Mode must be one of {list(self.MODES)}, got {mode!r}')
        if self.get_mode() != mode:
            self.lock()
            self._cmd(f'mode={self.MODES[mode]}')
        if unlock:
            self.unlock()

    def _require_mode(self, *modes: str):
        mode = self.get_mode()
        if mode not in modes:
            raise RuntimeError(f'SC30 must be in {" or ".join(modes)} mode for this (is {mode!r})')

    def _require_unlocked(self):
        if not self.is_unlocked():
            raise RuntimeError('SC30 is locked; unlock it on the front panel or via unlock()')

    # ── Manual mode: shutter state ───────────────────────────────────────
    def is_open(self, channel: int) -> bool:
        """Return the actual state of the shutter on a channel."""
        self._check_channel(channel)
        return self._cmd(f'shutterstate{channel}?') == '1'

    def set_open(self, channel: int, state: bool, wait: bool = True, timeout: float = 2.0):
        """Open (state=True) or close (state=False) a shutter in Manual mode.

        If wait=True, blocks until SHUTTERSTATE confirms the new state or
        timeout (seconds) elapses.
        """
        self._check_channel(channel)
        self._require_mode('manual')
        if state:
            self._require_unlocked()
        self._cmd(f'enable{channel}={int(state)}')
        if wait:
            self._wait_state(channel, state, timeout)

    def open(self, channel: int, wait: bool = True):
        self.set_open(channel, True, wait)

    def shut(self, channel: int, wait: bool = True):
        self.set_open(channel, False, wait)

    def open_all(self, wait: bool = True):
        for ch in self.CHANNELS:
            self.set_open(ch, True, wait)

    def shut_all(self, wait: bool = True):
        for ch in self.CHANNELS:
            self.set_open(ch, False, wait)

    def _wait_state(self, channel: int, state: bool, timeout: float):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_open(channel) == state:
                return
            time.sleep(0.02)
        word = 'open' if state else 'close'
        raise TimeoutError(f'Shutter {channel} did not {word} within {timeout}s '
                           f'(error code {self.get_error(channel)})')

    # ── Internal / triggered mode: timing ────────────────────────────────
    def _set_timing(self, prefix: str, channel: int, freq_hz, duty, open_ms, close_ms) -> dict:
        """Write timing as either freq/duty or open/close times; return read-back."""
        by_freq = freq_hz is not None or duty is not None
        by_time = open_ms is not None or close_ms is not None
        if by_freq == by_time:
            raise ValueError('Give either freq_hz/duty or open_ms/close_ms')
        c = self._cmd
        if by_freq:
            c(f'{prefix}signal{channel}=1')
            if freq_hz is not None:
                c(f'{prefix}freq{channel}={freq_hz:g}')
            if duty is not None:
                c(f'{prefix}duty{channel}={duty:g}')
            return {'freq_hz': float(c(f'{prefix}freq{channel}?')),
                    'duty': float(c(f'{prefix}duty{channel}?'))}
        c(f'{prefix}signal{channel}=0')
        for name, value in (('open', open_ms), ('close', close_ms)):
            if value is not None:
                c(f'{prefix}{name}uom{channel}=0')     # ms
                c(f'{prefix}{name}{channel}={value:g}')
        return {f'{name}_ms': self._read_ms(prefix, name, channel) for name in ('open', 'close')}

    def _read_ms(self, prefix: str, name: str, channel: int) -> float:
        value = float(self._cmd(f'{prefix}{name}{channel}?'))
        return value * 1000 if self._cmd(f'{prefix}{name}uom{channel}?') == '1' else value

    def configure_internal(self, channel: int, *, freq_hz: float = None, duty: float = None,
                           open_ms: float = None, close_ms: float = None,
                           cycles: int = 0) -> dict:
        """Set the free-run timing of a channel for Internal mode.

        Timing is given either as freq_hz (0.1-50, limited by the shutter) and
        duty (% open) or as open_ms/close_ms. cycles=0 runs until stop().
        Returns the values the controller actually applied (it clamps silently).
        """
        self._check_channel(channel)
        applied = self._set_timing('int', channel, freq_hz, duty, open_ms, close_ms)
        self._cmd(f'intcyclecnt{channel}={int(cycles)}')
        applied['cycles'] = int(self._cmd(f'intcyclecnt{channel}?'))
        return applied

    def configure_triggered(self, channel: int, *, freq_hz: float = None, duty: float = None,
                            open_ms: float = None, close_ms: float = None,
                            cycles: int = 1, source: str = 'sw', delay_ms: float = 0) -> dict:
        """Set the sequence a channel runs on each trigger in Triggered mode.

        Timing as in configure_internal; cycles is the number of open/close
        cycles per trigger. source: 't1' / 't2' (rising edge on the TRIGGER
        BNC) or 'sw' (software trigger via trigger()). delay_ms is the delay
        between the trigger and the start of the sequence.
        Returns the values the controller actually applied.
        """
        self._check_channel(channel)
        if source not in self.TRIGGER_SOURCES:
            raise ValueError(f'source must be one of {list(self.TRIGGER_SOURCES)}, got {source!r}')
        applied = self._set_timing('trig', channel, freq_hz, duty, open_ms, close_ms)
        c = self._cmd
        c(f'trigcyclecnt{channel}={int(cycles)}')
        c(f'trigtrigsrc{channel}={self.TRIGGER_SOURCES[source]}')
        c(f'trigtrigdlyuom{channel}=0')     # ms
        c(f'trigtrigdly{channel}={delay_ms:g}')
        src_code = int(c(f'trigtrigsrc{channel}?'))
        applied.update(
            cycles=int(c(f'trigcyclecnt{channel}?')),
            source=next(k for k, v in self.TRIGGER_SOURCES.items() if v == src_code),
            delay_ms=self._read_ms('trig', 'trigdly', channel))
        return applied

    # ── Internal / triggered mode: run control ───────────────────────────
    def start(self, channel: int):
        """Internal mode: start free-running. Triggered mode: arm the channel."""
        self._check_channel(channel)
        self._require_mode('internal', 'triggered')
        self._require_unlocked()
        self._cmd(f'enable{channel}=1')

    def stop(self, channel: int):
        """Stop (Internal) or disarm (Triggered) a channel; the shutter closes."""
        self._check_channel(channel)
        self._cmd(f'enable{channel}=0')

    def trigger(self, channel: int):
        """Fire a software trigger on an armed channel (Triggered mode, source='sw')."""
        self._check_channel(channel)
        self._require_mode('triggered')
        self._cmd(f'swtrig{channel}=1')

    def get_completed_cycles(self, channel: int) -> int:
        """Return the number of cycles completed in the current run."""
        self._check_channel(channel)
        return int(self._cmd(f'shuttercyccomp{channel}?'))

    def wait_cycles(self, channel: int, n: int, timeout: float = 10.0):
        """Block until at least n cycles have completed in the current run."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.get_completed_cycles(channel) >= n:
                return
            time.sleep(0.02)
        raise TimeoutError(f'Channel {channel} completed {self.get_completed_cycles(channel)} '
                           f'of {n} cycles within {timeout}s')

    # ── Lifecycle ────────────────────────────────────────────────────────
    def close(self):
        """Close the serial port. Shutter states are left unchanged."""
        self._ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
