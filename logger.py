"""logger.py -- Coloured console logging for Windows 11."""

import logging
import sys
import io
import ctypes
import ctypes.wintypes
from datetime import datetime


def _enable_ansi() -> bool:
    """Enable ANSI virtual terminal processing on Windows 10+."""
    try:
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)
        m = ctypes.wintypes.DWORD()
        k.GetConsoleMode(h, ctypes.byref(m))
        k.SetConsoleMode(h, m.value | 0x0004)
        k.SetConsoleOutputCP(65001)
        return True
    except Exception:
        return False


_ANSI = _enable_ansi()

_COLOURS = {
    "DEBUG":    "\033[36m",
    "INFO":     "\033[32m",
    "WARNING":  "\033[33m",
    "ERROR":    "\033[31m",
    "CRITICAL": "\033[35m",
}
_RESET = "\033[0m"


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Build timestamp manually -- %f not supported by time.strftime on Windows
        now = datetime.fromtimestamp(record.created)
        ts  = now.strftime("%H:%M:%S") + f".{now.microsecond // 1000:03d}"
        name  = f"{record.name:<12}"
        level = f"{record.levelname:<8}"
        msg   = record.getMessage()
        if _ANSI:
            c = _COLOURS.get(record.levelname, "")
            return f"{c}{ts}  {name}  {level}{_RESET}  {msg}"
        return f"{ts}  {name}  {level}  {msg}"


def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        h = logging.StreamHandler(stream)
        h.setFormatter(_Formatter())
        logger.addHandler(h)
        logger.propagate = False
    return logger
