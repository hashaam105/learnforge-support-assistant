"""Console setup that survives a Windows terminal.

Rich falls back to a legacy Win32 renderer when it cannot detect VT support,
and that renderer encodes through the console's ANSI code page — so a single
box-drawing or tick character raises UnicodeEncodeError and takes the whole
run down. Losing an evaluation run to a tick mark is a silly way to fail, so
we force UTF-8 on the streams and keep terminal output ASCII besides.
"""

from __future__ import annotations

import sys

from rich.console import Console


def make_console(**kwargs) -> Console:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - detached streams
                pass
    return Console(**kwargs)
