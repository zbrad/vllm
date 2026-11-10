# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`uvloop.run` where uvloop exists; the stdlib asyncio loop on Windows.

uvloop has no Windows build. Call sites import this module as `uvloop` so
their `uvloop.run(...)` calls stay unchanged; `asyncio.run` takes the same
arguments (coroutine, `debug=`, `loop_factory=`).

On Windows the default loop is the Proactor loop, which lacks the
`add_reader` family that `zmq.asyncio` needs, so the selector loop is the
default there (a call site's own `loop_factory=` still wins).
"""

import sys

if sys.platform == "win32":
    from asyncio import SelectorEventLoop
    from asyncio import run as _asyncio_run
    from functools import partial

    run = partial(_asyncio_run, loop_factory=SelectorEventLoop)
else:
    from uvloop import run

__all__ = ["run"]
