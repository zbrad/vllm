# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`uvloop.run` where uvloop exists; the stdlib asyncio loop on Windows.

uvloop has no Windows build. Call sites import this module as `uvloop` so
their `uvloop.run(...)` calls stay unchanged; `asyncio.run` takes the same
arguments (coroutine, `debug=`, `loop_factory=`).
"""

import sys

if sys.platform == "win32":
    from asyncio import run
else:
    from uvloop import run

__all__ = ["run"]
