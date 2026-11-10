# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys

# HTTP header limits for h11 parser
# These constants help mitigate header abuse attacks
H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT = 4194304  # 4 MB
H11_MAX_HEADER_COUNT_DEFAULT = 256

# Open HTTP connections an API server accepts at once (None: no limit). On
# Windows the event loop is a select() SelectorEventLoop, which raises
# "too many file descriptors in select()" past 511 sockets and takes the server
# down; the default stays well under that, leaving room for the listener and
# the engine's zmq sockets.
MAX_CONNECTIONS_DEFAULT = 256 if sys.platform == "win32" else None
