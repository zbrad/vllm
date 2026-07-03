# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility shim for importing PyNvVideoCodec on Python 3.13+.

Upstream bug: pynvvideocodec 2.0.4's ``__init__.py`` does
``from ast import Str`` at module scope. ``ast.Str`` was removed from the
standard library ``ast`` module; the import is otherwise dead code (never
referenced anywhere else in that file) but still raises ImportError before
any of the package's real functionality can be reached. This is the only
PyPI release with cp313/cp314 wheels -- 2.1.0 (latest) has none. Filed
upstream on the NVIDIA Developer Forums; no public issue tracker exists
for this NGC-distributed package.

Import PyNvVideoCodec through this module instead of importing it
directly, so the workaround travels with any environment that installs
this package rather than requiring a manual site-packages patch per venv.
"""

import ast

if not hasattr(ast, "Str"):
    setattr(ast, "Str", ast.Constant)  # noqa: B010

import PyNvVideoCodec  # noqa: E402

__all__ = ["PyNvVideoCodec"]
