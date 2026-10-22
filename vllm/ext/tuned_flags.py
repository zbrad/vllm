# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flags a tuned wheel carries (zbrad/vllm tuned-builds; no upstream counterpart).

In a source checkout nothing is asserted. `tuned/package.sh` rewrites the two values
below inside the wheel, and only after it has checked the compiled libraries. The
code then reads the wheel, not the environment of whatever launched it, so a tuned
install behaves the same however it is started.

Keep the two assignment lines exactly as written: package.sh edits them with sed, and
`tuned/tests/test_tuned_flags.py` fails if the two disagree.
"""


class TunedFlags:
    """Build-time assertions about this install."""

    # True only if package.sh verified, with cuobjdump, that
    # vllm_flash_attn/_vllm_fa2_C holds native cubins for the wheel's arch and no
    # PTX, so it can never need the driver to JIT-compile PTX at run time.
    FLASH_ATTN_FA2_NATIVE_ONLY: bool = False

    # Default for FLASHINFER_DISABLE_JIT: an op outside the shipped AOT cache is a
    # hard error, not a silent JIT build. Applied with setdefault in
    # vllm/env_override.py, so an explicit (even empty) value still wins.
    FLASHINFER_DISABLE_JIT: bool = False
