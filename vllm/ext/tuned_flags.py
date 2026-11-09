# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flags a tuned wheel carries (zbrad/vllm tuned-builds; no upstream counterpart).

In a source checkout nothing is asserted. `tuned/package.sh` rewrites the values
below inside the wheel, and only after it has checked the compiled libraries. The
code then reads the wheel, not the environment of whatever launched it, so a tuned
install behaves the same however it is started.

Keep the assignment lines exactly as written: package.sh edits them with sed, and
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

    # Default NCCL_MAX_NCHANNELS=2 when the whole job is one process (world size
    # across DP is 1). vLLM still creates an NCCL communicator per parallel group
    # with one rank each, and with NCCL's default channel count each held 0.38-0.51
    # GiB of buffers (about 3 GiB in all on a 24 GiB RTX 4090); 2 channels cut that
    # to 0.24 GiB. Applied with setdefault in vllm/v1/worker/gpu_worker.py, so an
    # explicit value still wins, and never for more than one rank.
    NCCL_SINGLE_RANK_FEW_CHANNELS: bool = False
