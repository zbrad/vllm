#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Run vllm on the DGX Spark (NVIDIA GB10, sm_121) with the runtime env vars
# a gb10 build needs. Pass through whatever vllm subcommand/args you want,
# e.g.:
#   tuned/run_gb10.sh serve Qwen/Qwen3-Coder-30B-A3B-Instruct
# See tuned/build.sh gb10 for the matching build step.

set -euo pipefail

# flashinfer-jit-cache (the AOT-compiled GB10 kernel cache, see
# requirements/gb10.txt) is now published at the same +gb10 version as
# flashinfer-python, so flashinfer.jit.env's version-equality check passes
# on its own -- FLASHINFER_DISABLE_VERSION_CHECK is no longer needed.
#
# That jit-cache release is scoped to two target models (NemotronH,
# DeepSeek-V4-Flash; see requirements/gb10.txt) -- FLASHINFER_DISABLE_JIT=1
# turns any op outside that filtered set into a hard MissingJITCacheError
# instead of a silent (working, just slower on first call) JIT compile.
#
# flashinfer checks `os.environ.get("FLASHINFER_DISABLE_JIT")` truthily
# (flashinfer/jit/core.py) -- FLASHINFER_DISABLE_JIT=0 is still a non-empty
# string and stays "disabled". `${VAR-default}` (no colon) substitutes only
# when VAR is completely unset, unlike `${VAR:-default}` which also
# substitutes for an empty string -- so to actually allow JIT for other
# models, set it to *empty* rather than unsetting it or using 0, e.g.:
#   FLASHINFER_DISABLE_JIT= tuned/run_gb10.sh serve ...
export FLASHINFER_DISABLE_JIT="${FLASHINFER_DISABLE_JIT-1}"

# Asserts that THIS deployment's _vllm_fa2_C (zbrad/flash-attention-vllm's
# tuned/build.sh gb10) was compiled with a real native sm_121a cubin
# (CUDA_ARCHS=12.1a), not upstream's PTX-only consumer-Blackwell build --
# verified via cuobjdump (no embedded PTX) at the time this was added, see
# vllm/platforms/cuda.py's _flash_attn_no_ptx_asserted(). Lets that check
# skip a false-positive PTX/driver-version hard failure it would otherwise
# raise for every GB10 box, gb10-tuned-build or not. Deliberately a
# separate, explicit assertion rather than inferred from the gb10 build
# marker alone -- a gb10 build isn't guaranteed to always be compiled this
# way (see that function's own docstring). Same unset-vs-empty truthiness
# as FLASHINFER_DISABLE_JIT above -- unset it (VLLM_FLASH_ATTN_NO_PTX=
# tuned/run_gb10.sh serve ...) if this build's cubin coverage is ever in
# doubt again.
export VLLM_FLASH_ATTN_NO_PTX="${VLLM_FLASH_ATTN_NO_PTX-1}"

# Fail fast, before spending minutes loading the real model, if this
# torch/flashinfer combination has a known fp4-quantization arch-coverage
# gap (torch built against CUDA>=12.9 redirects to a "120f" module this
# AOT cache doesn't have -- see the script's own docstring).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/verify_fp4_arch_match.py" 121

exec vllm "$@"
