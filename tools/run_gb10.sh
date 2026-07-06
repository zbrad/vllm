#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Run vllm on the DGX Spark (NVIDIA GB10, sm_121) with the runtime env vars
# a gb10 build needs. Pass through whatever vllm subcommand/args you want,
# e.g.:
#   tools/run_gb10.sh serve Qwen/Qwen3-Coder-30B-A3B-Instruct
# See tools/build_gb10.sh for the matching build step.

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
# Override with FLASHINFER_DISABLE_JIT=0 when running other models.
export FLASHINFER_DISABLE_JIT="${FLASHINFER_DISABLE_JIT:-1}"

exec vllm "$@"
