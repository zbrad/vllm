#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Run vllm on the DGX Spark (NVIDIA GB10, sm_121) with the runtime env vars
# a gb10 build needs. Pass through whatever vllm subcommand/args you want,
# e.g.:
#   tuned/run_gb10.sh serve Qwen/Qwen3-Coder-30B-A3B-Instruct
# See tuned/build.sh gb10 for the matching build step.

set -euo pipefail

# FLASHINFER_DISABLE_JIT and the FLASH_ATTN native-cubin assertion used to be exported here, so a vllm started any other
# way behaved differently. They are now carried by the wheel (vllm/ext/tuned_flags.py, written by tuned/package.sh) and
# apply however vllm is started. To allow JIT for an op outside the shipped AOT cache, set the variable to empty:
#   FLASHINFER_DISABLE_JIT= tuned/run_gb10.sh serve ...

# Fail fast, before spending minutes loading the real model, if this
# torch/flashinfer combination has a known fp4-quantization arch-coverage
# gap (torch built against CUDA>=12.9 redirects to a "120f" module this
# AOT cache doesn't have -- see the script's own docstring).
# The vllm that runs is the one the tuned build venv holds for the toolkit named by CUDA_TAG (no default:
# tuned/env.sh names it). The venv is not the user site, so a stock vllm on PATH is never picked up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh" gb10
[[ -x "${GPU_TUNED_VENV_DIR}/bin/vllm" ]] || {
    echo "ERROR: no vllm in ${GPU_TUNED_VENV_DIR}/bin. Run CUDA_TAG=${CUDA_TAG} bash tuned/build.sh gb10 and tuned/full_test.sh gb10 first." >&2
    exit 1
}
"${GPU_TUNED_VENV_DIR}/bin/python" "${SCRIPT_DIR}/verify_fp4_arch_match.py" 121

exec "${GPU_TUNED_VENV_DIR}/bin/vllm" "$@"
