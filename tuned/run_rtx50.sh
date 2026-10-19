#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Run vllm on RTX 5080/5090 (Blackwell consumer, sm_120) with the runtime
# env vars this variant's AOT flashinfer build needs. Pass through whatever
# vllm subcommand/args you want, e.g.:
#   tuned/run_rtx50.sh serve Qwen/Qwen2.5-32B
# See tuned/build.sh rtx50 for the matching build step, and
# flashinfer's tuned/devices/rtx50.conf for the target models this AOT
# cache is scoped to (NemotronH nano-30B-A3B-NVFP4 and Qwen 2.5 32B).

set -euo pipefail

# flashinfer checks `os.environ.get("FLASHINFER_DISABLE_JIT")` truthily
# (flashinfer/jit/core.py) -- FLASHINFER_DISABLE_JIT=0 is still a non-empty
# string and stays "disabled". `${VAR-default}` (no colon) substitutes only
# when VAR is completely unset, unlike `${VAR:-default}` which also
# substitutes for an empty string -- so to actually allow JIT for other
# models, set it to *empty* rather than unsetting it or using 0, e.g.:
#   FLASHINFER_DISABLE_JIT= tuned/run_rtx50.sh serve ...
#
# Deliberately enforced here despite a known open risk (see
# tuned/verify_fp4_arch_match.py and flashinfer's tuned/devices/rtx50.conf):
# a fault that identifies exactly what's missing is preferred over silently
# falling back to JIT.
export FLASHINFER_DISABLE_JIT="${FLASHINFER_DISABLE_JIT-1}"

# Fail fast, before spending minutes loading the real model, if this
# torch/flashinfer combination has a known fp4-quantization arch-coverage
# gap (torch built against CUDA>=12.9 redirects to a "120f" module this
# AOT cache doesn't have -- see the script's own docstring).
# The vllm that runs is the one the tuned build venv holds for the toolkit named by CUDA_TAG (no default:
# tuned/env.sh names it). The venv is not the user site, so a stock vllm on PATH is never picked up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh" rtx50
[[ -x "${GPU_TUNED_VENV_DIR}/bin/vllm" ]] || {
    echo "ERROR: no vllm in ${GPU_TUNED_VENV_DIR}/bin. Run CUDA_TAG=${CUDA_TAG} bash tuned/build.sh rtx50 and tuned/full_test.sh rtx50 first." >&2
    exit 1
}
"${GPU_TUNED_VENV_DIR}/bin/python" "${SCRIPT_DIR}/verify_fp4_arch_match.py" 120

exec "${GPU_TUNED_VENV_DIR}/bin/vllm" "$@"
