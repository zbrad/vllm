#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# tuned/wheel.sh <variant> — build a real, installable vllm wheel for a
# single GPU variant (gb10/rtx40/rtx50), instead of tuned/build.sh's
# editable (`pip install -e .`) install. Run tuned/build.sh <variant> at
# least once first (or just let this script's own dependency-install steps
# run) -- this reuses the same env vars (TORCH_CUDA_ARCH_LIST, MAX_JOBS,
# NVCC_THREADS, VLLM_FLASH_ATTN_PREBUILT_PKG, VLLM_GB10_BUILD) so a variant
# built via tuned/build.sh doesn't get silently rebuilt against different
# settings here.
#
# vllm's setup.py has no persisted incremental build_temp/ directory across
# invocations (confirmed empirically: no build/temp.* survives a prior
# tuned/build.sh run) -- so this is a real CUDA recompile, same cost as
# tuned/build.sh itself, not a quick repackage of already-built objects.
#
# Usage:
#   bash tuned/wheel.sh gb10
set -euo pipefail

GPU_TUNED_ARG_VARIANT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_TUNED_SELF_DIR="${REPO_ROOT}/tuned"
# shellcheck source=devices/rtx50.conf
source "${GPU_TUNED_SELF_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${GPU_TUNED_TORCH_ARCH}}"

DIST_DIR="${REPO_ROOT}/dist/${GPU_TUNED_VARIANT}"
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"

if [[ "${GPU_TUNED_NEEDS_PREBUILT_DEPS}" == "true" ]]; then
    # Same rationale as tuned/build.sh's own branch -- see its comments.
    export MAX_JOBS="${MAX_JOBS:-3}"
    export NVCC_THREADS="${NVCC_THREADS:-1}"
    export VLLM_FLASH_ATTN_PREBUILT_PKG=1
    export VLLM_GB10_BUILD=1

    echo "Installing build-time dependencies (cmake, ninja, setuptools-rust, ...)"
    grep -v '^torch' requirements/build/cuda.txt | pip install -r /dev/stdin
    echo "Installing GB10 requirements (torch @ URL, flashinfer, etc.)"
    pip install -r requirements/gb10.txt
    echo "Installing prebuilt vllm_flash_attn wheel (native sm_121, no PTX, no FA3)"
    pip install --no-deps "${GPU_TUNED_FLASH_ATTN_WHEEL_URL}"
else
    echo "Installing build-time dependencies (cmake, ninja, setuptools-rust, ...)"
    pip install -r requirements/build/cuda.txt
fi

echo "Building vllm wheel for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
pip wheel --no-build-isolation --no-deps -w "${DIST_DIR}" .

WHEEL_FILE="$(find "${DIST_DIR}" -maxdepth 1 -name '*.whl' | head -1)"
[[ -z "${WHEEL_FILE}" ]] && { echo "ERROR: no wheel produced in ${DIST_DIR}." >&2; exit 1; }
echo ""
echo "Built: $(basename "${WHEEL_FILE}") ($(du -sh "${WHEEL_FILE}" | awk '{print $1}'))"
