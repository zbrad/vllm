#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# tuned/build.sh <variant> — build the vllm wheel for one GPU variant (gb10/rtx40/rtx50) and one CUDA
# toolkit. It creates the toolkit's build venv (tuned/build/<cuda-tag>/venv), installs the build and
# runtime dependencies into it, then makes the unstamped wheel with tuned/wheel.sh. The compile is the
# long step. Nothing is installed into the user site or ~/.local: the venv is this toolkit's own.
#
#   CUDA_TAG=cu134 bash tuned/build.sh gb10
#
# Stage order (docs/ARCHITECTURE.md B3): build.sh -> full_test.sh -> package.sh -> smoke_test.sh ->
# (review and commit the release notes) -> release.sh -> publish.sh.
#
# requirements/gb10.txt stays under requirements/: setup.py reads it, and its `-r common.txt` resolves
# relative to that directory.
set -euo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/build.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"

command -v python3 &>/dev/null || { echo "ERROR: python3 not found on PATH." >&2; exit 1; }

VENV_DIR="${GPU_TUNED_VENV_DIR}"
# Create or reuse (verified), activate, upgrade pip, current setuptools: one shared setup (C9, tuned-common).
gpu_tuned_setup_build_venv "${VENV_DIR}" "${REPO_ROOT}" || exit 1

echo "=========================================="
echo "Building vllm for ${GPU_TUNED_HW_LABEL}"
echo "=========================================="
echo "Toolkit:   ${CUDA_TAG} (${CUDA_HOME})"
echo "Venv:      ${VENV_DIR}"
echo "Python:    $(python3 --version)"
echo ""

echo "Installing build-time requirements..."
pip install --upgrade pip
pip install build wheel

if [[ "${GPU_TUNED_NEEDS_PREBUILT_DEPS}" == "true" ]]; then
    # torch comes from requirements/gb10.txt (a pinned zbrad/pytorch wheel), so the cuda.txt pin is dropped.
    grep -v '^torch' requirements/build/cuda.txt | pip install -r /dev/stdin
    echo "Installing GB10 requirements (torch @ URL, flashinfer, etc.)"
    pip install -r requirements/gb10.txt
    echo "Installing prebuilt vllm_flash_attn wheel (native sm_121, no PTX, no FA3)"
    # --no-deps: its own metadata pins the stock torch it was built against, not the GB10 torch above.
    pip install --no-deps "${GPU_TUNED_FLASH_ATTN_WHEEL_URL}"

    echo "Auditing already-installed tuned packages"
    # Non-fatal: surfaces a stock torch or flashinfer that replaced a tuned one before the build relies on it.
    gpu_tuned_audit_pinned pip torch=gb10 vllm_flash_attn=gb10 flashinfer-python=gb10
    gpu_tuned_audit_stray pip flashinfer-cubin

    echo "Validating installed build-info stamps carry provenance for this variant"
    # Fatal on purpose: a missing or wrong stamp means an unstamped producer wheel or a wrong install.
    TORCH_CUDA_SO="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cuda-tuned.so"))')"
    gpu_tuned_verify_build_info "${TORCH_CUDA_SO}" torch "${GPU_TUNED_VARIANT}" pytorch_build_info
    FA2_SO="$(python3 -c 'import os, vllm_flash_attn; print(os.path.join(os.path.dirname(vllm_flash_attn.__file__), "_vllm_fa2_C.abi3.so"))')"
    gpu_tuned_verify_build_info "${FA2_SO}" vllm_flash_attn "${GPU_TUNED_VARIANT}" flash_attn_build_info
else
    echo "Installing build-time requirements (cmake, ninja, setuptools-rust, ...)"
    pip install -r requirements/build/cuda.txt
    echo "Installing runtime requirements (setup.py's install_requires)"
    pip install -r requirements/cuda.txt
fi
# The current setuptools, installed last (C8, tuned-common): requirements/build/cuda.txt above caps it at <81 and
# would otherwise leave the build venv on 80.x. wheel.sh builds with --no-isolation, so pyproject's cap is not enforced.
gpu_tuned_install_setuptools || exit 1

echo ""
bash "${REPO_ROOT}/tuned/wheel.sh" "${GPU_TUNED_VARIANT}"

echo ""
echo "=========================================="
echo "Build complete (${GPU_TUNED_HW_LABEL})."
echo "=========================================="
echo "Label:        $(cat "${GPU_TUNED_BUILD_ROOT}/release-label")"
echo "Build commit: $(cat "${GPU_TUNED_BUILD_ROOT}/build-commit")"
echo "Next: CUDA_TAG=${CUDA_TAG} bash tuned/full_test.sh ${GPU_TUNED_VARIANT}"
