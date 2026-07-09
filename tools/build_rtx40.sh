#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Build/install vllm from source restricted to RTX 4080/4090 (Ada Lovelace,
# sm_89) only. Skips every other CUDA arch vllm normally builds for, so this
# is much faster than a full multi-arch build but the resulting install only
# works on Ada Lovelace GPUs.
#
# Unlike tools/build_gb10.sh, this does NOT need prebuilt torch/flash-attn
# wheels or a dedicated requirements file: x86_64 + Ada Lovelace (sm_89) is
# already part of vLLM's standard supported arch matrix (see
# CUDA_SUPPORTED_ARCHS in CMakeLists.txt) with full upstream PyPI wheel
# support, so the normal requirements/cuda.txt path applies unchanged. Ada
# Lovelace has no "a" (family-specific) SASS variant -- unlike Blackwell,
# there's nothing beyond the compatible/generic sm_89 target to opt into.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

echo "Installing build-time dependencies (cmake, ninja, setuptools-rust, ...)"
pip install -r requirements/build/cuda.txt

echo "Building vllm for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
pip install -e .
