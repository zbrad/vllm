#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Build/install vllm from source restricted to RTX 5080/5090 (Blackwell
# consumer, sm_120) only. Skips every other CUDA arch vllm normally builds
# for, so this is much faster than a full multi-arch build but the resulting
# install only works on Blackwell consumer GPUs.
#
# Unlike tools/build_gb10.sh, this does NOT need prebuilt torch/flash-attn
# wheels or a dedicated requirements file: x86_64 + Blackwell consumer
# (sm_120) is already part of vLLM's standard supported arch matrix (see
# CUDA_SUPPORTED_ARCHS in CMakeLists.txt) with full upstream PyPI wheel
# support, so the normal requirements/cuda.txt path applies unchanged.
#
# "a" suffix (12.0a) targets the Blackwell family-specific SASS variant --
# same rationale as GB10's 12.1a in build_gb10.sh -- required for RTX
# 50-specific tensor core instructions not present in the compatible/generic
# sm_120 target.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0a}"

echo "Installing build-time dependencies (cmake, ninja, setuptools-rust, ...)"
pip install -r requirements/build/cuda.txt

echo "Building vllm for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
pip install -e .
