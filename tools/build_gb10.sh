#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Build/install vllm from source restricted to the DGX Spark (NVIDIA GB10,
# sm_121) architecture only. Skips every other CUDA arch vllm normally builds
# for, so this is much faster than a full multi-arch build but the resulting
# install only works on GB10.
#
# torch and flash-attention are consumed as prebuilt wheels (zbrad/pytorch and
# zbrad/flash-attention GB10 GitHub Releases) instead of being compiled here:
#   - torch is pinned via requirements/gb10.txt as a normal `@ URL` dependency.
#   - vllm_flash_attn is installed explicitly below (not via requirements/gb10.txt)
#     because it must be importable *before* vLLM's own CMake configure step
#     runs, so cmake/external_projects/vllm_flash_attn.cmake's
#     VLLM_FLASH_ATTN_PREBUILT_PKG path (set below) can locate it via
#     `import vllm_flash_attn` and copy its files into vllm/vllm_flash_attn/
#     instead of compiling flash-attention from source.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FLASH_ATTN_WHEEL_URL="https://github.com/zbrad/flash-attention/releases/download/v2.7.2.post1-gb10-cu133/vllm_flash_attn-2.7.2.post1%2Bcu133-cp314-cp314-linux_aarch64.whl"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
# Compilation here is compute-bound, not I/O-bound -- more parallel jobs than
# available cores/memory bandwidth can service doesn't increase throughput.
# vllm's setup.py computes num_jobs = max(1, MAX_JOBS // NVCC_THREADS) --
# NVCC_THREADS>1 divides away outer job parallelism, so a non-1 value here
# (e.g. picking a "nicer"/prime number) would silently floor concurrency
# to 1 given MAX_JOBS=3. Keep NVCC_THREADS=1 so MAX_JOBS=3 actually yields
# 3 concurrent compile jobs.
export MAX_JOBS="${MAX_JOBS:-3}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export VLLM_FLASH_ATTN_PREBUILT_PKG=1
# setup.py's get_requirements() hardcodes requirements/cuda.txt for CUDA
# builds (which install_requires bakes into the package metadata,
# independent of whatever -r file is passed to pip on the command line) --
# this tells it to read requirements/gb10.txt instead.
export VLLM_GB10_BUILD=1

echo "Installing build-time dependencies (cmake, ninja, setuptools-rust, ...)"
# --no-build-isolation below means these must be pre-installed rather than
# pip resolving them per-build. requirements/build/cuda.txt also carries its
# own torch==2.11.0 pin (for its own dependency resolution only) -- drop it
# so it doesn't fight the real GB10 torch installed just below.
grep -v '^torch' requirements/build/cuda.txt | pip install -r /dev/stdin

echo "Installing GB10 requirements (torch @ URL, flashinfer, etc.)"
pip install -r requirements/gb10.txt

echo "Installing prebuilt vllm_flash_attn wheel (native sm_121, no PTX, no FA3)"
# --no-deps: this wheel's own metadata pins torch==2.4.0 (whatever stock
# torch was present when it was built, before our custom GB10 torch build
# replaced it) -- that pin has no aarch64/cp314 wheel and would fight pip's
# resolver over the real (correct) torch already installed via
# requirements/gb10.txt above.
pip install --no-deps "${FLASH_ATTN_WHEEL_URL}"

echo "Building vllm for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} (MAX_JOBS=${MAX_JOBS}, NVCC_THREADS=${NVCC_THREADS})"
pip install --no-build-isolation -e .
