#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# tuned/build.sh <variant> — build/install vllm from source restricted to a
# single GPU variant (gb10/rtx40/rtx50) only. Skips every other CUDA arch
# vllm normally builds for, so this is much faster than a full multi-arch
# build but the resulting install only works on the targeted GPU.
#
# Consolidates what used to be three separate scripts (tools/build_gb10.sh,
# tools/build_rtx40.sh, tools/build_rtx50.sh) into one, parameterized by
# tuned/devices/<variant>.conf. gb10 remains structurally different from
# rtx40/rtx50 (no official aarch64/sm_121 upstream wheel exists for torch or
# flash-attention, so it consumes prebuilt personal-fork wheels instead of
# building them -- see tuned/devices/gb10.conf's GPU_TUNED_NEEDS_PREBUILT_DEPS
# note) -- handled below as an explicit branch, not forced into a shared
# path that doesn't apply to both.
#
# Note: requirements/gb10.txt is NOT relocated under tuned/ like the rest of
# this variant's tooling -- it has a `-r common.txt` recursive include that
# resolves relative to requirements/ (setup.py's get_requirements() always
# reads _read_requirements() relative to that one directory), so moving it
# would either break that include or require invasive changes to setup.py's
# requirements parser. Left in place as a deliberate, narrow exception to
# the "everything non-upstream moves under tuned/" reorg goal.
set -euo pipefail

GPU_TUNED_ARG_VARIANT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_TUNED_SELF_DIR="${REPO_ROOT}/tuned"
# shellcheck source=common.sh
source "${GPU_TUNED_SELF_DIR}/common.sh"
# shellcheck source=devices/rtx50.conf
source "${GPU_TUNED_SELF_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${GPU_TUNED_TORCH_ARCH}}"

if [[ "${GPU_TUNED_NEEDS_PREBUILT_DEPS}" == "true" ]]; then
    # setup.py derives the tuning.<N> counter from main..HEAD; refuse a stale main.
    gpu_tuned_check_main_current
    # GB10: torch and flash-attention are consumed as prebuilt wheels
    # (zbrad/pytorch and zbrad/flash-attention GB10 GitHub Releases) instead
    # of being compiled here:
    #   - torch is pinned via requirements/gb10.txt as a normal `@ URL` dependency.
    #   - vllm_flash_attn is installed explicitly below (not via requirements/gb10.txt)
    #     because it must be importable *before* vLLM's own CMake configure step
    #     runs, so cmake/external_projects/vllm_flash_attn.cmake's
    #     VLLM_FLASH_ATTN_PREBUILT_PKG path (set below) can locate it via
    #     `import vllm_flash_attn` and copy its files into vllm/vllm_flash_attn/
    #     instead of compiling flash-attention from source.

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

    echo "Auditing already-installed tuned packages before installing over them"
    # Non-fatal (WARNING only) -- this install is about to run regardless;
    # the point is surfacing a downgrade/replacement risk (e.g. a plain
    # PyPI torch that landed here via some unrelated pip install) before
    # it happens instead of discovering it later via an ABI mismatch or a
    # version-check error deep into a model load.
    gpu_tuned_audit_pinned pip torch=gb10 vllm_flash_attn=gb10 flashinfer-python=gb10
    gpu_tuned_audit_stray pip flashinfer-cubin

    echo "Installing GB10 requirements (torch @ URL, flashinfer, etc.)"
    pip install -r requirements/gb10.txt

    echo "Installing prebuilt vllm_flash_attn wheel (native sm_121, no PTX, no FA3)"
    # --no-deps: this wheel's own metadata pins torch==2.4.0 (whatever stock
    # torch was present when it was built, before our custom GB10 torch build
    # replaced it) -- that pin has no aarch64/cp314 wheel and would fight pip's
    # resolver over the real (correct) torch already installed via
    # requirements/gb10.txt above.
    pip install --no-deps "${GPU_TUNED_FLASH_ATTN_WHEEL_URL}"

    echo "Validating installed build-info stamps carry provenance for this variant"
    # Confirms what actually landed in this venv, independent of pip's own
    # (self-reported) version metadata -- traces back to a real git commit
    # and build time. Only checks torch/vllm_flash_attn: flashinfer has no
    # embed_build_info call yet (separate, tracked follow-up). Fatal on
    # purpose -- this fleet's stamping pipeline was silently discarding
    # every stamp until this session (see gpu_tuned_verify_build_info's
    # header comment in tuned-common.sh); a missing/wrong stamp here means
    # either an unfixed producer wheel or a genuinely wrong install.
    TORCH_CUDA_SO="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cuda.so"))')"
    gpu_tuned_verify_build_info "${TORCH_CUDA_SO}" torch "${GPU_TUNED_VARIANT}" pytorch_build_info
    FA2_SO="$(python3 -c 'import os, vllm_flash_attn; print(os.path.join(os.path.dirname(vllm_flash_attn.__file__), "_vllm_fa2_C.abi3.so"))')"
    gpu_tuned_verify_build_info "${FA2_SO}" vllm_flash_attn "${GPU_TUNED_VARIANT}" flash_attn_build_info

    echo "Building vllm for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} (MAX_JOBS=${MAX_JOBS}, NVCC_THREADS=${NVCC_THREADS})"
    pip install --no-build-isolation -e .
else
    echo "Installing build-time dependencies (cmake, ninja, setuptools-rust, ...)"
    pip install -r requirements/build/cuda.txt

    echo "Building vllm for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
    pip install -e .
fi

echo ""
echo "Smoke test: import vllm, confirm its compiled extensions actually load..."
# A wheel finishing install here only proves setup.py/CMake ran to
# completion -- it says nothing about whether the resulting extensions
# actually work at runtime. Bit us for real: a prebuilt vllm_flash_attn
# wheel (GB10's GPU_TUNED_NEEDS_PREBUILT_DEPS path) built against an
# older torch installed silently -- it imports fine, `pip show` looks
# correct, and the ONLY symptom was a swallowed ImportError two layers
# down (vllm.vllm_flash_attn.flash_attn_interface's own try/except),
# surfacing only much later during an actual model load, as an
# "architecture failed to be inspected" error with no obvious link back
# to flash-attn at all. Catch that class of failure here instead, right
# after the build that caused it.
python3 - <<'PYEOF'
import vllm  # noqa: F401 -- import alone exercises _C, _moe_C, etc.

from vllm.vllm_flash_attn import flash_attn_interface as fi

if not (fi.FA2_AVAILABLE or fi.FA3_AVAILABLE):
    raise SystemExit(
        "FAIL: vllm imported, but neither FA2 nor FA3 flash-attn extension "
        f"loaded (FA2: {fi.FA2_UNAVAILABLE_REASON}; FA3: {fi.FA3_UNAVAILABLE_REASON}). "
        "This usually means the prebuilt/compiled flash-attn extension is "
        "ABI-incompatible with the torch build actually installed -- check "
        "whether flash-attn (or its prebuilt wheel, on GB10) was rebuilt "
        "against the SAME torch build this vllm was just built against."
    )
print(f"OK: vllm {vllm.__version__} imports cleanly, FA2={fi.FA2_AVAILABLE} FA3={fi.FA3_AVAILABLE}.")
PYEOF
