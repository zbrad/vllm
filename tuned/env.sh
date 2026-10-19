#!/bin/bash
# tuned/env.sh <variant> — device, toolkit and build-environment setup for a tuned vllm wheel
# (gb10/rtx40/rtx50). Source this file with the variant as $1; do not execute it directly.
#
#   CUDA_TAG=cu134 bash tuned/build.sh gb10     (or CUDA_VER=13.4)
#
# Device facts come from tuned/common-data/devices/<variant>.conf and the CUDA toolkit from
# tuned/common-data/toolkits/<id>.env (both vendored from zbrad/tuned-common by sync-common.sh),
# never from /usr/local/cuda or from what happens to be installed. There is no default: CUDA_TAG or
# CUDA_VER must name one. Repo-specific settings live in tuned/repo.env, loaded after the shared
# definitions; it may not change any shared key, and it may restrict the toolkits this repo builds
# against (gb10: cu134 only).
#
# Exported: GPU_TUNED_VARIANT/PLATFORM/CUDA_ARCH/HW_LABEL (shared), GPU_TUNED_TORCH_ARCH,
# GPU_TUNED_NEEDS_PREBUILT_DEPS and GPU_TUNED_FLASH_ATTN_WHEEL_URL (repo.env), CUDA_HOME/CUDA_TAG/
# CUDA_VER (the toolkit), GPU_TUNED_BUILD_ROOT (tuned/build/<cuda-tag>, the one tree for this
# toolkit: venv/ wheel/ cmake/ dist/ releases/ ccache/), GPU_TUNED_VENV_DIR, GPU_TUNED_WHEEL_DIR,
# GPU_TUNED_DIST_DIR, GPU_TUNED_RELEASES_DIR, GPU_TUNED_CMAKE_DIR, CCACHE_DIR, TORCH_CUDA_ARCH_LIST,
# MAX_JOBS, NVCC_THREADS and, for prebuilt-dependency variants, VLLM_GB10_BUILD and
# VLLM_FLASH_ATTN_PREBUILT_PKG.
#
# The CMake build directory is not chosen here: setup.py's build_temp is pointed at
# GPU_TUNED_CMAKE_DIR by wheel.sh (build --build-base). The repo's everyday build/ is never used.

GPU_TUNED_ARG_VARIANT="${1:-}"
if [[ -z "${GPU_TUNED_ARG_VARIANT}" ]]; then
    echo "ERROR: env.sh requires a variant argument (gb10/rtx40/rtx50)" >&2
    return 1 2>/dev/null || exit 1
fi

GPU_TUNED_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_TUNED_DATA_DIR="${GPU_TUNED_SELF_DIR}/common-data"
if [[ ! -f "${GPU_TUNED_DATA_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf" ]]; then
    echo "ERROR: no device '${GPU_TUNED_ARG_VARIANT}' in ${GPU_TUNED_DATA_DIR}/devices (have: $(cd "${GPU_TUNED_DATA_DIR}/devices" && ls | sed 's/\.conf$//' | tr '\n' ' '))" >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck source=common-data/devices/gb10.conf
source "${GPU_TUNED_DATA_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf" || return 1 2>/dev/null || exit 1
export GPU_TUNED_VARIANT GPU_TUNED_PLATFORM GPU_TUNED_CUDA_ARCH GPU_TUNED_HW_LABEL

# shellcheck source=common.sh
# Vendored from https://github.com/zbrad/tuned-common (see sync-common.sh).
source "${GPU_TUNED_SELF_DIR}/common.sh" || return 1 2>/dev/null || exit 1

# Fail loudly if this script runs on the wrong host, rather than producing a wrong-architecture wheel.
gpu_tuned_assert_platform "${GPU_TUNED_PLATFORM}" "${GPU_TUNED_VARIANT}" || return 1 2>/dev/null || exit 1

# --- CUDA toolkit: CUDA_TAG (cu134) or CUDA_VER (13.4) name one from common-data/toolkits. ---
_vllm_tk="${CUDA_TAG:-}"
_vllm_req_ver="${CUDA_VER:-}"
if [ -z "${_vllm_tk}" ] && [ -n "${_vllm_req_ver}" ]; then
    _vllm_tk="cu${_vllm_req_ver//./}"                           # 13.4 -> cu134
fi
gpu_tuned_load_toolkit "${GPU_TUNED_DATA_DIR}" "${GPU_TUNED_VARIANT}" "${_vllm_tk}" || return 1 2>/dev/null || exit 1
if [ -n "${_vllm_req_ver}" ] && [ "${_vllm_req_ver}" != "${TOOLKIT_CUDA_VER}" ]; then
    echo "[tuned/env] ERROR: CUDA_VER=${_vllm_req_ver} conflicts with toolkit ${TOOLKIT_ID} (CUDA ${TOOLKIT_CUDA_VER})." >&2
    return 1 2>/dev/null || exit 1
fi
unset _vllm_tk _vllm_req_ver

# --- Repo-specific settings (loaded after the shared device and toolkit values; may not change them). ---
gpu_tuned_load_repo_env "${GPU_TUNED_SELF_DIR}/repo.env" || return 1 2>/dev/null || exit 1
# A sourced file's own `return 1` does not stop the loader, so check that repo.env really set its keys.
if [[ -z "${GPU_TUNED_NEEDS_PREBUILT_DEPS+x}" || -z "${GPU_TUNED_TORCH_ARCH:-}" ]]; then
    echo "ERROR: tuned/repo.env did not set its settings for ${GPU_TUNED_VARIANT}." >&2
    return 1 2>/dev/null || exit 1
fi
export GPU_TUNED_TORCH_ARCH GPU_TUNED_NEEDS_PREBUILT_DEPS GPU_TUNED_FLASH_ATTN_WHEEL_URL
export GPU_TUNED_REPO_TOOLKITS GPU_TUNED_REPO_TOOLKIT_REASON

# Which toolkits this variant may be built against for vllm (repo.env). Checked before the toolkit is
# verified, so a refused toolkit is reported as the restriction, not as a missing install.
if [[ -n "${GPU_TUNED_REPO_TOOLKITS:-}" && " ${GPU_TUNED_REPO_TOOLKITS} " != *" ${CUDA_TAG} "* ]]; then
    echo "[tuned/env] ERROR: ${GPU_TUNED_VARIANT} vllm builds support ${GPU_TUNED_REPO_TOOLKITS} only; ${CUDA_TAG} was asked for." >&2
    echo "  ${GPU_TUNED_REPO_TOOLKIT_REASON}" >&2
    return 1 2>/dev/null || exit 1
fi
gpu_tuned_verify_toolkit || return 1 2>/dev/null || exit 1

export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${GPU_TUNED_TOOLKIT_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_PATH="${CUDA_HOME}" CUDACXX="${CUDA_HOME}/bin/nvcc"
# CUDA_VERSION_COMPACT (e.g. "134" for CUDA 13.4) is the toolkit id's digits.
CUDA_VERSION_COMPACT="${CUDA_TAG#cu}"
export CUDA_VERSION_COMPACT

# --- The one build tree for this toolkit (docs/ARCHITECTURE.md C1/C2): tuned/build/<cuda-tag>/. ---
GPU_TUNED_REPO_ROOT="$(cd "${GPU_TUNED_SELF_DIR}/.." && pwd)"
# A tree last used by another variant is refused, never mixed (gpu_tuned_claim_toolkit_tree).
gpu_tuned_claim_toolkit_tree "${GPU_TUNED_REPO_ROOT}" "${CUDA_TAG}" "${GPU_TUNED_VARIANT}" || return 1 2>/dev/null || exit 1
GPU_TUNED_BUILD_ROOT="${GPU_TUNED_REPO_ROOT}/tuned/build/${CUDA_TAG}"
GPU_TUNED_VENV_DIR="${GPU_TUNED_BUILD_ROOT}/venv"
GPU_TUNED_WHEEL_DIR="${GPU_TUNED_BUILD_ROOT}/wheel"
GPU_TUNED_CMAKE_DIR="$(gpu_tuned_out_dir build "${GPU_TUNED_REPO_ROOT}" "${CUDA_TAG}")"
GPU_TUNED_DIST_DIR="$(gpu_tuned_out_dir dist "${GPU_TUNED_REPO_ROOT}" "${CUDA_TAG}")"
GPU_TUNED_RELEASES_DIR="$(gpu_tuned_out_dir releases "${GPU_TUNED_REPO_ROOT}" "${CUDA_TAG}")"
export GPU_TUNED_BUILD_ROOT GPU_TUNED_VENV_DIR GPU_TUNED_WHEEL_DIR GPU_TUNED_CMAKE_DIR GPU_TUNED_DIST_DIR GPU_TUNED_RELEASES_DIR

# ccache is per toolkit (C4). CCACHE_DIR makes ccache ignore ~/.config/ccache, whose default size is too small.
export CCACHE_DIR="${GPU_TUNED_BUILD_ROOT}/ccache"
export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-60G}"
# setup.py's FetchContent downloads default to the repo's .deps/, shared by every toolkit and the dev build.
export FETCHCONTENT_BASE_DIR="${GPU_TUNED_BUILD_ROOT}/deps"

# --- Build parallelism: gpu_tuned_jobs unless MAX_JOBS is set (C5). NVCC_THREADS=1 keeps MAX_JOBS
#     as the real number of compile jobs (setup.py divides MAX_JOBS by NVCC_THREADS). ---
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${GPU_TUNED_TORCH_ARCH}}"
export MAX_JOBS="${MAX_JOBS:-$(gpu_tuned_jobs)}"
export NVCC_THREADS="${NVCC_THREADS:-1}"

# Prebuilt-dependency variants (gb10): setup.py reads requirements/gb10.txt and tuned/gb10_version.py
# when VLLM_GB10_BUILD is set; vllm_flash_attn is a prebuilt wheel, not compiled here.
if [[ "${GPU_TUNED_NEEDS_PREBUILT_DEPS}" == "true" ]]; then
    export VLLM_GB10_BUILD=1 VLLM_FLASH_ATTN_PREBUILT_PKG=1
fi

echo "[tuned/env] GPU_TUNED_VARIANT=${GPU_TUNED_VARIANT} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} CUDA_HOME=${CUDA_HOME} MAX_JOBS=${MAX_JOBS} tree=${GPU_TUNED_BUILD_ROOT}"
