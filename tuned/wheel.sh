#!/bin/bash
# tuned/wheel.sh <variant> — make the unstamped vllm wheel for one GPU variant and one CUDA toolkit,
# into tuned/build/<cuda-tag>/wheel/, and record the release label and the build commit. This is the
# wheel step of tuned/build.sh, which installs the toolkit's build venv first; it refuses to run without
# that venv. It never publishes, tests or packages (package.sh stamps a copy, H3).
#
#   CUDA_TAG=cu134 bash tuned/wheel.sh gb10     (normally run by tuned/build.sh)
#
# The release label (E3/E4) is allocated here, once: HEAD's UTC commit date and the lowest unused sequence
# among origin's tags. setup.py puts it in the wheel's local version through VLLM_GB10_LOCAL_VERSION.
# release-label and build-commit are written last, so their presence means the build finished.
# The CMake build directory is GPU_TUNED_CMAKE_DIR: setuptools' build --build-base moves build_temp and
# build_lib under it, so the repo's everyday build/ is not written (C6, C1).
set -euo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/wheel.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"

[[ -d "${GPU_TUNED_VENV_DIR}" ]] || {
    echo "ERROR: ${GPU_TUNED_VENV_DIR} not found. Run CUDA_TAG=${CUDA_TAG} bash tuned/build.sh ${GPU_TUNED_VARIANT} (it installs the build venv, then calls this script)." >&2
    exit 1
}
gpu_tuned_verify_venv "${GPU_TUNED_VENV_DIR}" "${REPO_ROOT}"
# shellcheck source=/dev/null
source "${GPU_TUNED_VENV_DIR}/bin/activate"

LOCAL_VERSION="$(gpu_tuned_build_local_version "${GPU_TUNED_VARIANT}" "${REPO_ROOT}")" || exit 1
[[ "${LOCAL_VERSION}" == *".cu${CUDA_VERSION_COMPACT}."* ]] || {
    echo "ERROR: label ${LOCAL_VERSION} disagrees with ${CUDA_TAG} (from ${CUDA_HOME}); nvcc and the toolkit definition must report the same release." >&2
    exit 1
}
LABEL="${LOCAL_VERSION#*.cu${CUDA_VERSION_COMPACT}.}"
export VLLM_GB10_LOCAL_VERSION="${LOCAL_VERSION}"
echo "Release label: ${LABEL} (local version ${LOCAL_VERSION})"

# Remove the previous build's markers first: they are rewritten only if this build succeeds.
rm -f "${GPU_TUNED_BUILD_ROOT}/build-commit" "${GPU_TUNED_BUILD_ROOT}/release-label"
mkdir -p "${GPU_TUNED_WHEEL_DIR}" "${GPU_TUNED_CMAKE_DIR}"
rm -f "${GPU_TUNED_WHEEL_DIR}"/vllm-*.whl

echo "=========================================="
echo "Building vllm wheel for ${GPU_TUNED_HW_LABEL}"
echo "=========================================="
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
echo "CUDA_HOME:            ${CUDA_HOME}"
echo "MAX_JOBS:             ${MAX_JOBS} (NVCC_THREADS=${NVCC_THREADS})"
echo "CMake build tree:     ${GPU_TUNED_CMAKE_DIR}"
echo "Python:               $(python3 --version)"
echo "Git commit:           $(git rev-parse --short HEAD)"

# The unstamped wheel: built through setuptools' PEP 517 backend with --build-base pointed at the toolkit's
# cmake/ tree. --no-isolation: the build venv already holds the build requirements (build.sh installs them).
# --skip-dependency-check: pyproject.toml pins torch==2.13.0 for upstream's own build, but this venv deliberately
# holds the tuned torch wheel from requirements/gb10.txt (the old pip install --no-build-isolation never checked
# that pin either; build.sh dropped upstream's torch line from the build requirements for the same reason).
python3 -m build --wheel --no-isolation --skip-dependency-check --outdir "${GPU_TUNED_WHEEL_DIR}" \
    --config-setting="--build-option=build --build-base=${GPU_TUNED_CMAKE_DIR}"

WHEEL="$(ls "${GPU_TUNED_WHEEL_DIR}"/vllm-*.whl 2>/dev/null | head -1)"
[[ -n "${WHEEL}" ]] || { echo "ERROR: no wheel found in ${GPU_TUNED_WHEEL_DIR}" >&2; exit 1; }
VERSION="$(gpu_tuned_wheel_version "${WHEEL}" vllm)" || exit 1
[[ "${VERSION}" == *"+${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL}" ]] || {
    echo "ERROR: $(basename "${WHEEL}") is version ${VERSION}, not carrying +${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL} from the release label." >&2
    exit 1
}
gpu_tuned_check_wheel_label "${WHEEL}"
echo "Built wheel: $(basename "${WHEEL}") ($(du -sh "${WHEEL}" | awk '{print $1}'))"

# build-commit: the source the wheel was made from. A dirty tree is recorded as such and package.sh refuses it.
BUILD_COMMIT="$(git rev-parse HEAD)"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || BUILD_COMMIT="${BUILD_COMMIT}-dirty"
printf '%s\n' "${LABEL}" > "${GPU_TUNED_BUILD_ROOT}/release-label"
printf '%s\n' "${BUILD_COMMIT}" > "${GPU_TUNED_BUILD_ROOT}/build-commit"

echo ""
echo "Wheel:        ${WHEEL}"
echo "Label:        ${LABEL}"
echo "Build commit: ${BUILD_COMMIT}"
