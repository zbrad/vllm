#!/bin/bash
# tuned/full_test.sh <variant> — test the wheel tuned/build.sh made, for ONE variant and ONE CUDA toolkit,
# and write a results log in that toolkit's tree (tuned/build/<cuda-tag>/releases/TEST_RESULTS_<variant>.log).
#
#   CUDA_TAG=cu134 bash tuned/full_test.sh gb10
#
# The unstamped wheel (tuned/build/<cuda-tag>/wheel/vllm-*.whl) is installed --no-deps into the toolkit's
# build venv, and every test runs from an empty directory, so vllm and its compiled extensions are the
# wheel's, not the source tree's. The venv holds the dependency set build.sh pinned from requirements/gb10.txt
# (or cuda.txt); a clean-venv resolution would re-download the GB10 torch and flashinfer wheels for each test.
#
# RECORDED DECISION, docs/ARCHITECTURE.md G3 (owner, 2026-10-19): a narrower gate than upstream's full
# pytest suite (tests/), which is deliberately not run. The gate is:
#   1. the gates above: wheel label and build commit, H6 source check, the compiled extensions load from the
#      wheel, the toolkit libraries resolve (G2), the single-arch check (verify_arch) passes;
#   2. the import check build.sh used to run at its end: vllm imports and FA2 or FA3 flash-attn loads;
#   3. tuned/gb10_acceptance_test.sh: small stand-ins for NemotronH and DeepSeek-V4, load and generate
#      (needs a Hugging Face download; a stand-in is not the target model);
#   4. python -m pytest tuned/tests: the tuned tooling's own tests (no GPU).
# Skips by design, named in the log header: everything in tests/ (upstream's suite) and multi-GPU
# behaviour (a tuned build is single-GPU: CUDA_VISIBLE_DEVICES is pinned to one device).
#
# The log ends with "=== Full test suite summary: acceptance + tuned tests passed ===" only when every step
# passed; release.sh greps for that exact line. The other outcome is "=== Full test suite summary: FAILED
# (see above) ===". tuned/package.sh and tuned/release.sh refuse a log without the pass line.
set -uo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/full_test.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1

WHEEL="$(ls "${GPU_TUNED_WHEEL_DIR}"/vllm-*.whl 2>/dev/null | head -1)"
[[ -n "${WHEEL}" ]] || { echo "ERROR: no wheel in ${GPU_TUNED_WHEEL_DIR}. Run CUDA_TAG=${CUDA_TAG} bash tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
BUILD_COMMIT="$(cat "${GPU_TUNED_BUILD_ROOT}/build-commit" 2>/dev/null || true)"
LABEL="$(cat "${GPU_TUNED_BUILD_ROOT}/release-label" 2>/dev/null || true)"
[[ -n "${BUILD_COMMIT}" && -n "${LABEL}" ]] || { echo "ERROR: ${GPU_TUNED_BUILD_ROOT}/build-commit or release-label missing: the last build did not finish. Re-run tuned/build.sh ${GPU_TUNED_VARIANT}." >&2; exit 1; }
[[ "$(basename "${WHEEL}")" == vllm-*"+${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL}-"* ]] \
    || { echo "ERROR: $(basename "${WHEEL}") does not carry label ${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL} from release-label." >&2; exit 1; }
[[ "${BUILD_COMMIT}" != *-dirty ]] || { echo "ERROR: the build was made from a dirty tree (${BUILD_COMMIT}); commit and rebuild." >&2; exit 1; }
git cat-file -e "${BUILD_COMMIT}^{commit}" 2>/dev/null || { echo "ERROR: build commit ${BUILD_COMMIT} is not in this checkout." >&2; exit 1; }
# H6: the wheel must be from the source at HEAD; only tuned/ may differ.
STALE_SRC="$(git diff --name-only "${BUILD_COMMIT}" HEAD -- . ':!tuned' | head -5)"
if [[ -n "${STALE_SRC}" ]]; then
    echo "ERROR: the wheel is from ${BUILD_COMMIT}, but source outside tuned/ changed since:" >&2
    echo "${STALE_SRC}" | sed 's/^/    /' >&2
    echo "  Rebuild (tuned/build.sh ${GPU_TUNED_VARIANT}) before testing." >&2
    exit 1
fi

PY="${GPU_TUNED_VENV_DIR}/bin/python"
[[ -x "${PY}" ]] || { echo "ERROR: ${PY} not found. Run CUDA_TAG=${CUDA_TAG} bash tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
mkdir -p "${GPU_TUNED_RELEASES_DIR}"
RESULTS_FILE="${GPU_TUNED_RELEASES_DIR}/TEST_RESULTS_${GPU_TUNED_VARIANT}.log"
rm -f "${RESULTS_FILE}"

# Tuned builds are single-GPU: the tests see exactly one device (GPU_TUNED_TEST_GPUS, default 0).
export CUDA_VISIBLE_DEVICES="${GPU_TUNED_TEST_GPUS:-0}"
TESTDIR="$(mktemp -d)"
trap 'rm -rf "${TESTDIR}"' EXIT

echo "installing $(basename "${WHEEL}") into the build venv (--no-deps)"
"${PY}" -m pip install --quiet --no-deps --force-reinstall "${WHEEL}" -r tuned/requirements-test.txt \
    || { echo "ERROR: pip could not install the wheel and the test tooling." >&2; exit 1; }

# Where the wheel's compiled extensions load from, and the device-code checks, from an empty directory.
SITE_DIR="$(cd "${TESTDIR}" && "${PY}" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
{
    echo "vllm tuned test run -- variant=${GPU_TUNED_VARIANT} cuda=${CUDA_VER} (${CUDA_TAG})"
    echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit:  $(git rev-parse HEAD)"
    echo "build:   ${BUILD_COMMIT} (label ${LABEL})"
    echo "wheel:   $(basename "${WHEEL}") sha256 $(sha256sum "${WHEEL}" | awk '{print $1}')"
    echo "torch:   $(cd "${TESTDIR}" && "${PY}" -I -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no device")' 2>&1 | tail -1)"
    echo "tools:   $("${PY}" -m pip freeze 2>/dev/null | grep -iE '^(pytest|setuptools|wheel|ninja|cmake)[=@ ]' | tr '\n' ' ')"
    echo "scope:   gates; import check; tuned/gb10_acceptance_test.sh (stand-ins, all variants); python -m pytest tuned/tests"
    echo "devices: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (single-GPU build: multi-GPU tests are not part of its scope)"
    echo "skips by design: upstream's pytest suite (tests/) is not run (G3, recorded decision, docs/ARCHITECTURE.md); no multi-GPU tests"
    echo "gates: wheel installed --no-deps in the build venv; vllm and _C_stable_libtorch load from it; toolkit libs resolve under ${CUDA_HOME}; single-arch ${GPU_TUNED_TORCH_ARCH}; build commit ${BUILD_COMMIT} == HEAD outside tuned/"
    echo ""
} > "${RESULTS_FILE}"

# Gates and import check. A function, so its status comes back through the tee pipeline's PIPESTATUS.
gate_checks() {
    local rc=0 loaded_from so
    # vllm and the extension must be the installed wheel's, from an empty directory (no source tree on the path).
    (cd "${TESTDIR}" && "${PY}" -I -c '
import vllm
from vllm.vllm_flash_attn import flash_attn_interface as fi
print("vllm: " + vllm.__file__)
print("fa2: " + str(fi.FA2_AVAILABLE) + "  fa3: " + str(fi.FA3_AVAILABLE))
if not (fi.FA2_AVAILABLE or fi.FA3_AVAILABLE):
    raise SystemExit("FAIL: neither FA2 nor FA3 flash-attn extension loaded: FA2: %s; FA3: %s" % (fi.FA2_UNAVAILABLE_REASON, fi.FA3_UNAVAILABLE_REASON))
import vllm._C_stable_libtorch  # noqa: F401
import vllm._moe_C_stable_libtorch  # noqa: F401
print("loaded: _C_stable_libtorch and _moe_C_stable_libtorch")
') || rc=1
    loaded_from="$(cd "${TESTDIR}" && "${PY}" -I -c 'import vllm._C_stable_libtorch as m; print(m.__file__)' 2>/dev/null | tail -1)"
    if [[ "${loaded_from}" == "${SITE_DIR}/"* ]]; then echo "  ok:   _C_stable_libtorch is the wheel's: ${loaded_from}"
    else echo "  FAIL: _C_stable_libtorch loads from '${loaded_from}', not from ${SITE_DIR}"; rc=1; fi
    if gpu_tuned_verify_toolkit_libs "${SITE_DIR}/vllm/_C_stable_libtorch.abi3.so"; then echo "  ok:   toolkit libraries resolve under ${CUDA_HOME}"
    else echo "  FAIL: toolkit libraries do not resolve under ${CUDA_HOME}"; rc=1; fi
    for so in "${SITE_DIR}/vllm/_C_stable_libtorch.abi3.so" "${SITE_DIR}/vllm/_moe_C_stable_libtorch.abi3.so"; do
        gpu_tuned_verify_arch "${so}" "${GPU_TUNED_TORCH_ARCH}" || rc=1
    done
    return "${rc}"
}
# logrun <command...>: run with output to the terminal and the log; returns the command's own status.
logrun() { "$@" 2>&1 | tee -a "${RESULTS_FILE}"; return "${PIPESTATUS[0]}"; }

STATUS=0
echo "=== gates and import check ===" | tee -a "${RESULTS_FILE}"
logrun gate_checks || STATUS=1
echo "" | tee -a "${RESULTS_FILE}"
echo "=== tuned/gb10_acceptance_test.sh (stand-ins; needs a Hugging Face download) ===" | tee -a "${RESULTS_FILE}"
TUNED_PYTHON="${PY}" bash tuned/gb10_acceptance_test.sh 2>&1 | tee -a "${RESULTS_FILE}"
[[ ${PIPESTATUS[0]} -eq 0 ]] || { STATUS=1; echo "gb10_acceptance_test.sh FAILED" | tee -a "${RESULTS_FILE}"; }

echo "" | tee -a "${RESULTS_FILE}"
echo "=== python -m pytest tuned/tests ===" | tee -a "${RESULTS_FILE}"
(cd "${REPO_ROOT}" && env PYTHONPATH="${REPO_ROOT}/tuned" "${PY}" -m pytest -q -p no:cacheprovider tuned/tests) 2>&1 | tee -a "${RESULTS_FILE}"
[[ ${PIPESTATUS[0]} -eq 0 ]] || { STATUS=1; echo "tuned/tests FAILED" | tee -a "${RESULTS_FILE}"; }

echo "" >> "${RESULTS_FILE}"
echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${RESULTS_FILE}"
if [[ "${STATUS}" -eq 0 ]]; then
    echo "=== Full test suite summary: acceptance + tuned tests passed ===" | tee -a "${RESULTS_FILE}"
else
    echo "=== Full test suite summary: FAILED (see above) ===" | tee -a "${RESULTS_FILE}"
fi
echo "Results written to ${RESULTS_FILE}"
exit "${STATUS}"
