#!/bin/bash
# tuned/smoke_test.sh <variant> — prove the PACKAGED wheel works, before it is released. It installs the stamped
# wheel tuned/package.sh made (named in package-info.txt) into the toolkit's build venv with --no-deps, and checks:
#   1. the wheel's sha256 matches package-info.txt and its .sha256 sidecar;
#   2. the installed _C_stable_libtorch.abi3.so and _moe_C_stable_libtorch.abi3.so are byte-identical to the
#      packaged wheel's copies, so the bits in the package are the bits that run;
#   3. the installed distribution is exactly the packaged version, vllm loads from the venv, and each stamped
#      extension's build-info JSON reads back as this release, variant and toolkit (H7);
#   4. the libtorch and c10 libraries a process maps are the venv's, and the toolkit libraries resolve (G2);
#   5. three real tests from this repo pass against the package. They run from an empty directory with the repo
#      appended to sys.path AFTER site-packages, so `import vllm` is the package's and `tests` is the repo's.
# A smoke test, not the suite: tuned/full_test.sh ran the gate against the unstamped wheel. What is new here is that
# the bits in the package are the ones running. Writes SMOKE_RESULTS_<variant>.log beside TEST_RESULTS; release.sh
# requires a passing log newer than the packaged wheel. A pass needs the end marker, so an abort cannot pass.
set -uo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/smoke_test.sh <variant>}"
REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPODIR}" || exit 1
# shellcheck source=env.sh
source "${REPODIR}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1

INFO="${GPU_TUNED_DIST_DIR}/package-info.txt"
[[ -f "${INFO}" ]] || { echo "ERROR: ${INFO} not found. Run CUDA_TAG=${CUDA_TAG} bash tuned/package.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
info() { sed -n "s/^$1=//p" "${INFO}"; }
WHEEL="${GPU_TUNED_DIST_DIR}/$(info wheel)"; WANT_SHA="$(info sha256)"; VERSION="$(info version)"
[[ -f "${WHEEL}" ]] || { echo "ERROR: ${WHEEL} not found." >&2; exit 1; }
mkdir -p "${GPU_TUNED_RELEASES_DIR}"
LOG="${GPU_TUNED_RELEASES_DIR}/SMOKE_RESULTS_${GPU_TUNED_VARIANT}.log"
rm -f "${LOG}"

PY="${GPU_TUNED_VENV_DIR}/bin/python"
[[ -x "${PY}" ]] || { echo "ERROR: ${PY} not found. Run tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
FAILS=0
fail() { echo "  FAIL: $*"; FAILS=$((FAILS + 1)); }
ok() { echo "  ok:   $*"; }
STAMPED=(vllm/_C_stable_libtorch.abi3.so vllm/_moe_C_stable_libtorch.abi3.so)

{
    echo "vllm tuned smoke test -- variant=${GPU_TUNED_VARIANT} cuda=${CUDA_VER} (${CUDA_TAG})"
    echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "wheel:   $(basename "${WHEEL}")"
    echo "packaged commit: $(info commit)"
    echo ""

    got="$(sha256sum "${WHEEL}" | awk '{print $1}')"
    [[ "${got}" == "${WANT_SHA}" ]] && ok "wheel sha256 matches package-info.txt" || fail "wheel sha256 ${got} does not match package-info.txt (${WANT_SHA})"
    gpu_tuned_verify_sha256_sidecar "${WHEEL}" && ok "sidecar .sha256 verifies" || fail "sidecar .sha256 does not verify"

    # Install the packaged wheel over the unstamped one the test run used. --no-deps: the venv holds the dependency
    # set build.sh pinned.
    if "${PY}" -m pip install --quiet --no-deps --force-reinstall "${WHEEL}"; then
        ok "packaged wheel installed into ${GPU_TUNED_VENV_DIR} (--no-deps)"
    else
        fail "pip could not install the packaged wheel"
    fi
    SITE_DIR="$(cd "${WORK}" && "${PY}" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

    # Byte identity of the stamped extensions with the packaged wheel's copies.
    bytes_ok=1
    for so in "${STAMPED[@]}"; do
        same="$("${PY}" -I - "${WHEEL}" "${so}" "${SITE_DIR}/${so}" <<'PY'
import hashlib, sys, zipfile
wheel, member, installed = sys.argv[1:4]
with zipfile.ZipFile(wheel) as z:
    packed = hashlib.sha256(z.read(member)).hexdigest()
with open(installed, "rb") as f:
    here = hashlib.sha256(f.read()).hexdigest()
print("same" if packed == here else "differs")
PY
)"
        [[ "${same}" == same ]] || { fail "${so} installed differs from the packaged wheel's copy"; bytes_ok=0; }
    done
    [[ ${bytes_ok} -eq 1 ]] && ok "installed ${STAMPED[*]} are byte-identical to the packaged wheel's copies"

    # The distribution version and where vllm loads from (an empty directory, so the source tree is not on the path).
    meta="$(cd "${WORK}" && "${PY}" -I -c 'import importlib.metadata as md, vllm; print("dist=" + md.version("vllm")); print("file=" + vllm.__file__)' 2>&1)"
    echo "${meta}" | sed 's/^/  | /'
    [[ "$(sed -n 's/^dist=//p' <<< "${meta}")" == "${VERSION}" ]] && ok "installed vllm is exactly ${VERSION}" || fail "installed vllm is not ${VERSION}"
    [[ "$(sed -n 's/^file=//p' <<< "${meta}")" == "${SITE_DIR}/"* ]] && ok "vllm loads from the venv" || fail "vllm loads from outside ${SITE_DIR}"

    for so in "${STAMPED[@]}"; do
        if gpu_tuned_verify_build_info_json "${SITE_DIR}/${so}" "release=${VERSION}" "variant=${GPU_TUNED_VARIANT}" "cuda.tag=${CUDA_TAG}" \
            && gpu_tuned_verify_build_info_json "${SITE_DIR}/${so}" "fork.commit=$(info built_commit)" >/dev/null 2>&1; then
            ok "${so}: build-info stamp reads back as release ${VERSION}, ${GPU_TUNED_VARIANT}, ${CUDA_TAG} and the build commit"
        else
            fail "${so}: build-info stamp does not read back as this release"
        fi
    done

    # Mapped libraries: a process that imports the extension maps the venv's libtorch/libc10, from one file each.
    maps="$(cd "${WORK}" && "${PY}" -I -c '
import collections, hashlib, os, sys
import torch
import vllm._C_stable_libtorch
torch.zeros(1, device="cuda")
by = collections.defaultdict(set)
for line in open("/proc/self/maps"):
    p = line.split()[-1] if line.split() else ""
    if ".so" in p and p.startswith("/"):
        by[os.path.basename(p).split(".so")[0]].add(os.path.realpath(p))
# A name mapped from two files is a failure unless the files are byte-identical (the pip nvidia-cuda-runtime wheel and the
# toolkit ship the same libcudart; cuda-bindings loads one, torch the other). Python extension modules may share a file
# name across packages.
def digest(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()
bad = [f"{k}: {sorted(v)}" for k, v in by.items() if len(v) > 1 and ".cpython-" not in k and len({digest(x) for x in v}) > 1]
bad += [f"{k} from {sorted(v)}" for k, v in by.items() if k.startswith(("libtorch", "libc10", "_C_stable")) and not all(x.startswith(sys.prefix) for x in v)]
same = [k for k, v in by.items() if len(v) > 1 and ".cpython-" not in k and len({digest(x) for x in v}) == 1]
print("\n".join(bad) if bad else "clean" + (" (byte-identical copies of: " + ", ".join(sorted(same)) + ")" if same else ""))
' 2>&1 | tail -5)"
    [[ "${maps}" == clean* ]] && ok "every mapped libtorch/libc10/_C_stable library is the venv's, and no library name maps from two different files ${maps#clean}" \
        || fail "library identity: ${maps}"
    gpu_tuned_verify_toolkit_libs "${SITE_DIR}/vllm/_C_stable_libtorch.abi3.so" && ok "toolkit libraries resolve under ${CUDA_HOME}" \
        || fail "toolkit libraries do not resolve under ${CUDA_HOME}"

    "${PY}" -m pip install --quiet -r tuned/requirements-test.txt || fail "test tooling did not install"

    # Three real tests from this repo, against the installed package. Pytest runs in the empty directory with
    # the repo appended after site-packages; --import-mode=importlib keeps the repo root off sys.path[0].
    run_one() {
        local node="$1" out rc passed
        out="$(cd "${WORK}" && env -u PYTHONPATH "${PY}" -I -c '
import sys
sys.path.append(sys.argv[1])
import pytest
raise SystemExit(pytest.main(sys.argv[2:]))
' "${REPODIR}" --import-mode=importlib -q -p no:cacheprovider "${REPODIR}/tests/${node}" 2>&1)"; rc=$?
        printf '%s\n' "${out}" | tail -4 | sed 's/^/  | /'
        passed="$(printf '%s\n' "${out}" | grep -oE '[0-9]+ passed' | tail -1 | grep -oE '[0-9]+' || true)"
        if [[ ${rc} -eq 0 && "${passed:-0}" -gt 0 ]] && ! printf '%s\n' "${out}" | grep -qE '[0-9]+ (failed|error)'; then
            ok "tests/${node}: ${passed} passed"
        else
            fail "tests/${node} (exit ${rc}, passed ${passed:-0})"
        fi
    }
    run_one kernels/core/test_layernorm.py::test_rms_norm
    run_one kernels/core/test_activation.py::test_act_and_mul
    run_one kernels/core/test_rotary_embedding.py::test_rotary_embedding_opcheck

    echo ""
    if [[ ${FAILS} -eq 0 ]]; then
        echo "=== Smoke test summary: passed ==="
    else
        echo "=== Smoke test summary: FAILED (${FAILS}) ==="
    fi
    echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "${LOG}"
grep -q '^=== Smoke test summary: passed ===$' "${LOG}"
