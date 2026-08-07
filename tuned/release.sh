#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# tuned/release.sh <variant> — publish an already-built (tuned/wheel.sh
# <variant>) vllm wheel as a real GitHub release, matching
# zbrad/raft's/zbrad/cuvs's/zbrad/faiss's/zbrad/flashinfer's tuned/
# publish conventions (v<version>-<variant>-<cuda_tag> tag, --target
# tuned-builds). vllm previously only did an editable install -- no
# wheel, no publish step -- of everything in this repo set, it was the
# only true leaf with no artifact at all.
#
# vllm's own version is setuptools_scm-derived (git-describe style, e.g.
# 8.4.dev7+g2b9dcbd29.gb10.cu133) rather than a plain x.y.z like
# raft/cuvs/faiss/flashinfer -- there's no clean SHORT_VER reduction for
# it, so the release tag uses the dev-version's base (before the "+"
# local-version segment, which already separately encodes variant+cuda
# tag) plus an explicit -<variant>-<cuda_tag> suffix, for the same tag
# shape as every other repo in this set.
#
# Usage:
#   bash tuned/wheel.sh gb10       # first, produces dist/gb10/*.whl
#   bash tuned/release.sh gb10     # then, publish it
set -euo pipefail

GPU_TUNED_ARG_VARIANT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_TUNED_SELF_DIR="${REPO_ROOT}/tuned"
# shellcheck source=devices/rtx50.conf
source "${GPU_TUNED_SELF_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf"

DIST_DIR="${REPO_ROOT}/dist/${GPU_TUNED_VARIANT}"
WHEEL_FILE="$(find "${DIST_DIR}" -maxdepth 1 -name '*.whl' | head -1)"
if [[ -z "${WHEEL_FILE}" ]]; then
    echo "ERROR: no wheel found in ${DIST_DIR}." >&2
    echo "  Run 'bash tuned/wheel.sh ${GPU_TUNED_VARIANT}' first." >&2
    exit 1
fi
WHEEL_BASENAME="$(basename "${WHEEL_FILE}")"

# e.g. vllm-8.4.dev7+g2b9dcbd29.gb10.cu133-cp314-...whl -> full version
# 8.4.dev7+g2b9dcbd29.gb10.cu133, base 8.4.dev7 (before "+").
FULL_VERSION="$(echo "${WHEEL_BASENAME}" | sed -E 's/^vllm-([^-]+)-.*/\1/')"
BASE_VERSION="${FULL_VERSION%%+*}"

CUDA_TAG="$(echo "${FULL_VERSION}" | grep -oE 'cu[0-9]+' | head -1)"
[[ -z "${CUDA_TAG}" ]] && CUDA_TAG="unknown"

RELEASE_TAG="v${BASE_VERSION}-${GPU_TUNED_VARIANT}-${CUDA_TAG}"
RELEASE_TITLE="vLLM ${FULL_VERSION} — ${GPU_TUNED_HW_LABEL}"

echo "===================================================="
echo "vllm ${GPU_TUNED_HW_LABEL} Release"
echo "===================================================="
echo "  Wheel   : ${WHEEL_BASENAME}"
echo "  Version : ${FULL_VERSION}"
echo ""
echo "Publishing to GitHub release ${RELEASE_TAG}..."
gh release create "${RELEASE_TAG}" \
    --repo zbrad/vllm \
    --title "${RELEASE_TITLE}" \
    --target "tuned-builds" \
    --notes "vllm ${FULL_VERSION} wheel for ${GPU_TUNED_HW_LABEL}, single-arch (TORCH_CUDA_ARCH_LIST=${GPU_TUNED_TORCH_ARCH}). Depends on zbrad/pytorch's and zbrad/flash-attention's matching gb10 releases (see requirements/gb10.txt and tuned/devices/${GPU_TUNED_VARIANT}.conf)." \
    "${WHEEL_FILE}#${WHEEL_BASENAME}"

echo ""
echo "Release: https://github.com/zbrad/vllm/releases/tag/${RELEASE_TAG}"
echo "Done."
