#!/usr/bin/env bash
# tuned/publish.sh <variant> — publish the GitHub release that tuned/release.sh marked ready.
# It publishes exactly what release.sh checked: it refuses unless the marker exists and HEAD, the notes and
# the wheel are unchanged since release.sh ran (gpu_tuned_publish_ready). It builds nothing.
#
#   bash tuned/publish.sh gb10
set -uo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/publish.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

GPU_TUNED_SELF_DIR="${REPO_ROOT}/tuned"
# shellcheck source=devices/rtx50.conf
source "${GPU_TUNED_SELF_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf" || exit 1
# shellcheck source=common.sh
source "${GPU_TUNED_SELF_DIR}/common.sh" || exit 1

MARKER="${REPO_ROOT}/dist/${GPU_TUNED_VARIANT}/RELEASE_READY_${GPU_TUNED_VARIANT}"
TAG="$(sed -n 's/^tag=//p' "${MARKER}" 2>/dev/null)"
gpu_tuned_publish_ready "${MARKER}" || exit 1

echo ""
echo "Release: https://github.com/zbrad/vllm/releases/tag/${TAG}"
echo "Done."
