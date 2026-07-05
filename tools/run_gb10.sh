#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Run vllm on the DGX Spark (NVIDIA GB10, sm_121) with the runtime env vars
# a gb10 build needs. Pass through whatever vllm subcommand/args you want,
# e.g.:
#   tools/run_gb10.sh serve Qwen/Qwen3-Coder-30B-A3B-Instruct
# See tools/build_gb10.sh for the matching build step.

set -euo pipefail

# flashinfer.jit.env's _get_cubin_dir() does a strict string-equality check
# between flashinfer-python's version (0.6.13+gb10, our custom local build)
# and flashinfer-cubin's version (stock 0.6.13 from PyPI, since
# zbrad/flashinfer's gb10-only branch has no GitHub Release yet -- see
# requirements/gb10.txt) and raises RuntimeError on any mismatch, even
# though the two are actually compatible here. Drop this once that release
# exists and both versions are made to match.
export FLASHINFER_DISABLE_VERSION_CHECK=1

exec vllm "$@"
