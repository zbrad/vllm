#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail fast, loudly, and early if this torch build's CUDA version will
redirect flashinfer's fp4-quantization dispatch to an arch this variant's
AOT jit-cache wasn't built for.

flashinfer/quantization/fp4_quantization.py's get_fp4_quantization_module()
unconditionally redirects backend "120"/"121" to "120f" whenever the
CALLING process's installed torch was built against CUDA >= 12.9
(torch.version.cuda, checked at runtime) -- independent of what
FLASHINFER_CUDA_ARCH_LIST the AOT jit-cache was actually built with. Both
gb10 (tuned/devices/gb10.conf in flashinfer) and rtx50
(tuned/devices/rtx50.conf) build their jit-cache with an explicit "a"
suffix only ("12.1a"/"12.0a"), never "120f" -- so if torch.version.cuda on
this host is >= 12.9, any real code path that calls
get_fp4_quantization_module() (confirmed real call site:
flashinfer/activation.py) would raise MissingJITCacheError under strict
FLASHINFER_DISABLE_JIT=1 -- but only once that specific path is actually
exercised, potentially deep inside a real generation call.

This script calls that exact function directly, at startup, so a coverage
gap surfaces immediately with an unambiguous message instead of a random
crash mid-inference. Run before `exec vllm` in tuned/run_gb10.sh /
tuned/run_rtx50.sh (both already set FLASHINFER_DISABLE_JIT=1, which this
script relies on to get the real MissingJITCacheError rather than a silent
JIT compile).

Usage:
    python3 tuned/verify_fp4_arch_match.py <backend>
        <backend> is the SM backend this variant's AOT cache targets before
        any "f" redirect: "121" for gb10, "120" for rtx50.

Not yet empirically exercised end-to-end (no torch/flashinfer install was
available to test against when this was written) -- if it turns out
get_fp4_quantization_module() is never actually reached by the target
model's real forward pass, this script will report OK even though the
underlying question (does anything else in flashinfer hit this dispatch)
remains open; treat an OK result here as "this specific known risk is
covered", not "the whole AOT cache is guaranteed complete".
"""
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <backend: 120|121>", file=sys.stderr)
        return 2
    backend = sys.argv[1]

    import torch

    print(f"torch.version.cuda: {torch.version.cuda}")

    from flashinfer.quantization.fp4_quantization import get_fp4_quantization_module

    print(f"Requesting fp4 quantization module for backend={backend} "
          f"(flashinfer redirects to 120f internally if torch.version.cuda >= 12.9) ...")
    try:
        get_fp4_quantization_module(backend=backend)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure here is the fault we want surfaced
        print(
            f"FAULT: fp4 quantization module for backend={backend} is not "
            f"available in this build's AOT jit-cache (torch CUDA "
            f"{torch.version.cuda}).",
            file=sys.stderr,
        )
        print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "  This AOT cache was built with an explicit 'a'-suffixed "
            "FLASHINFER_CUDA_ARCH_LIST (not '120f') -- see the flashinfer "
            "repo's tuned/devices/<variant>.conf. Either rebuild flashinfer's "
            "jit-cache with '120f' included, or use a torch build with "
            "CUDA < 12.9.",
            file=sys.stderr,
        )
        return 1

    print("OK: fp4 quantization module available for this torch/AOT-cache combination.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
