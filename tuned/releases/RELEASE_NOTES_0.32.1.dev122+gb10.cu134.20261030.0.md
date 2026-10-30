# vLLM 0.32.1.dev122 — gb10 cu134 (0.32.1.dev122+gb10.cu134.20261030.0)

Single-arch vllm wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (TORCH_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261030.0 (UTC date of the tuned commit)
- Built from commit 078a5df1ec181daa130c10582a9fb242e8ac516d; tested at 078a5df1ec181daa130c10582a9fb242e8ac516d
- Upstream base: upstream/main merge-base fe341a53dda00472c281a2d1f4bd34a74e7eae6e
- Prebuilt dependencies: requirements/gb10.txt (torch, flashinfer) and https://github.com/zbrad/flash-attention-vllm/releases/download/v2.7.2.post1%2Bgb10.cu134.20261028.0/vllm_flash_attn-2.7.2.post1%2Bgb10.cu134.20261028.0-cp314-cp314-linux_aarch64.whl
- Wheel sha256: 178a73155f5712338f033f0852d4e48ebe38baa666fc6b112963ff4c57a585f0

## Changes since 0.32.1.dev117+gb10.cu134.20261028.0

- No upstream merge: the upstream base is unchanged (fe341a53dd). The version moves from dev117 to dev122 because it counts
  commits, and the five since 20261028.0 are this fork's own (pins, notes, tuned-common syncs).
- Prebuilt dependencies move to torch 2.15.0+gb10.cu134.20261030.0 and flashinfer 0.7.0+gb10.cu134.20261030.0 (python and
  jit-cache), in `requirements/gb10.txt`. Same torch and flashinfer source as before; the torch wheel now carries JSON build
  info (`.tuned_build_info`), and `tuned/build.sh` checks it with `gpu_tuned_verify_build_info_json`. vllm_flash_attn stays
  at 20261028.0.
- The vendored tuned-common is synced to fc2a630 (venv tool-resolution check, toolkit PATH gate, locked build requirements,
  `gpu_tuned_assert_versioned_runpath`).

## Tests

Scope (from the attached test log): gates; import check; tuned/gb10_acceptance_test.sh (stand-ins, all variants); python -m pytest tuned/tests
Skips by design: upstream's pytest suite (tests/) is not run (G3, recorded decision, docs/ARCHITECTURE.md); no multi-GPU tests
Result: === Full test suite summary: acceptance + tuned tests passed ===

## Install

```
pip install "vllm @ https://github.com/zbrad/vllm/releases/download/v0.32.1.dev122%2Bgb10.cu134.20261030.0/vllm-0.32.1.dev122+gb10.cu134.20261030.0-cp314-cp314-linux_aarch64.whl"
```
