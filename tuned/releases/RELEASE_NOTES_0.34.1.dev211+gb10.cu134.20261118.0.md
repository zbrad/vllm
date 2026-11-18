# vLLM 0.34.1.dev211 — gb10 cu134 (0.34.1.dev211+gb10.cu134.20261118.0)

Single-arch vllm wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (TORCH_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Channel: dev (a build of upstream main, not of an upstream release tag; published as a pre-release)
- Release label: 20261118.0 (UTC date of the tuned commit)
- Built from commit 68a8b55451473f2109f19e96bd56944249123760; tested at 68a8b55451473f2109f19e96bd56944249123760
- Upstream base: upstream/main merge-base 34b780fe66bdf3df173358774eb4f29f684b8368
- Prebuilt dependencies: requirements/gb10.txt (torch, torchvision, flashinfer) and https://github.com/zbrad/flash-attention-vllm/releases/download/v2.7.2.post1%2Bgb10.cu134.20261113.0/vllm_flash_attn-2.7.2.post1%2Bgb10.cu134.20261113.0-cp314-cp314-linux_aarch64.whl
- Wheel sha256: 379dae6e7e05f82a58d21e7c4ac3cdb4d61d2785a1f84bfa5fe1846fdae3e0b3

## What's new since 0.34.1.dev204

- **torchvision is now a dependency** (zbrad/vision 0.29.2+gb10.cu134.20261112.0, built against exactly the torch in
  requirements/gb10.txt; image codecs are not built). transformers' Qwen2-VL image processor imports torchvision at
  module level, so without it no Qwen VL model and no Qwen3.5 or later model could load. The Qwen2-VL stand-in
  (`yujiepan/qwen2-vl-tiny-random`) is now part of the acceptance test instead of a documented skip.
- **Fix: `reshape_and_cache_flash` hung on models with fewer than 32 K elements per token** (KV heads x head
  size). The launcher started fewer than 32 threads, and with a packed K/V cache the kernel's head loop
  stepped by zero warps and never finished. Only very small models (test stand-ins such as the Qwen2-VL one above,
  1 KV head x 8) were affected; real checkpoints were not. Upstream main has the same launcher line. Regression test:
  `tuned/tests/test_reshape_and_cache_flash.py`.
- **Load-time variant and CUDA toolkit check.** At `import vllm`, after torch loads and before vllm's own extensions
  do, every loaded library that carries a tuned build stamp (and the stamped extensions in `vllm/` and
  `vllm_flash_attn/`) is compared with vllm's own stamp on `variant` and `cuda.tag`. A library from another variant or
  toolkit now fails the import with an `ImportError` naming the file and both values, instead of later with a missing
  symbol. Unstamped libraries are not checked. Measured cost: about 20 ms on a fresh process.

## Tests

Scope (from the attached test log): gates; import check; tuned/gb10_acceptance_test.sh (stand-ins, all variants); python -m pytest tuned/tests
Skips by design: upstream's pytest suite (tests/) is not run (G3, recorded decision, docs/ARCHITECTURE.md); no multi-GPU tests
Result: === Full test suite summary: acceptance + tuned tests passed ===

## Install

```
pip install "vllm @ https://github.com/zbrad/vllm/releases/download/v0.34.1.dev211%2Bgb10.cu134.20261118.0/vllm-0.34.1.dev211+gb10.cu134.20261118.0-cp314-cp314-linux_aarch64.whl"
```
