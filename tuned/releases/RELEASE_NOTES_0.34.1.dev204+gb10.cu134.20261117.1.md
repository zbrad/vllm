# vLLM 0.34.1.dev204 — gb10 cu134 (0.34.1.dev204+gb10.cu134.20261117.1)

Single-arch vllm wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (TORCH_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Channel: dev (a build of upstream main, not of an upstream release tag; published as a pre-release)
- Release label: 20261117.1 (UTC date of the tuned commit)
- Built from commit 86a6780896c66798254f2dec4948e60c5f6d3fb2; tested at 86a6780896c66798254f2dec4948e60c5f6d3fb2
- Upstream base: upstream/main merge-base 34b780fe66bdf3df173358774eb4f29f684b8368
- Prebuilt dependencies: requirements/gb10.txt (torch, flashinfer) and https://github.com/zbrad/flash-attention-vllm/releases/download/v2.7.2.post1%2Bgb10.cu134.20261113.0/vllm_flash_attn-2.7.2.post1%2Bgb10.cu134.20261113.0-cp314-cp314-linux_aarch64.whl
- Wheel sha256: 8547f8ec048dd7ab489e48c7d5f4ef0e54244f6d535786ff15d816c18fda225f

## What's new since 0.34.1.dev200

- **DeepSeek-V4's code path runs on GB10** (tested with a tiny stand-in, not the real checkpoint). The FA4 CuteDSL kernels shipped in this wheel failed at import
  (`No module named 'flash_attn_2_cuda'`), because the prebuilt flash-attn path packaged them with
  unrewritten `flash_attn.cute` imports. They are now copied with their imports rewritten, as upstream's
  source build does. DeepSeek-V4's fused indexer is one user of them; with the DeepGEMM pin's SM12 scale-factor
  support (the gap in deepseek-ai/DeepGEMM#372), the DeepSeek-V4 stand-in (`yujiepan/deepseek-v4-tiny-random`)
  now loads and generates and is part of the acceptance test instead of a documented skip.
- **Direct I/O loader fixes from review** (zbrad/vllm#7): zero-size tensors stay on the GPU on the
  device-staged path instead of coming back on the CPU.

## Tests

Scope (from the attached test log): gates; import check; tuned/gb10_acceptance_test.sh (stand-ins, all variants); python -m pytest tuned/tests
Skips by design: upstream's pytest suite (tests/) is not run (G3, recorded decision, docs/ARCHITECTURE.md); no multi-GPU tests
Result: === Full test suite summary: acceptance + tuned tests passed ===

## Install

```
pip install "vllm @ https://github.com/zbrad/vllm/releases/download/v0.34.1.dev204%2Bgb10.cu134.20261117.1/vllm-0.34.1.dev204+gb10.cu134.20261117.1-cp314-cp314-linux_aarch64.whl"
```
