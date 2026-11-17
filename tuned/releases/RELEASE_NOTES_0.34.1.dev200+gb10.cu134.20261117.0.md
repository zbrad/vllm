# vLLM 0.34.1.dev200 — gb10 cu134 (0.34.1.dev200+gb10.cu134.20261117.0)

Single-arch vllm wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (TORCH_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Channel: dev (a build of upstream main, not of an upstream release tag; published as a pre-release)
- Release label: 20261117.0 (UTC date of the tuned commit)
- Built from commit e45bca34b17484aa9a9989c078a3e6dd568aa497; tested at e45bca34b17484aa9a9989c078a3e6dd568aa497
- Upstream base: upstream/main merge-base 34b780fe66bdf3df173358774eb4f29f684b8368
- Prebuilt dependencies: requirements/gb10.txt (torch, flashinfer) and https://github.com/zbrad/flash-attention-vllm/releases/download/v2.7.2.post1%2Bgb10.cu134.20261113.0/vllm_flash_attn-2.7.2.post1%2Bgb10.cu134.20261113.0-cp314-cp314-linux_aarch64.whl
- Wheel sha256: 8bd8fcd20ae81d1e8736e082a5882fa594603ff9ab2268bd1ea2385e1f8c813d

## What's new since 0.33.1.dev176

- **Upstream v0.34.0** merged (1,071 upstream commits). Upstream changes to know about: Marlin's 8-bit
  activation path (INT8/FP8) is removed in favour of Humming (#65542), QuTLASS and fp_quant are removed (#65230),
  and Model Runner V1 is gone (`VLLM_USE_V2_MODEL_RUNNER` other than `1` is an error).
- **Weight loading on integrated GPUs is no longer slow.** Includes upstream's fix for first copies from a mapped
  checkpoint (#58868): a 7.4 GB checkpoint that took about 18 minutes to load on GB10 now loads in under a second.
- **Direct I/O weight loading, default on GB10** (zbrad/vllm#7): with checkpoints on a local ext4/xfs and less than
  half of the checkpoint already in the page cache, safetensors are read with `O_DIRECT` through pinned buffers
  straight into GPU staging buffers, with read-ahead across files, and no page cache is filled. Measured on GB10,
  Nemotron-3-Super-120B-A12B-NVFP4 (80.3 GB), cold: 6.84 s to load weights, against 39.4 s memory-mapped and 8.2 s
  with fastsafetensors. A checkpoint that is already mostly cached is memory-mapped instead (Nemotron-3-Nano-4B,
  warm: 0.33 s). Force either with `--safetensors-load-strategy direct` or `--safetensors-load-strategy lazy`.

## Tests

Scope (from the attached test log): gates; import check; tuned/gb10_acceptance_test.sh (stand-ins, all variants); python -m pytest tuned/tests
Skips by design: upstream's pytest suite (tests/) is not run (G3, recorded decision, docs/ARCHITECTURE.md); no multi-GPU tests
Result: === Full test suite summary: acceptance + tuned tests passed ===

## Install

```
pip install "vllm @ https://github.com/zbrad/vllm/releases/download/v0.34.1.dev200%2Bgb10.cu134.20261117.0/vllm-0.34.1.dev200+gb10.cu134.20261117.0-cp314-cp314-linux_aarch64.whl"
```
