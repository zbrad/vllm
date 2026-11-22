# vLLM 0.34.1.dev221 — gb10 cu134 (0.34.1.dev221+gb10.cu134.20261122.0)

Single-arch vllm wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (TORCH_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Channel: dev (a build of upstream main, not of an upstream release tag; published as a pre-release)
- Release label: 20261122.0 (UTC date of the tuned commit)
- Built from commit 2429c3d3e3f5dae5b6b927ea09c5e85eb5c90a0c; tested at 2429c3d3e3f5dae5b6b927ea09c5e85eb5c90a0c
- Upstream base: upstream/main merge-base 34b780fe66bdf3df173358774eb4f29f684b8368 (unchanged since 0.34.1.dev214)
- Prebuilt dependencies: requirements/gb10.txt (torch, torchvision, flashinfer-python, flashinfer-jit-cache, flashinfer-jit-cache-sm121a) and https://github.com/zbrad/flash-attention-vllm/releases/download/v2.7.2.post1%2Bgb10.cu134.20261113.0/vllm_flash_attn-2.7.2.post1%2Bgb10.cu134.20261113.0-cp314-cp314-linux_aarch64.whl
- Wheel sha256: 7eada92e9b43e491d6151a619e34e724912ed38625aed6832e7a0d3e4a419eb1

## Behaviour changes

- **torch.compile is off by default** (759d8c2fea, `TunedFlags.TORCH_COMPILE_OFF_BY_DEFAULT`). When no compilation
  mode is given, it resolves to NONE instead of VLLM_COMPILE, and startup logs "Tuned build: torch.compile is off by
  default ...". CUDA graphs stay on: vLLM falls back from FULL_AND_PIECEWISE to FULL_DECODE_ONLY (models vLLM already
  runs with breakable CUDA graphs, such as Qwen3.5/3.6, keep FULL_AND_PIECEWISE). Pass `-cc.mode=3` (or
  `--compilation-config '{"mode": 3}'`) to compile as before. Why: each compiled model builds its own torch.compile
  cache and Inductor Triton kernels at first start, which a build meant to run pre-built kernels would have to build
  and ship per model, for little gain. Measured on one GB10 with 0.34.1.dev214 and mode 0 set by hand (empty caches,
  2 runs each, random 512 in / 256 out, tok/s compiled vs mode 0):

  | model | 1 request | 16 requests | time to ready |
  |---|---:|---:|---:|
  | Qwen3-4B-Instruct-2507 | 30.6 vs 30.4 | 354 vs 350 | 43 vs 28 s |
  | Hermes-3-Llama-3.1-8B | 16.5 vs 16.5 | 205 vs 203 | 46 vs 32 s |
  | Qwen3.6-35B-A3B-NVFP4 | already uncompiled upstream; unchanged | | |
  | NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | 71.7 vs 69.6 | 304 vs 296 | 89 vs 72 s |

  `--enforce-eager` is not a substitute: without CUDA graphs, 1-request decode fell 57% on the Qwen3.6 MoE and about
  27% on Nemotron-3-Nano. Breakable CUDA graphs (`VLLM_USE_BREAKABLE_CUDAGRAPH=1`) were also tried and were no faster
  than mode 0 and slower to start.
- **FlashInfer no longer downloads at run time** (2553af3155). With the wheel's `FLASHINFER_DISABLE_JIT` flag,
  `FLASHINFER_NO_DOWNLOAD=1` is now defaulted too (setdefault; an explicit value, even empty, still wins). A kernel
  artifact that is not installed fails by name instead of being fetched from NVIDIA. With flashinfer
  0.7.4+gb10.cu134.20261122.0 (pinned here) this also covers the DSL FMHA `checksums.txt` fetch. The default follows
  the wheel's flag, not the effective value: setting `FLASHINFER_DISABLE_JIT=` (empty) to allow JIT still leaves
  downloads off unless `FLASHINFER_NO_DOWNLOAD=` is cleared as well.
- **FlashInfer works without nvcc or ninja** (35b9da995c, cherry-pick of 4c8b64d25c from the Windows branch).
  `has_flashinfer()` required nvcc and `ninja` on PATH unless `flashinfer-cubin` was installed, ignoring the shipped
  jit-cache. On a machine without `ninja`, FlashInfer was silently disabled ("FlashInfer kernels are disabled") and the
  sampler and other paths fell back to other kernels, so the same model ran different kernels on different machines.
  With `FLASHINFER_DISABLE_JIT` set the check is skipped.
- **Accurate FlashInfer startup message** (2814b01103). The warning that FlashInfer "downloads and compiles kernels at
  startup" (wrong with JIT disabled) is gone. A normal start logs one line: "FlashInfer 0.7.4 runs pre-built kernels
  only (flashinfer-jit-cache 0.7.4): FLASHINFER_DISABLE_JIT and FLASHINFER_NO_DOWNLOAD are set, ...". A mismatched
  jit-cache still warns, naming `MissingJITCacheError` as the failure.

## Other changes since 0.34.1.dev214

- flashinfer 0.7.4+gb10.cu134.20261122.0 (2429c3d3e3; from 20261113.1): its one source change is the DSL FMHA
  `FLASHINFER_NO_DOWNLOAD` check above.
- tuned-common synced to 7f8a05d (8a1a680286): SemVer native-tag functions, not used by this wheel repo.

## Tests

Scope (from the attached test log): gates; import check; tuned/gb10_acceptance_test.sh (stand-ins, all variants); python -m pytest tuned/tests
Skips by design: upstream's pytest suite (tests/) is not run (G3, recorded decision, docs/ARCHITECTURE.md); no multi-GPU tests
Result: === Full test suite summary: acceptance + tuned tests passed ===

The gate tests the unstamped wheel, in which every TunedFlags flag is False, so its acceptance run (4 PASS: the two
NemotronH stand-ins, deepseek-v4-tiny-random, qwen2-vl-tiny-random; 3 documented SKIPs) ran with torch.compile on.
The baked defaults were checked separately on this packaged wheel, installed, serving with default environment
(`--gpu-memory-utilization 0.5`): every model ran compile mode NONE and logged the "runs pre-built kernels only" line,
with no refused or attempted download, no `MissingJITCacheError` and FlashInfer enabled.

| model | settings | CUDA graphs | ready | 1 / 16 requests, tok/s |
|---|---|---|---:|---|
| NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | `--mamba-backend flashinfer` | FULL_DECODE_ONLY | 72 s | 67.8 / 292.5 |
| Qwen3.6-35B-A3B-NVFP4 | | FULL_AND_PIECEWISE | 78 s | 78.7 / 324.3 |
| NVIDIA-Nemotron-3-Nano-4B-BF16 | | FULL_DECODE_ONLY | 74 s | 32.4 / 321.9 |
| deepseek-v4-tiny-random | `--kv-cache-dtype fp8_ds_mla --max-model-len 512` | FULL_AND_PIECEWISE | 108 s | 849 / 1,000 (128 in / 64 out) |
| qwen2-vl-tiny-random | `--max-model-len 512` | FULL_DECODE_ONLY | 35 s | 1,396 / 15,141 (128 in / 64 out) |

Throughput is random 512 in / 256 out unless marked; the tiny-random models use 128 / 64 to fit their 512-token
length, so their numbers are not comparable with the others.

## Install

```
pip install "vllm @ https://github.com/zbrad/vllm/releases/download/v0.34.1.dev221%2Bgb10.cu134.20261122.0/vllm-0.34.1.dev221+gb10.cu134.20261122.0-cp314-cp314-linux_aarch64.whl"
```
