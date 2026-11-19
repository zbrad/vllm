# vLLM 0.34.1.dev214 — gb10 cu134 (0.34.1.dev214+gb10.cu134.20261119.0)

Single-arch vllm wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (TORCH_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Channel: dev (a build of upstream main, not of an upstream release tag; published as a pre-release)
- Release label: 20261119.0 (UTC date of the tuned commit)
- Built from commit 5a6c658ed0ab7934d4c44421c2075125f4305118; tested at 5a6c658ed0ab7934d4c44421c2075125f4305118
- Upstream base: upstream/main merge-base 34b780fe66bdf3df173358774eb4f29f684b8368
- Prebuilt dependencies: requirements/gb10.txt (torch, torchvision, flashinfer-python, flashinfer-jit-cache, flashinfer-jit-cache-sm121a) and https://github.com/zbrad/flash-attention-vllm/releases/download/v2.7.2.post1%2Bgb10.cu134.20261113.0/vllm_flash_attn-2.7.2.post1%2Bgb10.cu134.20261113.0-cp314-cp314-linux_aarch64.whl
- Wheel sha256: 4abbf8d42c4e6ad82d7d1f662da1771833e7ddc8d3f470e54f0ebcd4de73ba7d

## What's new since 0.34.1.dev211

- **Faster low-concurrency decode for bf16 models: the skinny GEMM now covers every bf16 linear layer, not only the
  shapes upstream tuned** (5a6c658ed0). Upstream runs its CuTe DSL skinny GEMM on GB10 only for the weight shapes in
  its plan table (GLM-5.3-Flash). Every other bf16 linear ran cuBLAS, whose 1-token kernel reads weights well below
  GB10's memory bandwidth. Shapes without a measured plan now get a default config at 1, 2 and 4 tokens. Measured
  plans still take precedence, and other token counts (prefill, larger batches) still use cuBLAS.
  - The config is chosen inside the GEMM op at run time, so it is not baked into the torch.compile cache.
  - `VLLM_SKINNY_GEMM_DEFAULT_RULE=0` turns it off. It is read at run time and is not part of the compile-cache key.
  - Measured on one GB10, Qwen3-4B-Instruct-2507, random 512 in / 256 out, compile cache on, with this change
    applied to the dev211 build (not this wheel):

    | requests at once | rule off | rule on |
    |---|---:|---:|
    | 1 | 23.44 tok/s | 30.92 tok/s (+32%) |
    | 2 | 57.06 | 60.37 (+5.8%) |
    | 4 | 109.42 | 117.81 (+7.7%) |

    The same rule on Hermes-3-Llama-3.1-8B (Llama-3.1-8B shapes): 15.62 -> 16.92 tok/s at 1 request (+8%), where
    cuBLAS was already closer to the bandwidth limit. 16 requests are unchanged on both models. gsm8k (200
    questions, flexible match): Qwen3-4B 0.915 -> 0.920 and Hermes-3 0.755 -> 0.770, within noise.
  - This wheel, installed, same Qwen3-4B setup at 1 request: 30.34 tok/s (median time per token 32.78 ms).
  - Results vary by model: the gain depends on how far cuBLAS falls below the bandwidth limit on that model's shapes.
- Release notes list every prebuilt dependency from requirements/gb10.txt (82eb5d22b1).

## Tests

Scope (from the attached test log): gates; import check; tuned/gb10_acceptance_test.sh (stand-ins, all variants); python -m pytest tuned/tests
Skips by design: upstream's pytest suite (tests/) is not run (G3, recorded decision, docs/ARCHITECTURE.md); no multi-GPU tests
Result: === Full test suite summary: acceptance + tuned tests passed ===

## Install

```
pip install "vllm @ https://github.com/zbrad/vllm/releases/download/v0.34.1.dev214%2Bgb10.cu134.20261119.0/vllm-0.34.1.dev214+gb10.cu134.20261119.0-cp314-cp314-linux_aarch64.whl"
```
