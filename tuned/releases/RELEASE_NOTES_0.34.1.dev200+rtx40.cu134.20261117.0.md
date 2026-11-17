# vLLM 0.34.1.dev200 — rtx40 cu134 (0.34.1.dev200+rtx40.cu134.20261117.0)

Single-arch vllm wheel for RTX 4080/4090 (Ada Lovelace, SM 89) (TORCH_CUDA_ARCH_LIST=8.9), CUDA 13.4.

- Channel: dev (a build of upstream main, not of an upstream release tag; published as a pre-release)
- Release label: 20261117.0 (UTC date of the tuned commit)
- Built from commit e45bca34b17484aa9a9989c078a3e6dd568aa497; tested at e45bca34b17484aa9a9989c078a3e6dd568aa497
- Upstream base: upstream/main merge-base 34b780fe66bdf3df173358774eb4f29f684b8368
- Prebuilt dependencies: requirements/rtx40.txt (torch, flashinfer) and https://github.com/zbrad/flash-attention-vllm/releases/download/v2.7.2.post1%2Brtx40.cu134.20261109.0/vllm_flash_attn-2.7.2.post1%2Brtx40.cu134.20261109.0-cp314-cp314-linux_x86_64.whl
- Wheel sha256: 6690d365852694c3cfea15bdcfac2938d02539e001057d4548afe793998acab3

## What's new since 0.33.1.dev166

- **Upstream v0.34.0** merged (1,071 upstream commits). For SM 89 in particular: Marlin's 8-bit activation path
  (INT8/FP8) is removed in favour of Humming (#65542), so FP8-activation models run on different kernels than in
  dev166; this build keeps Marlin's 8.9 entries so the weight-only Marlin kernels are still built natively for SM 89.
  Also removed upstream: QuTLASS and fp_quant (#65230) and Model Runner V1 (`VLLM_USE_V2_MODEL_RUNNER` other than `1`
  is an error).
- **Weight loading is unchanged on RTX 40.** The direct I/O loader added in this series (zbrad/vllm#7) is only the
  default on integrated GPUs; on a discrete GPU under WSL, vLLM reports no pinned memory, so even an explicit
  `--safetensors-load-strategy direct` uses its plain host-buffer path. Loads stay memory-mapped by default (the
  acceptance models loaded in 2.3 s and 9.3 s here).

## Tests

Scope (from the attached test log): gates; import check; tuned/gb10_acceptance_test.sh (stand-ins, all variants); python -m pytest tuned/tests
Skips by design: upstream's pytest suite (tests/) is not run (G3, recorded decision, docs/ARCHITECTURE.md); no multi-GPU tests
Result: === Full test suite summary: acceptance + tuned tests passed ===

## Install

```
pip install "vllm @ https://github.com/zbrad/vllm/releases/download/v0.34.1.dev200%2Brtx40.cu134.20261117.0/vllm-0.34.1.dev200+rtx40.cu134.20261117.0-cp314-cp314-linux_x86_64.whl"
```
