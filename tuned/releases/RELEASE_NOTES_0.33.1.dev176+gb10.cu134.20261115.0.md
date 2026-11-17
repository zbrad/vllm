# vLLM 0.33.1.dev176 — gb10 cu134 (0.33.1.dev176+gb10.cu134.20261115.0)

Single-arch vllm wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (TORCH_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261115.0 (UTC date of the tuned commit)
- Built from commit d713282154e0944f1060f815f5afe8198379844b; tested at d713282154e0944f1060f815f5afe8198379844b
- Upstream base: upstream/main merge-base 4135590ffe0119dab806d78a7eee6dff3be2618e (unchanged from 0.33.1.dev170)
- Prebuilt dependencies: requirements/gb10.txt (torch, flashinfer) and https://github.com/zbrad/flash-attention-vllm/releases/download/v2.7.2.post1%2Bgb10.cu134.20261113.0/vllm_flash_attn-2.7.2.post1%2Bgb10.cu134.20261113.0-cp314-cp314-linux_aarch64.whl
- Wheel sha256: 4091c651bbdc26d858be978a8f99e8a3a4e39c58239cfe350d4e365cd30f0dd4

## Changes since 0.33.1.dev170+gb10.cu134.20261113.0

- **All three prebuilt dependencies move to torch `2.15.0+gb10.cu134.20261112.0`** (was `20261030.1`), so torch,
  flashinfer and vllm_flash_attn are again built against the same torch:
  - torch `2.15.0+gb10.cu134.20261112.0` (`3c12148784`)
  - flashinfer `0.7.4+gb10.cu134.20261113.1` (was `20261112.0`; same 0.7.4 source, rebuilt against the new torch;
    python, shim and provider wheels; `3c12148784`)
  - vllm_flash_attn `2.7.2.post1+gb10.cu134.20261113.0` (was `20261102.0`; upstream base b273a02 unchanged, rebuilt
    against the new torch; `d713282154`)
- `vllm.__version__` now carries the tuned label instead of setuptools_scm's `+g<sha>` (`901df064f9`); the smoke test
  checks it reads back as `0.33.1.dev176+gb10.cu134.20261115.0`.
- The other changes are tuned tooling synced from tuned-common (`50e0b9118e`, `5d93752588`: cu133 pins with no system
  libnccl2, a cu135 toolkit definition, rtx40-win device data, and `so_has_device_code` no longer hitting SIGPIPE under
  pipefail). This wheel is still built with CUDA 13.4.

## Tests

Scope (from the attached test log): gates; import check; tuned/gb10_acceptance_test.sh (stand-ins, all variants); python -m pytest tuned/tests
Skips by design: upstream's pytest suite (tests/) is not run (G3, recorded decision, docs/ARCHITECTURE.md); no multi-GPU tests
Result: === Full test suite summary: acceptance + tuned tests passed ===

- Acceptance: PASS for `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` and `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`;
  SKIP for five stand-ins (`deepgemm-sm121`, `bad-checkpoint`, `deps-gap-torchvision`, `mla-dims-too-small`,
  `trtllm-moe-sm100-only`), which are not coverage of this release. DeepSeek-V4 and Muse Spark were not run.
- tuned/tests: 45 passed.
- Smoke test of the packaged wheel passed (2026-11-15T05:17:34Z): `test_rms_norm` 216, `test_act_and_mul` 126, `test_rotary_embedding_opcheck` 192 passed.

## Install

Install flashinfer's three wheels (see the flashinfer release) with this one; `requirements/gb10.txt` has the URLs.

```
pip install "vllm @ https://github.com/zbrad/vllm/releases/download/v0.33.1.dev176%2Bgb10.cu134.20261115.0/vllm-0.33.1.dev176+gb10.cu134.20261115.0-cp314-cp314-linux_aarch64.whl"
```
