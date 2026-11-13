# vLLM 0.33.1.dev170 — gb10 cu134 (0.33.1.dev170+gb10.cu134.20261113.0)

Single-arch vllm wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (TORCH_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261113.0 (UTC date of the tuned commit)
- Built from commit 0a6a3e8e79756698589e5f443e0e90d9ee132517; tested at 0a6a3e8e79756698589e5f443e0e90d9ee132517
- Upstream base: upstream/main merge-base 4135590ffe0119dab806d78a7eee6dff3be2618e (unchanged from 0.33.1.dev149)
- Prebuilt dependencies: requirements/gb10.txt (torch, flashinfer) and https://github.com/zbrad/flash-attention-vllm/releases/download/v2.7.2.post1%2Bgb10.cu134.20261102.0/vllm_flash_attn-2.7.2.post1%2Bgb10.cu134.20261102.0-cp314-cp314-linux_aarch64.whl
- Wheel sha256: 56f6f24a66ba312711b304b46c3036f2cf54d060674c92759ede89deef2e22d9

## Changes since 0.33.1.dev149+gb10.cu134.20261102.1

- **flashinfer moves from 0.7.2.post1 to 0.7.4** (`0.7.4+gb10.cu134.20261112.0`; python, shim and provider wheels).
  It is built against the same torch, `2.15.0+gb10.cu134.20261030.1`. flashinfer 0.7.4 requires
  `nvidia-cudnn-frontend>=1.32.0`, so the build venv now resolves 1.32.3 (was 1.31.0); `requirements/gb10.txt` keeps
  `cuda.txt`'s `>=1.31.0` floor so the drift test against upstream's `cuda.txt` still holds.
- Upstream's `requirements/cuda.txt` still pins flashinfer 0.7.2.post1, so this is a newer flashinfer than the vLLM
  source was tested with upstream.
- DeepSeek-V4.1 specialized decode only enables when the installed flashinfer-python version is exactly `0.7.2`
  (`vllm/models/deepseek_v41/nvidia/specialized_decode/__init__.py`). The 0.7.2.post1+gb10 version string never
  matched that either, so the stock path was already used on gb10; this release does not change that.
- The pip CUDA component wheels are pinned to 13.4.92 in `requirements/gb10.txt` (`ca20dae820`), the set the build
  venv already resolved.
- Acceptance test: a Muse Spark stand-in is listed as a documented skip (`trtllm-moe-sm100-only`, `4b9084cd6e`).
- The other changes since dev149 are for the rtx40 variant (sm_89 entries in the Marlin/Hadacore arch lists in
  `CMakeLists.txt`, rtx40 requirements and NCCL defaults) or tuned tooling (common.sh sync, smoke-test
  library-identity check from tuned-common).

## Tests

Scope (from the attached test log): gates; import check; tuned/gb10_acceptance_test.sh (stand-ins, all variants); python -m pytest tuned/tests
Skips by design: upstream's pytest suite (tests/) is not run (G3, recorded decision, docs/ARCHITECTURE.md); no multi-GPU tests
Result: === Full test suite summary: acceptance + tuned tests passed ===

- Acceptance: PASS for `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` and `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`;
  SKIP for five stand-ins (`deepgemm-sm121`, `bad-checkpoint`, `deps-gap-torchvision`, `mla-dims-too-small`,
  `trtllm-moe-sm100-only`), which are not coverage of this release. So the flashinfer 0.7.4 paths exercised are the
  NemotronH ones; DeepSeek-V4 and Muse Spark were not run.
- tuned/tests: 39 passed.
- Smoke test of the packaged wheel passed (2026-11-13T09:07:05Z): `test_rms_norm` 216, `test_act_and_mul` 126, `test_rotary_embedding_opcheck` 192 passed.

## Install

Install flashinfer's three wheels (see the flashinfer release) with this one; `requirements/gb10.txt` has the URLs.

```
pip install "vllm @ https://github.com/zbrad/vllm/releases/download/v0.33.1.dev170%2Bgb10.cu134.20261113.0/vllm-0.33.1.dev170+gb10.cu134.20261113.0-cp314-cp314-linux_aarch64.whl"
```
