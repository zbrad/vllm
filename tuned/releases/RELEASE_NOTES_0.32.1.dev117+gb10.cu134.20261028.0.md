# vLLM 0.32.1.dev117 — gb10 cu134 (0.32.1.dev117+gb10.cu134.20261028.0)

Single-arch vllm wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (TORCH_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261028.0 (UTC date of the tuned commit)
- Built from commit 88862e64cbf6935e4d9b41f94ed1ca78fdea0098; tested at 88862e64cbf6935e4d9b41f94ed1ca78fdea0098
- Upstream base: upstream/main merge-base fe341a53dda00472c281a2d1f4bd34a74e7eae6e
- Prebuilt dependencies: requirements/gb10.txt (torch, flashinfer) and https://github.com/zbrad/flash-attention-vllm/releases/download/v2.7.2.post1%2Bgb10.cu134.20261028.0/vllm_flash_attn-2.7.2.post1%2Bgb10.cu134.20261028.0-cp314-cp314-linux_aarch64.whl
- Wheel sha256: 6ddb2fe719de2d2efbf2d41433403886b70997002a3ac063c8bcd7e7ef65b257

## Changes since 0.32.1.dev94+gb10.cu134.20261020.0

- Upstream: merged up to dev117 (the previous release was dev94).
- Prebuilt dependencies move to the 20261028.0 flashinfer (python and jit-cache) and vllm_flash_attn wheels, built against torch 2.15.0+gb10.cu134.20261022.0 (no CPU BLAS, LAPACK or MPI).
- The wheel carries `FLASH_ATTN_FA2_NATIVE_ONLY = True` and `FLASHINFER_DISABLE_JIT = True` (vllm/ext/tuned_flags.py), set by package.sh after cuobjdump found native sm_121a code and no PTX in `_vllm_fa2_C`. vllm's own PTX/driver guard (`check_flash_attn_ptx_compat`) needs that flag on a driver that supports CUDA 13.0 only; the test run now sets it in the installed copy too, so it tests what ships.
- The build venv has setuptools 84.2 or newer, and package.sh checks that and audits the wheel's runtime dependencies.
- The test tooling pins numpy 2.4.6 instead of upstream's 2.2.6, which has no Python 3.14 wheel and was being built from source against the host's OpenBLAS (a deviation from requirements/test/cuda.txt).

## Tests

Scope (from the attached test log): gates; import check; tuned/gb10_acceptance_test.sh (stand-ins, all variants); python -m pytest tuned/tests
Skips by design: upstream's pytest suite (tests/) is not run (G3, recorded decision, docs/ARCHITECTURE.md); no multi-GPU tests
Result: === Full test suite summary: acceptance + tuned tests passed ===

## Install

```
pip install "vllm @ https://github.com/zbrad/vllm/releases/download/v0.32.1.dev117%2Bgb10.cu134.20261028.0/vllm-0.32.1.dev117+gb10.cu134.20261028.0-cp314-cp314-linux_aarch64.whl"
```
