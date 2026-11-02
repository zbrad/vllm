# vLLM 0.33.1.dev149 — gb10 cu134 (0.33.1.dev149+gb10.cu134.20261102.1)

Single-arch vllm wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (TORCH_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261102.1 (UTC date of the tuned commit)
- Built from commit c35a6429d10238f8e93e4b011dd55415ec6d8c16; tested at c35a6429d10238f8e93e4b011dd55415ec6d8c16
- Upstream base: upstream/main merge-base 4135590ffe0119dab806d78a7eee6dff3be2618e
- Prebuilt dependencies: requirements/gb10.txt (torch, flashinfer) and https://github.com/zbrad/flash-attention-vllm/releases/download/v2.7.2.post1%2Bgb10.cu134.20261102.0/vllm_flash_attn-2.7.2.post1%2Bgb10.cu134.20261102.0-cp314-cp314-linux_aarch64.whl
- Wheel sha256: 5db81cceb5ca2b84b8c7d0d927245d23e124f376a1e4ed715a28c90fe5e4b325

## Changes since 0.33.1.dev147+gb10.cu134.20261102.0

- **flashinfer moves from 0.7.0 to 0.7.2.post1** (`0.7.2.post1+gb10.cu134.20261102.0`), the version v0.33.0's `requirements/cuda.txt` pins. This is the only dependency change. The flashinfer jit-cache is now three wheels, not two: `flashinfer-python`, the shim `flashinfer-jit-cache` (no binaries) and the provider `flashinfer-jit-cache-sm121a` (the compiled modules). `requirements/gb10.txt` pins all three by URL, because the provider is not on PyPI and the shim requires it at the same version. See the flashinfer release notes for the layout and why.
- **Names v0.33.0 imports that flashinfer 0.7.0 lacked.** With the 0.7.0 fork build, `flashinfer.comm.mnnvl_cutedsl_ag`, `flashinfer.comm.mnnvl_cutedsl_moe_finalize`, `flashinfer.comm.tp_moe_finalize_rmsnorm`, `flashinfer.norm.kernels.gated_residual_next_rmsnorm`, `flashinfer.moe_ep.NVLinkOneSidedAlltoAll`, `flashinfer.norm.gated_residual_rmsnorm`, `rmsnorm_sigmoid_gate` and `supports_gated_residual_rmsnorm` did not import (Claude's check, made before this release; the commands were not kept). With flashinfer 0.7.2.post1 all eight import. They are used by the Muse Spark model code and by `vllm/distributed/device_communicators/all2all.py`.
- **Not run: those code paths.** The acceptance models below are NemotronH. Importing a name is not the same as the Muse Spark or NVLink all2all paths working on GB10; no model that uses them was run, and there is no multi-GPU test.
- Acceptance (`tuned/gb10_acceptance_test.sh`): PASS for `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` and `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`; SKIP for the same four stand-ins as before (`deepgemm-sm121`, `bad-checkpoint`, `deps-gap-torchvision`, `mla-dims-too-small`), which are not coverage of this release.
- Smoke test of the packaged wheel passed (its last check: `test_rotary_embedding_opcheck`, 192 passed).
- **Still not checked: CPU LAPACK.** The torch has no CPU LAPACK or BLAS (see its release notes). A grep of v0.33.0's `vllm/` for CPU `torch.linalg` calls found none (Claude's grep, not a run; it would miss tensor methods such as `.inverse()`).

## Correction to the 0.33.1.dev147 release notes

Recorded here rather than silently changed; the dev147 notes themselves are not edited by this release.

- dev147 says the stale `vllm/_C_stable_libtorch.abi3.so` in the source tree is something "v0.33.0 imports when it loads the platform", implying earlier bases did not. **That is wrong.** The previous base (`fe341a53dd`) also imports `vllm._C_stable_libtorch` at the top of `vllm/platforms/cuda.py` and has the same registry subprocess (Claude's reading of both trees).
- dev147 also says earlier acceptance runs "probably imported the source tree too". That was unverified, and **why the 0.32.1.dev125 acceptance run passed with the stale `.so` present is still unexplained.** What is established: with the repo root as cwd, the registry subprocess imported the source tree and the stale `.so` and failed with `ImportError: libtorch_cpu.so`; running the test's python from `/` avoids it.

## Tests

Scope (from the attached test log): gates; import check; tuned/gb10_acceptance_test.sh (stand-ins, all variants); python -m pytest tuned/tests
Skips by design: upstream's pytest suite (tests/) is not run (G3, recorded decision, docs/ARCHITECTURE.md); no multi-GPU tests
Result: === Full test suite summary: acceptance + tuned tests passed ===

## Install

Install flashinfer's three wheels (see the flashinfer release) with this one; `requirements/gb10.txt` has the URLs.

```
pip install "vllm @ https://github.com/zbrad/vllm/releases/download/v0.33.1.dev149%2Bgb10.cu134.20261102.1/vllm-0.33.1.dev149+gb10.cu134.20261102.1-cp314-cp314-linux_aarch64.whl"
```
