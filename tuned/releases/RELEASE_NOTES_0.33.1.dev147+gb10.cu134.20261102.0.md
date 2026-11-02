# vLLM 0.33.1.dev147 — gb10 cu134 (0.33.1.dev147+gb10.cu134.20261102.0)

Single-arch vllm wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (TORCH_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261102.0 (UTC date of the tuned commit)
- Built from commit e0e14838e1c979767e8f3d8e66cd5e366c30fbde; tested at e0e14838e1c979767e8f3d8e66cd5e366c30fbde
- Upstream base: upstream/main merge-base 4135590ffe0119dab806d78a7eee6dff3be2618e
- Prebuilt dependencies: requirements/gb10.txt (torch, flashinfer) and https://github.com/zbrad/flash-attention-vllm/releases/download/v2.7.2.post1%2Bgb10.cu134.20261102.0/vllm_flash_attn-2.7.2.post1%2Bgb10.cu134.20261102.0-cp314-cp314-linux_aarch64.whl
- Wheel sha256: 758273fe1b5fc2cd5d493e80d0e6481c2bd5b5e57950bbde6492129f427b29c3

## Changes since 0.32.1.dev125+gb10.cu134.20261101.0

- **Upstream merge: v0.33.0** (`4ba787ec99`, 943 upstream commits; the previous base was `fe341a53dd`). The version moves
  from 0.32.1.dev125 to 0.33.1.dev147 because it counts commits after the newest upstream tag.
- Merge conflicts, 17 files. Where the fork had only carried upstream PRs that v0.33.0 contains at the same shas (#61718,
  #61858, #61427, #61409, #58297, #61556, #61167, #59815, #59604), upstream's version was taken. Fork patches kept: the
  GB10 prebuilt flash-attention block in `cmake/external_projects/vllm_flash_attn.cmake` (upstream's FetchContent
  `GIT_TAG` bump applied), the tuned-flags block in `vllm/env_override.py` (upstream removed the
  `VLLM_TRITON_FORCE_FIRST_CONFIG` block, so that is gone), and the torchvision guard in `kernel_warmup.py`.
- **New fork patch:** `step5_dsa_warmup` (new in v0.33.0) imports step5 model code, which imports torchvision, which GB10
  does not have. The first acceptance run on the merge failed both models with `ModuleNotFoundError: No module named
  'torchvision'` from `kernel_warmup`. `_step5_dsa_impl` now returns None on ImportError, as the fork already does for
  the MiniMax warmup.
- Dependencies: `vllm_flash_attn` 2.7.2.post1+gb10.cu134.20261102.0 (built from upstream `b273a02`, the commit v0.33.0
  pins; the previous pin, 20261028.0, was built from `9cd61de`). `requirements/gb10.txt` follows v0.33.0's `cuda.txt` for
  `nvidia-cudnn-frontend>=1.31.0`, `tokenspeed-mla==2.0.3` and `humming-kernels==0.1.19`. Torch (20261030.1) and
  flashinfer (0.7.0+gb10.cu134.20261101.0) are unchanged. Upstream's `cuda.txt` also moved to torch 2.14.0; the GB10 build
  does not use that file's torch pin.
- **Not checked: flashinfer.** v0.33.0's `requirements/cuda.txt` pins `flashinfer-python==0.7.2.post1`; this release ships
  the fork's flashinfer, which is built from a 0.7.0 base. The two acceptance models below pass with it. Other models and
  ops were not run, so a flashinfer API gap in v0.33.0 would show up there. **[Added 2026-11-02, see Corrections.]**
  There is such a gap: this flashinfer lacks names v0.33.0 imports; 0.33.1.dev149+gb10.cu134.20261102.1 fixes it.
- **Not checked: CPU LAPACK.** The torch has no CPU LAPACK or BLAS (see its release notes). The grep of this repo's
  runtime code for CPU `torch.linalg` calls was made on 2026-10-24 on the earlier base (Claude's grep, not a run) and was
  not repeated for v0.33.0.
- **Test harness:** `tuned/gb10_acceptance_test.sh` now runs its python from `/`. vllm's model-registry check spawns
  `python -m vllm.model_executor.models.registry`, which inherits the cwd; with the repo root as the cwd it imported the
  source tree and the stale `vllm/_C_stable_libtorch.abi3.so` in it (`ImportError: libtorch_cpu.so`)~~, which v0.33.0
  imports when it loads the platform~~. ~~Earlier releases' acceptance runs had the same cwd, so their registry check
  probably imported the source tree too; I did not verify that.~~ **[Corrected 2026-11-02, see Corrections.]**
- Acceptance (`tuned/gb10_acceptance_test.sh`): PASS for `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` and
  `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`; SKIP for the same four stand-ins as before (`deepgemm-sm121`,
  `bad-checkpoint`, `deps-gap-torchvision`, `mla-dims-too-small`), which are not coverage of this release.
- Smoke test of the packaged wheel passed (its last check: `test_rotary_embedding_opcheck`, 192 passed).

## Corrections (2026-11-02)

Made after publication; the struck text above is kept so the change is visible. The wheel and its sha256 are unchanged.

- **`_C_stable_libtorch` is not new in v0.33.0.** The notes said v0.33.0 "imports [it] when it loads the platform",
  implying earlier bases did not. That was wrong: the previous base (`fe341a53dd`) also imports `vllm._C_stable_libtorch`
  at the top of `vllm/platforms/cuda.py` and has the same registry subprocess (Claude's reading of both trees).
- **Earlier acceptance runs: unexplained, not "probably the same".** The guess that earlier runs' registry check imported
  the source tree too was never verified, and why the 0.32.1.dev125 acceptance run passed with the stale `.so` present is
  still unexplained. What is established: with the repo root as cwd, the registry subprocess imported the source tree and
  the stale `.so` and failed with `ImportError: libtorch_cpu.so`; running the test's python from `/` avoided it. The
  stale in-tree `.so` files have since been deleted from the build checkout.
- **flashinfer gap confirmed.** With this release's flashinfer (0.7.0+gb10.cu134.20261101.0), these did not import (Claude's
  check, made before dev149): the modules `flashinfer.comm.mnnvl_cutedsl_ag`, `flashinfer.comm.mnnvl_cutedsl_moe_finalize`,
  `flashinfer.comm.tp_moe_finalize_rmsnorm` and `flashinfer.norm.kernels.gated_residual_next_rmsnorm`, and the names
  `flashinfer.moe_ep.NVLinkOneSidedAlltoAll`, `flashinfer.norm.gated_residual_rmsnorm`, `rmsnorm_sigmoid_gate` and
  `supports_gated_residual_rmsnorm`. Users: the Muse Spark model code and the opt-in NVLink all2all backends, neither of
  which runs on GB10 (no sm12x MoE route for Muse Spark; no GPU-to-GPU NVLink). 0.33.1.dev149+gb10.cu134.20261102.1 pins
  flashinfer 0.7.2.post1, where all eight import.

## Tests

Scope (from the attached test log): gates; import check; tuned/gb10_acceptance_test.sh (stand-ins, all variants); python -m pytest tuned/tests
Skips by design: upstream's pytest suite (tests/) is not run (G3, recorded decision, docs/ARCHITECTURE.md); no multi-GPU tests
Result: === Full test suite summary: acceptance + tuned tests passed ===

## Install

```
pip install "vllm @ https://github.com/zbrad/vllm/releases/download/v0.33.1.dev147%2Bgb10.cu134.20261102.0/vllm-0.33.1.dev147+gb10.cu134.20261102.0-cp314-cp314-linux_aarch64.whl"
```
